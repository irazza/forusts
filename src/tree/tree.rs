use core::fmt::Debug;
use hashbrown::HashMap;
use rand::{thread_rng, Rng};
use serde_derive::{Deserialize, Serialize};
use std::{cmp::max, ops::Deref, sync::Arc};

use crate::{feature_extraction::statistics::stddev, utils::structures::Sample};

use super::node::Node;

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum MaxFeatures {
    All,
    Sqrt,
    Log2,
}
impl MaxFeatures {
    pub fn convert(&self, n_features: usize) -> usize {
        match self {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => (n_features as f64).sqrt() as usize,
            MaxFeatures::Log2 => (n_features as f64).log2() as usize,
        }
    }
}
#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
pub enum Criterion {
    Gini,
    Entropy,
    Random,
}
impl Criterion {
    pub fn to_string(self) -> &'static str {
        match self {
            Criterion::Gini => "gini",
            Criterion::Entropy => "entropy",
            Criterion::Random => "random",
        }
    }

    pub fn to_fn<T: Tree>(self) -> fn(&HashMap<isize, usize>) -> f64 {
        match self {
            Criterion::Gini => T::gini_impurity,
            Criterion::Entropy => T::entropy_impurity,
            Criterion::Random => T::random_impurity,
        }
    }
}
pub trait SplitParameters: Sync + Send + Debug + Ord + Eq {
    fn split(&self, samples: &Sample<'_>) -> bool;
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SplitTest {
    pub feature: usize,
    pub threshold: f64,
}

impl Ord for SplitTest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl Eq for SplitTest {}

impl SplitParameters for SplitTest {
    fn split(&self, sample: &Sample<'_>) -> bool {
        sample.data[self.feature] < self.threshold
    }
}
pub trait Tree: Sync + Send {
    type Config;
    type SplitParameters: SplitParameters;
    fn new(init: Self::Config) -> Self;
    fn get_max_depth(&self) -> usize;
    fn get_root(&self) -> &Node<Self::SplitParameters>;
    fn set_root(&mut self, root: Node<Self::SplitParameters>);
    fn get_split(&self, samples: &[Sample<'_>]) -> (Self::SplitParameters, f64);
    fn pre_split_conditions(&self, samples: &[Sample<'_>], current_depth: usize) -> bool;
    fn post_split_conditions(&self, new_impurity: f64, old_impurity: f64) -> bool;
    fn fit(&mut self, data: &[Sample<'_>]) {
        let _n_features = data[0].data.len();
        let data = &mut data.to_vec();
        let root = self.build_tree(data, self.get_max_depth(), f64::MAX);
        self.set_root(root);
    }
    fn build_tree(
        &mut self,
        samples: &mut [Sample<'_>],
        max_depth: usize,
        impurity: f64,
    ) -> Node<Self::SplitParameters> {
        let current_depth = max(1, self.get_max_depth() - max_depth);

        if self.pre_split_conditions(samples, current_depth) {
            return Node::Leaf {
                class: Self::get_most_common_class(samples),
                depth: current_depth,
                impurity: f64::EPSILON,
                n_samples: samples.len(),
            };
        }

        let (best_split_parameters, best_impurity) = self.get_split(samples);

        if self.post_split_conditions(best_impurity, impurity) {
            return Node::Leaf {
                class: Self::get_most_common_class(samples),
                depth: current_depth,
                impurity: f64::EPSILON,
                n_samples: samples.len(),
            };
        }

        let (left_data, right_data) = Self::split(samples, &best_split_parameters);

        assert!(
            left_data.len() > 0 && right_data.len() > 0,
            "{} {}",
            left_data.len(),
            right_data.len()
        );
        // Split the data and recursively build the left and right subtrees
        let left_subtree = self.build_tree(left_data, max_depth - 1, best_impurity);
        let right_subtree = self.build_tree(right_data, max_depth - 1, best_impurity);

        Node::Split {
            split_params: best_split_parameters,
            left: Box::new(left_subtree),
            right: Box::new(right_subtree),
            depth: current_depth,
            impurity: best_impurity,
            n_samples: samples.len(),
        }
    }
    fn predict(&self, x: &[Sample<'_>]) -> Vec<isize> {
        x.iter()
            .map(|sample| self.predict_leaf(sample).get_class())
            .collect()
    }
    fn get_diameter(n: &Node<Self::SplitParameters>) -> (usize, usize) {
        match n {
            Node::Leaf { .. } => (1, 1),
            Node::Split { left, right, .. } => {
                let (left_diameter, left_height) = Self::get_diameter(left);
                let (right_diameter, right_height) = Self::get_diameter(right);
                let current_diameter = left_height + right_height;

                (
                    max(current_diameter, max(left_diameter, right_diameter)),
                    1 + max(left_height, right_height),
                )
            }
        }
    }
    fn get_splits(&self, x: &Sample<'_>) -> Vec<&Self::SplitParameters> {
        let mut path = Vec::new();
        let mut node = self.get_root();
        while let Node::Split {
            split_params,
            left,
            right,
            depth: _,
            impurity: _,
            n_samples: _,
        } = node
        {
            path.push(split_params);
            if split_params.split(x) {
                node = left;
            } else {
                node = right;
            }
        }
        path
    }
    fn predict_leaf(&self, x: &Sample<'_>) -> &Node<Self::SplitParameters> {
        let mut node = self.get_root();

        while let Node::Split {
            split_params,
            left,
            right,
            depth: _,
            impurity: _,
            n_samples: _,
        } = node
        {
            if split_params.split(x) {
                node = left;
            } else {
                node = right;
            }
        }
        node
    }
    fn split<'a, 'b>(
        samples: &'a mut [Sample<'b>],
        parameters: &Self::SplitParameters,
    ) -> (&'a mut [Sample<'b>], &'a mut [Sample<'b>]) {
        let mut idx = 0;
        let mut last = samples.len();

        while idx < last {
            if parameters.split(&samples[idx]) {
                idx += 1;
            } else {
                samples.swap(idx, last - 1);
                last -= 1;
            }
        }

        samples.split_at_mut(idx)
    }
    fn get_most_common_class(samples: &[Sample<'_>]) -> isize {
        let mut class_counts = HashMap::new();
        for Sample { target, .. } in samples {
            *class_counts.entry(*target).or_insert(0) += 1;
        }

        let mut max_count = 0;
        let mut most_common_class = 0;

        for (class, count) in &class_counts {
            if *count > max_count {
                max_count = *count;
                most_common_class = *class;
            }
        }

        most_common_class
    }
    fn bfs(&self) -> Vec<&Node<Self::SplitParameters>> {
        let mut queue = vec![self.get_root()];
        let mut bfs = Vec::new();

        while !queue.is_empty() {
            let node = queue.remove(0);
            bfs.push(node);
            if let Node::Split { left, right, .. } = node {
                queue.push(left);
                queue.push(right);
            }
        }

        bfs
    }
    fn get_depth(&self) -> usize {
        let mut max_depth = 0;
        let mut queue = vec![(self.get_root(), 0)];

        while !queue.is_empty() {
            let (node, depth) = queue.remove(0);
            max_depth = max(max_depth, depth);
            if let Node::Split { left, right, .. } = node {
                queue.push((left, depth + 1));
                queue.push((right, depth + 1));
            }
        }

        max_depth
    }
    fn compute_ancestor<'a>(
        &'a self,
        node: &'a Node<Self::SplitParameters>,
    ) -> HashMap<*const Node<Self::SplitParameters>, &'a Node<Self::SplitParameters>> {
        let mut ancestors = HashMap::new();
        Self::compute_ancestor_rec(&self.get_root(), node, None, &mut ancestors);
        ancestors.insert(node as *const Node<Self::SplitParameters>, node);
        ancestors
    }
    fn compute_ancestor_rec<'a>(
        current: &'a Node<Self::SplitParameters>,
        target: &'a Node<Self::SplitParameters>,
        found_lca: Option<&'a Node<Self::SplitParameters>>,
        ancestors: &mut HashMap<
            *const Node<Self::SplitParameters>,
            &'a Node<Self::SplitParameters>,
        >,
    ) -> bool {
        if (current as *const Node<Self::SplitParameters>)
            == (target as *const Node<Self::SplitParameters>)
        {
            return true;
        }

        match current {
            Node::Leaf { .. } => {
                if let Some(found_lca) = found_lca {
                    ancestors.insert(current as *const Node<Self::SplitParameters>, found_lca);
                }
                false
            }
            Node::Split { left, right, .. } => {
                if let Some(found_lca) = found_lca {
                    Self::compute_ancestor_rec(left.deref(), target, Some(found_lca), ancestors);
                    Self::compute_ancestor_rec(right.deref(), target, Some(found_lca), ancestors);
                    false
                } else {
                    let left_found =
                        Self::compute_ancestor_rec(left.deref(), target, None, ancestors);
                    if left_found {
                        Self::compute_ancestor_rec(right.deref(), target, Some(current), ancestors);
                        true
                    } else {
                        let right_found =
                            Self::compute_ancestor_rec(right, target, None, ancestors);
                        if right_found {
                            Self::compute_ancestor_rec(
                                left.deref(),
                                target,
                                Some(current),
                                ancestors,
                            );
                            true
                        } else {
                            false
                        }
                    }
                }
            }
        }
    }
    fn entropy_impurity(class_counts: &HashMap<isize, usize>) -> f64 {
        let mut impurity = 0.0;
        let total_samples = class_counts.values().sum::<usize>() as f64;
        for &count in class_counts.values() {
            if count > 0 {
                let p = count as f64 / total_samples;
                impurity -= p * p.log2();
            }
        }

        impurity
    }
    fn gini_impurity(class_counts: &HashMap<isize, usize>) -> f64 {
        let mut impurity = 1.0;
        let total_samples = class_counts.values().sum::<usize>() as f64;
        for &count in class_counts.values() {
            if count > 0 {
                let p = count as f64 / total_samples;
                impurity -= p * p;
            }
        }

        impurity
    }
    fn random_impurity(class_counts: &HashMap<isize, usize>) -> f64 {
        return thread_rng().gen_range(0.0..1.0);
    }
    fn sd_gain(y_l: &[f64], y_r: &[f64]) -> f64 {
        let num = (stddev(y_l) + stddev(y_r)) / 2.0;
        let den = stddev(&[y_l, y_r].concat());
        1.0 - num / den
    }
}
