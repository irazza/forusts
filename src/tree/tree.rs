use crate::feature_extraction::statistics::EULER_MASCHERONI;
use core::fmt::Debug;
use hashbrown::HashMap;
use rand::{thread_rng, Rng};
use serde_derive::{Deserialize, Serialize};
use std::{cmp::max, ops::Deref};

use crate::{feature_extraction::statistics::stddev, utils::structures::Sample};

use super::node::{LeafClassification, Node};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum MaxFeatures {
    All,
    Sqrt,
    Log2,
}
impl MaxFeatures {
    pub fn convert(&self, n_features: usize) -> usize {
        match self {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => max(1, (n_features as f64).sqrt() as usize),
            MaxFeatures::Log2 => max(1, (n_features as f64).log2() as usize),
        }
    }
}
impl Debug for MaxFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaxFeatures::All => write!(f, "A"),
            MaxFeatures::Sqrt => write!(f, "S"),
            MaxFeatures::Log2 => write!(f, "L"),
        }
    }
}
#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum Criterion {
    Gini,
    Entropy,
    Random,
}
impl Criterion {
    pub fn to_fn<T: Tree>(self) -> fn(&HashMap<isize, usize>) -> f64 {
        match self {
            Criterion::Gini => T::gini_impurity,
            Criterion::Entropy => T::entropy_impurity,
            Criterion::Random => T::random_impurity,
        }
    }
}

impl Debug for Criterion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Criterion::Gini => write!(f, "G"),
            Criterion::Entropy => write!(f, "E"),
            Criterion::Random => write!(f, "R"),
        }
    }
}
pub trait SplitParameters: Sync + Send + Debug + Ord + Eq {
    fn split(&self, samples: &Sample, is_train: bool) -> bool;
    fn path_length<T: Tree<SplitParameters = Self>>(tree: &T, x: &Sample) -> f64;
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct StandardSplit {
    pub feature: usize,
    pub threshold: f64,
}

impl Ord for StandardSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl Eq for StandardSplit {}

impl SplitParameters for StandardSplit {
    fn split(&self, sample: &Sample, _is_train: bool) -> bool {
        sample.data[self.feature] < self.threshold
    }
    fn path_length<T: Tree<SplitParameters = Self>>(tree: &T, x: &Sample) -> f64 {
        let leaf = tree.predict_leaf(&x);

        let samples = leaf.get_samples() as f64;
        let path_length;

        if samples <= 1.0 {
            path_length = 0.0;
        } else if samples == 2.0 {
            path_length = 1.0;
        } else {
            path_length =
                2.0 * (f64::ln(samples - 1.0) + EULER_MASCHERONI) - 2.0 * (samples - 1.0) / samples;
        }
        path_length + leaf.get_depth() as f64
    }
}
pub trait Tree: Sync + Send {
    type Config;
    type SplitParameters: SplitParameters;
    fn new(init: Self::Config) -> Self;
    fn get_max_depth(&self) -> usize;
    fn get_root(&self) -> &Node<Self::SplitParameters>;
    fn set_root(&mut self, root: Node<Self::SplitParameters>);
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64);
    fn pre_split_conditions(&self, samples: &[Sample], current_depth: usize) -> bool;
    fn fit(&mut self, data: &[Sample]) {
        let mut data = data.to_vec();
        let root = self.build_tree(&mut data, 0, f64::INFINITY);
        self.set_root(root);
    }
    fn build_tree(
        &mut self,
        samples: &mut [Sample],
        depth: usize,
        _impurity: f64,
    ) -> Node<Self::SplitParameters> {
        if self.pre_split_conditions(samples, depth) {
            return Node::Leaf {
                class: Self::get_leaf_class(samples, None),
                depth: depth,
                impurity: f64::EPSILON,
                n_samples: samples.len(),
            };
        }

        let (best_split_parameters, best_impurity) = self.get_split(samples);

        let (left_data, right_data) = Self::split(samples, &best_split_parameters);

        // Split the data and recursively build the left and right subtrees
        let left_subtree = self.build_tree(left_data, depth + 1, best_impurity);
        let right_subtree = self.build_tree(right_data, depth + 1, best_impurity);

        Node::Split {
            split_params: best_split_parameters,
            left: Box::new(left_subtree),
            right: Box::new(right_subtree),
            depth: depth,
            impurity: best_impurity,
        }
    }

    fn predict(&self, x: &[Sample]) -> Vec<isize> {
        x.iter()
            .map(|sample| self.predict_leaf(sample).get_class(&sample.data))
            .collect()
    }

    fn get_leaves(&self) -> Vec<&Node<Self::SplitParameters>> {
        let mut leaves = Vec::new();
        let mut queue = vec![self.get_root()];

        while !queue.is_empty() {
            let node = queue.remove(0);
            if let Node::Leaf { .. } = node {
                leaves.push(node);
            } else if let Node::Split { left, right, .. } = node {
                queue.push(left);
                queue.push(right);
            }
        }

        leaves
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
    fn get_splits(&self, x: &Sample) -> Vec<&Self::SplitParameters> {
        let mut path = Vec::new();
        let mut node = self.get_root();
        while let Node::Split {
            split_params,
            left,
            right,
            depth: _,
            impurity: _,
        } = node
        {
            path.push(split_params);
            if split_params.split(x, true) {
                node = left;
            } else {
                node = right;
            }
        }
        path
    }
    fn predict_leaf(&self, x: &Sample) -> &Node<Self::SplitParameters> {
        let mut node = self.get_root();

        while let Node::Split {
            split_params,
            left,
            right,
            depth: _,
            impurity: _,
        } = node
        {
            if split_params.split(x, false) {
                node = left;
            } else {
                node = right;
            }
        }
        node
    }
    fn split<'a, 'b>(
        samples: &'a mut [Sample],
        parameters: &Self::SplitParameters,
    ) -> (&'a mut [Sample], &'a mut [Sample]) {
        let mut idx = 0;
        let mut last = samples.len();

        while idx < last {
            if parameters.split(&samples[idx], true) {
                idx += 1;
            } else {
                samples.swap(idx, last - 1);
                last -= 1;
            }
        }

        samples.split_at_mut(idx)
    }
    fn get_leaf_class(
        samples: &[Sample],
        _parameters: Option<&Self::SplitParameters>,
    ) -> LeafClassification {
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

        LeafClassification::Simple(most_common_class)
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
    fn random_impurity(_class_counts: &HashMap<isize, usize>) -> f64 {
        return thread_rng().gen_range(0.0..1.0);
    }
    fn sd_gain(y_l: &[f64], y_r: &[f64]) -> f64 {
        let num = (stddev(y_l) + stddev(y_r)) / 2.0;
        let den = stddev(&[y_l, y_r].concat());
        1.0 - num / den
    }
}
