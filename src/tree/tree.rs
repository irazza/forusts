use crate::{forest::forest::EGAMMA, utils::structures::Sample, RandomGenerator};
use core::fmt::Debug;
use hashbrown::HashMap;
use std::{collections::VecDeque, ops::Range};

use super::node::{LeafClassification, Node};

pub trait SplitParameters: Sync + Send + Debug + Ord + Eq {
    fn split(&self, sample: &Sample) -> usize;
    fn path_length<T: Tree<SplitParameters = Self>>(tree: &T, sample: &Sample) -> f64;
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
    fn split(&self, sample: &Sample) -> usize {
        (sample.features[self.feature] < self.threshold) as usize
    }
    fn path_length<T: Tree<SplitParameters = Self>>(tree: &T, x: &Sample) -> f64 {
        let leaf = tree.predict_leaf(x);
        return leaf.get_depth() as f64 + T::average_path_length(leaf.get_n_samples());
    }
}
pub trait Tree: Sync + Send {
    type Config;
    type SplitParameters: SplitParameters;
    fn new(init: Self::Config, random_state: &mut RandomGenerator) -> Self;
    fn transform(&self, data: &[Sample]) -> Vec<Sample>;
    fn get_max_depth(&self) -> usize;
    fn get_min_samples_split(&self) -> usize;
    fn get_min_samples_leaf(&self) -> usize;
    fn get_root(&self) -> &Node<Self::SplitParameters>;
    fn set_nodes(&mut self, nodes: Vec<Node<Self::SplitParameters>>);
    fn get_node_at(&self, id: usize) -> &Node<Self::SplitParameters>;
    fn get_split(
        &self,
        samples: &[Sample],
        min_samples_leaf: usize,
        non_constant_features: &mut Vec<usize>,
        random_state: &mut RandomGenerator,
    ) -> Option<(Self::SplitParameters, f64)>;
    fn fit(&mut self, data: &[Sample], random_state: &mut RandomGenerator) {
        let mut data = data.to_vec();
        let nodes = self.build_tree(&mut data, random_state);
        self.set_nodes(nodes);
    }
    fn build_tree(
        &mut self,
        samples: &mut [Sample],
        random_state: &mut RandomGenerator,
    ) -> Vec<Node<Self::SplitParameters>> {
        let features = (0..samples[0].features.len()).collect::<Vec<_>>();
        let mut queue = VecDeque::from(vec![(0..samples.len(), 0, None, features)]);
        let mut nodes = Vec::new();
        while let Some((range, depth, parent, mut non_constant_features)) = queue.pop_front() {
            let id = nodes.len();

            let is_leaf = depth >= self.get_max_depth()
                || range.len() < self.get_min_samples_split()
                || range.len() < 2 * self.get_min_samples_leaf()
                || non_constant_features.is_empty();

            let node_samples = &mut samples[range.clone()];

            let get_leaf = || Node::External {
                id,
                class: Self::get_leaf_class(node_samples, None),
                depth,
                n_samples: node_samples.len(),
            };

            let add_children = |nodes: &mut Vec<Node<Self::SplitParameters>>| {
                if let Some(parent) = parent {
                    if let Node::Internal { children, .. } = &mut nodes[parent] {
                        children.push(id);
                    }
                }
            };

            if is_leaf {
                nodes.push(get_leaf());
                add_children(&mut nodes);
                continue;
            }

            let Some((split_parameters, impurity)) = self.get_split(
                node_samples,
                self.get_min_samples_leaf(),
                &mut non_constant_features,
                random_state,
            ) else {
                nodes.push(get_leaf());
                add_children(&mut nodes);
                continue;
            };

            let splitted_data = Self::split(node_samples, &split_parameters);

            assert!(splitted_data.len() >= 2);

            nodes.push(Node::Internal {
                id,
                split_params: split_parameters,
                children: vec![],
                n_children: splitted_data.len(),
                depth: depth,
                impurity,
                n_samples: node_samples.len(),
            });

            for child_range in splitted_data {
                let child_depth = depth + 1;
                let child_range =
                    (range.start + child_range.start)..(range.start + child_range.end);
                queue.push_back((
                    child_range,
                    child_depth,
                    Some(id),
                    non_constant_features.clone(),
                ));
            }

            add_children(&mut nodes);
        }
        nodes
    }
    fn predict(&self, x: &[Sample]) -> Vec<isize> {
        x.iter()
            .map(|sample| self.predict_leaf(sample).get_class(&sample.features))
            .collect()
    }
    fn average_path_length(n_samples: usize) -> f64 {
        if n_samples == 1 {
            0.0
        } else {
            2.0 * (Self::harmonic_number(n_samples - 1))
                - (2.0 * (n_samples as f64 - 1.0) / n_samples as f64)
        }
    }
    #[inline]
    fn harmonic_number(n: usize) -> f64 {
        (n as f64).ln() + EGAMMA
    }

    // fn get_leaves(&self) -> Vec<&Node<Self::SplitParameters>> {
    //     let mut leaves = Vec::new();
    //     let mut queue = vec![self.get_root()];

    //     while !queue.is_empty() {
    //         let node = queue.remove(0);
    //         if let Node::External { .. } = node {
    //             leaves.push(node);
    //         } else if let Node::Internal { left, right, .. } = node {
    //             queue.push(left);
    //             queue.push(right);
    //         }
    //     }

    //     leaves
    // }

    // fn get_diameter(n: &Node<Self::SplitParameters>) -> (usize, usize) {
    //     match n {
    //         Node::External { .. } => (1, 1),
    //         Node::Internal { left, right, .. } => {
    //             let (left_diameter, left_height) = Self::get_diameter(left);
    //             let (right_diameter, right_height) = Self::get_diameter(right);
    //             let current_diameter = left_height + right_height;

    //             (
    //                 max(current_diameter, max(left_diameter, right_diameter)),
    //                 1 + max(left_height, right_height),
    //             )
    //         }
    //     }
    // }
    fn get_splits(&self, x: &Sample) -> Vec<&Self::SplitParameters> {
        let mut path = Vec::new();
        let mut node = self.get_root();
        while let Node::Internal {
            split_params,
            children,
            ..
        } = node
        {
            path.push(split_params);
            node = self.get_node_at(children[split_params.split(x)]);
        }
        path
    }
    fn predict_leaf(&self, x: &Sample) -> &Node<Self::SplitParameters> {
        let mut node = self.get_root();

        while let Node::Internal {
            split_params,
            children,
            ..
        } = node
        {
            let branch = split_params.split(x);
            node = self.get_node_at(children[branch]);
        }
        node
    }
    fn split<'a, 'b>(
        samples: &'a mut [Sample],
        parameters: &Self::SplitParameters,
    ) -> Vec<Range<usize>> {
        let mut branches = Vec::new();
        let mut counters = Vec::new();
        let mut positions = Vec::new();
        let mut ranges = Vec::new();
        let mut samples_cp = samples.iter().cloned().map(Some).collect::<Vec<_>>();

        for sample in samples.iter() {
            let branch = parameters.split(sample);
            branches.push(branch);

            if counters.len() <= branch {
                counters.resize(branch + 1, 0);
            }

            counters[branch] += 1;
        }

        let mut count = 0;
        for counter in counters.iter() {
            ranges.push(count..count + counter);
            positions.push(count);
            count += counter;
        }

        for idx in 0..samples_cp.len() {
            let branch = branches[idx];
            let position = positions[branch];
            samples[position] = samples_cp[idx].take().unwrap();
            positions[branch] += 1;
        }

        ranges
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
    // fn bfs(&self) -> Vec<&Node<Self::SplitParameters>> {
    //     let mut queue = vec![self.get_root()];
    //     let mut bfs = Vec::new();

    //     while !queue.is_empty() {
    //         let node = queue.remove(0);
    //         bfs.push(node);
    //         if let Node::Internal { left, right, .. } = node {
    //             queue.push(left);
    //             queue.push(right);
    //         }
    //     }

    //     bfs
    // }
    // fn get_depth(&self) -> usize {
    //     let mut max_depth = 0;
    //     let mut queue = vec![(self.get_root(), 0)];

    //     while !queue.is_empty() {
    //         let (node, depth) = queue.remove(0);
    //         max_depth = max(max_depth, depth);
    //         if let Node::Internal { left, right, .. } = node {
    //             queue.push((left, depth + 1));
    //             queue.push((right, depth + 1));
    //         }
    //     }

    //     max_depth
    // }
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
            Node::External { .. } => {
                if let Some(found_lca) = found_lca {
                    ancestors.insert(current as *const Node<Self::SplitParameters>, found_lca);
                }
                false
            }
            Node::Internal { children, .. } => {
                todo!("Implement this");
                // if let Some(found_lca) = found_lca {
                //     Self::compute_ancestor_rec(left.deref(), target, Some(found_lca), ancestors);
                //     Self::compute_ancestor_rec(right.deref(), target, Some(found_lca), ancestors);
                //     false
                // } else {
                //     let left_found =
                //         Self::compute_ancestor_rec(left.deref(), target, None, ancestors);
                //     if left_found {
                //         Self::compute_ancestor_rec(right.deref(), target, Some(current), ancestors);
                //         true
                //     } else {
                //         let right_found =
                //             Self::compute_ancestor_rec(right, target, None, ancestors);
                //         if right_found {
                //             Self::compute_ancestor_rec(
                //                 left.deref(),
                //                 target,
                //                 Some(current),
                //                 ancestors,
                //             );
                //             true
                //         } else {
                //             false
                //         }
                //     }
                // }
            }
        }
    }
    // fn entropy_impurity(class_counts: &HashMap<isize, usize>) -> f64 {
    //     let mut impurity = 0.0;
    //     let total_samples = class_counts.values().sum::<usize>() as f64;
    //     for &count in class_counts.values() {
    //         if count > 0 {
    //             let p = count as f64 / total_samples;
    //             impurity -= p * p.log2();
    //         }
    //     }

    //     impurity
    // }
    // fn gini_impurity(class_counts: &HashMap<isize, usize>) -> f64 {
    //     let mut impurity = 1.0;
    //     let total_samples = class_counts.values().sum::<usize>() as f64;
    //     for &count in class_counts.values() {
    //         if count > 0 {
    //             let p = count as f64 / total_samples;
    //             impurity -= p * p;
    //         }
    //     }

    //     impurity
    // }
    // fn random_impurity(_class_counts: &HashMap<isize, usize>) -> f64 {
    //     return thread_rng().gen_range(0.0..1.0);
    // }
    // fn sd_gain(y_l: &[f64], y_r: &[f64]) -> f64 {
    //     let num = (stddev(y_l) + stddev(y_r)) / 2.0;
    //     let den = stddev(&[y_l, y_r].concat());
    //     1.0 - num / den
    // }
}
