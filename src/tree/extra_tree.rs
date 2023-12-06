use super::{node::Node, tree::MaxFeatures};
use crate::{tree::tree::Tree, utils::structures::Sample};
use rand::{thread_rng, Rng};

pub struct ExtraTree {
    root: Node,
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize,
}

impl ExtraTree {
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        let root = Node::new();
        Self {
            root,
            max_depth,
            min_samples_split,
            max_features: 0,
        }
    }
}

impl Tree for ExtraTree {
    fn get_max_depth(&self) -> usize {
        self.max_depth
    }
    fn set_max_features(&mut self, max_features: usize) {
        self.max_features = max_features;
    }
    fn get_max_features(&self) -> MaxFeatures {
        MaxFeatures::All
    }
    fn get_root(&self) -> &Node {
        &self.root
    }
    fn set_root(&mut self, root: Node) {
        self.root = root;
    }
    fn pre_split_conditions(&self, samples: &[Sample<'_>], current_depth: usize) -> bool {
        // Base case: not enough samples or max depth reached
        return samples.len() <= self.min_samples_split || current_depth >= self.max_depth
    }
    fn post_split_conditions(&self, _new_impurity: f64, _old_impurity: f64) -> bool {
        return false;
    }
    fn get_split(&self, samples: &[Sample<'_>]) -> (usize, f64, f64) {
        let best_feature = thread_rng().gen_range(0..samples[0].data.len());
        // Exclude first and last thresholds, avoiding a split on the min and max values (could move all samples on the left or right child)
        let mut thresholds = samples.iter().map(|f| f.data[best_feature]).collect::<Vec<f64>>();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let best_threshold = thresholds[thread_rng().gen_range(1..thresholds.len()-1)];
        let best_impurity = thread_rng().gen_range(f64::EPSILON..1.0);

        (best_feature, best_threshold, best_impurity)
    }
}
