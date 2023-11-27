use crate::{tree::tree::Tree, utils::structures::Sample};
use rand::{seq::SliceRandom, thread_rng, Rng};
use super::{node::Node, tree::MaxFeatures};

pub struct ExtraTree {
    root: Node,
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize,
}

impl ExtraTree {
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self{
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
        if samples.len() <= self.min_samples_split || current_depth == self.max_depth {
            return true;
        }
        // Base case: samples are the same object
        let first_sample = samples[0].data;
        let is_all_same_data = samples.iter().all(|v| v.data == first_sample);
        if is_all_same_data {
            return true;
        }
        // Base case: all samples have the same class
        let first_class = samples[0].target;
        let is_all_same_target = samples.iter().all(|v| v.target == first_class);
        if is_all_same_target {
            return true;
        }

        return false;
    }
    fn post_split_conditions(&self, impurity: f64) -> bool {
        return false;
    }
    fn get_split(&self, samples: &[Sample<'_>]) -> (usize, f64, f64) {
        let best_feature = thread_rng().gen_range(0..samples[0].data.len());
        let best_threshold = samples[thread_rng().gen_range(0..samples.len())].data[best_feature];
        let best_impurity = thread_rng().gen_range(f64::EPSILON..1.0);

        (best_feature, best_threshold, best_impurity)
    }
}