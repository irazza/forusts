use super::node::Node;
use crate::{tree::tree::Tree, utils::structures::Sample};
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::cmp::Ordering;

pub struct IsolationTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
}

#[derive(Clone, Debug)]
pub struct IsolationTree {
    root: Node,
    max_depth: usize,
    min_samples_split: usize,
}

impl Tree for IsolationTree {
    type Config = IsolationTreeConfig;
    fn new(config: Self::Config) -> Self {
        Self {
            root: Node::new(),
            max_depth: config.max_depth,
            min_samples_split: config.min_samples_split,
        }
    }
    fn get_max_depth(&self) -> usize {
        self.max_depth
    }
    fn get_root(&self) -> &Node {
        &self.root
    }
    fn set_root(&mut self, root: Node) {
        self.root = root;
    }
    fn pre_split_conditions(&self, samples: &[Sample<'_>], current_depth: usize) -> bool {
        // Base case: not enough samples or max depth reached
        return samples.len() <= self.min_samples_split || current_depth >= self.max_depth;
    }
    fn post_split_conditions(&self, _new_impurity: f64, _old_impurity: f64) -> bool {
        return false;
    }
    fn get_split(&self, samples: &[Sample<'_>]) -> (usize, f64, f64) {
        let mut shuffled_features: Vec<usize> = (0..samples[0].data.len()).collect();
        shuffled_features.shuffle(&mut thread_rng());
        let best_feature = shuffled_features[thread_rng().gen_range(0..8)];
        // Exclude first and last thresholds, avoiding a split on the min and max values (could move all samples on the left or right child)
        let mut thresholds = samples
            .iter()
            .map(|f| f.data[best_feature])
            .collect::<Vec<f64>>();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let best_threshold = thresholds[thread_rng().gen_range(1..thresholds.len() - 1)];
        let best_impurity = thread_rng().gen_range(f64::EPSILON..1.0);

        (best_feature, best_threshold, best_impurity)
    }
}
