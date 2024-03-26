use core::panic;
use std::{
    borrow::Cow, cmp::{max, min}, fmt::Debug, sync::Arc
};

use super::{
    node::{LeafClassifier, Node},
    tree::{Criterion, SplitParameters},
};
use crate::{
    distance::distances::{dtw, euclidean, twe}, feature_extraction::statistics::{fisher_score, mean, slope, stddev, unique, value_counts}, forest::forest::{
        ClassificationForestConfig, ClassificationTree}, tree::tree::Tree, utils::structures::Sample
};
use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng, Rng, SeedableRng};

#[derive(Clone, PartialEq, PartialOrd)]
pub struct DistanceSplit {
    pub left_candidates: Vec<Arc<Vec<f64>>>,
    pub right_candidates: Vec<Arc<Vec<f64>>>,
}

impl Ord for DistanceSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl Eq for DistanceSplit {}

impl Debug for DistanceSplit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistanceSplit")
            .field("left_candidates", &self.left_candidates.len())
            .field("right_candidates", &self.right_candidates.len())
            .finish()
    }
}

impl SplitParameters for DistanceSplit {
    fn split(&self, sample: &Sample) -> bool {

        let mut left_dist = 0.0;
        for candidate in &self.left_candidates {
            let dist = twe(&sample.data, &candidate);
            if dist <= f64::EPSILON {
                return true;
            }
            // left_dist = left_dist.min(dist);
            left_dist += dist;
        }
        let mut right_dist = 0.0;
        for candidate in &self.right_candidates {
            let dist = twe(&sample.data, &candidate);
            if dist <= f64::EPSILON {
                return false;
            }
            // right_dist = right_dist.min(dist);
            right_dist += dist;
        }
        left_dist /= self.left_candidates.len() as f64;
        right_dist /= self.right_candidates.len() as f64;
        left_dist < right_dist
    }

    fn path_length<T: Tree<SplitParameters = Self>>(_tree: &T, _x: &Sample) -> f64 {
        unreachable!()
    }
}

#[derive(Clone, Debug)]
pub struct DistanceTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub criterion: Criterion,
}

#[derive(Clone, Debug)]
pub struct DistanceTree {
    root: Node<DistanceSplit>,
    config: DistanceTreeConfig,
}

impl ClassificationTree for DistanceTree {
    fn from_classification_config(config: &ClassificationForestConfig) -> Self {
        Self::new(DistanceTreeConfig {
            max_depth: config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: config.min_samples_split,
            // Setted to 2 to avoid empty child when splitting when there are only two samples
            criterion: config.criterion,
        })
    }
}

impl Tree for DistanceTree {
    type Config = DistanceTreeConfig;
    type SplitParameters = DistanceSplit;
    fn new(config: Self::Config) -> Self {
        Self {
            root: Node::new(),
            config,
        }
    }
    fn get_max_depth(&self) -> usize {
        self.config.max_depth
    }
    fn get_root(&self) -> &Node<Self::SplitParameters> {
        &self.root
    }
    fn set_root(&mut self, root: Node<Self::SplitParameters>) {
        self.root = root;
    }
    fn pre_split_conditions(&self, samples: &[Sample], current_depth: usize) -> bool {
        // Base case: not enough samples or max depth reached
        if samples.len() <= self.config.min_samples_split || current_depth == self.config.max_depth
        {
            return true;
        }
        // Base case: all samples have the same target
        if samples.iter().all(|s| s.target == samples[0].target) {
            return true;
        }
        // Base case: all samples have the same features
        if samples.iter().all(|s| s.data == samples[0].data) {
            return true;
        }
        return false;
    }
    fn post_split_conditions(&self, new_impurity: f64, _old_impurity: f64) -> bool {
        // Check if there is a non empty split
        if new_impurity == std::f64::INFINITY {
            return true;
        }
        return false;
    }

    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        let mut samples = samples.iter().map(|s| s.data.clone()).collect::<Vec<_>>();
        samples.shuffle(&mut thread_rng());
        let mut left_candidates = Vec::new();
        let mut right_candidates = Vec::new();

        for (i, sample) in samples.into_iter().enumerate() {
            if i % 2 == 0 {
                left_candidates.push(sample);
            } else {
                right_candidates.push(sample);
            }
        }

        let split = DistanceSplit {
            left_candidates,
            right_candidates,
        };

        let impurity = thread_rng().gen_range(0.0..1.0);

        (split, impurity)
    }
}
