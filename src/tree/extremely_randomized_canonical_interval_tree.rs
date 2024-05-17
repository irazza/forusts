use core::panic;
use std::hash::Hash;
use super::{node::Node, tree::SplitParameters};
use crate::{
    feature_extraction::catch22::compute_catch,
    forest::{
        extremely_randomized_canonical_interval_forest::ExtremelyRandomizedCanonicalIntervalForestConfig,
        forest::ClassificationTree,
    },
    tree::tree::Tree,
    utils::structures::Sample,
};
use rand::{thread_rng, Rng};

pub const MIN_INTERVAL_LEN: usize = 20;
pub const TOT_ATTRIBUTES: usize = 25;

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct ExtremelyRandomizedCanonicalIntervalSplit {
    pub interval: (usize, usize),
    pub feature: usize,
    pub threshold: f64,
}
impl Hash for ExtremelyRandomizedCanonicalIntervalSplit {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.interval.0.hash(state);
        self.interval.1.hash(state);
        self.feature.hash(state);
        self.threshold.to_bits().hash(state);
    }
}
impl Eq for ExtremelyRandomizedCanonicalIntervalSplit {}
impl Ord for ExtremelyRandomizedCanonicalIntervalSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl SplitParameters for ExtremelyRandomizedCanonicalIntervalSplit {
    fn split(&self, sample: &Sample, _is_train: bool) -> bool {
        let feature = compute_catch(self.feature)(&sample.data[self.interval.0..self.interval.1]);
        feature < self.threshold
    }
    fn path_length<T: Tree<SplitParameters = Self>>(_tree: &T, _x: &Sample) -> f64 {
        unreachable!();
    }
}

#[derive(Clone, Debug)]
pub struct ExtremelyRandomizedCanonicalIntervalTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
}

#[derive(Clone, Debug)]
pub struct ExtremelyRandomizedCanonicalIntervalTree {
    root: Node<ExtremelyRandomizedCanonicalIntervalSplit>,
    config: ExtremelyRandomizedCanonicalIntervalTreeConfig,
}

impl ClassificationTree for ExtremelyRandomizedCanonicalIntervalTree {
    type TreeConfig = ExtremelyRandomizedCanonicalIntervalForestConfig;
    fn from_classification_config(config: &Self::TreeConfig) -> Self {
        Self::new(ExtremelyRandomizedCanonicalIntervalTreeConfig {
            max_depth: config.classification_config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: config.classification_config.min_samples_split,
        })
    }
}

impl Tree for ExtremelyRandomizedCanonicalIntervalTree {
    type Config = ExtremelyRandomizedCanonicalIntervalTreeConfig;
    type SplitParameters = ExtremelyRandomizedCanonicalIntervalSplit;
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
        // Base case: samples are the same object
        let first_sample = &samples[0].data;
        let is_all_same_data = samples.iter().all(|v| &v.data == first_sample);
        if is_all_same_data {
            return true;
        }
        return false;
    }
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        let mut rng = thread_rng();
        // Generate a random interval
        let sample_len = samples[0].data.len();
        let start = rng.gen_range(0..sample_len - MIN_INTERVAL_LEN);
        let end = rng.gen_range(start + MIN_INTERVAL_LEN..sample_len);
        // Generate a random feature
        let feature = rng.gen_range(0..TOT_ATTRIBUTES);
        // Compute the feature in the interval for all samples, and keep unique values
        let mut thresholds = vec![0.0; samples.len()];
        for i in 0..samples.len() {
            thresholds[i] = compute_catch(feature)(&samples[i].data[start..end]);
        }
        thresholds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        thresholds.dedup();
        // Select a random threshold
        let threshold = match thresholds.len() {
            0 => panic!("Thresholds cannot be empty"),
            1 => thresholds[0],
            _ => thresholds[rng.gen_range(1..thresholds.len())],
        };

        (
            ExtremelyRandomizedCanonicalIntervalSplit {
                interval: (start, end),
                feature,
                threshold,
            },
            rng.gen_range(f64::EPSILON..1.0),
        )
    }
}
