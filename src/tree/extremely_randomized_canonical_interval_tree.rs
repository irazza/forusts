use core::panic;
use std::sync::Arc;

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
use rand::{seq::SliceRandom, thread_rng, Rng};

pub const MIN_INTERVAL_LEN: usize = 20;
pub const TOT_ATTRIBUTES: usize = 25;

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct ExtremelyRandomizedCanonicalIntervalSplit {
    pub interval: (usize, usize),
    pub feature: usize,
    pub threshold: f64,
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
    pub n_attributes: usize,
    pub n_intervals: usize,
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
            n_attributes: config.n_attributes,
            n_intervals: config.n_intervals,
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
        // Generate n_intervals random intervals
        let mut intervals = Vec::new();
        for _ in 0..self.config.n_intervals {
            let start = rng.gen_range(0..samples[0].data.len() - MIN_INTERVAL_LEN);
            let end = rng.gen_range(start + MIN_INTERVAL_LEN..samples[0].data.len());
            intervals.push((start, end));
        }
        // Select n_attributes random features from catch22
        let mut attributes = (0..TOT_ATTRIBUTES).collect::<Vec<usize>>();
        attributes.shuffle(&mut rng);
        attributes.truncate(self.config.n_attributes);

        // For each interval, compute the randomly selected features
        let mut transformed_samples = Vec::new();
        for sample in samples {
            let mut transformed_sample = Vec::new();
            for (start, end) in &intervals {
                for i in &attributes {
                    transformed_sample.push(compute_catch(*i)(&sample.data[*start..*end]));
                }
            }
            transformed_samples.push(Sample {
                data: Arc::new(transformed_sample),
                target: sample.target,
            });
        }

        // Select random feature
        let feature = rng.gen_range(0..self.config.n_attributes * self.config.n_intervals);

        let mut thresholds = transformed_samples
            .iter()
            .map(|f| f.data[feature])
            .collect::<Vec<f64>>();
        thresholds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        thresholds.dedup();

        let threshold = match thresholds.len() {
            0 => panic!("Thresholds cannot be empty"),
            1 => thresholds[0],
            _ => thresholds[rng.gen_range(1..thresholds.len())],
        };
        let interval = intervals[feature / self.config.n_attributes];
        let feature = attributes[feature % self.config.n_attributes];
        (
            ExtremelyRandomizedCanonicalIntervalSplit {
                interval,
                feature,
                threshold,
            },
            rng.gen_range(f64::EPSILON..1.0),
        )
    }
}
