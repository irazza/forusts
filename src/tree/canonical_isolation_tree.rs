use std::cmp::max;

use super::{node::Node, tree::SplitParameters};
use crate::{
    feature_extraction::{catch22::compute_catch, statistics::EULER_MASCHERONI},
    forest::{canonical_isolation_forest::CanonicalIsolationForestConfig, forest::OutlierTree},
    tree::tree::Tree,
    utils::structures::Sample,
};
use dashmap::DashMap;
use lazy_static::lazy_static;
use rand::{seq::SliceRandom, thread_rng, Rng};

pub const MIN_INTERVAL_LEN: usize = 20;
pub const TOT_ATTRIBUTES: usize = 25;
pub const MIN_INTERVAL_PERCENTAGE: f64 = 0.1;

lazy_static! {
    pub static ref CISOF_CACHE: DashMap<(usize, usize, usize, usize), f64> = DashMap::new();
}

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct CanonicalIsolationSplit {
    pub interval: (usize, usize),
    pub feature: usize,
    pub threshold: f64,
}
impl Eq for CanonicalIsolationSplit {}
impl Ord for CanonicalIsolationSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl SplitParameters for CanonicalIsolationSplit {
    fn split(&self, sample: &Sample, _is_train: bool) -> bool {
        let key_cache = (
            sample.data.as_ptr() as usize,
            self.interval.0,
            self.interval.1,
            self.feature,
        );
        if let Some(value) = CISOF_CACHE.get(&key_cache) {
            return *value.value() < self.threshold;
        }

        let feature = compute_catch(self.feature)(&sample.data[self.interval.0..self.interval.1]);
        // if CISOF_CACHE.len() > 1e8 as usize {
        //     CISOF_CACHE.clear();
        // }
        CISOF_CACHE.insert(key_cache, feature);
        return feature < self.threshold;
    }
    fn path_length<T: Tree<SplitParameters = Self>>(tree: &T, x: &Sample) -> f64 {
        let leaf = tree.predict_leaf(x);

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
        path_length + leaf.get_depth() as f64 - 1.0
    }
}

#[derive(Clone, Debug, Copy)]
pub struct CanonicalIsolationTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub n_intervals: usize,
    pub n_attributes: usize,
    pub ts_length: usize,
}

#[derive(Clone, Debug)]
pub struct CanonicalIsolationTree {
    root: Node<CanonicalIsolationSplit>,
    config: CanonicalIsolationTreeConfig,
    intervals: Vec<(usize, usize)>,
    attributes: Vec<usize>,
}

impl OutlierTree for CanonicalIsolationTree {
    type TreeConfig = CanonicalIsolationForestConfig;
    fn from_outlier_config(config: &Self::TreeConfig, max_samples: usize) -> Self {
        Self::new(CanonicalIsolationTreeConfig {
            max_depth: (max_samples as f64).max(2.0).log2().ceil() as usize + 1,
            min_samples_split: config.outlier_config.min_samples_split,
            n_intervals: config.n_intervals,
            n_attributes: config.n_attributes,
            ts_length: config.ts_length,
        })
    }
}

impl Tree for CanonicalIsolationTree {
    type Config = CanonicalIsolationTreeConfig;
    type SplitParameters = CanonicalIsolationSplit;
    fn new(config: Self::Config) -> Self {
        Self {
            root: Node::new(),
            config,
            intervals: {
                let mut rng = thread_rng();
                let mut intervals = vec![(0, 3000); config.n_intervals];
                if config.ts_length < MIN_INTERVAL_LEN {
                    panic!("Time series length too short");
                }
                let min_interval = max(
                    MIN_INTERVAL_LEN,
                    (config.ts_length as f64 * MIN_INTERVAL_PERCENTAGE).ceil() as usize,
                );
                for j in 0..config.n_intervals {
                    let start = rng.gen_range(0..config.ts_length - min_interval);
                    let end = rng.gen_range(start + min_interval..config.ts_length);
                    intervals[j] = (start, end);
                }
                intervals
            },
            attributes: {
                let mut attributes = (0..TOT_ATTRIBUTES).collect::<Vec<_>>();
                attributes.shuffle(&mut thread_rng());
                attributes.truncate(config.n_attributes);
                attributes
            },
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

        let interval_idx = rng.gen_range(0..self.intervals.len());
        let (start, end) = self.intervals[interval_idx];

        let feature_idx = rng.gen_range(0..self.attributes.len());
        let attribute = self.attributes[feature_idx];

        // Compute the thresholds for all the samples, and store them in the cache
        let mut thresholds = vec![0.0; samples.len()];
        for (i, sample) in samples.iter().enumerate() {
            // Create the key for the cache
            let key_cache = (sample.data.as_ptr() as usize, start, end, attribute);

            if let Some(value) = CISOF_CACHE.get(&key_cache) {
                thresholds[i] = *value.value();
                continue;
            }

            let feature = compute_catch(attribute)(&sample.data[start..end]);
            if CISOF_CACHE.len() > 1e8 as usize {
                CISOF_CACHE.clear();
            }
            CISOF_CACHE.insert(key_cache, feature);
            thresholds[i] = feature;
        }
        // Remove all minimum and maximum values from the thresholds
        let min_value = thresholds.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_value = thresholds.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        let thresholds = thresholds
            .iter()
            .filter(|&v| v != min_value && v != max_value)
            .collect::<Vec<_>>();
        
        let threshold = match thresholds.len() {
            0 => min_value, 
            _ => thresholds[rng.gen_range(0..thresholds.len())]
        };
        // Generate the new split
        (CanonicalIsolationSplit { 
            interval: (start, end), 
            feature: attribute, 
            threshold: *threshold
        }, 
        rng.gen_range(f64::EPSILON..1.0))
    }
}