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
use core::panic;
use std::cmp::max;
use dashmap::DashMap;
use lazy_static::lazy_static;
use rand::{seq::SliceRandom, thread_rng, Rng};

pub const MIN_INTERVAL_LEN: usize = 20;
pub const TOT_ATTRIBUTES: usize = 25;
pub const MIN_INTERVAL_PERCENTAGE: f64 = 0.1;

lazy_static! {
    pub static ref ERCIF_CACHE: DashMap<(usize, usize, usize, usize), f64> = DashMap::new();
}

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
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
        let key_cache = (
            sample.data.as_ptr() as usize,
            self.interval.0,
            self.interval.1,
            self.feature,
        );
        if let Some(value) = ERCIF_CACHE.get(&key_cache) {
            return *value.value() < self.threshold;
        }

        let feature = compute_catch(self.feature)(&sample.data[self.interval.0..self.interval.1]);
        if ERCIF_CACHE.len() > 1e8 as usize {
            ERCIF_CACHE.clear();
        }
        ERCIF_CACHE.insert(key_cache, feature);
        return feature < self.threshold;
    }
    fn path_length<T: Tree<SplitParameters = Self>>(_tree: &T, _x: &Sample) -> f64 {
        unreachable!();
    }
}

#[derive(Clone, Debug, Copy)]
pub struct ExtremelyRandomizedCanonicalIntervalTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub n_intervals: usize,
    pub n_attributes: usize,
    pub ts_length: usize,
}

#[derive(Clone, Debug)]
pub struct ExtremelyRandomizedCanonicalIntervalTree {
    root: Node<ExtremelyRandomizedCanonicalIntervalSplit>,
    config: ExtremelyRandomizedCanonicalIntervalTreeConfig,
    intervals: Vec<(usize, usize)>,
    attributes: Vec<usize>,
}

impl ClassificationTree for ExtremelyRandomizedCanonicalIntervalTree {
    type TreeConfig = ExtremelyRandomizedCanonicalIntervalForestConfig;
    fn from_classification_config(config: &Self::TreeConfig) -> Self {
        Self::new(ExtremelyRandomizedCanonicalIntervalTreeConfig {
            max_depth: config.classification_config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: config.classification_config.min_samples_split,
            n_intervals: config.n_intervals,
            n_attributes: config.n_attributes,
            ts_length: config.ts_length,
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

            if let Some(value) = ERCIF_CACHE.get(&key_cache) {
                thresholds[i] = *value.value();
                continue;
            }

            let feature = compute_catch(attribute)(&sample.data[start..end]);
            if ERCIF_CACHE.len() > 1e8 as usize {
                ERCIF_CACHE.clear();
            }
            ERCIF_CACHE.insert(key_cache, feature);
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
        (
            ExtremelyRandomizedCanonicalIntervalSplit {
                interval: (start, end),
                feature: attribute,
                threshold: *threshold,
            },
            rng.gen_range(f64::EPSILON..1.0),
        )
    }
}