use crate::{
    distance::distances::Distance,
    feature_extraction::statistics::{mean, EULER_MASCHERONI},
    forest::{distance_isolation_forest::DistanceIsolationForestConfig, forest::OutlierTree},
    utils::structures::Sample,
};
use std::hash::Hash;
use std::{fmt::Debug, sync::Arc};

use super::{
    node::Node,
    tree::{SplitParameters, Tree},
};

use rand::{seq::SliceRandom, thread_rng, Rng};

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct DistanceIsolationSplit {
    pub left_candidates: Vec<Arc<Vec<f64>>>,
    pub right_candidates: Vec<Arc<Vec<f64>>>,
    pub interval: (usize, usize),
    pub distance: Distance,
    pub band: f64,
}
impl Hash for DistanceIsolationSplit {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unreachable!();
    }
}
impl Eq for DistanceIsolationSplit {}
impl Ord for DistanceIsolationSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl SplitParameters for DistanceIsolationSplit {
    fn split(&self, sample: &Sample, _is_train: bool) -> bool {
        let left_distances = self
            .left_candidates
            .iter()
            .map(|c| {
                self.distance.distance(
                    &c,
                    &sample.data[self.interval.0..self.interval.1],
                    self.band,
                )
            })
            .collect::<Vec<_>>();
        let right_distances = self
            .right_candidates
            .iter()
            .map(|c| {
                self.distance.distance(
                    &c,
                    &sample.data[self.interval.0..self.interval.1],
                    self.band,
                )
            })
            .collect::<Vec<_>>();

        mean(&left_distances) < mean(&right_distances)
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
        path_length + leaf.get_depth() as f64
    }
}

#[derive(Clone, Debug)]
pub struct DistanceIsolationTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub distance: Distance,
}

#[derive(Clone, Debug)]
pub struct DistanceIsolationTree {
    root: Node<DistanceIsolationSplit>,
    config: DistanceIsolationTreeConfig,
}

impl OutlierTree for DistanceIsolationTree {
    type TreeConfig = DistanceIsolationForestConfig;
    fn from_outlier_config(config: &Self::TreeConfig, max_samples: usize) -> Self {
        Self::new(DistanceIsolationTreeConfig {
            max_depth: max_samples.ilog2() as usize + 1,
            min_samples_split: config.outlier_config.min_samples_split,
            distance: config.distance,
        })
    }
}

impl Tree for DistanceIsolationTree {
    type Config = DistanceIsolationTreeConfig;
    type SplitParameters = DistanceIsolationSplit;
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
        // Base case: all samples have the same features
        if samples.iter().all(|s| s.data == samples[0].data) {
            return true;
        }
        return false;
    }
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        // let mut rng = ChaCha8Rng::seed_from_u64(42 as u64);
        let mut rng = thread_rng();

        // Generate a random interval
        let start;
        let end;
        let ts_len = samples[0].data.len();
        let min_interval = (rng.gen_range(0.1..1.0) * ts_len as f64).ceil() as usize;
        if min_interval == ts_len {
            start = 0;
            end = ts_len;
        } else {
            start = rng.gen_range(0..ts_len - min_interval);
            end = rng.gen_range(start + min_interval..ts_len);
        }

        // Generate a random subsample (MaxFeatures) of elements
        let mut subsamples_indices = (0..samples.len()).collect::<Vec<_>>();
        subsamples_indices.shuffle(&mut rng);

        // Randomly split subsamples into two clusters
        let mut left_candidates = Vec::new();
        let mut right_candidates = Vec::new();
        for (i, s) in subsamples_indices.iter().enumerate() {
            if i % 2 == 0 {
                left_candidates.push(Arc::new(samples[*s].data[start..end].to_vec()));
            } else {
                right_candidates.push(Arc::new(samples[*s].data[start..end].to_vec()));
            }
        }

        (
            DistanceIsolationSplit {
                left_candidates,
                right_candidates,
                interval: (start, end),
                distance: self.config.distance,
                band: f64::log2((end - start) as f64).ceil(),
            },
            rng.gen_range(f64::EPSILON..1.0),
        )
    }
}
