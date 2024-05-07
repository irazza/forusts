use core::panic;
use std::sync::Arc;

use crate::{
    feature_extraction::statistics::{mean, slope, stddev, EULER_MASCHERONI},
    forest::{forest::OutlierTree, time_series_isolation_forest::TimeSeriesIsolationForestConfig},
    utils::structures::Sample,
};

use super::{
    node::Node,
    tree::{SplitParameters, Tree},
};

use rand::{thread_rng, Rng};

pub const MIN_INTERVAL_LEN: usize = 3;
pub const N_ATTRIBUTES: usize = 3;

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct TimeSeriesIsolationSplit {
    pub interval: (usize, usize),
    pub feature: usize,
    pub threshold: f64,
}
impl Eq for TimeSeriesIsolationSplit {}
impl Ord for TimeSeriesIsolationSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl SplitParameters for TimeSeriesIsolationSplit {
    fn split(&self, sample: &Sample, _is_train: bool) -> bool {
        let feature = match self.feature {
            0 => mean(&sample.data[self.interval.0..self.interval.1]),
            1 => stddev(&sample.data[self.interval.0..self.interval.1]),
            2 => slope(&sample.data[self.interval.0..self.interval.1]),
            _ => panic!("Invalid feature index"),
        };
        feature < self.threshold
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
pub struct TimeSeriesIsolationTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub n_intervals: usize,
}

#[derive(Clone, Debug)]
pub struct TimeSeriesIsolationTree {
    root: Node<TimeSeriesIsolationSplit>,
    config: TimeSeriesIsolationTreeConfig,
}

impl OutlierTree for TimeSeriesIsolationTree {
    type TreeConfig = TimeSeriesIsolationForestConfig;
    fn from_outlier_config(config: &Self::TreeConfig, max_samples: usize) -> Self {
        Self::new(TimeSeriesIsolationTreeConfig {
            max_depth: max_samples.ilog2() as usize + 1,
            min_samples_split: config.outlier_config.min_samples_split,
            n_intervals: config.n_intervals,
        })
    }
}

impl Tree for TimeSeriesIsolationTree {
    type Config = TimeSeriesIsolationTreeConfig;
    type SplitParameters = TimeSeriesIsolationSplit;
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
        // For each interval, compute the randomly selected features
        let mut transformed_samples = Vec::new();
        for sample in samples {
            let mut transformed_sample = Vec::new();
            for (start, end) in &intervals {
                transformed_sample.push(mean(&sample.data[*start..*end]));
                transformed_sample.push(stddev(&sample.data[*start..*end]));
                transformed_sample.push(slope(&sample.data[*start..*end]));
            }
            transformed_samples.push(Sample {
                data: Arc::new(transformed_sample),
                target: sample.target,
            });
        }

        let feature = rng.gen_range(0..N_ATTRIBUTES * self.config.n_intervals);
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
        let interval = intervals[feature / N_ATTRIBUTES];
        let feature = feature % N_ATTRIBUTES;
        (
            TimeSeriesIsolationSplit {
                interval,
                feature,
                threshold,
            },
            rng.gen_range(f64::EPSILON..1.0),
        )
    }
}
