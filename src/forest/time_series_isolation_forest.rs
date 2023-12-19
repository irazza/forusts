use crate::feature_extraction::statistics::{mean, slope, std};
use crate::grid_search_tuning;
use crate::tree::tree::Tree;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::isolation_tree::IsolationTree,
};
use rand::{thread_rng, Rng};

use super::forest::{OutlierForestConfig, OutlierForestConfigTuning};

pub const MIN_INTERVAL_PERC : usize = 10;

grid_search_tuning! {
    pub struct TimeSeriesIsolationForestConfig[TimeSeriesIsolationForestConfigTuning] {
        pub n_intervals: usize,
        pub outlier_config: OutlierForestConfig [OutlierForestConfigTuning],
    }
}
impl TuningConfig for TimeSeriesIsolationForestConfigTuning {
    type Tree = IsolationTree;
    type Forest = TimeSeriesIsolationForest;
}

pub struct TimeSeriesIsolationForest {
    trees: Vec<IsolationTree>,
    min_interval_perc: usize,
    intervals: Vec<Vec<(usize, usize)>>,
    config: TimeSeriesIsolationForestConfig,
}

impl Forest<IsolationTree> for TimeSeriesIsolationForest {
    type Config = TimeSeriesIsolationForestConfig;
    type TuningType = f64;

    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            min_interval_perc: MIN_INTERVAL_PERC,
            intervals: Vec::new(),
            config,
        }
    }
    fn fit(&mut self, data: &mut [Sample<'_>]) {
        self.fit_(data);
    }
    fn predict(&self, data: &[Sample<'_>]) -> Vec<isize> {
        self.predict_(data)
    }
    fn compute_intervals(&mut self, n_features: usize) {
        // Generate n_intervals, with random start and end
        let min_interval_length = (n_features as f64 * self.min_interval_perc as f64 / 100.0).round() as usize;
        for _i in 0..self.config.outlier_config.n_trees {
            let mut intervals = Vec::new();
            for _j in 0..self.config.n_intervals {
                let start = thread_rng().gen_range(0..n_features - min_interval_length);
                let end = thread_rng().gen_range(start + min_interval_length..n_features);
                intervals.push((start, end));
            }
            self.intervals.push(intervals);
        }
    }
    fn get_trees(&self) -> &Vec<IsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<IsolationTree> {
        &mut self.trees
    }
    fn transform<'a>(&self, data: &[Sample<'a>], intervals_index: usize) -> Vec<Sample<'a>> {
        let n_samples = data.len();
        let mut transformed_data: Vec<Sample<'_>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in self.intervals[intervals_index].iter().copied() {
                let mean = mean(&data[j].data[start..end]);
                let std = std(&data[j].data[start..end]);
                let slope = slope(&data[j].data[start..end]);
                sample.extend([mean, std, slope].into_iter());
            }
            transformed_data.push(Sample {
                data: std::borrow::Cow::Owned(sample),
                target: data[j].target,
            });
        }
        transformed_data
    }
    fn tuning_predict(&self, data: &[Sample<'_>]) -> Vec<Self::TuningType> {
        self.score_samples(data)
    }
}

impl OutlierForest for TimeSeriesIsolationForest {
    fn get_forest_config(&self) -> &OutlierForestConfig {
        &self.config.outlier_config
    }
}
