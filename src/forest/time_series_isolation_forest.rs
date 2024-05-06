use crate::forest::forest::{Forest, OutlierForest};
use crate::grid_search_tuning;
use crate::tree::time_series_isolation_tree::TimeSeriesIsolationTree;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use std::fmt::Debug;

use super::forest::{OutlierForestConfig, OutlierForestConfigTuning};

grid_search_tuning! {
    pub struct TimeSeriesIsolationForestConfig[TimeSeriesIsolationForestConfigTuning] {
        pub n_intervals: usize,
        pub outlier_config: OutlierForestConfig [OutlierForestConfigTuning],
    }
    impl Debug for TimeSeriesIsolationForestConfig {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "TimeSeriesIsolationForestConfig")
        }
    }
}
impl TuningConfig for TimeSeriesIsolationForestConfigTuning {
    type Tree = TimeSeriesIsolationTree;
    type Forest = TimeSeriesIsolationForest;
}
pub struct TimeSeriesIsolationForest {
    trees: Vec<TimeSeriesIsolationTree>,
    config: TimeSeriesIsolationForestConfig,
    max_samples: usize,
}

impl Forest<TimeSeriesIsolationTree> for TimeSeriesIsolationForest {
    type Config = TimeSeriesIsolationForestConfig;
    type TuningType = f64;

    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config,
            max_samples: 0,
        }
    }
    fn fit(&mut self, data: &mut [Sample]) {
        self.fit_(data);
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
    fn get_trees(&self) -> &Vec<TimeSeriesIsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<TimeSeriesIsolationTree> {
        &mut self.trees
    }
    fn tuning_predict(&self, _ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Self::TuningType> {
        self.score_samples(ds_test)
    }
}

impl OutlierForest<TimeSeriesIsolationTree> for TimeSeriesIsolationForest {
    fn get_forest_config(&self) -> (&OutlierForestConfig, &TimeSeriesIsolationForestConfig) {
        (&self.config.outlier_config, &self.config)
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
}
