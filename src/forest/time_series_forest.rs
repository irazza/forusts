use crate::forest::forest::{ClassificationForest, Forest};
use crate::grid_search_tuning;
use crate::tree::time_series_tree::TimeSeriesTree;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use std::fmt::Debug;

use super::forest::{ClassificationForestConfig, ClassificationForestConfigTuning};

grid_search_tuning! {
pub struct TimeSeriesForestConfig [TimeSeriesForestConfigTuning]{
    pub n_intervals: usize,
    pub classification_config: ClassificationForestConfig [ClassificationForestConfigTuning],
}
impl Debug for TimeSeriesForestConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TimeSeriesForestConfig")
    }
}
}
impl TuningConfig for TimeSeriesForestConfigTuning {
    type Tree = TimeSeriesTree;
    type Forest = TimeSeriesForest;
}
pub struct TimeSeriesForest {
    trees: Vec<TimeSeriesTree>,
    config: TimeSeriesForestConfig,
}

impl Forest<TimeSeriesTree> for TimeSeriesForest {
    type Config = TimeSeriesForestConfig;
    type TuningType = isize;
    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config,
        }
    }
    fn fit(&mut self, data: &mut [Sample]) {
        self.fit_(data);
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
    fn get_trees(&self) -> &Vec<TimeSeriesTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<TimeSeriesTree> {
        &mut self.trees
    }
    fn tuning_predict(&self, _ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Self::TuningType> {
        self.predict(ds_test)
    }
}

impl ClassificationForest<TimeSeriesTree> for TimeSeriesForest {
    fn get_forest_config(&self) -> (&ClassificationForestConfig, &TimeSeriesForestConfig) {
        (&self.config.classification_config, &self.config)
    }
}
