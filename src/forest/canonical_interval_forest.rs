use crate::forest::forest::{ClassificationForest, Forest};
use crate::grid_search_tuning;
use crate::tree::canonical_interval_tree::CanonicalIntervalTree;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use std::fmt::Debug;

use super::forest::{ClassificationForestConfig, ClassificationForestConfigTuning};

grid_search_tuning! {
pub struct CanonicalIntervalForestConfig [CanonicalIntervalForestConfigTuning]{
    pub n_intervals: usize,
    pub n_attributes: usize,
    pub classification_config: ClassificationForestConfig [ClassificationForestConfigTuning],
}
impl Debug for CanonicalIntervalForestConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CanonicalIntervalForestConfig")
    }
}
}
impl TuningConfig for CanonicalIntervalForestConfigTuning {
    type Tree = CanonicalIntervalTree;
    type Forest = CanonicalIntervalForest;
}
pub struct CanonicalIntervalForest {
    trees: Vec<CanonicalIntervalTree>,
    config: CanonicalIntervalForestConfig,
}

impl Forest<CanonicalIntervalTree> for CanonicalIntervalForest {
    type Config = CanonicalIntervalForestConfig;
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
    fn get_trees(&self) -> &Vec<CanonicalIntervalTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<CanonicalIntervalTree> {
        &mut self.trees
    }
    fn tuning_predict(&self, _ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Self::TuningType> {
        self.predict(ds_test)
    }
}

impl ClassificationForest<CanonicalIntervalTree> for CanonicalIntervalForest {
    fn get_forest_config(&self) -> (&ClassificationForestConfig, &CanonicalIntervalForestConfig) {
        (&self.config.classification_config, &self.config)
    }
}
