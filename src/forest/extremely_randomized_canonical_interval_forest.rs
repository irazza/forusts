use crate::grid_search_tuning;
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use crate::{
    forest::forest::Forest,
    tree::extremely_randomized_canonical_interval_tree::ExtremelyRandomizedCanonicalIntervalTree,
};
use std::any::type_name;
use std::fmt::Debug;

use super::forest::{
    ClassificationForest, ClassificationForestConfig, ClassificationForestConfigTuning,
};

grid_search_tuning! {
pub struct ExtremelyRandomizedCanonicalIntervalForestConfig [ExtremelyRandomizedCanonicalIntervalForestConfigTuning]{
    pub n_intervals: usize,
    pub n_attributes: usize,
    pub ts_length: usize,
    pub classification_config: ClassificationForestConfig [ClassificationForestConfigTuning],
}
impl Debug for ExtremelyRandomizedCanonicalIntervalForestConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let full_type_name = type_name::<Self>();
        let struct_name = full_type_name.split("::").last().unwrap_or(full_type_name);
        let struct_name = struct_name.chars().take(struct_name.len() - 6).collect::<String>();
        write!(
            f,
            "{}_{}_{}_{}",
            struct_name, self.classification_config.n_trees, self.n_intervals, self.n_attributes
        )
    }
}
}
impl TuningConfig for ExtremelyRandomizedCanonicalIntervalForestConfigTuning {
    type Tree = ExtremelyRandomizedCanonicalIntervalTree;
    type Forest = ExtremelyRandomizedCanonicalIntervalForest;
}

#[derive(Clone, Debug)]
pub struct ExtremelyRandomizedCanonicalIntervalForest {
    trees: Vec<ExtremelyRandomizedCanonicalIntervalTree>,
    config: ExtremelyRandomizedCanonicalIntervalForestConfig,
}

impl Forest<ExtremelyRandomizedCanonicalIntervalTree>
    for ExtremelyRandomizedCanonicalIntervalForest
{
    type Config = ExtremelyRandomizedCanonicalIntervalForestConfig;
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
    fn get_trees(&self) -> &Vec<ExtremelyRandomizedCanonicalIntervalTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<ExtremelyRandomizedCanonicalIntervalTree> {
        &mut self.trees
    }
    fn tuning_predict(&self, ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Self::TuningType> {
        let breiman_distance = self.pairwise_breiman(&ds_test, &ds_train);
        k_nearest_neighbor(
            1,
            &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(),
            &breiman_distance,
        )
    }
}
impl ClassificationForest<ExtremelyRandomizedCanonicalIntervalTree>
    for ExtremelyRandomizedCanonicalIntervalForest
{
    fn get_forest_config(
        &self,
    ) -> (
        &ClassificationForestConfig,
        &ExtremelyRandomizedCanonicalIntervalForestConfig,
    ) {
        (&self.config.classification_config, &self.config)
    }
}
