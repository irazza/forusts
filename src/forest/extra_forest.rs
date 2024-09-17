use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::utils::structures::Sample;
use crate::{
    forest::forest::Forest,
    tree::extra_tree::ExtraTree,
};
use std::any::type_name;
use std::fmt::Debug;

use super::forest::{
    ClassificationForest, ClassificationForestConfig, ClassificationForestConfigTuning,
};


pub struct ExtraForestConfig{
    pub n_intervals: usize,
    pub n_attributes: usize,
    pub ts_length: usize,
    pub classification_config: ClassificationForestConfig,
}

#[derive(Clone, Debug)]
pub struct ExtraForest {
    trees: Vec<ExtraTree>,
    config: ExtraForest,
}

impl Forest<ExtraTree>
    for ExtraForest
{
    type Config = ExtraForestConfig;
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
    fn get_trees(&self) -> &Vec<ExtraTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<ExtraTree> {
        &mut self.trees
    }
}
impl ClassificationForest<ExtraTree>
    for ExtraForest
{
    fn get_forest_config(
        &self,
    ) -> (
        &ClassificationForestConfig,
        &ExtraForest,
    ) {
        (&self.config.classification_config, &self.config)
    }
}
