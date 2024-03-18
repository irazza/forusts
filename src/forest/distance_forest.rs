use std::borrow::Cow;

use crate::feature_extraction::statistics::zscore;
use crate::tree::distance_tree::DistanceTree;
use crate::{tree::decision_tree::DecisionTree, utils::structures::Sample};

use crate::forest::forest::{ClassificationForest, Forest};

use super::forest::ClassificationForestConfig;

pub type DistanceForestConfig = ClassificationForestConfig;
pub struct DistanceForest {
    trees: Vec<DistanceTree>,
    config: DistanceForestConfig,
}
impl Forest<DistanceTree> for DistanceForest {
    type Config = DistanceForestConfig;
    type TuningType = isize;
    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config,
        }
    }
    fn fit(&mut self, data: &mut [Sample<'_>]) {
        //let mut data = data.iter().map(|x| Sample{ data: Cow::Owned(zscore(&x.data)), target: x.target}).collect::<Vec<_>>();
        self.fit_(data)
    }
    fn predict(&self, data: &[Sample<'_>]) -> Vec<isize> {
        self.predict_(data)
    }
    fn compute_intervals(&mut self, _n_features: usize) {}
    fn get_trees(&self) -> &Vec<DistanceTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<DistanceTree> {
        &mut self.trees
    }
    fn transform<'a>(&self, data: &[Sample<'a>], _intervals_index: usize) -> Vec<Sample<'a>> {
        data.to_vec()
    }
    fn tuning_predict(
        &self,
        ds_train: &[Sample<'_>],
        ds_test: &[Sample<'_>],
    ) -> Vec<Self::TuningType> {
        self.predict(ds_test)
    }
}
impl ClassificationForest<DistanceTree> for DistanceForest {
    fn get_forest_config(&self) -> &ClassificationForestConfig {
        &self.config
    }
}
