use crate::{tree::decision_tree::DecisionTree, utils::structures::Sample};

use crate::forest::forest::{ClassificationForest, Forest};

use super::forest::ClassificationForestConfig;

pub type RandomForestConfig = ClassificationForestConfig;
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    config: RandomForestConfig,
}
impl Forest<DecisionTree> for RandomForest {
    type Config = RandomForestConfig;
    type TuningType = isize;
    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config,
        }
    }
    fn fit(&mut self, data: &mut [Sample<'_>]) {
        self.fit_(data)
    }
    fn predict(&self, data: &[Sample<'_>]) -> Vec<isize> {
        self.predict_(data)
    }
    fn compute_intervals(&mut self, _n_features: usize) {}
    fn get_trees(&self) -> &Vec<DecisionTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<DecisionTree> {
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
impl ClassificationForest<DecisionTree> for RandomForest {
    fn get_forest_config(&self) -> &ClassificationForestConfig {
        &self.config
    }
}
