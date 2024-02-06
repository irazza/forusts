use super::forest::OutlierForestConfig;
use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::isolation_tree::IsolationTree,
    utils::structures::Sample,
};

pub type IsolationForestConfig = OutlierForestConfig;

pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    config: IsolationForestConfig,
}

impl Forest<IsolationTree> for IsolationForest {
    type Config = IsolationForestConfig;
    type TuningType = f64;

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
    fn get_trees(&self) -> &Vec<IsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<IsolationTree> {
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
        self.score_samples(ds_test)
    }
}
impl OutlierForest<IsolationTree> for IsolationForest {
    fn get_forest_config(&self) -> &OutlierForestConfig {
        &self.config
    }
}
