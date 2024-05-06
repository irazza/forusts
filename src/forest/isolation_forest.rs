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
    max_samples: usize,
}

impl Forest<IsolationTree> for IsolationForest {
    type Config = IsolationForestConfig;
    type TuningType = f64;

    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config,
            max_samples: 0,
        }
    }
    fn fit(&mut self, data: &mut [Sample]) {
        self.fit_(data)
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
    fn get_trees(&self) -> &Vec<IsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<IsolationTree> {
        &mut self.trees
    }
    fn tuning_predict(&self, _ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Self::TuningType> {
        self.score_samples(ds_test)
    }
}
impl OutlierForest<IsolationTree> for IsolationForest {
    fn get_forest_config(&self) -> (&OutlierForestConfig, &OutlierForestConfig) {
        (&self.config, &self.config)
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
}
