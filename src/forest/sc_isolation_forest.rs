use super::forest::OutlierForestConfig;
use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::isolation_tree::IsolationTree,
    utils::structures::Sample,
};

pub type SCIsolationForestConfig = OutlierForestConfig;

pub struct SCIsolationForest {
    trees: Vec<IsolationTree>,
    config: SCIsolationForestConfig,
    max_samples: usize,
}

impl Forest<IsolationTree> for SCIsolationForest {
    type Config = SCIsolationForestConfig;
    type TuningType = f64;

    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config,
            max_samples: 0,
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
impl OutlierForest<IsolationTree> for SCIsolationForest {
    fn get_forest_config(&self) -> &OutlierForestConfig {
        &self.config
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
}
