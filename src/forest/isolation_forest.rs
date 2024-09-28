use super::forest::OutlierForestConfig;
use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::isolation_tree::IsolationTree,
    utils::structures::Sample,
    RandomGenerator,
};
use rand::{thread_rng, SeedableRng};

pub type IsolationForestConfig = OutlierForestConfig;

pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    config: IsolationForestConfig,
    max_samples: usize,
}

impl Forest<IsolationTree> for IsolationForest {
    type Config = IsolationForestConfig;

    fn new(config: &Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config: config.clone(),
            max_samples: 0,
        }
    }
    fn fit(&mut self, samples: &mut [Sample], random_state: Option<RandomGenerator>) {
        let mut random_state =
            random_state.unwrap_or_else(|| RandomGenerator::from_rng(thread_rng()).unwrap());
        self.fit_(&samples, &mut random_state)
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
