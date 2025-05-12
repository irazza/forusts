use std::cmp::min;

use super::forest::{ForestConfig, SUBSAMPLE_SIZE};
use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::isolation_tree::IsolationTree,
    utils::structures::Sample,
    RandomGenerator,
};
use rand::SeedableRng;

pub type IsolationForestConfig = ForestConfig;

pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    config: IsolationForestConfig,
    max_samples: usize,
}

impl Forest<IsolationTree> for IsolationForest {
    type Config = IsolationForestConfig;

    fn get_forest_config(&self) -> (&ForestConfig, &ForestConfig) {
        (&self.config, &self.config)
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
    fn get_trees(&self) -> &Vec<IsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<IsolationTree> {
        &mut self.trees
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn new(config: &Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config: config.clone(),
            max_samples: 0,
        }
    }
    fn fit(&mut self, samples: &mut [Sample], random_state: Option<RandomGenerator>) {
        let mut random_state = match random_state {
            Some(rng) => rng,
            None => RandomGenerator::from_rng(&mut rand::rng()),
        };
        let max_samples = min(SUBSAMPLE_SIZE, samples.len());
        self.fit_(&samples, max_samples, false, &mut random_state)
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
}

impl OutlierForest<IsolationTree> for IsolationForest {}
