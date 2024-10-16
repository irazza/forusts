use std::cmp::min;

use super::forest::{ClassificationForestConfig};
use crate::{
    forest::forest::{Forest, ClassificationForest},
    tree::decision_tree::DecisionTree,
    utils::structures::Sample,
    RandomGenerator,
};
use rand::{thread_rng, SeedableRng};

pub type RandomForestConfig = ClassificationForestConfig;

pub struct RandomForest {
    trees: Vec<DecisionTree>,
    config: RandomForestConfig,
    max_samples: usize,
}

impl Forest<DecisionTree> for RandomForest {
    type Config = RandomForestConfig;

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
        let max_samples = samples.len();
        self.fit_(&samples, max_samples, false, &mut random_state)
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
    fn get_trees(&self) -> &Vec<DecisionTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<DecisionTree> {
        &mut self.trees
    }
}
impl ClassificationForest<DecisionTree> for RandomForest {
    fn get_forest_config(&self) -> (&ClassificationForestConfig, &ClassificationForestConfig) {
        (&self.config, &self.config)
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
}
