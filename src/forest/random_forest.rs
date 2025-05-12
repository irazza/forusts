use crate::{
    forest::forest::{ClassificationForest, Forest, ForestConfig},
    tree::decision_tree::DecisionTree,
    utils::structures::Sample,
    RandomGenerator,
};
use rand::{rng, SeedableRng};

pub type RandomForestConfig = ForestConfig;

pub struct RandomForest {
    trees: Vec<DecisionTree>,
    config: RandomForestConfig,
    max_samples: usize,
}

impl Forest<DecisionTree> for RandomForest {
    type Config = RandomForestConfig;

    fn get_forest_config(&self) -> (&ForestConfig, &ForestConfig) {
        (&self.config, &self.config)
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
    fn get_trees(&self) -> &Vec<DecisionTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<DecisionTree> {
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
        let max_samples = samples.len();
        self.fit_(&samples, max_samples, true, &mut random_state)
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
}
impl ClassificationForest<DecisionTree> for RandomForest {}
