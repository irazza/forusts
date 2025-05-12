use super::{ciso_forest::CIsoForestConfig, forest::ForestConfig};
use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::ciso_tree::CIsoTree,
    utils::structures::Sample,
    RandomGenerator,
};
use rand::{rng, SeedableRng};

pub struct ERCIForest {
    trees: Vec<CIsoTree>,
    config: CIsoForestConfig,
    max_samples: usize,
}

impl Forest<CIsoTree> for ERCIForest {
    type Config = CIsoForestConfig;

    fn get_forest_config(&self) -> (&ForestConfig, &CIsoForestConfig) {
        (&self.config.outlier_config, &self.config)
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
    fn get_trees(&self) -> &Vec<CIsoTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<CIsoTree> {
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

impl OutlierForest<CIsoTree> for ERCIForest {}
