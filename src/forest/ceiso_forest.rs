use std::cmp::min;

use super::{
    eiso_forest::ExtensionLevel,
    forest::{ForestConfig, SUBSAMPLE_SIZE},
};
use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::ceiso_tree::CEIsoTree,
    utils::structures::{IntervalType, Sample},
    RandomGenerator,
};
use rand::SeedableRng;

#[derive(Clone)]
pub struct CEIsoForestConfig {
    pub n_intervals: IntervalType,
    pub n_attributes: usize,
    pub extension_level: ExtensionLevel,
    pub outlier_config: ForestConfig,
}

pub struct CEIsoForest {
    trees: Vec<CEIsoTree>,
    config: CEIsoForestConfig,
    max_samples: usize,
}

impl Forest<CEIsoTree> for CEIsoForest {
    type Config = CEIsoForestConfig;

    fn get_forest_config(&self) -> (&ForestConfig, &CEIsoForestConfig) {
        (&self.config.outlier_config, &self.config)
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
    fn get_trees(&self) -> &Vec<CEIsoTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<CEIsoTree> {
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
impl OutlierForest<CEIsoTree> for CEIsoForest {}
