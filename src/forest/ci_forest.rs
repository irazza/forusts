use super::forest::OutlierForestConfig;
use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::ci_tree::CITree,
    utils::structures::Sample,
    RandomGenerator,
};
use rand::{thread_rng, SeedableRng};

#[derive(Clone)]
pub struct CIForestConfig {
    pub n_intervals: usize,
    pub n_attributes: usize,
    pub outlier_config: OutlierForestConfig,
}

pub struct CIForest {
    trees: Vec<CITree>,
    config: CIForestConfig,
    max_samples: usize,
}

impl Forest<CITree> for CIForest {
    type Config = CIForestConfig;

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
    fn get_trees(&self) -> &Vec<CITree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<CITree> {
        &mut self.trees
    }
}
impl OutlierForest<CITree> for CIForest {
    fn get_forest_config(&self) -> (&OutlierForestConfig, &CIForestConfig) {
        (&self.config.outlier_config, &self.config)
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
}
