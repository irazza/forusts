use crate::{
    forest::forest::{ClassificationForest, Forest, ForestConfig},
    utils::structures::{IntervalType, Sample},
    tree::ci_tree::CITree,
    RandomGenerator,
};
use rand::{thread_rng, SeedableRng};

#[derive(Clone)]
pub struct CIForestConfig {
    pub n_intervals: IntervalType,
    pub n_attributes: usize,
    pub classification_config: ForestConfig,
}

pub struct CIForest {
    trees: Vec<CITree>,
    config: CIForestConfig,
    max_samples: usize,
}

impl Forest<CITree> for CIForest {
    type Config = CIForestConfig;

    fn get_forest_config(&self) -> (&ForestConfig, &CIForestConfig) {
        (&self.config.classification_config, &self.config)
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
    fn get_trees(&self) -> &Vec<CITree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<CITree> {
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
        let mut random_state =
            random_state.unwrap_or_else(|| RandomGenerator::from_rng(thread_rng()).unwrap());
        let max_samples = samples.len();
        self.fit_(&samples, max_samples, true, &mut random_state)
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
}
impl ClassificationForest<CITree> for CIForest {}
