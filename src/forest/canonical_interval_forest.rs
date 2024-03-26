use std::sync::Arc;

use crate::feature_extraction::catch22::compute_catch;
use crate::forest::forest::{ClassificationForest, Forest};
use crate::grid_search_tuning;
use crate::tree::decision_tree::DecisionTree;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use rand::{seq::SliceRandom, thread_rng, Rng};

use super::forest::{ClassificationForestConfig, ClassificationForestConfigTuning};

pub const MIN_INTERVAL: usize = 20;
pub const TOTAL_ATTRIBUTES: usize = 25;
pub const N_ATTRIBUTES: usize = 8;

grid_search_tuning! {
pub struct CanonicalIntervalForestConfig [CanonicalIntervalForestConfigTuning]{
    pub n_intervals: usize,
    pub min_interval_length: usize,
    pub classification_config: ClassificationForestConfig [ClassificationForestConfigTuning],
}
}
impl TuningConfig for CanonicalIntervalForestConfigTuning {
    type Tree = DecisionTree;
    type Forest = CanonicalIntervalForest;
}
pub struct CanonicalIntervalForest {
    trees: Vec<DecisionTree>,
    intervals: Vec<Vec<(usize, usize)>>,
    attributes: Vec<Vec<usize>>,
    config: CanonicalIntervalForestConfig,
}

impl Forest<DecisionTree> for CanonicalIntervalForest {
    type Config = CanonicalIntervalForestConfig;
    type TuningType = isize;
    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            intervals: Vec::new(),
            attributes: Vec::new(),
            config,
        }
    }
    fn fit(&mut self, data: &mut [Sample]) {
        self.fit_(data);
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
    fn compute_intervals(&mut self, n_features: usize) {
        // Generate n_intervals, with random start and end
        for _i in 0..self.config.classification_config.n_trees {
            let mut intervals = Vec::new();
            for _j in 0..self.config.n_intervals {
                let start = thread_rng().gen_range(0..n_features - MIN_INTERVAL);
                let end = thread_rng().gen_range(start + MIN_INTERVAL..n_features);
                intervals.push((start, end));
            }
            // We leverage the larger total feature space and inject additional diversity into the ensemble by randomly sampling the 25 features for each tree.
            let mut attributes = (0..TOTAL_ATTRIBUTES).collect::<Vec<_>>();
            attributes.shuffle(&mut thread_rng());
            self.attributes.push(attributes[..N_ATTRIBUTES].to_vec());

            self.intervals.push(intervals);
        }
    }
    fn get_trees(&self) -> &Vec<DecisionTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<DecisionTree> {
        &mut self.trees
    }
    fn transform<'a>(&self, data: &[Sample], tree_index: usize) -> Vec<Sample> {
        let n_samples = data.len();
        let mut transformed_data: Vec<Sample> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in self.intervals[tree_index].iter().copied() {
                for i in 0..N_ATTRIBUTES {
                    sample.push(compute_catch(self.attributes[tree_index][i])(
                        &data[j].data[start..end],
                    ));
                }
            }
            transformed_data.push(Sample {
                data: Arc::new(sample),
                target: data[j].target,
            });
        }
        transformed_data
    }
    fn tuning_predict(
        &self,
        ds_train: &[Sample],
        ds_test: &[Sample],
    ) -> Vec<Self::TuningType> {
        self.predict(ds_test)
    }
}

impl ClassificationForest<DecisionTree> for CanonicalIntervalForest {
    fn get_forest_config(&self) -> &ClassificationForestConfig {
        &self.config.classification_config
    }
}
