use crate::feature_extraction::catch22::CATCH22;
use crate::feature_extraction::statistics::{mean, slope, std, zscore};
use crate::forest::forest::{ClassificationForest, Forest};
use crate::grid_search_tuning;
use crate::tree::{decision_tree::DecisionTree, tree::Tree};
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use rand::{seq::SliceRandom, thread_rng, Rng};
use rayon::prelude::*;

use super::forest::{ClassificationForestConfig, ClassificationForestConfigTuning};

pub const MIN_INTERVAL: usize = 10;
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
    fn fit(&mut self, data: &mut [Sample<'_>]) {
        self.fit_(data);
    }
    fn predict(&self, data: &[Sample<'_>]) -> Vec<isize> {
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
    fn transform<'a>(&self, data: &[Sample<'a>], tree_index: usize) -> Vec<Sample<'a>> {
        let n_samples = data.len();
        let mut transformed_data: Vec<Sample<'_>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in self.intervals[tree_index].iter().copied() {
                for i in 0..N_ATTRIBUTES {
                    sample.extend(
                        [CATCH22::get(self.attributes[tree_index][i])(
                            &data[j].data[start..end],
                        )]
                        .iter(),
                    );
                }
            }
            transformed_data.push(Sample {
                data: std::borrow::Cow::Owned(sample),
                target: data[j].target,
            });
        }
        transformed_data
    }
    fn tuning_predict(&self, ds_train: &[Sample<'_>], ds_test: &[Sample<'_>]) -> Vec<Self::TuningType> {
        self.predict(ds_test)
    }
}

impl ClassificationForest for CanonicalIntervalForest {
    fn get_forest_config(&self) -> &ClassificationForestConfig {
        &self.config.classification_config
    }
}
