use std::sync::Arc;

use crate::feature_extraction::catch22::compute_catch;
use crate::forest::forest::Forest;
use crate::grid_search_tuning;
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::tree::extra_tree::ExtraTree;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use rand::{seq::SliceRandom, thread_rng, Rng};

use super::forest::{
    ClassificationForest, ClassificationForestConfig, ClassificationForestConfigTuning,
};

pub const MIN_INTERVAL: usize = 20;
pub const TOTAL_ATTRIBUTES: usize = 25;
pub const N_ATTRIBUTES: usize = 8;

grid_search_tuning! {
pub struct ExtraCanonicalForestConfig [ExtraCanonicalForestConfigTuning]{
    pub n_intervals: usize,
    pub classification_config: ClassificationForestConfig [ClassificationForestConfigTuning],
}
}
impl TuningConfig for ExtraCanonicalForestConfigTuning {
    type Tree = ExtraTree;
    type Forest = ExtraCanonicalForest;
}

#[derive(Clone, Debug)]
pub struct ExtraCanonicalForest {
    trees: Vec<ExtraTree>,
    intervals: Vec<Vec<(usize, usize)>>,
    attributes: Vec<Vec<usize>>,
    config: ExtraCanonicalForestConfig,
}

impl Forest<ExtraTree> for ExtraCanonicalForest {
    type Config = ExtraCanonicalForestConfig;
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
    fn get_trees(&self) -> &Vec<ExtraTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<ExtraTree> {
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
        let breiman_distance = self.pairwise_breiman(&ds_test, &ds_train);
        k_nearest_neighbor(
            1,
            &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(),
            &breiman_distance,
        )
    }
}

impl ClassificationForest<ExtraTree> for ExtraCanonicalForest {
    fn get_forest_config(&self) -> &ClassificationForestConfig {
        &self.config.classification_config
    }
}
