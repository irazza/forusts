use std::sync::Arc;

use crate::feature_extraction::catch22::{compute_catch, compute_catch_features};
use crate::forest::forest::{Forest, OutlierForest};
use crate::grid_search_tuning;
use crate::tree::sc_isolation_tree::SCIsolationTree;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use rand::{seq::SliceRandom, thread_rng, Rng};

use super::forest::{OutlierForestConfig, OutlierForestConfigTuning};

pub const MIN_INTERVAL: usize = 20;
pub const TOTAL_ATTRIBUTES: usize = 25;
pub const N_ATTRIBUTES: usize = 8;

grid_search_tuning! {
    pub struct CanonicalSCIsolationForestConfig[CanonicalSCIsolationForestConfigTuning] {
        pub n_intervals: usize,
        pub outlier_config: OutlierForestConfig [OutlierForestConfigTuning],
    }
}
impl TuningConfig for CanonicalSCIsolationForestConfigTuning {
    type Tree = SCIsolationTree;
    type Forest = CanonicalSCIsolationForest;
}

#[derive(Clone)]
pub struct CanonicalSCIsolationForest {
    trees: Vec<SCIsolationTree>,
    intervals: Vec<Vec<(usize, usize)>>,
    attributes: Vec<Vec<usize>>,
    config: CanonicalSCIsolationForestConfig,
    max_samples: usize,
}

impl Forest<SCIsolationTree> for CanonicalSCIsolationForest {
    type Config = CanonicalSCIsolationForestConfig;
    type TuningType = f64;

    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            intervals: Vec::new(),
            attributes: Vec::new(),
            config,
            max_samples: 0,
        }
    }
    fn fit(&mut self, data: &mut [Sample]) {
        self.fit_(&data);
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
    fn compute_intervals(&mut self, n_features: usize) {
        // Generate n_intervals, with random start and end
        for _i in 0..self.config.outlier_config.n_trees {
            // let n_intervals = thread_rng().gen_range(1..=self.config.n_intervals);
            // let intervals = (0..=n_features-(n_features/n_intervals)).step_by(n_features/n_intervals).into_iter().map(|start| (start, start + n_features/n_intervals)).collect();
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
    fn get_trees(&self) -> &Vec<SCIsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<SCIsolationTree> {
        &mut self.trees
    }
    fn transform<'a>(&self, data: &[Sample], tree_index: usize) -> Vec<Sample> {
        let n_samples = data.len();
        let mut transformed_data: Vec<Sample> = Vec::new();
        for j in 0..n_samples {
            let mut sample = compute_catch_features(&data[j].data);
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
        self.score_samples(ds_test)
    }
}

impl OutlierForest<SCIsolationTree> for CanonicalSCIsolationForest {
    fn get_forest_config(&self) -> &OutlierForestConfig {
        &self.config.outlier_config
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
}
