use crate::feature_extraction::catch22::{compute_catch, compute_catch_features};
use crate::forest::forest::{Forest, OutlierForest};
use crate::grid_search_tuning;
use crate::tree::sc_isolation_tree::SCIsolationTree;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::fmt::Debug;
use std::sync::Arc;

use super::forest::{OutlierForestConfig, OutlierForestConfigTuning};

pub const MIN_INTERVAL: usize = 20;
pub const TOTAL_ATTRIBUTES: usize = 25;
pub const N_ATTRIBUTES: usize = 8;

grid_search_tuning! {
    pub struct CanonicalSCIsolationForestConfig[CanonicalSCIsolationForestConfigTuning] {
        pub n_intervals: usize,
        pub outlier_config: OutlierForestConfig [OutlierForestConfigTuning],
    }
    impl Debug for CanonicalSCIsolationForestConfig {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "CanonicalSCIsolationForestConfig")
        }
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
    fn get_trees(&self) -> &Vec<SCIsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<SCIsolationTree> {
        &mut self.trees
    }
    fn tuning_predict(&self, _ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Self::TuningType> {
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
