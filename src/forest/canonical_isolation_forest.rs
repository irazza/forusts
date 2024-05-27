use super::forest::{OutlierForestConfig, OutlierForestConfigTuning};
use crate::forest::forest::{Forest, OutlierForest};
use crate::grid_search_tuning;
use crate::tree::canonical_isolation_tree::CanonicalIsolationTree;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use std::any::type_name;
use std::fmt::Debug;

grid_search_tuning! {
pub struct CanonicalIsolationForestConfig [CanonicalIsolationForestConfigTuning]{
    pub n_intervals: usize,
    pub n_attributes: usize,
    pub ts_length: usize,
    pub outlier_config: OutlierForestConfig [OutlierForestConfigTuning],
}
impl Debug for CanonicalIsolationForestConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let full_type_name = type_name::<Self>();
        let struct_name = full_type_name.split("::").last().unwrap_or(full_type_name);
        let struct_name = struct_name.chars().take(struct_name.len() - 6).collect::<String>();
        write!(
            f,
            "{}_{}_{}_{}",
            struct_name, self.outlier_config.n_trees, self.n_intervals, self.n_attributes
        )
    }
}
}
impl TuningConfig for CanonicalIsolationForestConfigTuning {
    type Tree = CanonicalIsolationTree;
    type Forest = CanonicalIsolationForest;
}
pub struct CanonicalIsolationForest {
    trees: Vec<CanonicalIsolationTree>,
    config: CanonicalIsolationForestConfig,
    max_samples: usize,
}

impl Forest<CanonicalIsolationTree> for CanonicalIsolationForest {
    type Config = CanonicalIsolationForestConfig;
    type TuningType = isize;
    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config,
            max_samples: 0,
        }
    }
    fn fit(&mut self, data: &mut [Sample]) {
        self.fit_(data);
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
    fn get_trees(&self) -> &Vec<CanonicalIsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<CanonicalIsolationTree> {
        &mut self.trees
    }
    fn tuning_predict(&self, _ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Self::TuningType> {
        self.predict(ds_test)
    }
}

impl OutlierForest<CanonicalIsolationTree> for CanonicalIsolationForest {
    fn get_forest_config(&self) -> (&OutlierForestConfig, &CanonicalIsolationForestConfig) {
        (&self.config.outlier_config, &self.config)
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
}
