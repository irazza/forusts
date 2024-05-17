use core::fmt::Debug;
use std::any::type_name;
use std::fmt::Formatter;

use crate::distance::distances::Distance;
use crate::grid_search_tuning;
use crate::tree::distance_isolation_tree::DistanceIsolationTree;
use crate::utils::structures::Sample;

use crate::forest::forest::{Forest, OutlierForest};

use super::forest::{OutlierForestConfig, OutlierForestConfigTuning};

grid_search_tuning! {
    pub struct DistanceIsolationForestConfig[DistanceIsolationForestTuning]{
        pub distance: Distance,
        pub outlier_config: OutlierForestConfig [OutlierForestConfigTuning],
    }
    impl Debug for DistanceIsolationForestConfig {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            let full_type_name = type_name::<Self>();
            let struct_name = full_type_name.split("::").last().unwrap_or(full_type_name);
            let struct_name = struct_name.chars().take(struct_name.len() - 6).collect::<String>();
            write!(
                f,
                "{}_{}_{}",
                struct_name, self.outlier_config.n_trees, self.distance
            )
        }
    }
}
pub struct DistanceIsolationForest {
    trees: Vec<DistanceIsolationTree>,
    config: DistanceIsolationForestConfig,
    max_samples: usize,
}
impl Forest<DistanceIsolationTree> for DistanceIsolationForest {
    type Config = DistanceIsolationForestConfig;
    type TuningType = isize;
    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config,
            max_samples: 0,
        }
    }
    fn fit(&mut self, data: &mut [Sample]) {
        self.fit_(data)
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
    fn get_trees(&self) -> &Vec<DistanceIsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<DistanceIsolationTree> {
        &mut self.trees
    }
    fn tuning_predict(&self, _ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Self::TuningType> {
        self.predict(ds_test)
    }
}
impl OutlierForest<DistanceIsolationTree> for DistanceIsolationForest {
    fn get_forest_config(&self) -> (&OutlierForestConfig, &DistanceIsolationForestConfig) {
        (&self.config.outlier_config, &self.config)
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
}
