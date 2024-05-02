use core::fmt::Debug;
use std::any::type_name;
use std::fmt::Formatter;

use crate::distance::distances::Distance;
use crate::grid_search_tuning;
use crate::tree::distance_set_tree::DistanceSetTree;
use crate::tree::distance_tree::DistanceTree;
use crate::utils::structures::Sample;

use crate::forest::forest::{ClassificationForest, Forest};

use super::forest::{ClassificationForestConfig, ClassificationForestConfigTuning};

grid_search_tuning! {
    pub struct DistanceSetForestConfig[DistanceSetForestTuning]{
        pub distance: Distance,
        pub classification_config: ClassificationForestConfig [ClassificationForestConfigTuning],
    }
    impl Debug for DistanceSetForestConfig {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            let full_type_name = type_name::<Self>();
            let struct_name = full_type_name.split("::").last().unwrap_or(full_type_name);
            let struct_name = struct_name.chars().take(struct_name.len() - 6).collect::<String>();
            write!(
                f,
                "{}_{}_{:?}_{:?}_{}",
                struct_name, self.classification_config.n_trees, self.classification_config.max_features, self.classification_config.criterion, self.distance
            )
        }
    }
}
pub struct DistanceSetForest {
    trees: Vec<DistanceSetTree>,
    config: DistanceSetForestConfig,
}
impl Forest<DistanceSetTree> for DistanceSetForest {
    type Config = DistanceSetForestConfig;
    type TuningType = isize;
    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            config,
        }
    }
    fn fit(&mut self, data: &mut [Sample]) {
        //let mut data = data.iter().map(|x| Sample{ data: Cow::Owned(zscore(&x.data)), target: x.target}).collect::<Vec<_>>();
        self.fit_(data)
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize> {
        self.predict_(data)
    }
    fn compute_intervals(&mut self, _n_features: usize) {}
    fn get_trees(&self) -> &Vec<DistanceSetTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<DistanceSetTree> {
        &mut self.trees
    }
    fn transform<'a>(&self, data: &[Sample], _intervals_index: usize) -> Vec<Sample> {
        data.to_vec()
    }
    fn tuning_predict(&self, _ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Self::TuningType> {
        self.predict(ds_test)
    }
}
impl ClassificationForest<DistanceSetTree> for DistanceSetForest {
    fn get_forest_config(&self) -> (&ClassificationForestConfig, &DistanceSetForestConfig) {
        (&self.config.classification_config, &self.config)
    }
}
