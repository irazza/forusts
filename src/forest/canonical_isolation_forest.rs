use std::borrow::Cow;

use crate::feature_extraction::catch22::CATCH22;
use crate::feature_extraction::statistics::zscore;
use crate::grid_search_tuning;
use crate::tree::tree::Tree;
use crate::utils::structures::Sample;
use crate::utils::tuning::TuningConfig;
use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::isolation_tree::IsolationTree,
};
use rand::{seq::SliceRandom, thread_rng, Rng};

use super::forest::{OutlierForestConfig, OutlierForestConfigTuning};

pub const MIN_INTERVAL_PERC: usize = 10;

grid_search_tuning! {
    pub struct CanonicalIsolationForestConfig[CanonicalIsolationForestConfigTuning] {
        pub n_intervals: usize,
        pub outlier_config: OutlierForestConfig [OutlierForestConfigTuning],
    }
}
impl TuningConfig for CanonicalIsolationForestConfigTuning {
    type Tree = IsolationTree;
    type Forest = CanonicalIsolationForest;
}

#[derive(Clone)]
pub struct CanonicalIsolationForest {
    trees: Vec<IsolationTree>,
    min_interval_perc: usize,
    intervals: Vec<Vec<(usize, usize)>>,
    attributes: Vec<usize>,
    config: CanonicalIsolationForestConfig,
}

impl Forest<IsolationTree> for CanonicalIsolationForest {
    type Config = CanonicalIsolationForestConfig;
    type TuningType = f64;

    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            min_interval_perc: MIN_INTERVAL_PERC,
            intervals: Vec::new(),
            attributes: Vec::new(),
            config,
        }
    }
    fn fit(&mut self, data: &mut [Sample<'_>]) {
        self.fit_(&data);
    }
    fn predict(&self, data: &[Sample<'_>]) -> Vec<isize> {
        self.predict_(data)
    }
    fn compute_intervals(&mut self, n_features: usize) {
        // Generate n_intervals, with random start and end
        let min_interval_length =
            (n_features as f64 * self.min_interval_perc as f64 / 100.0).round() as usize;
        for _i in 0..self.config.outlier_config.n_trees {
            let mut intervals = Vec::new();
            for _j in 0..self.config.n_intervals {
                let start = thread_rng().gen_range(0..n_features - min_interval_length);
                let end = thread_rng().gen_range(start + min_interval_length..n_features);
                intervals.push((start, end));
            }
            self.intervals.push(intervals);
        }
        self.attributes = (0..22).collect();
        self.attributes.shuffle(&mut thread_rng());
    }
    fn get_trees(&self) -> &Vec<IsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<IsolationTree> {
        &mut self.trees
    }
    fn transform<'a>(&self, data: &[Sample<'a>], intervals_index: usize) -> Vec<Sample<'a>> {
        let n_samples = data.len();
        let mut transformed_data: Vec<Sample<'_>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in self.intervals[intervals_index].iter().copied() {
                for i in 0..8 {
                    sample.extend(
                        [CATCH22::get(self.attributes[i])(&data[j].data[start..end])].iter(),
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
    fn tuning_predict(&self, data: &[Sample<'_>]) -> Vec<Self::TuningType> {
        self.score_samples(data)
    }
}

impl OutlierForest for CanonicalIsolationForest {
    fn get_forest_config(&self) -> &OutlierForestConfig {
        &self.config.outlier_config
    }
}
