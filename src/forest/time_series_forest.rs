use crate::feature_extraction::statistics::{mean, slope, std};
use crate::forest::forest::{ClassificationForest, Forest};
use crate::tree::{
    decision_tree::DecisionTree,
    tree::{Criterion, MaxFeatures, Tree},
};
use crate::utils::structures::Sample;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use super::forest::ClassificationForestConfig;

pub struct TimeSeriesForestConfig {
    pub n_intervals: usize,
    pub min_interval_length: usize,
    pub classification_config: ClassificationForestConfig,
}
pub struct TimeSeriesForest {
    trees: Vec<DecisionTree>,
    intervals: Vec<Vec<(usize, usize)>>,
    config: TimeSeriesForestConfig,
}

impl Forest<DecisionTree> for TimeSeriesForest {
    type Config = TimeSeriesForestConfig;
    fn new(config: Self::Config) -> Self {
        Self {
            trees: Vec::new(),
            intervals: Vec::new(),
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
                let start = thread_rng().gen_range(0..n_features - self.config.min_interval_length);
                let end =
                    thread_rng().gen_range(start + self.config.min_interval_length..n_features);
                intervals.push((start, end));
            }
            self.intervals.push(intervals);
        }
    }
    fn get_trees(&self) -> &Vec<DecisionTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<DecisionTree> {
        &mut self.trees
    }
    fn transform<'a>(&self, data: &[Sample<'a>], intervals_index: usize) -> Vec<Sample<'a>> {
        let n_samples = data.len();
        let mut transformed_data: Vec<Sample<'_>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in self.intervals[intervals_index].iter().copied() {
                let mean = mean(&data[j].data[start..end]);
                let std = std(&data[j].data[start..end]);
                let slope = slope(&data[j].data[start..end]);
                sample.extend([mean, std, slope].into_iter());
            }
            transformed_data.push(Sample {
                data: std::borrow::Cow::Owned(sample),
                target: data[j].target,
            });
        }
        transformed_data
    }
}

impl ClassificationForest for TimeSeriesForest {
    fn get_forest_config(&self) -> &ClassificationForestConfig {
        &self.config.classification_config
    }
}
