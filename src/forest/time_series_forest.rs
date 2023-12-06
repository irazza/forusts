use crate::feature_extraction::statistics::{mean, slope, std};
use crate::forest::forest::ClassificationForest;
use crate::tree::{
    decision_tree::DecisionTree,
    tree::{Criterion, MaxFeatures, Tree},
};
use rand::{thread_rng, Rng};
use rayon::prelude::*;

pub struct TimeSeriesForest {
    trees: Vec<DecisionTree>,
    criterion: Criterion,
    n_trees: usize,
    n_intervals: usize,
    min_interval_length: usize,
    intervals: Vec<Vec<(usize, usize)>>,
    max_features: MaxFeatures,
    max_samples: usize,
    min_samples_split: usize,
    max_depth: Option<usize>,
}

impl TimeSeriesForest {
    pub fn new(
        n_trees: usize,
        criterion: Criterion,
        n_intervals: usize,
        max_features: MaxFeatures,
        max_depth: Option<usize>,
        min_samples_split: usize,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            criterion,
            n_intervals,
            min_interval_length: 3,
            intervals: Vec::new(),
            max_features,
            min_samples_split,
            max_samples: 256,
            max_depth,
        }
    }
}

impl ClassificationForest for TimeSeriesForest {
    fn get_trees_mut(&mut self) -> &mut Vec<DecisionTree> {
        &mut self.trees
    }
    fn get_trees(&self) -> &Vec<DecisionTree> {
        &self.trees
    }
    fn get_n_trees(&self) -> usize {
        self.n_trees
    }
    fn get_criterion(&self) -> Criterion {
        self.criterion
    }
    fn get_max_features(&self) -> MaxFeatures {
        self.max_features
    }
    fn get_max_depth(&self) -> Option<usize> {
        self.max_depth
    }
    fn get_min_samples_split(&self) -> usize {
        self.min_samples_split
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn compute_intervals(&mut self, n_features: usize) {
        // Generate n_intervals, with random start and end
        for _i in 0..self.get_n_trees() {
            let mut intervals = Vec::new();
            for _j in 0..self.n_intervals {
                let start = thread_rng().gen_range(0..n_features - self.min_interval_length);
                let end = thread_rng().gen_range(start + self.min_interval_length..n_features);
                intervals.push((start, end));
            }
            self.intervals.push(intervals);
        }
    }
    fn transform(&self, x: &Vec<Vec<f64>>, intervals_index: usize) -> Vec<Vec<f64>> {
        let n_samples = x.len();
        let mut transformed_x: Vec<Vec<f64>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in self.intervals[intervals_index].iter().copied() {
                let mean = mean(&x[j][start..end]);
                let std = std(&x[j][start..end]);
                let slope = slope(&x[j][start..end]);
                sample.extend([mean, std, slope].into_iter());
            }
            transformed_x.push(sample);
        }
        transformed_x
    }
}
