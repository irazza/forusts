use crate::feature_extraction::statistics::{mean, slope, std, EULER_MASCHERONI};
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
    enhanced_anomaly_score: Option<bool>,
    max_samples: usize,
    min_samples_split: usize,
    max_depth: Option<usize>,
}

impl TimeSeriesForest {
    pub fn new(
        n_trees: usize,
        criterion: Criterion,
        n_intervals: usize,
        min_interval_length: usize,
        max_features: MaxFeatures,
        max_depth: Option<usize>,
        min_samples_split: usize,
        enhanced_anomaly_score: Option<bool>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            criterion,
            n_intervals,
            min_interval_length,
            intervals: Vec::new(),
            max_features,
            min_samples_split,
            max_samples: 256,
            enhanced_anomaly_score,
            max_depth,
        }
    }

    pub fn score_samples(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        let mut scores = Vec::new();
        let path_length = self.path_length(x);
        let c_n = 2.0 * (f64::log2(self.max_samples as f64 - 1.0) + EULER_MASCHERONI)
            - 2.0 * (self.max_samples as f64 - 1.0) / self.max_samples as f64;
        if !self.enhanced_anomaly_score.unwrap_or(false) {
            let e_h = path_length
                .iter()
                .map(|x| x.iter().sum::<usize>() as f64 / self.n_trees as f64)
                .collect::<Vec<f64>>();
            for i in 0..x.len() {
                scores.push(2.0f64.powf(-e_h[i] / c_n));
            }
        } else {
            let enhanced_scores = path_length
                .iter()
                .map(|x| {
                    x.iter()
                        .map(|pl| 2.0f64.powf(-(*pl as f64) / c_n))
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();
            for i in 0..x.len() {
                scores.push(enhanced_scores[i].iter().sum::<f64>() / self.n_trees as f64);
            }
        }
        scores
    }

    fn path_length(&self, x: &Vec<Vec<f64>>) -> Vec<Vec<usize>> {
        let mut path_length = Vec::new();
        path_length.par_extend((0..x.len()).into_par_iter().map(|i| {
            let mut depth = Vec::new();
            for tree in &self.trees {
                depth.push(tree.predict_leaf(&x[i]).get_depth());
            }
            depth
        }));
        path_length
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
