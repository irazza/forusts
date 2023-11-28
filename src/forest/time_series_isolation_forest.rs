use crate::feature_extraction::statistics::{mean, slope, std, EULER_MASCHERONI};
use crate::tree::{extra_tree::ExtraTree, tree::Tree};
use rand::{seq::SliceRandom, thread_rng, Rng};
use rayon::prelude::*;
use std::cmp::min;

pub struct TimeSeriesIsolationForest {
    trees: Vec<ExtraTree>,
    n_trees: usize,
    n_intervals: usize,
    min_interval_length: usize,
    intervals: Vec<Vec<(usize, usize)>>,
    max_depth: Option<usize>,
    enhanced_anomaly_score: Option<bool>,
    max_samples: usize,
}

impl TimeSeriesIsolationForest {
    pub fn new(
        n_trees: usize,
        n_intervals: usize,
        min_interval_length: usize,
        enhanced_anomaly_score: Option<bool>,
        max_depth: Option<usize>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            n_intervals,
            min_interval_length,
            intervals: Vec::new(),
            max_depth,
            enhanced_anomaly_score,
            max_samples: 256,
        }
    }

    pub fn transform(x: &Vec<Vec<f64>>, intervals: &Vec<(usize, usize)>) -> Vec<Vec<f64>> {
        let n_samples = x.len();
        let mut transformed_x: Vec<Vec<f64>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in intervals {
                let mean = mean(&x[j][*start..*end]);
                let std = std(&x[j][*start..*end]);
                let slope = slope(&x[j][*start..*end]);
                sample.extend([mean, std, slope].into_iter());
            }
            transformed_x.push(sample);
        }
        transformed_x
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>) {
        self.max_samples = min(256, x.len());
        // Generate n_intervals, with random start and end
        let n_features = x[0].len();
        for _i in 0..self.n_trees {
            let mut intervals = Vec::new();
            for _j in 0..self.n_intervals {
                let start = thread_rng().gen_range(0..n_features - self.min_interval_length);
                let end = thread_rng().gen_range(start + self.min_interval_length..n_features);
                intervals.push((start, end));
            }
            self.intervals.push(intervals);
        }

        self.trees
            .par_extend((0..self.n_trees).into_par_iter().map(|i| {
                let transformed_x = Self::transform(x, &self.intervals[i]);
                let mut n_samples: Vec<usize> = (0..x.len()).collect();
                n_samples.shuffle(&mut rand::thread_rng());

                let mut tree = ExtraTree::new(
                    self.max_depth
                        .unwrap_or(self.max_samples.ilog2() as usize + 1),
                    1,
                );
                tree.fit(
                    &(0..self.max_samples)
                        .into_iter()
                        .map(|i| &transformed_x[n_samples[i]])
                        .collect::<Vec<&Vec<f64>>>(),
                    &(0..self.max_samples).collect(),
                );

                tree
            }));
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        let scores = self.score_samples(x);
        let mut predictions = Vec::new();
        for i in 0..x.len() {
            predictions.push(if scores[i] > 0.5 { 1 } else { 0 });
        }
        predictions
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
