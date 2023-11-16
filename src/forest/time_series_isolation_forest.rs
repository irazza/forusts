#![allow(dead_code)]
use crate::feature_extraction::statistics::{self, percentile, EULER_MASCHERONI};
use crate::forest::forest::Forest;
use crate::tree::decision_tree::{Criterion, DecisionTree, MaxFeatures, Splitter};
use crate::tree::node::Node;
use hashbrown::HashMap;
use parking_lot::Mutex;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::cmp::max;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TimeSeriesIsolationForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    n_intervals: usize,
    min_interval_length: usize,
    intervals: Vec<Vec<(usize, usize)>>,
    max_features: MaxFeatures,
    max_depth: Option<usize>,
    threshold: f64,
}

impl TimeSeriesIsolationForest {
    pub fn new(
        n_trees: usize,
        n_intervals: usize,
        min_interval_length: usize,
        max_features: MaxFeatures,
        max_depth: Option<usize>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            n_intervals,
            min_interval_length,
            intervals: Vec::new(),
            max_features,
            max_depth,
            threshold: 0.5,
        }
    }

    pub fn transform(x: &Vec<Vec<f64>>, intervals: &Vec<(usize, usize)>) -> Vec<Vec<f64>> {
        let n_samples = x.len();
        let mut transformed_x: Vec<Vec<f64>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in intervals {
                let mean = statistics::mean(&x[j][*start..*end]);
                let std = statistics::std(&x[j][*start..*end]);
                let slope = statistics::slope(&x[j][*start..*end]);
                sample.extend([mean, std, slope].into_iter());
            }
            transformed_x.push(sample);
        }
        transformed_x
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>) {
        let n_samples = x.len();

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
                let mut tree = DecisionTree::new(
                    Criterion::None,
                    Splitter::Random,
                    self.max_depth.unwrap_or(usize::MAX),
                    1,
                    1,
                    self.max_features,
                );
                tree.fit(
                    &(0..n_samples)
                        .into_iter()
                        .map(|i| &transformed_x[i])
                        .collect(),
                    &(0..n_samples).collect(),
                );
                
                tree
            }));

        let scores = self.score_samples(x);
        self.threshold = percentile(&scores, 95);
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        let scores = self.score_samples(x);
        let mut predictions = Vec::new();
        for i in 0..x.len() {
            predictions.push(if scores[i] > self.threshold { 1 } else { 0 });
        }
        predictions
    }

    pub fn score_samples(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        let n_samples = x.len();
        let mut scores = Vec::new();
        let mut leaves: Vec<Vec<usize>> = Vec::new();
        let c_n = 2.0 * (f64::ln((n_samples - 1) as f64) + EULER_MASCHERONI)
            - (2.0 * (n_samples - 1) as f64 / n_samples as f64);

        //Make predictions for each sample using each tree in the forest
        leaves.par_extend(self.trees.par_iter().enumerate().map(|(i, tree)| {
            Self::transform(x, &self.intervals[i])
                .iter()
                .map(|sample| tree.predict_leaf(sample).get_depth())
                .collect()
        }));

        for i in 0..n_samples {
            let e_h = leaves.iter().map(|leaf| leaf[i]).sum::<usize>() as f64 / n_samples as f64;
            scores.push(2.0f64.powf(-e_h / c_n));
        }
        scores
    }

    pub fn pairwise_breiman(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| AtomicUsize::new(0)).collect())
            .collect();

        self.trees.par_iter().enumerate().for_each(|(i, tree)| {
            let transformed_x1 = Self::transform(&x1, &self.intervals[i]);
            let transformed_x2 = Self::transform(&x2, &self.intervals[i]);
            let x1_nodes = transformed_x1
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();
            let x2_nodes = transformed_x2
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();

            for (i, &x1_node) in x1_nodes.iter().enumerate() {
                for (j, &x2_node) in x2_nodes.iter().enumerate() {
                    distance_matrix[i][j].fetch_add(
                        ((x1_node as *const Node) != (x2_node as *const Node)) as usize,
                        Ordering::Relaxed,
                    );
                }
            }
        });

        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| d.into_inner() as f64 / self.n_trees as f64)
                    .collect()
            })
            .collect()
    }

    pub fn pairwise_ancestor(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        self.trees.par_iter().enumerate().for_each(|(i, tree)| {
            let transformed_x1 = Self::transform(&x1, &self.intervals[i]);
            let transformed_x2 = Self::transform(&x2, &self.intervals[i]);
            let x1_nodes = transformed_x1
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();
            let x2_nodes = transformed_x2
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();

            for (i, &x1_node) in x1_nodes.iter().enumerate() {
                let distances = tree.compute_ancestor(x1_node);

                for (j, &x2_node) in x2_nodes.iter().enumerate() {
                    *distance_matrix[i][j].lock() += (x1_node.get_depth() + x2_node.get_depth()
                        - 2 * distances[&(x2_node as *const Node)].get_depth())
                        as f64
                        / max(x1_node.get_depth(), x2_node.get_depth()) as f64;
                }
            }
        });

        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| d.into_inner() as f64 / self.n_trees as f64)
                    .collect()
            })
            .collect()
    }

    pub fn pairwise_zhu(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        self.trees.par_iter().enumerate().for_each(|(i, tree)| {
            let transformed_x1 = Self::transform(&x1, &self.intervals[i]);
            let transformed_x2 = Self::transform(&x2, &self.intervals[i]);
            let x1_nodes = transformed_x1
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();
            let x2_nodes = transformed_x2
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();

            for (i, &x1_node) in x1_nodes.iter().enumerate() {
                let distances = tree.compute_ancestor(x1_node);

                for (j, &x2_node) in x2_nodes.iter().enumerate() {
                    *distance_matrix[i][j].lock() += distances[&(x2_node as *const Node)]
                        .get_depth() as f64
                        / max(x1_node.get_depth(), x2_node.get_depth()) as f64;
                }
            }
        });

        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| 1.0 - (d.into_inner() as f64 / self.n_trees as f64))
                    .collect()
            })
            .collect()
    }
}
