#![allow(dead_code)]
use crate::tree::decision_tree::{Criterion, DecisionTree, MaxFeatures, Splitter};
use crate::feature_extraction::ts_features;
use crate::tree::node::Node;
use hashbrown::HashMap;
use parking_lot::Mutex;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::cmp::max;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::forest::forest::Forest;

pub struct TimeSeriesForest {
    trees: Vec<DecisionTree>,
    criterion: Criterion,
    splitter: Splitter,
    n_trees: usize,
    n_intervals: usize,
    min_interval_length: usize,
    intervals: Vec<Vec<(usize, usize)>>,
    max_features: MaxFeatures,
    max_depth: Option<usize>,
}

impl TimeSeriesForest {
    pub fn new(
        n_trees: usize,
        criterion: Criterion,
        splitter: Splitter,
        n_intervals: usize,
        min_interval_length: usize,
        max_features: MaxFeatures,
        max_depth: Option<usize>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            criterion,
            splitter,
            n_intervals,
            min_interval_length,
            intervals: Vec::new(),
            max_features,
            max_depth,
        }
    }

    pub fn transform(x: &Vec<Vec<f64>>, intervals: &Vec<(usize, usize)>) -> Vec<Vec<f64>> {
        let n_samples = x.len();
        let mut transformed_x: Vec<Vec<f64>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in intervals {
                let mean = ts_features::mean(&x[j][*start..*end].to_vec());
                let std = ts_features::std(&x[j][*start..*end].to_vec());
                let slope = ts_features::slope(&x[j][*start..*end].to_vec());
                sample.extend([mean, std, slope].into_iter());
            }
            transformed_x.push(sample);
        }
        transformed_x
    }
}

impl Forest for TimeSeriesForest {
    fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>) {
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
                let bootstrap_indices: Vec<usize> = (0..n_samples)
                    .map(|_| thread_rng().gen_range(0..n_samples))
                    .collect();
                let transformed_x = Self::transform(x, &self.intervals[i]);
                let mut tree = DecisionTree::new(
                    self.criterion,
                    self.splitter,
                    self.max_depth.unwrap_or(usize::MAX),
                    2,
                    1,
                    self.max_features,
                );
                tree.fit(
                    &bootstrap_indices
                        .iter()
                        .map(|i| &transformed_x[*i])
                        .collect(),
                    &bootstrap_indices.iter().map(|i| y[*i]).collect(),
                );
                tree
            }));
    }

    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        let n_samples = x.len();
        let mut predictions = Vec::new();
        // Make predictions for each sample using each tree in the forest
        predictions.par_extend(self.trees.par_iter().enumerate().map(|(i, tree)| {
            let transformed_x = Self::transform(x, &self.intervals[i]);
            tree.predict(&transformed_x)
        }));

        // Combine predictions using a majority vote
        let mut final_predictions = vec![0; n_samples];

        for i in 0..n_samples {
            let mut class_counts = HashMap::new();
            for j in 0..self.n_trees {
                let class = predictions[j][i];
                *class_counts.entry(class).or_insert(0) += 1;
            }

            // Find the class with the maximum count
            let mut max_count = 0;
            let mut majority_class = 0;
            for (class, count) in &class_counts {
                if *count > max_count {
                    max_count = *count;
                    majority_class = *class;
                }
            }

            final_predictions[i] = majority_class;
        }

        final_predictions
    }

    fn pairwise_breiman(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

    fn pairwise_ancestor(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

    fn pairwise_zhu(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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
