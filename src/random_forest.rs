use std::sync::atomic::{AtomicUsize, Ordering};

use crate::decision_tree::{DecisionTree, Node};
use hashbrown::HashMap;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

pub enum MaxFeatures {
    All,
    Sqrt,
    Log2,
}

pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    max_features: MaxFeatures,
    max_depth: Option<usize>,
}

impl RandomForest {
    pub fn new(n_trees: usize, max_features: MaxFeatures, max_depth: Option<usize>) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_features,
            max_depth,
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>) {
        let n_samples = x.len();
        let n_features = x[0].len();

        self.trees
            .par_extend((0..self.n_trees).into_par_iter().map(|_i| {
                let mut bootstrap_indices = vec![0; n_samples];

                for i in 0..n_samples {
                    bootstrap_indices[i] = thread_rng().gen_range(0..n_samples);
                }

                let mut x_bootstrap = vec![vec![0.0; n_features]; n_samples];
                let mut y_bootstrap = vec![0; n_samples];

                for i in 0..n_samples {
                    x_bootstrap[i] = x[bootstrap_indices[i]].clone();
                    y_bootstrap[i] = y[bootstrap_indices[i]];
                }

                let mut tree = DecisionTree::new(
                    self.max_depth.unwrap_or(usize::MAX),
                    2,
                    match self.max_features {
                        MaxFeatures::All => x[0].len(),
                        MaxFeatures::Sqrt => (x[0].len() as f64).sqrt() as usize,
                        MaxFeatures::Log2 => x[0].len().ilog2() as usize,
                    },
                );
                tree.fit(&x_bootstrap, &y_bootstrap);
                tree
            }));
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        let n_samples = x.len();
        let mut predictions = Vec::new();

        // Make predictions for each sample using each tree in the forest
        predictions.par_extend(
            self.trees
                .par_iter()
                .enumerate()
                .map(|(_i, tree)| tree.predict(x)),
        );

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

    pub fn pairwise_breiman(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        self.trees.par_iter().for_each(|tree| {
            let x1_nodes = x1.iter().map(|x| tree.predict_leaf(x)).collect::<Vec<_>>();
            let x2_nodes = x2.iter().map(|x| tree.predict_leaf(x)).collect::<Vec<_>>();

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
            .map(|_| (0..x2.len()).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        self.trees.par_iter().for_each(|tree| {
            let x1_nodes = x1.iter().map(|x| tree.predict_leaf(x)).collect::<Vec<_>>();
            let x2_nodes = x2.iter().map(|x| tree.predict_leaf(x)).collect::<Vec<_>>();

            for (i, &x1_node) in x1_nodes.iter().enumerate() {
                let distances = tree.compute_distances(x1_node);

                for (j, &x2_node) in x2_nodes.iter().enumerate() {
                    distance_matrix[i][j].fetch_add(
                        distances[&(x2_node as *const Node)],
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
}
