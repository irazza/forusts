use std::{
    cmp::{max, min},
    sync::atomic::{AtomicUsize, Ordering},
};

use hashbrown::HashMap;
use parking_lot::Mutex;
use rand::{seq::SliceRandom, thread_rng, Rng};
use rayon::prelude::*;

use crate::{
    feature_extraction::statistics::{mean, EULER_MASCHERONI},
    tree::{
        decision_tree::DecisionTree,
        isolation_tree::IsolationTree,
        node::Node,
        tree::{Criterion, MaxFeatures, Tree},
    },
};

pub trait ClassificationForest: Sync + Send {
    fn get_trees_mut(&mut self) -> &mut Vec<DecisionTree>;
    fn get_trees(&self) -> &Vec<DecisionTree>;
    fn get_n_trees(&self) -> usize;
    fn get_criterion(&self) -> Criterion;
    fn get_max_features(&self) -> MaxFeatures;
    fn get_max_depth(&self) -> Option<usize>;
    fn get_min_samples_split(&self) -> usize;
    fn get_max_samples(&self) -> usize;
    fn set_max_samples(&mut self, max_samples: usize);
    fn transform(&self, x: &[Vec<f64>], intervals_index: usize) -> Vec<Vec<f64>>;
    fn compute_intervals(&mut self, n_features: usize);
    fn fit(&mut self, x: &[Vec<f64>], y: &Vec<usize>) {
        let n_samples = x.len();
        self.set_max_samples(n_samples);
        let n_trees = self.get_n_trees();
        self.compute_intervals(x[0].len());

        let mut trees = Vec::new();
        trees.par_extend((0..n_trees).into_par_iter().map(|i| {
            let transformed_x = self.transform(x, i);
            let bootstrap_indices: Vec<usize> = (0..n_samples)
                .map(|_| thread_rng().gen_range(0..n_samples))
                .collect();

            let mut tree = DecisionTree::new(
                self.get_criterion(),
                self.get_max_depth().unwrap_or(usize::MAX),
                self.get_min_samples_split(),
                self.get_max_features(),
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
        *self.get_trees_mut() = trees;
    }
    fn forest_depths(&self) -> f64 {
        self.get_trees()
            .iter()
            .map(|tree| tree.get_depth())
            .sum::<usize>() as f64
            / self.get_n_trees() as f64
    }
    fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        let n_samples = x.len();
        let mut predictions = Vec::new();
        // Make predictions for each sample using each tree in the forest
        predictions.par_extend(self.get_trees().par_iter().enumerate().map(|(i, tree)| {
            let transformed_x = self.transform(x, i);
            tree.predict(&transformed_x)
        }));

        // Combine predictions using a majority vote
        let mut final_predictions = vec![0; n_samples];

        for i in 0..n_samples {
            let mut class_counts = HashMap::new();
            for j in 0..self.get_n_trees() {
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

        self.get_trees()
            .par_iter()
            .enumerate()
            .for_each(|(i, tree)| {
                let transformed_x1 = self.transform(&x1, i);
                let transformed_x2 = self.transform(&x2, i);
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
                    .map(|d| d.into_inner() as f64 / self.get_n_trees() as f64)
                    .collect()
            })
            .collect()
    }
    fn pairwise_ancestor(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        self.get_trees()
            .par_iter()
            .enumerate()
            .for_each(|(i, tree)| {
                let transformed_x1 = self.transform(&x1, i);
                let transformed_x2 = self.transform(&x2, i);
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
                    .map(|d| d.into_inner() as f64 / self.get_n_trees() as f64)
                    .collect()
            })
            .collect()
    }

    fn pairwise_zhu(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        self.get_trees()
            .par_iter()
            .enumerate()
            .for_each(|(i, tree)| {
                let transformed_x1 = self.transform(&x1, i);
                let transformed_x2 = self.transform(&x2, i);
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
                        *distance_matrix[i][j].lock() +=
                            distances[&(x2_node as *const Node)].get_depth() as f64
                                / max(x1_node.get_depth(), x2_node.get_depth()) as f64;
                    }
                }
            });

        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| 1.0 - (d.into_inner() as f64 / self.get_n_trees() as f64))
                    .collect()
            })
            .collect()
    }

    fn hyperparams_tuning(&mut self, params: HashMap<String, Vec<f64>>) -> HashMap<String, f64> {
        let n_params = params.len();
        //TODO
        todo!();
    }
}

pub trait OutlierForest: Sync + Send {
    fn set_max_samples(&mut self, max_samples: usize);
    fn get_max_samples(&self) -> usize;
    fn get_trees_mut(&mut self) -> &mut Vec<IsolationTree>;
    fn get_trees(&self) -> &Vec<IsolationTree>;
    fn get_n_trees(&self) -> usize;
    fn get_max_depth(&self) -> Option<usize>;
    fn get_enhanced_anomaly_score(&self) -> bool;
    fn transform(&self, x: &[Vec<f64>], intervals_index: usize) -> Vec<Vec<f64>>;
    fn compute_intervals(&mut self, n_features: usize);
    fn fit(&mut self, x: &[Vec<f64>]) {
        self.set_max_samples(min(256, x.len()));
        self.compute_intervals(x[0].len());
        let mut trees = Vec::new();
        trees.par_extend((0..self.get_n_trees()).into_par_iter().map(|i| {
            let transformed_x = self.transform(x, i);
            let mut n_samples: Vec<usize> = (0..x.len()).collect();
            n_samples.shuffle(&mut rand::thread_rng());

            let mut tree = IsolationTree::new(
                self.get_max_depth()
                    .unwrap_or(self.get_max_samples().ilog2() as usize + 1),
                2, // Setted to 2 to avoid empty child when splitting when there are only two samples
            );
            tree.fit(
                &(0..self.get_max_samples())
                    .into_iter()
                    .map(|i| &transformed_x[n_samples[i]])
                    .collect::<Vec<&Vec<f64>>>(),
                &(0..self.get_max_samples()).collect(),
            );

            tree
        }));
        *self.get_trees_mut() = trees;
    }

    fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        let scores = self.score_samples(x);
        let mut predictions = Vec::new();
        for i in 0..x.len() {
            predictions.push(if scores[i] > 0.5 { 1 } else { 0 });
        }
        predictions
    }

    fn score_samples(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let mut scores = Vec::new();
        scores.par_extend(x.par_windows(1).map(|sample| {
            let mut average_depth = 0.0;
            let mut average_path_length = 0.0;
            let mut enhnaced_score = 0.0;

            for (i, tree) in self.get_trees().iter().enumerate() {
                let transformed_x = self.transform(sample, i).into_iter().next().unwrap();
                let leaf = tree.predict_leaf(&transformed_x);
                let path_length = Self::path_length(&leaf);
                let depth = leaf.get_depth() as f64;

                if self.get_enhanced_anomaly_score() {
                    enhnaced_score += 2.0f64.powf(-depth / path_length);
                } else {
                    average_depth += depth;
                    average_path_length += path_length;
                }
            }

            if self.get_enhanced_anomaly_score() {
                return enhnaced_score / self.get_n_trees() as f64;
            } else {
                return 2.0f64.powf(-average_depth / average_path_length);
            }
        }));
        // scores.par_extend(x.par_iter().map(|sample| {
        //     let mut transformed_sample = Vec::new();
        //     let mut depths = Vec::new();
        //     for (i, tree) in self.get_trees().iter().enumerate() {
        //         transformed_sample = self.transform(&vec![sample.clone()], i)[0].clone();
        //         let leaf = tree.predict_leaf(&transformed_sample);
        //         depths.push(leaf.get_depth() as f64);
        //     }
        // // ERROR PATH LENGHT ALWAYS USE THE LAST TRANSFORMED SAMPLE
        //     let path_length = self.path_length(&transformed_sample);
        //     if self.get_enhanced_anomaly_score() {
        //         let mut enhanced_scores = Vec::new();
        //         for (pl, d) in path_length.iter().zip(depths.iter()) {
        //             enhanced_scores.push(2.0f64.powf(-d/pl));
        //         }
        //         return mean(&enhanced_scores[..]);
        //     } else {
        //         let average_path_length = mean(&path_length[..]);
        //         let average_depth = mean(&depths[..]);
        //         return 2.0f64.powf(-average_depth / average_path_length);
        //     }
        // }));
        scores
    }

    fn path_length(leaf: &Node) -> f64 {
        let samples = leaf.get_samples() as f64;
        if samples > 1.0 {
            return leaf.get_depth() as f64
                + (2.0 * (f64::ln(samples - 1.0) + EULER_MASCHERONI)
                    - 2.0 * (samples - 1.0) / samples);
        } else {
            return leaf.get_depth() as f64;
        }
    }
    fn hyperparams_tuning(&mut self, params: HashMap<String, Vec<f64>>) -> HashMap<String, f64> {
        let n_params = params.len();
        //TODO
        todo!();
    }
}
