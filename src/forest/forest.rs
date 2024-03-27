use crate::{
    feature_extraction::statistics::EULER_MASCHERONI,
    grid_search_tuning,
    tree::{
        node::Node,
        tree::{Criterion, MaxFeatures, SplitParameters, Tree},
    },
    utils::structures::Sample,
};
use hashbrown::HashMap;
use parking_lot::Mutex;
use rand::{seq::SliceRandom, thread_rng, Rng};
use rayon::prelude::*;
use std::{
    cmp::max,
    sync::atomic::{AtomicUsize, Ordering},
};

pub const ANOMALY_SCORE: f64 = 2.0;

pub trait Forest<T: Tree>: Sync + Send {
    type Config;
    type TuningType;
    fn compute_intervals(&mut self, n_features: usize);
    fn get_trees(&self) -> &Vec<T>;
    fn get_trees_mut(&mut self) -> &mut Vec<T>;
    fn transform<'a>(&self, data: &[Sample], intervals_index: usize) -> Vec<Sample>;
    fn new(config: Self::Config) -> Self;
    fn fit(&mut self, data: &mut [Sample]);
    fn predict(&self, data: &[Sample]) -> Vec<isize>;
    fn tuning_predict(&self, ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Self::TuningType>;
}

grid_search_tuning! {
    pub struct ClassificationForestConfig[ClassificationForestConfigTuning] {
        pub n_trees: usize,
        pub min_samples_split: usize,
        pub max_features: MaxFeatures,
        pub max_depth: Option<usize>,
        pub criterion: Criterion,
        pub bootstrap: bool,
    }
}

pub trait ClassificationTree: Tree {
    fn from_classification_config(config: &ClassificationForestConfig) -> Self;
}

pub trait ClassificationForest<T: ClassificationTree>: Forest<T> {
    fn get_forest_config(&self) -> &ClassificationForestConfig;
    fn fit_(&mut self, data: &mut [Sample]) {
        self.compute_intervals(data[0].data.len());
        let mut trees = Vec::new();
        let config = self.get_forest_config();
        trees.par_extend((0..config.n_trees).into_par_iter().map(|i| {
            let transformed_data = self.transform(data, i);
            let mut tree = T::from_classification_config(&config);
            if config.bootstrap {
                let bootstrap_indices = (0..transformed_data.len())
                    .collect::<Vec<_>>()
                    .iter()
                    .map(|_| thread_rng().gen_range(0..transformed_data.len()))
                    .collect::<Vec<_>>();
                tree.fit(
                    &mut bootstrap_indices
                        .iter()
                        .map(|idx| transformed_data[*idx].to_ref())
                        .collect::<Vec<Sample>>(),
                );
            } else {
                tree.fit(
                    &mut transformed_data
                        .iter()
                        .map(|x| x.to_ref())
                        .collect::<Vec<Sample>>(),
                );
            }
            tree
        }));
        *self.get_trees_mut() = trees;
    }
    fn predict_(&self, data: &[Sample]) -> Vec<isize> {
        let n_samples = data.len();
        let mut predictions = Vec::new();
        // Make predictions for each sample using each tree in the forest
        let trees: &Vec<T> = self.get_trees();
        predictions.par_extend(trees.par_iter().enumerate().map(|(i, tree)| {
            let transformed_data = self.transform(data, i);
            tree.predict(&transformed_data)
        }));

        // Combine predictions using a majority vote
        let mut final_predictions = vec![0; n_samples];

        for i in 0..n_samples {
            let mut class_counts = HashMap::new();
            for j in 0..self.get_forest_config().n_trees {
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
    fn pairwise_breiman(&self, x1: &[Sample], x2: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().enumerate().for_each(|(i, tree)| {
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
                        ((x1_node as *const Node<_>) != (x2_node as *const Node<_>)) as usize,
                        Ordering::Relaxed,
                    );
                }
            }
        });
        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    .collect()
            })
            .collect()
    }
    fn pairwise_ancestor(&self, x1: &[Sample], x2: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().enumerate().for_each(|(i, tree)| {
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
                        - 2 * distances[&(x2_node as *const Node<_>)].get_depth())
                        as f64
                        / max(x1_node.get_depth(), x2_node.get_depth()) as f64;
                }
            }
        });
        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<_>>>()
    }
    fn pairwise_zhu(&self, x1: &[Sample], x2: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().enumerate().for_each(|(i, tree)| {
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
                    *distance_matrix[i][j].lock() += distances[&(x2_node as *const Node<_>)]
                        .get_depth() as f64
                        / max(x1_node.get_depth(), x2_node.get_depth()) as f64;
                }
            }
        });

        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| {
                        1.0 - (d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    })
                    .collect()
            })
            .collect()
    }

    fn pairwise_ratiorf(&self, x1: &[Sample], x2: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();

        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().enumerate().for_each(|(i, tree)| {
            let transformed_x1 = self.transform(&x1, i);
            let transformed_x2 = self.transform(&x2, i);

            for (i, x1) in transformed_x1.iter().enumerate() {
                for (j, x2) in transformed_x2.iter().enumerate() {
                    let mut union = Vec::new();
                    union.extend(tree.get_splits(x1).into_iter());
                    union.extend(tree.get_splits(x2).into_iter());
                    // Remove duplicates based on feature and threshold
                    union.sort_by(|s1, s2| s1.cmp(s2));
                    union.dedup_by(|a, b| a == b);
                    let agree = union
                        .iter()
                        .filter(|s| s.split(x1, false) == s.split(x2, false))
                        .count() as f64;
                    *distance_matrix[i][j].lock() += 1.0
                        - if union.len() == 0 {
                            1.0
                        } else {
                            agree / union.len() as f64
                        };
                }
            }
        });
        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    .collect()
            })
            .collect()
    }
}

grid_search_tuning! {
    pub struct OutlierForestConfig[OutlierForestConfigTuning] {
        pub n_trees: usize,
        pub enhanced_anomaly_score: bool,
        pub max_depth: Option<usize>,
    }
}

pub trait OutlierTree: Tree {
    fn from_outlier_config(max_samples: usize, config: &OutlierForestConfig) -> Self;
}

pub trait OutlierForest<T: OutlierTree>: Forest<T> {
    fn get_forest_config(&self) -> &OutlierForestConfig;
    fn set_max_samples(&mut self, max_samples: usize);
    fn get_max_samples(&self) -> usize;
    fn fit_(&mut self, data: &[Sample]) {
        let subsampling_ratio = f64::min(1.0, 256.0 / data.len() as f64);
        let max_samples = (data.len() as f64 * subsampling_ratio) as usize;
        self.set_max_samples(max_samples);
        self.compute_intervals(data[0].data.len());
        let mut trees = Vec::new();
        let config = self.get_forest_config();
        trees.par_extend((0..config.n_trees).into_par_iter().map(|i| {
            let mut n_samples: Vec<usize> = (0..data.len()).collect();
            n_samples.shuffle(&mut rand::thread_rng());
            let transformed_data = self.transform(data, i);
            let mut tree = T::from_outlier_config(max_samples, config);
            tree.fit(
                &mut (0..max_samples)
                    .into_iter()
                    .map(|i| transformed_data[n_samples[i]].to_ref())
                    .collect::<Vec<Sample>>(),
            );
            tree
        }));
        *self.get_trees_mut() = trees;
    }
    fn predict_(&self, data: &[Sample]) -> Vec<isize> {
        let scores = self.score_samples(data);
        let mut predictions = Vec::new();
        for i in 0..data.len() {
            predictions.push(if scores[i] > 0.5 { 1 } else { 0 });
        }
        predictions
    }
    fn score_samples(&self, data: &[Sample]) -> Vec<f64> {
        if self.get_forest_config().enhanced_anomaly_score {
            self.compute_enhanced_anomaly_scores(data)
        } else {
            self.compute_anomaly_scores(data)
        }
    }
    fn compute_enhanced_anomaly_scores(&self, data: &[Sample]) -> Vec<f64> {
        let mut scores = Vec::new();
        let max_samples = self.get_max_samples() as f64;
        let denominator = (2.0 * (f64::ln(max_samples - 1.0) + EULER_MASCHERONI))
            - 2.0 * ((max_samples - 1.0) / max_samples);
        scores.par_extend(data.par_windows(1).map(|sample| {
            let mut average_as = 0.0;
            let trees: &Vec<T> = self.get_trees();
            for (i, tree) in trees.iter().enumerate() {
                let transformed_x = self.transform(sample, i).into_iter().next().unwrap();
                average_as +=
                    ANOMALY_SCORE.powf(-Self::path_length(tree, &transformed_x) / denominator);
            }
            average_as /= self.get_forest_config().n_trees as f64;
            return average_as;
        }));
        scores
    }
    fn compute_anomaly_scores(&self, data: &[Sample]) -> Vec<f64> {
        let mut scores = Vec::new();
        let max_samples = self.get_max_samples() as f64;
        let denominator = (2.0 * (f64::ln(max_samples - 1.0) + EULER_MASCHERONI))
            - 2.0 * ((max_samples - 1.0) / max_samples);
        scores.par_extend(data.par_windows(1).map(|sample| {
            let mut average_depth = 0.0;
            let trees: &Vec<T> = self.get_trees();
            for (i, tree) in trees.iter().enumerate() {
                let transformed_x = self.transform(sample, i).into_iter().next().unwrap();
                average_depth += Self::path_length(tree, &transformed_x);
            }
            average_depth /= self.get_forest_config().n_trees as f64;
            return ANOMALY_SCORE.powf(-average_depth / denominator);
        }));
        scores
    }
    fn path_length(tree: &T, transformed_x: &Sample) -> f64 {
        T::SplitParameters::path_length(tree, transformed_x)
    }
}
