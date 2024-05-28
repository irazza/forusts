#![allow(dead_code)]
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
use std::cmp::min;
use std::fmt::Debug;
use std::{
    any::type_name,
    cmp::max,
    fmt::Formatter,
    sync::atomic::{AtomicUsize, Ordering},
};

pub const ANOMALY_SCORE: f64 = 2.0;

pub trait Forest<T: Tree>: Sync + Send {
    type Config;
    type TuningType;
    fn get_trees(&self) -> &Vec<T>;
    fn get_trees_mut(&mut self) -> &mut Vec<T>;
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
    impl Debug for ClassificationForestConfig {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            let full_type_name = type_name::<Self>();
            let struct_name = full_type_name.split("::").last().unwrap_or(full_type_name);
            let struct_name = struct_name.chars().take(struct_name.len() - 6).collect::<String>();
            write!(
                f,
                "{}_{}_{:?}_{:?}",
                struct_name, self.n_trees, self.max_features, self.criterion
            )
        }
    }

}

pub trait ClassificationTree: Tree {
    type TreeConfig: Sync + Send;
    fn from_classification_config(config: &Self::TreeConfig) -> Self;
}

pub trait ClassificationForest<T: ClassificationTree>: Forest<T> {
    fn get_forest_config(&self) -> (&ClassificationForestConfig, &T::TreeConfig);
    fn fit_(&mut self, data: &mut [Sample]) {
        let mut trees = Vec::new();
        let (config, tree_config) = self.get_forest_config();
        trees.par_extend((0..config.n_trees).into_par_iter().map(|_i| {
            let mut tree = T::from_classification_config(&tree_config);
            if config.bootstrap {
                let bootstrap_indices = (0..data.len())
                    .collect::<Vec<_>>()
                    .iter()
                    .map(|_| thread_rng().gen_range(0..data.len()))
                    .collect::<Vec<_>>();
                tree.fit(
                    &mut bootstrap_indices
                        .iter()
                        .map(|idx| data[*idx].to_ref())
                        .collect::<Vec<Sample>>(),
                );
            } else {
                tree.fit(&mut data.iter().map(|x| x.to_ref()).collect::<Vec<Sample>>());
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
        predictions.par_extend(trees.par_iter().map(|tree| tree.predict(data)));

        // Combine predictions using a majority vote
        let mut final_predictions = vec![0; n_samples];

        for i in 0..n_samples {
            let mut class_counts = HashMap::new();
            for j in 0..self.get_forest_config().0.n_trees {
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
    fn pairwise_breiman(&self, ds_test: &[Sample], ds_train: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..ds_test.len())
            .map(|_| (0..ds_train.len()).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().for_each(|tree| {
            let ds_test_leaves = ds_test
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();
            let ds_train_leaves = ds_train
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();

            for (i, &ds_test_node) in ds_test_leaves.iter().enumerate() {
                for (j, &ds_train_node) in ds_train_leaves.iter().enumerate() {
                    distance_matrix[i][j].fetch_add(
                        ((ds_test_node as *const Node<_>) != (ds_train_node as *const Node<_>))
                            as usize,
                        Ordering::Relaxed,
                    );
                }
            }
        });
        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().0.n_trees as f64)
                    .collect()
            })
            .collect()
    }
    fn pairwise_ancestor(&self, ds_test: &[Sample], ds_train: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..ds_test.len())
            .map(|_| (0..ds_train.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().for_each(|tree| {
            let ds_test_nodes = ds_test
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();
            let ds_train_nodes = ds_train
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();

            for (i, &ds_test_node) in ds_test_nodes.iter().enumerate() {
                let distances = tree.compute_ancestor(ds_test_node);

                for (j, &ds_train_node) in ds_train_nodes.iter().enumerate() {
                    *distance_matrix[i][j].lock() += (ds_test_node.get_depth()
                        + ds_train_node.get_depth()
                        - 2 * distances[&(ds_train_node as *const Node<_>)].get_depth())
                        as f64
                        / max(ds_test_node.get_depth(), ds_train_node.get_depth()) as f64;
                }
            }
        });
        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().0.n_trees as f64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<_>>>()
    }
    fn pairwise_zhu(&self, ds_test: &[Sample], ds_train: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..ds_test.len())
            .map(|_| (0..ds_train.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().for_each(|tree| {
            let ds_test_nodes = ds_test
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();
            let ds_train_nodes = ds_train
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();

            for (i, &ds_test_node) in ds_test_nodes.iter().enumerate() {
                let distances = tree.compute_ancestor(ds_test_node);

                for (j, &ds_train_node) in ds_train_nodes.iter().enumerate() {
                    *distance_matrix[i][j].lock() += distances[&(ds_train_node as *const Node<_>)]
                        .get_depth() as f64
                        / max(ds_test_node.get_depth(), ds_train_node.get_depth()) as f64;
                }
            }
        });

        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| {
                        1.0 - (d.into_inner() as f64 / self.get_forest_config().0.n_trees as f64)
                    })
                    .collect()
            })
            .collect()
    }

    fn pairwise_ratiorf(&self, ds_test: &[Sample], ds_train: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..ds_test.len())
            .map(|_| (0..ds_train.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();

        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().for_each(|tree| {
            let mut union = Vec::new();
            for (i, sample_test) in ds_test.iter().enumerate() {
                let ds_test_splits = tree.get_splits(sample_test);
                for (j, sample_train) in ds_train.iter().enumerate() {
                    union.clear();
                    union.extend(ds_test_splits.iter().copied());
                    union.extend(tree.get_splits(sample_train).into_iter());
                    union.sort_unstable();
                    union.dedup();
                    let agree = union
                        .iter()
                        .filter(|s| s.split(sample_test, false) == s.split(sample_train, false))
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
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().0.n_trees as f64)
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
        pub min_samples_split: usize,
        pub max_samples: f64,
    }
    impl Debug for OutlierForestConfig {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let full_type_name = type_name::<Self>();
            let struct_name = full_type_name.split("::").last().unwrap_or(full_type_name);
            let struct_name = struct_name.chars().take(struct_name.len() - 6).collect::<String>();
            write!(
                f,
                "{}_{}",
                struct_name, self.n_trees,
            )
        }
    }
}

pub trait OutlierTree: Tree {
    type TreeConfig: Sync + Send;
    fn from_outlier_config(config: &Self::TreeConfig, max_samples: usize) -> Self;
}
pub trait OutlierForest<T: OutlierTree>: Forest<T> {
    fn get_forest_config(&self) -> (&OutlierForestConfig, &T::TreeConfig);
    fn set_max_samples(&mut self, max_samples: usize);
    fn get_max_samples(&self) -> usize;
    fn fit_(&mut self, data: &[Sample]) {
        let mut trees = Vec::new();
        let config_max_samples = self.get_forest_config().0.max_samples;
        let max_samples = min(256, (data.len() as f64 * config_max_samples) as usize);
        self.set_max_samples(max_samples);
        let (config, tree_config) = self.get_forest_config();
        trees.par_extend((0..config.n_trees).into_par_iter().map(|_i| {
            let mut n_samples: Vec<usize> = (0..data.len()).collect();
            n_samples.shuffle(&mut rand::thread_rng());
            let mut tree = T::from_outlier_config(&tree_config, max_samples);
            tree.fit(
                &mut (0..max_samples)
                    .into_iter()
                    .map(|i| data[n_samples[i]].to_ref())
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
        self.compute_anomaly_scores(data)
    }
    fn compute_anomaly_scores(&self, data: &[Sample]) -> Vec<f64> {
        let mut scores = Vec::new();
        let max_samples = self.get_max_samples() as f64;
        let average_path_length_max_samples = 2.0 * (f64::ln(max_samples - 1.0) + EULER_MASCHERONI)
            - 2.0 * (max_samples - 1.0) / max_samples;
        scores.par_extend(data.par_windows(1).map(|sample| {
            let mut average_depth = 0.0;
            let sample = &sample[0];
            let trees: &Vec<T> = self.get_trees();
            for tree in trees.iter() {
                average_depth += Self::path_length(tree, sample);
            }
            let score = ANOMALY_SCORE
                .powf(-average_depth / (average_path_length_max_samples * trees.len() as f64));
            return score;
        }));
        scores
    }
    fn path_length(tree: &T, x: &Sample) -> f64 {
        T::SplitParameters::path_length(tree, x)
    }
}
