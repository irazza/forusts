#![allow(dead_code)]
use crate::{
    tree::{
        node::Node,
        tree::{SplitParameters, Tree},
    },
    utils::structures::Sample,
    RandomGenerator,
};
use atomic_float::AtomicF64;
use hashbrown::HashMap;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::{cmp::max, sync::atomic::AtomicUsize};

pub const SUBSAMPLE_SIZE: usize = 256;
const ANOMALY_SCORE: f64 = 2.0;
pub const EGAMMA: f64 = 0.577215664901532860606512090082402431_f64;

pub trait Forest<T: Tree>: Sync + Send {
    type Config;
    fn get_trees(&self) -> &Vec<T>;
    fn get_trees_mut(&mut self) -> &mut Vec<T>;
    fn new(config: &Self::Config) -> Self;
    fn fit(&mut self, data: &mut [Sample], random_state: Option<RandomGenerator>);
    fn predict(&self, data: &[Sample]) -> Vec<isize>;
    fn generate_indeces(
        n_samples: usize,
        n_population: usize,
        with_replacement: bool,
        random_state: &mut RandomGenerator,
    ) -> Vec<usize> {
        let mut indeces = Vec::with_capacity(n_samples);
        if with_replacement {
            for _ in 0..n_samples {
                indeces.push(random_state.gen_range(0..n_population));
            }
        } else {
            let mut population = (0..n_population).collect::<Vec<usize>>();
            population.shuffle(random_state);
            indeces.extend(population.iter().take(n_samples).copied());
        }
        indeces
    }
    fn pairwise_breiman(&self, ds_test: &[Sample], ds_train: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..ds_test.len())
            .map(|_| (0..ds_train.len()).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().for_each(|tree| {
            let ds_train = tree.transform(ds_train);
            let ds_test = tree.transform(ds_test);
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
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            }
        });
        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| d.into_inner() as f64 / trees.len() as f64)
                    .collect()
            })
            .collect()
    }
    fn pairwise_zhu(&self, ds_test: &[Sample], ds_train: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..ds_test.len())
            .map(|_| (0..ds_train.len()).map(|_| AtomicF64::new(0.0)).collect())
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
                    let value = distances[&(ds_train_node as *const Node<_>)].get_depth() as f64
                        / max(ds_test_node.get_depth(), ds_train_node.get_depth()) as f64;
                    distance_matrix[i][j].fetch_add(value, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });

        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| 1.0 - (d.into_inner() as f64 / trees.len() as f64))
                    .collect()
            })
            .collect()
    }

    fn pairwise_ratiorf(&self, ds_test: &[Sample], ds_train: &[Sample]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..ds_test.len())
            .map(|_| (0..ds_train.len()).map(|_| AtomicF64::new(0.0)).collect())
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
                        .filter(|s| s.split(sample_test) == s.split(sample_train))
                        .count() as f64;
                    let value = 1.0
                        - if union.len() == 0 {
                            1.0
                        } else {
                            agree / union.len() as f64
                        };
                    distance_matrix[i][j].fetch_add(value, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });
        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| d.into_inner() as f64 / trees.len() as f64)
                    .collect()
            })
            .collect()
    }
}

// pub struct ClassificationForestConfig {
//     pub n_trees: usize,
//     pub min_samples_split: usize,
//     pub max_depth: Option<usize>,
//     pub max_features: fn(usize) -> usize,
//     pub criterion: fn(&HashMap<isize, usize>, Vec<&HashMap<isize, usize>>) -> f64,
//     pub bootstrap: bool,
// }

// pub trait ClassificationTree: Tree {
//     type TreeConfig: Sync + Send;
//     fn from_classification_config(config: &Self::TreeConfig) -> Self;
// }

// pub trait ClassificationForest<T: ClassificationTree>: Forest<T> {
//     fn get_forest_config(&self) -> (&ClassificationForestConfig, &T::TreeConfig);
//     fn fit_(&mut self, data: &mut [Sample]) {
//         let mut trees = Vec::new();
//         let (config, tree_config) = self.get_forest_config();
//         trees.par_extend((0..config.n_trees).into_par_iter().map(|_i| {
//             let mut tree = T::from_classification_config(&tree_config);
//             if config.bootstrap {
//                 let bootstrap_indices = (0..data.len())
//                     .collect::<Vec<_>>()
//                     .iter()
//                     .map(|_| thread_rng().gen_range(0..data.len()))
//                     .collect::<Vec<_>>();
//                 tree.fit(
//                     &mut bootstrap_indices
//                         .iter()
//                         .map(|idx| data[*idx].to_ref())
//                         .collect::<Vec<Sample>>(),
//                 );
//             } else {
//                 tree.fit(&mut data.iter().map(|x| x.to_ref()).collect::<Vec<Sample>>());
//             }
//             tree
//         }));
//         *self.get_trees_mut() = trees;
//     }
//     fn predict_(&self, data: &[Sample]) -> Vec<isize> {
//         let n_samples = data.len();
//         let mut predictions = Vec::new();
//         // Make predictions for each sample using each tree in the forest
//         let trees: &Vec<T> = self.get_trees();
//         predictions.par_extend(trees.par_iter().map(|tree| tree.predict(data)));

//         // Combine predictions using a majority vote
//         let mut final_predictions = vec![0; n_samples];

//         for i in 0..n_samples {
//             let mut class_counts = HashMap::new();
//             for j in 0..self.get_forest_config().0.n_trees {
//                 let class = predictions[j][i];
//                 *class_counts.entry(class).or_insert(0) += 1;
//             }

//             // Find the class with the maximum count
//             let mut max_count = 0;
//             let mut majority_class = 0;
//             for (class, count) in &class_counts {
//                 if *count > max_count {
//                     max_count = *count;
//                     majority_class = *class;
//                 }
//             }

//             final_predictions[i] = majority_class;
//         }

//         final_predictions
//     }
//     fn pairwise_ancestor(&self, ds_test: &[Sample], ds_train: &[Sample]) -> Vec<Vec<f64>> {
//         let distance_matrix: Vec<Vec<_>> = (0..ds_test.len())
//             .map(|_| (0..ds_train.len()).map(|_| Mutex::new(0.0)).collect())
//             .collect();
//         let trees: &Vec<T> = self.get_trees();
//         trees.par_iter().for_each(|tree| {
//             let ds_test_nodes = ds_test
//                 .iter()
//                 .map(|x| tree.predict_leaf(x))
//                 .collect::<Vec<_>>();
//             let ds_train_nodes = ds_train
//                 .iter()
//                 .map(|x| tree.predict_leaf(x))
//                 .collect::<Vec<_>>();

//             for (i, &ds_test_node) in ds_test_nodes.iter().enumerate() {
//                 let distances = tree.compute_ancestor(ds_test_node);

//                 for (j, &ds_train_node) in ds_train_nodes.iter().enumerate() {
//                     *distance_matrix[i][j].lock() += (ds_test_node.get_depth()
//                         + ds_train_node.get_depth()
//                         - 2 * distances[&(ds_train_node as *const Node<_>)].get_depth())
//                         as f64
//                         / max(ds_test_node.get_depth(), ds_train_node.get_depth()) as f64;
//                 }
//             }
//         });
//         distance_matrix
//             .into_iter()
//             .map(|d| {
//                 d.into_iter()
//                     .map(|d| d.into_inner() as f64 / self.get_forest_config().0.n_trees as f64)
//                     .collect::<Vec<_>>()
//             })
//             .collect::<Vec<Vec<_>>>()
//     }
//     fn pairwise_zhu(&self, ds_test: &[Sample], ds_train: &[Sample]) -> Vec<Vec<f64>> {
//         let distance_matrix: Vec<Vec<_>> = (0..ds_test.len())
//             .map(|_| (0..ds_train.len()).map(|_| Mutex::new(0.0)).collect())
//             .collect();
//         let trees: &Vec<T> = self.get_trees();
//         trees.par_iter().for_each(|tree| {
//             let ds_test_nodes = ds_test
//                 .iter()
//                 .map(|x| tree.predict_leaf(x))
//                 .collect::<Vec<_>>();
//             let ds_train_nodes = ds_train
//                 .iter()
//                 .map(|x| tree.predict_leaf(x))
//                 .collect::<Vec<_>>();

//             for (i, &ds_test_node) in ds_test_nodes.iter().enumerate() {
//                 let distances = tree.compute_ancestor(ds_test_node);

//                 for (j, &ds_train_node) in ds_train_nodes.iter().enumerate() {
//                     *distance_matrix[i][j].lock() += distances[&(ds_train_node as *const Node<_>)]
//                         .get_depth() as f64
//                         / max(ds_test_node.get_depth(), ds_train_node.get_depth()) as f64;
//                 }
//             }
//         });

//         distance_matrix
//             .into_iter()
//             .map(|d| {
//                 d.into_iter()
//                     .map(|d| {
//                         1.0 - (d.into_inner() as f64 / self.get_forest_config().0.n_trees as f64)
//                     })
//                     .collect()
//             })
//             .collect()
//     }

//     fn pairwise_ratiorf(&self, ds_test: &[Sample], ds_train: &[Sample]) -> Vec<Vec<f64>> {
//         let distance_matrix: Vec<Vec<_>> = (0..ds_test.len())
//             .map(|_| (0..ds_train.len()).map(|_| Mutex::new(0.0)).collect())
//             .collect();

//         let trees: &Vec<T> = self.get_trees();
//         trees.par_iter().for_each(|tree| {
//             let mut union = Vec::new();
//             for (i, sample_test) in ds_test.iter().enumerate() {
//                 let ds_test_splits = tree.get_splits(sample_test);
//                 for (j, sample_train) in ds_train.iter().enumerate() {
//                     union.clear();
//                     union.extend(ds_test_splits.iter().copied());
//                     union.extend(tree.get_splits(sample_train).into_iter());
//                     union.sort_unstable();
//                     union.dedup();
//                     let agree = union
//                         .iter()
//                         .filter(|s| s.split(sample_test, false) == s.split(sample_train, false))
//                         .count() as f64;
//                     *distance_matrix[i][j].lock() += 1.0
//                         - if union.len() == 0 {
//                             1.0
//                         } else {
//                             agree / union.len() as f64
//                         };
//                 }
//             }
//         });
//         distance_matrix
//             .into_iter()
//             .map(|d| {
//                 d.into_iter()
//                     .map(|d| d.into_inner() as f64 / self.get_forest_config().0.n_trees as f64)
//                     .collect()
//             })
//             .collect()
//     }
// }

#[derive(Clone)]
pub struct OutlierForestConfig {
    pub n_trees: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_samples: f64,
    pub max_features: fn(usize) -> usize,
    pub criterion: fn(&HashMap<isize, usize>, Vec<&HashMap<isize, usize>>) -> f64,
}

pub trait OutlierTree: Tree {
    type TreeConfig: Sync + Send;
    fn from_outlier_config(
        config: &Self::TreeConfig,
        max_samples: usize,
        n_features: usize,
        random_state: &mut RandomGenerator,
    ) -> Self;
}
pub trait OutlierForest<T: OutlierTree>: Forest<T> {
    fn get_forest_config(&self) -> (&OutlierForestConfig, &T::TreeConfig);
    fn set_max_samples(&mut self, max_samples: usize);
    fn get_max_samples(&self) -> usize;
    fn fit_(
        &mut self,
        data: &[Sample],
        max_samples: usize,
        with_replacement: bool,
        mut random_state: &mut RandomGenerator,
    ) {
        let mut trees = Vec::new();
        self.set_max_samples(max_samples);
        let (config, tree_config) = self.get_forest_config();
        let random_generators = (0..config.n_trees).map(|_| {
            RandomGenerator::from_rng(&mut random_state).expect("Error creating random generator")
        });
        trees.par_extend(
            (0..config.n_trees)
                .into_iter()
                .zip(random_generators)
                .par_bridge()
                .map(|(_i, mut random_state)| {
                    let indeces = Self::generate_indeces(
                        max_samples,
                        data.len(),
                        with_replacement,
                        &mut random_state,
                    );
                    let samples = indeces
                        .iter()
                        .map(|idx| data[*idx].clone())
                        .collect::<Vec<Sample>>();
                    let mut tree =
                        T::from_outlier_config(&tree_config, max_samples, samples[0].features.len(), &mut random_state);
                    let samples = tree.transform(&samples);
                    tree.fit(&samples, &mut random_state);
                    tree
                }),
        );
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
        let average_path_length_max_samples = T::average_path_length(self.get_max_samples());
        let trees: &Vec<T> = self.get_trees();
        let scores = (0..data.len())
            .map(|_| AtomicF64::new(0.0))
            .collect::<Vec<_>>();
        trees.par_iter().for_each(|tree| {
            let samples = tree.transform(data);
            for (i, sample) in samples.iter().enumerate() {
                scores[i].fetch_add(
                    Self::path_length(tree, sample),
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        });
        let scores = scores
            .into_iter()
            .map(|x| {
                ANOMALY_SCORE
                    .powf(-x.into_inner() / (average_path_length_max_samples * data.len() as f64))
            })
            .collect::<Vec<_>>();
        scores
    }
    fn path_length(tree: &T, x: &Sample) -> f64 {
        T::SplitParameters::path_length(tree, x)
    }
}
