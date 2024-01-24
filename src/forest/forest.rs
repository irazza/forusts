use crate::{
    feature_extraction::statistics::EULER_MASCHERONI,
    grid_search_tuning,
    tree::{
        decision_tree::{DecisionTree, DecisionTreeConfig},
        isolation_tree::{IsolationTree, IsolationTreeConfig},
        node::Node,
        tree::{Criterion, MaxFeatures, Tree}, extra_tree::{ExtraTree, ExtraTreeConfig},
    },
    utils::structures::Sample,
};
use hashbrown::HashMap;
use parking_lot::Mutex;
use rand::{seq::SliceRandom, thread_rng, Rng};
use rayon::prelude::*;
use std::{
    cmp::{max, min},
    sync::atomic::{AtomicUsize, Ordering},
};

pub const ANOMALY_SCORE: f64 = 2.0;

pub trait Forest<T: Tree>: Sync + Send {
    type Config;
    type TuningType;
    fn compute_intervals(&mut self, n_features: usize);
    fn get_trees(&self) -> &Vec<T>;
    fn get_trees_mut(&mut self) -> &mut Vec<T>;
    fn transform<'a>(&self, data: &[Sample<'a>], intervals_index: usize) -> Vec<Sample<'a>>;
    fn new(config: Self::Config) -> Self;
    fn fit(&mut self, data: &mut [Sample<'_>]);
    fn predict(&self, data: &[Sample<'_>]) -> Vec<isize>;
    fn tuning_predict(&self, ds_train: &[Sample<'_>], ds_test: &[Sample<'_>]) -> Vec<Self::TuningType>;
}


grid_search_tuning! {
    pub struct DistanceForestConfig[DistanceForestConfigTuning] {
        pub n_trees: usize,
        pub max_depth: Option<usize>,
        pub min_samples_split: usize,
        pub max_features: MaxFeatures,
    }
}

pub trait DistanceForest: Forest<ExtraTree> {
    fn get_forest_config(&self) -> &DistanceForestConfig;
    fn fit_(&mut self, data: &mut [Sample<'_>]) {
        self.compute_intervals(data[0].data.len());
        let mut trees = Vec::new();
        let config = self.get_forest_config();
        trees.par_extend((0..config.n_trees).into_par_iter().map(|i| {
            let transformed_data = self.transform(data, i);
            let mut tree = ExtraTree::new(ExtraTreeConfig {
                max_depth: config.max_depth.unwrap_or(usize::MAX),
                min_samples_split: config.min_samples_split,
                max_features: config.max_features,
            });
            let bootstrap_indices = (0..transformed_data.len())
                .collect::<Vec<_>>()
                .iter()
                .map(|_| thread_rng().gen_range(0..transformed_data.len()))
                .collect::<Vec<_>>();
            tree.fit(
                &mut bootstrap_indices
                    .iter()
                    .map(|idx| transformed_data[*idx].to_ref())
                    .collect::<Vec<Sample<'_>>>(),
            );
            tree
        }));
        *self.get_trees_mut() = trees;
    }
    fn predict_(&self, data: &[Sample<'_>]) -> Vec<isize> {
        let n_samples = data.len();
        let mut predictions = Vec::new();
        // Make predictions for each sample using each tree in the forest
        let trees: &Vec<ExtraTree> = self.get_trees();
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
    fn pairwise_breiman(&self, x1: &[Sample<'_>], x2: &[Sample<'_>]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        let trees: &Vec<ExtraTree> = self.get_trees();
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
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    .collect()
            })
            .collect()
    }
    fn pairwise_ancestor(&self, x1: &[Sample<'_>], x2: &[Sample<'_>]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        let trees: &Vec<ExtraTree> = self.get_trees();
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
                        - 2 * distances[&(x2_node as *const Node)].get_depth())
                        as f64 / ExtraTree::get_diameter(tree.get_root()).0 as f64;
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
    fn pairwise_zhu(&self, x1: &[Sample<'_>], x2: &[Sample<'_>]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        let trees: &Vec<ExtraTree> = self.get_trees();
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
                    *distance_matrix[i][j].lock() += 1.0 - (distances[&(x2_node as *const Node)]
                        .get_depth() as f64
                        / max(x1_node.get_depth(), x2_node.get_depth()) as f64);
                }
            }
        });

        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| {
                        d.into_inner() as f64 / self.get_forest_config().n_trees as f64
                    })
                    .collect()
            })
            .collect()
    }
    fn pairwise_ratiorf(&self, x1: &[Sample<'_>], x2: &[Sample<'_>]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();

        let trees: &Vec<ExtraTree> = self.get_trees();
        trees.par_iter().enumerate().for_each(|(i, tree)| {
            let transformed_x1 = self.transform(&x1, i);
            let transformed_x2 = self.transform(&x2, i);

            for (i, x1) in transformed_x1.iter().enumerate() {
                for (j, x2) in transformed_x2.iter().enumerate() {
                    let mut union = Vec::new();
                    union.extend(tree.get_splits(x1).into_iter());
                    union.extend(tree.get_splits(x2).into_iter());
                    // Remove duplicates based on feature and threshold
                    union.sort_by(|(f1, t1), (f2, t2)| {
                       f1
                            .partial_cmp(f2)
                            .unwrap_or(std::cmp::Ordering::Equal) // If the feature is the same, compare the threshold
                            .then(t1.partial_cmp(t2).unwrap())
                    });
                    union.dedup_by(|a, b| a == b);
                    let agree = union.iter().filter(|(f, t)| (x1.data[*f] > *t) == (x2.data[*f] > *t)).count() as f64;
                    *distance_matrix[i][j].lock() += 1.0 - agree/union.len() as f64;
                }
            }
        });
        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| {
                        d.into_inner() as f64 / self.get_forest_config().n_trees as f64
                    })
                    .collect()
            })
            .collect()
    }
}


grid_search_tuning! {
    pub struct ClassificationForestConfig[ClassificationForestConfigTuning] {
        pub n_trees: usize,
        pub max_depth: Option<usize>,
        pub min_samples_split: usize,
        pub max_features: MaxFeatures,
        pub criterion: Criterion,
    }
}

pub trait ClassificationForest: Forest<DecisionTree> {
    fn get_forest_config(&self) -> &ClassificationForestConfig;
    fn fit_(&mut self, data: &mut [Sample<'_>]) {
        self.compute_intervals(data[0].data.len());
        let mut trees = Vec::new();
        let config = self.get_forest_config();
        trees.par_extend((0..config.n_trees).into_par_iter().map(|i| {
            let transformed_data = self.transform(data, i);
            let mut tree = DecisionTree::new(DecisionTreeConfig {
                criterion: config.criterion,
                max_depth: config.max_depth.unwrap_or(usize::MAX),
                min_samples_split: config.min_samples_split,
                max_features: config.max_features,
            });
            let bootstrap_indices = (0..transformed_data.len())
                .collect::<Vec<_>>()
                .iter()
                .map(|_| thread_rng().gen_range(0..transformed_data.len()))
                .collect::<Vec<_>>();
            tree.fit(
                &mut bootstrap_indices
                    .iter()
                    .map(|idx| transformed_data[*idx].to_ref())
                    .collect::<Vec<Sample<'_>>>(),
            );
            tree
        }));
        *self.get_trees_mut() = trees;
    }
    fn predict_(&self, data: &[Sample<'_>]) -> Vec<isize> {
        let n_samples = data.len();
        let mut predictions = Vec::new();
        // Make predictions for each sample using each tree in the forest
        let trees: &Vec<DecisionTree> = self.get_trees();
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
    fn pairwise_breiman(&self, x1: &[Sample<'_>], x2: &[Sample<'_>]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        let trees: &Vec<DecisionTree> = self.get_trees();
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
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    .collect()
            })
            .collect()
    }
    fn pairwise_ancestor(&self, x1: &[Sample<'_>], x2: &[Sample<'_>]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        let trees: &Vec<DecisionTree> = self.get_trees();
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
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<_>>>()
    }
    fn pairwise_zhu(&self, x1: &[Sample<'_>], x2: &[Sample<'_>]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        let trees: &Vec<DecisionTree> = self.get_trees();
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
                    .map(|d| {
                        1.0 - (d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    })
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

pub trait OutlierForest: Forest<IsolationTree> {
    fn get_forest_config(&self) -> &OutlierForestConfig;
    fn fit_(&mut self, data: &[Sample<'_>]) {
        let max_samples = min(256, data.len());
        self.compute_intervals(data[0].data.len());
        let mut trees = Vec::new();
        let config = self.get_forest_config();
        trees.par_extend((0..config.n_trees).into_par_iter().map(|i| {
            let mut n_samples: Vec<usize> = (0..data.len()).collect();
            n_samples.shuffle(&mut rand::thread_rng());
            let transformed_data = self.transform(data, i);
            let mut tree = IsolationTree::new(IsolationTreeConfig {
                max_depth: config.max_depth.unwrap_or(max_samples.ilog2() as usize + 1),
                min_samples_split: 2,
                // Setted to 2 to avoid empty child when splitting when there are only two samples
            });
            tree.fit(
                &mut (0..max_samples)
                    .into_iter()
                    .map(|i| transformed_data[n_samples[i]].to_ref())
                    .collect::<Vec<Sample<'_>>>(),
            );
            tree
        }));
        *self.get_trees_mut() = trees;
    }
    fn predict_(&self, data: &[Sample<'_>]) -> Vec<isize> {
        let scores = self.score_samples(data);
        let mut predictions = Vec::new();
        for i in 0..data.len() {
            predictions.push(if scores[i] > 0.5 { 1 } else { 0 });
        }
        predictions
    }
    fn score_samples(&self, data: &[Sample<'_>]) -> Vec<f64> {
        let mut scores = Vec::new();
        let max_samples = min(256, data.len()) as f64;
        let denominator = (2.0 * (f64::ln(max_samples - 1.0) + EULER_MASCHERONI))
            - 2.0 * ((max_samples - 1.0) / max_samples);
        scores.par_extend(data.par_windows(1).map(|sample| {
            let mut average_depth = 0.0;
            let trees: &Vec<IsolationTree> = self.get_trees();
            for (i, tree) in trees.iter().enumerate() {
                let transformed_x = self.transform(sample, i).into_iter().next().unwrap();
                let leaf = tree.predict_leaf(&transformed_x);
                average_depth += Self::path_length(leaf);
            }
            average_depth /= self.get_forest_config().n_trees as f64;
            return ANOMALY_SCORE.powf(-average_depth / denominator);
        }));
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
    fn pairwise_breiman(&self, x1: &[Sample<'_>], x2: &[Sample<'_>]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        let trees: &Vec<IsolationTree> = self.get_trees();
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
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    .collect()
            })
            .collect()
    }
    fn pairwise_ancestor(&self, x1: &[Sample<'_>], x2: &[Sample<'_>]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        let trees: &Vec<IsolationTree> = self.get_trees();
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
                    .map(|d| d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<_>>>()
    }
    fn pairwise_zhu(&self, x1: &[Sample<'_>], x2: &[Sample<'_>]) -> Vec<Vec<f64>> {
        let distance_matrix: Vec<Vec<_>> = (0..x1.len())
            .map(|_| (0..x2.len()).map(|_| Mutex::new(0.0)).collect())
            .collect();
        let trees: &Vec<IsolationTree> = self.get_trees();
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
                    .map(|d| {
                        1.0 - (d.into_inner() as f64 / self.get_forest_config().n_trees as f64)
                    })
                    .collect()
            })
            .collect()
    }
}
