use crate::utils::structures::MaxFeatures;
use crate::{
    tree::{
        node::Node,
        tree::{SplitParameters, Tree},
    },
    utils::{
        aggregation::{Aggregation, Combiner},
        statistics::transpose,
        structures::Sample,
    },
    RandomGenerator,
};
use atomic_float::AtomicF64;

use hashbrown::HashMap;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use serde::de;
use std::{cmp::max, sync::atomic::AtomicUsize};

pub const SUBSAMPLE_SIZE: usize = 256;
pub const EGAMMA: f64 = 0.577215664901532860606512090082402431_f64;

#[derive(Clone)]
pub struct ForestConfig {
    pub n_trees: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: MaxFeatures,
    pub criterion: fn(&HashMap<isize, usize>, &[HashMap<isize, usize>]) -> f64,
    pub aggregation: Option<Combiner>,
}

pub trait Forest<T: Tree>: Sync + Send {
    type Config;
    fn get_forest_config(&self) -> (&ForestConfig, &T::ForestTreeConfig);
    fn get_max_samples(&self) -> usize;
    fn get_trees(&self) -> &Vec<T>;
    fn get_trees_mut(&mut self) -> &mut Vec<T>;
    fn set_max_samples(&mut self, max_samples: usize);
    fn new(config: &Self::Config) -> Self;
    fn fit(&mut self, data: &mut [Sample], random_state: Option<RandomGenerator>);
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
                    let indices = generate_indices(
                        max_samples,
                        data.len(),
                        with_replacement,
                        &mut random_state,
                    );
                    let samples = indices
                        .iter()
                        .map(|idx| data[*idx].clone())
                        .collect::<Vec<Sample>>();
                    let mut tree = T::from_config(
                        &tree_config,
                        max_samples,
                        samples[0].features.len(),
                        &mut random_state,
                    );
                    let samples = tree.transform(&samples);
                    tree.fit(&samples, &mut random_state);
                    tree
                }),
        );
        *self.get_trees_mut() = trees;
    }
    fn predict(&self, data: &[Sample]) -> Vec<isize>;
    fn pairwise_breiman(&self, ds_a: &[Sample], ds_b: Option<&[Sample]>) -> Vec<Vec<f64>> {
        // let ds_b = ds_b.unwrap_or(ds_a);
        let ds_b_len = ds_b.map_or(ds_a.len(), |x| x.len());
        let distance_matrix: Vec<Vec<_>> = (0..ds_a.len())
            .map(|_| (0..ds_b_len).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().for_each(|tree| {
            let ds_a = tree.transform(ds_a);

            let ds_a_leaves = ds_a
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();

            let ds_b_leaves = if let Some(ds_b) = ds_b {
                let ds_b = tree.transform(ds_b);

                ds_b.iter()
                    .map(|x| tree.predict_leaf(x))
                    .collect::<Vec<_>>()
            } else {
                ds_a_leaves.clone()
            };

            for (i, &ds_a_node) in ds_a_leaves.iter().enumerate() {
                for (j, &ds_b_node) in ds_b_leaves.iter().enumerate() {
                    distance_matrix[i][j].fetch_add(
                        ((ds_a_node as *const Node<_>) != (ds_b_node as *const Node<_>)) as usize,
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

    // https://openaccess.thecvf.com/content_cvpr_2014/papers/Zhu_Constructing_Robust_Affinity_2014_CVPR_paper.pdf
    // VARIANT II
    fn pairwise_zhu(&self, ds_a: &[Sample], ds_b: Option<&[Sample]>) -> Vec<Vec<f64>> {
        let ds_b_len = ds_b.map_or(ds_a.len(), |x| x.len());
        let distance_matrix: Vec<Vec<_>> = (0..ds_a.len())
            .map(|_| (0..ds_b_len).map(|_| AtomicF64::new(0.0)).collect())
            .collect();
        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().for_each(|tree| {
            let ds_a = tree.transform(ds_a);

            let ds_a_leaves = ds_a
                .iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>();

            let ds_b_leaves = if let Some(ds_b) = ds_b {
                let ds_b = tree.transform(ds_b);

                ds_b.iter()
                    .map(|x| tree.predict_leaf(x))
                    .collect::<Vec<_>>()
            } else {
                ds_a_leaves.clone()
            };
            for (i, &ds_a_node) in ds_a_leaves.iter().enumerate() {
                let distances = tree.compute_ancestor(ds_a_node);

                for (j, &ds_b_node) in ds_b_leaves.iter().enumerate() {
                    let value = distances[&(ds_b_node as *const Node<_>)].get_depth() as f64
                        / max(ds_a_node.get_depth(), ds_b_node.get_depth()) as f64;
                    distance_matrix[i][j].fetch_add(value, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });

        distance_matrix
            .into_iter()
            .map(|d| {
                d.into_iter()
                    .map(|d| (1.0 - (d.into_inner() as f64 / trees.len() as f64)).sqrt())
                    .collect()
            })
            .collect()
    }

    // http://profs.sci.univr.it/~bicego/papers/2021_TKDE.pdf
    fn pairwise_ratiorf(&self, ds_a: &[Sample], ds_b: Option<&[Sample]>) -> Vec<Vec<f64>> {
        let ds_b_len = ds_b.map_or(ds_a.len(), |x| x.len());
        let distance_matrix: Vec<Vec<_>> = (0..ds_a.len())
            .map(|_| (0..ds_b_len).map(|_| AtomicF64::new(0.0)).collect())
            .collect();

        let trees: &Vec<T> = self.get_trees();
        trees.par_iter().for_each(|tree| {
            let ds_a = tree.transform(ds_a);
            let ds_b = ds_b
                .map(|ds_b| tree.transform(ds_b))
                .unwrap_or_else(|| ds_a.clone());

            let mut union = Vec::new();
            for (i, sample_test) in ds_a.iter().enumerate() {
                let ds_a_splits = tree.get_splits(sample_test);
                for (j, sample_train) in ds_b.iter().enumerate() {
                    union.clear();
                    union.extend(ds_a_splits.iter().copied());
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
                    distance_matrix[i][j]
                        .fetch_add(value.sqrt(), std::sync::atomic::Ordering::Relaxed);
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

pub trait ClassificationForest<T: Tree>: Forest<T> {
    fn predict_(&self, data: &[Sample]) -> Vec<isize> {
        let n_samples = data.len();
        let mut predictions = Vec::new();
        // Make predictions for each sample using each tree in the forest
        let trees = self.get_trees();
        predictions.par_extend(
            trees
                .par_iter()
                .map(|tree| tree.predict(&tree.transform(data))),
        );

        // Combine predictions using a majority vote
        let mut final_predictions = vec![0; n_samples];

        for i in 0..n_samples {
            let mut class_counts = HashMap::new();
            for j in 0..trees.len() {
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
}

pub trait OutlierForest<T: Tree>: Forest<T> {
    fn predict_(&self, data: &[Sample]) -> Vec<isize> {
        let scores = self.score_samples(data);
        let mut predictions = Vec::new();
        for i in 0..data.len() {
            predictions.push(if scores[i] > 0.5 { 1 } else { 0 });
        }
        predictions
    }
    fn depth_samples(&self, data: &[Sample]) -> impl ParallelIterator<Item = Vec<f64>> {
        let trees: &Vec<T> = self.get_trees();
        let depths = (0..data.len())
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0; trees.len()];
                for (tree_idx, tree) in trees.iter().enumerate() {
                    let sample = &tree.transform(&[data[i].clone()])[0];
                    row[tree_idx] = Self::path_length(tree, sample);
                }
                row
            });
        depths
    }
    fn score_samples(&self, data: &[Sample]) -> Vec<f64> {
        let (config, _) = self.get_forest_config();
        let average_path_length_max_samples = T::average_path_length(self.get_max_samples());
        let depths = self.depth_samples(data);
        let mut scores = vec![0.0; data.len()];
        if let Some(aggregation) = &config.aggregation {
            for (i, depth) in depths.iter().enumerate() {
                scores[i] = aggregation.combine(depth, average_path_length_max_samples);
            }
        } else {
            scores = depths
                .iter()
                .map(|depth| {
                    let mut sum = 0.0;
                    for d in depth {
                        sum += d;
                    }
                    2.0_f64.powf(-(sum / depth.len() as f64) / average_path_length_max_samples)
                })
                .collect();
        }
        // let combiner = config.aggregation.clone().unwrap_or(Combiner::new());
        // for (i, depth) in depths.iter().enumerate() {
        //     scores[i] = combiner.combine(depth, average_path_length_max_samples);
        // }
        scores
    }
    fn path_length(tree: &T, x: &Sample) -> f64 {
        T::SplitParameters::path_length(tree, x)
    }
}

pub fn generate_indices(
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
