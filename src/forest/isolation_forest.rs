#![allow(dead_code)]
use crate::tree::{
    decision_tree::{Criterion, DecisionTree, MaxFeatures, Splitter},
    node::Node,
};
use hashbrown::HashMap;
use parking_lot::Mutex;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::{
    cmp::max,
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct IsolationForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    max_features: MaxFeatures,
    max_depth: Option<usize>,
}

impl IsolationForest {

    pub fn new(
        n_trees: usize,
        max_features: MaxFeatures,
        max_depth: Option<usize>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_features,
            max_depth,
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>){
        let n_samples = x.len();

        self.trees
            .par_extend((0..self.n_trees).into_par_iter().map(|_i| {
                let bootstrap_indices: Vec<usize> = (0..n_samples)
                    .map(|_| thread_rng().gen_range(0..n_samples))
                    .collect();

                let mut tree = DecisionTree::new(
                    Criterion::None,
                    Splitter::Random,
                    self.max_depth.unwrap_or(usize::MAX),
                    1,
                    1,
                    self.max_features,
                );
                tree.fit(
                    &bootstrap_indices.iter().map(|i| &x[*i]).collect(),
                    &bootstrap_indices.iter().map(|i| *i).collect(),
                );
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
}
