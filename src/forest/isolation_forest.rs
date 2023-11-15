#![allow(dead_code)]
use crate::{tree::{
    decision_tree::{Criterion, DecisionTree, MaxFeatures, Splitter},
    node::Node,
}, feature_extraction::statistics::{EULER_MASCHERONI, percentile}};
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
                let mut tree = DecisionTree::new(
                    Criterion::None,
                    Splitter::Random,
                    self.max_depth.unwrap_or(usize::MAX),
                    1,
                    1,
                    self.max_features,
                );
                tree.fit(
                    &(0..n_samples).into_iter().map(|i| &x[i]).collect(),
                    &(0..n_samples).collect(),
                );
                tree
            }));
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        let scores = self.score_samples(x);
        let mut predictions = Vec::new();
        let threshold = percentile(&scores, 95);
        for i in 0..x.len() {
            predictions.push(if scores[i] >  threshold {1} else {0});
        }
        predictions
    }

    pub fn score_samples(&self, x: &Vec<Vec<f64>>) -> Vec<f64>
    {
        let n_samples = x.len();
        let mut scores = Vec::new();
        let mut leaves: Vec<Vec<usize>> = Vec::new();
        let c_n = 2.0*(f64::ln((n_samples-1) as f64) +  EULER_MASCHERONI) - (2.0* (n_samples-1) as f64 / n_samples as f64);

        //Make predictions for each sample using each tree in the forest
        leaves.par_extend(
            self.trees
                .par_iter()
                .map(| tree| x.iter().map(|sample| tree.predict_leaf(sample).get_depth()).collect())
        );

        for i in 0..n_samples {
            let e_h = leaves
                .iter()
                .map(|leaf| leaf[i])
                .sum::<usize>() as f64
                / n_samples as f64;
                scores.push(2.0f64.powf(-e_h / c_n));
        }
        scores
    }
}
