#![allow(dead_code)]
use crate::{
    feature_extraction::statistics::{percentile, EULER_MASCHERONI},
    tree::{
        decision_tree::{Criterion, DecisionTree, MaxFeatures, Splitter},
        node::Node, self,
    },
};
use hashbrown::HashMap;
use parking_lot::Mutex;
use rand::{seq::SliceRandom, thread_rng, Rng};
use rayon::{prelude::*, vec};
use std::{
    cmp::{max, min},
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct IsolationForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    max_features: MaxFeatures,
    max_depth: Option<usize>,
    max_samples: usize,
}

impl IsolationForest {
    pub fn new(n_trees: usize, max_features: MaxFeatures, max_depth: Option<usize>) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_features,
            max_depth,
            max_samples: 256,
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>) {
        self.max_samples = min(256, x.len());

        self.trees
            .par_extend((0..self.n_trees).into_par_iter().map(|_i| {
                let mut n_samples: Vec<usize> =  (0..x.len()).collect();
                n_samples.shuffle(&mut rand::thread_rng());
                let mut tree = DecisionTree::new(
                    Criterion::None,
                    Splitter::Random,
                    self.max_depth.unwrap_or(usize::MAX),
                    1,
                    1,
                    self.max_features,
                );
                tree.fit(
                    &(0..self.max_samples).into_iter().map(|i| &x[n_samples[i]]).collect::<Vec<&Vec<f64>>>(),
                    &(0..self.max_samples).collect(),
                );
                tree
            }));
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        let scores = self.score_samples(x);
        let mut predictions = Vec::new();
        for i in 0..x.len() {
            predictions.push(if scores[i] > 0.5 { 1 } else { 0 });
        }
        predictions
    }

    pub fn score_samples(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {

        let average_path_length_max_samples = self.average_path_length([self.max_samples].to_vec())[0] as f64;
        let mut depths = Vec::new();

        for tree in &self.trees {
            let node_order = x.iter().map(|sample| tree.predict_leaf(sample)).collect::<Vec<&Node>>();
            let average_path_length_per_tree = self.average_path_length(node_order.iter().map(|node| node.get_n_samples()).collect::<Vec<usize>>());
            let decision_path_lengths = node_order.iter().map(|node| node.get_depth()).collect::<Vec<usize>>();
            depths.push(average_path_length_per_tree.iter().zip(decision_path_lengths.iter()).map(|(x, y)| x + y -1).collect::<Vec<usize>>());
        }

        let depths= (0..depths[0].len())
        .into_iter()
        .map(|c| depths.iter().map(|r| r[c]).sum())
        .collect::<Vec<usize>>();

        let denominator = self.trees.len() as f64 * average_path_length_max_samples;

        depths.iter().map(|x| 2.0f64.powf(-(*x as f64) / denominator)).collect::<Vec<f64>>()

        //let average_path_length_max_samples = self.average_path_length();



        // let n_samples = x.len();
        // let mut scores = Vec::new();
        // let mut leaves: Vec<Vec<usize>> = Vec::new();
        // let c_n = 2.0 * (f64::ln((n_samples - 1) as f64) + EULER_MASCHERONI)
        //     - (2.0 * (n_samples - 1) as f64 / n_samples as f64);

        // //Make predictions for each sample using each tree in the forest
        // leaves.par_extend(self.trees.par_iter().map(|tree| {
        //     x.iter()
        //         .map(|sample| tree.predict_leaf(sample).get_depth())
        //         .collect()
        // }));

        // for i in 0..n_samples {
        //     let e_h = leaves.iter().map(|leaf| leaf[i]).sum::<usize>() as f64 / n_samples as f64;
        //     scores.push(2.0f64.powf(-e_h / c_n));
        // }
        // scores
    }

    fn average_path_length(&self, n_samples_leaf: Vec<usize>) -> Vec<usize> {
        let mut average_path_length = vec![0; n_samples_leaf.len()];
        let mask1 = &n_samples_leaf.iter().map(|x| *x <= 1).collect::<Vec<bool>>();
        let mask2 = &n_samples_leaf.iter().map(|x| *x == 2).collect::<Vec<bool>>();
        let not_mask = mask1.iter().zip(mask2).map(|(x, y)| !(*x || *y)).collect::<Vec<bool>>();

        mask1.iter().zip(average_path_length.iter_mut()).for_each(|(x, y)| {
            if *x {
                *y = 0;
            }
        });

        mask2.iter().zip(average_path_length.iter_mut()).for_each(|(x, y)| {
            if *x {
                *y = 1;
            }
        });

        for (i, (x, y)) in not_mask.iter().zip(average_path_length.iter_mut()).enumerate() {
            if *x {
                *y = ((2.0 * (f64::ln((n_samples_leaf[i] - 1) as f64) + EULER_MASCHERONI))
                    - (2.0 * (n_samples_leaf[i] - 1) as f64 / n_samples_leaf[i] as f64)) as usize;
            }
        }
        average_path_length
    }
}
