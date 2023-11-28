use crate::{
    feature_extraction::statistics::EULER_MASCHERONI,
    tree::{extra_tree::ExtraTree, tree::Tree},
};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::cmp::min;

pub struct IsolationForest {
    trees: Vec<ExtraTree>,
    n_trees: usize,
    max_samples: usize,
    enhanced_anomaly_score: Option<bool>,
    max_depth: Option<usize>,
}

impl IsolationForest {
    pub fn new(
        n_trees: usize,
        enhanced_anomaly_score: Option<bool>,
        max_depth: Option<usize>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_depth,
            enhanced_anomaly_score,
            max_samples: 256,
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>) {
        self.max_samples = min(256, x.len());

        self.trees
            .par_extend((0..self.n_trees).into_par_iter().map(|_i| {
                let mut n_samples: Vec<usize> = (0..x.len()).collect();
                n_samples.shuffle(&mut rand::thread_rng());
                let mut tree = ExtraTree::new(
                    self.max_depth
                        .unwrap_or(self.max_samples.ilog2() as usize + 1),
                    1,
                );
                tree.fit(
                    &(0..self.max_samples)
                        .into_iter()
                        .map(|i| &x[n_samples[i]])
                        .collect::<Vec<&Vec<f64>>>(),
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
        // let average_path_length_max_samples =
        //     self.average_path_length([self.max_samples].to_vec())[0] as f64;
        // let mut depths = Vec::new();

        // for tree in &self.trees {
        //     let node_order = x
        //         .iter()
        //         .map(|sample| tree.predict_leaf(sample))
        //         .collect::<Vec<&Node>>();
        //     let average_path_length_per_tree = self.average_path_length(
        //         node_order
        //             .iter()
        //             .map(|node| node.get_n_samples())
        //             .collect::<Vec<usize>>(),
        //     );
        //     let decision_path_lengths = node_order
        //         .iter()
        //         .map(|node| node.get_depth())
        //         .collect::<Vec<usize>>();
        //     depths.push(
        //         average_path_length_per_tree
        //             .iter()
        //             .zip(decision_path_lengths.iter())
        //             .map(|(x, y)| x + y - 1)
        //             .collect::<Vec<usize>>(),
        //     );
        // }

        // let depths = (0..depths[0].len())
        //     .into_iter()
        //     .map(|c| depths.iter().map(|r| r[c]).sum())
        //     .collect::<Vec<usize>>();

        // let denominator = self.trees.len() as f64 * average_path_length_max_samples;

        // depths
        //     .iter()
        //     .map(|x| 2.0f64.powf(-(*x as f64) / denominator))
        //     .collect::<Vec<f64>>()
        let mut scores = Vec::new();
        let path_length = self.path_length(x);
        let c_n = 2.0 * (f64::log2(self.max_samples as f64 - 1.0) + EULER_MASCHERONI)
            - 2.0 * (self.max_samples as f64 - 1.0) / self.max_samples as f64;
        if !self.enhanced_anomaly_score.unwrap_or(false) {
            let e_h = path_length
                .iter()
                .map(|x| x.iter().sum::<usize>() as f64 / self.n_trees as f64)
                .collect::<Vec<f64>>();
            for i in 0..x.len() {
                scores.push(2.0f64.powf(-e_h[i] / c_n));
            }
        } else {
            let enhanced_scores = path_length
                .iter()
                .map(|x| {
                    x.iter()
                        .map(|pl| 2.0f64.powf(-(*pl as f64) / c_n))
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();
            for i in 0..x.len() {
                scores.push(enhanced_scores[i].iter().sum::<f64>() / self.n_trees as f64);
            }
        }
        scores
    }

    fn path_length(&self, x: &Vec<Vec<f64>>) -> Vec<Vec<usize>> {
        let mut path_length = Vec::new();
        path_length.par_extend((0..x.len()).into_par_iter().map(|i| {
            let mut depth = Vec::new();
            for tree in &self.trees {
                depth.push(tree.predict_leaf(&x[i]).get_depth());
            }
            depth
        }));
        path_length
    }
}
