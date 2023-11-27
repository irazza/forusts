#![allow(dead_code)]
use crate::feature_extraction::statistics::{EULER_MASCHERONI, mean, std, slope};
use crate::tree::{
    extra_tree::ExtraTree,
    node::Node,
    tree::{Tree, MaxFeatures}
};
use rand::{seq::SliceRandom, thread_rng, Rng};
use rayon::prelude::*;
use std::cmp::min;

pub struct TimeSeriesIsolationForest {
    trees: Vec<ExtraTree>,
    n_trees: usize,
    n_intervals: usize,
    min_interval_length: usize,
    intervals: Vec<Vec<(usize, usize)>>,
    max_features: MaxFeatures,
    max_depth: Option<usize>,
    max_samples: usize,
}

impl TimeSeriesIsolationForest {
    pub fn new(
        n_trees: usize,
        n_intervals: usize,
        min_interval_length: usize,
        max_features: MaxFeatures,
        max_depth: Option<usize>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            n_intervals,
            min_interval_length,
            intervals: Vec::new(),
            max_features,
            max_depth,
            max_samples: 256,
        }
    }

    pub fn transform(x: &Vec<Vec<f64>>, intervals: &Vec<(usize, usize)>) -> Vec<Vec<f64>> {
        let n_samples = x.len();
        let mut transformed_x: Vec<Vec<f64>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in intervals {
                let mean = mean(&x[j][*start..*end]);
                let std = std(&x[j][*start..*end]);
                let slope = slope(&x[j][*start..*end]);
                sample.extend([mean, std, slope].into_iter());
            }
            transformed_x.push(sample);
        }
        transformed_x
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>) {
        self.max_samples = min(256, x.len());
        // Generate n_intervals, with random start and end
        let n_features = x[0].len();
        for _i in 0..self.n_trees {
            let mut intervals = Vec::new();
            for _j in 0..self.n_intervals {
                let start = thread_rng().gen_range(0..n_features - self.min_interval_length);
                let end = thread_rng().gen_range(start + self.min_interval_length..n_features);
                intervals.push((start, end));
            }
            self.intervals.push(intervals);
        }

        self.trees
            .par_extend((0..self.n_trees).into_par_iter().map(|i| {
                let transformed_x = Self::transform(x, &self.intervals[i]);
                let mut n_samples: Vec<usize> = (0..x.len()).collect();
                n_samples.shuffle(&mut rand::thread_rng());

                let mut tree = ExtraTree::new(
                    self.max_depth.unwrap_or(usize::MAX),
                    1
                );
                tree.fit(
                    &(0..self.max_samples)
                        .into_iter()
                        .map(|i| &transformed_x[n_samples[i]])
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
        let average_path_length_max_samples =
            self.average_path_length([self.max_samples].to_vec())[0] as f64;
        let mut depths = Vec::new();

        depths.par_extend((0..self.n_trees).into_par_iter().map(|i| {
            let tree = &self.trees[i];
            let transformed_x = Self::transform(x, &self.intervals[i]);
            let node_order = transformed_x
                .iter()
                .map(|sample| tree.predict_leaf(sample))
                .collect::<Vec<&Node>>();
            let average_path_length_per_tree = self.average_path_length(
                node_order
                    .iter()
                    .map(|node| node.get_n_samples())
                    .collect::<Vec<usize>>(),
            );
            let decision_path_lengths = node_order
                .iter()
                .map(|node| node.get_depth())
                .collect::<Vec<usize>>();
            average_path_length_per_tree
                .iter()
                .zip(decision_path_lengths.iter())
                .map(|(x, y)| x + y - 1)
                .collect::<Vec<usize>>()
        }));

        let depths = (0..depths[0].len())
            .into_iter()
            .map(|c| depths.iter().map(|r| r[c]).sum())
            .collect::<Vec<usize>>();

        let denominator = self.trees.len() as f64 * average_path_length_max_samples;

        // Anomaly score
        depths
            .iter()
            .map(|x| 2.0f64.powf(-(*x as f64) / denominator))
            .collect::<Vec<f64>>()

        // Enhanced anomaly score
        //TODO
    }

    fn average_path_length(&self, n_samples_leaf: Vec<usize>) -> Vec<usize> {
        let mut average_path_length = vec![0; n_samples_leaf.len()];
        let mask1 = &n_samples_leaf
            .iter()
            .map(|x| *x <= 1)
            .collect::<Vec<bool>>();
        let mask2 = &n_samples_leaf
            .iter()
            .map(|x| *x == 2)
            .collect::<Vec<bool>>();
        let not_mask = mask1
            .iter()
            .zip(mask2)
            .map(|(x, y)| !(*x || *y))
            .collect::<Vec<bool>>();

        mask1
            .iter()
            .zip(average_path_length.iter_mut())
            .for_each(|(x, y)| {
                if *x {
                    *y = 0;
                }
            });

        mask2
            .iter()
            .zip(average_path_length.iter_mut())
            .for_each(|(x, y)| {
                if *x {
                    *y = 1;
                }
            });

        for (i, (x, y)) in not_mask
            .iter()
            .zip(average_path_length.iter_mut())
            .enumerate()
        {
            if *x {
                *y = ((2.0 * (f64::ln((n_samples_leaf[i] - 1) as f64) + EULER_MASCHERONI))
                    - (2.0 * (n_samples_leaf[i] - 1) as f64 / n_samples_leaf[i] as f64))
                    as usize;
            }
        }
        average_path_length
    }
}
