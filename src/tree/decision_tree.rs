use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng};

use crate::{
    tree::{
        node::Node,
        tree::{Criterion, MaxFeatures, Tree},
    },
    utils::structures::Sample
};

pub struct DecisionTree {
    root: Node,
    criterion: Criterion,
    max_depth: usize,
    min_samples_split: usize,
    max_features_: MaxFeatures,
    max_features: usize,
}

impl DecisionTree {
    pub fn new(
        criterion: Criterion,
        max_depth: usize,
        min_samples_split: usize,
        max_features: MaxFeatures,
    ) -> Self {
        Self {
            root: Node::Leaf {
                class: 0,
                depth: 0,
                impurity: f64::MAX,
                n_samples: 0,
            },
            criterion,
            max_depth,
            min_samples_split,
            max_features_: max_features,
            max_features: 0,
        }
    }
}
impl Tree for DecisionTree {
    fn get_max_depth(&self) -> usize {
        self.max_depth
    }
    fn set_max_features(&mut self, max_features: usize) {
        self.max_features = max_features;
    }
    fn get_max_features(&self) -> MaxFeatures {
        self.max_features_
    }
    fn get_root(&self) -> &Node {
        &self.root
    }
    fn set_root(&mut self, root: Node) {
        self.root = root;
    }
    fn get_split(&self, samples: &[Sample<'_>]) -> (usize, f64, f64) {
        let n_samples = samples.len() as f64;
        let mut shuffled_features: Vec<usize> = (0..samples[0].data.len()).collect();
        shuffled_features.shuffle(&mut thread_rng());

        let mut best_feature = usize::MAX;
        let mut best_threshold = f64::MAX;
        let mut best_impurity = f64::MAX;

        for &feature_idx in &shuffled_features[..self.max_features] {
            let mut feature_values = samples
                .iter()
                .map(|v| (v.data[feature_idx], v.target))
                .collect::<Vec<_>>();
            feature_values.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());

            let mut left_class_counts = HashMap::new();
            let mut right_class_counts = HashMap::new();

            for (_, class) in &feature_values {
                *left_class_counts.entry(*class).or_insert(0) += 1;
            }

            for (i, &(threshold, class)) in feature_values.iter().enumerate() {
                left_class_counts.entry(class).and_modify(|e| *e -= 1);
                *right_class_counts.entry(class).or_insert(0) += 1;

                let left_impurity = (self.criterion.to_fn::<DecisionTree>())(&left_class_counts);
                let right_impurity = (self.criterion.to_fn::<DecisionTree>())(&right_class_counts);

                let right_size = (i + 1) as f64;
                let left_size = n_samples - right_size;

                let impurity = match self.criterion {
                    Criterion::Gini => {
                        (left_size / n_samples) * left_impurity
                            + (right_size / n_samples) * right_impurity
                    }
                    Criterion::Entropy => {
                        let mut class_counts = HashMap::new();
                        for Sample { target, .. } in samples {
                            *class_counts.entry(*target).or_insert(0) += 1;
                        }
                        let parent_entropy =
                            (self.criterion.to_fn::<DecisionTree>())(&class_counts);
                        1.0 / (parent_entropy
                            - ((left_size / n_samples) * left_impurity
                                + (right_size / n_samples) * right_impurity))
                    }
                };
                if impurity < best_impurity {
                    best_impurity = impurity;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }
        (best_feature, best_threshold, best_impurity)
    }
    fn pre_split_conditions(&self, samples: &[Sample<'_>], current_depth: usize) -> bool {
        // Base case: not enough samples or max depth reached
        if samples.len() <= self.min_samples_split || current_depth == self.max_depth {
            return true;
        }
        // Base case: samples are the same object
        let first_sample = &samples[0];
        let is_all_same_data = samples.iter().all(|v| v == first_sample);
        if is_all_same_data {
            return true;
        }
        // Base case: all samples have the same class
        let first_class = samples[0].target;
        let is_all_same_target = samples.iter().all(|v| v.target == first_class);
        if is_all_same_target {
            return true;
        }

        return false;
    }
    fn post_split_conditions(&self, new_impurity: f64, old_impurity: f64) -> bool {
        new_impurity <= f64::EPSILON
            || new_impurity.is_nan()
            || new_impurity - old_impurity <= f64::EPSILON
    }
}
