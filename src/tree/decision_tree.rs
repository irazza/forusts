use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng};

use crate::{
    forest::forest::{ClassificationForestConfig, ClassificationTree},
    tree::{
        node::Node,
        tree::{Criterion, MaxFeatures, Tree},
    },
    utils::structures::Sample,
};

use super::tree::SplitTest;

pub struct DecisionTreeConfig {
    pub criterion: Criterion,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub max_features: MaxFeatures,
}

pub struct DecisionTree {
    root: Node<SplitTest>,
    config: DecisionTreeConfig,
}

impl ClassificationTree for DecisionTree {
    fn from_classification_config(config: &ClassificationForestConfig) -> Self {
        Self::new(DecisionTreeConfig {
            criterion: config.criterion,
            max_depth: config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: config.min_samples_split,
            max_features: config.max_features,
        })
    }
}

impl Tree for DecisionTree {
    type Config = DecisionTreeConfig;
    type SplitParameters = SplitTest;
    fn new(config: Self::Config) -> Self {
        Self {
            root: Node::new(),
            config,
        }
    }
    fn get_max_depth(&self) -> usize {
        self.config.max_depth
    }
    fn get_root(&self) -> &Node<Self::SplitParameters> {
        &self.root
    }
    fn set_root(&mut self, root: Node<Self::SplitParameters>) {
        self.root = root;
    }
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        // Initialize the best split
        let mut best_feature = usize::MAX;
        let mut best_threshold = f64::MAX;
        let mut best_impurity = f64::MAX;

        // Generate a random subsample (MaxFeatures) of features (length of sample)
        let mut features_subsample = (0..samples[0].data.len()).collect::<Vec<_>>();
        features_subsample.shuffle(&mut thread_rng());
        let features_subsample =
            features_subsample[..self.config.max_features.convert(samples[0].data.len())].to_vec();

        for feature_idx in features_subsample {
            // Extract the thresholds and the classes for the current feature
            let thresholds = samples
                .iter()
                .map(|s| s.data[feature_idx])
                .collect::<Vec<_>>();

            for threshold in thresholds {
                // Split the samples based on the current threshold
                let mut left = HashMap::new();
                let mut right = HashMap::new();

                for sample in samples {
                    if sample.data[feature_idx] <= threshold {
                        *left.entry(sample.target).or_insert(0) += 1;
                    } else {
                        *right.entry(sample.target).or_insert(0) += 1;
                    }
                }

                // Compute the impurity of the split
                let left_impurity = self.config.criterion.to_fn::<DecisionTree>()(&left);
                let right_impurity = self.config.criterion.to_fn::<DecisionTree>()(&right);

                // Compute the weighted impurity of the split
                let impurity = match self.config.criterion {
                    Criterion::Gini => {
                        (left_impurity * left.values().sum::<usize>() as f64
                            + right_impurity * right.values().sum::<usize>() as f64)
                            / samples.len() as f64
                    }
                    Criterion::Entropy => {
                        let mut class_counts = HashMap::new();
                        for Sample { target, .. } in samples {
                            *class_counts.entry(*target).or_insert(0) += 1;
                        }
                        let parent = (self.config.criterion.to_fn::<DecisionTree>())(&class_counts);
                        let impurity = parent
                            - (left_impurity * left.values().sum::<usize>() as f64
                                + right_impurity * right.values().sum::<usize>() as f64)
                                / samples.len() as f64;
                        1.0 / impurity
                    }
                    Criterion::Random => 1.0,
                };

                // (left_impurity * left.values().sum::<usize>() as f64
                //     + right_impurity * right.values().sum::<usize>() as f64)
                //     / samples.len() as f64;

                // Update the best split if the current split is better
                if impurity < best_impurity {
                    best_feature = feature_idx;
                    best_threshold = threshold;
                    best_impurity = impurity;
                }
            }
        }
        (
            SplitTest {
                feature: best_feature,
                threshold: best_threshold,
            },
            best_impurity,
        )
    }
    fn pre_split_conditions(&self, samples: &[Sample], current_depth: usize) -> bool {
        // Base case: not enough samples or max depth reached
        if samples.len() <= self.config.min_samples_split || current_depth == self.config.max_depth
        {
            return true;
        }
        // Base case: samples are the same object
        let first_sample = &samples[0].data;
        let is_all_same_data = samples.iter().all(|v| &v.data == first_sample);
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
        return (old_impurity - new_impurity).abs() < f64::EPSILON;
    }
}
