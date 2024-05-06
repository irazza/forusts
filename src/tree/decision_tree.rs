use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng, Rng};

use crate::{
    forest::forest::{ClassificationForestConfig, ClassificationTree},
    tree::{
        node::Node,
        tree::{Criterion, MaxFeatures, Tree},
    },
    utils::structures::Sample,
};

use super::tree::StandardSplit;

pub struct DecisionTreeConfig {
    pub criterion: Criterion,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub max_features: MaxFeatures,
}

pub struct DecisionTree {
    root: Node<StandardSplit>,
    config: DecisionTreeConfig,
}

impl ClassificationTree for DecisionTree {
    type TreeConfig = ClassificationForestConfig;
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
    type SplitParameters = StandardSplit;
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
        // Base case: all samples have the same target
        let first_target = samples[0].target;
        let is_all_same_target = samples.iter().all(|v| v.target == first_target);
        if is_all_same_target {
            return true;
        }
        return false;
    }
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        let mut rng = thread_rng();

        // Select best split
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_impurity = 0.0;

        let parent_impurity =
            self.config.criterion.to_fn::<DecisionTree>()(&samples.iter().map(|s| s.target).fold(
                HashMap::new(),
                |mut acc, x| {
                    *acc.entry(x).or_insert(0) += 1;
                    acc
                },
            ));

        let n_features = self.config.max_features.convert(samples[0].data.len());
        let mut features = (0..samples[0].data.len()).collect::<Vec<usize>>();
        features.shuffle(&mut rng);
        features.truncate(n_features);

        for feature in &features {
            let mut thresholds = samples
                .iter()
                .map(|f| f.data[*feature])
                .collect::<Vec<f64>>();
            thresholds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            thresholds.dedup();

            for threshold in &thresholds {
                // Split the samples based on the current threshold
                let mut left = HashMap::new();
                let mut right = HashMap::new();
                let mut left_items = 0;
                let mut right_items = 0;

                for sample in samples {
                    if sample.data[*feature] < *threshold {
                        *left.entry(sample.target).or_insert(0) += 1;
                        left_items += 1;
                    } else {
                        *right.entry(sample.target).or_insert(0) += 1;
                        right_items += 1;
                    }
                }

                // Compute the impurity of the split
                let left_impurity = self.config.criterion.to_fn::<DecisionTree>()(&left);
                let right_impurity = self.config.criterion.to_fn::<DecisionTree>()(&right);

                // Compute the weighted impurity of the split
                let impurity = match self.config.criterion {
                    Criterion::Gini => {
                        (left_impurity * left_items as f64 + right_impurity * right_items as f64)
                            / samples.len() as f64
                    }
                    Criterion::Entropy => {
                        (left_impurity * left_items as f64 + right_impurity * right_items as f64)
                            / samples.len() as f64
                    }
                    Criterion::Random => rng.gen_range(0.0..1.0),
                };

                // Update the best split if the current split is better
                let impurity = parent_impurity - impurity;
                if impurity > best_impurity {
                    best_feature = *feature;
                    best_threshold = *threshold;
                    best_impurity = impurity;
                }
            }
        }
        (
            StandardSplit {
                feature: best_feature,
                threshold: best_threshold,
            },
            best_impurity,
        )
    }
}
