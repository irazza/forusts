use std::sync::Arc;
use std::hash::Hash;
use super::{
    node::Node,
    tree::{Criterion, MaxFeatures, SplitParameters},
};
use crate::{
    feature_extraction::catch22::compute_catch,
    forest::{
        canonical_interval_forest::CanonicalIntervalForestConfig, forest::ClassificationTree,
    },
    tree::tree::Tree,
    utils::structures::Sample,
};
use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng, Rng};

pub const MIN_INTERVAL_LEN: usize = 20;
pub const TOT_ATTRIBUTES: usize = 25;

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct CanonicalIntervalSplit {
    pub interval: (usize, usize),
    pub feature: usize,
    pub threshold: f64,
}
impl Eq for CanonicalIntervalSplit {}
impl Ord for CanonicalIntervalSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl Hash for CanonicalIntervalSplit {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unreachable!();
    }
}
impl SplitParameters for CanonicalIntervalSplit {
    fn split(&self, sample: &Sample, _is_train: bool) -> bool {
        let feature = compute_catch(self.feature)(&sample.data[self.interval.0..self.interval.1]);
        feature < self.threshold
    }
    fn path_length<T: Tree<SplitParameters = Self>>(_tree: &T, _x: &Sample) -> f64 {
        unreachable!();
    }
}

#[derive(Clone, Debug)]
pub struct CanonicalIntervalTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub n_attributes: usize,
    pub n_intervals: usize,
    pub criterion: Criterion,
    pub max_features: MaxFeatures,
}

#[derive(Clone, Debug)]
pub struct CanonicalIntervalTree {
    root: Node<CanonicalIntervalSplit>,
    config: CanonicalIntervalTreeConfig,
}

impl ClassificationTree for CanonicalIntervalTree {
    type TreeConfig = CanonicalIntervalForestConfig;
    fn from_classification_config(config: &Self::TreeConfig) -> Self {
        Self::new(CanonicalIntervalTreeConfig {
            max_depth: config.classification_config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: config.classification_config.min_samples_split,
            n_attributes: config.n_attributes,
            n_intervals: config.n_intervals,
            criterion: config.classification_config.criterion,
            max_features: config.classification_config.max_features,
        })
    }
}

impl Tree for CanonicalIntervalTree {
    type Config = CanonicalIntervalTreeConfig;
    type SplitParameters = CanonicalIntervalSplit;
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
        // Generate n_intervals random intervals
        let mut intervals = Vec::new();
        for _ in 0..self.config.n_intervals {
            let start = rng.gen_range(0..samples[0].data.len() - MIN_INTERVAL_LEN);
            let end = rng.gen_range(start + MIN_INTERVAL_LEN..samples[0].data.len());

            intervals.push((start, end));
        }
        // Select n_attributes random features from catch22
        let mut attributes = (0..TOT_ATTRIBUTES).collect::<Vec<usize>>();
        attributes.shuffle(&mut rng);
        attributes.truncate(self.config.n_attributes);

        // For each interval, compute the randomly selected features
        let mut transformed_samples = Vec::new();
        for sample in samples {
            let mut transformed_sample = Vec::new();
            for (start, end) in &intervals {
                for i in &attributes {
                    transformed_sample.push(compute_catch(*i)(&sample.data[*start..*end]));
                }
            }
            transformed_samples.push(Sample {
                data: Arc::new(transformed_sample),
                target: sample.target,
            });
        }

        // Select best split
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_impurity = 0.0;

        let parent_impurity = self.config.criterion.to_fn::<CanonicalIntervalTree>()(
            &samples
                .iter()
                .map(|s| s.target)
                .fold(HashMap::new(), |mut acc, x| {
                    *acc.entry(x).or_insert(0) += 1;
                    acc
                }),
        );

        let n_features = self
            .config
            .max_features
            .convert(self.config.n_attributes * self.config.n_intervals);
        let mut features =
            (0..self.config.n_attributes * self.config.n_intervals).collect::<Vec<usize>>();
        features.shuffle(&mut rng);
        features.truncate(n_features);

        for feature in &features {
            let mut thresholds = transformed_samples
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

                for sample in &transformed_samples {
                    if sample.data[*feature] < *threshold {
                        *left.entry(sample.target).or_insert(0) += 1;
                        left_items += 1;
                    } else {
                        *right.entry(sample.target).or_insert(0) += 1;
                        right_items += 1;
                    }
                }

                // Compute the impurity of the split
                let left_impurity = self.config.criterion.to_fn::<CanonicalIntervalTree>()(&left);
                let right_impurity = self.config.criterion.to_fn::<CanonicalIntervalTree>()(&right);

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
        let interval = intervals[best_feature / self.config.n_attributes];
        let feature = attributes[best_feature % self.config.n_attributes];
        (
            CanonicalIntervalSplit {
                interval,
                feature,
                threshold: best_threshold,
            },
            best_impurity,
        )
    }
}
