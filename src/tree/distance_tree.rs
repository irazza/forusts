use std::{cmp::max, fmt::Debug, hash::Hash, sync::Arc};

use super::{
    node::{LeafClassification, LeafClassifier, Node},
    tree::{Criterion, MaxFeatures, SplitParameters, Tree},
};
use crate::{
    distance::distances::{dtw, euclidean, msm, twe},
    feature_extraction::statistics::{mean, median},
    forest::forest::{ClassificationForestConfig, ClassificationTree},
    utils::structures::Sample,
};
use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

#[derive(Debug)]
pub struct DistanceLeafClassification {
    pub leaf_samples: Vec<Sample>,
}
impl LeafClassifier for DistanceLeafClassification {
    fn classify(&self, x: &[f64]) -> isize {
        let mut best_class = isize::MIN;
        let mut best_distance = f64::MAX;

        for s in &self.leaf_samples {
            let distance = twe(&s.data, x, 1.0);
            if distance < best_distance {
                best_distance = distance;
                best_class = s.target;
            }
        }
        best_class
    }
}
#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct DistanceSplit {
    pub candidate: Arc<Vec<f64>>,
    pub threshold: f64,
    pub interval: (usize, usize),
    pub dist_fn: fn(&[f64], &[f64], f64) -> f64,
}
impl Eq for DistanceSplit {}
impl Ord for DistanceSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl SplitParameters for DistanceSplit {
    fn split(&self, sample: &Sample, is_train: bool) -> bool {
        return (self.dist_fn)(
            &self.candidate,
            &sample.data[self.interval.0..self.interval.1],
            1.0,
        ) <= self.threshold;
    }
    fn path_length<T: Tree<SplitParameters = Self>>(_tree: &T, _x: &Sample) -> f64 {
        unreachable!();
    }
}

#[derive(Clone, Debug)]
pub struct DistanceTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub criterion: Criterion,
    pub max_features: MaxFeatures,
}

#[derive(Clone, Debug)]
pub struct DistanceTree {
    root: Node<DistanceSplit>,
    config: DistanceTreeConfig,
}

impl ClassificationTree for DistanceTree {
    type TreeConfig = ClassificationForestConfig;
    fn from_classification_config(config: &ClassificationForestConfig) -> Self {
        Self::new(DistanceTreeConfig {
            max_depth: config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: config.min_samples_split,
            criterion: config.criterion,
            max_features: config.max_features,
        })
    }
}

impl Tree for DistanceTree {
    type Config = DistanceTreeConfig;
    type SplitParameters = DistanceSplit;
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
        // Base case: all samples have the same target
        if samples.iter().all(|s| s.target == samples[0].target) {
            return true;
        }
        // Base case: all samples have the same features
        if samples.iter().all(|s| s.data == samples[0].data) {
            return true;
        }
        return false;
    }
    fn post_split_conditions(&self, _new_impurity: f64, _old_impurity: f64) -> bool {
        return false;
    }
    fn get_leaf_class(
        samples: &[Sample],
        parameters: Option<&Self::SplitParameters>,
    ) -> LeafClassification {
        let mut leaf_samples = Vec::new();
        for s in samples.iter() {
            leaf_samples.push(s.clone());
        }
        LeafClassification::Complex(Arc::new(DistanceLeafClassification { leaf_samples }))
    }
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        // let mut rng = ChaCha8Rng::seed_from_u64(42 as u64);
        let mut rng = thread_rng();
        let dist_fn = twe;

        let start;
        let end;

        // Generate a random interval
        let ts_len = samples[0].data.len();
        let min_interval = (rng.gen_range(0.0..1.0) * ts_len as f64).ceil() as usize;
        if min_interval == ts_len {
            start = 0;
            end = ts_len;
        } else {
            start = rng.gen_range(0..ts_len - min_interval);
            end = rng.gen_range(start + min_interval..ts_len);
        }

        // Initialize the best split
        let mut best_feature = usize::MAX;
        let mut best_threshold = f64::MAX;
        let mut best_impurity = 0.0;
        let mut best_interval = (0, 0);

        // Compute the impurity of the parent node
        let mut parent = HashMap::new();
        for s in samples.iter() {
            *parent.entry(s.target).or_insert(0) += 1;
        }
        let parent_impurity = self.config.criterion.to_fn::<DistanceTree>()(&parent);

        // Generate a random subsample (MaxFeatures) of features (length of sample)

        let mut subsamples_indeces = (0..samples.len()).collect::<Vec<_>>();
        subsamples_indeces.shuffle(&mut rng);
        subsamples_indeces.truncate(self.config.max_features.convert(samples.len()));

        for sample_idx in subsamples_indeces {
            // Compute the distances dtw the samples
            let distances = samples
                .iter()
                .map(|s| {
                    dist_fn(
                        &samples[sample_idx].data[start..end],
                        &s.data[start..end],
                        1.0,
                    )
                })
                .collect::<Vec<_>>();

            let mut distances_sorted = distances.clone();
            distances_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            for (low, high) in distances_sorted.windows(2).map(|w| (w[0], w[1])) {
                // Split the samples based on the current threshold
                let threshold = (low + high) / 2.0;
                let mut left = HashMap::new();
                let mut right = HashMap::new();
                for (i, _) in distances.iter().enumerate() {
                    if distances[i] <= threshold {
                        *left.entry(samples[i].target).or_insert(0) += 1;
                    } else {
                        *right.entry(samples[i].target).or_insert(0) += 1;
                    }
                }

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                // Compute the impurity of the split
                let n_samples = samples.len() as f64;
                let n_left = left.values().sum::<usize>() as f64;
                let n_right = right.values().sum::<usize>() as f64;
                let left_impurity =
                    self.config.criterion.to_fn::<DistanceTree>()(&left) * (n_left / n_samples);
                let right_impurity =
                    self.config.criterion.to_fn::<DistanceTree>()(&right) * (n_right / n_samples);

                // Compute the weighted impurity of the split
                let impurity = parent_impurity - (left_impurity + right_impurity);

                // Update the best split if the current split is better
                if impurity > best_impurity {
                    best_feature = sample_idx;
                    best_threshold = threshold;
                    best_impurity = impurity;
                    best_interval = (start, end);
                }
            }
        }
        (
            DistanceSplit {
                candidate: Arc::new(
                    samples[best_feature].data[best_interval.0..best_interval.1].to_vec(),
                ),
                interval: best_interval,
                threshold: best_threshold,
                dist_fn: dist_fn,
            },
            best_impurity,
        )
    }
}

// let mut thresholds = Vec::new();
//         let mut candidate = None;

//         // Iterate over feature to find a features which can be splitted
//         let mut features = (0..samples.len()).collect::<Vec<_>>();
//         features.shuffle(&mut rng);
//         features.truncate(self.config.max_features.convert(samples.len()));

//         for feature_idx in features {

//             // Choose a random threshold
//             thresholds = samples.iter().map(|s| dist_fn(&samples[feature_idx].data[start..end], &s.data[start..end])).collect::<Vec<_>>();
//             thresholds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
//             thresholds.dedup();

//             // If there is only one threshold, choose another feature
//             if thresholds.len() > 1 {
//                 candidate = Some(feature_idx);
//                 break;
//             }
//         }
//         // let normal: f64 = thread_rng().sample(StandardNormal);
//         // let normal = f64::max(-1.0, f64::min(1.0, normal));
//         // let r_max = 1.0;
//         // let r_min = -1.0;
//         // let t_max = thresholds.len() as f64 - 1.0;
//         // let t_min = 1.0;
//         // let threshold_idx = (((normal - r_min) / (r_max - r_min)) * (t_max - t_min) + t_min) as usize;

//         // let threshold = thresholds[threshold_idx];
//         let threshold = thresholds[rng.gen_range(1..thresholds.len())];

//         (
//             DistanceSplit {
//                 candidate: Arc::new(samples[candidate.unwrap()].data[start..end].to_vec()),
//                 threshold: threshold,
//                 interval: (start, end),
//                 dist_fn: dist_fn,
//             },
//             rng.gen_range(f64::EPSILON..1.0),
//         )
