use crate::{
    distance::distances::Distance,
    feature_extraction::statistics::mean,
    forest::{distance_set_forest::DistanceSetForestConfig, forest::ClassificationTree},
    utils::structures::Sample,
};
use std::hash::Hash;
use std::{cmp::max, fmt::Debug, sync::Arc};

use super::{
    node::{LeafClassification, LeafClassifier, Node},
    tree::{Criterion, MaxFeatures, SplitParameters, Tree},
};

use rand::{seq::SliceRandom, thread_rng, Rng};

#[derive(Debug)]
pub struct DistanceSetLeafClassification {
    pub leaf_samples: Vec<Sample>,
    pub distance: Distance,
}
impl LeafClassifier for DistanceSetLeafClassification {
    fn classify(&self, x: &[f64]) -> isize {
        let mut best_class = 0;
        let mut best_distance = f64::MAX;

        for s in &self.leaf_samples {
            let distance = self.distance.distance(&s.data, x, f64::log2(s.data.len() as f64).ceil());
            if distance < best_distance {
                best_distance = distance;
                best_class = s.target;
            }
        }
        best_class
    }
}

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct DistanceSetSplit {
    pub left_candidates: Vec<Arc<Vec<f64>>>,
    pub right_candidates: Vec<Arc<Vec<f64>>>,
    pub interval: (usize, usize),
    pub distance: Distance,
    pub band: f64,
}
impl Hash for DistanceSetSplit {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unreachable!();
    }
}
impl Eq for DistanceSetSplit {}
impl Ord for DistanceSetSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl SplitParameters for DistanceSetSplit {
    fn split(&self, sample: &Sample, _is_train: bool) -> bool {
        let left_distances = self
            .left_candidates
            .iter()
            .map(|c| self.distance.distance(&c, &sample.data[self.interval.0..self.interval.1], self.band))
            .collect::<Vec<_>>();
        let right_distances = self
            .right_candidates
            .iter()
            .map(|c| self.distance.distance(&c, &sample.data[self.interval.0..self.interval.1], self.band))
            .collect::<Vec<_>>();

        mean(&left_distances) < mean(&right_distances)
    }
    fn path_length<T: Tree<SplitParameters = Self>>(_tree: &T, _x: &Sample) -> f64 {
        unreachable!();
    }
}

#[derive(Clone, Debug)]
pub struct DistanceSetTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub criterion: Criterion,
    pub max_features: MaxFeatures,
    pub distance: Distance,
}

#[derive(Clone, Debug)]
pub struct DistanceSetTree {
    root: Node<DistanceSetSplit>,
    config: DistanceSetTreeConfig,
}

impl ClassificationTree for DistanceSetTree {
    type TreeConfig = DistanceSetForestConfig;
    fn from_classification_config(config: &Self::TreeConfig) -> Self {
        Self::new(DistanceSetTreeConfig {
            max_depth: config.classification_config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: config.classification_config.min_samples_split,
            criterion: config.classification_config.criterion,
            max_features: config.classification_config.max_features,
            distance: config.distance,
        })
    }
}

impl Tree for DistanceSetTree {
    type Config = DistanceSetTreeConfig;
    type SplitParameters = DistanceSetSplit;
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
    fn get_leaf_class(
        samples: &[Sample],
        _parameters: Option<&Self::SplitParameters>,
    ) -> LeafClassification {
        let mut leaf_samples = Vec::new();

        for s in samples.iter() {
            leaf_samples.push(s.clone());
        }
        LeafClassification::Complex(Arc::new(DistanceSetLeafClassification { leaf_samples, distance: Distance::DTW}))
    }
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        // let mut rng = ChaCha8Rng::seed_from_u64(42 as u64);
        let mut rng = thread_rng();

        // Generate a random interval
        let start;
        let end;
        let ts_len = samples[0].data.len();
        let min_interval = (rng.gen_range(0.1..1.0) * ts_len as f64).ceil() as usize;
        if min_interval == ts_len {
            start = 0;
            end = ts_len;
        } else {
            start = rng.gen_range(0..ts_len - min_interval);
            end = rng.gen_range(start + min_interval..ts_len);
        }

        // Generate a random subsample (MaxFeatures) of elements
        let mut subsamples_indices = (0..samples.len()).collect::<Vec<_>>();
        subsamples_indices.shuffle(&mut rng);
        subsamples_indices.truncate(max(2, self.config.max_features.convert(samples.len())));

        // Randomly split subsamples into two clusters
        let mut left_candidates = Vec::new();
        let mut right_candidates = Vec::new();
        for (i, s) in subsamples_indices.iter().enumerate() {
            if i%2 == 0 {
                left_candidates.push(Arc::new(samples[*s].data[start..end].to_vec()));
            } else {
                right_candidates.push(Arc::new(samples[*s].data[start..end].to_vec()));
            }
        }
        

        (
            DistanceSetSplit {
                left_candidates,
                right_candidates,
                interval: (start, end),
                distance: self.config.distance,
                band: f64::log2((end - start) as f64).ceil(),
            },
            rng.gen_range(f64::EPSILON..1.0),
        )
    }
}

        // let mut distance_matrix = vec![vec![0.0; subsamples_indices.len()]; subsamples_indices.len()];
        // let sakoe_chiba = f64::log2((end - start) as f64).ceil();
        // for i in 0..subsamples_indices.len() - 1 {
        //     for j in i+1..subsamples_indices.len() {
        //         distance_matrix[i][j] = self.config.distance.to_fn()(
        //             &samples[subsamples_indices[i]].data[start..end],
        //             &samples[subsamples_indices[j]].data[start..end],
        //             sakoe_chiba,
        //         );
        //         distance_matrix[j][i] = distance_matrix[i][j];
        //     }
        // }

        // let clusters = k_means(2, &distance_matrix);
        // let mut left_candidates = Vec::new();
        // let mut right_candidates = Vec::new();

        // assert_eq!(unique(&clusters).len(), 2, "{:?}", clusters);

        // for (i, cluster) in clusters.iter().enumerate() {
        //     if cluster == &0 {
        //         left_candidates.push(Arc::new(samples[subsamples_indices[i]].data[start..end].to_vec()));
        //     } else {
        //         right_candidates.push(Arc::new(samples[subsamples_indices[i]].data[start..end].to_vec()));
        //     }
        // }