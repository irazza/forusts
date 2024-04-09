use std::{cmp::max, fmt::Debug, sync::Arc};

use super::{
    node::{LeafClassification, Node},
    tree::{Criterion, MaxFeatures, SplitParameters},
};
use crate::{
    distance::{self, distances::{self, euclidean}}, feature_extraction::statistics::stddev, forest::forest::{ClassificationForestConfig, ClassificationTree}, tree::tree::Tree, utils::structures::Sample
};
use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const N_HYPERPLANES: usize = 10;

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct DistanceSplitHyperplane {
    c: Vec<f64>,
    means: Vec<f64>,
    idx_attributes: Vec<usize>,
    candidates: Vec<Arc<Vec<f64>>>,
    p: f64,
    best_gain: f64,
    limit: f64,
}
impl DistanceSplitHyperplane {
    pub fn get_dist(&self, sample: &Sample) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.c.len() {
            sum += self.c[i]
                * (euclidean(&sample.data, &self.candidates[i]) - self.means[self.idx_attributes[i]]);
        }
        sum - self.p
    }
}
impl Eq for DistanceSplitHyperplane {}
impl Ord for DistanceSplitHyperplane {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.p.partial_cmp(&other.p).unwrap()
    }
}
impl SplitParameters for DistanceSplitHyperplane {
    fn split(&self, sample: &Sample, _is_train: bool) -> bool {
        self.get_dist(sample) < 0.0
    }
    fn path_length<T: Tree<SplitParameters = Self>>(tree: &T, x: &Sample) -> f64 {
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
    root: Node<DistanceSplitHyperplane>,
    config: DistanceTreeConfig,
}
impl DistanceTree {
    fn get_random_hyperplane(
        &self,
        data: &[Sample],
        samples: &Vec<Vec<f64>>,
        targets: &Vec<isize>,
        n_attributes: usize,
        means: &[f64],
        stddevs: &[f64],
        random_state: &mut ChaCha8Rng
    ) -> DistanceSplitHyperplane {
        // Compute the impurity of the parent node
        let mut parent = HashMap::new();
        for s in targets.iter() {
            *parent.entry(*s).or_insert(0) += 1;
        }
        
        let parent_impurity = self.config.criterion.to_fn::<DistanceTree>()(&parent);

        let idxs_candidates = stddevs
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > f64::EPSILON)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let mut subsampled_features = idxs_candidates;
        subsampled_features.shuffle(random_state);
        subsampled_features.truncate(n_attributes);
        let c = (0..subsampled_features.len())
            .into_iter()
            .map(|i| random_state.gen_range(-1.0..=1.0) / stddevs[subsampled_features[i]])
            .collect::<Vec<_>>();
        let mut p = samples
            .iter()
            .map(|s| {
                c.iter()
                    .zip(subsampled_features.iter())
                    .map(|(c, i)| c * (s[*i] - means[*i]))
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();
        p.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        //let p_stddev = stddev(&p);
        let mut best_p = 0.0;
        let mut best_gain = 0.0;

        for i in 1..p.len() {
            let mut left = HashMap::new();
            let mut right = HashMap::new();

            let split = DistanceSplitHyperplane {
                c: c.clone(),
                means: means.to_vec(),
                idx_attributes: subsampled_features.clone(),
                candidates: subsampled_features.iter().map(|i| data[*i].data.clone()).collect::<Vec<_>>(),
                p: p[i],
                best_gain: 0.0,
                limit: p[p.len() - 1] - p[0],
            };

            for j in 0..samples.len() {
                if split.split(&data[j], true) {
                    *left.entry(targets[j]).or_insert(0) += 1;
                } else {
                    *right.entry(targets[j]).or_insert(0) += 1;
                }
            }

            // Compute the impurity of the split
            let n_samples = samples.len() as f64;
            let n_left = left.values().sum::<usize>() as f64;
            let n_right = right.values().sum::<usize>() as f64;
            let left_impurity = self.config.criterion.to_fn::<DistanceTree>()(&left) * (n_left / n_samples);
            let right_impurity = self.config.criterion.to_fn::<DistanceTree>()(&right) * (n_right / n_samples);

            // Compute the weighted impurity of the split
            let impurity = match self.config.criterion {
                Criterion::Gini => {
                    parent_impurity - (left_impurity + right_impurity)
                }
                Criterion::Entropy => {
                    parent_impurity - (left_impurity + right_impurity)
                }
                Criterion::Random => left_impurity + right_impurity,
            };

            // Compute the gain of the split
            if impurity > best_gain {
                best_gain = impurity;
                best_p = p[i];
                //println!("{}, {}", left.values().sum::<usize>(), right.values().sum::<usize>());
            }
        }
        DistanceSplitHyperplane {
            c: c,
            means: means.to_vec(),
            idx_attributes: subsampled_features.clone(),
            candidates:  subsampled_features.iter().map(|i| data[*i].data.clone()).collect::<Vec<_>>(),
            p: best_p,
            best_gain: best_gain,
            limit: p[p.len() - 1] - p[0],
        }
    }
}

impl ClassificationTree for DistanceTree {
    fn from_classification_config(config: &ClassificationForestConfig) -> Self {
        Self::new(DistanceTreeConfig {
            max_depth: config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: config.min_samples_split,
            // Setted to 2 to avoid empty child when splitting when there are only two samples
            criterion: config.criterion,
            max_features: config.max_features,
        })
    }
}

impl Tree for DistanceTree {
    type Config = DistanceTreeConfig;
    type SplitParameters = DistanceSplitHyperplane;
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
        if samples.len() <= self.config.min_samples_split || current_depth == self.config.max_depth {
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
    fn post_split_conditions(&self, new_impurity: f64, old_impurity: f64) -> bool {
        if (new_impurity - old_impurity).abs() < f64::EPSILON {
            return true;
        }
        return false;
    }

    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        //let mut rng = ChaCha8Rng::seed_from_u64(42 as u64);
        let mut rng = ChaCha8Rng::from_rng(thread_rng()).unwrap();
        let n_features = self.config.max_features.convert(samples.len());
        let mut stddev = vec![0.0; samples.len()];
        let mut means = vec![0.0; samples.len()];

        let mut distances = vec![vec![0.0; samples.len()]; samples.len()];
        for i in 0..samples.len() {
            for j in 0..samples.len() {
                distances[i][j] = euclidean(&samples[i].data, &samples[j].data);
                distances[j][i] = distances[i][j];
            }
        }
        let targets = samples.iter().map(|s| s.target).collect::<Vec<_>>();

        for i in 0..distances.len() {
            let mean = distances.iter().map(|v| v[i]).sum::<f64>() / distances.len() as f64;
            let variance = distances
                .iter()
                .map(|v| (v[i] - mean).powi(2))
                .sum::<f64>()
                / distances.len() as f64;
            stddev[i] = variance.sqrt();
            means[i] = mean;
        }
        let mut best_gain = 0.0;
        let mut best_hp = DistanceSplitHyperplane {
            c: Vec::new(),
            means: Vec::new(),
            idx_attributes: Vec::new(),
            candidates: Vec::new(),
            p: 0.0,
            best_gain: 0.0,
            limit: 0.0,
        };

        for _i in 0..N_HYPERPLANES {
            let hp = self.get_random_hyperplane(samples, &distances, &targets, n_features, &means, &stddev, &mut rng);
            if best_gain < hp.best_gain {
                best_gain = hp.best_gain;
                best_hp = hp.clone();
            }

        }
        (best_hp.clone(), best_hp.best_gain)
    }
}
