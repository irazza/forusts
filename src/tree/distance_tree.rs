use core::panic;
use std::{
    cmp::{max, min},
    sync::Arc,
};

use super::{
    node::{LeafClassifier, Node},
    tree::SplitParameters,
};
use crate::{
    distance::distances::{dtw, euclidean, twe}, feature_extraction::statistics::{fisher_score, mean, slope, stddev, unique, value_counts}, forest::forest::{
        ClassificationForestConfig, ClassificationTree}, tree::tree::Tree, utils::structures::Sample
};
use hashbrown::HashMap;
use rand::{thread_rng, Rng, seq::SliceRandom};

const MIN_INTERVAL_LENGTH: usize = 10;
const MIN_INTERVAL_PERC: f64 = 0.2;

#[derive(Clone, Debug)]
pub struct DistanceLeaf {
    pub clusters: HashMap<isize, Vec<Vec<f64>>>,
    pub interval: (usize, usize),
}
impl LeafClassifier for DistanceLeaf {
    fn classify(&self, x: &[f64]) -> isize {
        let mut min_dist = std::f64::INFINITY;
        let mut min_class = 0;
        for (class, cluster) in &self.clusters {
            let mut dist = 0.0;
            let mut counter = 0.0;
            for candidate in cluster {
                dist += euclidean(&x[self.interval.0..self.interval.1], &candidate);
                counter += 1.0;
            }
            let mean_dist = dist / counter;
            if mean_dist < min_dist {
                min_dist = mean_dist;
                min_class = *class;
            }
        }
        min_class
    }
}
pub struct KUnionFind {
    parent: Vec<usize>,
    k: usize,
}
impl KUnionFind {
    pub fn new(k: usize, n_samples: usize) -> Self {
        Self {
            parent: (0..n_samples).collect(),
            k,
        }
    }
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    pub fn union(&mut self, x: usize, y: usize) {
        let x_root = self.find(x);
        let y_root = self.find(y);
        self.parent[x_root] = y_root;
    }
    pub fn get_clusters(&mut self, distances: &Vec<Vec<f64>>) -> Vec<Vec<usize>> {
        for (i, obj) in distances.iter().enumerate() {
            let mut nn = obj.iter().enumerate().collect::<Vec<_>>();
            nn.select_nth_unstable_by(self.k, |a, b| a.1.partial_cmp(b.1).unwrap());
            for (j, _) in nn.iter().take(self.k) {
                self.union(i, *j);
            }
        }
        let mut clusters = HashMap::new();
        for i in 0..distances.len() {
            let root = self.find(i);
            clusters.entry(root).or_insert(Vec::new()).push(i);
        }
        clusters.into_iter().map(|(_, v)| v).collect()
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct DistanceSplit {
    pub left_candidates: Vec<Vec<f64>>,
    pub right_candidates: Vec<Vec<f64>>,
    pub interval: (usize, usize),
}

impl Ord for DistanceSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl Eq for DistanceSplit {}

impl SplitParameters for DistanceSplit {
    fn split(&self, sample: &Sample<'_>) -> bool {
        let mut min_l = std::f64::INFINITY;
        let mut min_r = std::f64::INFINITY;

        let sample = &sample.data[self.interval.0..self.interval.1];
        for candidate in &self.left_candidates {
            let dist = euclidean(&sample, &candidate);
            if dist < min_l {
                min_l = dist;
            }
        }
        for candidate in &self.right_candidates {
            let dist = euclidean(&sample, &candidate);
            if dist < min_r {
                min_r = dist;
            }
        }
        min_l < min_r
    }
    fn path_length<T: Tree<SplitParameters = Self>>(_tree: &T, _x: &Sample<'_>) -> f64 {
        unreachable!()
    }
}

#[derive(Clone, Debug)]
pub struct DistanceTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
}

#[derive(Clone, Debug)]
pub struct DistanceTree {
    root: Node<DistanceSplit>,
    config: DistanceTreeConfig,
}
impl DistanceTree {
    fn supervised_interval(samples: &[Sample<'_>], min_interval_len: usize) -> (usize, usize) {
        // let mut start = 0;
        // let mut end = samples[0].data.len();
        // let mut best_score = f64::INFINITY;
        // for i in 0..samples[0].data.len() - min_interval_len {
        //     for j in i + min_interval_len..samples[0].data.len() {
        //         let interval = (i, j);
        //         let mut classes = HashMap::new();
        //         for sample in samples {
        //             let entry = classes.entry(sample.target).or_insert(Vec::new());
        //             entry.push(&sample.data[interval.0..interval.1]);
        //         }
        //         let score = fisher_score(&classes);
        //         if score < best_score {
        //             best_score = score;
        //             start = i;
        //             end = j;
        //         }
        //     }
        // }
        // (start, end)
        let ts_length = samples[0].data.len();
        let start = thread_rng().gen_range(0..ts_length - min_interval_len);
        let end = thread_rng().gen_range(start + min_interval_len..ts_length);
        (start, end)
    }
}

impl ClassificationTree for DistanceTree {
    fn from_classification_config(config: &ClassificationForestConfig) -> Self {
        Self::new(DistanceTreeConfig {
            max_depth: config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: 2,
            // Setted to 2 to avoid empty child when splitting when there are only two samples
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
    fn pre_split_conditions(&self, samples: &[Sample<'_>], current_depth: usize) -> bool {
        // Base case: not enough samples or max depth reached
        if samples.len() <= self.config.min_samples_split || current_depth == self.config.max_depth
        {
            return true;
        }
        // Base case: all samples have the same target
        if samples.iter().all(|s| s.target == samples[0].target) {
            return true;
        }
        return false;
    }
    fn post_split_conditions(&self, new_impurity: f64, _old_impurity: f64) -> bool {
        // Check if there is a non empty split
        if new_impurity == std::f64::INFINITY {
            return true;
        }
        return false;
        // Update COnditions
    }
    fn get_leaf_class(
        samples: &[Sample<'_>],
        parameters: Option<&Self::SplitParameters>,
    ) -> super::node::LeafClassification {
        let interval = parameters
            .map(|p| p.interval)
            .unwrap_or((0, samples[0].data.len()));
        let mut distance_leaf = HashMap::new();
        for sample in samples {
            distance_leaf
                .entry(sample.target)
                .or_insert(Vec::new())
                .push(sample.data[interval.0..interval.1].to_vec());
        }
        super::node::LeafClassification::Complex(Arc::new(DistanceLeaf {
            clusters: distance_leaf,
            interval: interval,
        }))
    }

    fn get_split(&self, samples: &[Sample<'_>]) -> (Self::SplitParameters, f64) {
        let ts_length = samples[0].data.len();
        let mut min_interval_len = 0;

        if ((MIN_INTERVAL_PERC * ts_length as f64).ceil() as usize) < MIN_INTERVAL_LENGTH {
            min_interval_len = MIN_INTERVAL_LENGTH;
        } else {
            min_interval_len = (MIN_INTERVAL_PERC * ts_length as f64).ceil() as usize;
        }

        let (start, end) = Self::supervised_interval(samples, min_interval_len);
        
        let ts = samples
            .iter()
            .map(|s| &s.data[start..end])
            .collect::<Vec<_>>();

        let mut distances = vec![vec![0.0; ts.len()]; ts.len()];
        for i in 0..ts.len()-1 {
            for j in i+1..ts.len() {
                let dist = euclidean(&ts[i], &ts[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        let k = max(1, (samples.len() as f64).log2().log2().ceil() as usize);

        let mut sets = KUnionFind::new(k, ts.len());
        let mut clusters = sets.get_clusters(&distances);

        let mut splits: HashMap<bool, Vec<usize>> = HashMap::new();

        // Split all the clusters in pure (clusters with same target for every element) and impure
        for i in 0..clusters.len() {
            if clusters[i]
                .iter()
                .all(|x| samples[*x].target == samples[*clusters[i].first().unwrap()].target)
            {
                splits
                    .entry(true)
                    .or_insert(Vec::<usize>::new())
                    .extend(&clusters[i]);
            } else {
                splits
                    .entry(false)
                    .or_insert(Vec::<usize>::new())
                    .extend(&clusters[i]);
            }
        }

        // If there are only pure clusters, return a leaf node
        if splits.len() < 2 && splits.contains_key(&true) {
            let t = DistanceSplit {
                left_candidates: Vec::new(),
                right_candidates: Vec::new(),
                interval: (start, end),
            };
            return (t, f64::INFINITY);
        } else if splits.len() < 2 && splits.contains_key(&false){
            // If there are only impure clusters, split them randomically in two sets with balancing
            splits.clear();
            clusters.shuffle(&mut thread_rng());
            for i in 0..clusters.len() {
                if i%2 == 0 {
                    splits
                        .entry(true)
                        .or_insert(Vec::<usize>::new())
                        .extend(&clusters[i]);
                } else {
                    splits
                        .entry(false)
                        .or_insert(Vec::<usize>::new())
                        .extend(&clusters[i]);
                }
            }
            // If there is only one impure cluster, 
            if splits.len() < 2 {
                // TODO: Check if this is the best way to handle this case, because so many times we reach this point
                //println!("{:?}", clusters[0].iter().map(|x| samples[*x].target).collect::<Vec<_>>());
                let t = DistanceSplit {
                    left_candidates: Vec::new(),
                    right_candidates: Vec::new(),
                    interval: (start, end),
                };
                return (t, f64::INFINITY);
            }
        }

        let left_candidates = splits
            .get(&true)
            .unwrap_or_else(|| panic!("{:?}", splits))
            .iter()
            .map(|x| ts[*x].to_vec())
            .collect::<Vec<_>>();
        let right_candidates = splits
            .get(&false)
            .unwrap_or_else(|| panic!("{:?}", splits))
            .iter()
            .map(|x| ts[*x].to_vec())
            .collect::<Vec<_>>();

        (DistanceSplit {
            left_candidates,
            right_candidates,
            interval: (start, end),
        }, thread_rng().gen_range(0.0..1.0))
    }
}
