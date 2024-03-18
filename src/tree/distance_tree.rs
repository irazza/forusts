use core::panic;
use std::cmp::{max, min};

use super::{
    node::Node,
    tree::{SplitParameters, SplitTest},
};
use crate::{
    distance::distances::{dtw, euclidean, twe},
    feature_extraction::{
        scamp::compute_selfmp,
        statistics::{
            fisher_score, mean, median, slope, stddev, unique, value_counts, zscore,
            EULER_MASCHERONI,
        },
    },
    forest::forest::{
        ClassificationForestConfig, ClassificationTree, OutlierForestConfig, OutlierTree,
    },
    tree::tree::Tree,
    utils::structures::Sample,
};
use dtw_rs::{Algorithm, DynamicTimeWarping};
use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng, Rng};

const MIN_INTERVAL_LENGTH: usize = 10;

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
    pub fn get_clusters(
        &mut self,
        distances: &Vec<Vec<f64>>,
        targets: &Vec<isize>,
    ) -> Vec<Vec<usize>> {
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
    fn path_length<T: Tree<SplitParameters = Self>>(tree: &T, x: &Sample<'_>) -> f64 {
        let leaf = tree.predict_leaf(&x);
        let samples = leaf.get_samples() as f64;
        if samples > 1.0 {
            return leaf.get_depth() as f64
                + (2.0 * (f64::ln(samples - 1.0) + EULER_MASCHERONI)
                    - 2.0 * (samples - 1.0) / samples);
        } else {
            return leaf.get_depth() as f64;
        }
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
        // Update COnditions
    }
    fn post_split_conditions(&self, new_impurity: f64, _old_impurity: f64) -> bool {
        // Check if there is a non empty split
        if new_impurity == std::f64::INFINITY {
            return true;
        }
        return false;
        // Update COnditions
    }
    fn get_split(&self, samples: &[Sample<'_>]) -> (Self::SplitParameters, f64) {
        // Generate a random interval
        let n_features = samples[0].data.len();
        let targets = samples.iter().map(|s| s.target).collect::<Vec<_>>();
        let classes = unique(&targets);
        let class_counts = value_counts(&targets);
        let mut best_fisher = 0.0;
        let mut best_ts = Vec::new();
        let mut best_start = 0;
        let mut best_end = 0;
        for _ in 0..(n_features as f64).sqrt() as usize {
            let start = thread_rng().gen_range(0..n_features - MIN_INTERVAL_LENGTH);
            let end = thread_rng().gen_range(start + MIN_INTERVAL_LENGTH..n_features);
            let aggregation = thread_rng().gen_range(0..3);
            let ts;
            if aggregation == 0 {
                ts = samples
                    .iter()
                    .map(|s| mean(&s.data[start..end]))
                    .collect::<Vec<_>>();
            } else if aggregation == 1 {
                ts = samples
                    .iter()
                    .map(|s| stddev(&s.data[start..end]))
                    .collect::<Vec<_>>();
            } else {
                ts = samples
                    .iter()
                    .map(|s| slope(&s.data[start..end]))
                    .collect::<Vec<_>>();
            }

            let fisher = fisher_score(&ts, &targets, &classes, &class_counts);
            if fisher > best_fisher {
                best_ts = samples
                    .iter()
                    .map(|s: &Sample<'_>| &s.data[start..end])
                    .collect::<Vec<_>>();
                best_fisher = fisher;
                best_start = start;
                best_end = end;
            }
        }

        let mut distances = Vec::new();
        for i in 0..best_ts.len() {
            let mut dists = Vec::new();
            //let start_time = std::time::Instant::now();
            for j in 0..best_ts.len() {
                let dist = euclidean(&best_ts[i], &best_ts[j]);
                dists.push(dist);
            }
            //println!("Time: {:?}", start_time.elapsed());
            distances.push(dists);
        }
        //println!("k {}", (samples.len() as f64).log2().log2().ceil());
        for k in (1..=max(2, ((samples.len() as f64).log2().log2().ceil() as usize))).rev() {
            let mut sets = KUnionFind::new(k, best_ts.len());
            let clusters = sets.get_clusters(&distances, &targets);

            // Get two sets of labels from targets
            let mut to_left = Vec::new();
            let mut to_right = Vec::new();
            let mut splits: HashMap<isize, bool> = HashMap::new();
            for set in clusters {
                let mut classes = HashMap::new();
                for i in &set {
                    *classes.entry(samples[*i].target).or_insert(0) += 1;
                }
                let majority = *classes.iter().max_by_key(|(_, v)| **v).unwrap().0;

                let goes_to_right = if let Some(&goes_to_right) = splits.get(&majority) {
                    goes_to_right
                } else {
                    if to_left.len() == 0 && to_right.len() != 0 {
                        false
                    } else if to_right.len() == 0 && to_left.len() != 0 {
                        true
                    } else {
                        let goes_to_right = thread_rng().gen_bool(0.5);
                        splits.insert(majority, goes_to_right);
                        goes_to_right
                    }
                };
                if goes_to_right {
                    to_right.extend(set.iter().map(|x| best_ts[*x].to_vec()));
                } else {
                    to_left.extend(set.iter().map(|x| best_ts[*x].to_vec()));
                }
            }
            if splits.len() < 2 {
                if k == 2 {
                    let t = DistanceSplit {
                        left_candidates: Vec::new(),
                        right_candidates: Vec::new(),
                        interval: (best_start, best_end),
                    };
                    return (t, f64::INFINITY);
                } else {
                    continue;
                }
            }
            let t = DistanceSplit {
                left_candidates: to_left,
                right_candidates: to_right,
                interval: (best_start, best_end),
            };
            return (t, 0.0);
        }
        unreachable!()

        // // Find the minimum inter class distance
        // let mut threshold = std::f64::INFINITY;
        // for i in 0..ts.len() {
        //     for j in 0..ts.len() {
        //         if samples[i].target != samples[j].target {
        //             if distances[i][j] < threshold {
        //                 threshold = distances[i][j];
        //             }
        //         }
        //     }
        // }

        // let mut sets = UnionFind::new(ts.len(), &distances, threshold);
        // let clusters = sets.get_clusters(indexes);
        // let mut splits: HashMap<isize, Vec<usize>> = HashMap::new();
        // for i in clusters.keys() {
        //     if clusters
        //         .get(i)
        //         .unwrap()
        //         .iter()
        //         .all(|x| samples[*x].target == samples[*clusters.get(i).unwrap().first().unwrap()].target)
        //     {
        //         splits
        //             .entry(0)
        //             .or_insert(Vec::<usize>::new())
        //             .extend(clusters.get(i).unwrap().iter().copied());
        //     } else {
        //         splits
        //             .entry(1)
        //             .or_insert(Vec::<usize>::new())
        //             .extend(clusters.get(i).unwrap().iter().copied());
        //     }
        // }
        // if splits.len() < 2 {
        //     splits.clear();
        //     for i in clusters.keys() {
        //         let set = clusters.get(i).unwrap();
        //         let mut class_counts = HashMap::new();
        //         for j in set {
        //             let target = samples[*j].target;
        //             *class_counts.entry(target).or_insert(0) += 1;
        //         }
        //         let majority = *class_counts.iter().max_by_key(|(_, v)| **v).unwrap().0;
        //         splits
        //             .entry(majority)
        //             .or_insert(Vec::<usize>::new())
        //             .extend(set.iter().copied());
        //     }
        //     if splits.len() < 2 {
        //         let t = DistanceSplit {
        //             left_candidates: Vec::new(),
        //             right_candidates: Vec::new(),
        //             interval: (start, end),
        //         };
        //         return (t, f64::INFINITY);
        //     }
        // }
        // let left_candidates = splits
        //     .get(&0)
        //     .unwrap_or_else(|| {
        //         panic!(
        //             "{:?}, \n {:?}",
        //             clusters
        //                 .iter()
        //                 .map(|x| x.1.iter().map(|v| samples[*v].target).collect::<Vec<_>>())
        //                 .collect::<Vec<_>>(),
        //             splits
        //         )
        //     })
        //     .iter()
        //     .map(|x| ts[*x].to_vec())
        //     .collect::<Vec<_>>();
        // let right_candidates = splits
        //     .get(&1)
        //     .unwrap_or_else(|| {
        //         panic!(
        //             "{:?}, \n {:?}",
        //             clusters
        //                 .iter()
        //                 .map(|x| x.1.iter().map(|v| samples[*v].target).collect::<Vec<_>>())
        //                 .collect::<Vec<_>>(),
        //             splits
        //         )
        //     })
        //     .iter()
        //     .map(|x| ts[*x].to_vec())
        //     .collect::<Vec<_>>();

        // let t = DistanceSplit {
        //     left_candidates: left_candidates,
        //     right_candidates: right_candidates,
        //     interval: (start, end),
        // };
        // //panic!("{}",t.split(&samples[splits.get(&1).unwrap()[0]]));
        // (t, 0.0)
    }
}
