use std::cmp::min;

use super::{
    node::Node,
    tree::{SplitParameters, SplitTest},
};
use crate::{
    distance::distances::{euclidean, twe},
    feature_extraction::{
        scamp::compute_selfmp,
        statistics::{mean, median, stddev, zscore, EULER_MASCHERONI},
    },
    forest::forest::{OutlierForestConfig, OutlierTree},
    tree::tree::Tree,
    utils::structures::Sample,
};
use rand::{seq::SliceRandom, thread_rng, Rng};

const MIN_INTERVAL_LENGTH: usize = 10;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MPSplit {
    pub threshold: f64,
    pub candidates: Vec<Vec<f64>>,
    pub interval: (usize, usize),
}

impl Ord for MPSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl Eq for MPSplit {}

impl SplitParameters for MPSplit {
    fn split(&self, sample: &Sample) -> bool {
        let mut min = std::f64::INFINITY;
        let sample = zscore(&sample.data[self.interval.0..self.interval.1]);
        for candidate in &self.candidates {
            let dist = euclidean(&sample, &candidate);
            if dist < min && dist != 0.0 {
                min = dist;
            }
        }
        min < self.threshold
    }
    fn path_length<T: Tree<SplitParameters = Self>>(tree: &T, x: &Sample) -> f64 {
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
pub struct MPTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
}

#[derive(Clone, Debug)]
pub struct MPTree {
    root: Node<MPSplit>,
    config: MPTreeConfig,
}

impl OutlierTree for MPTree {
    fn from_outlier_config(max_samples: usize, config: &OutlierForestConfig) -> Self {
        Self::new(MPTreeConfig {
            max_depth: config.max_depth.unwrap_or(max_samples.ilog2() as usize + 1),
            min_samples_split: 2,
            // Setted to 2 to avoid empty child when splitting when there are only two samples
        })
    }
}

impl Tree for MPTree {
    type Config = MPTreeConfig;
    type SplitParameters = MPSplit;
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
        return false;
    }
    fn post_split_conditions(&self, new_impurity: f64, _old_impurity: f64) -> bool {
        // Check if there is a non empty split
        if new_impurity == std::f64::INFINITY {
            return true;
        }
        return false;
    }
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        // Generate a random interval
        let n_features = samples[0].data.len();
        let start = thread_rng().gen_range(0..n_features - MIN_INTERVAL_LENGTH);
        let end = thread_rng().gen_range(start + MIN_INTERVAL_LENGTH..n_features);

        let ts = samples
            .iter()
            .map(|s| zscore(&s.data[start..end]))
            .collect::<Vec<_>>();
        let mut mp = Vec::new();

        for i in 0..ts.len() {
            let mut min = std::f64::INFINITY;
            for j in 0..ts.len() {
                if i != j {
                    let dist = euclidean(&ts[i], &ts[j]);
                    if dist < min {
                        min = dist;
                    }
                }
            }
            mp.push(min);
        }
        // let index_class_0 = samples.iter().enumerate().filter(|(_, s)| s.target == 0).map(|(i, _)| i).collect::<Vec<_>>();
        // let mean_class_0 = mean(&index_class_0.iter().map(|i| mp[*i]).collect::<Vec<_>>());
        // let std_class_0 = stddev(&index_class_0.iter().map(|i| mp[*i]).collect::<Vec<_>>());
        // let index_class_1 = samples
        //     .iter()
        //     .enumerate()
        //     .filter(|(_, s)| s.target == 1)
        //     .map(|(i, _)| i)
        //     .collect::<Vec<_>>();
        // let mean_class_1 = mean(&index_class_1.iter().map(|i| mp[*i]).collect::<Vec<_>>());
        // let std_class_1 = stddev(&index_class_1.iter().map(|i| mp[*i]).collect::<Vec<_>>());
        // println!("Mean class 0: {}, Std class 0: {}", mean_class_0, std_class_0);
        // println!("Mean class 1: {}, Std class 1: {}", mean_class_1, std_class_1);
        // mp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // mp.dedup();

        if mp.len() > 2 {
            //println!("MP Mean: {}, MP Median: {}, MP Std: {}", mean(&mp), median(&mp), stddev(&mp));
            let split = MPSplit {
                threshold: mean(&mp),
                candidates: ts,
                interval: (start, end),
            };
            (split, 0.0)
        } else {
            let split = MPSplit {
                threshold: 0.0,
                candidates: ts,
                interval: (start, end),
            };
            (split, f64::INFINITY)
        }
    }
}
