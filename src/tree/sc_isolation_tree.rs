use std::path;

use super::{node::Node, tree::SplitParameters};
use crate::feature_extraction::statistics::stddev;
use crate::{
    forest::forest::{OutlierForestConfig, OutlierTree},
    tree::tree::Tree,
    utils::structures::Sample,
};
use rand::{seq::SliceRandom, thread_rng, Rng};

const N_HYPERPLANES: usize = 10;
const N_ATTRIBUTES: usize = 25;

#[derive(Clone, Debug)]
pub struct SCIsolationTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
}

#[derive(Clone, Debug)]
pub struct SCIsolationTree {
    root: Node<SplitHyperplane>,
    config: SCIsolationTreeConfig,
}
impl SCIsolationTree {
    fn get_random_hyperplane(
        &self,
        samples: &[Sample<'_>],
        n_attributes: usize,
        means: &[f64],
        stddevs: &[f64],
    ) -> SplitHyperplane {
        let idxs_candidates = stddevs
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > f64::EPSILON)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let mut subsampled_features = idxs_candidates;
        subsampled_features.shuffle(&mut thread_rng());
        subsampled_features.truncate(n_attributes);
        let c = (0..subsampled_features.len())
            .into_iter()
            .map(|i| thread_rng().gen_range(-1.0..=1.0) / stddevs[subsampled_features[i]])
            .collect::<Vec<_>>();
        let mut p = samples
            .iter()
            .map(|s| {
                c.iter()
                    .zip(subsampled_features.iter())
                    .map(|(c, i)| c * (s.data[*i] - means[*i]))
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();
        p.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let p_stddev = stddev(&p);
        let mut best_p = 0.0;
        let mut best_sd_gain = 0.0;
        for i in 1..p.len() {
            let p_l = stddev(&p[0..i]);
            let p_r = stddev(&p[i..]);
            let sd_gain = 1.0 - ((p_l + p_r) / 2.0) / p_stddev;
            if sd_gain > best_sd_gain {
                best_sd_gain = sd_gain;
                best_p = p[i];
            }
        }
        SplitHyperplane {
            c: c,
            means: means.to_vec(),
            idx_attributes: subsampled_features,
            p: best_p,
            best_sd_gain: best_sd_gain,
            limit: p[p.len() - 1] - p[0],
        }
    }
}

impl OutlierTree for SCIsolationTree {
    fn from_outlier_config(max_samples: usize, config: &OutlierForestConfig) -> Self {
        Self::new(SCIsolationTreeConfig {
            max_depth: config.max_depth.unwrap_or(max_samples.ilog2() as usize + 1),
            min_samples_split: 2,
            // Setted to 2 to avoid empty child when splitting when there are only two samples
        })
    }
}
#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct SplitHyperplane {
    c: Vec<f64>,
    means: Vec<f64>,
    idx_attributes: Vec<usize>,
    p: f64,
    best_sd_gain: f64,
    limit: f64,
}
impl SplitHyperplane {
    pub fn get_dist(&self, sample: &Sample<'_>) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.c.len() {
            sum += self.c[i]
                * (sample.data[self.idx_attributes[i]] - self.means[self.idx_attributes[i]]);
        }
        sum - self.p
    }
}
impl Eq for SplitHyperplane {}
impl Ord for SplitHyperplane {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.p.partial_cmp(&other.p).unwrap()
    }
}
impl SplitParameters for SplitHyperplane {
    fn split(&self, sample: &Sample<'_>) -> bool {
        self.get_dist(sample) < 0.0
    }
    fn path_length<T: Tree<SplitParameters = Self>>(tree: &T, x: &Sample<'_>) -> f64 {
        let splits = tree.get_splits(x);
        let mut path_length = 0.0;
        for split in splits {
            if split.get_dist(x).abs() < split.limit {
                path_length += 1.0;
            }
        }
        return path_length;
    }
}
impl Tree for SCIsolationTree {
    type Config = SCIsolationTreeConfig;
    type SplitParameters = SplitHyperplane;
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
        // Base case: samples are the same object
        let first_sample = &samples[0];
        for sample in samples {
            for i in 0..sample.data.len() {
                if sample.data[i] != first_sample.data[i] {
                    return false;
                }
            }
        }
        return true;
    }
    fn post_split_conditions(&self, new_impurity: f64, _old_impurity: f64) -> bool {
        return false;
    }
    fn get_split(&self, samples: &[Sample<'_>]) -> (SplitHyperplane, f64) {
        let mut stddev = vec![0.0; samples[0].data.len()];
        let mut means = vec![0.0; samples[0].data.len()];
        for i in 0..samples[0].data.len() {
            let mean = samples.iter().map(|v| v.data[i]).sum::<f64>() / samples.len() as f64;
            let variance = samples
                .iter()
                .map(|v| (v.data[i] - mean).powi(2))
                .sum::<f64>()
                / samples.len() as f64;
            stddev[i] = variance.sqrt();
            means[i] = mean;
        }
        let best_hp = (0..N_HYPERPLANES)
            .into_iter()
            .map(|_| self.get_random_hyperplane(samples, N_ATTRIBUTES, &means, &stddev))
            .max_by(|a, b| a.best_sd_gain.partial_cmp(&b.best_sd_gain).unwrap())
            .unwrap();
        let best_sd_gain = best_hp.best_sd_gain;
        (best_hp, best_sd_gain)
    }
}
