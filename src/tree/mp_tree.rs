use super::{
    node::Node,
    tree::{SplitParameters, SplitTest},
};
use crate::{
    distance::distances::twe, feature_extraction::{scamp::compute_scamp, statistics::EULER_MASCHERONI}, forest::forest::{OutlierForestConfig, OutlierTree}, tree::tree::Tree, utils::structures::Sample
};
use rand::{seq::SliceRandom, thread_rng, Rng};

const MIN_INTERVAL_LENGHT: usize = 10;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MPSplit {
    pub left_candidate: Vec<f64>,
    pub right_candidate: Vec<f64>,
    pub interval: (usize, usize),
}

impl Ord for MPSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl Eq for MPSplit {}

impl SplitParameters for MPSplit {
    fn split(&self, sample: &Sample<'_>) -> bool {
        twe(&sample.data[self.interval.0..self.interval.1], &self.left_candidate, 0.1, 0.2) - twe(&sample.data[self.interval.0..self.interval.1], &self.right_candidate, 0.1, 0.2) > 0.0
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
    fn pre_split_conditions(&self, samples: &[Sample<'_>], current_depth: usize) -> bool {
        // Base case: not enough samples or max depth reached
        if samples.len() <= self.config.min_samples_split || current_depth == self.config.max_depth
        {
            return true;
        }
        return false;
    }
    fn post_split_conditions(&self, new_impurity: f64, _old_impurity: f64) -> bool {
        return false;
    }
    fn get_split(&self, samples: &[Sample<'_>]) -> (Self::SplitParameters, f64) {
        // Generate a random interval
        let n_features = samples[0].data.len();
        let start = thread_rng().gen_range(0..n_features - MIN_INTERVAL_LENGHT);
        let end = thread_rng().gen_range(start + MIN_INTERVAL_LENGHT..n_features);

        let timeseries = samples.iter().flat_map(|s| s.data[start..end].to_vec()).collect::<Vec<_>>();

        let (mp, _) = compute_scamp(&timeseries, end-start);
        
        // Get arg max and min of mp
        let (max_idx, _) = mp.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
        let (min_idx, _) = mp.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();

        let left_candidate = timeseries[min_idx..min_idx+end-start].to_vec();
        let right_candidate = timeseries[max_idx..max_idx+end-start].to_vec();

        (MPSplit {
            left_candidate,
            right_candidate,
            interval: (start, end),
        }, 0.0)
    }
}
