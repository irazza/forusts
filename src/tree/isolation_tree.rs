use core::panic;

use super::{node::Node, tree::SplitTest};
use crate::{
    forest::forest::{OutlierForestConfig, OutlierTree},
    tree::tree::Tree,
    utils::structures::Sample,
};
use rand::{seq::SliceRandom, thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct IsolationTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
}

#[derive(Clone, Debug)]
pub struct IsolationTree {
    root: Node<SplitTest>,
    config: IsolationTreeConfig,
}

impl OutlierTree for IsolationTree {
    fn from_outlier_config(max_samples: usize, config: &OutlierForestConfig) -> Self {
        Self::new(IsolationTreeConfig {
            max_depth: config.max_depth.unwrap_or((max_samples as f64).log2().ceil() as usize),
            min_samples_split: 1,
            // Setted to 2 to avoid empty child when splitting when there are only two samples
        })
    }
}

impl Tree for IsolationTree {
    type Config = IsolationTreeConfig;
    type SplitParameters = SplitTest;
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

        return false;
    }
    fn post_split_conditions(&self, new_impurity: f64, _old_impurity: f64) -> bool {
        // Base case: no split found
        return false;
    }
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        //let mut rng = ChaCha8Rng::seed_from_u64(samples.len() as u64);
        let mut rng = thread_rng();
        let mut thresholds = Vec::new();
        let mut candidate = None;

        // Iterate over feature to find a features which can be splitted
        let mut features = (0..samples[0].data.len()).collect::<Vec<_>>();
        features.shuffle(&mut rng);

        for feature_idx in features {

            // Choose a random threshold
            thresholds = samples.iter().map(|f| f.data[feature_idx]).collect::<Vec<f64>>();
            thresholds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            thresholds.dedup();

            // If there is only one threshold, choose another feature
            if thresholds.len() > 1 {
                candidate = Some(feature_idx);
                break;
            }
        }

        let threshold = thresholds[rng.gen_range(1..thresholds.len())];

        (
            SplitTest {
                feature: candidate.unwrap(),
                threshold,
            },
            rng.gen_range(f64::EPSILON..1.0),
        )
    }
}
