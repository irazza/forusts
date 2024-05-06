use core::panic;

use super::{node::Node, tree::StandardSplit};
use crate::{
    forest::{forest::OutlierTree, isolation_forest::IsolationForestConfig},
    tree::tree::Tree,
    utils::structures::Sample,
};
use rand::{thread_rng, Rng};

#[derive(Clone, Debug)]
pub struct IsolationTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
}

#[derive(Clone, Debug)]
pub struct IsolationTree {
    root: Node<StandardSplit>,
    config: IsolationTreeConfig,
}

impl OutlierTree for IsolationTree {
    type TreeConfig = IsolationForestConfig;
    fn from_outlier_config(config: &Self::TreeConfig) -> Self {
        Self::new(IsolationTreeConfig {
            max_depth: config
                .max_depth
                .unwrap_or((config.max_samples as f64).log2() as usize + 1),
            min_samples_split: 2,
        })
    }
}

impl Tree for IsolationTree {
    type Config = IsolationTreeConfig;
    type SplitParameters = StandardSplit;
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
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        let mut rng = thread_rng();

        let feature = rng.gen_range(0..samples[0].data.len());
        let mut thresholds = samples
            .iter()
            .map(|f| f.data[feature])
            .collect::<Vec<f64>>();
        thresholds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        thresholds.dedup();

        let threshold = match thresholds.len() {
            0 => panic!("Thresholds cannot be empty"),
            1 => thresholds[0],
            _ => thresholds[rng.gen_range(1..thresholds.len())],
        };

        (
            StandardSplit {
                feature: feature,
                threshold,
            },
            rng.gen_range(f64::EPSILON..1.0),
        )
    }
}
