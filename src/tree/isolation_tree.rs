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
    fn from_outlier_config(config: &Self::TreeConfig, max_samples: usize) -> Self {
        Self::new(IsolationTreeConfig {
            max_depth: (max_samples as f64).max(2.0).log2().ceil() as usize + 1,
            min_samples_split: config.min_samples_split,
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

        let min_feature = samples
            .iter()
            .map(|f| f.data[feature])
            .fold(f64::INFINITY, f64::min);
        let max_feature = samples
            .iter()
            .map(|f| f.data[feature])
            .fold(f64::NEG_INFINITY, f64::max);

        let threshold;
        if f64::abs(max_feature - min_feature) < f64::EPSILON {
            threshold = min_feature;
        } else {
            threshold = rng.gen_range(min_feature + f64::EPSILON..max_feature);
        }

        (
            StandardSplit {
                feature: feature,
                threshold,
            },
            rng.gen_range(f64::EPSILON..1.0),
        )
    }
}
