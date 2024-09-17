use std::hash::RandomState;

use super::{node::Node, tree::StandardSplit};
use crate::{
    forest::{forest::OutlierTree, isolation_forest::IsolationForestConfig},
    tree::tree::Tree,
    utils::{float_handling::next_up, structures::Sample},
};
use hashbrown::HashSet;
use rand::{seq::SliceRandom, thread_rng, Rng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct IsolationTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
}

#[derive(Clone, Debug)]
pub struct IsolationTree {
    nodes: Vec<Node<StandardSplit>>,
    config: IsolationTreeConfig,
}

impl OutlierTree for IsolationTree {
    type TreeConfig = IsolationForestConfig;
    fn from_outlier_config(config: &Self::TreeConfig, max_samples: usize) -> Self {
        Self::new(IsolationTreeConfig {
            max_depth: (max_samples as f64).max(2.0).log2().ceil() as usize + 1,
            min_samples_split: config.min_samples_split,
            min_samples_leaf: config.min_samples_leaf,
        })
    }
}

impl Tree for IsolationTree {
    type Config = IsolationTreeConfig;
    type SplitParameters = StandardSplit;
    fn new(config: Self::Config) -> Self {
        Self {
            nodes: Vec::new(),
            config,
        }
    }

    fn get_split(&self, samples: &[Sample], non_constant_features: &mut Vec<usize>, random_state: ChaCha8Rng) -> Option<(Self::SplitParameters, f64)> {
        let mut rng = random_state;
        non_constant_features.shuffle(&mut rng);

        while let Some(feature) = non_constant_features.pop() {

            let min_feature = samples
                .iter()
                .map(|f| f.features[feature])
                .fold(f64::INFINITY, f64::min);

            let max_feature = samples
                .iter()
                .map(|f| f.features[feature])
                .fold(f64::NEG_INFINITY, f64::max);

            if min_feature >= max_feature {
                // Remove constant features
                continue;
            } else {
                let threshold = rng.gen_range(min_feature..max_feature);
                non_constant_features.push(feature);
                return Some((
                    StandardSplit {
                        feature,
                        threshold,
                    },
                    f64::NAN,
                ));
            }
        }
        return None;
    }

    fn get_max_depth(&self) -> usize {
        self.config.max_depth
    }
    fn get_root(&self) -> &Node<Self::SplitParameters> {
        &self.nodes[0]
    }

    fn get_min_samples_split(&self) -> usize {
        self.config.min_samples_split
    }

    fn get_min_samples_leaf(&self) -> usize {
        self.config.min_samples_leaf
    }

    fn set_nodes(&mut self, nodes: Vec<Node<Self::SplitParameters>>) {
        self.nodes = nodes;
    }

    fn get_node_at(&self, id: usize) -> &Node<Self::SplitParameters> {
        &self.nodes[id]
    }
}
