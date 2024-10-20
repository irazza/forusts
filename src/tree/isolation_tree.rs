use std::ops::Range;

use super::{
    node::Node,
    tree::StandardSplit,
    utils::get_random_split,
};
use crate::{
    forest::isolation_forest::IsolationForestConfig, tree::tree::Tree, utils::structures::Sample,
    RandomGenerator,
};

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

impl Tree for IsolationTree {
    type Config = IsolationTreeConfig;
    type ForestTreeConfig = IsolationForestConfig;
    type SplitParameters = StandardSplit;
    fn new(config: Self::Config, _random_state: &mut RandomGenerator) -> Self {
        Self {
            nodes: Vec::new(),
            config,
        }
    }
    fn transform(&self, data: &[Sample]) -> Vec<Sample> {
        data.to_vec()
    }
    fn get_max_depth(&self) -> usize {
        self.config.max_depth
    }

    fn get_min_samples_split(&self) -> usize {
        self.config.min_samples_split
    }
    fn get_min_samples_leaf(&self) -> usize {
        self.config.min_samples_leaf
    }

    fn get_root(&self) -> &Node<Self::SplitParameters> {
        &self.nodes[0]
    }

    fn set_nodes(&mut self, nodes: Vec<Node<Self::SplitParameters>>) {
        self.nodes = nodes;
    }

    fn get_node_at(&self, id: usize) -> &Node<Self::SplitParameters> {
        &self.nodes[id]
    }

    fn get_split(
        &self,
        samples: &mut [Sample],
        non_constant_features: &mut Vec<usize>,
        random_state: &mut RandomGenerator,
    ) -> Option<(Vec<Range<usize>>, Self::SplitParameters, f64)> {
        get_random_split(
            samples,
            non_constant_features,
            random_state,
            self.config.min_samples_leaf,
        )
    }

    fn from_config(
        config: &Self::ForestTreeConfig,
        max_samples: usize,
        _n_features: usize,
        random_state: &mut RandomGenerator,
    ) -> Self {
        Self::new(
            IsolationTreeConfig {
                max_depth: config
                    .max_depth
                    .unwrap_or((max_samples as f64).max(2.0).log2().ceil() as usize + 1),
                min_samples_split: config.min_samples_split,
                min_samples_leaf: config.min_samples_leaf,
            },
            random_state,
        )
    }
}
