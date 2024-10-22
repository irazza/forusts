use super::{
    node::Node,
    tree::StandardSplit,
};
use crate::utils::split::get_best_split;
use crate::{
    forest::random_forest::RandomForestConfig, tree::tree::Tree, utils::structures::Sample,
    RandomGenerator,
};
use hashbrown::HashMap;

#[derive(Clone, Debug)]
pub struct DecisionTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: usize,
    pub criterion: fn(&HashMap<isize, usize>, &[HashMap<isize, usize>]) -> f64,
}

#[derive(Clone, Debug)]
pub struct DecisionTree {
    nodes: Vec<Node<StandardSplit>>,
    config: DecisionTreeConfig,
}
impl Tree for DecisionTree {
    type Config = DecisionTreeConfig;
    type ForestTreeConfig = RandomForestConfig;
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
    ) -> Option<(Vec<std::ops::Range<usize>>, Self::SplitParameters, f64)> {
        get_best_split(
            samples,
            non_constant_features,
            self.config.min_samples_leaf,
            self.config.max_features,
            random_state,
        )
    }

    fn from_config(
        config: &Self::ForestTreeConfig,
        _max_samples: usize,
        n_features: usize,
        random_state: &mut RandomGenerator,
    ) -> Self {
        Self::new(
            DecisionTreeConfig {
                max_depth: config.max_depth.unwrap_or(usize::MAX),
                max_features: config.max_features.get_features(n_features),
                min_samples_split: config.min_samples_split,
                min_samples_leaf: config.min_samples_leaf,
                criterion: config.criterion,
            },
            random_state,
        )
    }
}
