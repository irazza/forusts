use super::{node::Node, tree::StandardSplit};
use crate::{
    forest::{forest::ClassificationTree, random_forest::RandomForestConfig},
    tree::tree::Tree,
    utils::structures::Sample,
    RandomGenerator,
};
use rand::{seq::SliceRandom, Rng};

#[derive(Clone, Debug)]
pub struct DecisionTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
}

#[derive(Clone, Debug)]
pub struct DecisionTree {
    nodes: Vec<Node<StandardSplit>>,
    config: DecisionTreeConfig,
}
impl Tree for DecisionTree {
    type Config = DecisionTreeConfig;
    type ForestConfig = RandomForestConfig;
    type SplitParameters = StandardSplit;
    fn new(config: Self::Config, _random_state: &mut RandomGenerator) -> Self {
        Self {
            nodes: Vec::new(),
            config,
        }
    }
    fn get_split(
        &self,
        samples: &[Sample],
        min_samples_leaf: usize,
        non_constant_features: &mut Vec<usize>,
        random_state: &mut RandomGenerator,
    ) -> Option<(Self::SplitParameters, f64)> {
        non_constant_features.shuffle(random_state);

        while let Some(feature) = non_constant_features.pop() {
            let min_feature = samples
                .iter()
                .map(|f| f.features[feature])
                .fold(f64::INFINITY, f64::min);

            let max_feature = samples
                .iter()
                .map(|f| f.features[feature])
                .fold(f64::NEG_INFINITY, f64::max);

            if max_feature - min_feature <= f64::EPSILON {
                // Remove constant features
                continue;
            } else {
                let threshold = random_state.gen_range(min_feature..max_feature);
                let left_count = samples
                    .iter()
                    .filter(|s| s.features[feature] < threshold)
                    .count();
                let right_count = samples.len() - left_count;

                if left_count < min_samples_leaf || right_count < min_samples_leaf {
                    continue;
                }

                non_constant_features.push(feature);

                return Some((StandardSplit { feature, threshold }, f64::NAN));
            }
        }
        return None;
    }
    
    fn from_config(
        config: &Self::ForestConfig,
        max_samples: usize,
        n_features: usize,
        random_state: &mut RandomGenerator,
    ) -> Self {
        Self::new(
            DecisionTreeConfig {
                max_depth: config
                    .max_depth
                    .unwrap_or(usize::MAX),
                min_samples_split: config.min_samples_split,
                min_samples_leaf: config.min_samples_leaf,
            },
            random_state,
        )
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

    fn transform(&self, data: &[Sample]) -> Vec<Sample> {
        data.to_vec()
    }
}
