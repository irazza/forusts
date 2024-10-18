use super::{node::Node, tree::StandardSplit};
use crate::{
    forest::isolation_forest::IsolationForestConfig, tree::tree::Tree, utils::structures::Sample,
    RandomGenerator,
};
use rand::{seq::SliceRandom, Rng};

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
    fn get_split(
        &self,
        samples: &[Sample],
        min_samples_leaf: usize,
        non_constant_features: &mut Vec<usize>,
        random_state: &mut RandomGenerator,
    ) -> Option<(Self::SplitParameters, f64)> {
        non_constant_features.shuffle(random_state);

        while let Some(feature) = non_constant_features.pop() {
            let thresholds = samples
                .iter()
                .map(|f| f.features[feature])
                .collect::<Vec<_>>();

            let min_feature = *thresholds
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            let max_feature = *thresholds
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            if max_feature - min_feature <= f64::EPSILON {
                // Remove constant features
                continue;
            } else {
                let threshold = random_state.gen_range(min_feature..max_feature);
                let split = StandardSplit { feature, threshold };
                let (min_samples_leaf_split, _) =
                    Self::min_samples_leaf_split(samples, min_samples_leaf, &split);
                if min_samples_leaf_split {
                    continue;
                }

                non_constant_features.push(feature);

                return Some((split, f64::NAN));
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

    fn transform(&self, data: &[Sample]) -> Vec<Sample> {
        data.to_vec()
    }
}
