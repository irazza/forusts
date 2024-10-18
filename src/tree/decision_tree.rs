use super::{node::Node, tree::StandardSplit};
use crate::{
    forest::random_forest::RandomForestConfig, tree::tree::Tree, utils::structures::Sample,
    RandomGenerator,
};
use hashbrown::HashMap;
use rand::seq::SliceRandom;

#[derive(Clone, Debug)]
pub struct DecisionTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: usize,
    pub criterion: fn(&HashMap<isize, usize>, Vec<&HashMap<isize, usize>>) -> f64,
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
    fn from_config(
        config: &Self::ForestTreeConfig,
        _max_samples: usize,
        n_features: usize,
        random_state: &mut RandomGenerator,
    ) -> Self {
        Self::new(
            DecisionTreeConfig {
                max_depth: config.max_depth.unwrap_or(usize::MAX),
                max_features: n_features,
                min_samples_split: config.min_samples_split,
                min_samples_leaf: config.min_samples_leaf,
                criterion: config.criterion,
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
        let mut current_feature_count = 0;
        let mut max_gain = f64::NEG_INFINITY;
        let mut best_split = None;

        non_constant_features.shuffle(random_state);
        non_constant_features.retain(|&feature| {
            if current_feature_count >= self.config.max_features {
                return true;
            }

            let mut thresholds = samples
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
                return false;
            }

            thresholds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            thresholds.dedup();
            
            for &threshold in thresholds[1..].iter() {
                let current_split = StandardSplit { feature, threshold };

                let (min_samples_leaf_split, splitted_data) =
                    Self::min_samples_leaf_split(samples, min_samples_leaf, &current_split);
                if min_samples_leaf_split {
                    continue;
                }

                let mut parent_count = HashMap::new();
                let mut splitted_vec = vec![HashMap::new(); splitted_data.len()];
                for (i, split) in splitted_data.into_iter().enumerate() {
                    for sample in samples[split].iter() {
                        *splitted_vec[i].entry(sample.target).or_insert(0) += 1;
                        *parent_count.entry(sample.target).or_insert(0) += 1;
                    }
                }

                let current_gain = (self.config.criterion)(
                    &parent_count,
                    splitted_vec.iter().map(|x| x).collect(),
                );
                if current_gain > max_gain {
                    max_gain = current_gain;
                    best_split = Some((current_split, max_gain));
                }
            }
            current_feature_count += 1;

            return true;
        });

        return best_split;
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
