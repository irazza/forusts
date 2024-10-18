use super::{
    node::Node,
    tree::{SplitParameters, StandardSplit},
};
use crate::{
    forest::random_forest::RandomForestConfig, tree::tree::Tree, utils::structures::Sample,
    RandomGenerator,
};
use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng, Rng};

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
        samples: &mut [Sample],
        non_constant_features: &mut Vec<usize>,
        random_state: &mut RandomGenerator,
    ) -> Option<(Vec<std::ops::Range<usize>>, Self::SplitParameters, f64)> {
        let mut current_feature_count = 0;
        let mut max_gain = f64::NEG_INFINITY;
        let mut best_split = None;
        let mut best_split_index = 0;

        let mut parent_count = HashMap::new();
        for sample in samples.iter() {
            *parent_count.entry(sample.target).or_insert(0) += 1;
        }

        non_constant_features.shuffle(random_state);
        non_constant_features.retain(|&feature| {
            if current_feature_count >= self.config.max_features {
                return true;
            }

            let min_feature = samples
                .iter()
                .min_by(|a, b| {
                    a.features[feature]
                        .partial_cmp(&b.features[feature])
                        .unwrap()
                })
                .unwrap()
                .features[feature];

            let max_feature = samples
                .iter()
                .max_by(|a, b| {
                    a.features[feature]
                        .partial_cmp(&b.features[feature])
                        .unwrap()
                })
                .unwrap()
                .features[feature];

            if max_feature - min_feature <= f64::EPSILON {
                // Remove constant features
                return false;
            }

            samples.sort_unstable_by(|a, b| {
                a.features[feature]
                    .partial_cmp(&b.features[feature])
                    .unwrap()
            });

            let mut thresholds = samples
                .iter()
                .map(|f| f.features[feature])
                .collect::<Vec<_>>();

            thresholds.dedup();

            let mut split_index = 0;

            let mut splitted_vec = vec![HashMap::new(); 2];
            splitted_vec[1] = parent_count.clone();

            for &threshold in thresholds[1..].iter() {
                let current_split = StandardSplit { feature, threshold };

                while split_index < samples.len() && current_split.split(&samples[split_index]) == 1
                {
                    split_index += 1;
                    *splitted_vec[1]
                        .get_mut(&samples[split_index].target)
                        .unwrap() -= 1;
                    *splitted_vec[0]
                        .entry(samples[split_index].target)
                        .or_insert(0) += 1;
                }

                // 0..split_index => class 0
                // split_index..samples.len() => class 1

                if split_index < self.config.min_samples_leaf
                    || (samples.len() - split_index) < self.config.min_samples_leaf
                {
                    continue;
                }

                let current_gain = random_state.gen_range(0.0..1.0); // (self.config.criterion)(&parent_count, &splitted_vec);
                if current_gain > max_gain {
                    max_gain = current_gain;
                    best_split = Some((current_split, max_gain));
                    best_split_index = split_index;
                }
            }
            current_feature_count = self.config.max_features;
            current_feature_count += 1;

            return true;
        });

        let best_split = best_split?;

        samples.sort_unstable_by(|a, b| {
            a.features[best_split.0.feature]
                .partial_cmp(&b.features[best_split.0.feature])
                .unwrap()
        });

        Some((
            vec![0..best_split_index, best_split_index..samples.len()],
            best_split.0,
            best_split.1,
        ))
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
