use super::{node::Node, tree::SplitParameters};
use crate::forest::eiso_forest::ExtensionLevel;
use crate::utils::split::get_extended_split;
use crate::{
    forest::eiso_forest::EIsoForestConfig, tree::tree::Tree, utils::structures::Sample,
    RandomGenerator,
};
use rand::Rng;
use rand_distr::StandardNormal;
#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct EIsoSplit {
    offset: Vec<f64>,
    weights: Vec<f64>,
}

impl EIsoSplit {
    pub fn from_features(
        features_idx: &[usize],
        min_values: &[f64],
        max_values: &[f64],
        samples: &[Sample],
        random_state: &mut RandomGenerator,
    ) -> Self {
        let mut offset = vec![0.0; samples[0].features.len()];
        let mut weights = vec![0.0f64; samples[0].features.len()];

        let mut vector_len: f64 = 0.0;
        for (&feature_idx, (&min_value, &max_value)) in features_idx
            .iter()
            .zip(min_values.iter().zip(max_values.iter()))
        {
            offset[feature_idx] = random_state.gen_range(min_value..=max_value);
            weights[feature_idx] = random_state.sample(StandardNormal);
            vector_len += weights[feature_idx].powi(2);
        }
        vector_len = vector_len.sqrt();
        weights.iter_mut().for_each(|x| *x /= vector_len);

        EIsoSplit { offset, weights }
    }
}

impl Eq for EIsoSplit {}
impl Ord for EIsoSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl SplitParameters for EIsoSplit {
    fn split(&self, sample: &Sample) -> usize {
        let mut sum = 0.0;
        for ((&feature, &offset), &weight) in sample
            .features
            .iter()
            .zip(self.offset.iter())
            .zip(self.weights.iter())
        {
            sum += (feature - offset) * weight;
        }
        if sum < 0.0 {
            0
        } else {
            1
        }
    }
    fn path_length<T: Tree<SplitParameters = Self>>(tree: &T, x: &Sample) -> f64 {
        let leaf = tree.predict_leaf(x);
        leaf.get_depth() as f64 + T::average_path_length(leaf.get_n_samples())
    }
}

#[derive(Clone, Debug)]
pub struct EIsoTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub extended_level: ExtensionLevel,
}

#[derive(Clone, Debug)]
pub struct EIsoTree {
    nodes: Vec<Node<EIsoSplit>>,
    config: EIsoTreeConfig,
}
impl Tree for EIsoTree {
    type Config = EIsoTreeConfig;
    type ForestTreeConfig = EIsoForestConfig;
    type SplitParameters = EIsoSplit;
    fn new(config: Self::Config, mut _random_state: &mut RandomGenerator) -> Self {
        Self {
            nodes: Vec::new(),
            config: config.clone(),
        }
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
        get_extended_split(
            samples,
            non_constant_features,
            random_state,
            self.config.min_samples_leaf,
            match self.config.extended_level {
                ExtensionLevel::Percentage(percentage) => {
                    (samples[0].features.len() as f64 * percentage) as usize
                }
                ExtensionLevel::ExtraFeatures(n_features) => n_features,
            },
        )
    }

    fn from_config(
        config: &Self::ForestTreeConfig,
        max_samples: usize,
        _n_features: usize,
        random_state: &mut RandomGenerator,
    ) -> Self {
        Self::new(
            EIsoTreeConfig {
                max_depth: config
                    .outlier_config
                    .max_depth
                    .unwrap_or((max_samples as f64).max(2.0).log2().ceil() as usize + 1),
                min_samples_split: config.outlier_config.min_samples_split,
                min_samples_leaf: config.outlier_config.min_samples_leaf,
                extended_level: config.extension_level,
            },
            random_state,
        )
    }
}
