use super::{node::Node, tree::SplitParameters};
use crate::tree::transform::catch_transform;
use crate::utils::split::{get_extended_split, get_random_split};
use crate::{
    forest::ciso_forest::CIsoForestConfig, tree::tree::Tree, utils::structures::Sample,
    RandomGenerator,
};
use catch22::N_CATCH22;
use rand::{seq::SliceRandom, Rng};
use rand_distr::StandardNormal;
use std::cmp::max;

const MIN_INTERVAL_PERC: f64 = 0.1;
const MIN_INTERVALS_LEN: usize = 3;

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub struct CEIsoSplit {
    offset: Vec<f64>,
    weights: Vec<f64>,
}

impl CEIsoSplit {
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

        CEIsoSplit { offset, weights }
    }
}

impl Eq for CEIsoSplit {}
impl Ord for CEIsoSplit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl SplitParameters for CEIsoSplit {
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
pub struct CEIsoTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub n_features: usize,
    pub n_intervals: usize,
    pub n_attributes: usize,
    pub extended_level: usize,
}

#[derive(Clone, Debug)]
pub struct CIsoTree {
    nodes: Vec<Node<CEIsoSplit>>,
    config: CEIsoTreeConfig,
    intervals: Vec<(usize, usize)>,
    attributes: Vec<usize>,
}
impl Tree for CIsoTree {
    type Config = CEIsoTreeConfig;
    type ForestTreeConfig = CIsoForestConfig;
    type SplitParameters = CEIsoSplit;
    fn new(config: Self::Config, mut random_state: &mut RandomGenerator) -> Self {
        Self {
            nodes: Vec::new(),
            config: config.clone(),
            intervals: {
                let mut intervals = Vec::with_capacity(config.n_intervals);
                let min_interval_len = max(
                    (config.n_features as f64 * MIN_INTERVAL_PERC).ceil() as usize,
                    MIN_INTERVALS_LEN,
                );
                for _ in 0..config.n_intervals {
                    let start = random_state.gen_range(0..=config.n_features - min_interval_len);
                    let end = random_state.gen_range(start + min_interval_len..=config.n_features);
                    intervals.push((start, end));
                }
                intervals
            },
            attributes: {
                let mut attributes = (0..N_CATCH22).collect::<Vec<usize>>();
                attributes.shuffle(&mut random_state);
                attributes.truncate(config.n_attributes);
                attributes
            },
        }
    }
    fn transform(&self, data: &[Sample]) -> Vec<Sample> {
        catch_transform(data, &self.intervals, &self.attributes)
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
            samples[0].features.len() - self.config.extended_level,
        )
    }

    fn from_config(
        config: &Self::ForestTreeConfig,
        max_samples: usize,
        n_features: usize,
        random_state: &mut RandomGenerator,
    ) -> Self {
        Self::new(
            CEIsoTreeConfig {
                max_depth: config
                    .outlier_config
                    .max_depth
                    .unwrap_or((max_samples as f64).max(2.0).log2().ceil() as usize + 1),
                min_samples_split: config.outlier_config.min_samples_split,
                min_samples_leaf: config.outlier_config.min_samples_leaf,
                n_features,
                n_intervals: config.n_intervals.get_interval(n_features),
                n_attributes: config.n_attributes,
                extended_level: todo!(), //config.extended_level,
            },
            random_state,
        )
    }
}
