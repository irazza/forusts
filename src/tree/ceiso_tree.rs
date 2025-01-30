use super::ei_tree::EIsoSplit;
use super::node::Node;
use crate::forest::eiso_forest::ExtensionLevel;
use crate::tree::transform::catch_transform;
use crate::utils::split::get_extended_split;
use crate::{
    forest::ceiso_forest::CEIsoForestConfig, tree::tree::Tree, utils::structures::Sample,
    RandomGenerator,
};
use catch22::N_CATCH22;
use rand::{seq::SliceRandom, Rng};
use std::cmp::max;

const MIN_INTERVAL_PERC: f64 = 0.1;
const MIN_INTERVALS_LEN: usize = 3;

#[derive(Clone, Debug)]
pub struct CEIsoTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub n_features: usize,
    pub n_intervals: usize,
    pub n_attributes: usize,
    pub extended_level: ExtensionLevel,
}

#[derive(Clone, Debug)]
pub struct CEIsoTree {
    nodes: Vec<Node<EIsoSplit>>,
    config: CEIsoTreeConfig,
    intervals: Vec<(usize, usize)>,
    attributes: Vec<usize>,
}
impl Tree for CEIsoTree {
    type Config = CEIsoTreeConfig;
    type ForestTreeConfig = CEIsoForestConfig;
    type SplitParameters = EIsoSplit;
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
                extended_level: config.extension_level,
            },
            random_state,
        )
    }
}
