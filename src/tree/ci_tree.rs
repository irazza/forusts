use super::{node::Node, tree::StandardSplit};
use crate::forest::forest::CACHE;
use crate::utils::split::get_best_split;
use crate::{
    forest::ci_forest::CIForestConfig, tree::tree::Tree, utils::structures::Sample, RandomGenerator,
};
use catch22::{compute, N_CATCH22};
use hashbrown::HashMap;
use rand::seq::SliceRandom;
use rand::Rng;
use std::sync::Arc;

const MIN_INTERVAL_PERC: f64 = 0.1;

#[derive(Clone, Debug)]
pub struct CITreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: usize,
    pub criterion: fn(&HashMap<isize, usize>, &[HashMap<isize, usize>]) -> f64,
    pub n_intervals: usize,
    pub n_attributes: usize,
}

#[derive(Clone, Debug)]
pub struct CITree {
    nodes: Vec<Node<StandardSplit>>,
    config: CITreeConfig,
    intervals: Vec<(f64, f64)>,
    attributes: Vec<usize>,
}
impl Tree for CITree {
    type Config = CITreeConfig;
    type ForestTreeConfig = CIForestConfig;
    type SplitParameters = StandardSplit;
    fn new(config: Self::Config, mut random_state: &mut RandomGenerator) -> Self {
        Self {
            nodes: Vec::new(),
            config: config.clone(),
            intervals: {
                let mut intervals = Vec::with_capacity(config.n_intervals);
                for _ in 0..config.n_intervals {
                    let start = random_state.gen_range(0.0..=1.0 - MIN_INTERVAL_PERC);
                    let end = random_state.gen_range(start + MIN_INTERVAL_PERC..=1.0);
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
        let mut transformed = Vec::with_capacity(data.len());
        for sample in data {
            let mut features = Vec::with_capacity(self.intervals.len() * self.attributes.len());
            for (start, end) in &self.intervals {
                let start = (start * sample.features.len() as f64).round() as usize;
                let end = (end * sample.features.len() as f64).round() as usize;
                for attribute in &self.attributes {
                    if let Some(value) = CACHE.get(&(sample.clone(), start, end, *attribute)) {
                        features.push(*value);
                    } else {
                        let value = compute(&sample.features[start..end], *attribute);
                        CACHE.insert((sample.clone(), start, end, *attribute), value);
                        features.push(value);
                    }
                }
            }
            transformed.push(Sample {
                features: Arc::new(features),
                target: sample.target,
            });
        }
        transformed
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
            CITreeConfig {
                max_depth: config.classification_config.max_depth.unwrap_or(usize::MAX),
                max_features: config.classification_config.max_features.get_features(
                    config.n_intervals.get_interval(n_features) * config.n_attributes,
                ),
                min_samples_split: config.classification_config.min_samples_split,
                min_samples_leaf: config.classification_config.min_samples_leaf,
                criterion: config.classification_config.criterion,
                n_intervals: config.n_intervals.get_interval(n_features),
                n_attributes: config.n_attributes,
            },
            random_state,
        )
    }
}
