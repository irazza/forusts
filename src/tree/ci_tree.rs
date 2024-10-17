use std::{sync::Arc, usize};

use super::{node::Node, tree::StandardSplit};
use crate::{
    forest::{ci_forest::CIForestConfig, forest::CACHE},
    tree::tree::Tree,
    utils::structures::Sample,
    RandomGenerator,
};
use catch22::{compute, N_CATCH22};
use rand::{seq::SliceRandom, Rng};
const MIN_INTERVAL_PERC: f64 = 0.1;

#[derive(Clone, Debug)]
pub struct CITreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
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
    fn from_config(
        config: &Self::ForestTreeConfig,
        max_samples: usize,
        n_features: usize,
        random_state: &mut RandomGenerator,
    ) -> Self {
        Self::new(
            CITreeConfig {
                max_depth: config
                    .outlier_config
                    .max_depth
                    .unwrap_or((max_samples as f64).max(2.0).log2().ceil() as usize + 1),
                min_samples_split: config.outlier_config.min_samples_split,
                min_samples_leaf: config.outlier_config.min_samples_leaf,
                n_intervals: config.n_intervals.get_interval(n_features),
                n_attributes: config.n_attributes,
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

            let min_feature = *thresholds.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

            let max_feature = *thresholds.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

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
}
