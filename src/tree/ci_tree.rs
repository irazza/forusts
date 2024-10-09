use std::{sync::Arc, usize};

use super::{node::Node, tree::StandardSplit};
use crate::{
    forest::{ci_forest::CIForestConfig, forest::OutlierTree},
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
    // cache: Arc<DashMap<(Sample, usize, usize, usize), f64>>,
}

impl OutlierTree for CITree {
    type TreeConfig = CIForestConfig;
    fn from_outlier_config(
        config: &Self::TreeConfig,
        max_samples: usize,
        n_features: usize,
        random_state: &mut RandomGenerator,
    ) -> Self {
        Self::new(
            CITreeConfig {
                max_depth: (max_samples as f64).max(2.0).log2().ceil() as usize + 1,
                min_samples_split: config.outlier_config.min_samples_split,
                min_samples_leaf: config.outlier_config.min_samples_leaf,
                n_intervals: config.n_intervals.get_interval(n_features),
                n_attributes: config.n_attributes,
            },
            random_state,
        )
    }
}

impl Tree for CITree {
    type Config = CITreeConfig;
    type SplitParameters = StandardSplit;
    fn new(config: Self::Config, mut random_state: &mut RandomGenerator) -> Self {
        Self {
            nodes: Vec::new(),
            config: config.clone(),
            intervals: {
                let mut intervals = Vec::with_capacity(config.n_intervals + 1);
                intervals.push((0.0, 1.0));
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
                    let value = compute(&sample.features[start..end], *attribute);
                    features.push(value);
                    // if CIFOREST_CACHE.contains_key(&(sample.clone(), start, end, *attribute)) {
                    //     features.push(
                    //         *CIFOREST_CACHE
                    //             .get(&(sample.clone(), start, end, *attribute))
                    //             .unwrap(),
                    //     );
                    // } else {
                    //     let value = compute(&sample.features[start..end], *attribute);
                    //     CIFOREST_CACHE.insert((sample.clone(), start, end, *attribute), value);
                    //     features.push(value);
                    // }
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
