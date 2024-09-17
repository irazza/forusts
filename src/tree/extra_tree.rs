use super::{node::Node, tree::{SplitParameters, StandardSplit}};
use crate::{
    feature_extraction::catch22::compute_catch,
    forest::{
        extra_forest::ExtraForest,
        forest::ClassificationTree,
    },
    tree::tree::Tree,
    utils::structures::Sample,
};
use core::panic;
use std::cmp::max;
use dashmap::DashMap;
use lazy_static::lazy_static;
use rand::{seq::SliceRandom, thread_rng, Rng};

#[derive(Clone, Debug, Copy)]
pub struct ExtraTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
}

#[derive(Clone, Debug)]
pub struct ExtraTree {
    root: Node<StandardSplit>,
    config: ExtraTreeConfig,
}

impl ClassificationTree for ExtraTree {
    type TreeConfig = ExtraForest;
    fn from_classification_config(config: &Self::TreeConfig) -> Self {
        Self::new(ExtraTreeConfig {
            max_depth: config.classification_config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: config.classification_config.min_samples_split,
        })
    }
}

impl Tree for ExtraTree {
    type Config = ExtraTreeConfig;
    type SplitParameters = StandardSplit;
    fn new(config: Self::Config) -> Self {
        Self {
            root: Node::new(),
            config,
        }
    }
    fn get_max_depth(&self) -> usize {
        self.config.max_depth
    }
    fn get_root(&self) -> &Node<Self::SplitParameters> {
        &self.root
    }
    fn set_root(&mut self, root: Node<Self::SplitParameters>) {
        self.root = root;
    }
    fn pre_split_conditions(&self, samples: &[Sample], current_depth: usize) -> bool {
        // Base case: not enough samples or max depth reached
        if samples.len() <= self.config.min_samples_split || current_depth == self.config.max_depth
        {
            return true;
        }
        // Base case: samples are the same object
        let first_sample = &samples[0].features;
        let is_all_same_data = samples.iter().all(|v| &v.features == first_sample);
        if is_all_same_data {
            return true;
        }
        return false;
    }
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        todo!()
    }
}