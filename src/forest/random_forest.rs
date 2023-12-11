use crate::tree::{
    decision_tree::DecisionTree,
    tree::{Criterion, MaxFeatures},
};

use crate::forest::forest::ClassificationForest;

pub struct RandomForest {
    trees: Vec<DecisionTree>,
    criterion: Criterion,
    min_samples_split: usize,
    n_trees: usize,
    max_features: MaxFeatures,
    max_depth: Option<usize>,
    max_samples: usize,
}
impl RandomForest {
    pub fn new(
        n_trees: usize,
        criterion: Criterion,
        min_samples_split: usize,
        max_features: MaxFeatures,
        max_depth: Option<usize>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            criterion,
            min_samples_split,
            max_features,
            max_depth,
            max_samples: 0,
        }
    }
}
impl ClassificationForest for RandomForest {
    fn get_trees_mut(&mut self) -> &mut Vec<DecisionTree> {
        &mut self.trees
    }
    fn get_trees(&self) -> &Vec<DecisionTree> {
        &self.trees
    }
    fn get_n_trees(&self) -> usize {
        self.n_trees
    }
    fn get_criterion(&self) -> Criterion {
        self.criterion
    }
    fn get_max_features(&self) -> MaxFeatures {
        self.max_features
    }
    fn get_max_depth(&self) -> Option<usize> {
        self.max_depth
    }
    fn get_min_samples_split(&self) -> usize {
        self.min_samples_split
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn transform(&self, x: &[Vec<f64>], _intervals_index: usize) -> Vec<Vec<f64>> {
        x.to_vec()
    }
    fn compute_intervals(&mut self, _n_features: usize) {}
}
