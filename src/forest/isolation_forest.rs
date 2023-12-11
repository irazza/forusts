use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::{isolation_tree::IsolationTree, tree::Tree},
    utils::structures::Sample,
};

pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    n_trees: usize,
    max_samples: usize,
    enhanced_anomaly_score: bool,
    max_depth: Option<usize>,
}

impl IsolationForest {
    pub fn new(n_trees: usize, enhanced_anomaly_score: bool, max_depth: Option<usize>) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_depth,
            enhanced_anomaly_score,
            max_samples: 256,
        }
    }
}
impl Forest<IsolationTree> for IsolationForest {
    fn compute_intervals(&mut self, _n_features: usize) {}
    fn get_max_depth(&self) -> Option<usize> {
        self.max_depth
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
    fn get_n_trees(&self) -> usize {
        self.n_trees
    }
    fn get_trees(&self) -> &Vec<IsolationTree> {
        &self.trees
    }
    fn get_trees_mut(&mut self) -> &mut Vec<IsolationTree> {
        &mut self.trees
    }
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn transform<'a>(&self, data: &[Sample<'a>], _intervals_index: usize) -> Vec<Sample<'a>> {
        data.to_vec()
    }
}
impl OutlierForest for IsolationForest {
    fn get_enhanced_anomaly_score(&self) -> bool {
        self.enhanced_anomaly_score
    }
}
