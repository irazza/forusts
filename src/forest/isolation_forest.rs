use crate::{forest::forest::OutlierForest, tree::extra_tree::ExtraTree};

pub struct IsolationForest {
    trees: Vec<ExtraTree>,
    n_trees: usize,
    max_samples: usize,
    enhanced_anomaly_score: Option<bool>,
    max_depth: Option<usize>,
}

impl IsolationForest {
    pub fn new(
        n_trees: usize,
        max_depth: Option<usize>,
        enhanced_anomaly_score: Option<bool>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_depth,
            enhanced_anomaly_score,
            max_samples: 256,
        }
    }
}

impl OutlierForest for IsolationForest {
    fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples;
    }
    fn get_max_samples(&self) -> usize {
        self.max_samples
    }
    fn get_trees_mut(&mut self) -> &mut Vec<ExtraTree> {
        &mut self.trees
    }
    fn get_trees(&self) -> &Vec<ExtraTree> {
        &self.trees
    }
    fn get_n_trees(&self) -> usize {
        self.n_trees
    }
    fn get_max_depth(&self) -> Option<usize> {
        self.max_depth
    }
    fn get_enhanced_anomaly_score(&self) -> Option<bool> {
        self.enhanced_anomaly_score
    }
    fn transform(&self, x: &Vec<Vec<f64>>, _intervals_index: usize) -> Vec<Vec<f64>> {
        x.clone()
    }
    fn compute_intervals(&mut self, _n_features: usize) {}
}
