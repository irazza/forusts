use super::node::Node;
use crate::{
    forest::forest::{OutlierForestConfig, OutlierTree},
    tree::tree::Tree,
    utils::structures::Sample,
};
use rand::{thread_rng, Rng};

#[derive(Clone, Debug)]
pub struct IsolationTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
}

#[derive(Clone, Debug)]
pub struct IsolationTree {
    root: Node,
    config: IsolationTreeConfig,
}

impl OutlierTree for IsolationTree {
    fn from_outlier_config(max_samples: usize, config: &OutlierForestConfig) -> Self {
        Self::new(IsolationTreeConfig {
            max_depth: config.max_depth.unwrap_or(max_samples.ilog2() as usize + 1),
            min_samples_split: 2,
            // Setted to 2 to avoid empty child when splitting when there are only two samples
        })
    }
}

impl Tree for IsolationTree {
    type Config = IsolationTreeConfig;
    fn new(config: Self::Config) -> Self {
        Self {
            root: Node::new(),
            config,
        }
    }
    fn get_max_depth(&self) -> usize {
        self.config.max_depth
    }
    fn get_root(&self) -> &Node {
        &self.root
    }
    fn set_root(&mut self, root: Node) {
        self.root = root;
    }
    fn pre_split_conditions(&self, samples: &[Sample<'_>], current_depth: usize) -> bool {
        // Base case: not enough samples or max depth reached
        if samples.len() <= self.config.min_samples_split || current_depth == self.config.max_depth
        {
            return true;
        }
        // Base case: samples are the same object
        let first_sample = &samples[0];
        let is_all_same_data = samples.iter().all(|v| v == first_sample);
        if is_all_same_data {
            return true;
        }
        return false;
    }
    fn post_split_conditions(&self, _new_impurity: f64, _old_impurity: f64) -> bool {
        return false;
    }
    fn get_split(&self, samples: &[Sample<'_>]) -> (usize, f64, f64) {
        let mut best_feature = thread_rng().gen_range(0..samples[0].data.len());
        let mut thresholds = samples
            .iter()
            .map(|f| f.data[best_feature])
            .collect::<Vec<f64>>();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        thresholds.dedup();
        if thresholds.len() == 1 {
            for i in 0..samples[0].data.len() {
                thresholds = samples.iter().map(|f| f.data[i]).collect::<Vec<f64>>();
                thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                thresholds.dedup();
                if thresholds.len() > 1 {
                    best_feature = i;
                    break;
                }
            }
        }

        let best_threshold = if thresholds.len() == 2 {
            thresholds[1]
        } else {
            thresholds[thread_rng().gen_range(1..thresholds.len() - 1)]
        };
        let best_impurity = f64::NAN;
        (best_feature, best_threshold, best_impurity)
    }
}
