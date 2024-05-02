use crate::{
    forest::forest::{ClassificationForestConfig, ClassificationTree},
    tree::{
        node::Node,
        tree::{MaxFeatures, Tree},
    },
    utils::structures::Sample,
};
use rand::{seq::SliceRandom, thread_rng, Rng};

use super::tree::SplitTest;

#[derive(Clone, Debug)]
pub struct ExtraTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub max_features: MaxFeatures,
}

#[derive(Clone, Debug)]
pub struct ExtraTree {
    root: Node<SplitTest>,
    config: ExtraTreeConfig,
}

impl ClassificationTree for ExtraTree {
    type TreeConfig = ClassificationForestConfig;
    fn from_classification_config(config: &ClassificationForestConfig) -> Self {
        Self::new(ExtraTreeConfig {
            max_depth: config.max_depth.unwrap_or(usize::MAX),
            min_samples_split: config.min_samples_split,
            max_features: config.max_features,
        })
    }
}

impl Tree for ExtraTree {
    type Config = ExtraTreeConfig;
    type SplitParameters = SplitTest;
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
    fn get_split(&self, samples: &[Sample]) -> (Self::SplitParameters, f64) {
        // Generate a random subsample (MaxFeatures) of features (length of sample)
        let mut features_subsample = (0..samples[0].data.len()).collect::<Vec<_>>();
        features_subsample.shuffle(&mut thread_rng());
        let features_subsample =
            features_subsample[..self.config.max_features.convert(samples[0].data.len())].to_vec();
        let mut feature_counter = 0;

        let mut thresholds = [f64::NAN].to_vec();
        while thresholds.len() == 1 && feature_counter < features_subsample.len() {
            thresholds = samples
                .iter()
                .map(|f| f.data[features_subsample[feature_counter]])
                .collect::<Vec<f64>>();
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            thresholds.dedup();
            feature_counter += 1;
        }
        if feature_counter == features_subsample.len() {
            // No split found
            return (
                SplitTest {
                    feature: usize::MAX,
                    threshold: f64::MAX,
                },
                f64::MAX,
            );
        }
        let best_feature = features_subsample[feature_counter - 1];
        let best_threshold = if thresholds.len() == 2 {
            thresholds[1]
        } else {
            thresholds[thread_rng().gen_range(1..thresholds.len() - 1)]
        };
        let best_impurity = f64::NAN;
        (
            SplitTest {
                feature: best_feature,
                threshold: best_threshold,
            },
            best_impurity,
        )
    }
    fn pre_split_conditions(&self, samples: &[Sample], current_depth: usize) -> bool {
        // Base case: not enough samples or max depth reached
        if samples.len() <= self.config.min_samples_split || current_depth == self.config.max_depth
        {
            return true;
        }
        // Base case: samples are the same object
        let first_sample = &samples[0].data;
        let is_all_same_data = samples.iter().all(|v| &v.data == first_sample);
        if is_all_same_data {
            return true;
        }
        return false;
    }
    fn post_split_conditions(&self, new_impurity: f64, _old_impurity: f64) -> bool {
        // Base case: no split found
        return new_impurity == f64::MAX;
    }
}
