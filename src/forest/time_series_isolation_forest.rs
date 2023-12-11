use crate::feature_extraction::statistics::{mean, slope, std};
use crate::tree::tree::Tree;
use crate::utils::structures::Sample;
use crate::{
    forest::forest::{Forest, OutlierForest},
    tree::isolation_tree::IsolationTree,
};
use rand::{thread_rng, Rng};

pub struct TimeSeriesIsolationForest {
    trees: Vec<IsolationTree>,
    n_trees: usize,
    n_intervals: usize,
    min_interval_length: usize,
    intervals: Vec<Vec<(usize, usize)>>,
    enhanced_anomaly_score: bool,
    max_samples: usize,
    max_depth: Option<usize>,
}

impl TimeSeriesIsolationForest {
    pub fn new(
        n_trees: usize,
        n_intervals: usize,
        enhanced_anomaly_score: bool,
        max_depth: Option<usize>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            n_intervals,
            intervals: Vec::new(),
            max_depth,
            enhanced_anomaly_score,
            max_samples: 0,
            min_interval_length: 3,
        }
    }
}

impl Forest<IsolationTree> for TimeSeriesIsolationForest {
    fn compute_intervals(&mut self, n_features: usize) {
        // Generate n_intervals, with random start and end
        for _i in 0..self.get_n_trees() {
            let mut intervals = Vec::new();
            for _j in 0..self.n_intervals {
                let start = thread_rng().gen_range(0..n_features - self.min_interval_length);
                let end = thread_rng().gen_range(start + self.min_interval_length..n_features);
                intervals.push((start, end));
            }
            self.intervals.push(intervals);
        }
    }
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
    fn transform<'a>(&self, data: &[Sample<'a>], intervals_index: usize) -> Vec<Sample<'a>> {
        let n_samples = data.len();
        let mut transformed_data: Vec<Sample<'_>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in self.intervals[intervals_index].iter().copied() {
                let mean = mean(&data[j].data[start..end]);
                let std = std(&data[j].data[start..end]);
                let slope = slope(&data[j].data[start..end]);
                sample.extend([mean, std, slope].into_iter());
            }
            transformed_data.push(Sample {
                data: std::borrow::Cow::Owned(sample),
                target: data[j].target,
            });
        }
        transformed_data
    }
}

impl OutlierForest for TimeSeriesIsolationForest {
    fn get_enhanced_anomaly_score(&self) -> bool {
        self.enhanced_anomaly_score
    }
}
