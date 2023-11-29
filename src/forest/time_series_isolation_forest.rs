use crate::feature_extraction::statistics::{mean, slope, std};
use crate::{forest::forest::OutlierForest, tree::extra_tree::ExtraTree};
use rand::{thread_rng, Rng};

pub struct TimeSeriesIsolationForest {
    trees: Vec<ExtraTree>,
    n_trees: usize,
    n_intervals: usize,
    min_interval_length: usize,
    intervals: Vec<Vec<(usize, usize)>>,
    max_depth: Option<usize>,
    enhanced_anomaly_score: Option<bool>,
    max_samples: usize,
}

impl TimeSeriesIsolationForest {
    pub fn new(
        n_trees: usize,
        n_intervals: usize,
        min_interval_length: usize,
        enhanced_anomaly_score: Option<bool>,
        max_depth: Option<usize>,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            n_intervals,
            min_interval_length,
            intervals: Vec::new(),
            max_depth,
            enhanced_anomaly_score,
            max_samples: 256,
        }
    }
}

impl OutlierForest for TimeSeriesIsolationForest {
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
    fn transform(&self, x: &Vec<Vec<f64>>, intervals_index: usize) -> Vec<Vec<f64>> {
        let n_samples = x.len();
        let mut transformed_x: Vec<Vec<f64>> = Vec::new();
        for j in 0..n_samples {
            let mut sample = Vec::new();
            for (start, end) in self.intervals[intervals_index].iter().copied() {
                let mean = mean(&x[j][start..end]);
                let std = std(&x[j][start..end]);
                let slope = slope(&x[j][start..end]);
                sample.extend([mean, std, slope].into_iter());
            }
            transformed_x.push(sample);
        }
        transformed_x
    }
}
