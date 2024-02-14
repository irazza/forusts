// use crate::feature_extraction::mep::compute_mep_all;
// use crate::grid_search_tuning;
// use crate::utils::structures::Sample;
// use crate::utils::tuning::TuningConfig;
// use crate::{
//     forest::forest::{Forest, OutlierForest},
//     tree::isolation_tree::IsolationTree,
// };
// use rand::{thread_rng, Rng};

// use super::forest::{OutlierForestConfig, OutlierForestConfigTuning};

// pub const MIN_INTERVAL: usize = 20;

// grid_search_tuning! {
//     pub struct MEPIsolationForestConfig[MEPIsolationForestConfigTuning] {
//         pub n_intervals: usize,
//         pub outlier_config: OutlierForestConfig [OutlierForestConfigTuning],
//     }
// }
// impl TuningConfig for MEPIsolationForestConfigTuning {
//     type Tree = IsolationTree;
//     type Forest = MEPIsolationForest;
// }

// pub struct MEPIsolationForest {
//     trees: Vec<IsolationTree>,
//     intervals: Vec<Vec<(usize, usize)>>,
//     config: MEPIsolationForestConfig,
// }

// impl Forest<IsolationTree> for MEPIsolationForest {
//     type Config = MEPIsolationForestConfig;
//     type TuningType = f64;

//     fn new(config: Self::Config) -> Self {
//         Self {
//             trees: Vec::new(),
//             intervals: Vec::new(),
//             config
//         }
//     }
//     fn fit(&mut self, data: &mut [Sample<'_>]) {
//         self.fit_(data);
//     }
//     fn predict(&self, data: &[Sample<'_>]) -> Vec<isize> {
//         self.predict_(data)
//     }
//     fn compute_intervals(&mut self, n_features: usize) {
//         // Generate n_intervals, with random start and end
//         for _i in 0..self.config.outlier_config.n_trees {
//             let mut intervals = Vec::new();
//             for _j in 0..self.config.n_intervals {
//                 let start = thread_rng().gen_range(0..n_features - MIN_INTERVAL);
//                 let end = thread_rng().gen_range(start + MIN_INTERVAL..n_features);
//                 intervals.push((start, end));
//             }
//             self.intervals.push(intervals);
//         }
//     }
//     fn get_trees(&self) -> &Vec<IsolationTree> {
//         &self.trees
//     }
//     fn get_trees_mut(&mut self) -> &mut Vec<IsolationTree> {
//         &mut self.trees
//     }
//     fn transform<'a>(&self, data: &[Sample<'a>], tree_index: usize) -> Vec<Sample<'a>> {
//         let n_samples = data.len();
//         let mut transformed_data: Vec<Sample<'_>> = Vec::new();
//         for j in 0..n_samples {
//             let mut sample = Vec::new();
//             for (start, end) in self.intervals[tree_index].iter().copied() {
//                 sample.extend(compute_mep_all(
//                     &data[j].data[start..end],
//                 ).iter());
//             }
//             transformed_data.push(Sample {
//                 data: std::borrow::Cow::Owned(sample),
//                 target: data[j].target,
//             });
//         }
//         transformed_data
//     }
//     fn tuning_predict(
//         &self,
//         ds_train: &[Sample<'_>],
//         ds_test: &[Sample<'_>],
//     ) -> Vec<Self::TuningType> {
//         self.score_samples(ds_test)
//     }
// }

// impl OutlierForest<IsolationTree> for MEPIsolationForest {
//     fn get_forest_config(&self) -> &OutlierForestConfig {
//         &self.config.outlier_config
//     }
// }
