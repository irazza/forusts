// #![allow(dead_code, unused_variables)]
// use crate::feature_extraction::catch22;
// use crate::tree::{tree::{Criterion, MaxFeatures}, decision_tree::DecisionTree};
// use crate::forest::forest::ClassificationForest;

// pub struct CanonicalIntervalForest {
//     trees: Vec<DecisionTree>,
//     criterion: Criterion,
//     n_trees: usize,
//     n_intervals: usize,
//     min_interval_length: usize,
//     intervals: Vec<Vec<(usize, usize)>>,
//     max_features: MaxFeatures,
//     max_depth: Option<usize>,
// }

// impl CanonicalIntervalForest {
//     pub fn new(
//         n_trees: usize,
//         criterion: Criterion,
//         n_intervals: usize,
//         min_interval_length: usize,
//         max_features: MaxFeatures,
//         max_depth: Option<usize>,
//     ) -> Self {
//         Self {
//             trees: Vec::new(),
//             n_trees,
//             criterion,
//             n_intervals,
//             min_interval_length,
//             intervals: Vec::new(),
//             max_features,
//             max_depth,
//         }
//     }

//     pub fn transform(x: &[Vec<f64>], intervals: &Vec<(usize, usize)>) -> Vec<Vec<f64>> {
//         let x1 = x[0].clone();
//         let dn_histogram_mode_5 = catch22::dn_histogram_mode_n(&x1, 5);
//         let dn_histogram_mode_10 = catch22::dn_histogram_mode_n(&x1, 10);
//         let md_hrv_classic_pnn40 = catch22::md_hrv_classic_pnn40(&x1);
//         let sb_binary_stats_mean_longstretch1 = catch22::sb_binary_stats_mean_longstretch1(&x1);
//         let in_auto_mutual_info_stats_40_gaussian_fmmi =
//             catch22::in_auto_mutual_info_stats_40_gaussian_fmmi(&x1);
//         let dn_outlier_include_p_001_mdrmd = catch22::dn_outlier_include_p_001_mdrmd(&x1);
//         let dn_outlier_include_n_001_mdrmd = catch22::dn_outlier_include_n_001_mdrmd(&x1);
//         let sb_binary_stats_diff_longstretch0 = catch22::sb_binary_stats_diff_longstretch0(&x1);
//         let n_samples = x.len();
//         let mut transformed_x: Vec<Vec<f64>> = Vec::new();
//         for j in 0..n_samples {
//             let mut sample = Vec::new();
//             for (start, end) in intervals {
//                 //TODO
//                 sample.extend([0.0, 1.0, 2.0].into_iter());
//             }
//             transformed_x.push(sample);
//         }
//         transformed_x
//     }
// }

// impl ClassificationForest for CanonicalIntervalForest {

// }
