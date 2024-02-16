// use super::node::Node;
// use crate::{
//     forest::forest::{OutlierForestConfig, OutlierTree},
//     tree::tree::Tree,
//     utils::structures::Sample,
// };
// use rand::{seq::SliceRandom, thread_rng, Rng};

// #[derive(Clone, Debug)]
// pub struct SCIsolationTreeConfig {
//     pub max_depth: usize,
//     pub min_samples_split: usize,
// }

// #[derive(Clone, Debug)]
// pub struct SCIsolationTree {
//     root: Node,
//     config: SCIsolationTreeConfig,
// }
// impl SCIsolationTree {
//     fn get_random_hyperplane(&self, samples: &[Sample<'_>]) -> (usize, f64){
//         let n_features = samples[0].data.len();
//         let mut subsampled_features = (0..n_features).collect::<Vec<_>>();
//         subsampled_features.shuffle(&mut thread_rng());
//         for i in subsampled_features {
//             let c_i = thread_rng().gen_range(-1.0..1.0);
//             let x = samples.iter().map(|s| s.data[i]).collect::<Vec<_>>();

//         }
//         todo!()
//     }
// }

// impl OutlierTree for SCIsolationTree {
//     fn from_outlier_config(max_samples: usize, config: &OutlierForestConfig) -> Self {
//         Self::new(SCIsolationTreeConfig {
//             max_depth: config.max_depth.unwrap_or(max_samples.ilog2() as usize + 1),
//             min_samples_split: 2,
//             // Setted to 2 to avoid empty child when splitting when there are only two samples
//         })
//     }
// }

// impl Tree for SCIsolationTree {
//     type Config = SCIsolationTreeConfig;
//     fn new(config: Self::Config) -> Self {
//         Self {
//             root: Node::new(),
//             config,
//         }
//     }
//     fn get_max_depth(&self) -> usize {
//         self.config.max_depth
//     }
//     fn get_root(&self) -> &Node {
//         &self.root
//     }
//     fn set_root(&mut self, root: Node) {
//         self.root = root;
//     }
//     fn pre_split_conditions(&self, samples: &[Sample<'_>], current_depth: usize) -> bool {
//         // Base case: not enough samples or max depth reached
//         if samples.len() <= self.config.min_samples_split || current_depth == self.config.max_depth
//         {
//             return true;
//         }
//         // Base case: samples are the same object
//         let first_sample = &samples[0];
//         let is_all_same_data = samples.iter().all(|v| v == first_sample);
//         if is_all_same_data {
//             return true;
//         }
//         return false;
//     }
//     fn post_split_conditions(&self, new_impurity: f64, _old_impurity: f64) -> bool {
//         return false;
//     }
//     fn get_split(&self, samples: &[Sample<'_>]) -> (usize, f64, f64) {
//         let mut best_f = 0;
//         let mut best_t = 0.0;
//         let mut best_sd_gain = 0.0;
//         for _ in 0..100 {
//             let (f, t) = self.get_random_hyperplane(samples);
//             let mut x_l = Vec::new();
//             let mut x_r = Vec::new();
//             for i in 0..samples.len()
//             {
//                 let x = samples[i].data[f];
//                 if x < t {
//                     x_l.push(x);
//                 } else {
//                     x_r.push(x);
//                 }
//             }
//             let sd_gain = SCIsolationTree::sd_gain(&x_l, &x_r);
//             if sd_gain > best_sd_gain {
//                 best_sd_gain = sd_gain;
//                 best_f = f;
//                 best_t = t;
//             }
//         }
//         (best_f, best_t, best_sd_gain)
//     }

// }
