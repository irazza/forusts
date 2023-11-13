#![allow(dead_code)]
use crate::tree::node::Node;
use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::{cmp::max, ops::Deref};

#[derive(Copy, Clone)]
pub enum MaxFeatures {
    All,
    Sqrt,
    Log2,
}

#[derive(Copy, Clone)]
pub enum Splitter {
    Best,
    Random,
}
impl Splitter {
    pub fn to_string(&self) -> &str {
        match self {
            Splitter::Best => "B",
            Splitter::Random => "R",
        }
    }
}

#[derive(Copy, Clone)]
pub enum Criterion {
    Gini,
    Entropy,
    None,
}
impl Criterion {
    pub fn to_string(&self) -> String {
        match self {
            Criterion::Gini => String::from("gini"),
            Criterion::Entropy => String::from("entropy"),
            Criterion::None => String::from("none"),
        }
    }
}

pub struct DecisionTree {
    root: Node,
    criterion: Criterion,
    splitter: Splitter,
    impurity: fn(&HashMap<usize, usize>) -> f64,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: MaxFeatures,
    _max_features: usize,
}

struct Sample<'a> {
    data: &'a [f64],
    target: usize,
}

impl DecisionTree {
    pub fn new(
        criterion: Criterion,
        splitter: Splitter,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: MaxFeatures,
    ) -> Self {
        Self {
            root: Node::Leaf {
                class: 0,
                depth: 0,
                impurity: f64::MAX,
            },
            criterion,
            splitter,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            impurity: match criterion {
                Criterion::Gini => gini_impurity,
                Criterion::Entropy => entropy_impurity,
                Criterion::None => random_impurity,
            },
            _max_features: 0,
        }
    }

    pub fn fit(&mut self, x: &Vec<&Vec<f64>>, y: &Vec<usize>) {
        // Start the iterative tree-building process
        let mut data = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| Sample {
                data: &x,
                target: *y,
            })
            .collect::<Vec<_>>();

        self._max_features = match self.max_features {
            MaxFeatures::All => data[0].data.len(),
            MaxFeatures::Sqrt => (data[0].data.len() as f64).sqrt() as usize,
            MaxFeatures::Log2 => (data[0].data.len() as f64).log2() as usize,
        };

        self.root = self.build_tree(&mut data, self.max_depth);
    }

    fn build_tree(&mut self, samples: &mut [Sample<'_>], max_depth: usize) -> Node {
        let current_depth = max(1, self.max_depth - max_depth);

        let (best_feature, best_threshold, best_impurity) = match self.splitter {
            Splitter::Best => self.get_best_split(samples),
            Splitter::Random => self.get_random_split(samples),
        };

        if self.stop_conditions(samples, current_depth, best_impurity) {
            let mut class_counts = HashMap::new();
            for Sample { target, .. } in &samples[0..samples.len()] {
                *class_counts.entry(*target).or_insert(0) += 1;
            }
            return Node::Leaf {
                class: get_most_common_class(samples),
                depth: current_depth,
                impurity: (self.impurity)(&class_counts),
            };
        }

        let (left_data, right_data) = split(samples, best_feature, best_threshold);

        if left_data.len() == 0 || right_data.len() == 0 {
            return Node::Leaf {
                class: get_most_common_class(samples),
                depth: current_depth,
                impurity: best_impurity,
            };
        }
        // Split the data and recursively build the left and right subtrees
        let left_subtree = self.build_tree(left_data, max_depth - 1);
        let right_subtree = self.build_tree(right_data, max_depth - 1);

        Node::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(left_subtree),
            right: Box::new(right_subtree),
            depth: current_depth,
            impurity: best_impurity,
        }
    }

    fn stop_conditions(
        &self,
        samples: &mut [Sample<'_>],
        current_depth: usize,
        impurity: f64,
    ) -> bool {
        // Base case: not enough samples or max depth reached
        if samples.len() <= self.min_samples_split || current_depth == self.max_depth {
            return true;
        }
        // Base case: all samples have the same class
        let first_class = samples[0].target;
        for Sample { target, .. } in samples {
            if *target != first_class {
                return false;
            }
        }
        // Base case: impurity is 0
        if impurity <= f64::EPSILON {
            return true;
        }
        true
    }

    fn get_random_split(&self, samples: &[Sample<'_>]) -> (usize, f64, f64) {
        let best_feature = thread_rng().gen_range(0..samples[0].data.len());
        let best_threshold = samples[thread_rng().gen_range(0..samples.len())].data[best_feature];
        let best_impurity = 1.0;

        (best_feature, best_threshold, best_impurity)
    }

    fn get_best_split(&self, samples: &[Sample<'_>]) -> (usize, f64, f64) {
        let mut best_feature = usize::MAX;
        let mut best_threshold = f64::MAX;
        let mut best_impurity = f64::MAX;
        let n_samples = samples.len() as f64;
        let mut parent_entropy = 0.0;

        if self.criterion.to_string() == "entropy" {
            // reset impurity
            best_impurity = 0.0;
            let mut class_counts = HashMap::new();
            for Sample { target, .. } in samples {
                *class_counts.entry(*target).or_insert(0) += 1;
            }
            parent_entropy = (self.impurity)(&class_counts);
        }

        let mut features_idxs: Vec<_> = (0..samples[0].data.len()).collect();
        features_idxs.shuffle(&mut thread_rng());
        for &idx in &features_idxs[0..self._max_features] {
            let mut samples_feature: Vec<(f64, usize)> = samples
                .iter()
                .map(
                    |Sample {
                         data: sample,
                         target,
                     }| (sample[idx], *target),
                )
                .collect();
            samples_feature.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());

            let mut left_class_counts = HashMap::new();
            let mut right_class_counts = HashMap::new();

            for Sample { target: v, .. } in samples {
                *right_class_counts.entry(*v).or_insert(0) += 1;
            }
            for (i, &(v, target)) in samples_feature.iter().enumerate() {
                right_class_counts.entry(target).and_modify(|e| *e -= 1);
                *left_class_counts.entry(target).or_insert(0) += 1;

                let left_impurity = (self.impurity)(&left_class_counts);
                let right_impurity = (self.impurity)(&right_class_counts);

                let left_size = (i + 1) as f64;
                let right_size = n_samples - left_size;

                let impurity = if self.criterion.to_string() == "gini" {
                    (left_size / n_samples) * left_impurity
                        + (right_size / n_samples) * right_impurity
                } else if self.criterion.to_string() == "entropy" {
                    1.0 / (parent_entropy
                        - ((left_size / n_samples) * left_impurity
                            + (right_size / n_samples) * right_impurity))
                } else {
                    1.0
                };

                if impurity < best_impurity
                {
                    best_impurity = impurity;
                    best_feature = idx;
                    best_threshold = v;
                }
            }
        }

        (best_feature, best_threshold, best_impurity)
    }

    pub fn predict_leaf(&self, x: &Vec<f64>) -> &Node {
        let mut node = &self.root;

        while let Node::Split {
            feature,
            threshold,
            left,
            right,
            depth: _,
            impurity: _,
        } = node
        {
            if x[*feature] <= *threshold {
                node = left;
            } else {
                node = right;
            }
        }
        node
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        x.iter()
            .map(|sample| self.predict_leaf(sample).get_class())
            .collect()
    }

    pub fn compute_ancestor<'a>(&'a self, node: &'a Node) -> HashMap<*const Node, &'a Node> {
        let mut ancestors = HashMap::new();
        compute_ancestor_rec(&self.root, node, None, &mut ancestors);
        ancestors.insert(node as *const Node, node);
        ancestors
    }
}

fn compute_ancestor_rec<'a>(
    current: &'a Node,
    target: &'a Node,
    found_lca: Option<&'a Node>,
    ancestors: &mut HashMap<*const Node, &'a Node>,
) -> bool {
    if (current as *const Node) == (target as *const Node) {
        return true;
    }

    match current {
        Node::Leaf { .. } => {
            if let Some(found_lca) = found_lca {
                ancestors.insert(current as *const Node, found_lca);
            }
            false
        }
        Node::Split { left, right, .. } => {
            if let Some(found_lca) = found_lca {
                compute_ancestor_rec(left.deref(), target, Some(found_lca), ancestors);
                compute_ancestor_rec(right.deref(), target, Some(found_lca), ancestors);
                false
            } else {
                let left_found = compute_ancestor_rec(left.deref(), target, None, ancestors);
                if left_found {
                    compute_ancestor_rec(right.deref(), target, Some(current), ancestors);
                    true
                } else {
                    let right_found = compute_ancestor_rec(right, target, None, ancestors);
                    if right_found {
                        compute_ancestor_rec(left.deref(), target, Some(current), ancestors);
                        true
                    } else {
                        false
                    }
                }
            }
        }
    }
}

fn split<'a, 'b>(
    samples: &'a mut [Sample<'b>],
    feature: usize,
    threshold: f64,
) -> (&'a mut [Sample<'b>], &'a mut [Sample<'b>]) {
    let mut idx = 0;
    let mut last = samples.len();

    while idx < last {
        if samples[idx].data[feature] <= threshold {
            idx += 1;
        } else {
            samples.swap(idx, last - 1);
            last -= 1;
        }
    }

    samples.split_at_mut(idx)
}

fn get_most_common_class(samples: &[Sample<'_>]) -> usize {
    let mut class_counts = HashMap::new();

    for Sample { target, .. } in samples {
        *class_counts.entry(*target).or_insert(0) += 1;
    }

    let mut max_count = 0;
    let mut most_common_class = 0;

    for (class, count) in &class_counts {
        if *count > max_count {
            max_count = *count;
            most_common_class = *class;
        }
    }

    most_common_class
}

fn random_impurity(_class_counts: &HashMap<usize, usize>) -> f64 {
    return thread_rng().gen_range(0.1..1.0);
}

fn entropy_impurity(class_counts: &HashMap<usize, usize>) -> f64 {
    let mut impurity = 0.0;
    let total_samples = class_counts.values().sum::<usize>() as f64;
    for &count in class_counts.values() {
        if count > 0 {
            let p = count as f64 / total_samples;
            impurity -= p * p.log2();
        }
    }

    impurity
}

fn gini_impurity(class_counts: &HashMap<usize, usize>) -> f64 {
    let mut impurity = 1.0;
    let total_samples = class_counts.values().sum::<usize>() as f64;
    for &count in class_counts.values() {
        if count > 0 {
            let p = count as f64 / total_samples;
            impurity -= p * p;
        }
    }

    impurity
}
