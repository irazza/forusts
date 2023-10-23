use hashbrown::HashMap;
use rand::{seq::SliceRandom, thread_rng};
use std::{
    cmp::{max, min},
    ops::Deref,
};

#[derive(Debug, Clone)]
pub enum Node {
    Leaf {
        class: usize,
        depth: usize,
        impurity: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
        depth: usize,
        impurity: f64,
    },
}

impl Node {
    pub fn get_depth(&self) -> usize {
        match self {
            Node::Leaf {
                class: _,
                depth,
                impurity: _,
            } => return *depth,
            Node::Split {
                feature: _,
                threshold: _,
                left: _,
                right: _,
                depth,
                impurity: _,
            } => return *depth,
        }
    }
}

pub struct DecisionTree {
    root: Node,
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize,
    tree_depth: usize,
}

struct Sample<'a> {
    features: &'a [f64],
    target: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize, min_samples_split: usize, max_features: usize) -> Self {
        Self {
            root: Node::Leaf {
                class: 0,
                depth: 0,
                impurity: 0.0,
            },
            max_depth,
            min_samples_split,
            max_features,
            tree_depth: 0,
        }
    }

    pub fn fit(&mut self, x: &Vec<&Vec<f64>>, y: &Vec<usize>) {
        // Start the iterative tree-building process
        let mut data = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| Sample {
                features: &x,
                target: *y,
            })
            .collect::<Vec<_>>();

        self.root = self.build_tree(&mut data, self.max_depth, self.min_samples_split);
    }

    fn build_tree(
        &mut self,
        samples: &mut [Sample<'_>],
        max_depth: usize,
        min_samples_split: usize,
    ) -> Node {
        let current_depth = max(1, self.max_depth - max_depth);
        if self.tree_depth < current_depth {
            self.tree_depth = current_depth;
        }

        if samples.len() < min_samples_split || max_depth == 0 {
            return Node::Leaf {
                class: get_most_common_class(samples),
                depth: current_depth,
                impurity: 0.0,
            };
        }

        let (best_feature, best_threshold, best_gini) = self.get_best_split(samples);

        if best_gini == 0.0 {
            return Node::Leaf {
                class: samples[0].target,
                depth: current_depth,
                impurity: 0.0,
            };
        }

        let (left_data, right_data) = split(samples, best_feature, best_threshold);

        let left_subtree = self.build_tree(left_data, max_depth - 1, min_samples_split);
        let right_subtree = self.build_tree(right_data, max_depth - 1, min_samples_split);

        Node::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(left_subtree),
            right: Box::new(right_subtree),
            depth: current_depth,
            impurity: best_gini,
        }
    }

    // pub fn lca(&self, x1: &Vec<f64>, x2: &Vec<f64>) -> &Node {
    //     let mut node = &self.root;
    //     // Traverse the tree to make a prediction
    //     while let Node::Split {
    //         feature,
    //         threshold,
    //         left,
    //         right,
    //         depth: _,
    //         impurity: _,
    //     } = node
    //     {
    //         let x1_test = x1[*feature] <= *threshold;
    //         let x2_test = x2[*feature] <= *threshold;
    //         if x1_test != x2_test {
    //             return node;
    //         }
    //         if x1_test {
    //             node = left;
    //         } else {
    //             node = right;
    //         }
    //     }
    //     node
    // }

    pub fn predict_leaf(&self, x: &Vec<f64>) -> &Node {
        let mut node = &self.root;

        // Traverse the tree to make a prediction
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
        let n_samples = x.len();
        let mut predictions = vec![0; n_samples];

        for (i, sample) in x.iter().enumerate() {
            let node = self.predict_leaf(sample);

            if let Node::Leaf {
                class,
                depth: _,
                impurity: _,
            } = node
            {
                predictions[i] = *class;
            }
        }

        predictions
    }

    fn get_best_split(&self, samples: &[Sample<'_>]) -> (usize, f64, f64) {
        let mut best_feature = usize::MAX;
        let mut best_threshold = f64::MAX;
        let mut best_gini = f64::MAX;

        let mut selected_features: Vec<_> = (0..samples[0].features.len()).collect();
        selected_features.shuffle(&mut thread_rng());

        for &feature in &selected_features[0..min(samples[0].features.len(), self.max_features)] {
            let mut features = Vec::new();

            for Sample {
                features: sample,
                target,
            } in samples
            {
                features.push((sample[feature], target));
            }

            features.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
            let mut left_class_counts = HashMap::new();
            let mut right_class_counts = HashMap::new();
            for Sample { target: v, .. } in samples {
                *right_class_counts.entry(*v).or_insert(0) += 1;
            }

            for (idx, &(v, &target)) in features.iter().enumerate() {
                right_class_counts.entry(target).and_modify(|e| *e -= 1);
                *left_class_counts.entry(target).or_insert(0) += 1;

                let left_gini = get_gini_impurity(&left_class_counts);
                let right_gini = get_gini_impurity(&right_class_counts);

                let left_size = (idx + 1) as f64;
                let right_size = samples.len() as f64 - left_size;

                let gini = (left_size / samples.len() as f64) * left_gini
                    + (right_size / samples.len() as f64) * right_gini;

                if gini < best_gini {
                    best_gini = gini;
                    best_feature = feature;
                    best_threshold = v;
                }
            }
        }

        (best_feature, best_threshold, best_gini)
    }

    // fn minimal_cost_complexity_pruning(&self, ccp_alpha: &f64) { }

    // pub fn ancestor(&self, x1: &Vec<f64>, x2: &Vec<f64>) -> usize {
    //     return (self.predict_leaf(&x1).get_depth() + self.predict_leaf(&x2).get_depth())
    //         - 2 * self.lca(&x1, &x2).get_depth();
    // }

    pub fn compute_ancestor<'a>(&'a self, node: &'a Node) -> HashMap<*const Node, &'a Node> {
        let mut ancestors = HashMap::new();
        compute_ancestor_rec(&self.root, node, None, &mut ancestors);
        ancestors.insert(node as *const Node, node);
        ancestors
    }

    // pub fn compute_zhu(&self, node: &Node) -> HashMap<*const Node, usize> {
    //     let mut distances = HashMap::new();
    //     compute_zhou_rec(&self.root, node, None, &mut distances);
    //     distances.insert(node as *const Node, 0);
    //     distances
    // }
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
        if samples[idx].features[feature] <= threshold {
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

fn get_gini_impurity(class_counts: &HashMap<usize, usize>) -> f64 {
    let mut impurity = 1.0;
    let total_samples = class_counts.values().sum::<usize>() as f64;
    for count in class_counts.values() {
        let p = *count as f64 / total_samples;
        impurity -= p * p;
    }

    impurity
}
