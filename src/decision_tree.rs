use ndarray::{Array, Array1, Array2, Axis};
use std::collections::HashMap;

#[derive(Debug)]
pub enum Node {
    Leaf { class: usize },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

#[derive(Debug)]
pub struct DecisionTree {
    root: Node,
}

impl DecisionTree {
    pub fn new() -> Self {
        Self { root: Node::Leaf { class: 0 } }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) {
        self.root = build_tree(x, y);
    }

    pub fn predict(&self, x: &Array1<f64>) -> usize {
        predict_sample(&self.root, x)
    }
}

fn build_tree(x: &Array2<f64>, y: &Array1<usize>) -> Node {
    let n_features = x.shape()[1];

    // Base case: all samples belong to the same class
    if y.iter().all(|&class| class == y[0]) {
        return Node::Leaf { class: y[0] };
    }

    // Base case: no more features to split on
    if n_features == 0 {
        let class_counts = count_classes(y);
        let class = *class_counts.iter().max_by_key(|&(_, count)| count).unwrap().0;
        return Node::Leaf { class };
    }

    // Find the best feature to split on
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_gini = f64::INFINITY;
    for feature in 0..n_features {
        let (threshold, gini) = find_best_split(x, y, feature);
        if gini < best_gini {
            best_feature = feature;
            best_threshold = threshold;
            best_gini = gini;
        }
    }

    // Split the data and recursively build the left and right subtrees
    let (left_indices, right_indices) = split_data(x, y, best_feature, best_threshold);
    let left = build_tree(&x.select(Axis(0), &left_indices), &y.select(Axis(0), &left_indices));
    let right = build_tree(&x.select(Axis(0), &right_indices), &y.select(Axis(0), &right_indices));

    Node::Split {
        feature: best_feature,
        threshold: best_threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn predict_sample(node: &Node, sample: &Array1<f64>) -> usize {
    match node {
        Node::Leaf { class } => *class,
        Node::Split { feature, threshold, left, right } => {
            if sample[*feature] <= *threshold {
                predict_sample(left, sample)
            } else {
                predict_sample(right, sample)
            }
        }
    }
}

fn count_classes(y: &Array1<usize>) -> HashMap<usize, usize> {
    let mut class_counts = HashMap::new();
    for &class in y {
        *class_counts.entry(class).or_insert(0) += 1;
    }
    class_counts
}

fn find_best_split(x: &Array2<f64>, y: &Array1<usize>, feature: usize) -> (f64, f64) {
    let n_samples = x.shape()[0];

    // Sort the samples by the feature value
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.sort_by(|&i, &j| x[[i, feature]].partial_cmp(&x[[j, feature]]).unwrap());

    // Initialize the class counts for the left and right subsets
    let mut left_class_counts = HashMap::new();
    let mut right_class_counts = count_classes(y);

    // Initialize the Gini impurity for the left and right subsets
    let mut left_gini = 0.0;
    let mut right_gini = gini_impurity(y);

    // Initialize the best split
    let mut best_threshold = 0.0;
    let mut best_gini = f64::INFINITY;

    // Iterate over the sorted samples and update the class counts and Gini impurities
    for i in 0..n_samples-1 {
        let class = y[indices[i]];
        *left_class_counts.entry(class).or_insert(0) += 1;
        *right_class_counts.get_mut(&class).unwrap() -= 1;

        let threshold = (x[[indices[i], feature]] + x[[indices[i+1], feature]]) / 2.0;
        let left_weight = (i + 1) as f64 / n_samples as f64;
        let right_weight = 1.0 - left_weight;
        let gini = left_weight * gini_impurity_from_counts(&left_class_counts) +
                   right_weight * gini_impurity_from_counts(&right_class_counts);

        if gini < best_gini {
            best_threshold = threshold;
            best_gini = gini;
        }
    }

    (best_threshold, best_gini)
}

fn split_data(x: &Array2<f64>, y: &Array1<usize>, feature: usize, threshold: f64) -> (Vec<usize>, Vec<usize>) {
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    for (i, sample) in x.outer_iter().enumerate() {
        if sample[feature] <= threshold {
            left_indices.push(i);
        } else {
            right_indices.push(i);
        }
    }
    (left_indices, right_indices)
}

fn gini_impurity(y: &Array1<usize>) -> f64 {
    let class_counts = count_classes(y);
    gini_impurity_from_counts(&class_counts)
}

fn gini_impurity_from_counts(class_counts: &HashMap<usize, usize>) -> f64 {
    let n_samples = class_counts.values().sum::<usize>() as f64;
    let mut impurity = 1.0;
    for &count in class_counts.values() {
        let p = count as f64 / n_samples;
        impurity -= p * p;
    }
    impurity
}
