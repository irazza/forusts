use crate::tree::decision_tree::DecisionTree;
use hashbrown::HashMap;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

#[allow(dead_code)]
pub enum MaxFeatures {
    All,
    Sqrt,
    Log2,
}

pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
    max_features: MaxFeatures,
    max_depth: Option<usize>,
}

#[allow(dead_code)]
impl RandomForest {
    pub fn new(n_trees: usize, max_features: MaxFeatures, max_depth: Option<usize>) -> Self {
        Self {
            trees: Vec::new(),
            n_trees,
            max_features,
            max_depth,
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>) {
        let n_samples = x.len();

        self.trees
            .par_extend((0..self.n_trees).into_par_iter().map(|_i| {
                let bootstrap_indices: Vec<usize> = (0..n_samples)
                    .map(|_| thread_rng().gen_range(0..n_samples))
                    .collect();
                let n_features = x[0].len();
                let mut tree = DecisionTree::new(
                    self.max_depth.unwrap_or(usize::MAX),
                    2,
                    match self.max_features {
                        MaxFeatures::All => n_features,
                        MaxFeatures::Sqrt => (n_features as f64).sqrt() as usize,
                        MaxFeatures::Log2 => n_features.ilog2() as usize,
                    },
                );
                tree.fit(
                    &bootstrap_indices.iter().map(|i| &x[*i]).collect(),
                    &bootstrap_indices.iter().map(|i| y[*i]).collect(),
                );
                tree
            }));
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        let n_samples = x.len();
        let mut predictions = Vec::new();

        // Make predictions for each sample using each tree in the forest
        predictions.par_extend(
            self.trees
                .par_iter()
                .enumerate()
                .map(|(_i, tree)| tree.predict(x)),
        );

        // Combine predictions using a majority vote
        let mut final_predictions = vec![0; n_samples];

        for i in 0..n_samples {
            let mut class_counts = HashMap::new();
            for j in 0..self.n_trees {
                let class = predictions[j][i];
                *class_counts.entry(class).or_insert(0) += 1;
            }

            // Find the class with the maximum count
            let mut max_count = 0;
            let mut majority_class = 0;
            for (class, count) in &class_counts {
                if *count > max_count {
                    max_count = *count;
                    majority_class = *class;
                }
            }

            final_predictions[i] = majority_class;
        }

        final_predictions
    }
}
