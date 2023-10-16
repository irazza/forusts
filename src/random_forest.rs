use ndarray::{Array1, Array2, s};
use std::collections::HashMap;
use rand::{Rng, thread_rng};
use rayon::prelude::*;
use crate::decision_tree::DecisionTree;

pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_trees: usize,
}

impl RandomForest {
    pub fn new(n_trees: usize) -> Self {
        Self {
            trees: vec![],
            n_trees,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
    
        self.trees.par_extend((0..self.n_trees).into_par_iter().map(|_| {
            let mut indices = (0..n_samples).collect::<Vec<usize>>();
            let mut bootstrap_indices = vec![0; n_samples];
    
            for i in 0..n_samples {
                bootstrap_indices[i] = thread_rng().gen_range(0..n_samples);
            }
    
            let mut x_bootstrap = Array2::zeros((n_samples, n_features));
            let mut y_bootstrap = Array1::zeros(n_samples);
    
            for i in 0..n_samples {
                x_bootstrap.row_mut(i).assign(&x.row(bootstrap_indices[i]));
                y_bootstrap[i] = y[bootstrap_indices[i]];
            }
    
            let mut tree = DecisionTree::new();
            tree.fit(&x_bootstrap, &y_bootstrap);
    
            tree
        }));
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let n_samples = x.shape()[0];
        
        let predictions: Vec<Array1<usize>> = self.trees.par_iter().map(|tree| {
            let mut y_pred = Array1::zeros(n_samples);
            
            for i in 0..n_samples {
                let sample = x.slice(s![i, ..]).to_owned();
                y_pred[i] = tree.predict(&sample);
            }
            
            y_pred
        }).collect();
        
        // Combine predictions from all trees
        let mut y_pred = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut class_votes: HashMap<usize, usize> = HashMap::new();
    
            for prediction in &predictions {
                *class_votes.entry(prediction[i]).or_insert(0) += 1;
            }
    
            let most_common_class = class_votes.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(class, _)| class)
                .unwrap();
    
            y_pred[i] = most_common_class;
        }
        
        y_pred
    }
}
