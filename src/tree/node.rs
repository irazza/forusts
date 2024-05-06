use std::{fmt::Debug, sync::Arc};

use super::tree::SplitParameters;

pub trait LeafClassifier: Sync + Send + Debug {
    fn classify(&self, x: &[f64]) -> isize;
}
#[derive(Debug, Clone)]
pub enum LeafClassification {
    Simple(isize),
    Complex(Arc<dyn LeafClassifier>),
}

#[derive(Debug, Clone)]
pub enum Node<S: SplitParameters> {
    Leaf {
        class: LeafClassification,
        depth: usize,
        impurity: f64,
        n_samples: usize,
    },
    Split {
        split_params: S,
        left: Box<Node<S>>,
        right: Box<Node<S>>,
        depth: usize,
        impurity: f64,
    },
}

impl<S: SplitParameters> Node<S> {
    pub fn new() -> Self {
        Node::Leaf {
            class: LeafClassification::Simple(0),
            depth: 0,
            impurity: 0.0,
            n_samples: 0,
        }
    }
    pub fn get_depth(&self) -> usize {
        match self {
            Node::Leaf {
                class: _,
                depth,
                impurity: _,
                n_samples: _,
            } => return *depth,
            Node::Split {
                split_params: _,
                left: _,
                right: _,
                depth,
                impurity: _,
            } => return *depth,
        }
    }

    pub fn get_class(&self, sample: &[f64]) -> isize {
        match self {
            Node::Leaf {
                class,
                depth: _,
                impurity: _,
                n_samples: _,
            } => {
                return match class {
                    LeafClassification::Simple(c) => *c,
                    LeafClassification::Complex(c) => c.classify(sample),
                }
            }
            Node::Split {
                split_params: _,
                left: _,
                right: _,
                depth: _,
                impurity: _,
            } => panic!("Cannot get class of a split node"),
        }
    }

    pub fn get_samples(&self) -> usize {
        match self {
            Node::Leaf {
                class: _,
                depth: _,
                impurity: _,
                n_samples,
            } => return *n_samples,
            Node::Split {
                split_params: _,
                left: _,
                right: _,
                depth: _,
                impurity: _,
            } => panic!("Cannot get samples of a split node"),
        }
    }
}
