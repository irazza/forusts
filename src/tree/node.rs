use std::{ops::Deref, sync::Arc};

use super::tree::SplitParameters;

#[derive(Debug, Clone)]
pub enum Node<S: SplitParameters> {
    Leaf {
        class: isize,
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
        n_samples: usize,
    },
}

impl<S: SplitParameters> Node<S> {
    pub fn new() -> Self {
        Node::Leaf {
            class: 0,
            depth: 0,
            impurity: 0.0,
            n_samples: 0,
        }
    }
    pub fn get_split_parameters(&self) -> &S {
        match self {
            Node::Leaf {
                class: _,
                depth: _,
                impurity: _,
                n_samples: _,
            } => panic!("Cannot get feature of a leaf node"),
            Node::Split {
                split_params,
                left: _,
                right: _,
                depth: _,
                impurity: _,
                n_samples: _,
            } => return split_params,
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
                n_samples: _,
            } => return *depth,
        }
    }

    pub fn get_class(&self) -> isize {
        match self {
            Node::Leaf {
                class,
                depth: _,
                impurity: _,
                n_samples: _,
            } => return *class,
            Node::Split {
                split_params: _,
                left: _,
                right: _,
                depth: _,
                impurity: _,
                n_samples: _,
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
                n_samples,
            } => return *n_samples,
        }
    }
}
