use std::fmt::Debug;

use super::tree::SplitParameters;

// pub trait LeafClassifier: Sync + Send + Debug {
//     fn classify(&self, x: &[f64]) -> isize;
// }
// #[derive(Debug, Clone)]
// pub enum LeafClassification {
//     Simple(isize),
//     Complex(Arc<dyn LeafClassifier>),
// }

#[derive(Debug, Clone)]
pub enum Node<S: SplitParameters> {
    External {
        id: usize,
        class: isize, //LeafClassification,
        depth: usize,
        n_samples: usize,
    },
    Internal {
        id: usize,
        split_params: S,
        children: Vec<usize>,
        n_children: usize,
        depth: usize,
        impurity: f64,
        n_samples: usize,
    },
}
impl<S: SplitParameters> Node<S> {
    // pub fn get_id(&self) -> usize {
    //     match self {
    //         Node::External { id, .. } => *id,
    //         Node::Internal { id, .. } => *id,
    //     }
    // }
    // pub fn get_n_children(&self) -> usize {
    //     match self {
    //         Node::External { .. } => 0,
    //         Node::Internal { children, .. } => children.len(),
    //     }
    // }
    // pub fn get_class(&self, sample: &[f64]) -> isize {
    //     match self {
    //         Node::External { class, .. } => match class {
    //             LeafClassification::Simple(c) => *c,
    //             LeafClassification::Complex(c) => c.classify(sample),
    //         },
    //         Node::Internal { .. } => panic!("Cannot get class of a split node"),
    //     }
    // }
    pub fn get_class(&self) -> isize {
        match self {
            Node::External { class, .. } => *class,
            Node::Internal { .. } => panic!("Cannot get class of a split node"),
        }
    }
    pub fn get_n_samples(&self) -> usize {
        match self {
            Node::External { n_samples, .. } => *n_samples,
            Node::Internal { n_samples, .. } => *n_samples,
        }
    }
    pub fn get_depth(&self) -> usize {
        match self {
            Node::External { depth, .. } => *depth,
            Node::Internal { depth, .. } => *depth,
        }
    }
}
