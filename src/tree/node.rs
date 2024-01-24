#[derive(Debug, Clone)]
pub enum Node {
    Leaf {
        class: isize,
        depth: usize,
        impurity: f64,
        n_samples: usize,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
        depth: usize,
        impurity: f64,
        n_samples: usize,
    },
}

impl Node {
    pub fn new() -> Self {
        Node::Leaf {
            class: 0,
            depth: 0,
            impurity: 0.0,
            n_samples: 0,
        }
    }
    pub fn get_feature(&self) -> usize {
        match self {
            Node::Leaf {
                class: _,
                depth: _,
                impurity: _,
                n_samples: _,
            } => panic!("Cannot get feature of a leaf node"),
            Node::Split {
                feature,
                threshold: _,
                left: _,
                right: _,
                depth: _,
                impurity: _,
                n_samples: _,
            } => return *feature,
        }
    }
    pub fn get_threshold(&self) -> f64 {
        match self {
            Node::Leaf {
                class: _,
                depth: _,
                impurity: _,
                n_samples: _,
            } => panic!("Cannot get threshold of a leaf node"),
            Node::Split {
                feature: _,
                threshold,
                left: _,
                right: _,
                depth: _,
                impurity: _,
                n_samples: _,
            } => return *threshold,
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
                feature: _,
                threshold: _,
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
                feature: _,
                threshold: _,
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
                feature: _,
                threshold: _,
                left: _,
                right: _,
                depth: _,
                impurity: _,
                n_samples,
            } => return *n_samples,
        }
    }
}
