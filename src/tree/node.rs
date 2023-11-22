#[derive(Debug, Clone)]
pub enum Node {
    Leaf {
        class: usize,
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

    pub fn get_class(&self) -> usize {
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

    pub fn get_n_samples(&self) -> usize {
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
