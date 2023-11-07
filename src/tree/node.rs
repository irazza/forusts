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

    pub fn get_class(&self) -> usize {
        match self {
            Node::Leaf {
                class,
                depth: _,
                impurity: _,
            } => return *class,
            Node::Split {
                feature: _,
                threshold: _,
                left: _,
                right: _,
                depth: _,
                impurity: _,
            } => panic!("Cannot get class of a split node"),
        }
    }
}
