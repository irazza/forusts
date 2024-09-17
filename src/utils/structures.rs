use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, PartialOrd)]
pub struct Sample {
    pub target: isize,
    pub features: Arc<Vec<f64>>,
}