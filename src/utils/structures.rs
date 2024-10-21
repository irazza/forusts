use serde::{Deserialize, Serialize};
use std::{hash::Hash, sync::Arc};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, PartialOrd)]
pub struct Sample {
    pub target: isize,
    pub features: Arc<Vec<f64>>,
}

impl Hash for Sample {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.target.hash(state);
        self.features.iter().for_each(|f| f.to_bits().hash(state));
    }
}

impl Eq for Sample {}

#[derive(Clone)]
pub enum IntervalType {
    LOG2,
    LOG10,
    LN,
    SQRT,
}
impl IntervalType {
    pub fn get_interval(&self, n: usize) -> usize {
        match self {
            IntervalType::LOG2 => (n as f64).log2() as usize,
            IntervalType::LOG10 => (n as f64).log10() as usize,
            IntervalType::LN => (n as f64).ln() as usize,
            IntervalType::SQRT => (n as f64).sqrt() as usize,
        }
    }
}

#[derive(Clone)]
pub enum MaxFeatures {
    LOG2,
    LOG10,
    LN,
    SQRT,
    ALL,
}
impl crate::utils::structures::MaxFeatures {
    pub fn get_features(&self, n: usize) -> usize {
        match self {
            crate::utils::structures::MaxFeatures::LOG2 => (n as f64).log2() as usize,
            crate::utils::structures::MaxFeatures::LOG10 => (n as f64).log10() as usize,
            crate::utils::structures::MaxFeatures::LN => (n as f64).ln() as usize,
            crate::utils::structures::MaxFeatures::SQRT => (n as f64).sqrt() as usize,
            crate::utils::structures::MaxFeatures::ALL => n,

        }
    }
}
