#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, PartialOrd)]
pub struct Sample {
    pub target: isize,
    pub features: Arc<Vec<f64>>,
}

#[derive(Clone)]
pub enum IntervalType {
    N(usize),
    LOG2,
    LOG10,
    LN,
    SQRT,
}
impl IntervalType {
    pub fn get_interval(&self, n: usize) -> usize {
        match self {
            IntervalType::N(n) => *n,
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

#[macro_export]
macro_rules! assert_eq_with_tol {
    ($left:expr, $right:expr, $tolerance:expr) => {
        let left = $left;
        let right = $right;
        let tolerance = $tolerance;

        if (left - right).abs() > tolerance {
            panic!(
                "assertion failed: `(left == right)` \
                \n   left: `{:?}`,\
                \n  right: `{:?}`,\
                \n  diff:  `{:?}`,\
                \n  max tolerance: `{:?}`",
                left,
                right,
                (left - right).abs(),
                tolerance
            );
        }
    };
}
