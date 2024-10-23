use serde::{Deserialize, Serialize};
use std::mem::transmute;
use std::ops::Deref;
use std::{hash::Hash, sync::Arc};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, PartialOrd)]
pub struct Sample {
    pub target: isize,
    pub features: Arc<Vec<f64>>,
}

pub struct HashVecF64(pub Arc<Vec<f64>>);

impl PartialEq for HashVecF64 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let self_: &Vec<HashF64> = transmute(self.0.deref());
            let other_: &Vec<HashF64> = transmute(other.0.deref());

            self_.eq(other_)
        }
    }
}

impl Eq for HashVecF64 {}

impl Hash for HashVecF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            let self_: &Vec<HashF64> = transmute(self.0.deref());
            self_.hash(state);
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct HashF64(pub f64);

impl PartialEq for HashF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for HashF64 {}

impl Hash for HashF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl PartialOrd for HashF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for HashF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

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
