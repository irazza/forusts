use std::{mem::transmute, ops::Deref, sync::Arc};

pub struct FloatVecEq(pub Arc<Vec<f64>>);

impl PartialEq for FloatVecEq {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let self_: &Vec<FloatEq> = transmute(self.0.deref());
            let other_: &Vec<FloatEq> = transmute(other.0.deref());

            self_.eq(other_)
        }
    }
}

impl Eq for FloatVecEq {}

impl std::hash::Hash for FloatVecEq {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            let self_: &Vec<FloatEq> = transmute(self.0.deref());
            self_.hash(state);
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct FloatEq(pub f64);

impl PartialEq for FloatEq {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for FloatEq {}

impl std::hash::Hash for FloatEq {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl std::cmp::PartialOrd for FloatEq {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl std::cmp::Ord for FloatEq {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

pub fn next_up(value: f64) -> f64 {
    // We must use strictly integer arithmetic to prevent denormals from
    // flushing to zero after an arithmetic operation on some platforms.
    const TINY_BITS: u64 = 0x1; // Smallest positive f64.
    const CLEAR_SIGN_MASK: u64 = 0x7fff_ffff_ffff_ffff;

    let bits = value.to_bits();
    if value.is_nan() || bits == f64::INFINITY.to_bits() {
        return value;
    }

    let abs = bits & CLEAR_SIGN_MASK;
    let next_bits = if abs == 0 {
        TINY_BITS
    } else if bits == abs {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next_bits)
}
