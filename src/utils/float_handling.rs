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