use serde::{Deserialize, Serialize};
use std::borrow::Cow;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Sample<'a> {
    pub target: isize,
    pub data: Cow<'a, [f64]>,
}
impl<'a> Sample<'a> {
    pub fn to_ref(&self) -> Sample {
        Sample {
            target: self.target,
            data: Cow::Borrowed(&self.data),
        }
    }
}
