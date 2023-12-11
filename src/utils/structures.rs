use std::borrow::Cow;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Sample<'a> {
    pub target: isize,
    pub data: Cow<'a, [f64]>
}