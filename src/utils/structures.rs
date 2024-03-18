use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    cmp::{max, min},
    mem::swap,
};

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
    pub fn to_samples(x: Vec<Vec<f64>>, y: Vec<isize>) -> Vec<Sample<'a>> {
        let mut samples = Vec::new();
        for i in 0..x.len() {
            samples.push(Sample {
                target: y[i],
                data: Cow::Owned(x[i].clone()),
            });
        }
        samples
    }
}

pub fn train_test_split<'a>(
    data: &'a Vec<Sample<'a>>,
    test_size: f64,
    stratify: bool,
    random_state: &mut rand_chacha::ChaCha8Rng,
) -> (Vec<Sample<'a>>, Vec<Sample<'a>>) {
    if data.len() < 2 && (test_size > 0. && test_size < 0.5) {
        panic!("The dataset is too small to be splitted.");
    }
    let mut indices: Vec<usize> = (0..data.len()).collect();

    // Shuffle indices
    indices.shuffle(random_state);

    let test_size = (data.len() as f64 * test_size) as usize;
    let test_size = min(data.len() - 1, max(1, test_size));

    let test_indices = &indices[..test_size];
    let train_indices = &indices[test_size..];

    let mut train_data: Vec<_> = train_indices.iter().map(|&i| data[i].clone()).collect();

    let mut test_data: Vec<_> = test_indices.iter().map(|&i| data[i].clone()).collect();

    if stratify {
        let mut count_train = train_data.iter().filter(|s| s.target == 0).count();
        let mut count_test = test_data.iter().filter(|s| s.target == 0).count();

        train_data.sort_by(|a, b| a.target.cmp(&b.target));
        test_data.sort_by(|a, b| a.target.cmp(&b.target));

        let mut idx = 1;
        while idx < test_size - 1 {
            let train_ratio = count_train as f64 / train_data.len() as f64;
            let test_ratio = count_test as f64 / test_data.len() as f64;
            if test_ratio > train_ratio {
                break;
            }
            let test_len = test_data.len();
            swap(&mut train_data[idx], &mut test_data[test_len - idx - 1]);
            count_train -= 1;
            count_test += 1;
            idx += 1;
        }
        let mut idx = 1;
        while idx < test_size - 1 {
            let train_ratio = count_train as f64 / train_data.len() as f64;
            let test_ratio = count_test as f64 / test_data.len() as f64;
            if test_ratio < train_ratio {
                break;
            }
            let train_len = train_data.len();
            swap(&mut train_data[train_len - idx - 1], &mut test_data[idx]);
            count_train += 1;
            count_test -= 1;
            idx += 1;
        }
    }
    train_data.shuffle(random_state);
    test_data.shuffle(random_state);
    (train_data, test_data)
}
