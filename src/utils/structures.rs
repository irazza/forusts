use hashbrown::HashMap;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::{
    cmp::{max, min},
    mem::swap,
    sync::Arc,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Sample {
    pub target: isize,
    pub data: Arc<Vec<f64>>,
}
impl Sample {
    pub fn to_ref(&self) -> Sample {
        Sample {
            target: self.target,
            data: self.data.clone(),
        }
    }
    pub fn to_samples(x: Vec<Vec<f64>>, y: Vec<isize>) -> Vec<Sample> {
        let mut samples = Vec::new();
        for i in 0..x.len() {
            samples.push(Sample {
                target: y[i],
                data: Arc::new(x[i].clone()),
            });
        }
        samples
    }
}

pub fn train_test_split(
    data: &Vec<Sample>,
    test_size: f64,
    stratify: bool,
    random_state: &mut rand_chacha::ChaCha8Rng,
) -> (Vec<Sample>, Vec<Sample>) {
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

pub struct KUnionFind {
    parent: Vec<usize>,
    k: usize,
}
impl KUnionFind {
    pub fn new(k: usize, n_samples: usize) -> Self {
        Self {
            parent: (0..n_samples).collect(),
            k,
        }
    }
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    pub fn union(&mut self, x: usize, y: usize) {
        let x_root = self.find(x);
        let y_root = self.find(y);
        self.parent[x_root] = y_root;
    }
    pub fn get_clusters(&mut self, distances: &Vec<Vec<f64>>) -> Vec<Vec<usize>> {
        for (i, obj) in distances.iter().enumerate() {
            let mut nn = obj.iter().enumerate().collect::<Vec<_>>();
            nn.select_nth_unstable_by(self.k, |a, b| a.1.partial_cmp(b.1).unwrap());
            for (j, _) in nn.iter().take(self.k) {
                self.union(i, *j);
            }
        }
        let mut clusters = HashMap::new();
        for i in 0..distances.len() {
            let root = self.find(i);
            clusters.entry(root).or_insert(Vec::new()).push(i);
        }
        clusters.into_iter().map(|(_, v)| v).collect()
    }
}

pub struct ZScoreTransformer {
    mean: Vec<f64>,
    std: Vec<f64>,
}
impl ZScoreTransformer {
    pub fn new() -> Self {
        Self {
            mean: Vec::new(),
            std: Vec::new(),
        }
    }

    pub fn fit(&mut self, samples: &[Sample]) {
        let mut mean = vec![0.0; samples[0].data.len()];
        let mut std = vec![0.0; samples[0].data.len()];
        for sample in samples {
            for (i, &x) in sample.data.iter().enumerate() {
                mean[i] += x;
            }
        }
        let n_samples = samples.len() as f64;
        for i in 0..mean.len() {
            mean[i] /= n_samples;
        }
        for sample in samples {
            for (i, &x) in sample.data.iter().enumerate() {
                std[i] += (x - mean[i]).powi(2);
            }
        }
        for i in 0..std.len() {
            std[i] = (std[i] / n_samples).sqrt();
        }
        self.mean = mean;
        self.std = std;
    }
    pub fn transform(&self, samples: &[Sample]) -> Vec<Sample> {
        let mut transformed = Vec::new();
        for sample in samples {
            let data = sample
                .data
                .iter()
                .enumerate()
                .map(|(i, &x)| (x - self.mean[i]) / self.std[i])
                .collect();
            transformed.push(Sample {
                target: sample.target,
                data: Arc::new(data),
            });
        }
        transformed
    }
    pub fn fit_transform(&mut self, samples: &[Sample]) -> Vec<Sample> {
        self.fit(samples);
        let transformed = self.transform(samples);
        transformed
    }
}
