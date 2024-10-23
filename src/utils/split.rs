#![allow(dead_code)]

use super::structures::Sample;
use rand::{seq::SliceRandom, SeedableRng};
use std::{
    cmp::{max, min},
    mem::swap,
};

pub fn train_test_split(
    data: &[Sample],
    test_size: f64,
    stratify: bool,
    random_state: Option<rand_chacha::ChaCha8Rng>,
) -> (Vec<Sample>, Vec<Sample>) {
    if data.len() < 2 && (test_size > 0. && test_size < 0.5) {
        panic!("The dataset is too small to be splitted.");
    }
    let mut indices: Vec<usize> = (0..data.len()).collect();
    let mut random_state =
        random_state.unwrap_or(rand_chacha::ChaCha8Rng::from_rng(rand::thread_rng()).unwrap());
    // Shuffle indices
    indices.shuffle(&mut random_state);

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
    train_data.shuffle(&mut random_state);
    test_data.shuffle(&mut random_state);
    (train_data, test_data)
}

use hashbrown::HashMap;
use rand::Rng;
use std::ops::Range;

use crate::tree::fast_gini::FastGini;
use crate::tree::tree::{SplitParameters, StandardSplit};
use crate::RandomGenerator;

pub fn get_random_split(
    samples: &mut [Sample],
    non_constant_features: &mut Vec<usize>,
    random_state: &mut RandomGenerator,
    min_samples_leaf: usize,
) -> Option<(Vec<Range<usize>>, StandardSplit, f64)> {
    non_constant_features.shuffle(random_state);

    while let Some(feature) = non_constant_features.pop() {
        let thresholds = samples
            .iter()
            .map(|f| f.features[feature])
            .collect::<Vec<_>>();

        let min_feature = *thresholds
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let max_feature = *thresholds
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        if max_feature - min_feature <= f64::EPSILON {
            // Remove constant features
            continue;
        } else {
            let threshold = random_state.gen_range(min_feature..max_feature);
            let rand_split = StandardSplit { feature, threshold };

            let mut start = 0;
            let mut end = samples.len();
            while start < end {
                if rand_split.split(&samples[start]) == 0 {
                    start += 1;
                } else {
                    samples.swap(start, end - 1);
                    end -= 1;
                }
            }

            if start < min_samples_leaf || (samples.len() - start) < min_samples_leaf {
                continue;
            }

            non_constant_features.push(feature);

            return Some((vec![0..start, start..samples.len()], rand_split, f64::NAN));
        }
    }
    return None;
}
pub fn get_best_split(
    samples: &mut [Sample],
    non_constant_features: &mut Vec<usize>,
    min_samples_leaf: usize,
    max_features: usize,
    random_state: &mut RandomGenerator,
) -> Option<(Vec<Range<usize>>, StandardSplit, f64)> {
    let mut current_feature_count = 0;
    let mut max_gain = f64::NEG_INFINITY;
    let mut best_split = None;
    let mut best_split_index = 0;

    let mut parent_impurity =
        FastGini::from_classes_count(samples.iter().fold(HashMap::new(), |mut acc, x| {
            *acc.entry(x.target).or_insert(0) += 1;
            acc
        }));
    non_constant_features.shuffle(random_state);
    non_constant_features.retain(|&feature| {
        if current_feature_count >= max_features {
            return true;
        }

        let min_feature = samples
            .iter()
            .min_by(|a, b| {
                a.features[feature]
                    .partial_cmp(&b.features[feature])
                    .unwrap()
            })
            .unwrap()
            .features[feature];

        let max_feature = samples
            .iter()
            .max_by(|a, b| {
                a.features[feature]
                    .partial_cmp(&b.features[feature])
                    .unwrap()
            })
            .unwrap()
            .features[feature];

        if max_feature - min_feature <= f64::EPSILON {
            // Remove constant features
            return false;
        }

        samples.sort_unstable_by(|a, b| {
            a.features[feature]
                .partial_cmp(&b.features[feature])
                .unwrap()
        });
        let mut thresholds = samples
            .iter()
            .map(|f| f.features[feature])
            .collect::<Vec<_>>();

        thresholds.dedup();

        let mut split_index = 0;
        let mut children_count = [FastGini::new(), parent_impurity.clone()];
        let mut left_count = 0;
        let mut right_count = samples.len();

        for &threshold in &thresholds {
            let current_split = StandardSplit { feature, threshold };

            while split_index < samples.len() && current_split.split(&samples[split_index]) == 1 {
                children_count[1].change_element(samples[split_index].target, -1);
                right_count -= 1;
                children_count[0].change_element(samples[split_index].target, 1);
                left_count += 1;
                split_index += 1;
            }

            if split_index < min_samples_leaf || (samples.len() - split_index) < min_samples_leaf {
                continue;
            }

            let current_gain = parent_impurity.get_gini()
                - (children_count[0].get_gini() * left_count as f64
                    + children_count[1].get_gini() * right_count as f64)
                    / samples.len() as f64;

            if current_gain > max_gain {
                max_gain = current_gain;
                best_split = Some((current_split, max_gain));
                best_split_index = split_index;
            }
        }
        current_feature_count += 1;

        return true;
    });

    let best_split = best_split?;

    // Reorder according to the split
    let mut start = 0;
    let mut end = samples.len();
    while start < end {
        if best_split.0.split(&samples[start]) == 0 {
            start += 1;
        } else {
            samples.swap(start, end - 1);
            end -= 1;
        }
    }
    let best_split_index = start;

    Some((
        vec![0..best_split_index, best_split_index..samples.len()],
        best_split.0,
        best_split.1,
    ))
}
