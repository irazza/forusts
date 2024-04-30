use std::{
    cmp::{max, min},
    mem::swap,
};

use dashmap::DashMap;
use lazy_static::lazy_static;
use rand::{thread_rng, Rng};
use rayon::vec;

const MSM_C: f64 = 1.0;

use crate::{
    metrics::classification::{self, accuracy_score},
    utils::{
        float_handling::FloatEq,
        structures::{train_test_split, Sample},
    },
};

pub fn euclidean(x1: &[f64], x2: &[f64], _sakoe_chiba: f64) -> f64 {
    assert!(x1.len() == x2.len());
    x1.iter()
        .zip(x2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
pub fn test_twe() {
    let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let s2 = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    // start time
    let start = std::time::Instant::now();
    let result = twe(&s1, &s2, 1.0);
    println!("TWE: {}, time: {:?}", result, start.elapsed());
}

pub fn find_sakoe_chiba_band(x: &[Sample]) -> f64 {
    // Using k=1NN to find the optimal band
    let mut best_band = 0.0;
    let mut band = 0.0;
    let max_band = 1.0;
    let mut best_accuracy = 0.0;

    let (ds_train, ds_test) = train_test_split(x, 0.2, false, None);
    let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();

    while band <= max_band {
        // For every test sample, find the nearest neighbor in the training set
        let mut y_pred = Vec::new();
        for test_sample in ds_test.iter() {
            let mut distances = vec![0.0; ds_train.len()];
            for train_sample in ds_train.iter() {
                distances.push(twe(&test_sample.data, &train_sample.data, band));
            }
            let min_distance = distances
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            y_pred.push(ds_train[min_distance.0].target);
        }
        let accuracy = accuracy_score(&y_pred, &y_true);
        if accuracy > best_accuracy {
            best_accuracy = accuracy;
            best_band = band;
        }
        band += 0.1;
    }
    best_band
}

pub fn find_optimal_band(ds: &[Sample], dist_fn: fn(&[f64], &[f64], f64) -> f64, band_values: &[f64]) -> f64 {
    let mut best_band = 0.0;
    let mut best_error = f64::MAX;

    //let band_values = (0..=100).step_by(10).map(|x| x as f64 / 100.0); // adjust as needed
    // println!("Band values: {:?}", band_values.clone().collect::<Vec<_>>());
    for band in band_values {
        let mut total_error = 0.0;
        for (i, sample) in ds.iter().enumerate() {
            let mut rest_ds = ds.to_vec();
            rest_ds.remove(i);
            let mut distances = rest_ds.iter()
                .map(|other_sample| (dist_fn(&sample.data, &other_sample.data, *band), other_sample.target))
                .collect::<Vec<_>>();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let predicted_label = distances[0].1;
            let error = if predicted_label == sample.target { 0.0 } else { 1.0 };
            total_error += error;
        }
        let mean_error = total_error / ds.len() as f64;
        if mean_error < best_error {
            best_error = mean_error;
            best_band = *band;
        }
    }

    best_band
}

pub fn twe(x1: &[f64], x2: &[f64], sakoe_chiba: f64) -> f64 {
    let nu = 0.001;
    let lambda = 1.0;

    let n = x1.len();
    let m = x2.len();

    let delete_addition = nu + lambda;
    let sakoe_chiba_window_radius = (n as f64 + 1.0) * sakoe_chiba;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut current = vec![f64::INFINITY; m + 1];
    let mut previous = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;

    for i in 1..=n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(f64::INFINITY);

        // closure to handle the insertion of zeros as padding in the first element of the timeseries
        let x1 = |i| if i == 0 { 0.0 } else { x1[i - 1] };
        let x2 = |i| if i == 0 { 0.0 } else { x2[i - 1] };

        for j in lower..=upper {
            // deletion in x1
            let deletion_x1_euclidean_dist = (x1(i - 1) - x2(i)).abs();
            let del_x1: f64 = previous[j] + deletion_x1_euclidean_dist + delete_addition;

            // deletion in x2
            let deletion_x2_euclidean_dist = (x1(j - 1) - x2(j)).abs();
            let del_x2 = current[j - 1] + deletion_x2_euclidean_dist + delete_addition;

            // match
            let match_same_euclid_dist = (x1(i) - x2(j)).abs();
            let match_previous_euclid_dist = (x1(i - 1) - x2(j - 1)).abs();

            let match_x1_x2 = previous[j - 1]
                + match_same_euclid_dist
                + match_previous_euclid_dist
                + (nu * (2.0 * (i as isize - j as isize).abs() as f64));

            current[j] = del_x1.min(del_x2.min(match_x1_x2));
        }
        swap(&mut previous, &mut current);
    }
    let distance = previous[m];
    distance
}

#[test]
pub fn test_dtw() {
    let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let s2 = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    let result = dtw(&s1, &s2, 1.0);
    println!("DTW: {}", result);
}

// lazy_static! {
//     static ref DTW_CACHE: DashMap<(Vec<FloatEq>, Vec<FloatEq>), f64> = DashMap::new();
// }
pub fn dtw(x1: &[f64], x2: &[f64], sakoe_chiba: f64) -> f64 {
    // // DTW_CACHE
    // let x1_cache = x1.iter().copied().map(FloatEq).collect::<Vec<_>>();
    // let x2_cache = x2.iter().copied().map(FloatEq).collect::<Vec<_>>();
    // let mut key_cache = (x1_cache, x2_cache);

    // if let Some(value) = DTW_CACHE.get(&key_cache) {
    //     return *value.value();
    // }

    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba_window_radius = (n as f64 + 1.0) * sakoe_chiba;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut current = vec![f64::INFINITY; m + 1];
    let mut previous = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;

    for i in 1..=n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(f64::INFINITY);

        // closure to handle the insertion of zeros as padding in the first element of the timeseries
        let x1 = |i| if i == 0 { 0.0 } else { x1[i - 1] };
        let x2 = |i| if i == 0 { 0.0 } else { x2[i - 1] };

        for j in lower..=upper {
            let dist = (x1(i) - x2(j)).powi(2);
            current[j] = dist + previous[j].min(current[j - 1].min(previous[j - 1]));
        }
        swap(&mut previous, &mut current);
    }

    let distance = previous[m].sqrt();

    // if DTW_CACHE.len() > 1e6 as usize {
    //     return distance;
    // }
    // DTW_CACHE.insert(key_cache.clone(), distance);
    // swap(&mut key_cache.0, &mut key_cache.1);
    // DTW_CACHE.insert(key_cache, distance);
    distance
}

#[test]
pub fn test_msm() {
    let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let s2 = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    let result = msm(&s1, &s2);
    println!("MSM: {}", result);
}

lazy_static! {
    static ref MSM_CACHE: DashMap<(Vec<FloatEq>, Vec<FloatEq>), f64> = DashMap::new();
}
pub fn msm(x1: &[f64], x2: &[f64]) -> f64 {
    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba = 1.0;
    let sakoe_chiba_window_radius = (n as f64 + 1.0) * sakoe_chiba;

    let alpha = ((m) as f64) / ((n) as f64);

    let mut previous = vec![0.0; m];
    let mut current = vec![0.0; m];
    previous[0] = (x1[0] - x2[0]).abs();
    for j in 1..m {
        previous[j] = previous[j - 1] + msm_cost_function(x2[j], x1[0], x2[j - 1]);
    }

    for i in 1..n {
        let lower = alpha * (i as f64) - sakoe_chiba_window_radius;
        let upper = alpha * (i as f64) + sakoe_chiba_window_radius;

        let lower = max(1, lower.ceil() as usize);
        let upper = min(m, upper.floor() as usize);

        current[..].fill(f64::INFINITY);

        current[0] = previous[0] + msm_cost_function(x1[i], x1[i - 1], x2[0]);
        for j in lower..upper {
            current[j] = (previous[j - 1] + (x1[i] - x2[j]).abs())
                .min(previous[j] + msm_cost_function(x1[i], x1[i - 1], x2[j]))
                .min(current[j - 1] + msm_cost_function(x2[j], x1[i], x2[j - 1]));
        }
        swap(&mut previous, &mut current);
    }
    let distance = previous[m - 1];

    return distance;
}
fn msm_cost_function(x_i: f64, x_i_1: f64, y_j: f64) -> f64 {
    if (x_i >= x_i_1 && x_i <= y_j) || (x_i_1 >= x_i && x_i >= y_j) {
        MSM_C
    } else {
        MSM_C + (x_i - x_i_1).abs().min((x_i - y_j).abs())
    }
}

pub fn adtw(x1: &[f64], x2: &[f64], w: f64) -> f64 {
    let n = x1.len();
    let m = x2.len();

    let mut current = vec![f64::INFINITY; m + 1];
    let mut previous = vec![f64::INFINITY; m + 1];
    previous[0] = 0.0;

    for i in 1..=n {
        // closure to handle the insertion of zeros as padding in the first element of the timeseries
        let x1 = |i| if i == 0 { 0.0 } else { x1[i - 1] };
        let x2 = |i| if i == 0 { 0.0 } else { x2[i - 1] };

        for j in 1..=m {
            let dist = (x1(i) - x2(j)).powi(2);
            current[j] = dist + (previous[j] + w).min((current[j - 1] + w).min(previous[j - 1]));
        }
        swap(&mut previous, &mut current);
    }

    let distance = previous[m].sqrt();
    distance
}
