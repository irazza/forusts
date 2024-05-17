#![allow(dead_code)]
use dashmap::DashMap;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::{
    cmp::{max, min},
    mem::swap,
};

const MSM_C: f64 = 1.0;

use crate::utils::{float_handling::FloatEq, structures::Sample};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum Distance {
    Euclidean,
    DTW,
    TWE,
    MSM,
    ADTW,
}
impl Distance {
    pub fn distance(&self, x1: &[f64], x2: &[f64], band: f64) -> f64 {
        match self {
            Distance::Euclidean => euclidean(x1, x2, band),
            Distance::DTW => dtw(x1, x2, band),
            Distance::TWE => twe(x1, x2, band),
            Distance::MSM => msm(x1, x2, band),
            Distance::ADTW => adtw(x1, x2, band),
        }
    }
    pub fn to_fn(&self) -> fn(&[f64], &[f64], f64) -> f64 {
        match self {
            Distance::Euclidean => euclidean,
            Distance::DTW => dtw,
            Distance::TWE => twe,
            Distance::MSM => msm,
            Distance::ADTW => adtw,
        }
    }
}
impl std::fmt::Display for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Distance::Euclidean => write!(f, "eucl"),
            Distance::DTW => write!(f, "dtw"),
            Distance::TWE => write!(f, "twe"),
            Distance::MSM => write!(f, "msm"),
            Distance::ADTW => write!(f, "adtw"),
        }
    }
}

pub fn euclidean(x1: &[f64], x2: &[f64], _band: f64) -> f64 {
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

pub fn find_optimal_band(
    ds: &[Sample],
    dist_fn: fn(&[f64], &[f64], f64) -> f64,
    band_values: &[f64],
) -> f64 {
    let mut best_band = 0.0;
    let mut best_error = f64::MAX;

    //let band_values = (0..=100).step_by(10).map(|x| x as f64 / 100.0); // adjust as needed
    // println!("Band values: {:?}", band_values.clone().collect::<Vec<_>>());
    for band in band_values {
        let mut total_error = 0.0;
        for (i, sample) in ds.iter().enumerate() {
            let mut rest_ds = ds.to_vec();
            rest_ds.remove(i);
            let mut distances = rest_ds
                .iter()
                .map(|other_sample| {
                    (
                        dist_fn(&sample.data, &other_sample.data, *band),
                        other_sample.target,
                    )
                })
                .collect::<Vec<_>>();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let predicted_label = distances[0].1;
            let error = if predicted_label == sample.target {
                0.0
            } else {
                1.0
            };
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

pub fn twe(x1: &[f64], x2: &[f64], band: f64) -> f64 {
    let nu = 0.001;
    let lambda = 1.0;

    let n = x1.len();
    let m = x2.len();

    let delete_addition = nu + lambda;
    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

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
pub fn dtw(x1: &[f64], x2: &[f64], band: f64) -> f64 {
    // // DTW_CACHE
    // let x1_cache = x1.iter().copied().map(FloatEq).collect::<Vec<_>>();
    // let x2_cache = x2.iter().copied().map(FloatEq).collect::<Vec<_>>();
    // let mut key_cache = (x1_cache, x2_cache);

    // if let Some(value) = DTW_CACHE.get(&key_cache) {
    //     return *value.value();
    // }

    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

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
    let result = msm(&s1, &s2, 1.0);
    println!("MSM: {}", result);
}

lazy_static! {
    static ref MSM_CACHE: DashMap<(Vec<FloatEq>, Vec<FloatEq>), f64> = DashMap::new();
}
pub fn msm(x1: &[f64], x2: &[f64], band: f64) -> f64 {
    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba_window_radius = (n as f64 + 1.0) * band;

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


pub fn test_adtw() {
    let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let s2 = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    let result = adtw(&s1, &s2, 1.0);
    println!("ADTW: {}", result);
}

lazy_static! {
    static ref ADTW_CACHE: DashMap<(Vec<FloatEq>, Vec<FloatEq>), f64> = DashMap::new();
}
pub fn adtw(x1: &[f64], x2: &[f64], band: f64) -> f64 {
    // DTW_CACHE
    let x1_cache = x1.iter().copied().map(FloatEq).collect::<Vec<_>>();
    let x2_cache = x2.iter().copied().map(FloatEq).collect::<Vec<_>>();
    let mut key_cache = (x1_cache, x2_cache);

    if let Some(value) = ADTW_CACHE.get(&key_cache) {
        return *value.value();
    }
    
    let n = x1.len();
    let m = x2.len();
    let w = band;

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

    if ADTW_CACHE.len() > 1e6 as usize {
        return distance;
    }
    ADTW_CACHE.insert(key_cache.clone(), distance);
    swap(&mut key_cache.0, &mut key_cache.1);
    ADTW_CACHE.insert(key_cache, distance);

    distance
}
