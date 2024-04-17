use std::{
    cmp::{max, min},
    mem::swap,
};

use dashmap::DashMap;
use lazy_static::lazy_static;
use rayon::vec;

const MSM_C: f64 = 1.0;

use crate::utils::float_handling::FloatEq;

pub fn euclidean(x1: &[f64], x2: &[f64]) -> f64 {
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
    let result = twe(&s1, &s2);
    println!("TWE: {}, time: {:?}", result, start.elapsed());
}

lazy_static! {
    static ref TWE_CACHE: DashMap<(Vec<FloatEq>, Vec<FloatEq>), f64> = DashMap::new();
}
pub fn twe(x1: &[f64], x2: &[f64]) -> f64 {
    // TWE_CACHE
    let x1_cache = x1.iter().copied().map(FloatEq).collect::<Vec<_>>();
    let x2_cache = x2.iter().copied().map(FloatEq).collect::<Vec<_>>();
    let mut key_cache = (x1_cache, x2_cache);

    if let Some(value) = TWE_CACHE.get(&key_cache) {
        return *value.value();
    }

    let nu = 0.001;
    let lambda = 1.0;

    let n = x1.len();
    let m = x2.len();

    let delete_addition = nu + lambda;
    let sakoe_chiba = 1.0;
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
    if TWE_CACHE.len() > 1e6 as usize {
        return distance;
    }
    TWE_CACHE.insert(key_cache.clone(), distance);
    swap(&mut key_cache.0, &mut key_cache.1);
    TWE_CACHE.insert(key_cache, distance);

    distance
}

#[test]
pub fn test_dtw() {
    let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let s2 = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    let result = dtw(&s1, &s2);
    println!("DTW: {}", result);
}

lazy_static! {
    static ref DTW_CACHE: DashMap<(Vec<FloatEq>, Vec<FloatEq>), f64> = DashMap::new();
}
pub fn dtw(x1: &[f64], x2: &[f64]) -> f64 {
    // DTW_CACHE
    let x1_cache = x1.iter().copied().map(FloatEq).collect::<Vec<_>>();
    let x2_cache = x2.iter().copied().map(FloatEq).collect::<Vec<_>>();
    let mut key_cache = (x1_cache, x2_cache);

    if let Some(value) = DTW_CACHE.get(&key_cache) {
        return *value.value();
    }

    let n = x1.len();
    let m = x2.len();

    let sakoe_chiba = 1.0;
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

    if DTW_CACHE.len() > 1e6 as usize {
        return distance;
    }
    DTW_CACHE.insert(key_cache.clone(), distance);
    swap(&mut key_cache.0, &mut key_cache.1);
    DTW_CACHE.insert(key_cache, distance);
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

    // MSM_CACHE
    let x1_cache = x1.iter().copied().map(FloatEq).collect::<Vec<_>>();
    let x2_cache = x2.iter().copied().map(FloatEq).collect::<Vec<_>>();
    let mut key_cache = (x1_cache, x2_cache);

    if let Some(value) = MSM_CACHE.get(&key_cache) {
        return *value.value();
    }

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
    let distance = previous[m-1];

    DTW_CACHE.insert(key_cache.clone(), distance);
    swap(&mut key_cache.0, &mut key_cache.1);
    DTW_CACHE.insert(key_cache, distance);

    return distance;
}
fn msm_cost_function(x_i: f64, x_i_1: f64, y_j: f64) -> f64 {
    if (x_i >= x_i_1 && x_i <= y_j) || (x_i_1 >= x_i && x_i >= y_j) {
        MSM_C
    } else {
        MSM_C + (x_i - x_i_1).abs().min((x_i - y_j).abs())
    }
}
