use std::ops::{Add, Div};

use hashbrown::HashMap;

pub const EULER_MASCHERONI: f64 =
    0.5772156649015328606065120900824024310421593359399235988057672348849;

pub fn mean(x: &[f64]) -> f64 {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    assert!(mean.is_finite(), "{:?}", x);
    mean
}

pub fn max(x: &[f64]) -> f64 {
    let max = x.iter().fold(f64::MIN, |max, &val| max.max(val));
    assert!(max.is_finite());
    max
}

pub fn min(x: &[f64]) -> f64 {
    let min = x.iter().fold(f64::MAX, |min, &val| min.min(val));
    assert!(min.is_finite());
    min
}

pub fn sum(x: Vec<Vec<f64>>, axis: usize) -> Vec<f64> {
    if axis == 0 {
        let mut sum = vec![0.0; x[0].len()];
        for i in 0..x.len() {
            for j in 0..x[i].len() {
                sum[j] += x[i][j];
            }
        }
        sum
    } else if axis == 1{
        let mut sum = vec![0.0; x.len()];
        for i in 0..x.len() {
            sum[i] = x[i].iter().sum();
        }
        sum
    } else {
        panic!("Axis must be either 0 or 1")
    }
}

pub fn unique<T: PartialOrd + Clone>(x: &[T]) -> Vec<T> {
    let mut unique = x.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup();
    unique
}

pub fn diff<T: std::ops::Sub<Output = T> + Clone>(x: &[T]) -> Vec<T> {
    let mut diff = Vec::new();
    for i in 0..x.len() - 1 {
        diff.push(x[i + 1].clone() - x[i].clone());
    }
    diff
}

pub fn median<T>(x: &[T]) -> T
where
    T: Add<Output = T> + Div<f64, Output = T> + Clone + PartialOrd + Copy,
{
    let mut x = x.to_vec();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if x.len() % 2 == 0 {
        (x[x.len() / 2 - 1] + x[x.len() / 2]) / 2.0
    } else {
        x[x.len() / 2]
    };
    median
}

pub fn stddev(x: &[f64]) -> f64 {
    let mean = mean(x);
    let std = (x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
    assert!(std.is_finite());
    std
}

pub fn percentile(x: &[f64], p: usize) -> f64 {
    let mut x = x.to_vec();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p_ind = (x.len() as f64 * p as f64 / 100.0).floor() as usize;
    let percentile = x[p_ind];
    assert!(percentile.is_finite());
    percentile
}

pub fn slope(x: &[f64]) -> f64 {
    let n = x.len();

    let x_mean = x.iter().sum::<f64>() / n as f64;

    let y = (1..n + 1).map(|x| x as f64).collect::<Vec<f64>>();
    let y_mean = y.iter().sum::<f64>() / n as f64;

    let xy_mean = x.iter().zip(y.iter()).map(|(x, y)| x * y).sum::<f64>() / n as f64;
    let y2_mean = y.iter().map(|y| y.powi(2)).sum::<f64>() / n as f64;

    let slope = (xy_mean - x_mean * y_mean) / (y2_mean - y_mean.powi(2));
    assert!(slope.is_finite());
    slope
}

pub fn histcounts(x: &[f64], n_bins: usize) -> (Vec<i32>, Vec<f64>) {
    // Check min and max of input arrax
    let (min_val, max_val) = x.iter().fold((f64::MAX, f64::MIN), |(min, max), &val| {
        (min.min(val), max.max(val))
    });

    // Derive bin width from it
    let bin_step = (max_val - min_val) / n_bins as f64;

    // Variable to store counted occurrences in
    let mut bin_counts = vec![0; n_bins as usize];

    for val in x {
        let bin_ind = ((val - min_val) / bin_step).floor() as usize;
        bin_counts[bin_ind.min(n_bins as usize - 1)] += 1;
    }

    // Calculate bin edges
    let bin_edges: Vec<f64> = (0..=n_bins)
        .map(|i| i as f64 * bin_step + min_val)
        .collect();

    (bin_counts, bin_edges)
}

pub fn quantile(x: &[f64], q: f64) -> f64 {
    let mut x = x.to_vec();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q_ind = (x.len() as f64 * q).floor() as usize;
    x[q_ind]
}

pub fn argsort<T: PartialOrd>(data: &[T], order: &str) -> Vec<usize> {
    match order {
        "asc" => {
            let mut indices: Vec<usize> = (0..data.len()).collect();
            indices.sort_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap());
            indices
        }
        "desc" => {
            let mut indices: Vec<usize> = (0..data.len()).collect();
            indices.sort_by(|&i, &j| data[j].partial_cmp(&data[i]).unwrap());
            indices
        }
        _ => panic!("Order must be either 'asc' or 'desc'"),
    }
}

pub fn cumsum(x: &[usize]) -> Vec<usize> {
    let mut cumsum = Vec::new();
    let mut sum = 0;
    for &val in x {
        sum += val;
        cumsum.push(sum);
    }
    cumsum
}

pub fn cov(x: &[f64], y: &[f64]) -> f64 {
    let x_mean = mean(x);
    let y_mean = mean(y);

    let cov = x
        .iter()
        .zip(y.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>()
        / x.len() as f64;
    assert!(cov.is_finite());
    cov
}

pub fn cov_mean(x: &[f64], y: &[f64], lag: usize) -> f64 {
    let mut cov = 0.0;
    for i in 0..x.len() - lag {
        cov += x[i] * y[lag + i];
    }
    assert!(cov.is_finite());
    cov
}

pub fn autocov_lag(x: &[f64], lag: usize) -> f64 {
    cov_mean(x, x, lag)
}

pub fn corr(x: &[f64], y: &[f64], lag: usize) -> f64 {
    let size = x.len();

    let mean_x = mean(x);
    let mean_y = mean(&y[lag..]);

    let (mut nom, mut denom_x, mut denom_y) = (0.0, 0.0, 0.0);

    for i in 0..size - lag {
        nom += (x[i] - mean_x) * (y[lag + i] - mean_y);
        denom_x += (x[i] - mean_x).powi(2);
        denom_y += (y[lag + i] - mean_y).powi(2);
    }

    nom / (denom_x * denom_y).sqrt()
}

pub fn autocorr_lag(x: &[f64], lag: usize) -> f64 {
    corr(x, x, lag)
}

pub fn f_entropy(a: &[f64]) -> f64 {
    let mut f = 0.0;
    for &val in a.iter() {
        if val > 0.0 {
            f += val * val.ln();
        }
    }
    return -1.0 * f;
}

pub fn zscore(x: &[f64]) -> Vec<f64> {
    let mean = mean(x);
    let std = stddev(x) + f64::EPSILON;
    x.iter().map(|x| (x - mean) / std).collect()
}

pub fn value_counts(x: &[isize]) -> HashMap<isize, usize> {
    let mut counts = HashMap::new();
    for &val in x {
        *counts.entry(val).or_insert(0) += 1;
    }
    counts
}

pub fn fisher_score(
    x: &Vec<f64>,
    y: &Vec<isize>,
    classes: &Vec<isize>,
    class_counts: &HashMap<isize, usize>,
) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;

    let x_mean = mean(x);

    for cls in classes.iter() {
        let x_cls = x
            .iter()
            .enumerate()
            .filter_map(|(j, &val)| if y[j] == *cls { Some(val) } else { None })
            .collect::<Vec<f64>>();
        let x_cls_mean = mean(&x_cls);
        let x_cls_std = stddev(&x_cls);

        num += class_counts[cls] as f64 * (x_cls_mean - x_mean).powi(2);
        den += class_counts[cls] as f64 * x_cls_std.powi(2);
    }

    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

pub fn transpose(x: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut transposed = vec![vec![0.0; x.len()]; x[0].len()];
    for i in 0..x.len() {
        for j in 0..x[i].len() {
            transposed[j][i] = x[i][j];
        }
    }
    transposed
}
