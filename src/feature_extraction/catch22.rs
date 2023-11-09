use crate::feature_extraction::ts_features::histcounts;
use crate::feature_extraction::ts_features::{mean, median, max};

pub fn dn_histogram_mode_n(x: &Vec<f64>, n_bins: usize) ->f64{

    let (hist_counts, bin_edges) = histcounts(x, n_bins);

    let mut max_count = 0;
    let mut num_maxs = 1;
    let mut dn_histogram_mode_10= 0.0;

    for i in 0..n_bins {
        if hist_counts[i] > max_count {
            max_count = hist_counts[i];
            num_maxs = 1;
            dn_histogram_mode_10 = (bin_edges[i] + bin_edges[i + 1]) * 0.5;
        } else if hist_counts[i] == max_count {
            num_maxs += 1;
            dn_histogram_mode_10 += (bin_edges[i] + bin_edges[i + 1]) * 0.5;
        }
    }
    dn_histogram_mode_10 /= num_maxs as f64;
    assert!(dn_histogram_mode_10.is_finite());
    dn_histogram_mode_10
}

pub fn dn_outlier_include_np_001_mdrmd(x: &Vec<f64>, sign: f64) -> f64{

    let inc = 0.01;
    let mut tot = 0;
    let mut x_work: Vec<f64> = x.iter().map(|&val| sign * val).collect();

    // Apply sign and check constant time series
    let constant_flag = x.iter().all(|&val| val == x[0]);
    if constant_flag {
        return 0.0; // if constant, return 0
    }

    // Count pos/negs and find max value
    for &val in &x_work {
        if val >= 0.0 {
            tot += 1;
        }
    }

    // Maximum value too small? Return 0
    let max_val = max(&x_work);
    if max_val < inc {
        return 0.0;
    }

    let n_thresh = (max_val / inc + 1.0) as usize;

    // Save the indices where y > threshold
    let mut r: Vec<f64> = Vec::with_capacity(x.len());

    // Save the median over indices with absolute value > threshold
    let mut ms_dti1 = vec![0.0; n_thresh];
    let mut ms_dti3 = vec![0.0; n_thresh];
    let mut ms_dti4 = vec![0.0; n_thresh];

    for j in 0..n_thresh {
        let mut high_size = 0;

        for (i, &val) in x_work.iter().enumerate() {
            if val >= j as f64 * inc {
                r.push((i + 1) as f64);
                high_size += 1;
            }
        }

        // Intervals between high-values
        let mut dt_exc: Vec<f64> = Vec::with_capacity(high_size);
        for i in 0..high_size - 1 {
            dt_exc.push(r[i as usize + 1] - r[i as usize]);
        }

        ms_dti1[j] = mean(&dt_exc);
        ms_dti3[j] = (high_size - 1) as f64 * 100.0 / tot as f64;
        ms_dti4[j] = median(&mut r) / (x.len() as f64 / 2.0) - 1.0;

        r.clear();
    }

    let trim_thr = 2;
    let mut mj = 0;
    let mut fbi = n_thresh - 1;

    for i in 0..n_thresh {
        if ms_dti3[i] > trim_thr as f64 {
            mj = i;
        }
        if ms_dti1[n_thresh - 1 - i].is_nan() {
            fbi = n_thresh - 1 - i;
        }
    }

    let trim_limit = mj.min(fbi);
    let output_scalar = median(&ms_dti4[0..trim_limit].to_vec());

    output_scalar
}

pub fn dn_outlier_include_p_001_mdrmd(x: &Vec<f64>) -> f64
{
    dn_outlier_include_np_001_mdrmd(x, 1.0)
}

pub fn dn_outlier_include_n_001_mdrmd(x: &Vec<f64>) -> f64
{
    dn_outlier_include_np_001_mdrmd(x, -1.0)
}

fn dn_outlier_include_abs_001(x: &Vec<f64>) -> f64 {
    let inc = 0.01;
    let mut max_abs = 0.0;
    let mut x_abs: Vec<f64> = Vec::with_capacity(x.len());

    for &val in x {
        x_abs.push(if val > 0.0 { val } else { -val });

        if x_abs.last().unwrap() > &max_abs {
            max_abs = *x_abs.last().unwrap();
        }
    }

    let n_thresh = (max_abs / inc + 1.0) as usize;

    // Save the indices where y > threshold
    let mut high_inds: Vec<f64> = Vec::with_capacity(x.len());

    // Save the median over indices with absolute value > threshold
    let mut ms_dti3 = vec![0.0; n_thresh];
    let mut ms_dti4 = vec![0.0; n_thresh];

    for j in 0..n_thresh {
        let mut high_size = 0;

        for (i, &val) in x_abs.iter().enumerate() {
            if val >= j as f64 * inc {
                high_inds.push(i as f64);
                high_size += 1;
            }
        }

        // Median
        let median_out = median(&high_inds);

        ms_dti3[j] = (high_size - 1) as f64 * 100.0 / x.len() as f64;
        ms_dti4[j] = median_out / (x.len() as f64 / 2.0) - 1.0;
    }

    let trim_thr = 2.0;
    let mut mj = 0;

    for i in 0..n_thresh {
        if ms_dti3[i] > trim_thr {
            mj = i;
        }
    }

    // Calculate output scalar
    let output_scalar = median(&ms_dti4[0..mj].to_vec());

    output_scalar
}

fn dn_spread_std(x: &Vec<f64>) -> f64 {
    let m = mean(x);
    let mut sd = 0.0;

    for &val in x {
        sd += (val - m).powi(2);
    }

    sd = (sd / (x.len() - 1) as f64).sqrt();
    sd
}