#![allow(dead_code,unused_variables, unused_mut)]
use crate::feature_extraction::statistics::histcounts;
use crate::feature_extraction::statistics::{
    autocorr_lag, autocov_lag, cov, diff, f_entropy, max, mean, median, quantile,
};

pub fn dn_histogram_mode_n(x: &[f64], n_bins: usize) -> f64 {
    let (hist_counts, bin_edges) = histcounts(x, n_bins);

    let mut max_count = 0;
    let mut num_maxs = 1;
    let mut dn_histogram_mode_10 = 0.0;

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

pub fn dn_outlier_include_np_001_mdrmd(x: &[f64], sign: f64) -> f64 {
    let inc = 0.01;
    let mut tot = 0;
    let x_work: Vec<f64> = x.iter().map(|&val| sign * val).collect();

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

pub fn dn_outlier_include_p_001_mdrmd(x: &[f64]) -> f64 {
    dn_outlier_include_np_001_mdrmd(x, 1.0)
}

pub fn dn_outlier_include_n_001_mdrmd(x: &[f64]) -> f64 {
    dn_outlier_include_np_001_mdrmd(x, -1.0)
}

fn dn_outlier_include_abs_001(x: &[f64]) -> f64 {
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

fn dn_spread_std(x: &[f64]) -> f64 {
    let m = mean(x);
    let mut sd = 0.0;

    for &val in x {
        sd += (val - m).powi(2);
    }

    sd = (sd / (x.len() - 1) as f64).sqrt();
    sd
}

pub fn md_hrv_classic_pnn40(x: &[f64]) -> f64 {
    let pnnx = 40;

    // compute diff
    let dx = diff(x, x.len());

    let mut pnn40 = 0;
    for &diff_value in &dx {
        if (diff_value.abs() * 1000.0) > pnnx as f64 {
            pnn40 += 1;
        }
    }

    pnn40 as f64 / (x.len() - 1) as f64
}

pub fn sb_binary_stats_mean_longstretch1(x: &[f64]) -> i32 {
    // binarize
    let mut x_bin = vec![0; x.len() - 1];
    let x_mean = mean(x);
    for i in 0..x.len() - 1 {
        x_bin[i] = if x[i] - x_mean <= 0.0 { 0 } else { 1 };
    }

    let mut max_stretch1 = 0;
    let mut last1 = 0;
    for i in 0..x.len() - 1 {
        if x_bin[i] == 0 || i == x.len() - 2 {
            let stretch1 = i - last1;
            if stretch1 > max_stretch1 {
                max_stretch1 = stretch1;
            }
            last1 = i;
        }
    }

    max_stretch1 as i32
}

fn co_firstzero(x: &[f64], maxtau: usize) -> usize {
    let autocorrs = vec![0.0, 1.0, 2.0]; //co_autocorrs(x);

    let mut zerocrossind = 0;
    while autocorrs[zerocrossind] > 0.0 && zerocrossind < maxtau {
        zerocrossind += 1;
    }

    zerocrossind
}

fn sb_coarsegrain(y: &[f64], num_groups: usize, labels: &mut [i32]) {
    let mut th = vec![0.0; num_groups + 1];

    let ls: Vec<usize> = (0..num_groups + 1).step_by(1).collect();

    for i in 0..=num_groups {
        th[i] = quantile(y, ls[i] as f64);
    }

    th[0] -= 1.0;

    for i in 0..num_groups {
        for j in 0..y.len() {
            if y[j] > th[i] && y[j] <= th[i + 1] {
                labels[j] = (i + 1) as i32;
            }
        }
    }
}

pub fn sb_transition_matrix_3ac_sumdiagcov(y: &[f64]) -> f64 {
    //const check
    let mut constant = true;
    for &value in y {
        if value != y[0] {
            constant = false;
        }
    }
    if constant {
        return f64::NAN;
    }

    const NUM_GROUPS: usize = 3;

    let tau = co_firstzero(y, y.len());
    let mut y_filt = vec![0.0; y.len()];

    for i in 0..y.len() {
        y_filt[i] = y[i];
    }

    let n_down = (y.len() - 1) / tau + 1;
    let mut y_down = vec![0.0; n_down];

    for i in 0..n_down {
        y_down[i] = y_filt[i * tau];
    }

    // transfer to alphabet
    let mut y_cg = vec![0; n_down];
    sb_coarsegrain(&y_down,  NUM_GROUPS, &mut y_cg);

    let mut t = [[0; NUM_GROUPS]; NUM_GROUPS];

    // more efficient way of doing the below
    for j in 0..n_down - 1 {
        t[y_cg[j] as usize - 1][y_cg[j + 1] as usize - 1] += 1;
    }

    for i in 0..NUM_GROUPS {
        for j in 0..NUM_GROUPS {
            t[i][j] /= (n_down - 1) as i32;
        }
    }

    let mut column1 = [0.0; NUM_GROUPS];
    let mut column2 = [0.0; NUM_GROUPS];
    let mut column3 = [0.0; NUM_GROUPS];

    for i in 0..NUM_GROUPS {
        column1[i] = t[i][0] as f64;
        column2[i] = t[i][1] as f64;
        column3[i] = t[i][2] as f64;
    }

    let mut columns = [column1, column2, column3].to_vec();

    let mut cov_mat = [[0.0; NUM_GROUPS]; NUM_GROUPS];

    for i in 0..NUM_GROUPS {
        for j in i..NUM_GROUPS {
            let cov_temp = cov(&columns[i], &columns[j]);
            cov_mat[i][j] = cov_temp;
            cov_mat[j][i] = cov_temp;
        }
    }

    let sum_diag_cov: f64 = cov_mat.iter().enumerate().map(|(i, row)| row[i]).sum();

    sum_diag_cov
}

// fn pd_periodicity_wang_th0_01(y: &[f64]) -> i32 {

//     const TH: f64 = 0.01;

//     let mut y_spline = vec![0.0; y.len()];
//     splinefit(y, y.len(), &mut y_spline);

//     let mut y_sub = vec![0.0; y.len()];
//     for i in 0..y.len() {
//         y_sub[i] = y[i] - y_spline[i];
//         // println!("y_sub[{}] = {}", i, y_sub[i]);
//     }

//     let acmax = ((y.len() as f64) / 3.0).ceil() as usize;

//     let mut acf = vec![0.0; acmax];
//     for tau in 1..=acmax {
//         acf[tau - 1] = autocov_lag(&y_sub, tau);
//     }

//     let mut troughs = vec![0.0; acmax];
//     let mut peaks = vec![0.0; acmax];
//     let mut n_troughs = 0;
//     let mut n_peaks = 0;
//     let mut slope_in = 0.0;
//     let mut slope_out = 0.0;

//     for i in 1..acmax - 1 {
//         slope_in = acf[i] - acf[i - 1];
//         slope_out = acf[i + 1] - acf[i];

//         if slope_in < 0.0 && slope_out > 0.0 {
//             troughs[n_troughs] = i as f64;
//             n_troughs += 1;
//         } else if slope_in > 0.0 && slope_out < 0.0 {
//             peaks[n_peaks] = i as f64;
//             n_peaks += 1;
//         }
//     }

//     let mut i_peak = 0;
//     let mut the_peak = 0.0;
//     let mut i_trough = 0;
//     let mut the_trough = 0.0;
//     let mut out = 0;

//     for i in 0..n_peaks {
//         i_peak = peaks[i] as usize;
//         the_peak = acf[i_peak];

//         let mut j:i32 = -1;
//         while troughs[j as usize + 1] < i_peak as f64 && (j as usize + 1) < n_troughs {
//             j += 1;
//         }
//         if j == -1 {
//             continue;
//         }

//         i_trough = troughs[j as usize] as usize;
//         the_trough = acf[i_trough];

//         if the_peak - the_trough < TH {
//             continue;
//         }

//         if the_peak < 0.0 {
//             continue;
//         }

//         out = i_peak;
//         break;
//     }

//     out as i32
// }

pub fn in_auto_mutual_info_stats_40_gaussian_fmmi(x: &[f64]) -> f64 {
    // maximum time delay
    let tau = 40;

    // don't go above half the signal length
    let tau = tau.min((x.len() as f64 / 2.0).ceil() as usize);

    // compute autocorrelations and compute automutual information
    let mut ami = vec![0.0; x.len()];
    for i in 0..tau {
        let ac = autocorr_lag(x, i + 1);
        ami[i] = -0.5 * (1.0 - ac.powi(2)).ln();
    }

    // find first minimum of automutual information
    let mut fmmi = tau;
    for i in 1..tau - 1 {
        if ami[i] < ami[i - 1] && ami[i] < ami[i + 1] {
            fmmi = i;
            break;
        }
    }
    fmmi as f64
}

pub fn sb_binary_stats_diff_longstretch0(x: &[f64]) -> f64 {
    // Binarize
    let mut x_bin: Vec<i32> = Vec::with_capacity(x.len() - 1);
    for i in 0..x.len() - 1 {
        let diff_temp = x[i + 1] - x[i];
        x_bin.push(if diff_temp < 0.0 { 0 } else { 1 });
    }

    let mut max_stretch0 = 0;
    let mut last1 = 0;
    for i in 0..x.len() - 1 {
        if x_bin[i] == 1 || i == x.len() - 2 {
            let stretch0 = i - last1;
            if stretch0 > max_stretch0 {
                max_stretch0 = stretch0;
            }
            last1 = i;
        }
    }

    // Free allocated memorx
    x_bin.clear();

    max_stretch0 as f64
}

// pub fn sb_motif_three_quantile_hh(y: &[f64]) -> f64 {

//     let mut tmp_idx;
//     let mut r_idx;
//     let mut dynamic_idx;
//     let alphabet_size = 3;
//     let mut array_size;
//     let mut yt = vec![0; y.len()]; // alphabetized array
//     let mut hh; // output
//     let mut out = vec![0.0; 124]; // output array

//     // transfer to alphabet
//     sb_coarsegrain(y, "quantile", 3, &mut yt);

//     // words of length 1
//     array_size = alphabet_size;
//     let mut r1 = vec![vec![0; y.len()]; alphabet_size];
//     let mut sizes_r1 = vec![0; alphabet_size];
//     let mut out1 = vec![0.0; alphabet_size];
//     for i in 0..alphabet_size {
//         r_idx = 0;
//         sizes_r1[i] = 0;
//         for j in 0..y.len() {
//             if yt[j] == i as i32 + 1 {
//                 r1[i][r_idx] = j;
//                 r_idx += 1;
//                 sizes_r1[i] += 1;
//             }
//         }
//     }

//     // words of length 2
//     array_size *= alphabet_size;
//     for i in 0..alphabet_size {
//         if sizes_r1[i] != 0 && r1[i][sizes_r1[i] - 1] == y.len() - 1 {
//             let mut tmp_ar = r1[i][0..sizes_r1[i]].clone_from_slice(src);
//             r1[i].copy_from_slice(&tmp_ar[..sizes_r1[i] - 1]);
//             sizes_r1[i] -= 1;
//         }
//     }

//     let mut r2 = vec![vec![vec![0; y.len()]; alphabet_size]; alphabet_size];
//     let mut sizes_r2 = vec![vec![0; alphabet_size]; alphabet_size];
//     let mut out2 = vec![vec![0.0; alphabet_size]; alphabet_size];

//     for i in 0..alphabet_size {
//         for j in 0..alphabet_size {
//             sizes_r2[i][j] = 0;
//             dynamic_idx = 0;
//             for k in 0..sizes_r1[i] {
//                 tmp_idx = yt[r1[i][k] + 1];
//                 if tmp_idx == j as i32 + 1 {
//                     r2[i][j][dynamic_idx] = r1[i][k];
//                     dynamic_idx += 1;
//                     sizes_r2[i][j] += 1;
//                 }
//             }
//             let tmp = sizes_r2[i][j] as f64 / (y.len() as f64 - 1.0);
//             out2[i][j] = tmp;
//         }
//     }

//     hh = 0.0;
//     for i in 0..alphabet_size {
//         hh += f_entropy(&out2[i]);
//     }

//     hh
// }
