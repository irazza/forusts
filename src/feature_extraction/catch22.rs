mod bindings {
    #![allow(warnings)]

    include! {concat ! (env ! ("OUT_DIR"),"/bindings.rs")}
}

pub fn compute_catch_features(x: &[f64]) -> Vec<f64> {
    let mut features = Vec::new();
    let mut x_zscored = vec![0.0; x.len()];
    unsafe { bindings::stats::zscore_norm2(x.as_ptr(), x.len() as i32, x_zscored.as_mut_ptr()) };
    features.push(dn_outlier_include_n_001_mdrmd(&x_zscored));
    features.push(dn_outlier_include_p_001_mdrmd(&x_zscored));
    features.push(dn_histogram_mode_5(&x_zscored));
    features.push(dn_histogram_mode_10(&x_zscored));
    features.push(co_embed2_dist_tau_d_expfit_meandiff(&x_zscored));
    features.push(co_f1ecac(&x_zscored));
    features.push(co_first_min_ac(&x_zscored));
    features.push(co_histogram_ami_even_2_5(&x_zscored));
    features.push(co_trev_1_num(&x_zscored));
    features.push(fc_localsimple_mean1_tauresrat(&x_zscored));
    features.push(fc_localsimple_mean3_stderr(&x_zscored));
    features.push(in_auto_mutual_info_stats_40_gaussian_fmmi(&x_zscored));
    features.push(md_hrv_classic_pnn40(&x_zscored));
    features.push(sb_binarystats_diff_longstretch0(&x_zscored));
    features.push(sb_binary_stats_mean_longstretch1(&x_zscored));
    features.push(sb_motifthree_quantile_hh(&x_zscored));
    features.push(sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1(&x_zscored));
    features.push(sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1(&x_zscored));
    features.push(sp_summaries_welch_rect_area_5_1(&x_zscored));
    features.push(sp_summaries_welch_rect_centroid(&x_zscored));
    features.push(sb_transition_matrix_3ac_sumdiagcov(&x_zscored));
    features.push(pd_periodicity_wang_th0_01(&x_zscored));
    features
}

pub fn dn_outlier_include_n_001_mdrmd(x: &[f64]) -> f64 {
    
    unsafe {
        let result = bindings::DN_OutlierInclude::DN_OutlierInclude_n_001_mdrmd(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn dn_outlier_include_p_001_mdrmd(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::DN_OutlierInclude::DN_OutlierInclude_p_001_mdrmd(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
    
}

pub fn dn_histogram_mode_5(x: &[f64]) -> f64 {
    unsafe { let result = bindings::DN_HistogramMode_5::DN_HistogramMode_5(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn dn_histogram_mode_10(x: &[f64]) -> f64 {
    unsafe { let result = bindings::DN_HistogramMode_10::DN_HistogramMode_10(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn co_embed2_dist_tau_d_expfit_meandiff(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::CO_AutoCorr::CO_Embed2_Dist_tau_d_expfit_meandiff(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn co_f1ecac(x: &[f64]) -> f64 {
    unsafe { let result = bindings::CO_AutoCorr::CO_f1ecac(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn co_first_min_ac(x: &[f64]) -> f64 {
    unsafe { let result = bindings::CO_AutoCorr::CO_FirstMin_ac(x.as_ptr(), x.len() as i32) as f64;
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn co_histogram_ami_even_2_5(x: &[f64]) -> f64 {
    unsafe { let result = bindings::CO_AutoCorr::CO_HistogramAMI_even_2_5(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn co_trev_1_num(x: &[f64]) -> f64 {
    unsafe { let result = bindings::CO_AutoCorr::CO_trev_1_num(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn fc_localsimple_mean1_tauresrat(x: &[f64]) -> f64 {
    unsafe { let result = bindings::FC_LocalSimple::FC_LocalSimple_mean1_tauresrat(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn fc_localsimple_mean3_stderr(x: &[f64]) -> f64 {
    unsafe { let result = bindings::FC_LocalSimple::FC_LocalSimple_mean3_stderr(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn in_auto_mutual_info_stats_40_gaussian_fmmi(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::IN_AutoMutualInfoStats::IN_AutoMutualInfoStats_40_gaussian_fmmi(
            x.as_ptr(),
            x.len() as i32,
        );
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn md_hrv_classic_pnn40(x: &[f64]) -> f64 {
    unsafe { let result = bindings::MD_hrv::MD_hrv_classic_pnn40(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn sb_binarystats_diff_longstretch0(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::SB_BinaryStats::SB_BinaryStats_diff_longstretch0(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn sb_binary_stats_mean_longstretch1(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::SB_BinaryStats::SB_BinaryStats_mean_longstretch1(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn sb_motifthree_quantile_hh(x: &[f64]) -> f64 {
    unsafe { let result = bindings::SB_MotifThree::SB_MotifThree_quantile_hh(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::SC_FluctAnal::SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(
            x.as_ptr(),
            x.len() as i32,
        );
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::SC_FluctAnal::SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn sp_summaries_welch_rect_area_5_1(x: &[f64]) -> f64 {
    unsafe { let result = bindings::SP_Summaries::SP_Summaries_welch_rect_area_5_1(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn sp_summaries_welch_rect_centroid(x: &[f64]) -> f64 {
    unsafe { let result = bindings::SP_Summaries::SP_Summaries_welch_rect_centroid(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}

pub fn sb_transition_matrix_3ac_sumdiagcov(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::SB_TransitionMatrix::SB_TransitionMatrix_3ac_sumdiagcov(
            x.as_ptr(),
            x.len() as i32
        );
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn pd_periodicity_wang_th0_01(x: &[f64]) -> f64 {
    unsafe { let result = bindings::PD_PeriodicityWang::PD_PeriodicityWang_th0_01(x.as_ptr(), x.len() as i32) as f64;
        if result.is_finite() {
            result
        } else {
            0.0
        } }
}
