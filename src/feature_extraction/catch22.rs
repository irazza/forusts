use super::statistics::zscore;

mod bindings {
    #![allow(warnings)]

    include! {concat ! (env ! ("OUT_DIR"),"/bindings.rs")}
}

pub enum CATCH22 {
    DN_OutlierInclude_n_001_mdrmd,
    DN_OutlierInclude_p_001_mdrmd,
    DN_HistogramMode_5,
    DN_HistogramMode_10,
    CO_Embed2_Dist_tau_d_expfit_meandiff,
    CO_f1ecac,
    CO_FirstMin_ac,
    CO_HistogramAMI_even_2_5,
    CO_trev_1_num,
    FC_LocalSimple_mean1_tauresrat,
    FC_LocalSimple_mean3_stderr,
    IN_AutoMutualInfoStats_40_gaussian_fmmi,
    MD_hrv_classic_pnn40,
    SB_BinaryStats_diff_longstretch0,
    SB_BinaryStats_mean_longstretch1,
    SB_MotifThree_quantile_hh,
    SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
    SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    SP_Summaries_welch_rect_area_5_1,
    SP_Summaries_welch_rect_centroid,
    SB_TransitionMatrix_3ac_sumdiagcov,
    PD_PeriodicityWang_th0_01,
}
impl CATCH22 {
    pub fn to_fn(self) -> fn(&[f64]) -> f64 {
        match self {
            CATCH22::DN_OutlierInclude_n_001_mdrmd => dn_outlier_include_n_001_mdrmd,
            CATCH22::DN_OutlierInclude_p_001_mdrmd => dn_outlier_include_p_001_mdrmd,
            CATCH22::DN_HistogramMode_5 => dn_histogram_mode_5,
            CATCH22::DN_HistogramMode_10 => dn_histogram_mode_10,
            CATCH22::CO_Embed2_Dist_tau_d_expfit_meandiff => co_embed2_dist_tau_d_expfit_meandiff,
            CATCH22::CO_f1ecac => co_f1ecac,
            CATCH22::CO_FirstMin_ac => co_first_min_ac,
            CATCH22::CO_HistogramAMI_even_2_5 => co_histogram_ami_even_2_5,
            CATCH22::CO_trev_1_num => co_trev_1_num,
            CATCH22::FC_LocalSimple_mean1_tauresrat => fc_localsimple_mean1_tauresrat,
            CATCH22::FC_LocalSimple_mean3_stderr => fc_localsimple_mean3_stderr,
            CATCH22::IN_AutoMutualInfoStats_40_gaussian_fmmi => {
                in_auto_mutual_info_stats_40_gaussian_fmmi
            }
            CATCH22::MD_hrv_classic_pnn40 => md_hrv_classic_pnn40,
            CATCH22::SB_BinaryStats_diff_longstretch0 => sb_binarystats_diff_longstretch0,
            CATCH22::SB_BinaryStats_mean_longstretch1 => sb_binary_stats_mean_longstretch1,
            CATCH22::SB_MotifThree_quantile_hh => sb_motifthree_quantile_hh,
            CATCH22::SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1 => {
                sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1
            }
            CATCH22::SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1 => {
                sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1
            }
            CATCH22::SP_Summaries_welch_rect_area_5_1 => sp_summaries_welch_rect_area_5_1,
            CATCH22::SP_Summaries_welch_rect_centroid => sp_summaries_welch_rect_centroid,
            CATCH22::SB_TransitionMatrix_3ac_sumdiagcov => sb_transition_matrix_3ac_sumdiagcov,
            CATCH22::PD_PeriodicityWang_th0_01 => pd_periodicity_wang_th0_01,
        }
    }
    pub fn get(i: usize) -> fn(&[f64]) -> f64 {
        match i {
            0 => dn_outlier_include_n_001_mdrmd,
            1 => dn_outlier_include_p_001_mdrmd,
            2 => dn_histogram_mode_5,
            3 => dn_histogram_mode_10,
            4 => co_embed2_dist_tau_d_expfit_meandiff,
            5 => co_f1ecac,
            6 => co_first_min_ac,
            7 => co_histogram_ami_even_2_5,
            8 => co_trev_1_num,
            9 => fc_localsimple_mean1_tauresrat,
            10 => fc_localsimple_mean3_stderr,
            11 => in_auto_mutual_info_stats_40_gaussian_fmmi,
            12 => md_hrv_classic_pnn40,
            13 => sb_binarystats_diff_longstretch0,
            14 => sb_binary_stats_mean_longstretch1,
            15 => sb_motifthree_quantile_hh,
            16 => sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1,
            17 => sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1,
            18 => sp_summaries_welch_rect_area_5_1,
            19 => sp_summaries_welch_rect_centroid,
            20 => sb_transition_matrix_3ac_sumdiagcov,
            21 => pd_periodicity_wang_th0_01,
            _ => panic!("Invalid index for CATCH22 (valide range 0..22)"),
        }
    }
}
pub fn compute_catch_features(x: &[f64]) -> Vec<f64> {
    let mut features = Vec::new();
    let mut x_zscored = vec![0.0; x.len()];
    unsafe { bindings::stats::zscore_norm2(x.as_ptr(), x.len() as i32, x_zscored.as_mut_ptr()) };
    features.push(dn_outlier_include_n_001_mdrmd(&x_zscored));
    features.push(dn_outlier_include_p_001_mdrmd(&x_zscored));
    features.push(dn_histogram_mode_5(x));
    features.push(dn_histogram_mode_10(x));
    features.push(co_embed2_dist_tau_d_expfit_meandiff(x));
    features.push(co_f1ecac(x));
    features.push(co_first_min_ac(x));
    features.push(co_histogram_ami_even_2_5(x));
    features.push(co_trev_1_num(x));
    features.push(fc_localsimple_mean1_tauresrat(x));
    features.push(fc_localsimple_mean3_stderr(x));
    features.push(in_auto_mutual_info_stats_40_gaussian_fmmi(x));
    features.push(md_hrv_classic_pnn40(x));
    features.push(sb_binarystats_diff_longstretch0(x));
    features.push(sb_binary_stats_mean_longstretch1(x));
    features.push(sb_motifthree_quantile_hh(x));
    features.push(sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1(x));
    features.push(sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1(x));
    features.push(sp_summaries_welch_rect_area_5_1(x));
    features.push(sp_summaries_welch_rect_centroid(x));
    features.push(sb_transition_matrix_3ac_sumdiagcov(x));
    features.push(pd_periodicity_wang_th0_01(x));
    features
}

pub fn dn_outlier_include_n_001_mdrmd(x: &[f64]) -> f64 {
    let x = zscore(x);
    unsafe {
        let result =
            bindings::DN_OutlierInclude::DN_OutlierInclude_n_001_mdrmd(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn dn_outlier_include_p_001_mdrmd(x: &[f64]) -> f64 {
    let x = zscore(x);
    unsafe {
        let result =
            bindings::DN_OutlierInclude::DN_OutlierInclude_p_001_mdrmd(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn dn_histogram_mode_5(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::DN_HistogramMode_5::DN_HistogramMode_5(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn dn_histogram_mode_10(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::DN_HistogramMode_10::DN_HistogramMode_10(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn co_embed2_dist_tau_d_expfit_meandiff(x: &[f64]) -> f64 {
    unsafe {
        let result =
            bindings::CO_AutoCorr::CO_Embed2_Dist_tau_d_expfit_meandiff(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn co_f1ecac(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::CO_AutoCorr::CO_f1ecac(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn co_first_min_ac(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::CO_AutoCorr::CO_FirstMin_ac(x.as_ptr(), x.len() as i32) as f64;
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn co_histogram_ami_even_2_5(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::CO_AutoCorr::CO_HistogramAMI_even_2_5(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn co_trev_1_num(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::CO_AutoCorr::CO_trev_1_num(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn fc_localsimple_mean1_tauresrat(x: &[f64]) -> f64 {
    unsafe {
        let result =
            bindings::FC_LocalSimple::FC_LocalSimple_mean1_tauresrat(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn fc_localsimple_mean3_stderr(x: &[f64]) -> f64 {
    unsafe {
        let result =
            bindings::FC_LocalSimple::FC_LocalSimple_mean3_stderr(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
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
    unsafe {
        let result = bindings::MD_hrv::MD_hrv_classic_pnn40(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn sb_binarystats_diff_longstretch0(x: &[f64]) -> f64 {
    unsafe {
        let result =
            bindings::SB_BinaryStats::SB_BinaryStats_diff_longstretch0(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn sb_binary_stats_mean_longstretch1(x: &[f64]) -> f64 {
    unsafe {
        let result =
            bindings::SB_BinaryStats::SB_BinaryStats_mean_longstretch1(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn sb_motifthree_quantile_hh(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::SB_MotifThree::SB_MotifThree_quantile_hh(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
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
        let result = bindings::SC_FluctAnal::SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(
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

pub fn sp_summaries_welch_rect_area_5_1(x: &[f64]) -> f64 {
    unsafe {
        let result =
            bindings::SP_Summaries::SP_Summaries_welch_rect_area_5_1(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn sp_summaries_welch_rect_centroid(x: &[f64]) -> f64 {
    unsafe {
        let result =
            bindings::SP_Summaries::SP_Summaries_welch_rect_centroid(x.as_ptr(), x.len() as i32);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

pub fn sb_transition_matrix_3ac_sumdiagcov(x: &[f64]) -> f64 {
    unsafe {
        let result = bindings::SB_TransitionMatrix::SB_TransitionMatrix_3ac_sumdiagcov(
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

pub fn pd_periodicity_wang_th0_01(x: &[f64]) -> f64 {
    unsafe {
        let result =
            bindings::PD_PeriodicityWang::PD_PeriodicityWang_th0_01(x.as_ptr(), x.len() as i32)
                as f64;
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}
