mod bindings{
    #![allow(warnings)]
    
    include ! {concat ! (env ! ("OUT_DIR"),"/bindings.rs")}
}

pub fn DN_OutlierInclude_n_001_mdrmd(x: &[f64]) -> f64 {
    unsafe{bindings::DN_OutlierInclude::DN_OutlierInclude_n_001_mdrmd(x.as_ptr(), x.len() as i32)}
}

pub fn  DN_OutlierInclude_p_001_mdrmd(x: &[f64]) -> f64 {
    unsafe{bindings::DN_OutlierInclude::DN_OutlierInclude_p_001_mdrmd(x.as_ptr(), x.len() as i32)}
}

pub fn DN_HistogramMode_5(x: &[f64]) -> f64 {
    unsafe{bindings::DN_HistogramMode_5::DN_HistogramMode_5(x.as_ptr(), x.len() as i32)}
}

pub fn DN_HistogramMode_10(x: &[f64]) -> f64 {
    unsafe{bindings::DN_HistogramMode_10::DN_HistogramMode_10(x.as_ptr(), x.len() as i32)}
}

pub fn CO_Embed2_Dist_tau_d_expfit_meandiff(x: &[f64]) -> f64 {
    unsafe{bindings::CO_AutoCorr::CO_Embed2_Dist_tau_d_expfit_meandiff(x.as_ptr(), x.len() as i32)}
}

pub fn CO_f1ecac(x: &[f64]) -> f64 {
    unsafe{bindings::CO_AutoCorr::CO_f1ecac(x.as_ptr(), x.len() as i32)}
}

pub fn CO_FirstMin_ac(x: &[f64]) -> i32 {
    unsafe{bindings::CO_AutoCorr::CO_FirstMin_ac(x.as_ptr(), x.len() as i32)}
}

pub fn CO_HistogramAMI_even_2_5(x: &[f64]) -> f64 {
    unsafe{bindings::CO_AutoCorr::CO_HistogramAMI_even_2_5(x.as_ptr(), x.len() as i32)}
}

pub fn CO_trev_1_num(x: &[f64]) -> f64 {
    unsafe{bindings::CO_AutoCorr::CO_trev_1_num(x.as_ptr(), x.len() as i32)}
}

pub fn FC_LocalSimple_mean1_tauresrat(x: &[f64]) -> f64 {
    unsafe{bindings::FC_LocalSimple::FC_LocalSimple_mean1_tauresrat(x.as_ptr(), x.len() as i32)}
}

pub fn FC_LocalSimple_mean3_stderr(x: &[f64]) -> f64 {
    unsafe{bindings::FC_LocalSimple::FC_LocalSimple_mean3_stderr(x.as_ptr(), x.len() as i32)}
}

pub fn IN_AutoMutualInfoStats_40_gaussian_fmmi(x: &[f64]) -> f64 {
    unsafe{bindings::IN_AutoMutualInfoStats::IN_AutoMutualInfoStats_40_gaussian_fmmi(x.as_ptr(), x.len() as i32)}
}

pub fn MD_hrv_classic_pnn40(x: &[f64]) -> f64 {
    unsafe{bindings::MD_hrv::MD_hrv_classic_pnn40(x.as_ptr(), x.len() as i32)}
}

pub fn SB_BinaryStats_diff_longstretch0(x: &[f64]) -> f64 {
    unsafe{bindings::SB_BinaryStats::SB_BinaryStats_diff_longstretch0(x.as_ptr(), x.len() as i32)}
}

pub fn SB_BinaryStats_mean_longstretch1(x: &[f64]) -> f64 {
    unsafe{bindings::SB_BinaryStats::SB_BinaryStats_mean_longstretch1(x.as_ptr(), x.len() as i32)}
}

pub fn SB_MotifThree_quantile_hh(x: &[f64]) -> f64 {
    unsafe{bindings::SB_MotifThree::SB_MotifThree_quantile_hh(x.as_ptr(), x.len() as i32)}
}

pub fn SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(x: &[f64]) -> f64 {
    unsafe{bindings::SC_FluctAnal::SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(x.as_ptr(), x.len() as i32)}
}

pub fn SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(x: &[f64]) -> f64 {
    unsafe{bindings::SC_FluctAnal::SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(x.as_ptr(), x.len() as i32)}
}

pub fn SP_Summaries_welch_rect_area_5_1(x: &[f64]) -> f64 {
    unsafe{bindings::SP_Summaries::SP_Summaries_welch_rect_area_5_1(x.as_ptr(), x.len() as i32)}
}

pub fn SP_Summaries_welch_rect_centroid(x: &[f64]) -> f64 {
    unsafe{bindings::SP_Summaries::SP_Summaries_welch_rect_centroid(x.as_ptr(), x.len() as i32)}
}

pub fn SB_TransitionMatrix_3ac_sumdiagcov(x: &[f64]) ->f64 {
    unsafe{bindings::SB_TransitionMatrix::SB_TransitionMatrix_3ac_sumdiagcov(x.as_ptr(), x.len() as i32)}
}

pub fn PD_PeriodicityWang_th0_01(x: &[f64]) -> i32 {
    unsafe{bindings::PD_PeriodicityWang::PD_PeriodicityWang_th0_01(x.as_ptr(), x.len() as i32)}
}