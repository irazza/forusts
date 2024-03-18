mod bindings {
    #![allow(warnings)]

    include! {concat ! (env ! ("OUT_DIR"),"/scamp.rs")}
}

pub fn compute_mp(x1: &[f64], x2: &[f64], window: usize) -> (Vec<f32>, Vec<i32>) {
    let mut matrix_profile = vec![0.0; x1.len() - window + 1];
    let mut profile_index = vec![0; x1.len() - window + 1];

    unsafe {
        bindings::compute_mp(
            x1.as_ptr(),
            x1.len(),
            x2.as_ptr(),
            x2.len(),
            window as i32,
            matrix_profile.as_mut_ptr(),
            profile_index.as_mut_ptr(),
        );
    }

    (matrix_profile, profile_index)
}

pub fn compute_selfmp(x1: &[f64], window: usize) -> (Vec<f32>, Vec<i32>) {
    let mut matrix_profile = vec![0.0; x1.len() - window + 1];
    let mut profile_index = vec![0; x1.len() - window + 1];

    unsafe {
        bindings::compute_selfmp(
            x1.as_ptr(),
            x1.len(),
            window as i32,
            matrix_profile.as_mut_ptr(),
            profile_index.as_mut_ptr(),
        );
    }

    (matrix_profile, profile_index)
}
