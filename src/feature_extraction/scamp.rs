mod bindings {
    #![allow(warnings)]

    include! {concat ! (env ! ("OUT_DIR"),"/scamp.rs")}
}

pub fn compute_scamp(x: &[f64], window: usize) -> (Vec<f32>, Vec<i32>) {
    let mut matrix_profile = vec![0.0; x.len()-window+1];
    let mut profile_index = vec![0; x.len()-window+1];

    unsafe {
        bindings::compute_scamp(
            x.as_ptr(),
            x.len(),
            window as i32,
            matrix_profile.as_mut_ptr(),
            profile_index.as_mut_ptr(),
        );
    }

    (matrix_profile, profile_index)
}