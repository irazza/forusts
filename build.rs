use cmake::Config;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::{env, fs};

fn main() {
    // Catch22
    let paths = fs::read_dir("./catch22/C/").unwrap();
    let filtered_paths = paths
        .filter_map(|path| {
            let path = path.unwrap();
            if path.path().extension() == Some(OsStr::new("h")) {
                Some(path.path())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    println!("cargo:rerun-if-changed=catch22/C/");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let mut header_paths = Vec::new();

    for path in filtered_paths {
        let bindings = bindgen::Builder::default()
            .blocklist_item("FP_.*")
            .header(path.to_string_lossy())
            .generate()
            .expect("Unable to generate bindings");

        let name = path.file_stem().unwrap().to_string_lossy().into_owned();

        let temp_path = out_dir.join(format!("bindings_{}.rs", name));
        bindings
            .write_to_file(&temp_path)
            .expect("Couldn't write bindings!");
        header_paths.push((name, temp_path));
    }

    let mut bindings = String::new();
    for (name, path) in header_paths {
        bindings.push_str(&format!(
            "pub mod {}{{include ! {{ \"{}\" }} }}\n",
            name,
            path.display()
        ));
    }
    fs::write(out_dir.join("catch22.rs"), bindings).unwrap();

    compile_from_path("./catch22/C/", "catch22", None);

    // FFTW
    let dst = Config::new("catch22/fftw-3.3.10")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("ENABLE_OPENMP", "ON")
        .out_dir(out_dir.join("fftw"))
        .build();
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("build").display()
    );
    println!("cargo:rustc-link-lib=static=fftw3");

    // // SCAMP

    // let bindings = bindgen::Builder::default()
    //     .blocklist_item("FP_.*")
    //     .header("SCAMP/src/common/api.h")
    //     .generate()
    //     .expect("Unable to generate bindings");

    // let temp_path = out_dir.join("scamp.rs");
    // bindings
    //     .write_to_file(&temp_path)
    //     .expect("Couldn't write bindings!");

    // let dst = Config::new("SCAMP").out_dir(out_dir.join("scamp")).build();
    // for src_dir in [
    //     "/src/common",
    //     "/src/core",
    //     "/src/core/cpu_kernel",
    //     "/src/core/cpu_kernel/baseline",
    // ] {
    //     println!(
    //         "cargo:rustc-link-search=native={}",
    //         dst.join("build").display().to_string() + src_dir
    //     );
    // }
    // println!("cargo:rustc-link-lib=static=scamp_api");
    // println!("cargo:rustc-link-lib=static=scamp_interface");
    // println!("cargo:rustc-link-lib=static=common");
    // println!("cargo:rustc-link-lib=static=scamp_utils");
    // println!("cargo:rustc-link-lib=static=scamp_args");
    // println!("cargo:rustc-link-lib=static=profile");
    // println!("cargo:rustc-link-lib=static=scamp_op");
    // println!("cargo:rustc-link-lib=static=tile");
    // println!("cargo:rustc-link-lib=static=cpu_stats");
    // println!("cargo:rustc-link-lib=static=cpu_kernels");
    // println!("cargo:rustc-link-lib=static=tile");
    // println!("cargo:rustc-link-lib=static=qt_helper");
    // println!("cargo:rustc-link-lib=static=stdc++");
    // println!("cargo:rustc-link-lib=static=cpu_kernels");
    // println!("cargo:rustc-link-lib=static=kernel_common");
    // println!("cargo:rustc-link-lib=static=dispatch_baseline");
    // println!("cargo:rustc-link-lib=static=kernel_baseline");
}

pub fn compile_from_path(path: &str, libname: &str, inc: Option<Vec<&str>>) {
    let paths = fs::read_dir(path).unwrap();
    let filtered_paths = paths
        .filter_map(|path| {
            let path = path.unwrap();
            if path.path().extension() == Some(OsStr::new("c")) {
                Some(path.path())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    cc::Build::new()
        .files(filtered_paths)
        .includes(inc.unwrap_or_default())
        .opt_level(3)
        .compile(libname);
}
