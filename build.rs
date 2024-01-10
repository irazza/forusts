use cmake::Config;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::{env, fs};

fn main() {
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

    let mut header_paths = Vec::new();

    for path in filtered_paths {
        let bindings = bindgen::Builder::default()
            .blocklist_item("FP_.*")
            .header(path.to_string_lossy())
            .generate()
            .expect("Unable to generate bindings");

        let name = path.file_stem().unwrap().to_string_lossy().into_owned();

        let temp_path =
            PathBuf::from(env::var("OUT_DIR").unwrap()).join(format!("bindings_{}.rs", name));
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
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"),
        bindings,
    )
    .unwrap();

    let paths = fs::read_dir("./catch22/C/").unwrap();
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
        .opt_level(3)
        .compile("catch22");

    // Run the configure script
    let dst = Config::new("catch22/fftw-3.3.10")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("ENABLE_OPENMP", "ON")
        .build();
    println!("cargo:rustc-link-arg=-fopenmp");
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("build").display()
    );
    println!("cargo:rustc-link-lib=static=fftw3");
}
