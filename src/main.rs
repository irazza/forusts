use crate::forest::extremely_randomized_canonical_interval_forest::{
    ExtremelyRandomizedCanonicalIntervalForest, ExtremelyRandomizedCanonicalIntervalForestConfig,
};
use crate::forest::forest::{ClassificationForest, Forest};
use crate::tree::extremely_randomized_canonical_interval_tree::ERCIF_CACHE;
use crate::utils::csv_io::{read_csv, vec_vec_to_csv};
use forest::forest::ClassificationForestConfig;
use std::error::Error;
use std::fs::{self};
use tree::tree::Criterion;

mod distance;
mod feature_extraction;
mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;

fn main() -> Result<(), Box<dyn Error>> {
    // Settings for the experiments
    let n_repetitions = 10;
    let paths = fs::read_dir("/media/aazzari/UCRArchive_2018/")?;

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    for i in 0..n_repetitions {
        println!("Repetition {}", i + 1);
        for path in &datasets {
            print!("\tProcessing {}", path.file_name().to_string_lossy());
            let train_path = path
                .path()
                .join(format!("{}_TRAIN.tsv", path.file_name().to_string_lossy()));
            let test_path = path
                .path()
                .join(format!("{}_TEST.tsv", path.file_name().to_string_lossy()));

            let ds_train = read_csv(train_path, b'\t', false)?;
            let ds_test = read_csv(test_path, b'\t', false)?;

            let mut ds = ds_train.clone();
            ds.extend(ds_test.clone());

            let config = ExtremelyRandomizedCanonicalIntervalForestConfig {
                n_intervals: (ds[0].data.len() as f64).sqrt() as usize,
                n_attributes: 8,
                ts_length: ds_train[0].data.len(),
                classification_config: ClassificationForestConfig {
                    n_trees: 500,
                    max_depth: Some((ds_train[0].data.len() as f64).sqrt() as usize),
                    min_samples_split: 2,
                    max_features: tree::tree::MaxFeatures::Sqrt,
                    criterion: Criterion::Random,
                    bootstrap: true,
                },
            };

            let mut model = ExtremelyRandomizedCanonicalIntervalForest::new(config);
            let model_time = std::time::Instant::now();
            model.fit(&mut ds);
            let model_time = model_time.elapsed().as_secs_f64();
            println!(": built in {}s", model_time);

            let breiman_distance = model.pairwise_breiman(&ds, &ds);
            
            let zhu_distance = model.pairwise_zhu(&ds, &ds);
            
            let ratiorf_distance = model.pairwise_ratiorf(&ds, &ds);

            vec_vec_to_csv(
                format!(
                    "/media/aazzari/ERCIF_DISTANCES/HEAVY/{}/{:?}_breiman{}.csv",
                    path.file_name().to_string_lossy(),
                    config,
                    i
                ),
                &breiman_distance,
            )?;
            vec_vec_to_csv(
                format!(
                    "/media/aazzari/ERCIF_DISTANCES/HEAVY/{}/{:?}_zhu{}.csv",
                    path.file_name().to_string_lossy(),
                    config,
                    i
                ),
                &zhu_distance,
            )?;
            vec_vec_to_csv(
                format!(
                    "/media/aazzari/ERCIF_DISTANCES/{}/{:?}_ratiorf{}.csv",
                    path.file_name().to_string_lossy(),
                    config,
                    i
                ),
                &ratiorf_distance,
            )?;
            ERCIF_CACHE.clear();
        }
    }
    Ok(())
}
