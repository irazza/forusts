use crate::forest::extremely_randomized_canonical_interval_forest::{
    ExtremelyRandomizedCanonicalIntervalForest, ExtremelyRandomizedCanonicalIntervalForestConfig,
};
use crate::forest::forest::{ClassificationForest, Forest};
use crate::metrics::classification::accuracy_score;
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
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
    let paths = fs::read_dir("/media/DATA/albertoazzari/UCRArchive_2018/")?;

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    let mut wtr = csv::Writer::from_path("ercif_light.csv")?;
    wtr.write_record(&[
        "Dataset",
        "Breiman",
        "Time Breiman",
        "Zhu",
        "Time Zhu",
        "RatioRF",
        "Time RatioRF",
    ])?;
    wtr.flush()?;
    for i in 0..n_repetitions {
        println!("Repetition {}", i + 1);
        for path in &datasets {
            println!("\tProcessing {}", path.file_name().to_string_lossy());
            let train_path = path
                .path()
                .join(format!("{}_TRAIN.tsv", path.file_name().to_string_lossy()));
            let test_path = path
                .path()
                .join(format!("{}_TEST.tsv", path.file_name().to_string_lossy()));

            let mut ds_train = read_csv(train_path, b'\t', false)?;
            let ds_test = read_csv(test_path, b'\t', false)?;

            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();

            let config = ExtremelyRandomizedCanonicalIntervalForestConfig {
                n_intervals: (ds_train[0].data.len() as f64).log10() as usize,
                n_attributes: 8,
                ts_length: ds_train[0].data.len(),
                classification_config: ClassificationForestConfig {
                    n_trees: 100,
                    max_depth: Some((ds_train[0].data.len() as f64).sqrt() as usize),
                    min_samples_split: 2,
                    max_features: tree::tree::MaxFeatures::Sqrt,
                    criterion: Criterion::Random,
                    bootstrap: true,
                },
            };

            let mut model = ExtremelyRandomizedCanonicalIntervalForest::new(config);
            let model_time = std::time::Instant::now();
            model.fit(&mut ds_train);
            let model_time = model_time.elapsed().as_secs_f64();
            println!("\tModel built in {}s", model_time);

            let breiman_time = std::time::Instant::now();
            let breiman_distance = model.pairwise_breiman(&ds_test, &ds_train);
            let breiman_time = breiman_time.elapsed().as_secs_f64();
            let prediction_breiman = k_nearest_neighbor(
                1,
                &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(),
                &breiman_distance,
            );
            let accuracy_breiman = accuracy_score(&prediction_breiman, &y_true);

            let zhu_time = std::time::Instant::now();
            let zhu_distance = model.pairwise_zhu(&ds_test, &ds_train);
            let zhu_time = zhu_time.elapsed().as_secs_f64();
            let prediction_zhu = k_nearest_neighbor(
                1,
                &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(),
                &zhu_distance,
            );
            let accuracy_zhu = accuracy_score(&prediction_zhu, &y_true);
            let ratio_time = std::time::Instant::now();
            let ratiorf_distance = model.pairwise_ratiorf(&ds_test, &ds_train);

            let ratio_time = ratio_time.elapsed().as_secs_f64();
            let prediction_ratiorf = k_nearest_neighbor(
                1,
                &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(),
                &ratiorf_distance,
            );
            let accuracy_ratiorf = accuracy_score(&prediction_ratiorf, &y_true);
            println!(
                "\t\tBreiman: {}\n\t\tZhu: {}\n\t\tRatioRF: {}",
                accuracy_breiman, accuracy_zhu, accuracy_ratiorf
            );
            vec_vec_to_csv(
                format!(
                    "/media/DATA/albertoazzari/ERCIF_DISTANCES/{}/{}_T{}I{}F{}breiman{}.csv",
                    path.file_name().to_string_lossy(),
                    path.file_name().to_string_lossy(),
                    config.classification_config.n_trees,
                    config.n_intervals,
                    config.n_attributes,
                    i
                ),
                &breiman_distance,
            )?;
            vec_vec_to_csv(
                format!(
                    "/media/DATA/albertoazzari/ERCIF_DISTANCES/{}/{}_T{}I{}F{}zhu{}.csv",
                    path.file_name().to_string_lossy(),
                    path.file_name().to_string_lossy(),
                    config.classification_config.n_trees,
                    config.n_intervals,
                    config.n_attributes,
                    i
                ),
                &zhu_distance,
            )?;
            vec_vec_to_csv(
                format!(
                    "/media/DATA/albertoazzari/ERCIF_DISTANCES/{}/{}_T{}I{}F{}ratiorf{}.csv",
                    path.file_name().to_string_lossy(),
                    path.file_name().to_string_lossy(),
                    config.classification_config.n_trees,
                    config.n_intervals,
                    config.n_attributes,
                    i
                ),
                &ratiorf_distance,
            )?;
            wtr.write_record(&[
                path.file_name().to_string_lossy().to_string(),
                accuracy_breiman.to_string(),
                breiman_time.to_string(),
                accuracy_zhu.to_string(),
                zhu_time.to_string(),
                accuracy_ratiorf.to_string(),
                ratio_time.to_string(),
            ])?;
            wtr.flush()?;
            ERCIF_CACHE.clear();
        }
    }
    Ok(())
}
