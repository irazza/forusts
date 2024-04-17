#![allow(dead_code)]
use crate::feature_extraction::statistics::mean;
use crate::forest::distance_forest::{DistanceForest, DistanceForestConfig};
use crate::forest::forest::Forest;
use crate::metrics::classification::accuracy_score;
use crate::utils::csv_io::read_csv;
use crate::utils::structures::ZScoreTransformer;

use std::error::Error;
use std::fs::{self};

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
    let paths = fs::read_dir("/media/aazzari/DATA/UCRArchive_2018")?;

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        datasets.push(entry.path());
    }
    datasets.sort_by_key(|dir| dir.file_name().unwrap().to_string_lossy().to_string());
    let mut wtr = csv::Writer::from_path("dist_forest.csv")?;
    wtr.write_record(&["Dataset", "ROC-AUC"])?;
    wtr.flush()?;
    let mut mean_acc = Vec::new();
    for i in 0..n_repetitions {
        println!("Repetition {}", i + 1);
        //let mut predictions = Vec::new();
        for path in &datasets[1..2] {
            println!(
                "\tProcessing {}",
                path.file_name().unwrap().to_string_lossy()
            );
            let train_path = path.join(format!(
                "{}_TRAIN.tsv",
                path.file_name().unwrap().to_string_lossy()
            ));
            let test_path = path.join(format!(
                "{}_TEST.tsv",
                path.file_name().unwrap().to_string_lossy()
            ));

            let mut zst = ZScoreTransformer::new();

            let ds_train = read_csv(train_path, b'\t', false)?;

            let mut ds_train = zst.fit_transform(&ds_train);

            let ds_test = read_csv(test_path, b'\t', false)?;
            let ds_test = zst.transform(&ds_test);

            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();

            let config = DistanceForestConfig {
                n_trees: 500,
                min_samples_split: 2,
                max_features: tree::tree::MaxFeatures::Log2,
                max_depth: None,
                criterion: tree::tree::Criterion::Gini,
                bootstrap: true,
            };
            let mut model = DistanceForest::new(config);
            //let mut ds_train = ds_train.iter().map(|x| Sample{data: Arc::new(zscore(&x.data)), target: x.target}).collect::<Vec<_>>();
            let start_time = std::time::Instant::now();
            model.fit(&mut ds_train);
            println!("\t\tTraining time: {:?}", start_time.elapsed());
            //let ds_test = ds_test.iter().map(|x| Sample{data: Arc::new(zscore(&x.data)), target: x.target}).collect::<Vec<_>>();
            let start_time = std::time::Instant::now();
            let y_pred = model.predict(&ds_test);
            println!("\t\tPrediction time: {:?}", start_time.elapsed());
            let acc = accuracy_score(&y_pred, &y_true);
            mean_acc.push(acc);

            println!("\t\tAccuracy: {}", acc);
            break;
            wtr.write_record(&[
                path.file_name().unwrap().to_string_lossy().to_string(),
                acc.to_string(),
            ])?;
            wtr.flush()?;
        }
    }
    println!("Mean accuracy: {}", mean(&mean_acc));
    Ok(())
}
