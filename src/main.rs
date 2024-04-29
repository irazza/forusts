use crate::forest::distance_set_forest::{DistanceSetForest, DistanceSetForestConfig};
use crate::forest::forest::{ClassificationForestConfig, Forest};
use crate::metrics::classification::accuracy_score;
use crate::utils::csv_io::read_csv;

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
    let config = DistanceSetForestConfig {
        classification_config: ClassificationForestConfig {
            n_trees: 400,
            min_samples_split: 2,
            max_features: tree::tree::MaxFeatures::Sqrt,
            max_depth: None,
            criterion: tree::tree::Criterion::Random,
            bootstrap: true,
        },
    };
    let paths = fs::read_dir("/media/aazzari/UCRArchive_2018/")?;
    let n_repetitions = 10;
    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        datasets.push(entry.path());
    }
    datasets.sort_by_key(|dir| dir.file_name().unwrap().to_string_lossy().to_string());
    let mut wtr = csv::Writer::from_path(format!("{:?}.csv", config))?;
    wtr.write_record(&["Dataset", "ACC"])?;
    wtr.flush()?;
    for i in 0..n_repetitions {
        println!("Repetition {}", i + 1);
        for path in &datasets {
            let train_path = path.join(format!(
                "{}_TRAIN.tsv",
                path.file_name().unwrap().to_string_lossy()
            ));
            let test_path = path.join(format!(
                "{}_TEST.tsv",
                path.file_name().unwrap().to_string_lossy()
            ));

            let mut ds_train = read_csv(train_path, b'\t', false)?;

            let ds_test = read_csv(test_path, b'\t', false)?;

            if (ds_train.len() + ds_test.len() > 200) || (ds_train[0].data.len() > 1000) {
                continue;
            }
            println!(
                "\tProcessing {}",
                path.file_name().unwrap().to_string_lossy()
            );

            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();

            let mut model = DistanceSetForest::new(config);

            let start_time = std::time::Instant::now();
            model.fit(&mut ds_train);
            println!("\t\tTraining time: {:?}", start_time.elapsed());

            let start_time = std::time::Instant::now();
            let y_pred = model.predict(&ds_test);
            println!("\t\tPrediction time: {:?}", start_time.elapsed());

            let acc = accuracy_score(&y_pred, &y_true);
            println!("\tAccuracy: {}", acc);

            wtr.write_record(&[
                path.file_name().unwrap().to_string_lossy().to_string(),
                acc.to_string(),
            ])?;
            wtr.flush()?;
        }
    }
    Ok(())
}
