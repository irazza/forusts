use forest::isolation_forest::IsolationForestConfig;
use crate::forest::isolation_forest::IsolationForest;
use crate::forest::forest::{Forest, OutlierForest};
use crate::metrics::classification::roc_auc_score;
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
    let config = IsolationForestConfig {
        n_trees: 100,
        min_samples_split: 2,
        max_depth: None,
        enhanced_anomaly_score: false,
        max_samples: 1.0,
    };
    let n_repetitions = 100;
    let paths = fs::read_dir("/media/DATA/albertoazzari/IFDatasets")?;

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_file() {
            datasets.push(entry);
        }
    }
    let mut wtr = csv::Writer::from_path("if.csv")?;
    wtr.write_record(&["Dataset", "ROC-AUC"])?;
    wtr.flush()?;
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    for i in 0..n_repetitions {
        println!("Repetition {}", i + 1);
        //let mut predictions = Vec::new();
        for path in &datasets[5..6] {
            println!("\tProcessing {}", path.file_name().to_string_lossy());

            let mut ds_train = read_csv(path.path(), b',', false)?;
            let ds_test = read_csv(path.path(), b',', false)?;
            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();

            let mut model = IsolationForest::new(config);
            let start_time = std::time::Instant::now();
            model.fit(&mut ds_train);
            println!("\t\tTraining time: {:?}", start_time.elapsed());
            let start_time = std::time::Instant::now();
            let prediction = model.score_samples(&ds_test);
            println!("\t\tPrediction time: {:?}", start_time.elapsed());
            let roc = roc_auc_score(&prediction, &y_true);
            println!("\tROC AUC: {}", roc);
            wtr.write_record(&[
                path.file_name().to_string_lossy().to_string(),
                roc.to_string(),
            ])?;
            wtr.flush()?;
        }
    }
    Ok(())
}