use crate::forest::canonical_sc_isolation_forest::{
    CanonicalSCIsolationForest, CanonicalSCIsolationForestConfig,
};
use crate::forest::forest::{ClassificationForest, Forest, OutlierForest, OutlierForestConfig};
use crate::metrics::classification::{accuracy_score, roc_auc_score};
use crate::utils::csv_io::{read_csv, vec_vec_to_csv};
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
    // let mut predictions = Vec::new();
    // Settings for the experiments
    let n_repetitions = 10;
    let paths = fs::read_dir("/Users/albertoazzari/projects/MEP_cascade/admep")?;

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    let mut wtr = csv::Writer::from_path("admep.csv")?;
    wtr.write_record(&["Dataset", "ROC-AUC"])?;
    wtr.flush()?;
    let mut bw = csv::WriterBuilder::new()
        .flexible(true)
        .from_path("admep_scores.csv")?;
    for i in 0..n_repetitions {
        println!("Repetition {}", i + 1);
        //let mut predictions = Vec::new();
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

            let n_features = ds_train[0].data.len() as f64;

            let config = CanonicalSCIsolationForestConfig {
                n_intervals: n_features.log10() as usize,
                outlier_config: OutlierForestConfig {
                    n_trees: 200,
                    enhanced_anomaly_score: false,
                    max_depth: None,
                },
            };
            let mut model = CanonicalSCIsolationForest::new(config);
            model.fit(&mut ds_train);

            let y_pred = model.score_samples(&ds_test);
            bw.write_record(y_pred.iter().map(|v| v.to_string()))?;
            bw.flush()?;
            let roc_auc = roc_auc_score(&y_pred, &y_true);
            println!("\t\tROC-AUC: {}", roc_auc);
            panic!();
            wtr.write_record(&[
                path.file_name().to_string_lossy().to_string(),
                roc_auc.to_string(),
            ])?;
            wtr.flush()?;
        }
    }
    Ok(())
}
