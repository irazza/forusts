use crate::forest::canonical_isolation_forest::{
    CanonicalIsolationForest, CanonicalIsolationForestConfig,
};
use crate::forest::canonical_sc_isolation_forest::{
    CanonicalSCIsolationForest, CanonicalSCIsolationForestConfig,
};
use crate::forest::extra_canonical_forest::ExtraCanonicalForest;
use crate::forest::forest::{ClassificationForest, Forest, OutlierForest, OutlierForestConfig};
use crate::forest::sc_isolation_forest::{SCIsolationForest, SCIsolationForestConfig};
use crate::metrics::classification::{accuracy_score, roc_auc_score};
use crate::neighbors::local_outlier_factor::local_outlier_factor;
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::utils::csv_io::{read_csv, vec_vec_to_csv};
use forest::extra_canonical_forest::ExtraCanonicalForestConfig;
use forest::forest::ClassificationForestConfig;
use rayon::vec;
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
    // let mut predictions = Vec::new();
    // Settings for the experiments
    let n_repetitions = 10;
    let paths = fs::read_dir("/media/aazzari/DATA/admep")?;

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    let mut wtr = csv::Writer::from_path("admep_raw.csv")?;
    wtr.write_record(&["Dataset", "ROC-AUC"])?;
    wtr.flush()?;
    let mut bw = csv::WriterBuilder::new().flexible(true)
        .from_path("admep_raw_scores.csv")?;
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
                n_intervals: n_features.log2() as usize,
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
            wtr.write_record(&[
                path.file_name().to_string_lossy().to_string(),
                roc_auc.to_string(),
            ])?;
            wtr.flush()?;
        }
    }
    Ok(())
}
