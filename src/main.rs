use hashbrown::HashMap;
use utils::csv_io::read_csv_to_vec;

use crate::feature_extraction::{catch22::compute_catch_features, scamp::compute_scamp};
use crate::forest::canonical_isolation_forest::{CanonicalIsolationForest, CanonicalIsolationForestConfig};
use crate::forest::canonical_sc_isolation_forest::{
    CanonicalSCIsolationForest, CanonicalSCIsolationForestConfig,
};
use crate::forest::forest::{Forest, OutlierForest, OutlierForestConfig};
use crate::forest::sc_isolation_forest::{SCIsolationForest, SCIsolationForestConfig};
use crate::metrics::classification::roc_auc_score;
use crate::utils::csv_io::read_csv;
use crate::utils::structures::Sample;
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
    let (mp, idxs) = compute_scamp(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 3);
    panic!();
    // Settings for the experiments
    let n_repetitions = 10;
    let paths = fs::read_dir("/media/aazzari/DATA/UCRArchive_2018")?;
    let mut candidate_reader = csv::Reader::from_path("candidate.csv")?;
    let mut candidates = Vec::new();
    for result in candidate_reader.deserialize() {
        let record: String = result.unwrap();
        candidates.push(record);
    }

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let current_ds = entry.file_name().to_string_lossy().to_string();
            for candidate in &candidates {
                if current_ds.eq(candidate) {
                    datasets.push(entry.path());
                }
            }
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().unwrap().to_string_lossy().to_string());
    assert!(datasets.len() == candidates.len());
    let mut wtr = csv::Writer::from_path("CSCIForest_heavy.csv")?;
    wtr.write_record(&["Dataset", "ROC-AUC"])?;
    wtr.flush()?;
    let mut bw = csv::WriterBuilder::new()
        .flexible(true)
        .from_path("CSCIForest_heavy_scores.csv")?;
    for i in 0..n_repetitions {
        println!("Repetition {}", i + 1);
        //let mut predictions = Vec::new();
        for path in &datasets {
            println!("\tProcessing {}", path.file_name().unwrap().to_string_lossy());
            let train_path = path
                .join(format!("{}_TRAIN.tsv", path.file_name().unwrap().to_string_lossy()));
            let test_path = path
                .join(format!("{}_TEST.tsv", path.file_name().unwrap().to_string_lossy()));

            let mut ds_train = read_csv(train_path, b'\t', false)?;
            let mut ds_test = read_csv(test_path, b'\t', false)?;
            // Get frequency of classes
            let mut mapping = HashMap::new();
            for sample in &ds_train {
                *mapping.entry(sample.target).or_insert(0) += 1;
            }
            for sample in &ds_test {
                *mapping.entry(sample.target).or_insert(0) += 1;
            }
            // Most frequent class is 0, the other is 1
            let majority = mapping.iter().max_by_key(|(_, v)| **v).unwrap().0;
            for sample in &mut ds_test {
                sample.target = if sample.target == *majority { 0 } else { 1 };
            }
            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();

            let n_features = ds_train[0].data.len() as f64;

            let config = CanonicalSCIsolationForestConfig {
                n_intervals: n_features.sqrt() as usize,
                outlier_config: OutlierForestConfig {
                    n_trees: 500,
                    enhanced_anomaly_score: false,
                    max_depth: Some(usize::MAX),
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
                path.file_name().unwrap().to_string_lossy().to_string(),
                roc_auc.to_string(),
            ])?;
            wtr.flush()?;
        }
    }
    Ok(())
}
