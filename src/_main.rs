#![allow(unused_imports)]
use crate::forest::canonical_interval_forest::CanonicalIntervalForest;
use crate::forest::forest::Forest;
use crate::forest::isolation_forest::IsolationForest;
use crate::forest::random_forest::RandomForest;
use crate::forest::time_series_forest::TimeSeriesForest;
use crate::forest::time_series_isolation_forest::TimeSeriesIsolationForest;
use crate::metrics::classification::{accuracy_score, matthews_corrcoef};
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::tree::decision_tree::{Criterion, MaxFeatures, Splitter};
use crate::utils::csv_io::read_csv;
use hashbrown::HashMap;
use std::error::Error;
use std::fs;
use utils::csv_io::write_csv;

mod feature_extraction;
mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;

fn main() -> Result<(), Box<dyn Error>> {
    let paths = fs::read_dir("AD/")?;
    let n_repetitions = 1;
    let n_trees = 200;

    let mut datasets: Vec<_> = Vec::new();

    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    for path in &datasets {
        println!("Processing {}", path.file_name().to_string_lossy());
        let train_path = path
            .path()
            .join(format!("{}_TRAIN.tsv", path.file_name().to_string_lossy()));
        let test_path = path
            .path()
            .join(format!("{}_TEST.tsv", path.file_name().to_string_lossy()));

        let mut mapping = HashMap::new();
        let ds_train = read_csv(train_path, b'\t', &mut mapping)?;
        let n_features = ds_train.get_data()[0].len() as f64;
        let ds_test = read_csv(test_path,  b'\t', &mut mapping)?;
        let y_true = ds_test.get_targets().clone();

        for _i in 0..n_repetitions {
            let mut clf = TimeSeriesIsolationForest::new(n_trees, n_features.sqrt() as usize,
            3, MaxFeatures::All, None);

            clf.fit(&ds_train.get_data());

            let y_pred = clf.predict(&ds_test.get_data());
            let mcc = matthews_corrcoef(&y_pred, &y_true);
            let acc = accuracy_score(&y_pred, &y_true);
            println!("MCC: {}, Accuracy: {}", mcc, acc);
        }
    }

    Ok(())
}
