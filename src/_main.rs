#![allow(unused_imports, unused_variables)]
use crate::feature_extraction::statistics::unique;
use crate::forest::canonical_interval_forest::CanonicalIntervalForest;
use crate::forest::forest::Forest;
use crate::forest::isolation_forest::IsolationForest;
use crate::forest::random_forest::RandomForest;
use crate::forest::time_series_forest::TimeSeriesForest;
use crate::forest::time_series_isolation_forest::TimeSeriesIsolationForest;
use crate::metrics::classification::{accuracy_score, matthews_corrcoef, roc_auc_score};
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::tree::decision_tree::{Criterion, MaxFeatures, Splitter};
use crate::utils::csv_io::read_csv;
use csv::ReaderBuilder;
use hashbrown::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::BufReader;
use std::process::exit;
use utils::csv_io::write_csv;

mod feature_extraction;
mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;

fn main() -> Result<(), Box<dyn Error>> {
    let paths = fs::read_dir("ADMEP/")?;
    let mut predictions = Vec::new();
    let n_repetitions = 1;
    let n_trees = 500;

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
        let ds_test = read_csv(test_path, b'\t', &mut mapping)?;
        let y_true = ds_test.get_targets().clone();
        if unique(&y_true[..]).len() == 1 {
            println!("Only one class in y_true: {}", y_true[0]);
            continue;
        }

        for _i in 0..n_repetitions {
            let mut clf = TimeSeriesIsolationForest::new(
                n_trees,
                n_features.sqrt() as usize,
                3,
                MaxFeatures::All,
                None,
            );
            //let mut clf = IsolationForest::new(n_trees, MaxFeatures::All, None);

            clf.fit(&ds_train.get_data());

            let y_score = clf.score_samples(&ds_test.get_data());
            let accuracy = roc_auc_score(&y_score, &y_true);

            let breiman_distance =
                clf.pairwise_breiman(ds_test.get_data().clone(), ds_train.get_data().clone());
            let prediction_breiman =
                k_nearest_neighbor(1, &ds_train.get_targets(), &breiman_distance);
            let accuracy_breiman = matthews_corrcoef(&prediction_breiman, &y_true);

            let ancestor_distance =
                clf.pairwise_ancestor(ds_test.get_data().clone(), ds_train.get_data().clone());
            let prediction_ancestor =
                k_nearest_neighbor(1, &ds_train.get_targets(), &ancestor_distance);
            let accuracy_ancestor = matthews_corrcoef(&prediction_ancestor, &y_true);

            let zhu_distance =
                clf.pairwise_zhu(ds_test.get_data().clone(), ds_train.get_data().clone());
            let prediction_zhu = k_nearest_neighbor(1, &ds_train.get_targets(), &zhu_distance);
            let accuracy_zhu = matthews_corrcoef(&prediction_zhu, &y_true);
            predictions
                .push([accuracy, accuracy_breiman, accuracy_ancestor, accuracy_zhu].to_vec());
        }
    }
    // Create index modifying datasets multiplyng by n_repetitions
    let mut index = Vec::new();
    for i in 0..datasets.len() {
        for _j in 0..n_repetitions {
            index.push(datasets[i].file_name().to_string_lossy().to_string());
        }
    }

    let header = vec!["Dataset", "ROC-AUC", "Breiman", "Ancestor", "Zhu"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    write_csv(format!("admep_{}.csv", n_trees), predictions, header, index)?;

    Ok(())
}
