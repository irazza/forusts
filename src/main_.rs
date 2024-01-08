use crate::forest::canonical_interval_forest::{CanonicalIntervalForestConfig, CanonicalIntervalForest};
use crate::forest::canonical_isolation_forest::CanonicalIsolationForest;
use crate::forest::forest::{ClassificationForest, Forest, ClassificationForestConfig, OutlierForest};
use crate::forest::time_series_forest::{TimeSeriesForest, TimeSeriesForestConfig};
use crate::forest::time_series_isolation_forest::TimeSeriesIsolationForest;
use crate::metrics::classification::{accuracy_score, roc_auc_score};
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::tree::tree::{Criterion, MaxFeatures, Tree};
use crate::utils::csv_io::read_csv;
use csv::WriterBuilder;
use forest::canonical_isolation_forest::CanonicalIsolationForestConfig;
use forest::forest::OutlierForestConfig;
use forest::time_series_isolation_forest::TimeSeriesIsolationForestConfig;
use hashbrown::HashMap;
use std::error::Error;
use std::fs::{self, File};
use utils::csv_io::write_csv;

mod feature_extraction;
mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;

// TODO 
// Check why chinatown, italypowerdemand throws segmentation fault on catch22 features, probably related to time series constant interval??
fn main() -> Result<(), Box<dyn Error>> {
    // Settings for the experiments
    let mut config = CanonicalIsolationForestConfig {
        n_intervals: 0,
        outlier_config: OutlierForestConfig {
            n_trees: 200,
            enhanced_anomaly_score: false,
            max_depth: None,
        }
    };
    let n_repetitions = 10;
    let paths = fs::read_dir("/media/aazzari/DATA/UCRArchive_2018/")?;

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());

    let mut predictions = Vec::new();
    let mut training_scores = Vec::new();
    for path in &datasets {
        println!("Processing {}", path.file_name().to_string_lossy());
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
        config.n_intervals = n_features.log10() as usize;

        let mut mean_accuracy_breiman = 0.0;
        let mut mean_accuracy_ancestor = 0.0;
        let mut mean_accuracy_zhu = 0.0;

        for _i in 0..n_repetitions {
            let mut model = CanonicalIsolationForest::new(config);
            model.fit(&mut ds_train);
            training_scores.push(model.score_samples(&ds_train));

            let breiman_distance =
                model.pairwise_breiman(&ds_test, &ds_train);
            let prediction_breiman =
                k_nearest_neighbor(1, &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(), &breiman_distance);
            let accuracy_breiman = accuracy_score(&prediction_breiman, &y_true);

            let ancestor_distance =
                model.pairwise_ancestor(&ds_test, &ds_train);
            let prediction_ancestor =
                k_nearest_neighbor(1, &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(), &ancestor_distance);
            let accuracy_ancestor = accuracy_score(&prediction_ancestor, &y_true);

            let zhu_distance =
                model.pairwise_zhu(&ds_test, &ds_train);
            let prediction_zhu = k_nearest_neighbor(1, &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(), &zhu_distance);
            let accuracy_zhu = accuracy_score(&prediction_zhu, &y_true);

            predictions.push(
                [
                    accuracy_breiman,
                    accuracy_ancestor,
                    accuracy_zhu,
                ]
                .to_vec(),
            );
            mean_accuracy_breiman += accuracy_breiman;
            mean_accuracy_ancestor += accuracy_ancestor;
            mean_accuracy_zhu += accuracy_zhu;
        }
        println!(
            "\tBreiman: {}, Ancestor: {}, Zhu: {}",
                mean_accuracy_breiman/n_repetitions as f64, mean_accuracy_ancestor/n_repetitions as f64, mean_accuracy_zhu/n_repetitions as f64);
    }
    // Create index modifying datasets multiplyng by n_repetitions
    let mut index = Vec::new();
    for i in 0..datasets.len() {
        for _j in 0..n_repetitions {
            index.push(datasets[i].file_name().to_string_lossy().to_string());
        }
    }
    let header = vec!["Dataset", "Breiman", "Ancestor", "Zhu"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    write_csv(
        format!("experimental_results/ucrCIF_T{}_R{}_I{}_scores.csv", config.outlier_config.n_trees, n_repetitions, config.n_intervals),
        predictions,
        header,
        index,
    )?;

    let file = File::create(format!("experimental_results/ucrCIF_T{}_R{}_I{}_scores.csv", config.outlier_config.n_trees, n_repetitions, config.n_intervals))?;
    let mut csv_writer = WriterBuilder::new().flexible(true).from_writer(file);
    for record in &training_scores {
        csv_writer.serialize(record)?;
    }
    csv_writer.flush()?;

    Ok(())
}
