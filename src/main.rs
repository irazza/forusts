use crate::feature_extraction::catch22::DN_HistogramMode_5;
use crate::forest::catch_isolation_forest::{CatchIsolationForest, CatchIsolationForestConfig};
use crate::forest::forest::{Forest, OutlierForest, OutlierForestConfig, OutlierForestConfigTuning};
use crate::forest::time_series_isolation_forest::{
    TimeSeriesIsolationForest, TimeSeriesIsolationForestConfig, TimeSeriesIsolationForestConfigTuning,
};
use crate::metrics::classification::roc_auc_score;
use crate::utils::csv_io::read_csv;
use crate::utils::tuning::grid_search;
use core::panic;
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
    let paths = fs::read_dir("/media/aazzari/DATA/admep/")?;
    let mut predictions = Vec::new();
    // let mut hyperparameters = Vec::new();
    let n_repetitions = 10;
    let n_trees = 100;

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

        let mut ds_train = read_csv(train_path, b'\t', false)?;
        let ds_test = read_csv(test_path, b'\t', false)?;
        let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();
        let n_features = ds_train[0].data.len() as f64;
        // let config = TimeSeriesIsolationForestConfigTuning {
        //     n_intervals: (1..=20).step_by(5).collect(),
        //     outlier_config: OutlierForestConfigTuning {
        //         n_trees: (100..=500).step_by(100).collect(),
        //         enhanced_anomaly_score: vec![false],
        //         max_depth: (5..=50).step_by(5).map(Some).collect(),
        //     },
        // };
        //     hyperparameters.push(grid_search(&mut ds_train, &ds_test, config, 1, roc_auc_score));

        // }

        // serde_json::to_writer_pretty(
        //     std::fs::File::create("admepTSIF_hyperparameters.json")?,
        //     &hyperparameters,
        // )?;

        let config = CatchIsolationForestConfig {
            n_intervals: n_features.sqrt() as usize,
            outlier_config: OutlierForestConfig {
                n_trees,
                enhanced_anomaly_score: false,
                max_depth: None,
            }
        };

        for _i in 0..n_repetitions {
            let mut clf = CatchIsolationForest::new(config);
            clf.fit(&mut ds_train);
            let y_score = clf.score_samples(&ds_test);
            let roc_auc = roc_auc_score(&y_score, &y_true);
            predictions.push([roc_auc].to_vec());
        }
    }
    // Create index modifying datasets multiplying by n_repetitions
    let mut index = Vec::new();
    for i in 0..datasets.len() {
        for _j in 0..n_repetitions {
            index.push(datasets[i].file_name().to_string_lossy().to_string());
        }
    }

    let header = vec!["Dataset", "ROC-AUC"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    write_csv(
        format!("admepTSIF_T{}_R{}.csv", n_trees, n_repetitions),
        predictions,
        header,
        index,
    )?;

    Ok(())
}
