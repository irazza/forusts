use crate::forest::canonical_interval_forest::{
    CanonicalIntervalForest, CanonicalIntervalForestConfig,
};
use crate::forest::canonical_isolation_forest::{
    CanonicalIsolationForest, CanonicalIsolationForestConfig,
};

use crate::forest::extremely_randomized_canonical_interval_forest::{
    ExtremelyRandomizedCanonicalIntervalForest, ExtremelyRandomizedCanonicalIntervalForestConfig,
};
use crate::forest::forest::{
    ClassificationForest, ClassificationForestConfig, Forest, OutlierForest, OutlierForestConfig,
};

use crate::forest::time_series_forest::{TimeSeriesForest, TimeSeriesForestConfig};
use crate::forest::time_series_isolation_forest::{
    TimeSeriesIsolationForest, TimeSeriesIsolationForestConfig,
};
use crate::metrics::classification::{accuracy_score, roc_auc_score};
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::tree::tree::{Criterion, MaxFeatures};
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
    // Settings for the experiments
    let n_repetitions = 1;
    let paths = fs::read_dir("../DATA/admep")?;

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    for i in 0..n_repetitions {
        println!("Repetition {}", i + 1);
        //let mut predictions = Vec::new();
        for path in &datasets[1..] {
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

            let config = TimeSeriesIsolationForestConfig {
                n_intervals: n_features.sqrt() as usize,
                outlier_config: OutlierForestConfig {
                    n_trees: 100,
                    min_samples_split: 2,
                    max_depth: None,
                    enhanced_anomaly_score: false,
                    max_samples: 256,
                },
            };
            let mut model = TimeSeriesIsolationForest::new(config);
            let start_time = std::time::Instant::now();
            model.fit(&mut ds_train);
            println!("\t\tTraining time: {:?}", start_time.elapsed());
            let start_time = std::time::Instant::now();
            let prediction = model.score_samples(&ds_test);
            println!("\t\tPrediction time: {:?}", start_time.elapsed());
            let accuracy = roc_auc_score(&prediction, &y_true);
            println!("\tROC AUC: {}", accuracy);
            // let prediction = model.predict(&ds_test);
            // println!("\t\tPrediction time: {:?}", start_time.elapsed());
            // let accuracy = accuracy_score(&prediction, &y_true);
            // println!("\tACC: {}", accuracy);
            // let start_time = std::time::Instant::now();
            // let ratiorf_distance = model.pairwise_ratiorf(&ds_test, &ds_train);
            // println!("\t\tDistance Matrix time: {:?}", start_time.elapsed());
            // let start_time = std::time::Instant::now();
            // let prediction_ratiorf = k_nearest_neighbor(
            //     1,
            //     &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(),
            //     &ratiorf_distance,
            // );
            // println!("\t\tPrediction time: {:?}", start_time.elapsed());
            // let accuracy_ratiorf = accuracy_score(&prediction_ratiorf, &y_true);
            // println!("\tACC: {}", accuracy_ratiorf);
            panic!();
        }
    }
    Ok(())
}

// use crate::feature_extraction::statistics::EULER_MASCHERONI;
// use crate::forest::extremely_randomized_canonical_interval_forest::{ExtremelyRandomizedCanonicalIntervalForest, ExtremelyRandomizedCanonicalIntervalForestConfig};
// use crate::forest::forest::{Forest, OutlierForest};
// use crate::forest::isolation_forest::{IsolationForest, IsolationForestConfig};
// use crate::metrics::classification::{roc_auc_score, roc_auc_score_c};
// use crate::tree::tree::Tree;
// use crate::utils::csv_io::read_csv;
// use csv::Writer;
// use feature_extraction::statistics::transpose;
// use std::error::Error;
// use std::fs::{self};

// mod distance;
// mod feature_extraction;
// mod forest;
// mod metrics;
// mod neighbors;
// mod tree;
// mod utils;

// fn main() -> Result<(), Box<dyn Error>> {
//     // Settings for the experiments
//     let paths = fs::read_dir("../DATA/ADDatasets/")?;

//     let mut datasets = Vec::new();
//     for entry in paths {
//         // Unwrap the entry or handle the error, if any.
//         let entry = entry?;
//         datasets.push(entry.path());
//     }
//     datasets.sort_by_key(|dir| dir.file_name().unwrap().to_string_lossy().to_string());
//     for path in &datasets {
//         //let mut roc_aucs = Vec::new();
//         println!(
//             "Processing {}",
//             path.file_name().unwrap().to_string_lossy()
//         );
//         let mut ds = read_csv(path, b',', false)?;

//         let y_true = ds.iter().map(|s| s.target).collect::<Vec<_>>();

//         // let config = IsolationForestConfig {
//         //     n_trees: 1000,
//         //     enhanced_anomaly_score: false,
//         //     max_depth: None,

//         // };
//         // let mut model = IsolationForest::new(config);
//         // model.fit(&mut ds);
//         // let scores = model.score_samples(&ds);
//         // let roc_auc = roc_auc_score( &scores, &y_true);
//         // println!("\tROC AUC: {}", roc_auc);
//         // anomaly_scores.push(y_true.clone().iter().map(|s| *s as f64).collect::<Vec<_>>());
//         // // Transpose the anomaly scores
//         // let transposed = transpose(anomaly_scores.clone());
//         // // Get the name of the dataset
//         // let out_path = path.file_stem().unwrap().to_string_lossy();
//         // fs::create_dir_all(format!("s+sspr24/{}", out_path))?;
//         // let mut scores_wrt = Writer::from_path(format!("s+sspr24/{}/{}_scores.csv", out_path, out_path))?;
//         // for i in 0..transposed.len() {
//         //     scores_wrt.write_record(transposed[i].iter().map(|s| s.to_string()).collect::<Vec<_>>())?;
//         // }
//         // scores_wrt.flush()?;
//     }
//     Ok(())
// }

// // let denominator = (2.0 * (f64::ln(model.get_max_samples() as f64 - 1.0) + EULER_MASCHERONI))
//         //     - 2.0 * ((model.get_max_samples() as f64 - 1.0) / model.get_max_samples() as f64);
//         // let mut ans = Vec::new();
//         // for i in 0..anomaly_scores[0].len() {
//         //     let mut ascore = 0.0;        //     for t in anomaly_scores.iter() {
//         //         ascore += t[i];
//         //     }
//         //     ascore /= denominator;
//         //     ans.push(2.0f64.powf(-ascore/model.get_trees().len() as f64));
//         // }

//         // let scores = model.score_samples(&ds);

//         // assert_eq!(ans.len(), y_true.len());
