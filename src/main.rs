
use crate::forest::extra_canonical_forest::{ExtraCanonicalForest, ExtraCanonicalForestConfigTuning};
use crate::forest::forest::{DistanceForest, DistanceForestConfigTuning, Forest};
use crate::metrics::classification::accuracy_score;
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::utils::csv_io::read_csv;
use crate::utils::tuning::grid_search;
use forest::extra_canonical_forest::ExtraCanonicalForestConfig;
use forest::forest::{DistanceForestConfig};
use hashbrown::HashMap;
use core::panic;
use std::error::Error;
use std::fs::{self};
use utils::csv_io::write_csv;

mod feature_extraction;
mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;
mod distance;

fn main() -> Result<(), Box<dyn Error>> {

    let mut predictions = Vec::new();
    let mut config = ExtraCanonicalForestConfig {
        n_intervals: 0,
        classification_config: DistanceForestConfig {
            n_trees: 0,
            max_depth: None,
            min_samples_split: 2,
            max_features: tree::tree::MaxFeatures::Sqrt,
        },
    };
    // Settings for the experiments
    let n_repetitions = 1;
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

    //let mut predictions = Vec::new();
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

        
        let mut mean_accuracy_breiman = 0.0;
        let mut max_accuracy_breiman = 0.0;
        let mut mean_accuracy_ancestor = 0.0;
        let mut max_accuracy_ancestor = 0.0;
        let mut mean_accuracy_zhu = 0.0;
        let mut max_accuracy_zhu = 0.0;
        let mut mean_accuracy_ratiorf = 0.0;
        let mut max_accuracy_ratiorf = 0.0;

        config = ExtraCanonicalForestConfig {
            n_intervals: n_features.log2() as usize,
            classification_config: DistanceForestConfig {
                n_trees: 200,
                max_depth: None,
                min_samples_split: 2,
                max_features: tree::tree::MaxFeatures::Sqrt,
            },
        };

        for _i in 0..n_repetitions {
            let mut model = ExtraCanonicalForest::new(config);
            model.fit(&mut ds_train);

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

            let ratiorf_distance =
                model.pairwise_ratiorf(&ds_test, &ds_train);
            let prediction_ratiorf = k_nearest_neighbor(1, &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(), &ratiorf_distance);
            let accuracy_ratiorf = accuracy_score(&prediction_ratiorf, &y_true);

            predictions.push(
                [
                    accuracy_breiman,
                    accuracy_ancestor,
                    accuracy_zhu,
                    accuracy_ratiorf,
                ]
                .to_vec(),
            );

            mean_accuracy_breiman += accuracy_breiman;
            mean_accuracy_ancestor += accuracy_ancestor;
            mean_accuracy_zhu += accuracy_zhu;
            mean_accuracy_ratiorf += accuracy_ratiorf;

            if accuracy_breiman > max_accuracy_breiman {
                max_accuracy_breiman = accuracy_breiman;
            }
            if accuracy_ancestor > max_accuracy_ancestor {
                max_accuracy_ancestor = accuracy_ancestor;
            }
            if accuracy_zhu > max_accuracy_zhu {
                max_accuracy_zhu = accuracy_zhu;
            }
            if accuracy_ratiorf > max_accuracy_ratiorf {
                max_accuracy_ratiorf = accuracy_ratiorf;
            }
        }
        println!(
            "MEAN\tBreiman: {}, Ancestor: {}, Zhu: {}, RatioRF: {}\nMAX\tBreiman: {}, Ancestor: {}, Zhu: {}, RatioRF: {}",
                mean_accuracy_breiman/n_repetitions as f64, mean_accuracy_ancestor/n_repetitions as f64, mean_accuracy_zhu/n_repetitions as f64, mean_accuracy_ratiorf/n_repetitions as f64, max_accuracy_breiman, max_accuracy_ancestor, max_accuracy_zhu, max_accuracy_ratiorf);
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
        format!("experimental_results/ucrCIF_T{}_R{}_I{}.csv", config.classification_config.n_trees, n_repetitions, config.n_intervals),
        predictions,
        header,
        index,
    )?;
    Ok(())
}
