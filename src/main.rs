use crate::forest::forest::{Forest, OutlierForest};
use crate::utils::csv_io::{read_csv, write_csv};
use crate::utils::structures::MaxFeatures;
use forest::ciso_forest::{CIsoForest, CIsoForestConfig};
use forest::forest::ForestConfig;
use metrics::classification::roc_auc_score;
use rand_chacha::rand_core::SeedableRng;
use std::error::Error;
use std::fs;
use std::sync::Arc;
use tree::transform::zscore;
use utils::structures::{IntervalType, Sample};

mod cluster;
mod forest;
mod metrics;
mod neighbors;
mod tests;
mod tree;
mod utils;

type RandomGenerator = rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn Error>> {
    // Settings for the experiments
    let config = CIsoForestConfig {
        n_intervals: IntervalType::SQRT,
        n_attributes: 4,
        outlier_config: ForestConfig {
            n_trees: 500,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_samples: 1.0,
            max_features: MaxFeatures::ALL,
            criterion: |_a, _b| 1.0,
            aggregation: None,
        },
    };
    let n_repetitions = 10;
    let paths = fs::read_dir("../../DATA/ADMEP/").unwrap();

    let mut datasets = Vec::new();
    for entry in paths {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    let mut predictions = vec![vec![0.0; n_repetitions]; datasets.len()];

    for (i, path) in datasets.iter().enumerate() {
        let ds_train = read_csv(
            path.path()
                .join(format!("{}_TRAIN.csv", path.file_name().to_string_lossy())),
            b',',
            false,
        )
        .unwrap();
        let mut ds_train = ds_train
            .iter()
            .map(|s| Sample {
                features: Arc::new(zscore(&s.features)),
                target: s.target,
            })
            .collect::<Vec<_>>();
        let ds_test = read_csv(
            path.path()
                .join(format!("{}_TEST.csv", path.file_name().to_string_lossy())),
            b',',
            false,
        )
        .unwrap();
        let ds_test = ds_test
            .iter()
            .map(|s| Sample {
                features: Arc::new(zscore(&s.features)),
                target: s.target,
            })
            .collect::<Vec<_>>();
        let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();
        for j in 0..n_repetitions {
            let mut model = CIsoForest::new(&config);
            model.fit(
                &mut ds_train,
                Some(rand_chacha::ChaCha8Rng::seed_from_u64(j as u64)),
            );
            let prediction = model.score_samples(&ds_test);
            predictions[i][j] = roc_auc_score(&prediction, &y_true);
        }
        println!(
            "{}: {:.2}",
            path.file_name().to_string_lossy(),
            predictions[i].iter().sum::<f64>() / n_repetitions as f64
        );
    }
    println!(
        "Mean: {}",
        predictions.iter().flatten().sum::<f64>() / (n_repetitions * datasets.len()) as f64
    );
    // add first column with dataset names
    let predictions = datasets
        .iter()
        .zip(predictions)
        .map(|(d, p)| {
            let mut row = vec![d.file_name().to_string_lossy().to_string()];
            row.extend(p.iter().map(|v| v.to_string()));
            row
        })
        .collect::<Vec<Vec<_>>>();
    write_csv("cisof_H_R.csv", predictions, None);
    Ok(())
}
