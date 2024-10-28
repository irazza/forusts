use crate::forest::forest::{Forest, OutlierForest};
use crate::forest::isolation_forest::{IsolationForest, IsolationForestConfig};
use crate::metrics::classification::precision_at_k;
use crate::utils::aggregation::Combiner;
use crate::utils::csv_io::{read_csv, write_csv};
use crate::utils::structures::MaxFeatures;
use rand_chacha::rand_core::SeedableRng;
use std::error::Error;
use std::fs;

mod cluster;
mod forest;
mod metrics;
mod neighbors;
mod tests;
mod tree;
mod utils;

type RandomGenerator = rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn Error>> {
    let n_repetitions = 201;
    let n_combiners = 6;
    let paths = fs::read_dir("../DATA/IF_BENCHMARK").unwrap();

    let mut datasets = Vec::new();
    for entry in paths {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());

    let mut precision_at_10 = vec![vec![0; datasets.len()]; n_combiners * n_repetitions];

    for (i, path) in datasets.iter().enumerate() {
        println!(
            "Dataset: {}",
            path.path().file_stem().unwrap().to_string_lossy()
        );

        let mut ds_train = read_csv(path.path(), b',', false).unwrap();

        let ds_test = ds_train.clone();
        let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();
        let n_anomalies = y_true.iter().filter(|&&x| x == 1).count();
        for (j, combiner) in [
            Combiner::PROD,
            Combiner::SUM,
            Combiner::TRIMMEDSUM,
            Combiner::MEDIAN,
            Combiner::MIN,
            Combiner::MAX,
        ]
        .iter()
        .enumerate()
        {
            let config = IsolationForestConfig {
                n_trees: 100,
                max_depth: None,
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_samples: 1.0,
                max_features: MaxFeatures::ALL,
                criterion: |_a, _b| 1.0,
                aggregation: Some(combiner.clone()),
            };

            for k in 0..n_repetitions {
                let mut model = IsolationForest::new(&config);
                model.fit(
                    &mut ds_train,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                        ((i + 2) * (j + 2) * (k + 2)) as u64,
                    )),
                );
                precision_at_10[j * n_repetitions + k][i] =
                    (precision_at_k(&model.score_samples(&ds_test), &y_true, n_anomalies)
                        * n_anomalies as f64) as usize;
            }
        }
    }
    write_csv(
        "../DATA/precision_at_10.csv",
        precision_at_10,
        Some(
            datasets
                .iter()
                .map(|x| x.path().file_stem().unwrap().to_string_lossy().to_string())
                .collect::<Vec<_>>(),
        ),
    );
    Ok(())
}
