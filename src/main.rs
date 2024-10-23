use crate::forest::forest::Forest;
use crate::forest::random_forest::{RandomForest, RandomForestConfig};
use crate::metrics::classification::accuracy_score;
use crate::utils::csv_io::read_csv;
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
    // Settings for the experiments
    let config = RandomForestConfig {
        n_trees: 100,
        max_depth: None,
        min_samples_split: 2,
        min_samples_leaf: 1,
        max_samples: 1.0,
        max_features: MaxFeatures::ALL,
        criterion: |_a, _b| f64::NAN,
        aggregation: None,
    };
    let n_repetitions = 1;
    let paths = fs::read_dir("../DATA/ucr").unwrap();

    let mut datasets = Vec::new();
    for entry in paths {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    let mut predictions = vec![0.0; datasets.len()];
    for (i, path) in datasets[..5].iter().enumerate() {
        let mut ds_train = read_csv(
            path.path()
                .join(format!("{}_TRAIN.tsv", path.file_name().to_string_lossy())),
            b'\t',
            false,
        )
        .unwrap();
        let ds_test = read_csv(
            path.path()
                .join(format!("{}_TEST.tsv", path.file_name().to_string_lossy())),
            b'\t',
            false,
        )
        .unwrap();
        let start_time = std::time::Instant::now();
        for j in 0..n_repetitions {
            let mut model = RandomForest::new(&config);
            model.fit(
                &mut ds_train,
                Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                    ((i + 2) * (j + 2)) as u64,
                )),
            );
            let prediction = model.predict(&ds_test);
            predictions[i] += accuracy_score(
                &prediction,
                &ds_test.iter().map(|s| s.target).collect::<Vec<_>>(),
            );
        }
        println!(
            "{}: {:.2} in {:.2} seconds",
            path.file_name().to_string_lossy(),
            predictions[i] / n_repetitions as f64,
            start_time.elapsed().as_secs_f64(),
        );
    }
    Ok(())
}
