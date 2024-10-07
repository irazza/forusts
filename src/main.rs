use crate::forest::forest::{Forest, OutlierForest};
use crate::metrics::classification::roc_auc_score;
use crate::utils::csv_io::read_csv;
use forest::ci_forest::{CIForest, CIForestConfig};
use forest::forest::OutlierForestConfig;
use rand::SeedableRng;
use std::error::Error;
use std::fs::{self};

mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;
mod tests;

type RandomGenerator = rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn Error>> {
    // Settings for the experiments
    let config = CIForestConfig {
        n_intervals: 3,
        n_attributes: 8,
        outlier_config: OutlierForestConfig{
            n_trees: 500,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_samples: 1.0,
            max_features: |x| x,
            criterion: |_a, _b| 1.0,
        },
    };
    let n_repetitions = 10;
    let paths = fs::read_dir("/media/DATA/albertoazzari/admep_old/")?;

    let mut datasets = Vec::new();
    for entry in paths {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            datasets.push(entry);
        }
    }
    let mut wtr = csv::Writer::from_path("cif.csv")?;
    wtr.write_record(&["Dataset", "ROC-AUC"])?;
    wtr.flush()?;
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    let mut predictions = vec![0.0; datasets.len()];
    for _i in 0..n_repetitions {
        for (j, path) in datasets.iter().enumerate() {
            let mut ds_train = read_csv(path.path().join(format!("{}_TRAIN.tsv", path.file_name().to_string_lossy())), b'\t', false)?;
            let ds_test = read_csv(path.path().join(format!("{}_TEST.tsv", path.file_name().to_string_lossy())), b'\t', false)?;
            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();

            let mut model = CIForest::new(&config);
            model.fit(
                &mut ds_train,
                Some(rand_chacha::ChaCha8Rng::seed_from_u64(0)),
            );
            let prediction = model.score_samples(&ds_test);
            predictions[j] += roc_auc_score(&prediction, &y_true);
            println!("{}: {:.2}", path.file_name().to_string_lossy(), roc_auc_score(&prediction, &y_true));
        }
    }
    let predictions = predictions
        .iter()
        .map(|x| x / n_repetitions as f64)
        .collect::<Vec<_>>();
    println!("Mean ROC-AUC: {:.2}", predictions.iter().sum::<f64>() / predictions.len() as f64);
    for (path, mean) in datasets.iter().zip(predictions.iter()) {
        wtr.write_record(&[
            path.file_name().to_string_lossy().to_string(),
            mean.to_string(),
        ])?;
        wtr.flush()?;
    }
    Ok(())
}