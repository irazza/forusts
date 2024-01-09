use crate::forest::canonical_isolation_forest::{
    CanonicalIsolationForest, CanonicalIsolationForestConfig,
};
use crate::forest::forest::{Forest, OutlierForest, OutlierForestConfig};
use crate::feature_extraction::statistics::{mean, std};
use crate::metrics::classification::roc_auc_score;
use crate::utils::csv_io::{read_csv, vec_to_csv};
use crate::utils::structures::Sample;

use std::borrow::Cow;
use std::error::Error;
use std::fs::{self, File};
use csv::WriterBuilder;
use utils::csv_io::write_csv;

mod feature_extraction;
mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;

fn main() -> Result<(), Box<dyn Error>> {
    let paths = fs::read_dir("/media/aazzari/DATA/admep/")?;
    // Store ROC-AUC results
    let mut predictions = Vec::new();
    let mut training_scores = Vec::new();
    // let mut hyperparameters = Vec::new();
    let n_repetitions = 5;
    let n_trees = 200;
    let mut config = CanonicalIsolationForestConfig {
        outlier_config: OutlierForestConfig {
            n_trees,
            enhanced_anomaly_score: false,
            max_depth: None,
        },
        n_intervals: 0,
    };

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
        config.n_intervals = n_features.ln() as usize;

        let mut mean_roc = 0.0;
        for _i in 0..n_repetitions {
            let mut clf = CanonicalIsolationForest::new(config);
            clf.fit(&mut ds_train);

            let scores = clf.score_samples(&ds_train);
            training_scores.push(scores);

            let roc_auc = roc_auc_score(&clf.score_samples(&ds_test), &y_true);
            mean_roc += roc_auc;
            predictions.push([roc_auc].to_vec());
        }
        mean_roc /= n_repetitions as f64;
        println!("\tMean ROC-AUC: {}", mean_roc);
    }
    // Create index
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
        format!("experimental_results/admepCIF_T{}_R{}_I{}.csv", n_trees, n_repetitions,config.n_intervals),
        predictions,
        header,
        index.clone(),
    )?;
    
    let file = File::create(format!("experimental_results/admepCIF_T{}_R{}_I{}_scores.csv", n_trees, n_repetitions,config.n_intervals))?;
    let mut csv_writer = WriterBuilder::new().flexible(true).from_writer(file);
    for record in &training_scores {
        csv_writer.serialize(record)?;
    }
    csv_writer.flush()?;


    Ok(())
}
