use crate::forest::distance_forest::{DistanceForest, DistanceForestConfig};
use crate::forest::forest::Forest;
use crate::metrics::classification::accuracy_score;
use crate::utils::csv_io::read_csv;
use rand::SeedableRng;
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
    // Settings for the experiments
    let n_repetitions = 30;
    let paths = fs::read_dir("/media/aazzari/DATA/UCRArchive_2018")?;
    // let mut candidate_reader = csv::Reader::from_path("candidates.csv")?;
    // let mut candidates = Vec::new();
    // for result in candidate_reader.deserialize() {
    //     let record: String = result.unwrap();
    //     candidates.push(record);
    // }

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        datasets.push(entry.path());
    }
    datasets.sort_by_key(|dir| dir.file_name().unwrap().to_string_lossy().to_string());
    let mut wtr = csv::Writer::from_path("test.csv")?;
    wtr.write_record(&["Dataset", "ROC-AUC"])?;
    wtr.flush()?;
    let mut bw = csv::WriterBuilder::new()
        .flexible(true)
        .from_path("test_scores.csv")?;
    let mut m = 0.0;
    for i in 0..n_repetitions {
        println!("Repetition {}", i + 1);
        //let mut predictions = Vec::new();
        for path in &datasets[0..1] {
            println!(
                "\tProcessing {}",
                path.file_name().unwrap().to_string_lossy()
            );
            let train_path = path.join(format!(
                "{}_TRAIN.tsv",
                path.file_name().unwrap().to_string_lossy()
            ));
            let test_path = path.join(format!(
                "{}_TEST.tsv",
                path.file_name().unwrap().to_string_lossy()
            ));

            let mut ds_train = read_csv(train_path, b'\t', false)?;
            let ds_test = read_csv(test_path, b'\t', false)?;

            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();

            let config = DistanceForestConfig {
                n_trees: 100,
                min_samples_split: 2,
                max_features: tree::tree::MaxFeatures::All,
                max_depth: None,
                criterion: tree::tree::Criterion::Gini,
            };
            let mut model = DistanceForest::new(config);
            //let mut ds_train = ds_train.iter().map(|x| Sample{data: Cow::Owned(zscore(&x.data)), target: x.target}).collect::<Vec<_>>();
            model.fit(&mut ds_train);
            //let ds_test = ds_test.iter().map(|x| Sample{data: Cow::Owned(zscore(&x.data)), target: x.target}).collect::<Vec<_>>();
            let y_pred = model.predict(&ds_test);
            bw.write_record(y_pred.iter().map(|v| v.to_string()))?;
            bw.flush()?;
            let acc = accuracy_score(&y_pred, &y_true);
            m += acc;
            println!("\t\tAccuracy: {}", acc);
            wtr.write_record(&[
                path.file_name().unwrap().to_string_lossy().to_string(),
                acc.to_string(),
            ])?;
            wtr.flush()?;
        }
    }
    println!("Mean accuracy: {}", m / (n_repetitions as f64));
    Ok(())
}
