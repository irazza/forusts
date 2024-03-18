use crate::forest::distance_forest::{DistanceForest, DistanceForestConfig};
use crate::forest::forest::{Forest, OutlierForest, OutlierForestConfig};
use crate::forest::mp_isolation_forest::{MPIsolationForest, MPIsolationForestConfig};
use crate::forest::sc_isolation_forest::{SCIsolationForest, SCIsolationForestConfig};
use crate::metrics::classification::{accuracy_score, roc_auc_score};
use crate::utils::csv_io::{read_csv, write_csv};
use crate::utils::structures::train_test_split;
use hashbrown::HashMap;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::borrow::Cow;
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
    let mut random_state = rand_chacha::ChaCha8Rng::seed_from_u64(42);
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
        // if entry.file_type()?.is_dir() {
        //     let current_ds = entry.file_name().to_string_lossy().to_string();
        //     for candidate in &candidates {
        //         if current_ds.eq(candidate) {
        //             datasets.push(entry.path());
        //         }
        //     }
        // }
        datasets.push(entry.path());
    }
    datasets.sort_by_key(|dir| dir.file_name().unwrap().to_string_lossy().to_string());
    // assert!(datasets.len() == candidates.len());
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
        for path in &datasets[1..2] {
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
            let mut ds_test = read_csv(test_path, b'\t', false)?;
            // Concate the two datasets
            // let mut ds = Vec::new();
            // ds.append(&mut ds_train);
            // ds.append(&mut ds_test);
            // Get frequency of classes
            // let mut mapping = HashMap::new();
            // for sample in &ds_train {
            //     *mapping.entry(sample.target).or_insert(0) += 1;
            // }
            // for sample in &ds_test {
            //     *mapping.entry(sample.target).or_insert(0) += 1;
            // }
            // // Most frequent class is 0, the other is 1
            // let majority = mapping.iter().max_by_key(|(_, v)| **v).unwrap().0;
            // for sample in &mut ds_train {
            //     sample.target = if sample.target == *majority { 0 } else { 1 };
            // }
            // for sample in &mut ds_test {
            //     sample.target = if sample.target == *majority { 0 } else { 1 };
            // }
            // // Reduce class 1 to at most 10% of the dataset
            // let mut minority = 0;
            // let mut new_ds = Vec::new();
            // for sample in &ds {
            //     if sample.target == 1 {
            //         minority += 1;
            //         if minority <= mapping[majority] / 10 {
            //             new_ds.push(sample.clone());
            //         }
            //     } else {
            //         new_ds.push(sample.clone());
            //     }
            // }
            // // Shuffle the dataset and split it, keeping the same class distribution in the train and test set
            // new_ds.shuffle(&mut rand::thread_rng());
            // let (mut ds_train, ds_test) = train_test_split(&new_ds, 0.2, true, &mut random_state);
            // // Store to file the train and test set
            // let train_path = format!("UCR_Archive2018_AD/{}/{}_TRAIN.tsv", path.file_name().unwrap().to_string_lossy(), path.file_name().unwrap().to_string_lossy());
            // let test_path = format!("UCR_Archive2018_AD/{}/{}_TEST.tsv", path.file_name().unwrap().to_string_lossy(), path.file_name().unwrap().to_string_lossy());
            // write_csv(train_path, &ds_train, None)?;
            // write_csv(test_path, &ds_test, None)?;

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
