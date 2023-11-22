#![allow(unused_imports)]
use crate::forest::canonical_interval_forest::CanonicalIntervalForest;
use crate::forest::forest::Forest;
use crate::forest::random_forest::RandomForest;
use crate::forest::time_series_forest::TimeSeriesForest;
use crate::metrics::classification::accuracy_score;
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::tree::decision_tree::{Criterion, MaxFeatures, Splitter};
use crate::utils::csv_io::read_csv;
use hashbrown::HashMap;
use std::error::Error;
use std::fs;
use structopt::StructOpt;
use utils::csv_io::write_csv;

mod feature_extraction;
mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;

#[derive(Debug, StructOpt)]
#[structopt(name = "FTSD", about = "Forest-based Time-Series Distances")]
struct Opt {
    // Select model
    #[structopt(short, long, default_value = "tsf")]
    model: String,

    // Set number of trees
    #[structopt(short, long, default_value = "200")]
    trees: usize,

    // Set criterion
    #[structopt(short, long, default_value = "gini")]
    criterion: String,

    //Set splitter
    #[structopt(short, long, default_value = "best")]
    splitter: String,

    // Inpur directory
    #[structopt(short, long, default_value = "UCRArchive_2018/")]
    input_dir: String,

    // Number of repetitions
    #[structopt(short, long, default_value = "1")]
    repetitions: usize,
    // Output file
    #[structopt(short, long, default_value = "test.csv")]
    output_file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    let paths = fs::read_dir(opt.input_dir)?;
    let n_repetitions = opt.repetitions;
    let n_trees = opt.trees;
    let criterion = match opt.criterion.as_str() {
        "gini" => Criterion::Gini,
        "entropy" => Criterion::Entropy,
        "none" => Criterion::None,
        _ => Criterion::Gini,
    };
    let splitter = match opt.splitter.as_str() {
        "best" => Splitter::Best,
        "random" => Splitter::Random,
        _ => Splitter::Best,
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
    let mut predictions: Vec<Vec<f64>> = Vec::new();
    for path in &datasets {
        println!("Processing {}", path.file_name().to_string_lossy());
        let train_path = path
            .path()
            .join(format!("{}_TRAIN.tsv", path.file_name().to_string_lossy()));
        let test_path = path
            .path()
            .join(format!("{}_TEST.tsv", path.file_name().to_string_lossy()));

        let mut mapping = HashMap::new();
        let ds_train = read_csv(train_path, b'\t', &mut mapping)?;
        let ds_test = read_csv(test_path, b'\t', &mut mapping)?;
        let y_true = ds_test.get_targets().clone();

        let n_features = ds_train.get_data()[0].len() as f64;
        for _i in 0..n_repetitions {
            let mut clf: Box<dyn Forest> = match opt.model.as_str() {
                "tsf" => Box::new(TimeSeriesForest::new(
                    n_trees,
                    criterion,
                    splitter,
                    n_features.sqrt() as usize,
                    3,
                    MaxFeatures::Sqrt,
                    None,
                )),
                "rf" => Box::new(RandomForest::new(
                    n_trees,
                    criterion,
                    splitter,
                    MaxFeatures::Sqrt,
                    None,
                )),
                "cif" => Box::new(CanonicalIntervalForest::new(
                    n_trees,
                    criterion,
                    splitter,
                    n_features.sqrt() as usize,
                    3,
                    MaxFeatures::Sqrt,
                    None,
                )),
                _ => Box::new(TimeSeriesForest::new(
                    n_trees,
                    criterion,
                    splitter,
                    n_features.sqrt() as usize,
                    3,
                    MaxFeatures::Sqrt,
                    None,
                )),
            };

            clf.fit(&ds_train.get_data(), &ds_train.get_targets());
            let breiman_distance =
                clf.pairwise_breiman(ds_test.get_data().clone(), ds_train.get_data().clone());
            let prediction_breiman =
                k_nearest_neighbor(1, &ds_train.get_targets(), &breiman_distance);
            let accuracy_breiman = accuracy_score(&prediction_breiman, &y_true);

            let ancestor_distance =
                clf.pairwise_ancestor(ds_test.get_data().clone(), ds_train.get_data().clone());
            let prediction_ancestor =
                k_nearest_neighbor(1, &ds_train.get_targets(), &ancestor_distance);
            let accuracy_ancestor = accuracy_score(&prediction_ancestor, &y_true);

            let zhu_distance =
                clf.pairwise_zhu(ds_test.get_data().clone(), ds_train.get_data().clone());
            let prediction_zhu = k_nearest_neighbor(1, &ds_train.get_targets(), &zhu_distance);
            let accuracy_zhu = accuracy_score(&prediction_zhu, &y_true);
            predictions.push([accuracy_breiman, accuracy_ancestor, accuracy_zhu].to_vec());
            println!(
                "Breiman: {}, Ancestor: {}, Zhu: {}",
                accuracy_breiman, accuracy_ancestor, accuracy_zhu);
        }
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
    write_csv(opt.output_file, predictions, header, index)?;

    Ok(())
}
