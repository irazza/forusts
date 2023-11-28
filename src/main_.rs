use crate::forest::forest::Forest;
use crate::forest::random_forest::RandomForest;
use crate::metrics::classification::accuracy_score;
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::tree::tree::{Criterion, MaxFeatures};
use crate::utils::csv_io::read_csv;
use hashbrown::HashMap;
use std::error::Error;
use std::fs;
use utils::csv_io::write_csv;

mod feature_extraction;
mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;

fn main() -> Result<(), Box<dyn Error>> {
    let paths = fs::read_dir("/media/aazzari/DATA/UCRArchive_2018/")?;
    let n_repetitions = 1;
    let n_trees = 100;
    let criterion = Criterion::Gini;

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
        for _i in 0..n_repetitions {
            // let mut clf = TimeSeriesForest::new(
            //     n_trees,
            //     criterion,
            //     (ds_train.get_data()[0].len() as f64).sqrt() as usize,
            //     3,
            //     MaxFeatures::Sqrt,
            //     None,
            // );
            let mut clf = RandomForest::new(n_trees, criterion, MaxFeatures::Sqrt, None);

            clf.fit(&ds_train.get_data(), &ds_train.get_targets());
            let tsf_accuracy =
                accuracy_score(&clf.predict(&ds_test.get_data()), &ds_test.get_targets());

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
            predictions.push(
                [
                    tsf_accuracy,
                    accuracy_breiman,
                    accuracy_ancestor,
                    accuracy_zhu,
                ]
                .to_vec(),
            );

            println!("\tAverage depth of trees: {}", clf.forest_depth());
        }
    }
    // Create index modifying datasets multiplyng by n_repetitions
    let mut index = Vec::new();
    for i in 0..datasets.len() {
        for _j in 0..n_repetitions {
            index.push(datasets[i].file_name().to_string_lossy().to_string());
        }
    }
    let header = vec!["Dataset", "TSF", "Breiman", "Ancestor", "Zhu"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    write_csv(
        format!(
            "tsf_{}_{}_{}.csv",
            n_trees,
            criterion.to_string(),
            n_repetitions
        ),
        predictions,
        header,
        index,
    )?;

    Ok(())
}
