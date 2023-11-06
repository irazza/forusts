use crate::forest::random_forest::MaxFeatures;
use crate::forest::time_series_forest;
use crate::metrics::classification::accuracy_score;
use crate::utils::csv_reader::read_csv;
use hashbrown::HashMap;
use std::error::Error;
use std::fs;

mod forest;
mod metrics;
mod nearest_neighbour;
mod tree;
mod utils;

fn main() -> Result<(), Box<dyn Error>> {
    let paths = fs::read_dir("UCRArchive_2018/")?;
    //let paths = fs::read_dir("UCRArchive_2018/")?;
    let n_repetitions = 50;
    let n_trees = 500;

    println!("Number of trees: {}", n_trees);
    
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
        let ds_train = read_csv(train_path, &mut mapping)?;
        let ds_test = read_csv(test_path, &mut mapping)?;

        let n_features = ds_train.get_data()[0].len() as f64;
        for _i in 0..n_repetitions {
            let mut clf = time_series_forest::TimeSeriesForest::new(
                n_trees,
                n_features.sqrt() as usize,
                MaxFeatures::Sqrt,
                None,
            );

            clf.fit(&ds_train.get_data(), &ds_train.get_targets());

            let y_true = ds_test.get_targets().clone();

            let breiman_distance =
                clf.pairwise_breiman(ds_test.get_data().clone(), ds_train.get_data().clone());
            let prediction_breiman = nearest_neighbour::k_nearest_neighbour(
                1,
                &ds_train.get_targets(),
                &breiman_distance,
            );
            let accuracy_breiman = accuracy_score(&prediction_breiman, &y_true);

            let ancestor_distance =
                clf.pairwise_ancestor(ds_test.get_data().clone(), ds_train.get_data().clone());
            let prediction_ancestor = nearest_neighbour::k_nearest_neighbour(
                1,
                &ds_train.get_targets(),
                &ancestor_distance,
            );
            let accuracy_ancestor = accuracy_score(&prediction_ancestor, &y_true);

            let zhu_distance =
                clf.pairwise_zhu(ds_test.get_data().clone(), ds_train.get_data().clone());
            let prediction_zhu =
                nearest_neighbour::k_nearest_neighbour(1, &ds_train.get_targets(), &zhu_distance);
            let accuracy_zhu = accuracy_score(&prediction_zhu, &y_true);
            predictions.push([accuracy_breiman, accuracy_ancestor, accuracy_zhu].to_vec());
        }
    }
    let mut csv_writer = csv::Writer::from_path(format!("UCR_tsf_{}.csv", n_trees))?;
    csv_writer.write_record(&["dataset", "mcc_breiman", "mcc_ancestor", "mcc_zhu"])?;
    for (i, prediction) in predictions.iter().enumerate() {
        csv_writer.write_record(
            [datasets[i / n_repetitions]
                .file_name()
                .to_string_lossy()
                .into_owned()]
            .into_iter()
            .chain(prediction.iter().map(|f| f.to_string())),
        )?;
    }
    csv_writer.flush()?;

    Ok(())
}
