use crate::feature_extraction::statistics::unique;
use crate::forest::forest::OutlierForest;
use crate::forest::isolation_forest::IsolationForest;
use crate::forest::time_series_isolation_forest::TimeSeriesIsolationForest;
use crate::metrics::classification::roc_auc_score;
use crate::utils::csv_io::read_csv;
use crate::utils::tuning::tuning;
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
    let paths = fs::read_dir("/media/aazzari/DATA/admep/")?;
    //let mut predictions = Vec::new();
    let n_repetitions = 10;
    let n_trees = 50;

    let mut datasets: Vec<_> = Vec::new();
    let mut hyperparameters = Vec::new();
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

        let mut mapping = HashMap::new();
        // mapping {0: 0, 1: 1}
        mapping.insert(0, 0);
        mapping.insert(1, 1);
        let ds_train = read_csv(train_path, b'\t', &mut mapping)?;
        let ds_test = read_csv(test_path, b'\t', &mut mapping)?;
        hyperparameters.push(tuning(&ds_train, &ds_test));
    }
    println!("{:?}", hyperparameters);
    serde_json::to_writer(fs::File::create("hyperparameters.json")?, &hyperparameters)?;

    //     let n_features = ds_train.get_data()[0].len() as f64;
    //     for _i in 0..n_repetitions {
    //         let mut clf = TimeSeriesIsolationForest::new(
    //             50, //n_trees,
    //             2, //n_features.sqrt() as usize,
    //             Some(false),
    //             Some(5),
    //         );
    //         clf.fit(&ds_train.get_data());
    //         let y_score = clf.score_samples(&ds_test.get_data());
    //         let roc_auc = roc_auc_score(&y_score, &y_true);
    //         predictions.push([roc_auc].to_vec());
    //     }
    // }
    // // Create index modifying datasets multiplying by n_repetitions
    // let mut index = Vec::new();
    // for i in 0..datasets.len() {
    //     for _j in 0..n_repetitions {
    //         index.push(datasets[i].file_name().to_string_lossy().to_string());
    //     }
    // }

    // let header = vec!["Dataset", "ROC-AUC"]
    //     .iter()
    //     .map(|s| s.to_string())
    //     .collect();
    // write_csv(
    //     format!("admepTSIF_T{}_R{}.csv", n_trees, n_repetitions),
    //     predictions,
    //     header,
    //     index,
    // )?;

    Ok(())
}
