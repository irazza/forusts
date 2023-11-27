use crate::feature_extraction::statistics::unique;
use crate::forest::isolation_forest::IsolationForest;
use crate::metrics::classification::{matthews_corrcoef, roc_auc_score};
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
    let mut predictions = Vec::new();
    let n_repetitions = 1;
    let n_trees = 100;

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

        let mut mapping = HashMap::new();
        let ds_train = read_csv(train_path, b'\t', &mut mapping)?;
        let ds_test = read_csv(test_path, b'\t', &mut mapping)?;
        let y_true = ds_test.get_targets().clone();
        if unique(&y_true[..]).len() == 1 {
            println!("Only one class in y_true: {}", y_true[0]);
            continue;
        }

        for _i in 0..n_repetitions {
            let mut clf = IsolationForest::new(
                n_trees,
                None,
            );
            //let mut clf = IsolationForest::new(n_trees, MaxFeatures::All, None);

            clf.fit(&ds_train.get_data());
            println!("\tAverage depth of trees: {}", clf.forest_depth());
            let y_score = clf.score_samples(&ds_test.get_data());
            let y_pred = clf.predict(&ds_test.get_data());
            let roc_auc = roc_auc_score(&y_score, &y_true);
            let mcc = matthews_corrcoef(&y_pred, &y_true);

            predictions
                .push([roc_auc, mcc].to_vec());
        }
    }
    // Create index modifying datasets multiplying by n_repetitions
    let mut index = Vec::new();
    for i in 0..datasets.len() {
        for _j in 0..n_repetitions {
            index.push(datasets[i].file_name().to_string_lossy().to_string());
        }
    }

    let header = vec!["Dataset", "ROC-AUC", "MCC"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    write_csv(format!("admep_{}.csv", n_trees), predictions, header, index)?;

    Ok(())
}
