use crate::forest::extra_canonical_forest::ExtraCanonicalForest;
use crate::forest::forest::{ClassificationForest, Forest};
use crate::metrics::classification::accuracy_score;
use crate::neighbors::nearest_neighbor::k_nearest_neighbor;
use crate::utils::csv_io::read_csv;
use forest::extra_canonical_forest::ExtraCanonicalForestConfig;
use forest::forest::ClassificationForestConfig;
use std::error::Error;
use std::fs::{self};
use tree::tree::Criterion;

mod distance;
mod feature_extraction;
mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;

fn main() -> Result<(), Box<dyn Error>> {
    let mut predictions = Vec::new();
    let mut config = ExtraCanonicalForestConfig {
        n_intervals: 0,
        classification_config: ClassificationForestConfig {
            n_trees: 0,
            max_depth: None,
            min_samples_split: 2,
            max_features: tree::tree::MaxFeatures::Sqrt,
            criterion: Criterion::Random,
        },
    };
    // Settings for the experiments
    let n_repetitions = 1;
    let paths = fs::read_dir("ucr/")?;

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    let mut wtr = csv::Writer::from_path("ercif_heavy.csv")?;
    wtr.write_record(&["Dataset", "Breiman", "Ancestor", "Zhu", "RatioRF"])?;
    wtr.flush()?;
    for i in 0..n_repetitions {
        println!("Repetition {}", i + 1);
        //let mut predictions = Vec::new();
        for path in &datasets {
            println!("\tProcessing {}", path.file_name().to_string_lossy());
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

            config = ExtraCanonicalForestConfig {
                n_intervals: n_features.log10() as usize,
                classification_config: ClassificationForestConfig {
                    n_trees: 100,
                    max_depth: None,
                    min_samples_split: 2,
                    max_features: tree::tree::MaxFeatures::Sqrt,
                    criterion: Criterion::Random,
                },
            };

            let mut model = ExtraCanonicalForest::new(config);
            model.fit(&mut ds_train);

            let breiman_distance = model.pairwise_breiman(&ds_test, &ds_train);
            let prediction_breiman = k_nearest_neighbor(
                1,
                &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(),
                &breiman_distance,
            );
            let accuracy_breiman = accuracy_score(&prediction_breiman, &y_true);

            let ancestor_distance = model.pairwise_ancestor(&ds_test, &ds_train);
            let prediction_ancestor = k_nearest_neighbor(
                1,
                &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(),
                &ancestor_distance,
            );
            let accuracy_ancestor = accuracy_score(&prediction_ancestor, &y_true);

            let zhu_distance = model.pairwise_zhu(&ds_test, &ds_train);
            let prediction_zhu = k_nearest_neighbor(
                1,
                &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(),
                &zhu_distance,
            );
            let accuracy_zhu = accuracy_score(&prediction_zhu, &y_true);

            let ratiorf_distance = model.pairwise_ratiorf(&ds_test, &ds_train);
            let prediction_ratiorf = k_nearest_neighbor(
                1,
                &ds_train.iter().map(|v| v.target).collect::<Vec<_>>(),
                &ratiorf_distance,
            );
            let accuracy_ratiorf = accuracy_score(&prediction_ratiorf, &y_true);

            predictions.push(
                [
                    accuracy_breiman,
                    accuracy_ancestor,
                    accuracy_zhu,
                    accuracy_ratiorf,
                ]
                .to_vec(),
            );
            println!(
                "Breiman: {}, Ancestor: {}, Zhu: {}, RatioRF: {}",
                accuracy_breiman, accuracy_ancestor, accuracy_zhu, accuracy_ratiorf
            );
            wtr.write_record(&[
                path.file_name().to_string_lossy().to_string(),
                accuracy_breiman.to_string(),
                accuracy_ancestor.to_string(),
                accuracy_zhu.to_string(),
                accuracy_ratiorf.to_string(),
            ])?;
            wtr.flush()?;
        }
    }
    Ok(())
}
