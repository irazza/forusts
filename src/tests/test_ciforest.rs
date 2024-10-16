#[cfg(test)]
mod tests {

    use rand::SeedableRng;
    use std::fs;

    use crate::{
        cluster::agglomerative::agglomerative_clustering,
        forest::{
            ci_forest::{CIForest, CIForestConfig},
            erci_forest::ERCIForest,
            forest::{Forest, OutlierForest, OutlierForestConfig},
            isolation_forest::{IsolationForest, IsolationForestConfig},
        },
        metrics::{
            classification::{accuracy_score, roc_auc_score},
            clustering::adjusted_rand_score,
        },
        neighbors::nearest_neighbor::k_nearest_neighbor,
        utils::{csv_io::read_csv, structures::IntervalType},
    };

    #[test]
    fn test_cif() {
        // Settings for the experiments
        let config = CIForestConfig {
            n_intervals: IntervalType::LOG10,
            n_attributes: 4,
            outlier_config: OutlierForestConfig {
                n_trees: 100,
                max_depth: None,
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_samples: 1.0,
                max_features: |x| x,
                criterion: |_a, _b| 1.0,
                aggregation: None,
            },
        };
        let n_repetitions = 1;
        let paths = fs::read_dir("/media/DATA/admep/").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
        let mut predictions = vec![0.0; datasets.len()];
        for i in 0..n_repetitions {
            for (j, path) in datasets.iter().enumerate() {
                let mut ds_train = read_csv(
                    path.path()
                        .join(format!("{}_TRAIN.tsv", path.file_name().to_string_lossy())),
                    b'\t',
                    false,
                )
                .unwrap();
                let ds_test = read_csv(
                    path.path()
                        .join(format!("{}_TEST.tsv", path.file_name().to_string_lossy())),
                    b'\t',
                    false,
                )
                .unwrap();
                let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();

                let mut model = CIForest::new(&config);
                model.fit(
                    &mut ds_train,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64((i * j) as u64)),
                );
                let prediction = model.score_samples(&ds_test);
                predictions[j] += roc_auc_score(&prediction, &y_true);
            }
        }
        let predictions = predictions
            .iter()
            .map(|x| x / n_repetitions as f64)
            .collect::<Vec<_>>();
        for (i, path) in datasets.iter().enumerate() {
            println!(
                "{}: {:.2}",
                path.file_name().to_string_lossy(),
                predictions[i]
            );
        }
        println!(
            "Mean ROC-AUC: {:.2}",
            predictions.iter().sum::<f64>() / predictions.len() as f64
        );
    }

    #[test]
    fn test_dmkd() {
        // Settings for the experiments
        let config = CIForestConfig {
            n_intervals: IntervalType::LOG2,
            n_attributes: 8,
            outlier_config: OutlierForestConfig {
                n_trees: 200,
                max_depth: Some(usize::MAX),
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_samples: 1.0,
                max_features: |x| x,
                criterion: |_a, _b| 1.0,
                aggregation: None,
            },
        };
        let n_repetitions = 10;
        let paths = fs::read_dir("/media/DATA/UCRArchive_2018/").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
        let mut predictions = vec![0.0; datasets.len()];
        for i in 0..n_repetitions {
            for (j, path) in datasets.iter().enumerate() {
                println!("{:?} ", path.file_name().to_string_lossy());
                let ds_train = read_csv(
                    path.path()
                        .join(format!("{}_TRAIN.tsv", path.file_name().to_string_lossy())),
                    b'\t',
                    false,
                )
                .unwrap();
                let ds_test = read_csv(
                    path.path()
                        .join(format!("{}_TEST.tsv", path.file_name().to_string_lossy())),
                    b'\t',
                    false,
                )
                .unwrap();

                let mut ds = ds_train.clone();
                ds.extend(ds_test.clone());
                let y_true = ds.iter().map(|s| s.target).collect::<Vec<_>>();

                let mut classes = y_true.clone();
                classes.sort();
                classes.dedup();

                let mut model = ERCIForest::new(&config);
                model.fit(
                    &mut ds,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                        ((i + 2) * (j + 2)) as u64,
                    )),
                );
                let distance_matrix = model.pairwise_breiman(&ds, &ds);

                let prediction = agglomerative_clustering(
                    classes.len(),
                    kodama::Method::Average,
                    distance_matrix,
                );

                predictions[j] += adjusted_rand_score(&prediction, &y_true);
            }
        }
    }
}
