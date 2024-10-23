#[cfg(test)]
mod tests {

    use rand::SeedableRng;
    use std::fs;

    use crate::forest::forest::CACHE;
    use crate::metrics::classification::accuracy_score;
    use crate::utils::csv_io::write_csv;
    use crate::utils::structures::MaxFeatures;
    use crate::{
        forest::{
            ci_forest::{CIForest, CIForestConfig},
            ciso_forest::{CIsoForest, CIsoForestConfig},
            erci_forest::ERCIForest,
            forest::{Forest, ForestConfig, OutlierForest},
        },
        metrics::classification::roc_auc_score,
        utils::{csv_io::read_csv, structures::IntervalType},
    };

    #[test]
    fn test_cif() {
        // Settings for the experiments
        let config = CIsoForestConfig {
            n_intervals: IntervalType::LOG10,
            n_attributes: 4,
            outlier_config: ForestConfig {
                n_trees: 100,
                max_depth: None,
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_samples: 1.0,
                max_features: MaxFeatures::ALL,
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

        for (i, path) in datasets.iter().enumerate() {
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
            for j in 0..n_repetitions {
                let mut model = CIsoForest::new(&config);
                model.fit(
                    &mut ds_train,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64((i * j) as u64)),
                );
                let prediction = model.score_samples(&ds_test);
                predictions[i] += roc_auc_score(&prediction, &y_true);
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
        let config = CIsoForestConfig {
            n_intervals: IntervalType::LOG2,
            n_attributes: 8,
            outlier_config: ForestConfig {
                n_trees: 200,
                max_depth: Some(usize::MAX),
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_samples: 1.0,
                max_features: MaxFeatures::ALL,
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
        let mut times = vec![vec![0.0; 4]; datasets.len()];

        for (i, path) in datasets.iter().enumerate() {

            println!("Dataset: {}", path.file_name().to_string_lossy());

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

            for j in 0..n_repetitions {
                CACHE.clear();
                let mut model = ERCIForest::new(&config);
                let start_time = std::time::Instant::now();
                model.fit(
                    &mut ds_train,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                        ((i + 2) * (j + 2)) as u64,
                    )),
                );
                times[i][0] += start_time.elapsed().as_secs_f64();
                // breiman
                let distance_matrix = model.pairwise_breiman(&ds_test, &ds_train);
                times[i][1] += start_time.elapsed().as_secs_f64();
                let breiman_path = format!(
                    "/media/DATA/TSRFDist/LIGHT/breiman/{}_{}.csv",
                    path.file_name().to_string_lossy(),
                    j
                );
                write_csv(breiman_path, distance_matrix);

                // zhu
                let distance_matrix = model.pairwise_zhu(&ds_test, &ds_train);
                times[i][2] += start_time.elapsed().as_secs_f64();
                let zhu_path = format!(
                    "/media/DATA/TSRFDist/LIGHT/zhu/{}_{}.csv",
                    path.file_name().to_string_lossy(),
                    j
                );
                write_csv(zhu_path, distance_matrix);

                // ratiorf
                let distance_matrix = model.pairwise_ratiorf(&ds_test, &ds_train);
                times[i][3] += start_time.elapsed().as_secs_f64();
                let ratiorf_path = format!(
                    "/media/DATA/TSRFDist/LIGHT/ratiorf/{}_{}.csv",
                    path.file_name().to_string_lossy(),
                    j
                );
                write_csv(ratiorf_path, distance_matrix);
            }
            times[i] = times[i].iter().map(|x| x / n_repetitions as f64).collect::<Vec<_>>();
        }
        let times_path = "/media/DATA/TSRFDist/LIGHT/times.csv";
        write_csv(times_path, times);

    }

    #[test]
    fn test_cit() {
        // Settings for the experiments
        let config = CIForestConfig {
            n_intervals: IntervalType::LOG2,
            n_attributes: 8,
            classification_config: ForestConfig {
                n_trees: 200,
                max_depth: Some(usize::MAX),
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_samples: 1.0,
                max_features: MaxFeatures::ALL,
                criterion: |_a, _b| 1.0,
                aggregation: None,
            },
        };
        let n_repetitions = 10;
        let paths = fs::read_dir("../UCRArchive_2018/").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
        let mut predictions = vec![0.0; datasets.len()];
        for (i, path) in datasets[..5].iter().enumerate() {
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
            let start_time = std::time::Instant::now();
            for j in 0..n_repetitions {
                let mut model = CIForest::new(&config);
                model.fit(
                    &mut ds_train,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                        ((i + 2) * (j + 2)) as u64,
                    )),
                );
                let prediction = model.predict(&ds_test);
                predictions[i] += accuracy_score(
                    &prediction,
                    &ds_test.iter().map(|s| s.target).collect::<Vec<_>>(),
                );
            }
            println!(
                "{}: {:.2} in {:.2} seconds",
                path.file_name().to_string_lossy(),
                predictions[i] / n_repetitions as f64,
                start_time.elapsed().as_secs_f64(),
            );
        }
    }
}
