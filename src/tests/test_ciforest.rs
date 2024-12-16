#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use std::fs;

    use crate::forest::ceiso_forest::{CEIsoForest, CEIsoForestConfig};
    use crate::metrics::classification::{accuracy_score, precision_at_k, roc_auc_score};
    use crate::tree::transform::CACHE;
    use crate::utils::csv_io::{write_bin, write_csv};
    use crate::utils::split::binarize;
    use crate::utils::structures::MaxFeatures;
    use crate::{
        forest::{
            ci_forest::{CIForest, CIForestConfig},
            ciso_forest::{CIsoForest, CIsoForestConfig},
            erci_forest::ERCIForest,
            forest::{Forest, ForestConfig, OutlierForest},
        },
        utils::{csv_io::read_csv, structures::IntervalType},
    };

    #[test]
    fn test_admep() {
        // Settings for the experiments
        let config = CEIsoForestConfig {
            n_intervals: IntervalType::LOG2,
            n_attributes: 8,
            extension_level: 0.1,
            outlier_config: ForestConfig {
                n_trees: 200,
                max_depth: None,
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_samples: 1.0,
                max_features: MaxFeatures::ALL,
                criterion: |_a, _b| 1.0,
                aggregation: None,
            },
        };
        let n_repetitions = 10;
        let paths = fs::read_dir("../admep/ADMEP/").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
        let mut predictions = vec![vec![0.0; n_repetitions]; datasets.len()];

        for (i, path) in datasets.iter().enumerate() {
            let mut ds_train = read_csv(
                path.path()
                    .join(format!("{}_TRAIN.csv", path.file_name().to_string_lossy())),
                b',',
                false,
            )
            .unwrap();
            let ds_test = read_csv(
                path.path()
                    .join(format!("{}_TEST.csv", path.file_name().to_string_lossy())),
                b',',
                false,
            )
            .unwrap();
            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();
            let n_anomalies = y_true.iter().sum::<isize>() as usize;
            for j in 0..n_repetitions {
                let mut model = CEIsoForest::new(&config);
                model.fit(
                    &mut ds_train,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                        j as u64,
                    )),
                );
                let prediction = model.score_samples(&ds_test);
                predictions[i][j] = roc_auc_score(&prediction, &y_true);
            }
            println!(
                "{}: {:.2}",
                path.file_name().to_string_lossy(),
                predictions[i].iter().sum::<f64>() / n_repetitions as f64
            );
        }
        write_csv("admep_L.csv", predictions, None);
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
        let paths = fs::read_dir("../../DATA/ucr").unwrap();

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
            ds.extend(ds_test);

            for j in 0..n_repetitions {
                let mut model = ERCIForest::new(&config);
                let start_time = std::time::Instant::now();
                model.fit(
                    &mut ds,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                        ((i + 2) * (j + 2)) as u64,
                    )),
                );
                times[i][0] += start_time.elapsed().as_secs_f64();

                // breiman
                let start_time = std::time::Instant::now();
                let distance_matrix = model.pairwise_breiman(&ds, None);
                times[i][1] += start_time.elapsed().as_secs_f64();
                let breiman_path = format!(
                    "/media/DATA/albertoazzari/tsrf/LIGHT/breiman/{}_{}.csv",
                    path.file_name().to_string_lossy(),
                    j
                );
                write_csv(breiman_path, distance_matrix, None);

                // zhu
                let start_time = std::time::Instant::now();
                let distance_matrix = model.pairwise_zhu(&ds, None);
                times[i][2] += start_time.elapsed().as_secs_f64();
                let zhu_path = format!(
                    "/media/DATA/albertoazzari/tsrf/LIGHT/zhu/{}_{}.csv",
                    path.file_name().to_string_lossy(),
                    j
                );
                write_csv(zhu_path, distance_matrix, None);

                // ratiorf
                let start_time = std::time::Instant::now();
                let distance_matrix = model.pairwise_ratiorf(&ds, None);
                times[i][3] += start_time.elapsed().as_secs_f64();
                let ratiorf_path = format!(
                    "/media/DATA/albertoazzari/tsrf/LIGHT/ratiorf/{}_{}.csv",
                    path.file_name().to_string_lossy(),
                    j
                );
                write_csv(ratiorf_path, distance_matrix, None);
            }
            CACHE.clear();
            times[i][0] /= n_repetitions as f64;
            times[i][1] /= n_repetitions as f64;
            times[i][2] /= n_repetitions as f64;
            times[i][3] /= n_repetitions as f64;
            println!(
                "{}: Fit in {:.2}s, breiman in {:.2}s, zhu in {:.2}s, ratiorf in {:.2}s",
                path.file_name().to_string_lossy(),
                times[i][0],
                times[i][1],
                times[i][2],
                times[i][3],
            );
        }
        write_csv(
            "/media/DATA/albertoazzari/tsrf/LIGHT/times.csv",
            times,
            None,
        );
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
        let n_repetitions = 1;
        let paths = fs::read_dir("../../DATA/ucr/").unwrap();

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

    #[test]
    fn test_cisof() {
        // Settings for the experiments
        let datasets_name = [
            "Adiac",
            "ArrowHead",
            "Beef",
            "BeetleFly",
            "BirdChicken",
            "CBF",
            "ChlorineConcentration",
            "Coffee",
            "ECG200",
            "ECGFiveDays",
            "FaceFour",
            "GunPoint",
            "Ham",
            "Herring",
            "Lightning2",
            "Lightning7",
            "Meat",
            "MedicalImages",
            "MoteStrain",
            "Plane",
            "Strawberry",
            "Symbols",
            "ToeSegmentation1",
            "ToeSegmentation2",
            "Trace",
            "TwoLeadECG",
            "Wafer",
            "Wine",
        ];
        // let normal_classes = [9, 2, 1, 1, 1, 2, 1, 0, 1, 1, 3, 1, 1, 1, 1, 3, 2, 5, 1, 5, 1, 6, 0, 0, 1, 1, 1, 1];
        let config = CIsoForestConfig {
            n_intervals: IntervalType::LOG10,
            n_attributes: 8,
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

        let n_repetitions = 10;
        let paths = fs::read_dir("../../DATA/ucr_AD/").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir()
                && datasets_name.contains(&entry.file_name().to_string_lossy().to_string().as_str())
            {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
        let mut aucs = vec![0.0; datasets.len()];
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
            let y_test = binarize(&ds_test.iter().map(|s| s.target).collect::<Vec<isize>>());

            let start_time = std::time::Instant::now();
            for j in 0..n_repetitions {
                let mut model = CIsoForest::new(&config);
                model.fit(
                    &mut ds_train,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(j as u64)),
                );
                let prediction = model.score_samples(&ds_test);
                aucs[i] += roc_auc_score(&prediction, &y_test);
                write_bin(
                    format!(
                        "/media/albertoazzari/CISOF/{}/{}_{}.bin",
                        path.file_name().to_string_lossy(),
                        path.file_name().to_string_lossy(),
                        j
                    ),
                    prediction,
                )
                .unwrap();
            }
            println!(
                "{}: {:.2} in {:.2} seconds",
                path.file_name().to_string_lossy(),
                aucs[i] / n_repetitions as f64,
                start_time.elapsed().as_secs_f64(),
            );
        }
    }
}
