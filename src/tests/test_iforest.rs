#[cfg(test)]
mod tests {

    use rand::SeedableRng;
    use std::fs;

    use crate::forest::eiso_forest::{EIsoForest, EIsoForestConfig, ExtensionLevel};
    use crate::forest::forest::ForestConfig;
    use crate::utils::csv_io::write_bin;
    use crate::utils::structures::MaxFeatures;
    use crate::{
        forest::{
            forest::{Forest, OutlierForest},
            isolation_forest::{IsolationForest, IsolationForestConfig},
        },
        metrics::classification::roc_auc_score,
        utils::{aggregation::Combiner, csv_io::read_csv},
    };

    #[test]
    fn test_liu() {
        let config = IsolationForestConfig {
            n_trees: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_samples: 1.0,
            max_features: MaxFeatures::ALL,
            criterion: |_a, _b| 1.0,
            aggregation: None,
        };
        let n_repetitions = 10;
        let paths = fs::read_dir("../DATA/IF_BENCHMARK").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
        let mut predictions = vec![0.0; datasets.len()];

        for (i, path) in datasets.iter().enumerate() {
            let mut ds_train = read_csv(path.path(), b',', false).unwrap();
            let ds_test = ds_train.clone(); // read_csv(path.path(), b',', false).unwrap();
            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();
            for j in 0..n_repetitions {
                let mut model = IsolationForest::new(&config);
                model.fit(
                    &mut ds_train,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                        ((i + 2) * (j + 2)) as u64,
                    )),
                );
                let prediction = model.score_samples(&ds_test);
                predictions[i] += roc_auc_score(&prediction, &y_true);
            }
            println!(
                "{}: {:.2?}",
                path.file_name().to_string_lossy(),
                predictions[i] / n_repetitions as f64
            );
        }
    }

    #[test]
    fn test_hariri() {
        let mut config: EIsoForestConfig = EIsoForestConfig {
            extension_level: ExtensionLevel::ExtraFeatures(0),
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
        let paths = fs::read_dir("../../DATA/IF_BENCHMARK").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
        let mut predictions = vec![0.0; datasets.len()];

        for (i, path) in datasets.iter().enumerate() {
            let mut ds_train = read_csv(path.path(), b',', false).unwrap();

            config.extension_level = ExtensionLevel::ExtraFeatures(ds_train[0].features.len() - 1);

            let ds_test = ds_train.clone(); // read_csv(path.path(), b',', false).unwrap();
            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();
            for j in 0..n_repetitions {
                let mut model = EIsoForest::new(&config);
                model.fit(
                    &mut ds_train,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(j as u64)),
                );
                let prediction = model.score_samples(&ds_test);
                predictions[i] += roc_auc_score(&prediction, &y_true);
            }
            println!(
                "{}: {:.2?}",
                path.file_name().to_string_lossy(),
                predictions[i] / n_repetitions as f64
            );
        }
    }

    #[test]
    fn test_stability() {
        let n_repetitions = 201;
        let paths = fs::read_dir("../../DATA/IF_BENCHMARK").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());

        for path in datasets.iter() {
            println!(
                "Dataset: {}",
                path.path().file_stem().unwrap().to_string_lossy()
            );

            let mut ds_train = read_csv(path.path(), b',', false).unwrap();

            let ds_test = ds_train.clone();

            for combiner in [
                Combiner::PROD,
                Combiner::SUM,
                Combiner::TRIMMEDSUM,
                Combiner::MEDIAN,
                Combiner::MIN,
                Combiner::MAX,
            ]
            .iter()
            {
                let config = IsolationForestConfig {
                    n_trees: 100,
                    max_depth: None,
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    max_samples: 1.0,
                    max_features: MaxFeatures::ALL,
                    criterion: |_a, _b| 1.0,
                    aggregation: Some(combiner.clone()),
                };
                let mut scores = vec![vec![0.0; ds_test.len()]; n_repetitions];
                for k in 0..n_repetitions {
                    let mut model = IsolationForest::new(&config);
                    model.fit(
                        &mut ds_train,
                        None,
                        // Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                        //     ((i + 2) * (j + 2) * (k + 2)) as u64,
                        // )),
                    );

                    scores[k] = model.score_samples(&ds_test);
                }
                let _ = write_bin(
                    format!(
                        "/media/albertoazzari/STABILITY/{}/{}.bin",
                        combiner,
                        path.path().file_stem().unwrap().to_string_lossy(),
                    ),
                    scores.clone(),
                );
            }
        }
    }
}
