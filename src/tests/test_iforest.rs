#[cfg(test)]
mod tests {

    use rand::SeedableRng;
    use std::fs;

    use crate::{
        forest::{
            forest::{Forest, OutlierForest},
            isolation_forest::{IsolationForest, IsolationForestConfig},
        },
        metrics::classification::{precision_at_k, roc_auc_score},
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
            max_features: |x| x,
            criterion: |_a, _b| 1.0,
            aggregation: None,
        };
        let n_repetitions = 10;
        let paths = fs::read_dir("/media/DATA/IFDatasets").unwrap();

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
    fn test_stability() {
        let n_repetitions = 200;
        let n_combiners = 6;
        let paths = fs::read_dir("/media/DATA/IFDatasets").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());

        for (i, path) in datasets.iter().enumerate() {
            println!(
                "Dataset: {}",
                path.path().file_stem().unwrap().to_string_lossy()
            );
            let mut anomaly_scores = Vec::with_capacity(n_combiners * n_repetitions);
            let mut auc_scores = Vec::with_capacity(n_combiners * n_repetitions);
            let mut precision_at_10 = Vec::with_capacity(n_combiners * n_repetitions);
            let mut ds_train = read_csv(path.path(), b',', false).unwrap();
            let ds_test = ds_train.clone();
            let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();
            let n_anomalies = y_true.iter().filter(|&&x| x == 1).count();

            for (j, combiner) in [
                Combiner::PROD,
                Combiner::SUM,
                Combiner::TRIMMEDSUM,
                Combiner::MEDIAN,
                Combiner::MIN,
                Combiner::MAX,
            ]
            .iter()
            .enumerate()
            {
                let config = IsolationForestConfig {
                    n_trees: 100,
                    max_depth: None,
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    max_samples: 1.0,
                    max_features: |x| x,
                    criterion: |_a, _b| 1.0,
                    aggregation: Some(combiner.clone()),
                };
                for k in 0..n_repetitions {
                    let mut model = IsolationForest::new(&config);
                    model.fit(
                        &mut ds_train,
                        Some(rand_chacha::ChaCha8Rng::seed_from_u64(
                            ((i + 2) * (j + 2) * (k + 2)) as u64,
                        )),
                    );
                    let prediction = model.score_samples(&ds_test);
                    anomaly_scores.push(prediction.clone());
                    precision_at_10.push(precision_at_k(
                        &prediction,
                        &y_true,
                        (n_anomalies as f64 * 0.1).round() as usize,
                    ));
                    auc_scores.push(roc_auc_score(&prediction, &y_true));
                }
            }
            let folder = format!(
                "results/{}",
                path.path().file_stem().unwrap().to_string_lossy()
            );
            if !std::path::Path::new(&folder).exists() {
                fs::create_dir_all(&folder).unwrap();
            }
            let mut wtr = csv::Writer::from_path(format!(
                "results/{}/anomaly_scores.csv",
                path.path().file_stem().unwrap().to_string_lossy()
            ))
            .unwrap();
            for scores in anomaly_scores {
                wtr.write_record(scores.iter().map(|x| x.to_string()))
                    .unwrap();
            }
            wtr.flush().unwrap();
            let mut wtr = csv::Writer::from_path(format!(
                "results/{}/auc_scores.csv",
                path.path().file_stem().unwrap().to_string_lossy()
            ))
            .unwrap();
            wtr.write_record(auc_scores.iter().map(|x| x.to_string()))
                .unwrap();
            wtr.flush().unwrap();
            let mut wtr = csv::Writer::from_path(format!(
                "results/{}/precision_at_10.csv",
                path.path().file_stem().unwrap().to_string_lossy()
            ))
            .unwrap();
            wtr.write_record(precision_at_10.iter().map(|x| x.to_string()))
                .unwrap();
            wtr.flush().unwrap();
        }
    }
}
