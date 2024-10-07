

#[cfg(test)]
mod tests {

    use std::fs;

    use rand::SeedableRng;

    use crate::{forest::{forest::{Forest, OutlierForest}, isolation_forest::{IsolationForest, IsolationForestConfig}}, metrics::classification::roc_auc_score, utils::csv_io::read_csv};

    #[test]
    fn test_if() {
        let config = IsolationForestConfig {
            n_trees: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_samples: 1.0,
            max_features: |x| x,
            criterion: |_a, _b| 1.0,
        };
        let n_repetitions = 10;
        let paths = fs::read_dir("/media/DATA/albertoazzari/IFDatasets").unwrap();

        let mut datasets = Vec::new();
        for entry in paths {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                datasets.push(entry);
            }
        }
        datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
        let mut predictions = vec![0.0; datasets.len()];
        for i in 0..n_repetitions {
            for (j, path) in datasets.iter().enumerate() {

                let mut ds_train = read_csv(path.path(), b',', false).unwrap();
                let ds_test = ds_train.clone(); // read_csv(path.path(), b',', false).unwrap();
                let y_true = ds_test.iter().map(|s| s.target).collect::<Vec<_>>();

                let mut model = IsolationForest::new(&config);
                model.fit(
                    &mut ds_train,
                    Some(rand_chacha::ChaCha8Rng::seed_from_u64(i as u64)),
                );
                assert_eq!(ds_test, ds_test);
                let prediction = model.score_samples(&ds_test);
                predictions[j] += roc_auc_score(&prediction, &y_true);
            }
        }
        
        for (i, path) in datasets.iter().enumerate() {
            println!("{}: {:.2?}", path.file_name().to_string_lossy(), predictions[i] / n_repetitions as f64);
        }
    }
}
