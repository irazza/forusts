#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand::Rng;
    use rand::SeedableRng;
    use rand_distr::StandardNormal;

    use crate::forest::ci_forest::{CIForest, CIForestConfig};
    use crate::forest::forest::{Forest, OutlierForest};
    use crate::forest::isolation_forest::{IsolationForest, IsolationForestConfig};
    use crate::forest::random_forest::{RandomForest, RandomForestConfig};
    use crate::utils::split::train_test_split;
    use crate::utils::structures::{IntervalType, MaxFeatures, Sample};
    use crate::RandomGenerator;

    fn make_samples(n_samples: usize, n_features: usize, seed: u64) -> Vec<Sample> {
        let mut rng = RandomGenerator::seed_from_u64(seed);
        (0..n_samples)
            .map(|row| {
                let mut features = Vec::with_capacity(n_features);
                let mut score = 0.0;
                for feature_idx in 0..n_features {
                    let value = rng.sample::<f64, _>(StandardNormal) + feature_idx as f64 * 0.05;
                    score += value * (feature_idx as f64 + 1.0);
                    features.push(value);
                }
                Sample {
                    target: if score + row as f64 * 0.001 >= 0.0 { 1 } else { 0 },
                    features: Arc::new(features),
                }
            })
            .collect()
    }

    #[test]
    fn random_forest_same_seed_same_predictions() {
        let config = RandomForestConfig {
            n_trees: 32,
            max_depth: Some(8),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::SQRT,
            criterion: |_a, _b| f64::NAN,
            aggregation: None,
        };
        let mut train_a = make_samples(256, 16, 7);
        let train_b = train_a.clone();
        let test = make_samples(96, 16, 11);

        let mut model_a = RandomForest::new(&config);
        model_a.fit(&mut train_a, Some(RandomGenerator::seed_from_u64(42)));
        let prediction_a = model_a.predict(&test);

        let mut model_b = RandomForest::new(&config);
        model_b.fit(&mut train_b.clone(), Some(RandomGenerator::seed_from_u64(42)));
        let prediction_b = model_b.predict(&test);

        assert_eq!(prediction_a, prediction_b);
    }

    #[test]
    fn isolation_forest_default_rng_is_reproducible() {
        let config = IsolationForestConfig {
            n_trees: 32,
            max_depth: Some(8),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::ALL,
            criterion: |_a, _b| 1.0,
            aggregation: None,
        };
        let mut train_a = make_samples(256, 16, 19);
        let mut train_b = train_a.clone();
        let test = make_samples(96, 16, 23);

        let mut model_a = IsolationForest::new(&config);
        model_a.fit(&mut train_a, None);
        let scores_a = model_a.score_samples(&test);

        let mut model_b = IsolationForest::new(&config);
        model_b.fit(&mut train_b, None);
        let scores_b = model_b.score_samples(&test);

        assert_eq!(scores_a, scores_b);
    }

    #[test]
    fn train_test_split_default_rng_is_reproducible() {
        let data = make_samples(64, 8, 29);
        let split_a = train_test_split(&data, 0.25, true, None);
        let split_b = train_test_split(&data, 0.25, true, None);

        assert_eq!(split_a.0, split_b.0);
        assert_eq!(split_a.1, split_b.1);
    }

    #[test]
    fn pairwise_ratiorf_handles_different_input_lengths() {
        let config = CIForestConfig {
            n_intervals: IntervalType::LOG2,
            n_attributes: 4,
            classification_config: crate::forest::forest::ForestConfig {
                n_trees: 16,
                max_depth: Some(6),
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_features: MaxFeatures::SQRT,
                criterion: |_a, _b| 1.0,
                aggregation: None,
            },
        };

        let mut train = make_samples(96, 12, 41);
        let probe = make_samples(37, 12, 43);

        let mut model = CIForest::new(&config);
        model.fit(&mut train, Some(RandomGenerator::seed_from_u64(47)));

        let distances = model.pairwise_ratiorf(&probe, Some(&train));
        assert_eq!(distances.len(), probe.len());
        assert!(distances.iter().all(|row| row.len() == train.len()));
    }
}
