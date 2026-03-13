use atomic_float::AtomicF64;
use forust::forest::ci_forest::{CIForest, CIForestConfig};
use forust::forest::forest::{Forest, ForestConfig};
use forust::tree::node::Node;
use forust::tree::tree::{SplitParameters, Tree};
use forust::utils::structures::{IntervalType, MaxFeatures, Sample};
use forust::RandomGenerator;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::cmp::max;
use std::sync::Arc;
use std::time::Instant;

fn make_samples(n_samples: usize, n_features: usize, seed: u64) -> Vec<Sample> {
    let mut rng = RandomGenerator::seed_from_u64(seed);
    (0..n_samples)
        .map(|_| {
            let mut features = Vec::with_capacity(n_features);
            let mut score = 0.0;
            for feature_idx in 0..n_features {
                let value = rng.sample::<f64, _>(StandardNormal) + feature_idx as f64 * 0.03;
                score += value * (feature_idx as f64 + 1.0);
                features.push(value);
            }
            Sample {
                target: if score >= 0.0 { 1 } else { 0 },
                features: Arc::new(features),
            }
        })
        .collect()
}

fn pairwise_zhu_old<F, T>(forest: &F, ds_a: &[Sample], ds_b: Option<&[Sample]>) -> Vec<Vec<f64>>
where
    F: Forest<T>,
    T: Tree,
{
    let ds_b_len = ds_b.map_or(ds_a.len(), |x| x.len());
    let distance_matrix: Vec<Vec<_>> = (0..ds_a.len())
        .map(|_| (0..ds_b_len).map(|_| AtomicF64::new(0.0)).collect())
        .collect();
    let trees = forest.get_trees();
    trees.par_iter().for_each(|tree| {
        let ds_a = tree.transform(ds_a);

        let ds_a_leaves = ds_a
            .iter()
            .map(|x| tree.predict_leaf(x))
            .collect::<Vec<_>>();

        let ds_b_leaves = if let Some(ds_b) = ds_b {
            let ds_b = tree.transform(ds_b);

            ds_b.iter()
                .map(|x| tree.predict_leaf(x))
                .collect::<Vec<_>>()
        } else {
            ds_a_leaves.clone()
        };
        for (i, &ds_a_node) in ds_a_leaves.iter().enumerate() {
            let distances = tree.compute_ancestor(ds_a_node);

            for (j, &ds_b_node) in ds_b_leaves.iter().enumerate() {
                let value = distances[&(ds_b_node as *const Node<_>)].get_depth() as f64
                    / max(ds_a_node.get_depth(), ds_b_node.get_depth()) as f64;
                distance_matrix[i][j].fetch_add(value, std::sync::atomic::Ordering::Relaxed);
            }
        }
    });

    distance_matrix
        .into_iter()
        .map(|d| {
            d.into_iter()
                .map(|d| (1.0 - (d.into_inner() as f64 / trees.len() as f64)).sqrt())
                .collect()
        })
        .collect()
}

fn pairwise_ratiorf_old<F, T>(forest: &F, ds_a: &[Sample], ds_b: Option<&[Sample]>) -> Vec<Vec<f64>>
where
    F: Forest<T>,
    T: Tree,
{
    let ds_b_len = ds_b.map_or(ds_a.len(), |x| x.len());
    let distance_matrix: Vec<Vec<_>> = (0..ds_a.len())
        .map(|_| (0..ds_b_len).map(|_| AtomicF64::new(0.0)).collect())
        .collect();

    let trees = forest.get_trees();
    trees.par_iter().for_each(|tree| {
        let ds_a = tree.transform(ds_a);
        let ds_b = ds_b
            .map(|ds_b| tree.transform(ds_b))
            .unwrap_or_else(|| ds_a.clone());

        let mut union = Vec::new();
        for (i, sample_test) in ds_a.iter().enumerate() {
            let ds_a_splits = tree.get_splits(sample_test);
            for (j, sample_train) in ds_b.iter().enumerate() {
                union.clear();
                union.extend(ds_a_splits.iter().copied());
                union.extend(tree.get_splits(sample_train).into_iter());
                union.sort_unstable();
                union.dedup();
                let agree = union
                    .iter()
                    .filter(|s| s.split(sample_test) == s.split(sample_train))
                    .count() as f64;
                let value = 1.0
                    - if union.is_empty() {
                        1.0
                    } else {
                        agree / union.len() as f64
                    };
                distance_matrix[i][j].fetch_add(value.sqrt(), std::sync::atomic::Ordering::Relaxed);
            }
        }
    });
    distance_matrix
        .into_iter()
        .map(|d| {
            d.into_iter()
                .map(|d| d.into_inner() as f64 / trees.len() as f64)
                .collect()
        })
        .collect()
}

fn checksum(matrix: &[Vec<f64>]) -> f64 {
    matrix.iter().flatten().sum::<f64>()
}

fn main() {
    let mut train = make_samples(512, 48, 17);
    let test = make_samples(48, 48, 29);
    let config = CIForestConfig {
        n_intervals: IntervalType::LOG10,
        n_attributes: 8,
        classification_config: ForestConfig {
            n_trees: 48,
            max_depth: Some(10),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::SQRT,
            criterion: |_a, _b| f64::NAN,
            aggregation: None,
        },
    };

    let mut forest = CIForest::new(&config);
    forest.fit(&mut train, Some(RandomGenerator::seed_from_u64(303)));

    let start = Instant::now();
    let zhu_old = pairwise_zhu_old(&forest, &test, None);
    let zhu_old_sec = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let zhu_new = forest.pairwise_zhu(&test, None);
    let zhu_new_sec = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let ratio_old = pairwise_ratiorf_old(&forest, &test, None);
    let ratio_old_sec = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let ratio_new = forest.pairwise_ratiorf(&test, None);
    let ratio_new_sec = start.elapsed().as_secs_f64();

    println!("zhu_old_sec={zhu_old_sec:.6}");
    println!("zhu_new_sec={zhu_new_sec:.6}");
    println!("zhu_checksum_delta={:.12}", (checksum(&zhu_old) - checksum(&zhu_new)).abs());
    println!("ratio_old_sec={ratio_old_sec:.6}");
    println!("ratio_new_sec={ratio_new_sec:.6}");
    println!(
        "ratio_checksum_delta={:.12}",
        (checksum(&ratio_old) - checksum(&ratio_new)).abs()
    );
}
