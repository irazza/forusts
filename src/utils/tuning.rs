use hashbrown::HashMap;

use crate::forest::forest::{ClassificationForest, OutlierForest};
use crate::forest::time_series_isolation_forest::TimeSeriesIsolationForest;
use crate::metrics::classification::roc_auc_score;
use crate::utils::csv_io::Dataset;

enum Numerical {
    Integer(i64),
    Float(f64),
}

pub fn tuning(ds_train: &Dataset, ds_test: &Dataset) -> HashMap<String, f64> {
    // Set parameters
    let n_trees = (1..11).into_iter().map(|x| x * 50).collect::<Vec<usize>>();
    let mut intervals = (1..6).into_iter().map(|x| x * 2).collect::<Vec<i32>>();
    intervals.push(-1);
    intervals.push(-2);
    let mut max_depths = (1..11).into_iter().map(|x| x * 5).collect::<Vec<i32>>();
    max_depths.push(-1);

    let n_features = ds_train.get_data()[0].len() as f64;

    // Grid search
    let mut best_parameters: HashMap<String, f64> = HashMap::new();
    let mut parameters_permutations = Vec::new();
    for trees in n_trees {
        for depth in &max_depths {
            for intervals in &intervals {
                parameters_permutations.push((trees, *depth, *intervals));
            }
        }
    }

    // Find best parameters
    let n_repeats = 20;
    let mut best_score = 0.0;
    for (i, (tree, depth, interval)) in parameters_permutations.iter().enumerate() {
        // Print loading bar
        print_loading_bar(i, parameters_permutations.len());

        // Store parameters
        let t = *tree;
        let d = if *depth == -1 {
            None
        } else {
            Some(*depth as usize)
        };
        let i = if *interval == -1 {
            n_features.sqrt() as usize
        } else if *interval == -2 {
            n_features.log2() as usize
        } else {
            *interval as usize
        };

        let mut scores = 0.0;
        for _i in 0..n_repeats {
            let mut clf = TimeSeriesIsolationForest::new(t, i, false, d);
            clf.fit(&ds_train.get_data());
            let y_pred = clf.score_samples(&ds_test.get_data());
            scores += roc_auc_score(&y_pred, &ds_test.get_targets());
        }
        if scores > best_score {
            best_score = scores;
            best_parameters.insert("n_trees".to_string(), t as f64);
            best_parameters.insert("max_depth".to_string(), d.unwrap_or(usize::MAX) as f64);
            best_parameters.insert("n_intervals".to_string(), i as f64);
            best_parameters.insert("score".to_string(), scores / n_repeats as f64);
        }
    }
    print_loading_bar(1, 1);
    best_parameters
}

fn print_loading_bar(current: usize, total: usize) {
    let progress = (current as f64 / total as f64 * 100.0) as usize;
    print!("\r[");
    for i in 0..50 {
        if i < progress / 2 {
            print!("=");
        } else {
            print!(" ");
        }
    }
    print!("] {}%", progress);
}
