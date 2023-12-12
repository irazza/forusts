use hashbrown::HashMap;

use crate::forest::forest::{ClassificationForest, OutlierForest};
use crate::forest::time_series_isolation_forest::TimeSeriesIsolationForest;
use crate::metrics::classification::roc_auc_score;

use crate::utils::structures::Sample;


pub trait GridSearch {
    type Config;
    fn generate_grid(&self, cb: impl FnMut(Self::Config));
}

#[macro_export]
macro_rules! generate_grid_for_loop {
    ($self_:ident, $first_field: ident [$tuning_impl:ident] => $body:tt) => {
        use $crate::utils::tuning::GridSearch;
        $self_.$first_field.generate_grid(|$first_field| {
            $body
        });
    };
    ($self_:ident, $first_field: ident => $body:tt) => {
        for $first_field in $self_.$first_field.iter() {
            $body
        }
    };
    ($self_:ident, $first_field: ident, $($other_fields:ident $([$other_tuning_impl:ident])?),+ => $body:tt) => {
        for $first_field in $self_.$first_field.iter() {
            $crate::generate_grid_for_loop!($self_, $($other_fields $([$other_tuning_impl])?),* => $body);
        }
    };
    ($self_:ident, $first_field: ident [$tuning_impl:ident], $($other_fields:ident$([$other_tuning_impl:ident])?),+ => $body:tt) => {
        use $crate::utils::tuning::GridSearch;
        $self_.$first_field.generate_grid(|$first_field| {
            $crate::generate_grid_for_loop!($self_, $($other_fields $([$other_tuning_impl])?),* => $body);
        });
    };

}

#[macro_export]
macro_rules! generate_tuning_type {
    ($type_:ty) => { Vec<$type_> };
    ($type_:ty [$tuning_impl:ident]) => { $type_ };
}

#[macro_export]
macro_rules! grid_search_tuning {
    (
        pub struct $name:ident[$tuning_name:ident] {
            $(
                pub $field:ident: $type_:ty $([$tuning_impl:ident])?
            ),+ $(,)?
        }

    ) => {
        #[derive(Clone, Copy)]
        pub struct $name {
            $(
               pub $field: $type_
            ),+
        }

        pub struct $tuning_name {
            $(
               pub $field: $crate::generate_tuning_type!($type_$([$tuning_impl])?)
            ),+
        }

        impl crate::utils::tuning::GridSearch for $tuning_name {
            type Config = $name;
            fn generate_grid(&self, mut cb: impl FnMut($name)) {

                $crate::generate_grid_for_loop!(self, $($field$([$tuning_impl])?),+ => {
                    cb($name {
                        $(
                            $field: $field.clone()
                        ),+
                    });
                });
            }
        }
    };
}

pub fn tuning(ds_train: &[Sample<'_>], ds_test: &[Sample<'_>]) -> HashMap<String, f64> {
    // Set parameters
    let n_trees = (1..11).into_iter().map(|x| x * 50).collect::<Vec<usize>>();
    let mut intervals = (1..6).into_iter().map(|x| x * 2).collect::<Vec<i32>>();
    intervals.push(-1);
    intervals.push(-2);
    let mut max_depths = (1..11).into_iter().map(|x| x * 5).collect::<Vec<i32>>();
    max_depths.push(-1);

    let n_features = ds_train[0].data.len() as f64;

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
            clf.fit(&ds_train);
            let y_pred = clf.score_samples(&ds_test);
            scores += roc_auc_score(&y_pred, &ds_test.iter().map(|s| s.target).collect::<Vec<_>>());
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
    println!();
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
