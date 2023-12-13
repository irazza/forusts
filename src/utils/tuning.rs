use hashbrown::HashMap;

use crate::forest::forest::{ClassificationForest, Forest, OutlierForest, OutlierForestConfig};
use crate::forest::time_series_isolation_forest::{
    TimeSeriesIsolationForest, TimeSeriesIsolationForestConfig,
};
use crate::metrics::classification::roc_auc_score;

use crate::tree::tree::Tree;
use crate::utils::structures::Sample;

pub trait GridSearch {
    type Config: Clone;
    fn generate_grid(&self, cb: impl FnMut(Self::Config));
    fn compute_n_parameters(&self) -> usize;
}

pub trait TuningConfig: GridSearch {
    type Tree: Tree;
    type Forest: Forest<Self::Tree, Config = Self::Config>;
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
    ($type_:ty [$tuning_impl:ident]) => { $tuning_impl };
}

#[macro_export]
macro_rules! compute_n_parameters {
    ($field: expr) => { $field.len() };
    ($field: expr, [$tuning_impl:ident]) => { $field.compute_n_parameters() };
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
        #[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
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
            fn compute_n_parameters(&self) -> usize {
                let mut n_parameters = 1;
                $(
                    n_parameters *= $crate::compute_n_parameters!(self.$field $(, [$tuning_impl])?);
                )+
                n_parameters
            }
        }
    };
}

pub fn grid_search<T: GridSearch + TuningConfig>(
    ds_train: &mut [Sample<'_>],
    ds_test: &[Sample<'_>],
    parameters: T,
    repetition: usize,
    metric: fn(&[<T::Forest as Forest<T::Tree>>::TuningType], &[isize]) -> f64,
) -> Option<(T::Config, f64)> {
    let mut best_score = None;
    let n_parameters = parameters.compute_n_parameters();
    let mut counter = 0;
    parameters.generate_grid(|config| {
        let mut score = 0.0;
        print_loading_bar(counter, n_parameters);
        counter += 1;
        for _i in 0..repetition { 
            let mut forest = T::Forest::new(config.clone());
            forest.fit(ds_train);
            let y_pred = forest.tuning_predict(ds_test);
            score += metric(
                &y_pred,
                &ds_test.iter().map(|s| s.target).collect::<Vec<_>>(),
            );
        }
        score /= repetition as f64;
        if let Some((_, last_best_score)) = &best_score {
            if score > *last_best_score {
                best_score = Some((config, score));
            }
        } else {
            best_score = Some((config, score));
        }
    });
    print_loading_bar(n_parameters, n_parameters);
    best_score
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
    if progress == 100 {
        println!();
    }
}
