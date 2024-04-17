use crate::feature_extraction::statistics::EULER_MASCHERONI;
use crate::forest::forest::{Forest, OutlierForest};
use crate::forest::isolation_forest::{IsolationForest, IsolationForestConfig};
use crate::metrics::classification::{roc_auc_score, roc_auc_score_c};
use crate::utils::csv_io::read_csv;
use csv::Writer;
use feature_extraction::statistics::transpose;
use std::error::Error;
use std::fs::{self};

mod distance;
mod feature_extraction;
mod forest;
mod metrics;
mod neighbors;
mod tree;
mod utils;

fn main() -> Result<(), Box<dyn Error>> {
    // Settings for the experiments
    let paths = fs::read_dir("../DATA/ADDatasets/")?;

    let mut datasets = Vec::new();
    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        datasets.push(entry.path());
    }
    datasets.sort_by_key(|dir| dir.file_name().unwrap().to_string_lossy().to_string());
    for path in &datasets {
        //let mut roc_aucs = Vec::new();
        println!(
            "Processing {}",
            path.file_name().unwrap().to_string_lossy()
        );
        let mut ds = read_csv(path, b',', false)?;
        // ds.sort_unstable_by(|a, b| a.partial_cmp(&b).unwrap());
        // ds.dedup();
        // println!("\t{} samples with {} features", ds.len(), ds[0].data.len());
        // let class_0 = ds.iter().filter(|s| s.target == 0).count() as f64;
        // let class_1 = ds.iter().filter(|s| s.target == 1).count() as f64;
        // println!("\tClass 0: {}, Class 1: {}", class_0/ds.len() as f64, class_1/ds.len() as f64);
        let y_true = ds.iter().map(|s| s.target).collect::<Vec<_>>();

        let config = IsolationForestConfig {
            n_trees: 1000,
            enhanced_anomaly_score: false,
            max_depth: None,
            
        };
        let mut model = IsolationForest::new(config);
        model.fit(&mut ds);
        let mut anomaly_scores = model.compute_as_per_tree(&ds);
        let roc_auc = roc_auc_score( &model.score_samples(&ds), &y_true);
        println!("\tROC AUC: {}", roc_auc);
        anomaly_scores.push(y_true.clone().iter().map(|s| *s as f64).collect::<Vec<_>>());
        // Transpose the anomaly scores
        let transposed = transpose(anomaly_scores.clone());
        // Get the name of the dataset
        let out_path = path.file_stem().unwrap().to_string_lossy();
        fs::create_dir_all(format!("s+sspr24/{}", out_path))?;
        let mut scores_wrt = Writer::from_path(format!("s+sspr24/{}/{}_scores.csv", out_path, out_path))?;
        for i in 0..transposed.len() {
            scores_wrt.write_record(transposed[i].iter().map(|s| s.to_string()).collect::<Vec<_>>())?;
        }
        scores_wrt.flush()?;
    }
    Ok(())
}

// let denominator = (2.0 * (f64::ln(model.get_max_samples() as f64 - 1.0) + EULER_MASCHERONI))
        //     - 2.0 * ((model.get_max_samples() as f64 - 1.0) / model.get_max_samples() as f64);
        // let mut ans = Vec::new();
        // for i in 0..anomaly_scores[0].len() {
        //     let mut ascore = 0.0;        //     for t in anomaly_scores.iter() {
        //         ascore += t[i];
        //     }
        //     ascore /= denominator;
        //     ans.push(2.0f64.powf(-ascore/model.get_trees().len() as f64));
        // }

        // let scores = model.score_samples(&ds);

        // assert_eq!(ans.len(), y_true.len());