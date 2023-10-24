use csv::ReaderBuilder;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::random_forest::MaxFeatures;
mod decision_tree;
mod nearest_neighbour;
mod random_forest;
mod time_series_forest;

struct Dataset {
    targets: Vec<usize>,
    data: Vec<Vec<f64>>,
}

fn read_csv(path: impl AsRef<Path>) -> Result<Dataset, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b'\t')
        .from_reader(reader);

    let mut targets = Vec::new();
    let mut data = Vec::new();

    for result in reader.records() {
        let record = result?;
        let mut row = Vec::new();

        // Assuming the first column is the target and the rest are data
        if let Some(target) = record.get(0) {
            let class = target.parse::<isize>()?;
            let class = if class < 0 { -class * 2 } else { class * 2 + 1 };
            targets.push(class as usize);
        }

        for value in record.iter().skip(1) {
            row.push(value.parse()?);
        }

        data.push(row);
    }

    Ok(Dataset { targets, data })
}

fn main() -> Result<(), Box<dyn Error>> {
    let paths = fs::read_dir("UCRArchive_2018/")?;
    let mut datasets: Vec<_> = Vec::new();

    for entry in paths {
        // Unwrap the entry or handle the error, if any.
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            datasets.push(entry);
        }
    }
    datasets.sort_by_key(|dir| dir.file_name().to_string_lossy().to_string());
    let mut predictions: Vec<Vec<f64>> = Vec::new();
    let n_repetitions = 50;
    let n_trees = 200;
    for path in &datasets {
        println!("Processing {}", path.file_name().to_string_lossy());

        let train_path = path
            .path()
            .join(format!("{}_TRAIN.tsv", path.file_name().to_string_lossy()));
        let test_path = path
            .path()
            .join(format!("{}_TEST.tsv", path.file_name().to_string_lossy()));

        let ds_train = &read_csv(train_path)?;
        let ds_test = &read_csv(test_path)?;

        if [&ds_train, &ds_test]
            .into_iter()
            .any(|d| d.data.iter().any(|row| row.iter().any(|v| !v.is_finite())))
        {
            println!("NaN in training data");
            continue;
        }

        // Train the model
        for _i in 0..n_repetitions {
            let mut clf = time_series_forest::TimeSeriesForest::new(
                n_trees,
                (ds_train.data[0].len() as f64).sqrt() as usize,
                MaxFeatures::Sqrt,
                Some(1000),
            );
            // check fitting time
            clf.fit(&ds_train.data, &ds_train.targets);
            let y_pred = clf.predict(&ds_test.data);
            let accuracy_rf = (y_pred
                .iter()
                .zip(ds_test.targets.iter())
                .filter(|&(a, b)| a == b)
                .count() as f64)
                / (ds_test.targets.len() as f64);
            let breiman_distance =
                clf.pairwise_breiman(ds_test.data.clone(), ds_train.data.clone());
            let prediction_breiman =
                nearest_neighbour::k_nearest_neighbour(1, &ds_train.targets, &breiman_distance);
            let accuracy_breiman = (prediction_breiman
                .iter()
                .zip(ds_test.targets.iter())
                .filter(|&(a, b)| a == b)
                .count() as f64)
                / (ds_test.targets.len() as f64);
            let ancestor_distance =
                clf.pairwise_ancestor(ds_test.data.clone(), ds_train.data.clone());
            let prediction_ancestor =
                nearest_neighbour::k_nearest_neighbour(1, &ds_train.targets, &ancestor_distance);
            let accuracy_ancestor = (prediction_ancestor
                .iter()
                .zip(ds_test.targets.iter())
                .filter(|&(a, b)| a == b)
                .count() as f64)
                / (ds_test.targets.len() as f64);
            let zhu_distance = clf.pairwise_zhu(ds_test.data.clone(), ds_train.data.clone());
            let prediction_zhu =
                nearest_neighbour::k_nearest_neighbour(1, &ds_train.targets, &zhu_distance);
            let accuracy_zhu = (prediction_zhu
                .iter()
                .zip(ds_test.targets.iter())
                .filter(|&(a, b)| a == b)
                .count() as f64)
                / (ds_test.targets.len() as f64);

            predictions.push([accuracy_rf, accuracy_breiman, accuracy_ancestor, accuracy_zhu].to_vec());
        }
    }

    let mut csv_writer = csv::Writer::from_path(format!("results_{}.csv", n_trees))?;
    csv_writer.write_record(&[
        "dataset",
        "accuracy_rf",
        "accuracy_breiman",
        "accuracy_ancestor",
        "accuracy_zhu",
    ])?;
    for (i, prediction) in predictions.iter().enumerate() {
        csv_writer.write_record(
            [datasets[i/n_repetitions].file_name().to_string_lossy().into_owned()]
                .into_iter()
                .chain(prediction.iter().map(|f| f.to_string())),
        )?;
    }
    csv_writer.flush()?;

    Ok(())
}
