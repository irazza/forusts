use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use crate::random_forest::MaxFeatures;
mod decision_tree;
mod random_forest;
mod nearest_neighbour;

struct Dataset {
    targets: Vec<usize>,
    data: Vec<Vec<f64>>,
}

fn read_csv(path: &str) -> Result<Dataset, Box<dyn Error>> {
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
            targets.push(target.parse()?);
        }

        for value in record.iter().skip(1) {
            row.push(value.parse()?);
        }

        data.push(row);
    }

    Ok(Dataset { targets, data })
}

fn main() -> Result<(), Box<dyn Error>> {
    let ds_train = read_csv("UCRArchive_2018/Adiac/Adiac_TEST.tsv")?;
    let ds_test = read_csv("UCRArchive_2018/Adiac/Adiac_TRAIN.tsv")?;

    // Train the model

    let mut clf = random_forest::RandomForest::new(1000, MaxFeatures::Sqrt, Some(1000));
    // check fitting time
    let start = std::time::Instant::now();
    clf.fit(&ds_train.data, &ds_train.targets);
    let duration = start.elapsed();
    println!("Time elapsed in training is: {:?}", duration);
    let y_pred = clf.predict(&ds_test.data);
    let accuracy = (y_pred
        .iter()
        .zip(ds_test.targets.iter())
        .filter(|&(a, b)| a == b)
        .count() as f32)
        / (ds_test.targets.len() as f32);

    println!("Accuracy: {}", accuracy);
    let start = std::time::Instant::now();
    let breiman_distance = clf.pairwise_breiman(ds_test.data.clone(), ds_train.data.clone());
    let duration = start.elapsed();
    println!("Time elapsed in training is: {:?}", duration);
    let prediction_breiman = nearest_neighbour::k_nearest_neighbour(1, &ds_train.targets, &breiman_distance);
    let accuracy = (prediction_breiman
        .iter()
        .zip(ds_test.targets.iter())
        .filter(|&(a, b)| a == b)
        .count() as f32)
        / (ds_test.targets.len() as f32);
    println!("Accuracy: {}", accuracy);
    let ancestor_distance = clf.pairwise_ancestor(ds_test.data, ds_train.data);
    println!("Time elapsed in training is: {:?}", duration);
    let prediction_ancestor = nearest_neighbour::k_nearest_neighbour(1, &ds_train.targets, &ancestor_distance);
    let accuracy = (prediction_ancestor
        .iter()
        .zip(ds_test.targets.iter())
        .filter(|&(a, b)| a == b)
        .count() as f32)
        / (ds_test.targets.len() as f32);
    println!("Accuracy: {}", accuracy);
    Ok(())
}
