use csv::ReaderBuilder;
use ndarray::{Array2, s};
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;
mod random_forest;
mod decision_tree;

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("UCRArchive_2018/Adiac/Adiac_TRAIN.tsv")?;
    let mut reader = ReaderBuilder::new().has_headers(true)
        .delimiter(b'\t').
        from_reader(file);
    let data: Array2<f64> = reader.deserialize_array2_dynamic()?;
    let X = data.slice(s![.., 1..]).to_owned();
    let y = data.slice(s![.., 0]).to_owned().mapv(|x| x as usize);

    // Split the data into training and test sets
    let n_samples = X.shape()[0];
    let n_train = (n_samples as f64 * 0.8) as usize;
    let X_train = X.slice(s![..n_train, ..]).to_owned();
    let y_train = y.slice(s![..n_train]).to_owned();
    let X_test = X.slice(s![n_train.., ..]).to_owned();
    let y_test = y.slice(s![n_train..]).to_owned();

    // Train the model

    let mut clf = random_forest::RandomForest::new(100);
    clf.fit(&X_train, &y_train);
    let y_pred = clf.predict(&X_test);
    let accuracy = (y_pred.iter().zip(y_test.iter()).filter(|&(a, b)| a == b).count() as f32) / (y_test.len() as f32);

    println!("Accuracy: {}", accuracy);
    Ok(())
}
