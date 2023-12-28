use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::utils::structures::Sample;

pub fn read_csv(
    path: impl AsRef<Path>,
    delimiter: u8,
    header: bool,
) -> Result<Vec<Sample<'static>>, Box<dyn Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut reader = ReaderBuilder::new()
        .has_headers(header)
        .delimiter(delimiter)
        .from_reader(reader);

    let mut samples = Vec::new();

    for result in reader.deserialize() {
        let record: Sample = result.unwrap();
        samples.push(record);
    }
    Ok(samples)
}

pub fn write_csv(
    path: impl AsRef<Path>,
    data: Vec<Vec<f64>>,
    header: Vec<String>,
    index: Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let mut csv_writer = csv::Writer::from_path(path)?;
    csv_writer.write_record(header)?;
    for (i, prediction) in data.iter().enumerate() {
        csv_writer.write_record(
            [index[i].clone()]
                .into_iter()
                .chain(prediction.iter().map(|f| f.to_string())),
        )?;
    }
    csv_writer.flush()?;
    Ok(())
}

pub fn vec_to_csv(path: impl AsRef<Path>, data: &[f64]) -> Result<(), Box<dyn Error>> {
    let mut csv_writer = csv::Writer::from_path(path)?;
    csv_writer.write_record(data.iter().map(|v| v.to_string()))?;
    csv_writer.flush()?;
    Ok(())
}

pub fn read_csv_to_vec(path: impl AsRef<Path>) -> Result<Vec<f64>, Box<dyn Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_reader(reader);

    let mut samples = Vec::new();

    for result in reader.deserialize() {
        let record: f64 = result.unwrap();
        samples.push(record);
    }
    Ok(samples)
}
