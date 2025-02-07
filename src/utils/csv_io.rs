use csv::ReaderBuilder;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::utils::structures::Sample;

pub fn read_csv(
    path: impl AsRef<Path>,
    delimiter: u8,
    header: bool,
) -> Result<Vec<Sample>, Box<dyn Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut reader = ReaderBuilder::new()
        .has_headers(header)
        .delimiter(delimiter)
        .from_reader(reader);

    let mut samples = Vec::new();

    for result in reader.deserialize() {
        let record: Sample = result.unwrap();
        // Replace NaNs with 0s
        let record = Sample {
            target: record.target,
            features: Arc::new(
                record
                    .features
                    .to_vec()
                    .iter()
                    .map(|v| if v.is_nan() { 0.0 } else { *v })
                    .collect(),
            ),
        };
        samples.push(record);
    }
    Ok(samples)
}

pub fn write_csv<T>(path: impl AsRef<Path>, data: Vec<Vec<T>>, header: Option<Vec<String>>)
where
    T: std::fmt::Display,
{
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut writer = csv::Writer::from_path(path).unwrap();
    if header.is_some() {
        writer.write_record(header.unwrap()).unwrap();
    }
    for row in data {
        writer
            .write_record(row.iter().map(|v| v.to_string()))
            .unwrap();
    }
    writer.flush().unwrap();
}

pub fn write_bin<T>(path: impl AsRef<Path>, data: Vec<T>) -> Result<(), std::io::Error>
where
    T: serde::Serialize,
{
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).unwrap();
    }

    let mut f = OpenOptions::new().create(true).append(true).open(path).unwrap();
    let encoded: Vec<u8> = rmp_serde::to_vec(&data).unwrap();
    f.write_all(&encoded)?;
    f.flush()
}

pub fn read_bin<T>(path: impl AsRef<Path>) -> Vec<T>
where
    T: serde::de::DeserializeOwned,
{
    let mut f = BufReader::new(File::open(path).unwrap());
    let mut decoded = Vec::new();
    while f.fill_buf().unwrap().len() > 0 {
        let line = rmp_serde::from_read::<_, T>(&mut f).expect("Error reading binary file");
        decoded.push(line);
    }
    decoded
}
