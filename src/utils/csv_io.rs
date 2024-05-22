#![allow(dead_code)]
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use crate::utils::structures::Sample;

pub fn read_a_file(path_to_file: &str) -> Vec<String> {
    // Open the file
    let file = match File::open(path_to_file) {
        Ok(file) => file,
        Err(_) => {
            panic!("Error: Unable to open the file");
        }
    };

    // Create a buffer reader to read the file line by line
    let reader = BufReader::new(file);

    // Create a vector to store the lines
    let mut lines: Vec<String> = Vec::new();

    // Read each line and push it into the vector
    for line in reader.lines() {
        match line {
            Ok(line) => lines.push(line),
            Err(_) => {
                panic!("Error: Unable to read a line from the file");
            }
        }
    }
    lines
}

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
            data: Arc::new(
                record
                    .data
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

pub fn write_csv(
    path: impl AsRef<Path>,
    data: &Vec<Sample>,
    header: Option<Vec<String>>,
) -> Result<(), Box<dyn Error>> {
    // Make dir if it does not exist
    let parent = path.as_ref().parent().unwrap();
    if !parent.exists() {
        std::fs::create_dir_all(parent)?;
    }
    let mut csv_writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_path(path)?;
    if let Some(header) = header {
        csv_writer.write_record(header)?;
    }
    for record in data {
        csv_writer.serialize(record)?;
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

pub fn vec_vec_to_csv(path: impl AsRef<Path>, data: &[Vec<f64>]) -> Result<(), Box<dyn Error>> {
    // Create path if it does not exist
    let parent = path.as_ref().parent().unwrap();
    if !parent.exists() {
        std::fs::create_dir_all(parent)?;
    }
    let mut csv_writer = csv::Writer::from_path(path)?;
    for record in data {
        csv_writer.write_record(record.iter().map(|v| v.to_string()))?;
    }
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
