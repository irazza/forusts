use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
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

pub fn write_csv(path: impl AsRef<Path>, data: Vec<Vec<f64>>) {
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut writer = csv::Writer::from_path(path).unwrap();
    for row in data {
        writer
            .write_record(row.iter().map(|v| v.to_string()))
            .unwrap();
    }
    writer.flush().unwrap();
}
