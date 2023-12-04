#![allow(dead_code)]
use csv::ReaderBuilder;
use hashbrown::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub struct Dataset {
    targets: Vec<usize>,
    data: Vec<Vec<f64>>,
}

impl Dataset {
    pub fn get_targets(&self) -> &Vec<usize> {
        &self.targets
    }

    pub fn get_data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }
}

pub fn read_csv(
    path: impl AsRef<Path>,
    delimiter: u8,
    mapping: &mut HashMap<isize, i32>,
) -> Result<Dataset, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(delimiter)
        .from_reader(reader);

    let mut targets = Vec::new();
    let mut data = Vec::new();

    let mut class_counter = 0;

    for result in reader.records() {
        let record = result?;
        let mut row = Vec::new();

        // Assuming the first column is the target and the rest are data
        if let Some(target) = record.get(0) {
            let class = target.parse::<f64>()? as isize;
            let remapped_class = if mapping.contains_key(&class) {
                mapping.get(&class).unwrap()
            } else {
                mapping.insert(class, class_counter);
                class_counter += 1;
                mapping.get(&class).unwrap()
            };
            targets.push(*remapped_class as usize);
        }

        for value in record.iter().skip(1) {
            row.push(value.parse()?);
        }

        data.push(row);
    }

    Ok(Dataset { targets, data })
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

pub fn vec_to_csv(
    path: impl AsRef<Path>,
    data: &Vec<f64>
) -> Result<(), Box<dyn Error>> {
    let mut csv_writer = csv::Writer::from_path(path)?;
    csv_writer.write_record(data.iter().map(|v| v.to_string()))?;
    csv_writer.flush()?;
    Ok(())
}
