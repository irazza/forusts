use std::fmt;

use super::statistics::quartiles;

const ANOMALY_SCORE: f64 = 2.0;
const TRIM: f64 = 0.05;

pub trait Aggregation: Clone + Send + Sync {
    fn combine(&self, values: &[f64], average_path_length: f64) -> f64;
}

#[derive(Clone)]
pub struct Combiner {
    pub aggregation: AggregationFunction,
    pub quantile: Option<Quantile>,
}
impl Combiner {
    pub fn new() -> Self {
        Self {
            aggregation: AggregationFunction::PROD,
            quantile: None,
        }
    }
    pub fn enumerate() -> impl Iterator<Item = Self> {
        [
            AggregationFunction::PROD,
            AggregationFunction::SUM,
            AggregationFunction::TRIMMEDSUM,
            AggregationFunction::MEDIAN,
            AggregationFunction::MIN,
            AggregationFunction::MAX,
        ]
        .into_iter()
        .flat_map(|aggregation| {
            [Quantile::Q1, Quantile::Q2, Quantile::Q3, Quantile::Q4]
                .into_iter()
                .map(move |quantile| Self {
                    aggregation: aggregation.clone(),
                    quantile: Some(quantile),
                })
        })
    }
}
impl Aggregation for Combiner {
    fn combine(&self, values: &[f64], average_path_length: f64) -> f64 {
        let (_, values) = quartiles(
            values,
            match self.quantile {
                Some(Quantile::Q1) => 1,
                Some(Quantile::Q2) => 2,
                Some(Quantile::Q3) => 3,
                Some(Quantile::Q4) => 4,
                None => 4,
            },
        );
        self.aggregation.combine(&values, average_path_length)
    }
}
impl fmt::Display for Combiner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            match &self.quantile {
                Some(quantile) => {
                    format!("{}_", quantile)
                }
                None => {
                    "".to_string()
                }
            },
            self.aggregation,
        )
    }
}

#[derive(Clone)]
pub enum Quantile {
    Q1,
    Q2,
    Q3,
    Q4,
}
impl fmt::Display for Quantile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Quantile::Q1 => {
                    "Q1"
                }
                Quantile::Q2 => {
                    "Q2"
                }
                Quantile::Q3 => {
                    "Q3"
                }
                Quantile::Q4 => {
                    "Q4"
                }
            }
        )
    }
}
#[derive(Clone)]
pub enum AggregationFunction {
    PROD,
    SUM,
    TRIMMEDSUM,
    MEDIAN,
    MIN,
    MAX,
}
impl Aggregation for AggregationFunction {
    fn combine(&self, values: &[f64], average_path_length: f64) -> f64 {
        match self {
            AggregationFunction::PROD => {
                let mut result = 1.0;
                for value in values {
                    result *=
                        ANOMALY_SCORE.powf(-*value / (average_path_length * values.len() as f64));
                }
                result
            }
            AggregationFunction::SUM => {
                let mut result = 0.0;
                for value in values {
                    result +=
                        ANOMALY_SCORE.powf(-*value / (average_path_length * values.len() as f64));
                }
                result
            }
            AggregationFunction::TRIMMEDSUM => {
                let mut values = values.to_vec();
                values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let n = values.len();
                let p = (n as f64 * TRIM).round() as usize;

                let mut result = 0.0;
                for value in values[p..n - p].iter() {
                    result +=
                        ANOMALY_SCORE.powf(-*value / (average_path_length * values.len() as f64));
                }
                result
            }
            AggregationFunction::MEDIAN => {
                let mut values = values.to_vec();
                values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = values.len() / 2;
                if values.len() % 2 == 0 {
                    ANOMALY_SCORE.powf(
                        -((values[mid - 1] + values[mid]) / 2.0)
                            / (average_path_length * values.len() as f64),
                    )
                } else {
                    ANOMALY_SCORE.powf(-values[mid] / (average_path_length * values.len() as f64))
                }
            }
            AggregationFunction::MIN => {
                let value = values
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                ANOMALY_SCORE.powf(-*value / (average_path_length * values.len() as f64))
            }
            AggregationFunction::MAX => {
                let value = values
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                ANOMALY_SCORE.powf(-*value / (average_path_length * values.len() as f64))
            }
        }
    }
}
impl fmt::Display for AggregationFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                AggregationFunction::PROD => {
                    "PROD"
                }
                AggregationFunction::SUM => {
                    "SUM"
                }
                AggregationFunction::TRIMMEDSUM => {
                    "TRIMMEDSUM"
                }
                AggregationFunction::MEDIAN => {
                    "MEDIAN"
                }
                AggregationFunction::MIN => {
                    "MIN"
                }
                AggregationFunction::MAX => {
                    "MAX"
                }
            }
        )
    }
}
