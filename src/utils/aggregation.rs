use std::slice::Iter;

const ANOMALY_SCORE: f64 = 2.0;
const TRIM: f64 = 0.05;

pub trait Aggregation: Clone + Send + Sync {
    fn combine(&self, values: &[f64], average_path_length: f64) -> f64;
}

#[derive(Clone)]
pub enum Combiner {
    PROD,
    SUM,
    TRIMMEDSUM,
    MEDIAN,
    MIN,
    MAX,
}
impl Aggregation for Combiner {
    fn combine(&self, values: &[f64], average_path_length: f64) -> f64 {
        match self {
            Combiner::PROD => {
                let mut result = 1.0;
                for value in values {
                    result *=
                        ANOMALY_SCORE.powf(-*value / (average_path_length * values.len() as f64));
                }
                result
            }
            Combiner::SUM => {
                let mut result = 0.0;
                for value in values {
                    result +=
                        ANOMALY_SCORE.powf(-*value / (average_path_length * values.len() as f64));
                }
                result
            }
            Combiner::TRIMMEDSUM => {
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
            Combiner::MEDIAN => {
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
            Combiner::MIN => {
                let value = values
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                ANOMALY_SCORE.powf(-*value / (average_path_length * values.len() as f64))
            }
            Combiner::MAX => {
                let value = values
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                ANOMALY_SCORE.powf(-*value / (average_path_length * values.len() as f64))
            }
        }
    }
}
