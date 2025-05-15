#[derive(Clone, Copy, Debug)]
pub enum Subset {
    ALL,
    Q1,
    Q2,
    Q3,
    Q4,
    Q1Q2,
    Q3Q4,
    Q2Q3,
    Q1Q4,
    Q1Q3,
    Q2Q4,
    X84(f64),
    MODE(usize),
}

impl Subset {

    pub fn compute(self, x: &[f64]) -> Vec<f64> {
        // WARNING: These are not real quartiles
        let mut x = x.to_vec();
        x.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let q1 = (x.len() as f64 / 4.0) as usize;
        let q2 = (x.len() as f64 / 2.0) as usize;
        let q3 = (x.len() as f64 * (3.0 / 4.0)) as usize;

        match self {
            Subset::ALL => x.to_vec(),
            Subset::Q1 => x[..q1].to_vec(),
            Subset::Q2 => x[q1..q2].to_vec(),
            Subset::Q3 => x[q2..q3].to_vec(),
            Subset::Q4 => x[q3..].to_vec(),
            Subset::Q1Q2 => x[..q2].to_vec(),
            Subset::Q3Q4 => x[q2..].to_vec(),
            Subset::Q2Q3 => x[q1..q3].to_vec(),
            Subset::Q1Q4 => {
                let mut q1q4 = x[..q1].to_vec();
                q1q4.extend_from_slice(&x[q3..]);
                q1q4
            }
            Subset::Q1Q3 => x[..q3].to_vec(),
            Subset::Q2Q4 => x[q1..].to_vec(),
            Subset::X84(k) => {
                // Implement X84 rule - rejects points more than k*MAD from median
                let median_value = {
                    let mid = x.len() / 2;
                    if x.len() & 1 == 0 {
                        (x[mid - 1] + x[mid]) / 2.0
                    } else {
                        x[mid]
                    }
                };

                // Calculate median absolute deviation (MAD)
                let mut deviations: Vec<f64> =
                    x.iter().map(|&val| (val - median_value).abs()).collect();
                deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let mad = {
                    let mid = deviations.len() / 2;
                    if deviations.len() % 2 == 0 {
                        (deviations[mid - 1] + deviations[mid]) / 2.0
                    } else {
                        deviations[mid]
                    }
                };

                // Reject outliers: keep only values within k*MAD of median
                let v = x.into_iter()
                    .filter(|&val| (val - median_value).abs() <= k * mad)
                    .collect::<Vec<_>>();
                // println!("DBG X84 len() : {}", v.len());
                v
            }
            Subset::MODE(n_bins) => {
                let min = *x.last().unwrap(); // it is sorted descending
                let max = *x.first().unwrap();
                let bin_width = (max - min) / n_bins as f64;

                // Create bins and count frequencies
                let mut bins = vec![0; n_bins];
                for &value in &x {
                    let bin_index = ((value - min) / bin_width).floor() as usize;
                    let bin_index = bin_index.min(n_bins - 1); // Ensure index is within bounds
                    bins[bin_index] += 1;
                }

                // Find the most frequent bin
                let max_bin_count = bins.iter().max().unwrap();
                let most_frequent_bin_index = bins
                    .iter()
                    .position(|&count| count == *max_bin_count)
                    .unwrap();

                let v: Vec<f64> = x
                    .clone()
                    .into_iter()
                    .filter(|&value| {
                        ((value - min) / bin_width).floor() as usize == most_frequent_bin_index
                    })
                    .collect();
                // println!("DBG MODE len() : {}", v.len());
                v
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CombinerType {
    Prod,
    Sum,
    TSum,
    Median,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug)]
pub struct Combiner {
    pub subset: Subset,
    pub combiner: CombinerType,
}
impl Combiner {
    pub fn default() -> Self {
        Combiner {
            subset: Subset::ALL,
            combiner: CombinerType::Prod,
        }
    }
    pub fn new(subset: Subset, combiner: CombinerType) -> Self {
        Combiner { subset, combiner }
    }
    pub fn compute(self, x: &[f64], average_path_length: f64) -> f64 {
        let subset = self.subset.compute(x);
        let scores = if self.combiner == CombinerType::Prod {
            subset
        } else {
            subset.iter().map(|&v| 2.0f64.powf(-v / average_path_length)).collect::<Vec<_>>()
        };
        let score = match self.combiner {
            CombinerType::Prod => {
                let prod = scores.iter().sum::<f64>() / scores.len() as f64;
                2.0f64.powf(-prod / average_path_length)
            }
            CombinerType::Sum => {
                let sum = scores.iter().sum::<f64>() / scores.len() as f64;
                sum
            }
            CombinerType::TSum => {
                let s = (scores.len() as f64 * 0.05) as usize;
                let e = (scores.len() as f64 * 0.95) as usize;
                let tsum = scores[s..e].iter().sum::<f64>() / (scores.len() as f64 * 0.9);
                tsum
            }
            CombinerType::Median => {
                let n = scores.len();
                let median = if n % 2 == 0 {
                    (scores[n / 2 - 1] + scores[n / 2]) / 2.0
                } else {
                    scores[n / 2]
                };
                median
            }
            CombinerType::Min => {
                let min = scores[0];
                min
            }
            CombinerType::Max => {
                let max = scores[scores.len() - 1];
                max
            }
        };
        score
    }
}
