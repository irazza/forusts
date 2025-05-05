#[derive(Clone, Copy)]
pub enum Quartile {
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
}

impl Quartile {
    pub fn iter() -> impl Iterator<Item = Quartile> {
        [
            Quartile::ALL,
            Quartile::Q1,
            Quartile::Q2,
            Quartile::Q3,
            Quartile::Q4,
            Quartile::Q1Q2,
            Quartile::Q3Q4,
            Quartile::Q2Q3,
            Quartile::Q1Q4,
            Quartile::Q1Q3,
            Quartile::Q2Q4,
        ]
        .iter()
        .copied()
    }

    pub fn len() -> usize {
        Quartile::iter().count()
    }

    pub fn compute(self, x: &[f64]) -> Vec<f64> {
        // WARNING: These are not real quartiles
        let mut x = x.to_vec();
        x.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let q1 = (x.len() as f64 / 4.0) as usize;
        let q2 = (x.len() as f64 / 2.0) as usize;
        let q3 = (x.len() as f64 * (3.0 / 4.0)) as usize;

        match self {
            Quartile::ALL => x.to_vec(),
            Quartile::Q1 => x[..q1].to_vec(),
            Quartile::Q2 => x[q1..q2].to_vec(),
            Quartile::Q3 => x[q2..q3].to_vec(),
            Quartile::Q4 => x[q3..].to_vec(),
            Quartile::Q1Q2 => x[..q2].to_vec(),
            Quartile::Q3Q4 => x[q2..].to_vec(),
            Quartile::Q2Q3 => x[q1..q3].to_vec(),
            Quartile::Q1Q4 => {
                let mut q1q4 = x[..q1].to_vec();
                q1q4.extend_from_slice(&x[q3..]);
                q1q4
            }
            Quartile::Q1Q3 => x[..q3].to_vec(),
            Quartile::Q2Q4 => x[q1..].to_vec(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum CombinerType {
    Prod,
    Sum,
    TSum,
    Median,
    Min,
    Max,
}
impl CombinerType {
    pub fn iter() -> impl Iterator<Item = CombinerType> {
        [
            CombinerType::Prod,
            CombinerType::Sum,
            CombinerType::TSum,
            CombinerType::Median,
            CombinerType::Min,
            CombinerType::Max,
        ]
        .iter()
        .copied()
    }

    pub fn len() -> usize {
        CombinerType::iter().count()
    }
}

#[derive(Clone, Copy)]
pub struct Combiner {
    pub quartile: Quartile,
    pub combiner: CombinerType,
}
impl Combiner {
    pub fn default() -> Self {
        Combiner {
            quartile: Quartile::ALL,
            combiner: CombinerType::Prod,
        }
    }
    pub fn compute(self, x: &[f64], average_path_length: f64) -> f64 {
        let x = self.quartile.compute(x);
        let value = match self.combiner {
            CombinerType::Prod => {
                let prod = x.iter().sum::<f64>() / x.len() as f64;
                prod
            }
            CombinerType::Sum => {
                let sum = x.iter().sum::<f64>() / x.len() as f64;
                sum
            }
            CombinerType::TSum => {
                let s = (x.len() as f64 * 0.05) as usize;
                let e = (x.len() as f64 * 0.95) as usize;
                let tsum = x[s..e].iter().sum::<f64>() / (x.len() as f64 * 0.9);
                tsum
            }
            CombinerType::Median => {
                let n = x.len();
                let median = if n % 2 == 0 {
                    (x[n / 2 - 1] + x[n / 2]) / 2.0
                } else {
                    x[n / 2]
                };
                median
            }
            CombinerType::Min => {
                let min = x[0];
                min
            }
            CombinerType::Max => {
                let max = x[x.len() - 1];
                max
            }
        };
        2.0f64.powf(-value / average_path_length)
    }
}
