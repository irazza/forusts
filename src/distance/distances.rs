use dtw_rs::{Algorithm, DynamicTimeWarping};

pub fn euclidean(x1: &[f64], x2: &[f64]) -> f64 {
    x1.iter()
        .zip(x2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
pub fn test_twe() {
    let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let s2 = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let result = twe(&s1, &s2);
    println!("TWE: {}", result);
}

pub fn twe(x1: &[f64], x2: &[f64]) -> f64 {
    // Insert 0.0 at the beginning of the vectors
    let x1 = vec![0.0]
        .into_iter()
        .chain(x1.to_vec().into_iter())
        .collect::<Vec<f64>>();
    let x2 = vec![0.0]
        .into_iter()
        .chain(x2.to_vec().into_iter())
        .collect::<Vec<f64>>();

    let n = x1.len();
    let m = x2.len();

    let nu = 0.1;
    let lambda = 0.2;

    let mut current = vec![f64::INFINITY; m];
    let mut previous = vec![f64::INFINITY; m];

    previous[0] = 0.0;

    for i in 1..n {
        for j in 1..m {
            let mut current_cost = vec![f64::NAN; 3];

            // Deletion x1
            current_cost[0] = previous[j] + (x1[i - 1] - x1[i]).abs() + nu + lambda;

            // Deletion x2
            current_cost[1] = current[j - 1] + (x2[j - 1] - x2[j]).abs() + nu + lambda;

            // Match
            current_cost[2] = previous[j - 1]
                + (x1[i] - x2[j]).abs()
                + (x1[i - 1] - x2[j - 1]).abs()
                + (2.0 * ((i as isize - j as isize) as f64).abs()) * nu;

            current[j] = current_cost
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .clone();
        }

        previous = current.clone();
    }
    current[m - 1]
}

#[test]
pub fn test_dtw() {
    let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let s2 = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let result = dtw(&s1, &s2);
    println!("DTW: {}", result);
}

pub fn dtw(x1: &[f64], x2: &[f64]) -> f64 {
    let dtw = DynamicTimeWarping::between(x1, x2);
    dtw.distance()
}
