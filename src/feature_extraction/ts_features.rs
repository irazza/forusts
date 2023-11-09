pub fn mean(x: &Vec<f64>) -> f64
{
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    assert!(mean.is_finite());
    mean
}

pub fn max(x: &Vec<f64>) -> f64
{
    let max = x.iter().fold(f64::MIN, |max, &val| max.max(val));
    assert!(max.is_finite());
    max
}

pub fn min(x: &Vec<f64>) -> f64
{
    let min = x.iter().fold(f64::MAX, |min, &val| min.min(val));
    assert!(min.is_finite());
    min
}

pub fn median(x: &Vec<f64>) -> f64
{
    let mut x = x.clone();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if x.len() % 2 == 0 {
        (x[x.len() / 2 - 1] + x[x.len() / 2]) / 2.0
    } else {
        x[x.len() / 2]
    };
    assert!(median.is_finite());
    median
}

pub fn std(x: &Vec<f64>) -> f64
{
    let mean = mean(x);
    let std = (x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
    assert!(std.is_finite());
    std
}

pub fn slope(x: &Vec<f64>) -> f64 {
    let n = x.len();

    let x_mean = x.iter().sum::<f64>() / n as f64;

    let y = (1..n + 1).map(|x| x as f64).collect::<Vec<f64>>();
    let y_mean = y.iter().sum::<f64>() / n as f64;

    let xy_mean = x.iter().zip(y.iter()).map(|(x, y)| x * y).sum::<f64>() / n as f64;
    let y2_mean = y.iter().map(|y| y.powi(2)).sum::<f64>() / n as f64;

    let slope = (xy_mean - x_mean * y_mean) / (y2_mean - y_mean.powi(2));
    assert!(slope.is_finite());
    slope
}

pub fn histcounts(x: &Vec<f64>, n_bins: usize) -> (Vec<i32>, Vec<f64>) {
    // Check min and max of input arrax
    let (min_val, max_val) = x.iter().fold((f64::MAX, f64::MIN), |(min, max), &val| {
        (min.min(val), max.max(val))
    });

    // Derive bin width from it
    let bin_step = (max_val - min_val) / n_bins as f64;

    // Variable to store counted occurrences in
    let mut bin_counts = vec![0; n_bins as usize];

    for val in x {
        let bin_ind = ((val - min_val) / bin_step).floor() as usize;
        bin_counts[bin_ind.min(n_bins as usize - 1)] += 1;
    }

    // Calculate bin edges
    let bin_edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 * bin_step + min_val).collect();

    (bin_counts, bin_edges)
}

