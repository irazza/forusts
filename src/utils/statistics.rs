use statrs::distribution::{ChiSquared, ContinuousCDF};

pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}
pub fn chi_square_proportion_test(
    precision_counts: &[(usize, usize)],
    p0: Option<f64>,
) -> (f64, f64) {
    // Calculate p0 (hypothesized proportion) if not provided
    let p0 = p0.unwrap_or_else(|| {
        let total_true_positives: usize = precision_counts.iter().map(|(count, _)| count).sum();
        let total_k: usize = precision_counts.iter().map(|(_, k)| k).sum();
        total_true_positives as f64 / total_k as f64
    });

    // Calculate the chi-square test statistic
    let mut test_statistic = 0.0;
    for &(count, k) in precision_counts {
        let expected_count = k as f64 * p0;
        test_statistic += (count as f64 - expected_count).powi(2) / (k as f64 * p0 * (1.0 - p0));
    }

    // Degrees of freedom = number of proportions - 1
    let df = precision_counts.len() as f64 - 1.0;

    // Calculate the p-value using the chi-squared distribution
    let chi_squared = ChiSquared::new(df).expect("Failed to create ChiSquared distribution");
    let p_value = 1.0 - chi_squared.cdf(test_statistic);

    (test_statistic, p_value)
}

#[test]
fn test_chi_square_proportion_test() {
    let precision_counts = vec![(10, 100), (20, 100), (30, 100), (40, 100)];
    let (_, p_value) = chi_square_proportion_test(&precision_counts, None);
    println!("p-value: {}", p_value);
}
