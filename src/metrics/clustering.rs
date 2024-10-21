#![allow(dead_code)]
use hashbrown::HashMap;

pub fn adjusted_rand_score(y_pred: &[isize], y_true: &[isize]) -> f64 {
    let n = y_true.len();
    assert_eq!(n, y_pred.len());

    let mut contingency_table: HashMap<(isize, isize), usize> = HashMap::new();
    let mut a: HashMap<isize, usize> = HashMap::new();
    let mut b: HashMap<isize, usize> = HashMap::new();

    for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
        *contingency_table
            .entry((true_label, pred_label))
            .or_insert(0) += 1;
        *a.entry(true_label).or_insert(0) += 1;
        *b.entry(pred_label).or_insert(0) += 1;
    }

    let sum_comb_c = contingency_table
        .values()
        .map(|&n_ij| comb2(n_ij))
        .sum::<usize>();
    let sum_comb_a = a.values().map(|&n_i| comb2(n_i)).sum::<usize>();
    let sum_comb_b = b.values().map(|&n_j| comb2(n_j)).sum::<usize>();

    let comb_n = comb2(n);
    let index = sum_comb_c as f64;
    let expected_index = (sum_comb_a as f64 * sum_comb_b as f64) / comb_n as f64;
    let max_index = (sum_comb_a + sum_comb_b) as f64 / 2.0;

    (index - expected_index) / (max_index - expected_index)
}

#[inline(always)]
fn comb2(n: usize) -> usize {
    n * (n - 1) / 2
}

#[test]
pub fn test_ari() {
    let y_true = vec![0, 0, 1, 1];
    let y_pred = vec![1, 1, 0, 0];
    let ari = adjusted_rand_score(&y_pred, &y_true);
    println!("Adjusted Rand Index: {:.2}", ari);
}
