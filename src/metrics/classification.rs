#![allow(dead_code)]

use crate::feature_extraction::statistics::{argsort, cumsum, unique};

pub fn accuracy_score(y_pred: &Vec<usize>, y_true: &Vec<usize>) -> f64 {
    (y_pred
        .iter()
        .zip(y_true.iter())
        .filter(|&(a, b)| a == b)
        .count() as f64)
        / (y_true.len() as f64)
}

pub fn precision_score(y_pred: &Vec<usize>, y_true: &Vec<usize>) -> f64 {
    let confusion_matrix = confusion_matrix(y_pred, y_true);
    let mut precision = 0.0;

    for i in 0..confusion_matrix.len() {
        let tp = confusion_matrix[i][i];
        let fp = confusion_matrix[i].iter().sum::<usize>() - tp;

        if tp + fp > 0 {
            precision += tp as f64 / (tp + fp) as f64;
        }
    }

    precision / confusion_matrix.len() as f64
}

pub fn recall_score(y_pred: &Vec<usize>, y_true: &Vec<usize>) -> f64 {
    let confusion_matrix = confusion_matrix(y_pred, y_true);
    let mut recall = 0.0;

    for i in 0..confusion_matrix.len() {
        let tp = confusion_matrix[i][i];
        let fn_ = confusion_matrix.iter().map(|r| r[i]).sum::<usize>() - tp;

        if tp + fn_ > 0 {
            recall += tp as f64 / (tp + fn_) as f64;
        }
    }

    recall / confusion_matrix.len() as f64
}

pub fn confusion_matrix(y_pred: &Vec<usize>, y_true: &Vec<usize>) -> Vec<Vec<usize>> {
    let n_unique_true = y_true.iter().max().unwrap() + 1;
    let n_unique_pred = y_pred.iter().max().unwrap() + 1;
    assert!(n_unique_pred > 1, "y_pred contains only one class");
    let mut matrix = vec![vec![0; n_unique_true]; n_unique_true];
    for (a, b) in y_pred.iter().zip(y_true.iter()) {
        matrix[*a][*b] += 1;
    }

    matrix
}

pub fn matthews_corrcoef(y_pred: &Vec<usize>, y_true: &Vec<usize>) -> f64 {
    // Base case: only one class predicted
    if unique(y_true).len() == 1 || unique(y_pred).len() == 1 {
        return 0.0;
    }

    let confusion_matrix = confusion_matrix(y_pred, y_true);

    let t_sum: Vec<usize> = confusion_matrix.iter().map(|r| r.iter().sum()).collect();
    let p_sum: Vec<usize> = (0..confusion_matrix[0].len())
        .into_iter()
        .map(|c| confusion_matrix.iter().map(|r| r[c]).sum())
        .collect();
    let n_correct = confusion_matrix
        .iter()
        .enumerate()
        .map(|(i, r)| r[i])
        .sum::<usize>();
    let n_samples = p_sum.iter().sum::<usize>();
    let cov_ytyp = (n_correct * n_samples) as f64
        - t_sum
            .iter()
            .zip(p_sum.iter())
            .map(|(a, b)| a * b)
            .sum::<usize>() as f64;
    let cov_ypyp = (n_samples * n_samples) as f64
        - p_sum
            .iter()
            .zip(p_sum.iter())
            .map(|(a, b)| a * b)
            .sum::<usize>() as f64;
    let cov_ytyt = (n_samples * n_samples) as f64
        - t_sum
            .iter()
            .zip(t_sum.iter())
            .map(|(a, b)| a * b)
            .sum::<usize>() as f64;

    if cov_ypyp * cov_ytyt == 0.0 {
        return 0.0;
    } else {
        return cov_ytyp / (cov_ytyt * cov_ypyp).sqrt();
    }
}

pub fn roc_auc_score(y_pred: &Vec<f64>, y_true: &Vec<usize>) -> f64 {
    let (fpr, tpr, _) = roc_curve(y_pred, y_true);
    auc(&fpr, &tpr)
}

fn auc(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    let mut auc = 0.0;
    for i in 1..x.len() {
        auc += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0;
    }
    auc
}

fn roc_curve(y_pred: &Vec<f64>, y_true: &Vec<usize>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut fps, mut tps, mut threshold) = _binary_clf_curve(y_pred, y_true, 1);

    fps.insert(0, 0);
    tps.insert(0, 0);

    threshold.insert(0, f64::INFINITY);

    let fpr;
    let tpr;

    if fps[fps.len() - 1] <= 0 {
        println!("No negative samples in y_true, false positive value should be meaningless");
        fpr = vec![f64::NAN; fps.len()];
    } else {
        fpr = fps
            .iter()
            .map(|v| *v as f64 / fps[fps.len() - 1] as f64)
            .collect::<Vec<f64>>();
    }

    if tps[tps.len() - 1] <= 0 {
        println!("No positive samples in y_true, true positive value should be meaningless");
        tpr = vec![f64::NAN; tps.len()];
    } else {
        tpr = tps
            .iter()
            .map(|v| *v as f64 / tps[tps.len() - 1] as f64)
            .collect::<Vec<f64>>();
    }

    (fpr, tpr, threshold)
}

fn _binary_clf_curve(
    y_score: &Vec<f64>,
    y_true: &Vec<usize>,
    pos_label: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    // Transform y_true in a boolean vector
    let boolean_y_true = y_true
        .iter()
        .map(|v| *v == pos_label)
        .collect::<Vec<bool>>();

    let desc_score_indices = argsort(y_score, "desc");

    let y_score_ordered = desc_score_indices
        .iter()
        .map(|i| y_score[*i])
        .collect::<Vec<f64>>();
    let y_true_ordered = desc_score_indices
        .iter()
        .map(|i| if boolean_y_true[*i] { 1 } else { 0 })
        .collect::<Vec<usize>>();

    let mut threshold_idxs = (1..y_score_ordered.len())
        .into_iter()
        .filter(|i| y_score_ordered[*i] != y_score_ordered[*i - 1])
        .collect::<Vec<usize>>();

    threshold_idxs.push(y_true_ordered.len() - 1);

    let tps = cumsum(&y_true_ordered);

    let fps = threshold_idxs
        .iter()
        .map(|i| 1 + *i - tps[*i])
        .collect::<Vec<usize>>();

    (
        fps,
        tps,
        threshold_idxs
            .iter()
            .map(|i| y_score_ordered[*i])
            .collect::<Vec<f64>>(),
    )

    // // sort scores and corresponding truth values
    // let mut desc_score_indices = y_score
    //     .iter()
    //     .enumerate()
    //     .map(|(i, v)| (*v, i))
    //     .collect::<Vec<_>>();

    // desc_score_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().reverse());

    // let mut y_score_sorted = vec![0.0; y_score.len()];
    // let mut y_true_sorted = vec![0; y_true.len()];

    // for (i, (v, j)) in desc_score_indices.iter().enumerate() {
    //     y_score_sorted[i] = *v;
    //     y_true_sorted[i] = y_true[*j];
    // }

    // let distinct_value_indices = (1..y_score_sorted.len())
    //     .into_iter()
    //     .filter(|i| y_score_sorted[*i] != y_score_sorted[*i - 1])
    //     .collect::<Vec<_>>();

    // let mut threshold_idxs = vec![0; distinct_value_indices.len() + 1];
    // threshold_idxs[distinct_value_indices.len()] = y_score_sorted.len()-1;

    // for (i, v) in distinct_value_indices.iter().enumerate() {
    //     threshold_idxs[i] = *v;
    // }

    // let mut tps = vec![0; threshold_idxs.len()];
    // let mut fps = vec![0; threshold_idxs.len()];

    // for (i, v) in threshold_idxs.iter().enumerate() {
    //     for j in 0..*v {
    //         if y_true_sorted[j] == 1 {
    //             tps[i] += 1;
    //         } else {
    //             fps[i] += 1;
    //         }
    //     }
    // }

    // (tps, fps, threshold_idxs.iter().map(|i| y_score[*i]).collect::<Vec<f64>>())
}
