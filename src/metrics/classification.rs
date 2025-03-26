use std::cmp::Ordering;

use crate::utils::statistics::{argsort, class_counts, unique};

pub fn accuracy_score(y_pred: &[isize], y_true: &[isize]) -> f64 {
    (y_pred
        .iter()
        .zip(y_true.iter())
        .filter(|&(a, b)| a == b)
        .count() as f64)
        / (y_true.len() as f64)
}

pub fn f1_score(y_pred: &[isize], y_true: &[isize]) -> f64 {
    let precision = precision_score(y_pred, y_true);
    let recall = recall_score(y_pred, y_true);

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * (precision * recall) / (precision + recall)
    }
}

pub fn precision_score(y_pred: &[isize], y_true: &[isize]) -> f64 {
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

pub fn recall_score(y_pred: &[isize], y_true: &[isize]) -> f64 {
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

pub fn confusion_matrix(y_pred: &[isize], y_true: &[isize]) -> Vec<Vec<usize>> {
    let n_unique_true = y_true.iter().max().unwrap() + 1;
    let n_unique_pred = y_pred.iter().max().unwrap() + 1;
    assert!(n_unique_pred > 1, "y_pred contains only one class");
    let mut matrix = vec![vec![0; n_unique_true as usize]; n_unique_true as usize];
    for (a, b) in y_pred.iter().zip(y_true.iter()) {
        matrix[*a as usize][*b as usize] += 1;
    }

    matrix
}

pub fn matthews_corrcoef(y_pred: &[isize], y_true: &[isize]) -> f64 {
    // Base case: only one class predicted
    if class_counts(y_true) == 1 || class_counts(y_pred) == 1 {
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

pub fn precision_at_k(y_pred: &[f64], y_true: &[isize], k: usize) -> f64 {
    // Sort the predictions
    let indices = argsort(&y_pred);
    let indices = &indices[y_pred.len() - k..];

    // Initialize variables to store the number of true positives and the precision at k
    let mut n_true_positives = 0;

    // Iterate through the top k predictions
    for i in indices {
        if y_true[*i] == 1 {
            // Increment the number of true positives when the prediction is correct
            n_true_positives += 1;
        }
    }

    // Calculate the precision at k
    n_true_positives as f64 / k as f64
}

pub fn pr_auc_score(y_pred: &[f64], y_true: &[isize]) -> f64 {
    // Calculate PR curve
    let (recalls, precisions, _) = pr_curve(y_pred, y_true);

    // Calculate AUC from the PR curve
    let auc_value = auc(&recalls, &precisions);

    auc_value
}

pub fn roc_auc_score(y_pred: &[f64], y_true: &[isize]) -> f64 {
    // Calculate ROC curve
    let (fprs, tprs, _) = roc_curve(y_pred, y_true);

    // Calculate AUC from the ROC curve
    let auc_value = auc(&fprs, &tprs);

    auc_value
}

fn auc(x: &[f64], y: &[f64]) -> f64 {
    // Sort the vectors based on x (false positive rates) in ascending order
    let mut sorted_data: Vec<_> = x.iter().zip(y.iter()).collect();
    sorted_data.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());

    // Calculate AUC using the trapezoidal rule
    let mut auc_value = 0.0;
    let n = sorted_data.len();

    for i in 0..(n - 1) {
        let x1 = sorted_data[i].0;
        let x2 = sorted_data[i + 1].0;
        let y1 = sorted_data[i].1;
        let y2 = sorted_data[i + 1].1;

        // Calculate the area of the trapezoid and add it to the total AUC
        auc_value += 0.5 * (x2 - x1) * (y1 + y2);
    }

    auc_value
}

pub fn true_positive_rate(y_pred: &[usize], y_true: &[isize]) -> f64 {
    // Ensure the input vectors have the same length
    assert_eq!(
        y_pred.len(),
        y_true.len(),
        "Input vectors must have the same length"
    );

    // Ensure that is a binary problem
    assert_eq!(
        class_counts(y_true),
        2,
        "TPR is only defined for binary problems"
    );

    // Count true positives (TP) and false negatives (FN)
    let (mut true_positives, mut false_negatives) = (0usize, 0usize);

    for (&pred, &true_val) in y_pred.iter().zip(y_true.iter()) {
        if pred == 1 && true_val == 1 {
            // Increment true positives when prediction is positive and true value is positive
            true_positives += 1;
        } else if pred == 0 && true_val == 1 {
            // Increment false negatives when prediction is negative, but true value is positive
            false_negatives += 1;
        }
    }
    // Calculate true positive rate
    let tpr = if true_positives == 0 {
        0.0 // Handle the case where there are no true positives to avoid division by zero
    } else {
        true_positives as f64 / (true_positives + false_negatives) as f64
    };

    tpr
}

pub fn false_positive_rate(y_pred: &[usize], y_true: &[isize]) -> f64 {
    // Ensure the input vectors have the same length
    assert_eq!(
        y_pred.len(),
        y_true.len(),
        "Input vectors must have the same length"
    );

    // Ensure that is a binary problem
    assert_eq!(
        class_counts(y_true),
        2,
        "FPR is only defined for binary problems"
    );

    // Count false positives (FP) and true negatives (TN)
    let (mut false_positives, mut true_negatives) = (0usize, 0usize);

    for (&pred, &true_val) in y_pred.iter().zip(y_true.iter()) {
        if pred == 1 && true_val == 0 {
            // Increment false positives when prediction is positive, but true value is negative
            false_positives += 1;
        } else if pred == 0 && true_val == 0 {
            // Increment true negatives when both prediction and true value are negative
            true_negatives += 1;
        }
    }

    // Calculate false positive rate
    let fpr = if true_negatives + false_positives == 0 {
        0.0 // Handle the case where there are no true negatives to avoid division by zero
    } else {
        false_positives as f64 / (true_negatives + false_positives) as f64
    };

    fpr
}

fn roc_curve(y_pred: &[f64], y_true: &[isize]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Ensure that is a binary problem
    assert_eq!(
        class_counts(y_true),
        2,
        "ROC curve is only defined for binary problems"
    );

    // Initialize vectors to store true positive rate (sensitivity), false positive rate, and thresholds
    let mut tprs = Vec::new();
    let mut fprs = Vec::new();
    let thresholds = unique(y_pred);

    let mut pred_with_true_class = y_pred
        .iter()
        .copied()
        .zip(y_true.iter().copied())
        .collect::<Vec<_>>();
    pred_with_true_class.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut true_positives = 0;
    let mut false_positives = 0;

    let mut false_negatives = pred_with_true_class.iter().filter(|x| x.1 == 1).count();
    let mut true_negatives = pred_with_true_class.iter().filter(|x| x.1 == 0).count();

    let mut index = 0;

    let not_zero = |x: usize| if x == 0 { 1 } else { x };

    // Iterate through a range of thresholds
    for threshold in &thresholds {
        // Create a binary vector based on the current threshold
        while index < pred_with_true_class.len() && pred_with_true_class[index].0 < *threshold {
            if pred_with_true_class[index].1 == 1 {
                true_positives += 1;
                false_negatives -= 1;
            } else {
                false_positives += 1;
                true_negatives -= 1;
            }
            index += 1;
        }

        // Store TPR, FPR, and threshold for the current iteration
        tprs.push(1.0 - true_positives as f64 / not_zero(true_positives + false_negatives) as f64);
        fprs.push(1.0 - false_positives as f64 / not_zero(true_negatives + false_positives) as f64);
    }
    (fprs, tprs, thresholds)
}

fn pr_curve(y_pred: &[f64], y_true: &[isize]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Ensure that is a binary problem
    assert_eq!(
        class_counts(y_true),
        2,
        "ROC curve is only defined for binary problems"
    );

    // Initialize vectors to store true positive rate (sensitivity), false positive rate, and thresholds
    let mut recalls = Vec::new();
    recalls.push(1.0);
    let mut precisions = Vec::new();
    precisions.push(y_true.iter().filter(|&&x| x == 1).count() as f64 / y_true.len() as f64);
    let mut thresholds = unique(y_pred);

    println!("{}, {}, {}", &recalls[0], &precisions[0], &thresholds[0]);

    let mut pred_with_true_class = y_pred
        .iter()
        .copied()
        .zip(y_true.iter().copied())
        .collect::<Vec<_>>();
    pred_with_true_class.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut true_positives = 0;
    let mut false_positives = 0;

    let mut false_negatives = pred_with_true_class.iter().filter(|x| x.1 == 1).count();
    let mut true_negatives = pred_with_true_class.iter().filter(|x| x.1 == 0).count();

    let mut index = 0;

    let not_zero = |x: usize| if x == 0 { 1 } else { x };

    // Iterate through a range of thresholds
    for threshold in &thresholds {
        // Create a binary vector based on the current threshold
        while index < pred_with_true_class.len() && pred_with_true_class[index].0 < *threshold {
            if pred_with_true_class[index].1 == 1 {
                true_positives += 1;
                false_negatives -= 1;
            } else {
                false_positives += 1;
                true_negatives -= 1;
            }
            index += 1;
        }

        // Store TPR, FPR, and threshold for the current iteration
        let recall = true_positives as f64 / not_zero(true_positives + false_negatives) as f64;
        let precision = true_positives as f64 / not_zero(true_positives + false_positives) as f64;
        println!("{}, {}, {}", recall, precision, threshold);
        recalls.push(true_positives as f64 / not_zero(true_positives + false_negatives) as f64);
        precisions.push(true_positives as f64 / not_zero(true_positives + false_positives) as f64);
    }
    (precisions, recalls, thresholds)
}
// 4.000000000000000222e-01
// 4.444444444444444198e-01
// 3.750000000000000000e-01
// 4.285714285714285476e-01
// 3.333333333333333148e-01
// 4.000000000000000222e-01
// 5.000000000000000000e-01
// 3.333333333333333148e-01
// 5.000000000000000000e-01
// 0.000000000000000000e+00
// 1.000000000000000000e+00