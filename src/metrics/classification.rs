#![allow(dead_code)]

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
    let n_unique = y_true.iter().max().unwrap() + 1;
    let mut matrix = vec![vec![0; n_unique]; n_unique];

    for (a, b) in y_pred.iter().zip(y_true.iter()) {
        matrix[*a][*b] += 1;
    }

    matrix
}

pub fn matthews_corrcoef(y_pred: &Vec<usize>, y_true: &Vec<usize>) -> f64 {
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
