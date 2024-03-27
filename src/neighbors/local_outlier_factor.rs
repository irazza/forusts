// TODO FIX THIS BECAUSE IT DOES NOT WORK CORRECLTY, NEED TO HANDLE DUPLICATE IN THE DISTANCES
fn transpose<T: Copy>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut transposed = Vec::new();
    for i in 0..matrix[0].len() {
        let mut row = Vec::new();
        for j in 0..matrix.len() {
            row.push(matrix[j][i]);
        }
        transposed.push(row);
    }
    transposed
}

fn reachability_distance(x: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let k_distances = x
        .iter()
        .cloned()
        .map(|mut x| {
            let (_, n, _) = x.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap());
            *n
        })
        .collect::<Vec<_>>();
    let mut reachability_distance = Vec::new();
    for i in 0..k_distances.len() {
        let mut reachability_distance_i = Vec::new();
        for j in 0..x[i].len() {
            reachability_distance_i.push(f64::max(x[i][j], k_distances[i]));
        }
        reachability_distance.push(reachability_distance_i);
    }
    reachability_distance
}
fn local_reachability_density(x: &[Vec<f64>], k: usize) -> Vec<f64> {
    let reachability_distance = reachability_distance(&transpose(x), k);

    let mut local_reachability_density = Vec::new();
    for i in 0..x.len() {
        let mut x_clone = x[i].iter().copied().enumerate().collect::<Vec<_>>();
        let (priork, _, _) =
            x_clone.select_nth_unstable_by(k, |(_, a), (_, b)| a.partial_cmp(b).unwrap());
        let mut local_reachability_density_i = 0.0;
        for (j, _) in priork.iter() {
            local_reachability_density_i += reachability_distance[*j][i];
        }
        local_reachability_density.push(priork.len() as f64 / local_reachability_density_i);
    }
    local_reachability_density
}
pub fn local_outlier_factor(k: usize, x: &[Vec<f64>]) -> Vec<f64> {
    assert!(
        k > 0,
        "k must be greater than 0, otherwise the neighbors are not defined."
    );

    let local_reachability_density_test = local_reachability_density(x, k);
    let local_reachability_density_train = local_reachability_density(&transpose(x), k);
    let mut local_outlier_factor = Vec::new();
    for i in 0..x.len() {
        let mut x_clone = x[i].iter().copied().enumerate().collect::<Vec<_>>();
        let (priork, _, _) =
            x_clone.select_nth_unstable_by(k, |(_, a), (_, b)| a.partial_cmp(b).unwrap());
        let mut local_outlier_score = 0.0;
        for (j, _) in priork.iter() {
            local_outlier_score += local_reachability_density_train[*j];
        }
        local_outlier_factor
            .push(local_outlier_score / (priork.len() as f64 * local_reachability_density_test[i]));
    }
    local_outlier_factor
}
