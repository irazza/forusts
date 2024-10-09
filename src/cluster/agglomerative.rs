use kodama::linkage;

pub fn agglomerative_clustering(
    n_clusters: usize,
    linkage_method: kodama::Method,
    distance_matrix: Vec<Vec<f64>>,
) -> Vec<isize> {
    let n = distance_matrix.len();
    let mut condensed_matrix = Vec::with_capacity((n * (n - 1)) / 2);
    for i in 0..n - 1 {
        for j in i + 1..n {
            condensed_matrix.push(distance_matrix[i][j]);
        }
    }
    let dendrogram = linkage(&mut condensed_matrix, n, linkage_method);
    let steps = dendrogram.steps();

    todo!()
}
