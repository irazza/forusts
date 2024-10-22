#[warn(dead_code)]
use hashbrown::HashMap;
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

    let mut clusters = HashMap::new();
    for i in 0..n {
        clusters.insert(i, vec![i]);
    }

    for (i, step) in steps.iter().enumerate() {
        if i >= n - n_clusters {
            break;
        }

        let (mut a, mut b) = (
            clusters.remove(&step.cluster1).unwrap(),
            clusters.remove(&step.cluster2).unwrap(),
        );

        let new_cluster = if a.len() > b.len() {
            a.extend(b);
            a
        } else {
            b.extend(a);
            b
        };

        clusters.insert(n + i, new_cluster);
    }
    assert_eq!(n_clusters, clusters.len());
    let mut labels = vec![-1; n];
    for (i, cluster) in clusters.iter() {
        for &j in cluster {
            labels[j] = *i as isize;
        }
    }
    labels
}
