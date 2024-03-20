use dtw_rs::{Algorithm, DynamicTimeWarping};

pub fn euclidean(x1: &[f64], x2: &[f64]) -> f64 {
    assert!(x1.len() == x2.len());
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
    // start time
    let start = std::time::Instant::now();
    let result = twe(&s1, &s2);
    println!("TWE: {}, time: {:?}", result, start.elapsed());
}

pub fn twe(x1: &[f64], x2: &[f64]) -> f64 {
    let nu = 0.1;
    let lambda = 0.2;
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

    let bounding_matrix: Vec<Vec<bool>> = create_bounding_matrix(&x1, &x2, 0.7);
    // print bounding_matrix
    // for i in 0..bounding_matrix.len() {
    //     for j in 0..bounding_matrix[i].len() {
    //         print!("{}\t", bounding_matrix[i][j]);
    //     }
    //     println!();
    // }
    let mut cost_matrix = vec![vec![0.0; n + 1]; m + 1];
    // Set to inf the first row and column exluding the first element
    for i in 1..n + 1 {
        cost_matrix[0][i] = f64::INFINITY;
    }
    for i in 1..m + 1 {
        cost_matrix[i][0] = f64::INFINITY;
    }

    let delete_addition = nu + lambda;

    for i in 1..n {
        for j in 1..m {
            if bounding_matrix[i][j] {
                // deletion in x1
                let deletion_x1_euclidean_dist = x1[i - 1] - x2[i];
                let del_x1 = cost_matrix[i - 1][j] + deletion_x1_euclidean_dist + delete_addition;

                // deletion in x2
                let deletion_x2_euclidean_dist = x1[j - 1] - x2[j];
                let del_x2 = cost_matrix[i][j - 1] + deletion_x2_euclidean_dist + delete_addition;

                // match
                let match_same_euclid_dist = x1[i] - x2[j];
                let match_previous_euclid_dist = x1[i - 1] - x2[j - 1];

                let match_x1_x2 = cost_matrix[i - 1][j - 1]
                    + match_same_euclid_dist
                    + match_previous_euclid_dist
                    + (nu * (2.0 * (i as isize - j as isize).abs() as f64));

                // Choose the operation with the minimal cost and update DP Matrix
                cost_matrix[i][j] = vec![del_x1, del_x2, match_x1_x2]
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .clone();
            }
        }
    }
    return cost_matrix[n - 1][m - 1];
    
}

pub fn create_bounding_matrix(x1: &[f64], x2: &[f64], sakoe_chiba: f64) -> Vec<Vec<bool>> {
    let n = x1.len();
    let m = x2.len();
    let mut bounding_matrix = vec![vec![false; n]; m];
    let sakoe_chiba_window_radius = ((n as f64 / 100.0) * sakoe_chiba) * 100.0;

    let x_upper_line_values = interp(
        &(0..n)
            .collect::<Vec<usize>>()
            .iter()
            .map(|x| *x as f64)
            .collect::<Vec<_>>(),
        &vec![0.0, (n - 1) as f64],
        &vec![
            0.0 - sakoe_chiba_window_radius,
            (m - 1) as f64 - sakoe_chiba_window_radius,
        ],
    );
    let x_lower_line_values = interp(
        &(0..n)
            .collect::<Vec<usize>>()
            .iter()
            .map(|x| *x as f64)
            .collect::<Vec<_>>(),
        &vec![0.0, (n - 1) as f64],
        &vec![
            0.0 + sakoe_chiba_window_radius,
            (m - 1) as f64 + sakoe_chiba_window_radius,
        ],
    );

    let bounding_matrix = create_shape_on_matrix(
        &mut bounding_matrix,
        x_upper_line_values,
        x_lower_line_values,
    );
    bounding_matrix.clone()
}

fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(x.len());

    for &x_val in x {
        let mut i = 0;
        while i < xp.len() - 1 && x_val > xp[i + 1] {
            i += 1;
        }

        if i == xp.len() - 1 {
            // Extrapolation on the right side
            result.push(fp[i]);
        } else {
            let x0 = xp[i];
            let x1 = xp[i + 1];
            let y0 = fp[i];
            let y1 = fp[i + 1];

            let interpolated_value = y0 + (y1 - y0) * ((x_val - x0) / (x1 - x0));
            result.push(interpolated_value);
        }
    }

    result
}

pub fn create_shape_on_matrix(
    bounding_matrix: &mut Vec<Vec<bool>>,
    y_upper_line: Vec<f64>,
    y_lower_line: Vec<f64>,
) -> &mut Vec<Vec<bool>> {
    let upper_line_y_values = y_upper_line.len();
    let lower_line_y_values = y_lower_line.len();

    //let half_way = upper_line_y_values / 2;
    let step = 1;
    let y_size = bounding_matrix.len();

    for i in 0..upper_line_y_values {
        let x = i * step;
        let upper_y;
        let lower_y;

        // if i > half_way {
        upper_y = f64::max(0.0, f64::min((y_size - 1) as f64, y_upper_line[i].ceil()));
        lower_y = f64::max(0.0, f64::min((y_size - 1) as f64, y_lower_line[i].floor()));
        // } else {
        //     upper_y = f64::max(0.0, f64::min((y_size - 1) as f64, y_upper_line[i].floor()));
        //     lower_y = f64::max(0.0, f64::min((y_size - 1) as f64, y_lower_line[i].ceil()));
        // }

        if upper_line_y_values == lower_line_y_values {
            if upper_y == lower_y {
                bounding_matrix[upper_y as usize][x] = true;
            } else {
                for y in upper_y as usize..(lower_y + 1.0) as usize {
                    bounding_matrix[y][x] = true;
                }
            }
        }
    }
    bounding_matrix
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
