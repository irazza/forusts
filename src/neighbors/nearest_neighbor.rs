#![allow(dead_code)]
use hashbrown::HashMap;

pub fn k_nearest_neighbor(k: usize, y_train: &Vec<usize>, x_test: &Vec<Vec<f64>>) -> Vec<usize> {
    let mut predictions = Vec::new();

    for i in 0..x_test.len() {
        let mut indices = x_test[i].iter().copied().enumerate().collect::<Vec<_>>();
        indices.sort_unstable_by(|&(_, a), &(_, b)| {
            a.partial_cmp(&b)
                .ok_or_else(|| panic!("{}-{}", a, b))
                .unwrap()
        });
        let mut neighbours = HashMap::new();
        for j in 0..k {
            let index = indices[j].0;
            let label = y_train[index];
            let count = neighbours.entry(label).or_insert(0);
            *count += 1;
        }
        let mut max_count = 0;
        let mut majority_class = 0;
        for (class, count) in &neighbours {
            if *count > max_count {
                max_count = *count;
                majority_class = *class;
            }
        }

        predictions.push(majority_class);
    }
    predictions
}
