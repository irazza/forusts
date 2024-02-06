use crate::utils::structures::Sample;

pub fn euclidean(ds_train: &[Sample], ds_test: &[Sample]) -> Vec<Vec<f64>> {
    let mut distance = Vec::new();
    for i in 0..ds_test.len() {
        let mut row = Vec::new();
        for j in 0..ds_train.len() {
            let mut sum = 0.0;
            for k in 0..ds_test[i].data.len() {
                sum += (ds_test[i].data[k] - ds_train[j].data[k]).powi(2);
            }
            row.push(sum.sqrt());
        }
        distance.push(row);
    }
    distance
}
