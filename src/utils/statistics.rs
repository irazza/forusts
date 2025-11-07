use hashbrown::HashSet;
use std::hash::Hash;

pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

pub fn variance(v: &[f64]) -> f64 {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64
}

pub fn class_counts<T: Hash + Eq>(arr: &[T]) -> usize {
    let mut count = HashSet::new();
    for x in arr {
        count.insert(x);
    }
    return count.len();
}

pub fn argsort<T: PartialOrd>(v: &[T]) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

pub fn unique<T: PartialOrd + Clone>(x: &[T]) -> Vec<T> {
    let mut unique = x.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique.dedup();
    unique
}

pub fn quantile(v: &[f64], p: f64) -> (f64, Vec<f64>) {
    if p == 1.0 {
        return (*v.last().unwrap(), v[..v.len() - 1].to_vec());
    }
    let mut v = v.to_vec();
    let n = v.len();
    let k = (n as f64 * p).round() as usize;
    v.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
    (v[k], v[..k].to_vec())
}

pub fn quartiles(v: &[f64], q: i32) -> (f64, Vec<f64>) {
    match q {
        1 => {
            let (q1, v) = quantile(v, 0.25);
            (q1, v)
        }
        2 => {
            let k = (v.len() as f64 * 0.25).round() as usize;
            let (q2, v) = quantile(v, 0.5);
            (q2, v[k..].to_vec())
        }
        3 => {
            let k = (v.len() as f64 * 0.5).round() as usize;
            let (q3, v) = quantile(v, 0.75);
            (q3, v[k..].to_vec())
        }
        4 => {
            let (q4, v) = quantile(v, 0.75);
            (q4, v)
        }
        _ => panic!("Invalid quartile"),
    }
}
