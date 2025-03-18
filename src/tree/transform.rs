use crate::utils::structures::Sample;
use catch22::compute;
use dashmap::DashMap;
use lazy_static::lazy_static;
use std::sync::Arc;
use sys_info;

lazy_static! {
    pub static ref CACHE: DashMap<(usize, usize, usize, usize), f64> = DashMap::new();
    static ref TOT_RAM: usize = sys_info::mem_info().unwrap().total as usize * 1024;
}

fn cache_size(cache: &DashMap<(usize, usize, usize, usize), f64>) -> usize {
    return 2 * cache.capacity() * (std::mem::size_of::<(usize, usize, usize, usize, f64)>() + 1);
}

pub fn zscore(data: &[f64]) -> Vec<f64> {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
    (0..data.len()).map(|i| (data[i] - mean) / std).collect()
}

pub fn catch_transform(
    data: &[Sample],
    intervals: &[(usize, usize)],
    attributes: &[usize],
) -> Vec<Sample> {
    let mut transformed = Vec::with_capacity(data.len());
    for sample in data {
        let mut features = Vec::with_capacity(intervals.len() * attributes.len());
        for (start, end) in intervals {
            for attribute in attributes {
                let key_cache = (sample.features.as_ptr() as usize, *start, *end, *attribute);
                if let Some(value) = CACHE.get(&key_cache) {
                    features.push(*value);
                } else {
                    let mut value = compute(&sample.features[*start..*end], *attribute);
                    if !value.is_finite() {
                        value = 0.0;
                    }

                    if cache_size(&CACHE) < (*TOT_RAM as f64 * 0.8) as usize {
                        CACHE.insert(key_cache, value);
                    }
                    features.push(value);
                }
            }
        }

        transformed.push(Sample {
            features: Arc::new(features),
            target: sample.target,
        });
    }
    transformed
}
