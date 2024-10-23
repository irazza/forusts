use crate::utils::structures::Sample;
use catch22::compute;
use dashmap::DashMap;
use lazy_static::lazy_static;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

lazy_static! {
    pub static ref CACHE: DashMap<(usize, usize, usize, usize), f64> = DashMap::new();
}

pub fn catch_transform(
    data: &[Sample],
    intervals: &[(usize, usize)],
    attributes: &[usize],
) -> Vec<Sample> {
    // if CACHE.len() > 100_000 {
    //     CACHE.clear();
    // }
    let mut transformed = Vec::with_capacity(data.len());
    static COUNTER_GET: AtomicUsize = AtomicUsize::new(0);
    for sample in data {
        let mut features = Vec::with_capacity(intervals.len() * attributes.len());
        for (start, end) in intervals {
            for attribute in attributes {
                // static COUNTER: AtomicUsize = AtomicUsize::new(0);
                // if COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 1000 == 0 {
                //     println!("COUNTER: {}", COUNTER.load(std::sync::atomic::Ordering::Relaxed));
                //     println!("COUNTER GET: {}", COUNTER_GET.load(std::sync::atomic::Ordering::Relaxed));
                //
                // }

                let key_cache = (sample.features.as_ptr() as usize, *start, *end, *attribute);
                if let Some(value) = CACHE.get(&key_cache) {
                    // if COUNTER_GET.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 1000 == 0 {
                    //     println!("COUNTER GET: {}", COUNTER_GET.load(std::sync::atomic::Ordering::Relaxed));
                    // }
                    features.push(*value);
                } else {
                    let value = compute(&sample.features[*start..*end], *attribute);
                    CACHE.insert(key_cache, value);
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
