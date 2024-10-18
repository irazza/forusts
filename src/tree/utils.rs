use std::ops::Range;

use rand::{seq::SliceRandom, Rng};

use crate::{utils::structures::Sample, RandomGenerator};

use super::tree::{SplitParameters, StandardSplit};

pub fn get_random_split(
    samples: &mut [Sample],
    non_constant_features: &mut Vec<usize>,
    random_state: &mut RandomGenerator,
    min_samples_leaf: usize,
) -> Option<(Vec<Range<usize>>, StandardSplit, f64)> {
    non_constant_features.shuffle(random_state);

    while let Some(feature) = non_constant_features.pop() {
        let thresholds = samples
            .iter()
            .map(|f| f.features[feature])
            .collect::<Vec<_>>();

        let min_feature = *thresholds
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let max_feature = *thresholds
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        if max_feature - min_feature <= f64::EPSILON {
            // Remove constant features
            continue;
        } else {
            let threshold = random_state.gen_range(min_feature..max_feature);
            let rand_split = StandardSplit { feature, threshold };

            let mut start = 0;
            let mut end = samples.len();
            while start < end {
                if rand_split.split(&samples[start]) == 0 {
                    start += 1;
                } else {
                    samples.swap(start, end - 1);
                    end -= 1;
                }
            }

            if start < min_samples_leaf || (samples.len() - start) < min_samples_leaf {
                continue;
            }

            non_constant_features.push(feature);

            return Some((vec![0..start, start..samples.len()], rand_split, f64::NAN));
        }
    }
    return None;
}
