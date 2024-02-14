// use find_peaks::PeakFinder;

// use super::statistics::max;

// pub fn compute_mep_all(x: &[f64]) -> Vec<f64> {
//     let mut features = Vec::new();

//     let mut pf = PeakFinder::new(x);
//     pf.with_min_prominence(0.001);

//     // Sort peaks by start position
//     let mut peaks = pf.find_peaks();

//     if peaks.len() <= 4 {
//         return vec![0.0; 8];
//     }

//     peaks.sort_by(|a, b| a.position.start.partial_cmp(&b.position.start).unwrap());

//     // First 4 peaks are ignored, because ...
//     // let first_peak = if peaks.len() < 4 {
//     //         0
//     //     } else {
//     //         4
//     //     };

//     let peaks = peaks[4..].to_vec();

//     // Amplitude: Max amplitude
//     features.push(max(&peaks
//         .iter()
//         .map(|p| p.height.unwrap_or(0.0))
//         .collect::<Vec<_>>()));

//     // Area: Total area within duration
//     features.push(x.iter().map(|v| f64::abs(*v)).sum::<f64>());

//     // Duration: length of the signal response
//     features.push((peaks[peaks.len() - 1].position.end - peaks[0].position.start) as f64);

//     // Thickness: Area / Amplitude
//     features.push(features[1] / features[0]);

//     // Size index: Normalized thickness, sometimes the value of amplitude is too low that the log10 is asyntotic
//     if features[0].log10().is_finite() {
//         features.push(features[2] / features[0].log10());
//     } else {
//         features.push(features[3]);
//     }

//     // No. of phases: zero-crossing + 1
//     let mut zc = 0;
//     for i in 0..x.len() - 1 {
//         if x[i] * x[i + 1] < 0.0 {
//             zc += 1;
//         }
//     }
//     features.push(zc as f64 + 1.0);

//     // No. of turns: number of direction changes
//     features.push(peaks.len() as f64 + 4.0);

//     // No.of spikes: number of peaks within the signal response
//     features.push(peaks.len() as f64);
//     features
// }
