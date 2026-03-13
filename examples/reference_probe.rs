use forust::forest::forest::{Forest, OutlierForest};
use forust::forest::isolation_forest::{IsolationForest, IsolationForestConfig};
use forust::forest::random_forest::{RandomForest, RandomForestConfig};
use forust::utils::io::read_csv;
use forust::utils::structures::MaxFeatures;
use forust::RandomGenerator;
use rand::SeedableRng;
use serde_json::json;

fn usage() -> ! {
    panic!("usage: cargo run --release --example reference_probe -- <rf|iforest> <train> <test> <tab|comma> <seed>");
}

fn parse_delimiter(value: &str) -> u8 {
    match value {
        "tab" => b'\t',
        "comma" => b',',
        _ => panic!("unsupported delimiter: {value}"),
    }
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 5 {
        usage();
    }

    let mode = &args[0];
    let train_path = &args[1];
    let test_path = &args[2];
    let delimiter = parse_delimiter(&args[3]);
    let seed = args[4].parse::<u64>().expect("invalid seed");

    let mut train = read_csv(train_path, delimiter, false).expect("failed to read train dataset");
    let test = read_csv(test_path, delimiter, false).expect("failed to read test dataset");

    match mode.as_str() {
        "rf" => {
            let config = RandomForestConfig {
                n_trees: 100,
                max_depth: None,
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_features: MaxFeatures::ALL,
                criterion: |_a, _b| f64::NAN,
                aggregation: None,
            };
            let mut model = RandomForest::new(&config);
            model.fit(&mut train, Some(RandomGenerator::seed_from_u64(seed)));
            let predictions = model.predict(&test);
            println!("{}", json!({ "predictions": predictions }));
        }
        "iforest" => {
            let config = IsolationForestConfig {
                n_trees: 100,
                max_depth: None,
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_features: MaxFeatures::ALL,
                criterion: |_a, _b| 1.0,
                aggregation: None,
            };
            let mut model = IsolationForest::new(&config);
            model.fit(&mut train, Some(RandomGenerator::seed_from_u64(seed)));
            let scores = model.score_samples(&test);
            println!("{}", json!({ "scores": scores }));
        }
        _ => usage(),
    }
}
