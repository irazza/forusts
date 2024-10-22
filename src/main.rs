use std::error::Error;

use rand::SeedableRng;

mod cluster;
mod forest;
mod metrics;
mod neighbors;
mod tests;
mod tree;
mod utils;

type RandomGenerator = rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Hello, world!");
    Ok(())
}
