pub mod cluster;
pub mod forest;
pub mod metrics;
pub mod neighbors;
pub mod tests;
pub mod tree;
pub mod utils;

pub type RandomGenerator = rand_chacha::ChaCha8Rng;
pub(crate) const DEFAULT_RANDOM_SEED: u64 = 0;

#[inline]
pub(crate) fn default_random_generator() -> RandomGenerator {
    rand::SeedableRng::seed_from_u64(DEFAULT_RANDOM_SEED)
}
