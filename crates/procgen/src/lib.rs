//! Procedural map generators (wild now; rural and urban later) and the on-disk map cache.
//!
//! Generators are deterministic: the same configuration and seed give a bit-identical map
//! (and [`MapHash`](autonomousim_world::MapHash)) on any machine and with any number of
//! threads. Transcendental functions come from `libm`, parallel stages write disjoint slots,
//! and anything order-dependent runs sequentially.

pub mod cache;
mod hydrology;
pub mod noise;
pub mod rural;
mod scatter;
mod terrain;
pub mod wild;

pub use cache::MapCache;
pub use rural::{RuralConfig, RuralPreset, RuralStats};
pub use scatter::{RocksConfig, TreesConfig};
pub use terrain::{ErosionConfig, TerrainConfig};
pub use wild::{WildConfig, WildPreset, WildStats};

use autonomousim_world::MapFileError;

#[derive(Debug, thiserror::Error)]
pub enum ProcgenError {
    #[error("invalid generator config: {0}")]
    Config(String),
    #[error(transparent)]
    MapFile(#[from] MapFileError),
    #[error("map cache: {0}")]
    Io(#[from] std::io::Error),
}
