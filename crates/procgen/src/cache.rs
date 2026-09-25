//! On-disk cache of generated maps, keyed by generator, version, configuration and seed.
//!
//! Entries are [map files](autonomousim_world::mapfile): loading one checks its content hash,
//! and an unreadable or stale entry is regenerated and replaced. Files are written atomically,
//! so several processes can share a cache directory.

use crate::ProcgenError;
use crate::rural::{self, RURAL_VERSION, RuralConfig, RuralStats};
use crate::wild::{self, WILD_VERSION, WildConfig, WildStats};
use autonomousim_world::{MapHash, StaticWorld, mapfile};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct MapCache {
    dir: PathBuf,
}

/// A map from the cache or freshly generated.
pub struct Cached<S> {
    pub world: StaticWorld,
    pub hash: MapHash,
    pub path: PathBuf,
    /// Generator statistics if the map was generated now, `None` if it was loaded.
    pub generated: Option<S>,
}

impl MapCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// `$AUTONOMOUSIM_MAP_CACHE`, else `$XDG_CACHE_HOME/autonomousim/maps`, else
    /// `~/.cache/autonomousim/maps`.
    pub fn default_dir() -> Option<PathBuf> {
        let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        if let Some(d) = var("AUTONOMOUSIM_MAP_CACHE") {
            return Some(d);
        }
        let base = var("XDG_CACHE_HOME").or_else(|| var("HOME").map(|h| h.join(".cache")))?;
        Some(base.join("autonomousim").join("maps"))
    }

    /// The cache in [`default_dir`](Self::default_dir).
    pub fn user() -> Option<Self> {
        Self::default_dir().map(Self::new)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Cache key of a generator run: the BLAKE3 hash (hex) of the generator name, its version,
    /// the configuration as JSON (fields in declaration order, shortest round-trip floats) and
    /// the seed.
    pub fn key(generator: &str, version: u32, config: &impl Serialize, seed: u64) -> String {
        let mut h = blake3::Hasher::new();
        h.update(b"autonomousim map cache v1\0");
        h.update(generator.as_bytes());
        h.update(&[0]);
        h.update(&version.to_le_bytes());
        h.update(serde_json::to_string(config).expect("configs serialise to JSON").as_bytes());
        h.update(&[0]);
        h.update(&seed.to_le_bytes());
        h.finalize().to_hex().to_string()
    }

    pub fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.map"))
    }

    /// Load the entry `key`, or build it with `generate` and store it.
    pub fn load_or_generate<S>(
        &self,
        key: &str,
        generate: impl FnOnce() -> Result<(StaticWorld, S), ProcgenError>,
    ) -> Result<Cached<S>, ProcgenError> {
        let path = self.path(key);
        if path.exists()
            && let Ok((world, hash)) = mapfile::load(&path)
        {
            return Ok(Cached { world, hash, path, generated: None });
        }
        let (world, stats) = generate()?;
        std::fs::create_dir_all(&self.dir)?;
        let hash = mapfile::save(&world, &path)?;
        Ok(Cached { world, hash, path, generated: Some(stats) })
    }

    /// A wild map from the cache, generated on a miss.
    pub fn wild(&self, config: &WildConfig, seed: u64) -> Result<Cached<WildStats>, ProcgenError> {
        config.validate()?;
        let key = Self::key("wild", WILD_VERSION, config, seed);
        self.load_or_generate(&key, || wild::generate(config, seed))
    }

    /// A rural map from the cache, generated on a miss.
    pub fn rural(&self, config: &RuralConfig, seed: u64) -> Result<Cached<RuralStats>, ProcgenError> {
        config.validate()?;
        let key = Self::key("rural", RURAL_VERSION, config, seed);
        self.load_or_generate(&key, || rural::generate(config, seed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_separate_configs_and_seeds() {
        let a = WildConfig::training();
        let mut b = a.clone();
        b.trees.min_spacing += 0.5;
        let k = |c: &WildConfig, s| MapCache::key("wild", WILD_VERSION, c, s);
        assert_eq!(k(&a, 1), k(&a.clone(), 1));
        assert_ne!(k(&a, 1), k(&a, 2));
        assert_ne!(k(&a, 1), k(&b, 1));
        assert_ne!(k(&a, 1), MapCache::key("wild", WILD_VERSION + 1, &a, 1));
        assert_eq!(k(&a, 1).len(), 64);
    }
}
