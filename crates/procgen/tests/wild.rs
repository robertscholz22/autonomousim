//! Wild generator: golden hashes (independent of the thread count), map invariants, cache.
//!
//! Regenerate the golden hashes after an intended change of the generator output (and bump
//! `WILD_VERSION`) with
//! `AUTONOMOUSIM_BLESS=1 cargo test -p autonomousim-procgen --test wild`.

use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::Terrain;
use autonomousim_procgen::wild::{self, WILD_VERSION, WildConfig, WildPreset};
use autonomousim_procgen::{MapCache, ProcgenError};
use autonomousim_world::obstacles::tags;
use autonomousim_world::{ObstacleShape, StaticWorld};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
struct Golden {
    wild: Vec<GoldenMap>,
}

#[derive(Clone, Serialize, Deserialize)]
struct GoldenMap {
    preset: WildPreset,
    size: f64,
    seed: u64,
    hash: String,
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden_hashes.toml")
}

fn config(preset: WildPreset, size: f64) -> WildConfig {
    WildConfig { size, ..preset.config() }
}

fn generate_with_threads(c: &WildConfig, seed: u64, threads: usize) -> StaticWorld {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| wild::generate(c, seed)).unwrap().0
}

#[test]
fn golden_hashes_do_not_depend_on_the_thread_count() {
    let text = std::fs::read_to_string(golden_path()).expect("fixtures/golden_hashes.toml");
    let golden: Golden = toml::from_str(&text).unwrap();
    let bless = std::env::var_os("AUTONOMOUSIM_BLESS").is_some();
    let mut blessed = Vec::new();
    for g in &golden.wild {
        let c = config(g.preset, g.size);
        let one = generate_with_threads(&c, g.seed, 1).content_hash().to_string();
        let many = generate_with_threads(&c, g.seed, 12).content_hash().to_string();
        assert_eq!(one, many, "{:?} {} m seed {}: 1 vs 12 threads", g.preset, g.size, g.seed);
        if bless {
            blessed.push(GoldenMap { hash: one, ..g.clone() });
        } else {
            assert_eq!(
                one, g.hash,
                "{:?} {} m seed {} changed (bump WILD_VERSION and bless)",
                g.preset, g.size, g.seed
            );
        }
    }
    if bless {
        let header = "# Content hashes of generated maps (see crates/procgen/tests/wild.rs).\n\
                      # Regenerate: AUTONOMOUSIM_BLESS=1 cargo test -p autonomousim-procgen --test wild\n\n";
        let body = toml::to_string(&Golden { wild: blessed }).unwrap();
        std::fs::write(golden_path(), format!("{header}{body}")).unwrap();
    }
}

#[test]
fn wild_map_invariants() {
    let c = config(WildPreset::Training, 512.0);
    let (w, stats) = wild::generate(&c, 3).unwrap();
    assert_eq!((w.meta.generator.as_str(), w.meta.generator_version, w.meta.seed), ("wild", WILD_VERSION, 3));
    assert!(stats.trees > 1000 && stats.rocks > 50 && stats.lakes > 0, "{stats:?}");
    let t = w.terrain();
    assert_eq!(t.dims(), (513, 513));
    let (cw, ch) = t.cells();
    let cell_of = |x: f64, y: f64| {
        let q = (glam::DVec2::new(x, y) - t.origin()) / t.cell_size();
        ((q.x as usize).min(cw - 1), (q.y as usize).min(ch - 1))
    };

    // Lakes: every wet cell has a corner below its level, lakebeds are sand or mud, and two
    // neighbouring wet cells that share a submerged corner have the same level.
    let water = t.water().expect("lakes");
    for cy in 0..ch {
        for cx in 0..cw {
            let Some(level) = t.cell_water(cx, cy) else { continue };
            let corners = [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)];
            assert!(corners.iter().any(|&(x, y)| t.vertex_height(x, y) < level), "dry cell {cx} {cy}");
            assert!(matches!(t.cell_material(cx, cy), MaterialId::SAND | MaterialId::MUD));
            if cx + 1 < cw
                && let Some(right) = t.cell_water(cx + 1, cy)
            {
                let shared = [(cx + 1, cy), (cx + 1, cy + 1)];
                if shared.iter().any(|&(x, y)| t.vertex_height(x, y) < level.min(right)) {
                    assert_eq!(level, right, "lake level jumps at {cx} {cy}");
                }
            }
        }
    }
    assert_eq!(water.iter().filter(|v| !v.is_nan()).count(), stats.lake_cells);

    // Trees stand on dry, moderately sloped ground, sunk slightly into it; rocks stay dry.
    let max_slope = c.trees.max_slope_deg.to_radians().tan();
    let mut trunks = 0;
    for o in w.obstacles().obstacles() {
        let p = o.pose.pos;
        let (cx, cy) = cell_of(p.x, p.y);
        match o.tag {
            tags::TRUNK => {
                trunks += 1;
                assert!(t.cell_water(cx, cy).is_none(), "tree in a lake at {p}");
                let ObstacleShape::Capsule { half_height, radius } = o.shape else { panic!("trunk shape") };
                let ground = t.height(p.x, p.y);
                let bottom = p.z - half_height - radius;
                assert!(bottom < ground && bottom > ground - 2.5, "floating or buried trunk at {p}");
                let d = 1.5;
                let gx = (t.height(p.x + d, p.y) - t.height(p.x - d, p.y)) / (2.0 * d);
                let gy = (t.height(p.x, p.y + d) - t.height(p.x, p.y - d)) / (2.0 * d);
                assert!((gx * gx + gy * gy).sqrt() < 1.3 * max_slope, "tree on a cliff at {p}");
            }
            tags::ROCK => assert!(t.cell_water(cx, cy).is_none(), "rock in a lake at {p}"),
            tags::CANOPY | tags::CANOPY_BROADLEAF => {}
            tag => panic!("unexpected tag {tag}"),
        }
    }
    assert_eq!(trunks, stats.trees);
}

#[test]
fn config_round_trips_and_is_validated() {
    for c in [WildConfig::training(), WildConfig::showcase()] {
        let text = toml::to_string(&c).unwrap();
        assert_eq!(toml::from_str::<WildConfig>(&text).unwrap(), c);
    }
    // Missing fields come from the training preset; unknown ones are rejected.
    let c: WildConfig = toml::from_str("size = 256.0\n[trees]\nmin_spacing = 4.0").unwrap();
    assert_eq!((c.size, c.trees.min_spacing, c.terrain.relief), (256.0, 4.0, WildConfig::training().terrain.relief));
    assert!(toml::from_str::<WildConfig>("sise = 256.0").is_err());
    for bad in [
        WildConfig { size: 255.0, ..WildConfig::training() },
        WildConfig { size: 16.0, ..WildConfig::training() },
        WildConfig { cell: 0.0, ..WildConfig::training() },
    ] {
        assert!(matches!(wild::generate(&bad, 0), Err(ProcgenError::Config(_))));
    }
}

#[test]
fn cache_stores_and_reloads_maps() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("procgen-cache");
    let _ = std::fs::remove_dir_all(&dir);
    let cache = MapCache::new(&dir);
    let c = config(WildPreset::Training, 128.0);
    let first = cache.wild(&c, 5).unwrap();
    assert!(first.generated.is_some() && first.path.exists());
    assert_eq!(first.hash, first.world.content_hash());
    let second = cache.wild(&c, 5).unwrap();
    assert!(second.generated.is_none(), "second request must hit the cache");
    assert_eq!(second.hash, first.hash);
    assert_eq!(second.world.obstacles().len(), first.world.obstacles().len());
    // Another seed is another entry.
    assert_ne!(cache.wild(&c, 6).unwrap().path, first.path);
    // A damaged entry is regenerated.
    let bytes = std::fs::read(&first.path).unwrap();
    std::fs::write(&first.path, &bytes[..bytes.len() / 2]).unwrap();
    let third = cache.wild(&c, 5).unwrap();
    assert!(third.generated.is_some());
    assert_eq!(third.hash, first.hash);
    std::fs::remove_dir_all(&dir).unwrap();
}
