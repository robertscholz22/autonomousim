//! Tiled large wild maps: golden hashes of the coarse layer and of sample tiles, tiles that do
//! not depend on access order or threads, seams that agree, queries against one stitched grid,
//! scatter invariants, the bounded tile cache and the coarse-layer cache.
//!
//! Regenerate the golden hashes after an intended change (and bump `TILED_VERSION`) with
//! `AUTONOMOUSIM_BLESS=1 cargo test --release -p autonomousim-procgen --test large`.

use autonomousim_core::geometry::{HitMask, Ray, StaticGeometry};
use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::Terrain;
use autonomousim_procgen::large;
use autonomousim_procgen::wild::{self, WildConfig};
use autonomousim_procgen::{MapCache, TilesConfig};
use autonomousim_world::tiles::{Tile, TiledMap};
use autonomousim_world::{HeightGrid, ObstacleClass, StaticWorld};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// A 2 km tiled map with the large preset's shapes (fast enough for tests).
fn small() -> WildConfig {
    WildConfig { size: 2048.0, ..WildConfig::large() }
}

fn tiles(w: &StaticWorld) -> &Arc<TiledMap> {
    w.terrain().tiled().expect("a tiled map")
}

/// BLAKE3 of everything a tile holds.
fn tile_hash(t: &Tile) -> String {
    let mut h = blake3::Hasher::new();
    for v in t.grid.heights() {
        h.update(&v.to_bits().to_le_bytes());
    }
    h.update(&t.grid.materials().iter().map(|m| m.0).collect::<Vec<_>>());
    if let Some(w) = t.grid.water() {
        for v in w {
            h.update(&v.to_bits().to_le_bytes());
        }
    }
    for id in &t.ids {
        h.update(&id.to_le_bytes());
    }
    h.update(serde_json::to_string(t.obstacles.obstacles()).unwrap().as_bytes());
    h.finalize().to_hex().to_string()
}

#[derive(Serialize, Deserialize)]
struct Golden {
    map: String,
    tiles: Vec<GoldenTile>,
}

#[derive(Clone, Serialize, Deserialize)]
struct GoldenTile {
    tx: u32,
    ty: u32,
    hash: String,
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden_hashes_tiled.toml")
}

const SAMPLE_TILES: [(u32, u32); 4] = [(0, 0), (3, 4), (4, 4), (7, 2)];

#[test]
fn golden_hashes_and_tiles_do_not_depend_on_order_or_threads() {
    let c = small();
    let pool = |n| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    let a = pool(1).install(|| wild::generate(&c, 11)).unwrap().0;
    let b = pool(12).install(|| wild::generate(&c, 11)).unwrap().0;
    assert_eq!(a.content_hash(), b.content_hash());
    // Tiles of `a` in order on this thread; of `b` in reverse order on other threads.
    let ta = tiles(&a);
    let forward: Vec<String> = SAMPLE_TILES.iter().map(|&(x, y)| tile_hash(&ta.tile(x, y))).collect();
    let tb = tiles(&b).clone();
    let backward: Vec<String> = std::thread::scope(|s| {
        SAMPLE_TILES
            .iter()
            .rev()
            .map(|&(x, y)| {
                let tb = tb.clone();
                s.spawn(move || tile_hash(&tb.tile(x, y)))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });
    assert_eq!(forward, backward.into_iter().rev().collect::<Vec<_>>());

    let golden = Golden {
        map: a.content_hash().to_string(),
        tiles: SAMPLE_TILES.iter().zip(&forward).map(|(&(tx, ty), h)| GoldenTile { tx, ty, hash: h.clone() }).collect(),
    };
    if std::env::var_os("AUTONOMOUSIM_BLESS").is_some() {
        std::fs::write(golden_path(), toml::to_string(&golden).unwrap()).unwrap();
        return;
    }
    let stored: Golden = toml::from_str(&std::fs::read_to_string(golden_path()).expect("golden file")).unwrap();
    assert_eq!(golden.map, stored.map, "coarse layer changed (bump TILED_VERSION and bless)");
    for (g, s) in golden.tiles.iter().zip(&stored.tiles) {
        assert_eq!(g.hash, s.hash, "tile ({}, {}) changed (bump TILED_VERSION and bless)", g.tx, g.ty);
    }
}

#[test]
fn neighbouring_tiles_agree_on_their_overlap() {
    let w = wild::generate(&small(), 5).unwrap().0;
    let t = tiles(&w);
    let (nx, _) = t.layout().tiles;
    let mut shared = 0;
    for ty in 2..5u32 {
        for tx in 2..5u32 {
            let a = t.tile(tx, ty);
            for (dx, dy) in [(1i32, 0i32), (0, 1), (1, 1)] {
                let (bx, by) = ((tx as i32 + dx) as u32, (ty as i32 + dy) as u32);
                if bx >= nx {
                    continue;
                }
                let b = t.tile(bx, by);
                // Vertices both grids hold must be identical.
                let (ga, gb) = (&a.grid, &b.grid);
                let (oa, ob) = (ga.origin(), gb.origin());
                let (na, _) = ga.dims();
                for iy in 0..na {
                    for ix in 0..na {
                        let p = oa + DVec2::new(ix as f64, iy as f64);
                        let q = (p - ob).round();
                        let (jx, jy) = (q.x as i64, q.y as i64);
                        if jx < 0 || jy < 0 || jx >= na as i64 || jy >= na as i64 {
                            continue;
                        }
                        assert_eq!(
                            ga.vertex_height(ix, iy).to_bits(),
                            gb.vertex_height(jx as usize, jy as usize).to_bits(),
                            "tiles ({tx},{ty}) / ({bx},{by}) at {p}"
                        );
                        if ix + 1 < na && iy + 1 < na && jx + 1 < na as i64 && jy + 1 < na as i64 {
                            assert_eq!(ga.cell_material(ix, iy), gb.cell_material(jx as usize, jy as usize));
                            assert_eq!(
                                ga.cell_water(ix, iy).map(f64::to_bits),
                                gb.cell_water(jx as usize, jy as usize).map(f64::to_bits)
                            );
                        }
                    }
                }
                // Obstacles stored with both tiles are identical.
                for (i, id) in a.ids.iter().enumerate() {
                    if let Ok(j) = b.ids.binary_search(id) {
                        assert_eq!(a.obstacles.obstacles()[i], b.obstacles.obstacles()[j]);
                        shared += 1;
                    }
                }
            }
        }
    }
    assert!(shared > 50, "{shared} obstacles near seams");
}

#[test]
fn queries_agree_with_one_stitched_grid() {
    let w = wild::generate(&small(), 5).unwrap().0;
    let t = tiles(&w);
    // Tiles (2..6)² stitched into one grid over their cores.
    let (lo, _) = t.layout().core(2, 2);
    let (_, hi) = t.layout().core(5, 5);
    let n = (hi.x - lo.x) as usize + 1;
    let heights: Vec<f32> = (0..n * n)
        .map(|k| {
            let p = lo + DVec2::new((k % n) as f64, (k / n) as f64);
            let tile = t.tile_at(p.x - 0.25, p.y - 0.25);
            let q = p - tile.grid.origin();
            tile.grid.vertex_height(q.x.round() as usize, q.y.round() as usize) as f32
        })
        .collect();
    let grid = HeightGrid::new(lo, 1.0, n, n, heights, vec![MaterialId::GRASS; (n - 1) * (n - 1)]);
    let mut rng = autonomousim_core::rng::Seed::from_u64(9).rng();
    for _ in 0..20_000 {
        // Rays up to 60 m stay inside the stitched grid.
        let (x, y) = (rng.range(lo.x + 70.0, hi.x - 70.0), rng.range(lo.y + 70.0, hi.y - 70.0));
        assert!((w.terrain().height(x, y) - grid.height(x, y)).abs() < 1e-6, "{x} {y}");
        let p = DVec3::new(x, y, grid.height(x, y) + rng.range(-0.5, 3.0));
        let (a, b) = (w.terrain().closest_point(p, 4.0), grid.closest_point(p, 4.0));
        assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            assert!((a.distance - b.distance).abs() < 1e-6);
        }
        let dir = DVec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-0.5, -0.02)).normalize();
        let ray = Ray::new(p + DVec3::Z * 10.0, dir);
        let (a, b) = (w.terrain().raycast(&ray, 60.0, HitMask::TERRAIN), grid.raycast(&ray, 60.0, HitMask::TERRAIN));
        assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            assert!((a.toi - b.toi).abs() < 1e-6, "{} {}", a.toi, b.toi);
        }
    }
}

#[test]
fn scatter_and_water_invariants() {
    let c = small();
    let w = wild::generate(&c, 5).unwrap().0;
    let t = tiles(&w);
    let max_slope = c.trees.max_slope_deg.to_radians().tan();
    let (mut trees, mut rocks, mut lakes) = (0, 0, 0);
    for ty in 0..8 {
        for tx in 0..8 {
            let tile = t.tile(tx, ty);
            let (lo, hi) = t.layout().core(tx, ty);
            for o in tile.obstacles.obstacles() {
                let p = o.pose.pos;
                if p.x < lo.x || p.y < lo.y || p.x >= hi.x || p.y >= hi.y {
                    continue;
                }
                if o.tag == autonomousim_world::obstacles::tags::TRUNK {
                    trees += 1;
                    assert!(w.terrain().water_level(p.x, p.y).is_none(), "tree in water at {p}");
                    let (_, n) = w.terrain().height_normal(p.x, p.y);
                    assert!(n.z > 0.5, "tree on a cliff");
                    // Trunks start at the ground.
                    assert!(o.aabb().0.z < w.terrain().height(p.x, p.y));
                } else if o.tag == autonomousim_world::obstacles::tags::ROCK {
                    rocks += 1;
                }
            }
            if let Some(water) = tile.grid.water() {
                let levels: Vec<f32> = water.iter().copied().filter(|l| !l.is_nan()).collect();
                lakes += levels.len();
            }
            let _ = max_slope;
        }
    }
    assert!(trees > 5_000, "{trees} trees");
    assert!(rocks > 100, "{rocks} rocks");
    assert!(lakes > 0, "some lake water on a 2 km map");
    // No two trunks closer than the spacing (the local rule keeps at most one of a pair).
    let tile = t.tile(3, 3);
    let trunks: Vec<DVec3> = tile
        .obstacles
        .obstacles()
        .iter()
        .filter(|o| o.tag == autonomousim_world::obstacles::tags::TRUNK)
        .map(|o| o.pose.pos)
        .collect();
    for (i, a) in trunks.iter().enumerate() {
        for b in &trunks[i + 1..] {
            assert!(a.truncate().distance(b.truncate()) >= c.trees.min_spacing - 1e-9);
        }
    }
    // Obstacle queries see trees near seams from both sides.
    let solid = HitMask::SOLID;
    let o = tile.obstacles.obstacles().iter().find(|o| o.class == ObstacleClass::Solid).unwrap();
    assert!(w.obstacles().nearest_distance(o.pose.pos, 1.0, solid).is_some());
}

#[test]
fn the_tile_cache_stays_bounded() {
    let mut c = small();
    c.tiles = Some(TilesConfig::default());
    let w = wild::generate(&c, 5).unwrap().0;
    let t = tiles(&w);
    for ty in 0..8 {
        for tx in 0..8 {
            let _ = t.terrain_probe(tx, ty);
        }
    }
    assert!(t.loaded() <= t.capacity());
}

trait Probe {
    fn terrain_probe(&self, tx: u32, ty: u32) -> f64;
}

impl Probe for TiledMap {
    fn terrain_probe(&self, tx: u32, ty: u32) -> f64 {
        let (lo, _) = self.layout().core(tx, ty);
        self.height(lo.x + 1.0, lo.y + 1.0)
    }
}

#[test]
fn the_coarse_layer_is_cached() {
    let dir = std::env::temp_dir().join(format!("autonomousim-tiled-cache-{}", std::process::id()));
    let cache = MapCache::new(&dir);
    let c = WildConfig { size: 1024.0, ..small() };
    let first = cache.wild(&c, 3).unwrap();
    assert!(first.generated.is_some());
    let second = cache.wild(&c, 3).unwrap();
    assert!(second.generated.is_none());
    assert_eq!(first.hash, second.hash);
    let p = DVec2::new(100.0, -200.0);
    assert_eq!(first.world.terrain().height(p.x, p.y), second.world.terrain().height(p.x, p.y));
    std::fs::remove_dir_all(&dir).ok();
}

/// `cargo test --release -p autonomousim-procgen --test large -- --ignored --nocapture`
#[test]
#[ignore = "timing of the full 16 km preset"]
fn large_preset_timing() {
    let c = WildConfig::large();
    let t0 = Instant::now();
    let (w, stats) = wild::generate(&c, 1).unwrap();
    println!("coarse layer {:.2} s: {:?}", t0.elapsed().as_secs_f64(), stats.stages);
    println!("height range {:?}, lakes {}", stats.height_range, stats.lakes);
    let t = tiles(&w);
    let n = 16;
    let t1 = Instant::now();
    let mut bytes = 0;
    let mut obstacles = 0;
    for k in 0..n {
        let tile = t.tile(10 + k, 30);
        bytes += tile.bytes();
        obstacles += tile.obstacles.len();
    }
    println!(
        "tile {:.1} ms, {:.2} MB, {} obstacles (mean of {n})",
        t1.elapsed().as_secs_f64() * 1e3 / n as f64,
        bytes as f64 / n as f64 / 1e6,
        obstacles / n as usize
    );
    let _ = large::DEFAULT_TILE_CACHE;
}
