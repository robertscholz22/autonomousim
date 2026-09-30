//! Tiled large wild maps (10–20 km) for aircraft.
//!
//! The wild pipeline runs on a **coarse layer** that covers the whole map (e.g. 16 km at
//! 8 m): noise terrain, erosion on cells twice as large, Priority-Flood lakes and drainage
//! moisture ([`WildConfig`] with [`TilesConfig`]). The final grid is never built as a whole:
//! each **tile** (e.g. 256 m at 1 m) is a pure function of the coarse layer and global
//! coordinates:
//!
//! - heights: Catmull–Rom interpolation of the coarse heights plus a band-filling noise
//!   (`mid_amplitude`, `mid_wavelength`) and the fine detail noise;
//! - water: the level of a lake in the coarse cell or its neighbours wherever the fine cell
//!   lies below it (lakes stay flat, shores follow the fine terrain);
//! - materials: the wild rules on the fine cells with moisture from the coarse layer;
//! - trees and rocks: candidates drawn per 16 m scatter cell from that cell's own seed, a
//!   tree kept when no candidate within the minimum spacing has a higher priority (a local
//!   rule, so the same trees survive whichever tile decides), rocks kept clear of trees.
//!
//! Tiles are computed over their core plus a margin wide enough for every decision that
//! reaches into the core, so neighbouring tiles agree exactly on their seams. Generation of a
//! tile is sequential (it may run inside rayon workers of a batch step).

use crate::ProcgenError;
use crate::hydrology::Lakes;
use crate::noise::{Fractal, fbm};
use crate::scatter::{EDGE_MARGIN, broadleaf, rock_shape};
use crate::terrain::{TerrainConfig, TerrainSeeds, detail_height};
use crate::wild::{
    Land, LineSeeds, MaterialRule, TilesConfig, WILD_VERSION, WildConfig, WildStats, landform, moisture_scaled,
    terrain_seeds,
};
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::rng::Seed;
use autonomousim_world::testworlds::tree as conifer;
use autonomousim_world::tiles::{Tile, TileLayout, TileSource, TiledMap};
use autonomousim_world::{HeightGrid, MapHash, MapMeta, Obstacle, ObstacleSet, StaticWorld};
use glam::DVec2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// Bumped whenever the tiles for a given configuration and seed change.
pub const TILED_VERSION: u32 = 1;

/// Edge of a scatter cell (m): candidates are drawn per cell from the cell's own seed.
pub const SCATTER_CELL: f64 = 16.0;

/// Terrain stored around each tile core (final cells): the closest-point search radius.
const RING_CELLS: usize = autonomousim_world::heightgrid::MAX_SEARCH_CELLS as usize;

/// Obstacles are stored with every tile whose core grown by this much (m) they overlap.
const REACH: f64 = 4.0;

/// Rays use the tiles up to this distance and the coarse layer beyond it (m): about one tile,
/// so an aircraft's long-range LiDAR keeps only the few tiles around it in use; farther
/// terrain comes from the coarse layer (8 m) and misses trees and rocks.
const DETAIL_RANGE: f64 = 300.0;

/// Default number of tiles kept in memory (about 2.5 MB each with forest); override
/// with `$AUTONOMOUSIM_TILE_CACHE`.
pub const DEFAULT_TILE_CACHE: usize = 128;

const MID_FRACTAL: Fractal = Fractal { octaves: 3, lacunarity: 2.0, gain: 0.5 };

/// The coarse layer: everything a tile needs besides the configuration and the seed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Coarse {
    /// Vertices per edge.
    pub n: usize,
    /// Vertex heights (`n × n`).
    pub heights: Vec<f32>,
    /// Lake level per cell (`(n−1)²`, NaN = dry).
    pub water: Vec<f32>,
    /// Moisture per vertex in [0, 1].
    pub moisture: Vec<f32>,
    pub droplets: u64,
    pub lakes: usize,
    pub lake_cells: usize,
}

/// Terrain with the fine detail removed: the coarse layer carries only the large shapes.
fn coarse_terrain(t: &TerrainConfig) -> TerrainConfig {
    TerrainConfig { detail_amplitude: 0.0, ..t.clone() }
}

/// Build the coarse layer (the expensive, whole-map part).
pub fn coarse(config: &WildConfig, seed: u64, stats: &mut WildStats) -> Result<Coarse, ProcgenError> {
    config.validate()?;
    let tiles = config.tiles.as_ref().ok_or_else(|| ProcgenError::Config("not a tiled map".into()))?;
    let mut t = Instant::now();
    let root = Seed::from_u64(seed).child("map/wild");
    let terrain = coarse_terrain(&config.terrain);
    let land = landform(
        &Land {
            size: config.size,
            cell: tiles.coarse_cell,
            terrain: &terrain,
            erosion: &config.erosion,
            water: &config.water,
            shape: None,
        },
        &root,
        &mut |name| stats.stage(name, &mut t),
    );
    let n = (config.size / tiles.coarse_cell).round() as usize + 1;
    let area = (tiles.coarse_cell / config.cell).powi(2);
    let moisture = moisture_scaled(&land.accumulation, n, area);
    stats.stage("moisture", &mut t);
    let Lakes { water, count, cells } = land.lakes;
    Ok(Coarse { n, heights: land.heights, water, moisture, droplets: land.droplets, lakes: count, lake_cells: cells })
}

/// Content hash of a tiled wild map: the generator inputs and the coarse layer (tiles are
/// functions of these).
pub fn map_hash(config: &WildConfig, seed: u64, coarse: &Coarse) -> MapHash {
    let mut h = blake3::Hasher::new();
    h.update(b"autonomousim tiled wild map v1\0");
    h.update(&WILD_VERSION.to_le_bytes());
    h.update(&TILED_VERSION.to_le_bytes());
    h.update(serde_json::to_string(config).expect("configs serialise to JSON").as_bytes());
    h.update(&[0]);
    h.update(&seed.to_le_bytes());
    for v in [&coarse.heights, &coarse.water, &coarse.moisture] {
        for x in v.iter() {
            h.update(&x.to_bits().to_le_bytes());
        }
    }
    MapHash(*h.finalize().as_bytes())
}

/// The coarse layer as bytes (zstd-compressed postcard after a magic and a BLAKE3 checksum).
pub fn encode_coarse(c: &Coarse) -> Vec<u8> {
    let body = zstd::encode_all(&postcard::to_stdvec(c).expect("coarse layers serialise")[..], 3)
        .expect("in-memory compression cannot fail");
    let mut out = Vec::with_capacity(body.len() + 40);
    out.extend_from_slice(COARSE_MAGIC);
    out.extend_from_slice(blake3::hash(&body).as_bytes());
    out.extend_from_slice(&body);
    out
}

/// A coarse layer from [`encode_coarse`] bytes, or `None` if they are damaged.
pub fn decode_coarse(bytes: &[u8]) -> Option<Coarse> {
    let rest = bytes.strip_prefix(COARSE_MAGIC.as_slice())?;
    let (sum, body) = rest.split_at_checked(32)?;
    if blake3::hash(body).as_bytes() != sum {
        return None;
    }
    let c: Coarse = postcard::from_bytes(&zstd::decode_all(body).ok()?).ok()?;
    let valid = c.n >= 2
        && c.heights.len() == c.n * c.n
        && c.moisture.len() == c.n * c.n
        && c.water.len() == (c.n - 1) * (c.n - 1);
    valid.then_some(c)
}

const COARSE_MAGIC: &[u8; 8] = b"AUTOSIMC";

/// Generate a tiled wild map (the coarse layer now, tiles on demand).
pub fn generate(config: &WildConfig, seed: u64) -> Result<(StaticWorld, WildStats), ProcgenError> {
    let mut stats = WildStats::default();
    let coarse = coarse(config, seed, &mut stats)?;
    let world = assemble(config, seed, coarse, &mut stats)?;
    Ok((world, stats))
}

/// The map from its coarse layer (cheap: coarse materials and bounds).
pub fn assemble(
    config: &WildConfig,
    seed: u64,
    coarse: Coarse,
    stats: &mut WildStats,
) -> Result<StaticWorld, ProcgenError> {
    config.validate()?;
    let mut t = Instant::now();
    let hash = map_hash(config, seed, &coarse);
    let source = WildTiles::new(config, seed, coarse);
    stats.vertices = source.n * source.n;
    stats.droplets = source.coarse.droplets;
    stats.lakes = source.coarse.lakes;
    stats.lake_cells = source.coarse.lake_cells;
    let grid = source.coarse_grid();
    stats.height_range = grid.height_range();
    let pad = source.pad();
    stats.stage("coarse materials", &mut t);
    let capacity =
        std::env::var("AUTONOMOUSIM_TILE_CACHE").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_TILE_CACHE);
    let layout = source.layout;
    let map = TiledMap::new(layout, grid, pad, Box::new(source), capacity);
    let mut meta = MapMeta::new("wild", "wild", seed);
    meta.generator_version = WILD_VERSION;
    meta.geo_origin = config.geo_origin;
    Ok(StaticWorld::tiled(meta, Arc::new(map), MaterialTable::standard(), hash))
}

/// Catmull–Rom weights at fraction `t` of the interval between the middle samples.
fn catmull_rom(t: f64) -> [f64; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [0.5 * (-t3 + 2.0 * t2 - t), 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0), 0.5 * (-3.0 * t3 + 4.0 * t2 + t), 0.5 * (t3 - t2)]
}

/// A tree or rock candidate that passed the ground rules.
struct TreeCandidate {
    p: DVec2,
    key: u32,
    priority: f64,
    height: f64,
    conifer: bool,
    slope: f64,
}

/// Tiles of a wild map.
pub struct WildTiles {
    c: WildConfig,
    t: TilesConfig,
    coarse: Coarse,
    /// Coarse vertices per edge.
    n: usize,
    /// Coarse cell / final cell.
    ratio: i64,
    weights: Vec<[f64; 4]>,
    origin: DVec2,
    ts: TerrainSeeds,
    mid_seed: u64,
    lines: LineSeeds,
    tree_seed: Seed,
    rock_seed: Seed,
    layout: TileLayout,
    /// Final cells computed around each core (for decisions that reach into it).
    margin_cells: i64,
    /// Candidates are considered this far (m) around a core.
    scatter_margin: f64,
    /// Scatter cells along x.
    scatter_nx: u32,
}

impl WildTiles {
    fn new(config: &WildConfig, seed: u64, coarse: Coarse) -> Self {
        let t = config.tiles.clone().expect("a tiled configuration");
        let root = Seed::from_u64(seed).child("map/wild");
        let ratio = (t.coarse_cell / config.cell).round() as i64;
        let tile_size = t.tile_cells as f64 * config.cell;
        let tiles = (config.size / tile_size).round() as u32;
        let origin = DVec2::splat(-0.5 * config.size);
        // How far obstacles reach from their anchors: crowns (0.3 of the tree height) and
        // rock hulls (up to 0.6 of the size).
        let reach = (0.3 * config.trees.max_height).max(0.6 * config.rocks.max_size);
        let rock_clearance = 1.1 * config.rocks.max_size + 0.5;
        let scatter_margin = REACH + reach.max(rock_clearance) + config.trees.min_spacing;
        // Slopes need 2 cells, shores `shore_width`, the stored ring RING_CELLS.
        let shore = (config.materials.shore_width / config.cell).ceil() as i64;
        let margin_cells = ((scatter_margin / config.cell).ceil() as i64).max(RING_CELLS as i64) + 2 + shore.max(0) + 1;
        Self {
            c: config.clone(),
            n: coarse.n,
            coarse,
            ratio,
            weights: (0..ratio).map(|k| catmull_rom(k as f64 / ratio as f64)).collect(),
            origin,
            ts: terrain_seeds(&root),
            mid_seed: root.child("terrain").child("mid").short(),
            lines: LineSeeds::new(&root),
            tree_seed: root.child("tiles").child("trees"),
            rock_seed: root.child("tiles").child("rocks"),
            layout: TileLayout {
                origin,
                tile_size,
                tiles: (tiles, tiles),
                ring: RING_CELLS as f64 * config.cell,
                reach: REACH,
                detail_range: DETAIL_RANGE,
            },
            margin_cells,
            scatter_margin,
            scatter_nx: (config.size / SCATTER_CELL).round() as u32,
            t,
        }
    }

    #[inline]
    fn coarse_at(&self, ix: i64, iy: i64) -> f64 {
        let m = self.n as i64 - 1;
        self.coarse.heights[(iy.clamp(0, m) as usize) * self.n + ix.clamp(0, m) as usize] as f64
    }

    /// Height (m) of final vertex `(gx, gy)` (global indices from the map's corner).
    fn vertex_height(&self, gx: i64, gy: i64) -> f64 {
        let (ix, iy) = (gx.div_euclid(self.ratio), gy.div_euclid(self.ratio));
        let wx = &self.weights[gx.rem_euclid(self.ratio) as usize];
        let wy = &self.weights[gy.rem_euclid(self.ratio) as usize];
        let mut h = 0.0;
        for (a, wa) in wy.iter().enumerate() {
            let yy = iy + a as i64 - 1;
            let mut row = 0.0;
            for (b, wb) in wx.iter().enumerate() {
                row += wb * self.coarse_at(ix + b as i64 - 1, yy);
            }
            h += wa * row;
        }
        let (x, y) = (self.origin.x + gx as f64 * self.c.cell, self.origin.y + gy as f64 * self.c.cell);
        let mid = if self.t.mid_amplitude > 0.0 {
            self.t.mid_amplitude
                * fbm(self.mid_seed, x / self.t.mid_wavelength, y / self.t.mid_wavelength, &MID_FRACTAL)
        } else {
            0.0
        };
        h + mid + detail_height(&self.c.terrain, &self.ts, x, y)
    }

    /// Moisture at final vertex `(gx, gy)`: bilinear in the coarse layer.
    fn vertex_moisture(&self, gx: i64, gy: i64) -> f32 {
        let r = self.ratio;
        let (ix, iy) = (gx.div_euclid(r), gy.div_euclid(r));
        let (fx, fy) = (gx.rem_euclid(r) as f32 / r as f32, gy.rem_euclid(r) as f32 / r as f32);
        let m = self.n as i64 - 1;
        let at = |x: i64, y: i64| self.coarse.moisture[(y.clamp(0, m) as usize) * self.n + x.clamp(0, m) as usize];
        let a = at(ix, iy) + fx * (at(ix + 1, iy) - at(ix, iy));
        let b = at(ix, iy + 1) + fx * (at(ix + 1, iy + 1) - at(ix, iy + 1));
        a + fy * (b - a)
    }

    /// Highest lake level in coarse cell `(cx, cy)` and its eight neighbours (−∞ if none).
    fn lake_level_near(&self, cx: i64, cy: i64) -> f32 {
        let m = self.n as i64 - 2;
        let mut level = f32::NEG_INFINITY;
        for y in (cy - 1).max(0)..=(cy + 1).min(m) {
            for x in (cx - 1).max(0)..=(cx + 1).min(m) {
                let w = self.coarse.water[y as usize * (self.n - 1) + x as usize];
                if !w.is_nan() {
                    level = level.max(w);
                }
            }
        }
        level
    }

    /// Ground height at `p` on the final triangulation (from global vertices, so every tile
    /// computes the same value).
    fn surface(&self, p: DVec2) -> f64 {
        let cell = self.c.cell;
        let (u, v) = ((p.x - self.origin.x) / cell, (p.y - self.origin.y) / cell);
        let (gx, gy) = (u.floor() as i64, v.floor() as i64);
        let (fx, fy) = (u - gx as f64, v - gy as f64);
        let h00 = self.vertex_height(gx, gy) as f32 as f64;
        let h11 = self.vertex_height(gx + 1, gy + 1) as f32 as f64;
        if fx >= fy {
            let h10 = self.vertex_height(gx + 1, gy) as f32 as f64;
            h00 + fx * (h10 - h00) + fy * (h11 - h10)
        } else {
            let h01 = self.vertex_height(gx, gy + 1) as f32 as f64;
            h00 + fx * (h11 - h01) + fy * (h01 - h00)
        }
    }

    /// The coarse layer as a grid with materials (for far queries and distant rendering).
    fn coarse_grid(&self) -> HeightGrid {
        let n = self.n;
        let cw = n - 1;
        let cell = self.t.coarse_cell;
        let h = &self.coarse.heights;
        let rule = MaterialRule::new(&self.c.materials, &self.lines, self.c.terrain.relief);
        let treeline = |x: f64, y: f64| self.lines.treeline(&self.c.materials, self.c.terrain.relief, x, y);
        let materials: Vec<MaterialId> = (0..cw * cw)
            .into_par_iter()
            .map(|k| {
                let (cx, cy) = (k % cw, k / cw);
                let i = cy * n + cx;
                let v = |x: usize, y: usize| h[y.min(n - 1) * n + x.min(n - 1)] as f64;
                let gx = (v(cx + 1, cy) + v(cx + 1, cy + 1) - v(cx, cy) - v(cx, cy + 1)) / (2.0 * cell);
                let gy = (v(cx, cy + 1) + v(cx + 1, cy + 1) - v(cx, cy) - v(cx + 1, cy)) / (2.0 * cell);
                let z = 0.25 * (h[i] as f64 + h[i + 1] as f64 + h[i + n] as f64 + h[i + n + 1] as f64);
                let m = &self.coarse.moisture;
                let wet = m[i].max(m[i + 1]).max(m[i + n]).max(m[i + n + 1]) as f64;
                let (x, y) = (self.origin.x + (cx as f64 + 0.5) * cell, self.origin.y + (cy as f64 + 0.5) * cell);
                rule.material(x, y, z, gx.hypot(gy), self.coarse.water[k], None, wet, &treeline)
            })
            .collect();
        HeightGrid::new(self.origin, cell, n, n, h.clone(), materials).with_water(self.coarse.water.clone())
    }

    /// Bound (m) on how far tile terrain can leave the coarse grid's bounds: Catmull–Rom
    /// overshoot (at most 9/16 of the range of a 4×4 stencil) plus the noise amplitudes.
    fn pad(&self) -> f64 {
        let n = self.n;
        let h = &self.coarse.heights;
        let range = (0..n)
            .into_par_iter()
            .map(|y| {
                let mut worst = 0.0f32;
                for x in 0..n {
                    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                    for yy in y.saturating_sub(1)..=(y + 2).min(n - 1) {
                        for xx in x.saturating_sub(1)..=(x + 2).min(n - 1) {
                            lo = lo.min(h[yy * n + xx]);
                            hi = hi.max(h[yy * n + xx]);
                        }
                    }
                    worst = worst.max(hi - lo);
                }
                worst
            })
            .reduce(|| 0.0, f32::max) as f64;
        // Simplex fBm stays within ±1 by construction; 1.25 leaves headroom.
        0.5625 * range + 1.25 * (self.t.mid_amplitude + self.c.terrain.detail_amplitude) + 0.01
    }

    /// Global id of part `part` (0, 1) of candidate `k` in scatter cell `cell`, `rock` for rocks.
    #[inline]
    fn id(cell: u32, k: usize, part: u32, rock: bool) -> u32 {
        (cell << 8) | ((k as u32) << 2) | (part << 1) | rock as u32
    }
}

/// Per-tile arrays over the core plus the margin, indexed from the region's first vertex.
struct Region {
    /// First global vertex.
    gx0: i64,
    gy0: i64,
    /// Vertices per edge (cells per edge + 1).
    m: usize,
    heights: Vec<f32>,
    /// Per cell: water level (NaN = dry), slope tangent, material.
    water: Vec<f32>,
    slope: Vec<f32>,
    materials: Vec<MaterialId>,
}

impl Region {
    /// Local cell of point `p` (global cell index minus the region's first).
    #[inline]
    fn cell(&self, w: &WildTiles, p: DVec2) -> Option<usize> {
        let cx = ((p.x - w.origin.x) / w.c.cell).floor() as i64 - self.gx0;
        let cy = ((p.y - w.origin.y) / w.c.cell).floor() as i64 - self.gy0;
        let mc = self.m as i64 - 1;
        ((0..mc).contains(&cx) && (0..mc).contains(&cy)).then(|| cy as usize * (self.m - 1) + cx as usize)
    }
}

impl WildTiles {
    fn region(&self, tx: u32, ty: u32) -> Region {
        let tc = self.t.tile_cells as i64;
        let e = self.margin_cells;
        let (gx0, gy0) = (tx as i64 * tc - e, ty as i64 * tc - e);
        let m = (tc + 2 * e + 1) as usize;
        let mc = m - 1;
        let mut heights = Vec::with_capacity(m * m);
        let mut moisture = Vec::with_capacity(m * m);
        for j in 0..m as i64 {
            for i in 0..m as i64 {
                heights.push(self.vertex_height(gx0 + i, gy0 + j) as f32);
                moisture.push(self.vertex_moisture(gx0 + i, gy0 + j));
            }
        }
        let r = self.ratio;
        let mut water = vec![f32::NAN; mc * mc];
        for j in 0..mc {
            for i in 0..mc {
                let (gx, gy) = (gx0 + i as i64, gy0 + j as i64);
                let level = self.lake_level_near(gx.div_euclid(r), gy.div_euclid(r));
                if level > f32::NEG_INFINITY {
                    let k = j * m + i;
                    let low = heights[k].min(heights[k + 1]).min(heights[k + m]).min(heights[k + m + 1]);
                    if low < level {
                        water[j * mc + i] = level;
                    }
                }
            }
        }
        let cell = self.c.cell;
        let mat = &self.c.materials;
        let shore = (mat.shore_width / cell).ceil() as usize;
        let near = (mat.shore_width > 0.0 && water.iter().any(|w| !w.is_nan())).then(|| nearby_max(&water, mc, shore));
        let rule = MaterialRule::new(mat, &self.lines, self.c.terrain.relief);
        let treeline = |x: f64, y: f64| self.lines.treeline(mat, self.c.terrain.relief, x, y);
        let inv = 0.5 / cell;
        let v = |x: usize, y: usize| heights[y.min(m - 1) * m + x.min(m - 1)] as f64;
        let mut slope = vec![0.0f32; mc * mc];
        let mut materials = vec![MaterialId::GRASS; mc * mc];
        for cy in 0..mc {
            let y = self.origin.y + ((gy0 + cy as i64) as f64 + 0.5) * cell;
            for cx in 0..mc {
                let i = cy * m + cx;
                let (x0, x1, y0, y1) = (cx.saturating_sub(1), cx + 2, cy.saturating_sub(1), cy + 2);
                let gx = (v(x1, cy) + v(x1, cy + 1) - v(x0, cy) - v(x0, cy + 1)) * inv / (x1.min(m - 1) - x0) as f64;
                let gy = (v(cx, y1) + v(cx + 1, y1) - v(cx, y0) - v(cx + 1, y0)) * inv / (y1.min(m - 1) - y0) as f64;
                let s = (gx * gx + gy * gy).sqrt();
                let k = cy * mc + cx;
                slope[k] = s as f32;
                let z = 0.25
                    * (heights[i] as f64 + heights[i + 1] as f64 + heights[i + m] as f64 + heights[i + m + 1] as f64);
                let x = self.origin.x + ((gx0 + cx as i64) as f64 + 0.5) * cell;
                let wet = moisture[i].max(moisture[i + 1]).max(moisture[i + m]).max(moisture[i + m + 1]) as f64;
                materials[k] = rule.material(x, y, z, s, water[k], near.as_ref().map(|l| l[k]), wet, &treeline);
            }
        }
        Region { gx0, gy0, m, heights, water, slope, materials }
    }

    /// Kept trees around tile `(tx, ty)` (candidates within the scatter margin).
    fn trees(&self, reg: &Region, lo: DVec2, hi: DVec2) -> Vec<TreeCandidate> {
        let c = &self.c.trees;
        let max_density = c.forest_density.max(c.meadow_density);
        if max_density <= 0.0 {
            return Vec::new();
        }
        let max_slope = libm::tan(c.max_slope_deg.to_radians());
        let per_cell = (max_density * SCATTER_CELL * SCATTER_CELL / 10_000.0).ceil() as usize;
        let (mlo, mhi) = self.layout.extent();
        let mut found = Vec::new();
        self.scatter_cells(lo, hi, |cell, clo| {
            let mut rng = self.tree_seed.child_index(cell as u64).rng();
            for k in 0..per_cell {
                let p = DVec2::new(rng.range(clo.x, clo.x + SCATTER_CELL), rng.range(clo.y, clo.y + SCATTER_CELL));
                let (accept, u_height, u_species, priority) =
                    (rng.uniform(), rng.uniform(), rng.uniform(), rng.uniform());
                if p.x < lo.x || p.y < lo.y || p.x > hi.x || p.y > hi.y {
                    continue;
                }
                if p.x < mlo.x + EDGE_MARGIN
                    || p.y < mlo.y + EDGE_MARGIN
                    || p.x > mhi.x - EDGE_MARGIN
                    || p.y > mhi.y - EDGE_MARGIN
                {
                    continue;
                }
                let Some(i) = reg.cell(self, p) else { continue };
                let slope = reg.slope[i] as f64;
                if !reg.water[i].is_nan() || slope > max_slope {
                    continue;
                }
                let density = match reg.materials[i] {
                    MaterialId::FOREST_FLOOR => c.forest_density,
                    MaterialId::GRASS => c.meadow_density,
                    _ => 0.0,
                };
                let z = self.surface(p);
                let treeline = self.lines.treeline(&self.c.materials, self.c.terrain.relief, p.x, p.y);
                if accept * max_density >= density || z >= treeline {
                    continue;
                }
                let alt = (z / treeline).clamp(0.0, 1.0);
                let conifer = u_species < c.conifer_low + (c.conifer_high - c.conifer_low) * alt;
                let height = (c.min_height + u_height * (c.max_height - c.min_height)) * (1.0 - 0.4 * alt);
                found.push(TreeCandidate {
                    p,
                    key: WildTiles::id(cell, k, 0, false),
                    priority,
                    height: height.max(c.min_height * 0.6),
                    conifer,
                    slope,
                });
            }
        });
        // Keep a candidate when no other within the spacing ranks higher.
        let spacing = c.min_spacing;
        let hash = Buckets::new(lo, hi, spacing, found.iter().map(|t| t.p));
        let rank = |t: &TreeCandidate| (t.priority, t.key);
        let keep: Vec<bool> = found
            .iter()
            .map(|t| {
                !hash.any(t.p, spacing, |j| {
                    let u = &found[j];
                    u.p.distance_squared(t.p) < spacing * spacing && rank(u) > rank(t)
                })
            })
            .collect();
        found.into_iter().zip(keep).filter_map(|(t, k)| k.then_some(t)).collect()
    }

    /// Calls `f(global cell index, cell corner)` for the scatter cells overlapping `[lo, hi]`
    /// in ascending index order.
    fn scatter_cells(&self, lo: DVec2, hi: DVec2, mut f: impl FnMut(u32, DVec2)) {
        let n = self.scatter_nx as i64;
        let idx = |v: f64, o: f64| (((v - o) / SCATTER_CELL).floor() as i64).clamp(0, n - 1);
        let (x0, x1) = (idx(lo.x, self.origin.x), idx(hi.x, self.origin.x));
        let (y0, y1) = (idx(lo.y, self.origin.y), idx(hi.y, self.origin.y));
        for sy in y0..=y1 {
            for sx in x0..=x1 {
                let corner = self.origin + DVec2::new(sx as f64, sy as f64) * SCATTER_CELL;
                f((sy * n + sx) as u32, corner);
            }
        }
    }
}

/// Highest water level within `r` cells (−∞ where there is none), sequentially.
fn nearby_max(water: &[f32], w: usize, r: usize) -> Vec<f32> {
    let h = water.len() / w;
    let level = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
    let mut tmp = vec![f32::NEG_INFINITY; water.len()];
    for iy in 0..h {
        for ix in 0..w {
            let (a, b) = (ix.saturating_sub(r), (ix + r).min(w - 1));
            tmp[iy * w + ix] = water[iy * w + a..=iy * w + b].iter().fold(f32::NEG_INFINITY, |m, &v| m.max(level(v)));
        }
    }
    let mut out = vec![f32::NEG_INFINITY; water.len()];
    for iy in 0..h {
        let (a, b) = (iy.saturating_sub(r), (iy + r).min(h - 1));
        for ix in 0..w {
            out[iy * w + ix] = (a..=b).fold(f32::NEG_INFINITY, |m, y| m.max(tmp[y * w + ix]));
        }
    }
    out
}

/// Points in square buckets for neighbour queries.
struct Buckets {
    lo: DVec2,
    size: f64,
    w: usize,
    h: usize,
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Buckets {
    fn new(lo: DVec2, hi: DVec2, size: f64, points: impl Iterator<Item = DVec2> + Clone) -> Self {
        let w = ((hi.x - lo.x) / size).ceil() as usize + 1;
        let h = ((hi.y - lo.y) / size).ceil() as usize + 1;
        let bucket = |p: DVec2| {
            let bx = (((p.x - lo.x) / size).floor() as usize).min(w - 1);
            let by = (((p.y - lo.y) / size).floor() as usize).min(h - 1);
            by * w + bx
        };
        let mut count = vec![0u32; w * h + 1];
        for p in points.clone() {
            count[bucket(p) + 1] += 1;
        }
        for i in 1..count.len() {
            count[i] += count[i - 1];
        }
        let mut fill = count.clone();
        let mut items = vec![0u32; *count.last().unwrap() as usize];
        for (i, p) in points.enumerate() {
            let b = bucket(p);
            items[fill[b] as usize] = i as u32;
            fill[b] += 1;
        }
        Self { lo, size, w, h, start: count, items }
    }

    /// Whether `pred` holds for an item in the buckets within `r` of `p`.
    fn any(&self, p: DVec2, r: f64, mut pred: impl FnMut(usize) -> bool) -> bool {
        let k = (r / self.size).ceil() as i64;
        let bx = ((p.x - self.lo.x) / self.size).floor() as i64;
        let by = ((p.y - self.lo.y) / self.size).floor() as i64;
        for y in (by - k).max(0)..=(by + k).min(self.h as i64 - 1) {
            for x in (bx - k).max(0)..=(bx + k).min(self.w as i64 - 1) {
                let b = y as usize * self.w + x as usize;
                for &i in &self.items[self.start[b] as usize..self.start[b + 1] as usize] {
                    if pred(i as usize) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl TileSource for WildTiles {
    fn tile(&self, tx: u32, ty: u32) -> Tile {
        let reg = self.region(tx, ty);
        let (core_lo, core_hi) = self.layout.core(tx, ty);

        // Stored grid: the core plus the ring.
        let e = self.margin_cells as usize;
        let ring = RING_CELLS;
        let tc = self.t.tile_cells as usize;
        let ms = tc + 2 * ring + 1;
        let off = e - ring;
        let (m, mc) = (reg.m, reg.m - 1);
        let mut heights = Vec::with_capacity(ms * ms);
        for j in 0..ms {
            heights.extend_from_slice(&reg.heights[(off + j) * m + off..(off + j) * m + off + ms]);
        }
        let (mut materials, mut water) =
            (Vec::with_capacity((ms - 1) * (ms - 1)), Vec::with_capacity((ms - 1) * (ms - 1)));
        for j in 0..ms - 1 {
            let row = (off + j) * mc + off;
            materials.extend_from_slice(&reg.materials[row..row + ms - 1]);
            water.extend_from_slice(&reg.water[row..row + ms - 1]);
        }
        let grid_origin = core_lo - ring as f64 * self.c.cell;
        let mut grid = HeightGrid::new(grid_origin, self.c.cell, ms, ms, heights, materials);
        if water.iter().any(|w| !w.is_nan()) {
            grid = grid.with_water(water);
        }

        // Scatter around the core.
        let s = self.scatter_margin;
        let (lo, hi) = (core_lo - s, core_hi + s);
        let trees = self.trees(&reg, lo, hi);
        let (qlo, qhi) = (core_lo - REACH, core_hi + REACH);
        let overlaps = |o: &Obstacle| {
            let (a, b) = o.aabb();
            b.x >= qlo.x && a.x <= qhi.x && b.y >= qlo.y && a.y <= qhi.y
        };
        let mut out: Vec<(u32, Obstacle)> = Vec::new();
        let tree_reach = REACH + 0.3 * self.c.trees.max_height;
        for t in &trees {
            if t.p.x < core_lo.x - tree_reach
                || t.p.y < core_lo.y - tree_reach
                || t.p.x > core_hi.x + tree_reach
                || t.p.y > core_hi.y + tree_reach
            {
                continue;
            }
            let sink = 0.15 + 0.04 * t.height * t.slope;
            let base = t.p.extend(self.surface(t.p) - sink);
            let parts = if t.conifer { conifer(base, t.height) } else { broadleaf(base, t.height) };
            for (part, o) in parts.into_iter().enumerate() {
                if overlaps(&o) {
                    out.push((t.key | ((part as u32) << 1), o));
                }
            }
        }
        self.rocks(&reg, &trees, core_lo, core_hi, &overlaps, &mut out);
        out.sort_unstable_by_key(|(id, _)| *id);
        let ids = out.iter().map(|(id, _)| *id).collect();
        let obstacles = ObstacleSet::new(out.into_iter().map(|(_, o)| o).collect());
        Tile { grid, obstacles, ids }
    }

    fn owner(&self, id: u32) -> (u32, u32) {
        let cell = id >> 8;
        let (sx, sy) = (cell % self.scatter_nx, cell / self.scatter_nx);
        let per_tile = (self.layout.tile_size / SCATTER_CELL).round() as u32;
        (sx / per_tile, sy / per_tile)
    }
}

impl WildTiles {
    /// Rocks anchored near the core that overlap it (grown by `REACH`).
    fn rocks(
        &self,
        reg: &Region,
        trees: &[TreeCandidate],
        core_lo: DVec2,
        core_hi: DVec2,
        overlaps: &dyn Fn(&Obstacle) -> bool,
        out: &mut Vec<(u32, Obstacle)>,
    ) {
        let c = &self.c.rocks;
        let max_density = c.rocky_density.max(c.other_density);
        if max_density <= 0.0 {
            return;
        }
        let per_cell = (max_density * SCATTER_CELL * SCATTER_CELL / 10_000.0).ceil() as usize;
        let reach = REACH + 0.6 * c.max_size;
        let (lo, hi) = (core_lo - reach, core_hi + reach);
        let (mlo, mhi) = self.layout.extent();
        let clearance = 0.5 * c.max_size + 0.5;
        let tree_hash = Buckets::new(lo - clearance, hi + clearance, clearance.max(1.0), trees.iter().map(|t| t.p));
        self.scatter_cells(lo, hi, |cell, clo| {
            let mut rng = self.rock_seed.child_index(cell as u64).rng();
            for k in 0..per_cell {
                let p = DVec2::new(rng.range(clo.x, clo.x + SCATTER_CELL), rng.range(clo.y, clo.y + SCATTER_CELL));
                let accept = rng.uniform();
                let u = 1.0 - rng.uniform();
                let size = (c.min_size * libm::pow(u, -1.0 / c.size_exponent)).min(c.max_size);
                let (shape, bottom, top) = rock_shape(&mut rng, size);
                let yaw = rng.range(0.0, std::f64::consts::TAU);
                if p.x < lo.x || p.y < lo.y || p.x > hi.x || p.y > hi.y {
                    continue;
                }
                if p.x < mlo.x + EDGE_MARGIN
                    || p.y < mlo.y + EDGE_MARGIN
                    || p.x > mhi.x - EDGE_MARGIN
                    || p.y > mhi.y - EDGE_MARGIN
                {
                    continue;
                }
                let Some(i) = reg.cell(self, p) else { continue };
                if !reg.water[i].is_nan() {
                    continue;
                }
                let density = match reg.materials[i] {
                    MaterialId::ROCK | MaterialId::SCREE => c.rocky_density,
                    _ => c.other_density,
                };
                if accept * max_density >= density {
                    continue;
                }
                let r = 0.5 * size + 0.5;
                if tree_hash.any(p, r, |j| trees[j].p.distance_squared(p) < r * r) {
                    continue;
                }
                let ground = self.surface(p);
                let centre = p.extend(ground - c.sunk * (top - bottom) - bottom);
                let (s, co) = (libm::sin(0.5 * yaw), libm::cos(0.5 * yaw));
                let pose = autonomousim_core::math::Pose { pos: centre, rot: glam::DQuat::from_xyzw(0.0, 0.0, s, co) };
                let o =
                    Obstacle::solid(shape, pose, MaterialId::ROCK).with_tag(autonomousim_world::obstacles::tags::ROCK);
                if overlaps(&o) {
                    out.push((WildTiles::id(cell, k, 0, true), o));
                }
            }
        });
    }
}
