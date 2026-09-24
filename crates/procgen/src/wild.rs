//! Wild (untouched nature) maps: hills and mountains, lakes, meadows, forest, rock and snow.
//!
//! Pipeline:
//! 1. Noise terrain on a grid of twice the final cell size ([`TerrainConfig`]).
//! 2. Particle hydraulic and thermal erosion on that grid ([`ErosionConfig`]).
//! 3. Catmull–Rom upsampling to the final grid plus fine detail noise.
//! 4. Priority-Flood depression filling: lakes where depressions are deep and large enough,
//!    and upstream areas that make valleys and stream beds moist.
//! 5. Materials per cell from slope, altitude (tree- and snowline), moisture, shore distance
//!    and a forest noise mask.
//! 6. Trees (conifer or broadleaf) on forest floor and meadows, rocks mostly on rock and scree.
//!
//! The result depends only on the configuration and the seed, never on the thread count.

use crate::ProcgenError;
use crate::hydrology::{accumulation, lakes, priority_flood};
use crate::noise::{Fractal, fbm};
use crate::scatter::{Ground, RocksConfig, TreesConfig, rocks, trees};
use crate::terrain::{
    ErosionConfig, Field, TerrainConfig, TerrainSeeds, base_height, detail_height, hydraulic_erosion, thermal_erosion,
    upsample2,
};
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::rng::Seed;
use autonomousim_world::{GeoOrigin, HeightGrid, MapMeta, ObstacleSet, StaticWorld};
use glam::DVec2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Bumped whenever the output for a given configuration and seed changes.
pub const WILD_VERSION: u32 = 1;

/// Upstream area (vertices) at which the moisture from drainage saturates.
const MOISTURE_AREA: f64 = 4000.0;
/// Radius (cells) of the box blur that spreads stream moisture into valleys.
const MOISTURE_BLUR: usize = 8;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WaterConfig {
    pub enabled: bool,
    /// Depressions shallower than this (m) stay dry.
    pub min_depth: f64,
    /// Depressions smaller than this (m²) stay dry.
    pub min_area: f64,
    /// A lake covers at most this share of its catchment; larger depressions fill only
    /// partly (lakes without outflow).
    pub max_catchment_share: f64,
}

impl Default for WaterConfig {
    fn default() -> Self {
        Self { enabled: true, min_depth: 0.5, min_area: 200.0, max_catchment_share: 0.06 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MaterialsConfig {
    /// Bare rock above this slope.
    pub rock_slope_deg: f64,
    /// Scree above this slope over the treeline.
    pub scree_slope_deg: f64,
    /// Treeline and snowline as fractions of the relief.
    pub treeline: f64,
    pub snowline: f64,
    /// Snow does not stick above this slope.
    pub snow_max_slope_deg: f64,
    /// Amplitude of the noise on tree- and snowline (fraction of the relief).
    pub line_noise: f64,
    pub forest_wavelength: f64,
    /// Forest where the forest noise (plus a moisture bonus) exceeds this, in [−1, 1].
    pub forest_threshold: f64,
    /// Sand up to this distance (m) from and height (m) above a lake.
    pub shore_width: f64,
    pub shore_height: f64,
    /// Mud on flat ground wetter than this (moisture in [0, 1]).
    pub marsh_moisture: f64,
}

impl Default for MaterialsConfig {
    fn default() -> Self {
        Self {
            rock_slope_deg: 42.0,
            scree_slope_deg: 30.0,
            treeline: 0.55,
            snowline: 0.72,
            snow_max_slope_deg: 38.0,
            line_noise: 0.05,
            forest_wavelength: 300.0,
            forest_threshold: -0.1,
            shore_width: 3.0,
            shore_height: 0.6,
            marsh_moisture: 0.85,
        }
    }
}

impl MaterialsConfig {
    fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("rock_slope_deg", self.rock_slope_deg),
            ("scree_slope_deg", self.scree_slope_deg),
            ("snow_max_slope_deg", self.snow_max_slope_deg),
        ] {
            if !(v > 0.0 && v < 90.0) {
                return Err(format!("materials.{name} must be in (0, 90)"));
            }
        }
        if !(self.forest_wavelength > 0.0 && self.shore_width >= 0.0 && self.shore_height >= 0.0) {
            return Err("materials: forest_wavelength must be positive, shore sizes non-negative".into());
        }
        if !(self.line_noise >= 0.0 && self.treeline.is_finite() && self.snowline.is_finite()) {
            return Err("materials: invalid tree- or snowline".into());
        }
        Ok(())
    }
}

/// Everything that shapes a wild map (the seed is separate).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WildConfig {
    /// Edge length of the square map (m), centred on the origin.
    pub size: f64,
    /// Final grid cell (m); erosion runs on cells twice as large.
    pub cell: f64,
    pub geo_origin: GeoOrigin,
    pub terrain: TerrainConfig,
    pub erosion: ErosionConfig,
    pub water: WaterConfig,
    pub materials: MaterialsConfig,
    pub trees: TreesConfig,
    pub rocks: RocksConfig,
}

impl Default for WildConfig {
    fn default() -> Self {
        Self::training()
    }
}

/// Named starting points for [`WildConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WildPreset {
    /// 512 m with 100 m relief: fast to generate (well under a second), for training pools.
    Training,
    /// 2 km with 300 m relief: mountains, valleys, lakes and about 60k trees.
    Showcase,
}

impl WildPreset {
    pub fn config(self) -> WildConfig {
        match self {
            Self::Training => WildConfig::training(),
            Self::Showcase => WildConfig::showcase(),
        }
    }
}

impl std::str::FromStr for WildPreset {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "training" => Ok(Self::Training),
            "showcase" => Ok(Self::Showcase),
            _ => Err(format!("unknown preset {s:?} (training, showcase)")),
        }
    }
}

impl WildConfig {
    pub fn showcase() -> Self {
        Self {
            size: 2048.0,
            cell: 1.0,
            geo_origin: GeoOrigin::default(),
            terrain: TerrainConfig::default(),
            erosion: ErosionConfig::default(),
            water: WaterConfig::default(),
            materials: MaterialsConfig::default(),
            trees: TreesConfig::default(),
            rocks: RocksConfig::default(),
        }
    }

    pub fn training() -> Self {
        Self {
            size: 512.0,
            terrain: TerrainConfig {
                relief: 100.0,
                hills_wavelength: 400.0,
                mountains_wavelength: 800.0,
                mask_wavelength: 800.0,
                warp_wavelength: 400.0,
                warp_strength: 40.0,
                ..TerrainConfig::default()
            },
            ..Self::showcase()
        }
    }

    /// `preset` with `overrides` merged on top: a (partial) configuration in the same layout,
    /// e.g. `{"size": 256.0, "trees": {"forest_density": 500.0}}`. Unknown keys are errors.
    pub fn from_preset(preset: WildPreset, overrides: Option<&serde_json::Value>) -> Result<Self, ProcgenError> {
        let config = preset.config();
        let Some(overrides) = overrides else { return Ok(config) };
        let mut value = serde_json::to_value(&config).expect("configs serialise to JSON");
        merge(&mut value, overrides);
        let config: Self = serde_json::from_value(value).map_err(|e| ProcgenError::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Vertices along each edge of the final grid.
    pub fn vertices(&self) -> usize {
        (self.size / self.cell).round() as usize + 1
    }

    pub fn validate(&self) -> Result<(), ProcgenError> {
        let check = || -> Result<(), String> {
            if !(0.25..=8.0).contains(&self.cell) {
                return Err("cell must be in [0.25, 8] m".into());
            }
            let cells = self.size / self.cell;
            if !(cells.fract() == 0.0 && (cells as u64).is_multiple_of(2) && (64.0..=16384.0).contains(&cells)) {
                return Err("size / cell must be an even integer in [64, 16384]".into());
            }
            self.terrain.validate()?;
            self.erosion.validate()?;
            self.materials.validate()?;
            self.trees.validate()?;
            self.rocks.validate()?;
            let w = &self.water;
            if !(w.min_depth >= 0.0 && w.min_area >= 0.0 && (0.0..=1.0).contains(&w.max_catchment_share)) {
                return Err("water: need min_depth, min_area ≥ 0 and max_catchment_share in [0, 1]".into());
            }
            Ok(())
        };
        check().map_err(ProcgenError::Config)
    }
}

/// Recursively overlay `top` onto `base` (objects merge key by key, anything else replaces).
fn merge(base: &mut serde_json::Value, top: &serde_json::Value) {
    match (base, top) {
        (serde_json::Value::Object(b), serde_json::Value::Object(t)) => {
            for (k, v) in t {
                match b.get_mut(k) {
                    Some(slot) => merge(slot, v),
                    None => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, t) => *b = t.clone(),
    }
}

/// What happened during generation (for logs and the CLI).
#[derive(Clone, Debug, Default, Serialize)]
pub struct WildStats {
    /// Wall time of each stage (s).
    pub stages: Vec<(&'static str, f64)>,
    pub vertices: usize,
    pub droplets: u64,
    pub lakes: usize,
    pub lake_cells: usize,
    pub trees: usize,
    pub conifers: usize,
    pub rocks: usize,
    pub height_range: (f64, f64),
    /// Slope (degrees, over a 3-cell baseline) at the 50th, 90th and 99th percentile of cells.
    pub slope_percentiles_deg: [f64; 3],
    /// Cells per material id.
    pub materials: Vec<(String, usize)>,
}

impl WildStats {
    fn stage(&mut self, name: &'static str, t: &mut Instant) {
        self.stages.push((name, t.elapsed().as_secs_f64()));
        *t = Instant::now();
    }

    pub fn total_seconds(&self) -> f64 {
        self.stages.iter().map(|s| s.1).sum()
    }
}

/// Noise seeds of the material rules.
struct LineSeeds {
    treeline: u64,
    snowline: u64,
    forest: u64,
}

const LINE_FRACTAL: Fractal = Fractal { octaves: 3, lacunarity: 2.0, gain: 0.5 };
const LINE_WAVELENGTH: f64 = 250.0;

/// Generate a wild map. Uses the current rayon pool; the output does not depend on its size.
pub fn generate(config: &WildConfig, seed: u64) -> Result<(StaticWorld, WildStats), ProcgenError> {
    config.validate()?;
    let c = config;
    let mut stats = WildStats::default();
    let mut t = Instant::now();
    let root = Seed::from_u64(seed).child("map/wild");
    let terrain_seed = root.child("terrain");
    let layer = |name: &str| terrain_seed.child(name).short();
    let ts = TerrainSeeds {
        warp: [0, 1, 2, 3].map(|i| terrain_seed.child("warp").child_index(i).short()),
        mask: layer("mask"),
        hills: layer("hills"),
        mountains: layer("mountains"),
        detail: layer("detail"),
    };
    let relief = c.terrain.relief;
    let n = c.vertices();
    let nc = (n - 1) / 2 + 1;
    let origin = DVec2::splat(-0.5 * c.size);
    let coarse_cell = 2.0 * c.cell;
    stats.vertices = n * n;

    // 1. Terrain on the coarse grid.
    let mut coarse = Field::from_rows(nc, nc, |iy, row| {
        let y = origin.y + iy as f64 * coarse_cell;
        for (ix, h) in row.iter_mut().enumerate() {
            *h = base_height(&c.terrain, &ts, origin.x + ix as f64 * coarse_cell, y);
        }
    });
    stats.stage("terrain", &mut t);

    // 2. Erosion (hydraulic erosion works in units of the relief).
    if c.erosion.enabled {
        coarse.data.iter_mut().for_each(|h| *h /= relief);
        stats.droplets = hydraulic_erosion(&mut coarse, &c.erosion, &mut root.child("erosion").rng());
        coarse.data.iter_mut().for_each(|h| *h *= relief);
        stats.stage("hydraulic erosion", &mut t);
        thermal_erosion(&mut coarse, coarse_cell, &c.erosion);
        stats.stage("thermal erosion", &mut t);
    }

    // 3. Final resolution.
    let mut fine = upsample2(&coarse);
    drop(coarse);
    fine.data.par_chunks_mut(n).enumerate().for_each(|(iy, row)| {
        let y = origin.y + iy as f64 * c.cell;
        for (ix, h) in row.iter_mut().enumerate() {
            *h += detail_height(&c.terrain, &ts, origin.x + ix as f64 * c.cell, y);
        }
    });
    let heights: Vec<f32> = fine.data.iter().map(|&h| h as f32).collect();
    drop(fine);
    stats.stage("upsample", &mut t);

    // 4. Hydrology.
    let drainage = priority_flood(&heights, n, n);
    let acc = accumulation(&drainage);
    let min_vertices = (c.water.min_area / (c.cell * c.cell)).ceil() as usize;
    let lakes = if c.water.enabled {
        let w = &c.water;
        lakes(&heights, &drainage.filled, &acc, n, n, w.min_depth as f32, min_vertices, w.max_catchment_share)
    } else {
        crate::hydrology::Lakes { water: vec![f32::NAN; (n - 1) * (n - 1)], count: 0, cells: 0 }
    };
    drop(drainage);
    stats.lakes = lakes.count;
    stats.lake_cells = lakes.cells;
    stats.stage("hydrology", &mut t);

    // 5. Moisture and materials.
    let moisture = moisture(&acc, n);
    drop(acc);
    let line_seed = root.child("lines");
    let lines = LineSeeds {
        treeline: line_seed.child("treeline").short(),
        snowline: line_seed.child("snowline").short(),
        forest: line_seed.child("forest").short(),
    };
    let m = &c.materials;
    let treeline = |x: f64, y: f64| {
        relief
            * (m.treeline + m.line_noise * fbm(lines.treeline, x / LINE_WAVELENGTH, y / LINE_WAVELENGTH, &LINE_FRACTAL))
    };
    let (materials, slope) = materials(c, origin, &heights, &lakes.water, &moisture, &lines, &treeline);
    drop(moisture);
    let mut counts = [0usize; 256];
    for id in &materials {
        counts[id.0 as usize] += 1;
    }
    let table = MaterialTable::standard();
    stats.materials = (0..256)
        .filter(|&i| counts[i] > 0)
        .map(|i| {
            let name = if i < table.len() { table.get(MaterialId(i as u8)).name.clone() } else { format!("#{i}") };
            (name, counts[i])
        })
        .collect();
    let grid = HeightGrid::new(origin, c.cell, n, n, heights, materials).with_water(lakes.water);
    stats.height_range = grid.height_range();
    stats.stage("materials", &mut t);

    // 6. Vegetation and rocks.
    let ground = Ground { grid: &grid, slope: &slope, treeline: &treeline };
    let trees = trees(&ground, &c.trees, &root.child("trees"));
    let rocks = rocks(&ground, &c.rocks, &trees, &root.child("rocks"));
    stats.trees = trees.count;
    stats.conifers = trees.conifers;
    stats.rocks = rocks.obstacles.len();
    let mut sorted = slope;
    for (k, q) in [0.5, 0.9, 0.99].into_iter().enumerate() {
        let i = ((sorted.len() - 1) as f64 * q) as usize;
        let (_, v, _) = sorted.select_nth_unstable_by(i, f32::total_cmp);
        stats.slope_percentiles_deg[k] = libm::atan(*v as f64).to_degrees();
    }
    drop(sorted);
    stats.stage("scatter", &mut t);

    let mut obstacles = trees.obstacles;
    obstacles.extend(rocks.obstacles);
    let obstacles = ObstacleSet::new(obstacles);
    let mut meta = MapMeta::new("wild", "wild", seed);
    meta.generator_version = WILD_VERSION;
    meta.geo_origin = c.geo_origin;
    let world = StaticWorld::new(meta, grid, obstacles, table);
    stats.stage("obstacle index", &mut t);
    Ok((world, stats))
}

/// Moisture in [0, 1] per vertex: log upstream area, blurred.
fn moisture(acc: &[u32], n: usize) -> Vec<f32> {
    let scale = 1.0 / libm::log(MOISTURE_AREA);
    let raw: Vec<f32> = acc.par_iter().map(|&a| (libm::log(a as f64) * scale).min(1.0) as f32).collect();
    box_blur(&raw, n, n, MOISTURE_BLUR)
}

/// Separable box blur with clamped borders; every output sample is summed in a fixed order.
fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let norm = 1.0 / (2 * r + 1) as f32;
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(iy, row)| {
        let s = &src[iy * w..(iy + 1) * w];
        for (ix, out) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for k in 0..=2 * r {
                sum += s[(ix + k).saturating_sub(r).min(w - 1)];
            }
            *out = sum * norm;
        }
    });
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(iy, row)| {
        for (ix, o) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for k in 0..=2 * r {
                sum += tmp[(iy + k).saturating_sub(r).min(h - 1) * w + ix];
            }
            *o = sum * norm;
        }
    });
    out
}

/// Highest water level within `r` cells (−∞ where there is none).
fn nearby_water(water: &[f32], w: usize, r: usize) -> Vec<f32> {
    let h = water.len() / w;
    let level = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
    let mut tmp = vec![0.0f32; water.len()];
    tmp.par_chunks_mut(w).enumerate().for_each(|(iy, row)| {
        for (ix, out) in row.iter_mut().enumerate() {
            let (a, b) = (ix.saturating_sub(r), (ix + r).min(w - 1));
            *out = water[iy * w + a..=iy * w + b].iter().fold(f32::NEG_INFINITY, |m, &v| m.max(level(v)));
        }
    });
    let mut out = vec![0.0f32; water.len()];
    out.par_chunks_mut(w).enumerate().for_each(|(iy, row)| {
        let (a, b) = (iy.saturating_sub(r), (iy + r).min(h - 1));
        for (ix, o) in row.iter_mut().enumerate() {
            *o = (a..=b).fold(f32::NEG_INFINITY, |m, y| m.max(tmp[y * w + ix]));
        }
    });
    out
}

/// Material and slope tangent of every cell.
fn materials(
    c: &WildConfig,
    origin: DVec2,
    heights: &[f32],
    water: &[f32],
    moisture: &[f32],
    lines: &LineSeeds,
    treeline: &(dyn Fn(f64, f64) -> f64 + Sync),
) -> (Vec<MaterialId>, Vec<f32>) {
    let m = &c.materials;
    let n = c.vertices();
    let cw = n - 1;
    let relief = c.terrain.relief;
    let tan = |deg: f64| libm::tan(deg.to_radians());
    let (rock, scree, snow_max) = (tan(m.rock_slope_deg), tan(m.scree_slope_deg), tan(m.snow_max_slope_deg));
    let near = if m.shore_width > 0.0 && water.iter().any(|w| !w.is_nan()) {
        Some(nearby_water(water, cw, (m.shore_width / c.cell).ceil() as usize))
    } else {
        None
    };
    // Slopes over a 3-cell baseline: rock and scree follow the landform, not the fine detail.
    let inv = 0.5 / c.cell;
    let v = |x: usize, y: usize| heights[y.min(n - 1) * n + x.min(n - 1)] as f64;
    let mut materials = vec![MaterialId::GRASS; cw * cw];
    let mut slopes = vec![0.0f32; cw * cw];
    materials.par_chunks_mut(cw).zip(slopes.par_chunks_mut(cw)).enumerate().for_each(|(cy, (mrow, srow))| {
        let y = origin.y + (cy as f64 + 0.5) * c.cell;
        for cx in 0..cw {
            let i = cy * n + cx;
            let (h00, h10, h01, h11) =
                (heights[i] as f64, heights[i + 1] as f64, heights[i + n] as f64, heights[i + n + 1] as f64);
            let (x0, x1, y0, y1) = (cx.saturating_sub(1), cx + 2, cy.saturating_sub(1), cy + 2);
            let gx = (v(x1, cy) + v(x1, cy + 1) - v(x0, cy) - v(x0, cy + 1)) * inv / (x1.min(n - 1) - x0) as f64;
            let gy = (v(cx, y1) + v(cx + 1, y1) - v(cx, y0) - v(cx + 1, y0)) * inv / (y1.min(n - 1) - y0) as f64;
            let slope = (gx * gx + gy * gy).sqrt();
            srow[cx] = slope as f32;
            let z = 0.25 * (h00 + h10 + h01 + h11);
            let x = origin.x + (cx as f64 + 0.5) * c.cell;
            let k = cy * cw + cx;
            let wet = moisture[i].max(moisture[i + 1]).max(moisture[i + n]).max(moisture[i + n + 1]) as f64;
            mrow[cx] = if !water[k].is_nan() {
                if water[k] as f64 - z < 0.5 { MaterialId::SAND } else { MaterialId::MUD }
            } else if slope > rock {
                MaterialId::ROCK
            } else if z > relief
                * (m.snowline
                    + m.line_noise * fbm(lines.snowline, x / LINE_WAVELENGTH, y / LINE_WAVELENGTH, &LINE_FRACTAL))
                && slope < snow_max
            {
                MaterialId::SNOW
            } else if z > treeline(x, y) {
                if slope > scree { MaterialId::SCREE } else { MaterialId::GRASS }
            } else if near.as_ref().is_some_and(|l| z < l[k] as f64 + m.shore_height) {
                MaterialId::SAND
            } else if wet > m.marsh_moisture && slope < 0.05 {
                MaterialId::MUD
            } else {
                let f = fbm(lines.forest, x / m.forest_wavelength, y / m.forest_wavelength, &Fractal::new(4));
                if f + 0.3 * wet > m.forest_threshold { MaterialId::FOREST_FLOOR } else { MaterialId::GRASS }
            };
        }
    });
    (materials, slopes)
}
