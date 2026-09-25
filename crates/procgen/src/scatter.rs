//! Trees and rocks.
//!
//! Candidates are generated per 64 m tile in parallel, each tile from its own seed, and then
//! accepted or rejected in one sequential pass in tile order (minimum spacing through a
//! spatial hash). The result does not depend on the thread count.

use autonomousim_core::material::MaterialId;
use autonomousim_core::math::Pose;
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::terrain::Terrain;
use autonomousim_world::obstacles::tags;
use autonomousim_world::testworlds::tree as conifer;
use autonomousim_world::{HeightGrid, Obstacle, ObstacleShape};
use glam::{DQuat, DVec2, DVec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

const TILE: f64 = 64.0;
/// Candidates stay this far from the map edge.
const EDGE_MARGIN: f64 = 2.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TreesConfig {
    /// Trees per hectare on forest floor.
    pub forest_density: f64,
    /// Trees per hectare on grass below the treeline.
    pub meadow_density: f64,
    /// Smallest distance between two trunks (m).
    pub min_spacing: f64,
    pub max_slope_deg: f64,
    pub min_height: f64,
    pub max_height: f64,
    /// Share of conifers at the lowest altitude and at the treeline.
    pub conifer_low: f64,
    pub conifer_high: f64,
}

impl Default for TreesConfig {
    fn default() -> Self {
        Self {
            forest_density: 350.0,
            meadow_density: 6.0,
            min_spacing: 3.0,
            max_slope_deg: 35.0,
            min_height: 8.0,
            max_height: 26.0,
            conifer_low: 0.3,
            conifer_high: 0.95,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RocksConfig {
    /// Rocks per hectare on rock and scree.
    pub rocky_density: f64,
    /// Rocks per hectare on other dry ground.
    pub other_density: f64,
    /// Sizes (m, longest diameter) follow a Pareto law with this minimum and exponent,
    /// truncated at `max_size`.
    pub min_size: f64,
    pub max_size: f64,
    pub size_exponent: f64,
    /// Fraction of a rock's height below the ground.
    pub sunk: f64,
}

impl Default for RocksConfig {
    fn default() -> Self {
        Self { rocky_density: 120.0, other_density: 2.0, min_size: 0.4, max_size: 4.0, size_exponent: 2.2, sunk: 0.3 }
    }
}

impl TreesConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if !(self.forest_density >= 0.0 && self.meadow_density >= 0.0 && self.forest_density <= 2000.0) {
            return Err("trees: densities must be in [0, 2000] per hectare".into());
        }
        if self.min_spacing.is_nan() || self.min_spacing < 0.5 {
            return Err("trees.min_spacing must be at least 0.5 m".into());
        }
        if !(self.min_height > 1.0 && self.max_height >= self.min_height && self.max_height <= 80.0) {
            return Err("trees: need 1 < min_height ≤ max_height ≤ 80".into());
        }
        if !(0.0..=1.0).contains(&self.conifer_low) || !(0.0..=1.0).contains(&self.conifer_high) {
            return Err("trees: conifer shares must be in [0, 1]".into());
        }
        if !(self.max_slope_deg > 0.0 && self.max_slope_deg < 90.0) {
            return Err("trees.max_slope_deg must be in (0, 90)".into());
        }
        Ok(())
    }
}

impl RocksConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if !(self.rocky_density >= 0.0 && self.other_density >= 0.0 && self.rocky_density <= 2000.0) {
            return Err("rocks: densities must be in [0, 2000] per hectare".into());
        }
        if !(self.min_size >= 0.05 && self.max_size >= self.min_size && self.size_exponent > 0.0) {
            return Err("rocks: need 0.05 ≤ min_size ≤ max_size and size_exponent > 0".into());
        }
        if !(0.0..0.5).contains(&self.sunk) {
            return Err("rocks.sunk must be in [0, 0.5)".into());
        }
        Ok(())
    }
}

/// What the scatter passes need to know about the ground.
pub(crate) struct Ground<'a> {
    pub grid: &'a HeightGrid,
    /// Tangent of the slope per cell.
    pub slope: &'a [f32],
    /// Treeline height (m) at a position.
    pub treeline: &'a (dyn Fn(f64, f64) -> f64 + Sync),
}

impl Ground<'_> {
    fn cell(&self, p: DVec2) -> (usize, usize) {
        let (cw, ch) = self.grid.cells();
        let q = (p - self.grid.origin()) / self.grid.cell_size();
        ((q.x as usize).min(cw - 1), (q.y as usize).min(ch - 1))
    }

    fn slope(&self, cx: usize, cy: usize) -> f64 {
        self.slope[cy * self.grid.cells().0 + cx] as f64
    }
}

/// Tiles covering the map, row-major.
fn tiles(grid: &HeightGrid) -> Vec<(DVec2, DVec2)> {
    let (cw, ch) = grid.cells();
    let size = DVec2::new(cw as f64, ch as f64) * grid.cell_size();
    let lo = grid.origin() + EDGE_MARGIN;
    let hi = grid.origin() + size - EDGE_MARGIN;
    let (nx, ny) = (((hi.x - lo.x) / TILE).ceil() as usize, ((hi.y - lo.y) / TILE).ceil() as usize);
    let mut out = Vec::with_capacity(nx * ny);
    for ty in 0..ny {
        for tx in 0..nx {
            let a = lo + DVec2::new(tx as f64, ty as f64) * TILE;
            out.push((a, (a + TILE).min(hi)));
        }
    }
    out
}

/// Draws `per_tile(area_ha)` candidates in every tile (in parallel) with `make` and returns
/// the accepted ones in tile order.
fn candidates<T: Send>(
    grid: &HeightGrid,
    seed: &Seed,
    max_density: f64,
    make: impl Fn(&mut SimRng, DVec2) -> Option<T> + Sync,
) -> Vec<T> {
    if max_density <= 0.0 {
        return Vec::new();
    }
    let tiles = tiles(grid);
    let per_tile: Vec<Vec<T>> = tiles
        .par_iter()
        .enumerate()
        .map(|(i, &(lo, hi))| {
            let mut rng = seed.child_index(i as u64).rng();
            let area_ha = (hi - lo).x * (hi - lo).y / 10_000.0;
            let n = (max_density * area_ha).ceil() as usize;
            (0..n)
                .filter_map(|_| {
                    let p = DVec2::new(rng.range(lo.x, hi.x), rng.range(lo.y, hi.y));
                    make(&mut rng, p)
                })
                .collect()
        })
        .collect();
    per_tile.into_iter().flatten().collect()
}

/// Points in a uniform grid of buckets for neighbour queries.
struct SpatialHash {
    origin: DVec2,
    inv_cell: f64,
    cell: f64,
    w: usize,
    h: usize,
    head: Vec<u32>,
    next: Vec<u32>,
    points: Vec<DVec2>,
}

impl SpatialHash {
    fn new(grid: &HeightGrid, cell: f64) -> Self {
        let (cw, ch) = grid.cells();
        let size = DVec2::new(cw as f64, ch as f64) * grid.cell_size();
        let w = (size.x / cell).ceil() as usize + 1;
        let h = (size.y / cell).ceil() as usize + 1;
        Self {
            origin: grid.origin(),
            inv_cell: 1.0 / cell,
            cell,
            w,
            h,
            head: vec![u32::MAX; w * h],
            next: Vec::new(),
            points: Vec::new(),
        }
    }

    fn bucket(&self, p: DVec2) -> (isize, isize) {
        let q = (p - self.origin) * self.inv_cell;
        (q.x.floor() as isize, q.y.floor() as isize)
    }

    fn any_within(&self, p: DVec2, r: f64) -> bool {
        let (bx, by) = self.bucket(p);
        let k = (r / self.cell).ceil() as isize;
        for y in (by - k).max(0)..=(by + k).min(self.h as isize - 1) {
            for x in (bx - k).max(0)..=(bx + k).min(self.w as isize - 1) {
                let mut i = self.head[y as usize * self.w + x as usize];
                while i != u32::MAX {
                    if self.points[i as usize].distance_squared(p) < r * r {
                        return true;
                    }
                    i = self.next[i as usize];
                }
            }
        }
        false
    }

    fn insert(&mut self, p: DVec2) {
        let (bx, by) = self.bucket(p);
        let b = by.clamp(0, self.h as isize - 1) as usize * self.w + bx.clamp(0, self.w as isize - 1) as usize;
        self.next.push(self.head[b]);
        self.head[b] = self.points.len() as u32;
        self.points.push(p);
    }
}

pub(crate) struct Trees {
    pub obstacles: Vec<Obstacle>,
    pub count: usize,
    pub conifers: usize,
    /// Trunk positions for the rock pass.
    hash: SpatialHash,
}

struct TreeCandidate {
    p: DVec2,
    height: f64,
    conifer: bool,
}

pub(crate) fn trees(g: &Ground, c: &TreesConfig, seed: &Seed) -> Trees {
    let max_slope = libm::tan(c.max_slope_deg.to_radians());
    let max_density = c.forest_density.max(c.meadow_density);
    let found = candidates(g.grid, seed, max_density, |rng, p| {
        // Always draw the same numbers so that one rejection does not shift the others.
        let (accept, u_height, u_species) = (rng.uniform(), rng.uniform(), rng.uniform());
        let (cx, cy) = g.cell(p);
        if g.grid.cell_water(cx, cy).is_some() || g.slope(cx, cy) > max_slope {
            return None;
        }
        let density = match g.grid.cell_material(cx, cy) {
            MaterialId::FOREST_FLOOR => c.forest_density,
            MaterialId::GRASS => c.meadow_density,
            _ => 0.0,
        };
        let z = g.grid.height(p.x, p.y);
        let treeline = (g.treeline)(p.x, p.y);
        if accept * max_density >= density || z >= treeline {
            return None;
        }
        // Higher up: more conifers, smaller trees.
        let alt = (z / treeline).clamp(0.0, 1.0);
        let conifer = u_species < c.conifer_low + (c.conifer_high - c.conifer_low) * alt;
        let height = (c.min_height + u_height * (c.max_height - c.min_height)) * (1.0 - 0.4 * alt);
        Some(TreeCandidate { p, height: height.max(c.min_height * 0.6), conifer })
    });
    let mut hash = SpatialHash::new(g.grid, c.min_spacing);
    let mut obstacles = Vec::with_capacity(2 * found.len());
    let mut conifers = 0;
    for t in found {
        if hash.any_within(t.p, c.min_spacing) {
            continue;
        }
        hash.insert(t.p);
        let (cx, cy) = g.cell(t.p);
        // Sink the trunk so that it meets the ground on its downhill side too.
        let sink = 0.15 + 0.04 * t.height * g.slope(cx, cy);
        let base = t.p.extend(g.grid.height(t.p.x, t.p.y) - sink);
        if t.conifer {
            conifers += 1;
            obstacles.extend(conifer(base, t.height));
        } else {
            obstacles.extend(broadleaf(base, t.height));
        }
    }
    Trees { count: hash.points.len(), conifers, obstacles, hash }
}

/// Broadleaf tree standing at `base`: a solid capsule trunk and a spherical foliage crown.
pub(crate) fn broadleaf(base: DVec3, height: f64) -> [Obstacle; 2] {
    let trunk_r = 0.03 * height;
    let trunk_hh = 0.25 * height;
    let crown_r = 0.3 * height;
    let trunk = Obstacle::solid(
        ObstacleShape::Capsule { half_height: trunk_hh, radius: trunk_r },
        Pose::from_translation(base + DVec3::Z * trunk_hh),
        MaterialId::WOOD,
    )
    .with_tag(tags::TRUNK);
    let crown = Obstacle::foliage(
        ObstacleShape::Sphere { radius: crown_r },
        Pose::from_translation(base + DVec3::Z * (height - crown_r)),
    )
    .with_tag(tags::CANOPY_BROADLEAF);
    [trunk, crown]
}

pub(crate) struct Rocks {
    pub obstacles: Vec<Obstacle>,
}

struct RockCandidate {
    p: DVec2,
    size: f64,
    shape: ObstacleShape,
    yaw: f64,
    /// Lowest and highest point in the rock frame.
    bottom: f64,
    top: f64,
}

pub(crate) fn rocks(g: &Ground, c: &RocksConfig, trees: &Trees, seed: &Seed) -> Rocks {
    let max_density = c.rocky_density.max(c.other_density);
    let found = candidates(g.grid, seed, max_density, |rng, p| {
        let accept = rng.uniform();
        let (cx, cy) = g.cell(p);
        if g.grid.cell_water(cx, cy).is_some() {
            return None;
        }
        let density = match g.grid.cell_material(cx, cy) {
            MaterialId::ROCK | MaterialId::SCREE => c.rocky_density,
            _ => c.other_density,
        };
        if accept * max_density >= density {
            return None;
        }
        let u = 1.0 - rng.uniform();
        let size = (c.min_size * libm::pow(u, -1.0 / c.size_exponent)).min(c.max_size);
        let (shape, bottom, top) = rock_shape(rng, size);
        Some(RockCandidate { p, size, shape, yaw: rng.range(0.0, TAU), bottom, top })
    });
    let mut obstacles = Vec::with_capacity(found.len());
    for r in found {
        if trees.hash.any_within(r.p, 0.5 * r.size + 0.5) {
            continue;
        }
        let ground = g.grid.height(r.p.x, r.p.y);
        let centre = r.p.extend(ground - c.sunk * (r.top - r.bottom) - r.bottom);
        let (s, co) = (libm::sin(0.5 * r.yaw), libm::cos(0.5 * r.yaw));
        let pose = Pose { pos: centre, rot: DQuat::from_xyzw(0.0, 0.0, s, co) };
        obstacles.push(Obstacle::solid(r.shape, pose, MaterialId::ROCK).with_tag(tags::ROCK));
    }
    Rocks { obstacles }
}

/// Perturbed ellipsoid with diameter `size`: 6 axis points (so the hull contains its centre)
/// and 8 random directions, each pushed in or out by up to 20 %. Returns the shape and its
/// lowest and highest `z`.
fn rock_shape(rng: &mut SimRng, size: f64) -> (ObstacleShape, f64, f64) {
    let a = 0.5 * size;
    let axes = DVec3::new(a, a * rng.range(0.6, 1.0), a * rng.range(0.35, 0.75));
    let mut dirs = [DVec3::ZERO; 14];
    dirs[..6].copy_from_slice(&[DVec3::X, -DVec3::X, DVec3::Y, -DVec3::Y, DVec3::Z, -DVec3::Z]);
    for d in &mut dirs[6..] {
        let z = rng.range(-1.0, 1.0);
        let phi = rng.range(0.0, TAU);
        let r = (1.0 - z * z).sqrt();
        *d = DVec3::new(r * libm::cos(phi), r * libm::sin(phi), z);
    }
    let points: Vec<DVec3> = dirs.iter().map(|d| *d * axes * rng.range(0.8, 1.2)).collect();
    let bottom = points.iter().map(|p| p.z).fold(0.0, f64::min);
    let top = points.iter().map(|p| p.z).fold(0.0, f64::max);
    (ObstacleShape::ConvexHull { points }, bottom, top)
}
