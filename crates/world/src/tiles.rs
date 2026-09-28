//! Large maps made of square tiles generated on demand.
//!
//! A [`TiledMap`] covers `tiles.0 × tiles.1` square tile cores of `tile_size` metres. A
//! [`TileSource`] (a generator) builds each tile as a pure function of its index: a
//! [`HeightGrid`] over the core plus a `ring` of terrain around it, and the obstacles whose
//! bounding boxes overlap the core grown by `reach`. Every value a tile holds (heights,
//! materials, water, obstacles) is a function of global coordinates only, so the terrain and
//! obstacles near a seam are the same whichever tile answers, and a map answers every query
//! identically whether its tiles were generated earlier, now, in another order or on another
//! thread.
//!
//! Queries go to the tile whose core contains the query point; point queries that reach no
//! further than the ring (terrain) or `reach` (obstacles) need that tile alone. Rays walk the
//! tiles along their path up to `detail_range` and continue on the coarse layer (the map's
//! overview grid) beyond it. Obstacles carry global ids, unique across tiles.
//!
//! Tiles live in a shared cache that keeps at most `capacity` tiles (least recently used go
//! first; a tile in use stays alive through its `Arc`) and a small per-thread list of the
//! tiles each thread used last, which answers most queries without touching shared state.

use crate::heightgrid::{HeightGrid, MAX_SEARCH_CELLS};
use crate::obstacles::ObstacleSet;
use autonomousim_core::geometry::{HitKind, HitMask, Ray, RayHit, StaticGeometry, SurfacePoint};
use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::Terrain;
use glam::{DVec2, DVec3};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// One generated tile.
#[derive(Clone, Debug)]
pub struct Tile {
    /// Terrain over the core and the ring around it.
    pub grid: HeightGrid,
    /// Obstacles whose bounding boxes overlap the core grown by the layout's `reach`, sorted
    /// by global id.
    pub obstacles: ObstacleSet,
    /// Global id of each obstacle (ascending).
    pub ids: Vec<u32>,
}

impl Tile {
    /// Approximate heap size (bytes).
    pub fn bytes(&self) -> usize {
        let (w, h) = self.grid.cells();
        let (nx, ny) = self.grid.dims();
        // Heights, materials, water (if any) and the min/max pyramid (≈ 4/3 of two floats per cell).
        let grid = 4 * nx * ny + w * h + self.grid.water().map_or(0, |w| 4 * w.len()) + 8 * w * h * 4 / 3;
        grid + self.obstacles.len() * 400 + 4 * self.ids.len()
    }
}

/// Builds tiles (a map generator).
pub trait TileSource: Send + Sync {
    /// The tile at index `(tx, ty)`.
    fn tile(&self, tx: u32, ty: u32) -> Tile;

    /// Index of the tile whose core holds the anchor of obstacle `id` (which is in that
    /// tile's obstacles).
    fn owner(&self, id: u32) -> (u32, u32);
}

/// Geometry of a tiled map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileLayout {
    /// Corner of tile `(0, 0)` (the map's minimum corner).
    pub origin: DVec2,
    /// Edge length of a tile core (m).
    pub tile_size: f64,
    /// Number of tiles along x and y.
    pub tiles: (u32, u32),
    /// Width of the terrain ring stored around each core (m); at least the closest-point
    /// search radius of its grid.
    pub ring: f64,
    /// Obstacles are stored with every tile whose core grown by `reach` (m) they overlap.
    pub reach: f64,
    /// Rays use the tiles up to this distance and the coarse layer beyond it (m).
    pub detail_range: f64,
}

impl TileLayout {
    /// Horizontal extent of the map.
    pub fn extent(&self) -> (DVec2, DVec2) {
        let size = DVec2::new(self.tiles.0 as f64, self.tiles.1 as f64) * self.tile_size;
        (self.origin, self.origin + size)
    }

    /// Tile whose core contains `(x, y)`, clamped to the map.
    #[inline]
    pub fn tile_at(&self, x: f64, y: f64) -> (u32, u32) {
        let u = (x - self.origin.x) / self.tile_size;
        let v = (y - self.origin.y) / self.tile_size;
        ((u.max(0.0) as u32).min(self.tiles.0 - 1), (v.max(0.0) as u32).min(self.tiles.1 - 1))
    }

    /// Core rectangle of tile `(tx, ty)`.
    pub fn core(&self, tx: u32, ty: u32) -> (DVec2, DVec2) {
        let lo = self.origin + DVec2::new(tx as f64, ty as f64) * self.tile_size;
        (lo, lo + self.tile_size)
    }

    /// Tile index range `(tx0, ty0, tx1, ty1)` (inclusive) whose cores overlap `[min, max]`.
    fn tiles_overlapping(&self, min: DVec2, max: DVec2) -> (u32, u32, u32, u32) {
        let (a, b) = self.tile_at(min.x, min.y);
        let (c, d) = self.tile_at(max.x, max.y);
        (a, b, c, d)
    }

    #[inline]
    fn index(&self, tx: u32, ty: u32) -> u32 {
        ty * self.tiles.0 + tx
    }

    /// Whether the rectangle `[min, max]` lies inside the core of `(tx, ty)` grown by `grow`.
    fn within(&self, tx: u32, ty: u32, min: DVec2, max: DVec2, grow: f64) -> bool {
        let (lo, hi) = self.core(tx, ty);
        min.x >= lo.x - grow && min.y >= lo.y - grow && max.x <= hi.x + grow && max.y <= hi.y + grow
    }
}

struct Entry {
    tile: OnceLock<Arc<Tile>>,
    /// Clock of the last use (for eviction).
    used: AtomicU64,
}

/// Tiles each thread used last: `(map id, tile index, entry, tile)`.
type Recent = Vec<(u64, u32, Arc<Entry>, Arc<Tile>)>;

/// Tiles a thread remembers (per thread, across maps).
const RECENT: usize = 4;

thread_local! {
    static RECENT_TILES: RefCell<Recent> = const { RefCell::new(Vec::new()) };
}

static NEXT_MAP_ID: AtomicU64 = AtomicU64::new(1);

/// A map made of tiles generated on demand (see the module docs).
pub struct TiledMap {
    layout: TileLayout,
    coarse: HeightGrid,
    pad: f64,
    source: Box<dyn TileSource>,
    capacity: usize,
    id: u64,
    clock: AtomicU64,
    generated: AtomicU64,
    cache: Mutex<HashMap<u32, Arc<Entry>>>,
}

impl std::fmt::Debug for TiledMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiledMap").field("layout", &self.layout).field("loaded", &self.loaded()).finish()
    }
}

impl TiledMap {
    /// A map with the given layout; `coarse` is the overview grid over the whole map (for rays
    /// beyond `detail_range`, conservative bounds and distant rendering) and `pad` bounds how
    /// far the tile terrain can deviate from it (m). At most `capacity` tiles stay cached.
    pub fn new(layout: TileLayout, coarse: HeightGrid, pad: f64, source: Box<dyn TileSource>, capacity: usize) -> Self {
        assert!(layout.tiles.0 > 0 && layout.tiles.1 > 0 && layout.tile_size > 0.0);
        assert!(layout.reach >= 0.0 && layout.detail_range > 0.0 && pad >= 0.0);
        Self {
            layout,
            coarse,
            pad,
            source,
            capacity: capacity.max(1),
            id: NEXT_MAP_ID.fetch_add(1, Ordering::Relaxed),
            clock: AtomicU64::new(0),
            generated: AtomicU64::new(0),
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn layout(&self) -> &TileLayout {
        &self.layout
    }

    /// The overview grid over the whole map.
    pub fn coarse(&self) -> &HeightGrid {
        &self.coarse
    }

    /// Bound on the difference between tile terrain and the coarse grid (m).
    pub fn pad(&self) -> f64 {
        self.pad
    }

    pub fn source(&self) -> &dyn TileSource {
        &*self.source
    }

    /// Tiles currently in the shared cache.
    pub fn loaded(&self) -> usize {
        self.cache.lock().expect("tile cache").values().filter(|e| e.tile.get().is_some()).count()
    }

    /// Tiles generated so far (a tile evicted and needed again counts twice).
    pub fn generated(&self) -> u64 {
        self.generated.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Tile `(tx, ty)`, generated if it is not cached.
    pub fn tile(&self, tx: u32, ty: u32) -> Arc<Tile> {
        let index = self.layout.index(tx, ty);
        let clock = self.clock.load(Ordering::Relaxed);
        let hit = RECENT_TILES.with_borrow_mut(|recent| {
            let k = recent.iter().position(|r| r.0 == self.id && r.1 == index)?;
            let r = recent.remove(k);
            if r.2.used.load(Ordering::Relaxed) != clock {
                r.2.used.store(clock, Ordering::Relaxed);
            }
            let tile = r.3.clone();
            recent.insert(0, r);
            Some(tile)
        });
        if let Some(tile) = hit {
            return tile;
        }
        let entry = {
            let mut cache = self.cache.lock().expect("tile cache");
            let clock = self.clock.fetch_add(1, Ordering::Relaxed) + 1;
            let entry = cache
                .entry(index)
                .or_insert_with(|| Arc::new(Entry { tile: OnceLock::new(), used: AtomicU64::new(clock) }))
                .clone();
            entry.used.store(clock, Ordering::Relaxed);
            entry
        };
        let tile = entry
            .tile
            .get_or_init(|| {
                self.generated.fetch_add(1, Ordering::Relaxed);
                Arc::new(self.source.tile(tx, ty))
            })
            .clone();
        self.evict(index);
        RECENT_TILES.with_borrow_mut(|recent| {
            recent.insert(0, (self.id, index, entry, tile.clone()));
            recent.truncate(RECENT);
        });
        tile
    }

    /// Drop least recently used tiles (never `keep`) until at most `capacity` are cached.
    fn evict(&self, keep: u32) {
        let mut cache = self.cache.lock().expect("tile cache");
        while cache.len() > self.capacity {
            let oldest = cache
                .iter()
                .filter(|(k, e)| **k != keep && e.tile.get().is_some())
                .min_by_key(|(k, e)| (e.used.load(Ordering::Relaxed), **k))
                .map(|(k, _)| *k);
            match oldest {
                Some(k) => {
                    cache.remove(&k);
                }
                None => break,
            }
        }
    }

    /// Tile whose core contains `(x, y)` (clamped to the map).
    #[inline]
    pub fn tile_at(&self, x: f64, y: f64) -> Arc<Tile> {
        let (tx, ty) = self.layout.tile_at(x, y);
        self.tile(tx, ty)
    }

    /// `(x, y)` clamped to the map extent.
    #[inline]
    fn clamp(&self, x: f64, y: f64) -> (f64, f64) {
        let (lo, hi) = self.layout.extent();
        (x.clamp(lo.x, hi.x), y.clamp(lo.y, hi.y))
    }

    /// Calls `f(tx, ty, t_enter, t_exit)` for the tiles the ray crosses within `[t0, t1]` (and
    /// inside the map), in order, while `f` returns `true`.
    fn walk(&self, ray: &Ray, t0: f64, t1: f64, mut f: impl FnMut(u32, u32, f64, f64) -> bool) {
        let (lo, hi) = self.layout.extent();
        let mut range = (t0, t1);
        for (o, d, a, b) in [(ray.origin.x, ray.dir.x, lo.x, hi.x), (ray.origin.y, ray.dir.y, lo.y, hi.y)] {
            if d == 0.0 {
                if o < a || o > b {
                    return;
                }
            } else {
                let (s, e) = ((a - o) / d, (b - o) / d);
                let (s, e) = if s < e { (s, e) } else { (e, s) };
                range = (range.0.max(s), range.1.min(e));
            }
        }
        let (mut t, t_end) = range;
        if t > t_end {
            return;
        }
        let p = ray.at(t);
        let (mut tx, mut ty) = self.layout.tile_at(p.x, p.y);
        let size = self.layout.tile_size;
        loop {
            let (clo, chi) = self.layout.core(tx, ty);
            let exit = |o: f64, d: f64, a: f64, b: f64| {
                if d > 0.0 {
                    (b - o) / d
                } else if d < 0.0 {
                    (a - o) / d
                } else {
                    f64::INFINITY
                }
            };
            let ex = exit(ray.origin.x, ray.dir.x, clo.x, chi.x);
            let ey = exit(ray.origin.y, ray.dir.y, clo.y, chi.y);
            let t_exit = ex.min(ey).min(t_end).max(t);
            if !f(tx, ty, t, t_exit) || t_exit >= t_end {
                return;
            }
            if ex <= ey {
                match (ray.dir.x > 0.0, tx) {
                    (true, x) if x + 1 < self.layout.tiles.0 => tx += 1,
                    (false, x) if x > 0 => tx -= 1,
                    _ => return,
                }
            } else {
                match (ray.dir.y > 0.0, ty) {
                    (true, y) if y + 1 < self.layout.tiles.1 => ty += 1,
                    (false, y) if y > 0 => ty -= 1,
                    _ => return,
                }
            }
            t = t_exit;
            debug_assert!(size > 0.0);
        }
    }

    /// Global ids of the obstacles in `mask` whose bounds overlap `[min, max]`, ascending.
    fn candidates_multi(&self, min: DVec3, max: DVec3, mask: HitMask, out: &mut Vec<u32>) {
        let (tx0, ty0, tx1, ty1) = self.layout.tiles_overlapping(min.truncate(), max.truncate());
        let start = out.len();
        let mut local = Vec::new();
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let tile = self.tile(tx, ty);
                local.clear();
                tile.obstacles.query_candidates(min, max, mask, &mut local);
                out.extend(local.iter().map(|&i| tile.ids[i as usize]));
            }
        }
        out[start..].sort_unstable();
        let mut w = start;
        for r in start..out.len() {
            if w == start || out[r] != out[w - 1] {
                out[w] = out[r];
                w += 1;
            }
        }
        out.truncate(w);
    }

    /// Tile holding obstacle `id` near `near` (the tile there, else its owner) and the
    /// obstacle's index in it.
    fn find(&self, id: u32, near: DVec3) -> (Arc<Tile>, usize) {
        let tile = self.tile_at(near.x, near.y);
        if let Ok(i) = tile.ids.binary_search(&id) {
            return (tile, i);
        }
        let (tx, ty) = self.source.owner(id);
        let tile = self.tile(tx, ty);
        let i = tile.ids.binary_search(&id).expect("an obstacle is stored with its owner tile");
        (tile, i)
    }
}

/// The hit kind with the tile's local obstacle index replaced by the global id.
#[inline]
fn global(kind: HitKind, ids: &[u32]) -> HitKind {
    match kind {
        HitKind::Solid(i) => HitKind::Solid(ids[i as usize]),
        HitKind::Foliage(i) => HitKind::Foliage(ids[i as usize]),
        k => k,
    }
}

impl Terrain for TiledMap {
    fn extent(&self) -> (DVec2, DVec2) {
        self.layout.extent()
    }

    fn height(&self, x: f64, y: f64) -> f64 {
        let (x, y) = self.clamp(x, y);
        self.tile_at(x, y).grid.height(x, y)
    }

    fn height_normal(&self, x: f64, y: f64) -> (f64, DVec3) {
        let (x, y) = self.clamp(x, y);
        self.tile_at(x, y).grid.height_normal(x, y)
    }

    fn material(&self, x: f64, y: f64) -> MaterialId {
        let (x, y) = self.clamp(x, y);
        self.tile_at(x, y).grid.material(x, y)
    }

    fn water_level(&self, x: f64, y: f64) -> Option<f64> {
        let (x, y) = self.clamp(x, y);
        self.tile_at(x, y).grid.water_level(x, y)
    }

    fn height_bounds(&self, min: DVec2, max: DVec2) -> (f64, f64) {
        let (tx, ty) = self.layout.tile_at(min.x, min.y);
        if self.layout.within(tx, ty, min, max, 0.0) {
            return self.tile(tx, ty).grid.height_bounds(min, max);
        }
        // Catmull–Rom stencils reach one coarse cell beyond the rectangle.
        let grow = DVec2::splat(2.0 * self.coarse.cell_size());
        let (lo, hi) = self.coarse.height_bounds(min - grow, max + grow);
        (lo - self.pad, hi + self.pad)
    }

    fn closest_point(&self, p: DVec3, max_dist: f64) -> Option<SurfacePoint> {
        let tile = self.tile_at(p.x, p.y);
        debug_assert!(self.layout.ring >= MAX_SEARCH_CELLS * tile.grid.cell_size());
        tile.grid.closest_point(p, max_dist)
    }

    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        if !mask.intersects(HitMask::TERRAIN | HitMask::WATER) {
            return None;
        }
        let near = max_toi.min(self.layout.detail_range);
        let mut hit = None;
        self.walk(ray, 0.0, near, |tx, ty, _, t_exit| {
            hit = self.tile(tx, ty).grid.raycast(ray, t_exit, mask);
            hit.is_none()
        });
        if hit.is_some() || max_toi <= near {
            return hit;
        }
        // Beyond the detail range: the coarse grid, from where the tiles ended.
        let far = Ray::new(ray.at(near), ray.dir);
        self.coarse.raycast(&far, max_toi - near, mask).map(|mut h| {
            h.toi += near;
            h
        })
    }
}

impl StaticGeometry for TiledMap {
    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        if !mask.intersects(HitMask::SOLID | HitMask::FOLIAGE) {
            return None;
        }
        let mut best: Option<RayHit> = None;
        let end = max_toi.min(self.layout.detail_range);
        // Any hit point lies in some tile's core, and that tile stores the obstacle, so the
        // tiles entered before the best hit so far are all that need checking.
        self.walk(ray, 0.0, end, |tx, ty, t_enter, _| {
            if best.is_some_and(|b| b.toi < t_enter) {
                return false;
            }
            let tile = self.tile(tx, ty);
            let limit = best.map_or(end, |b| b.toi);
            if let Some(mut h) = tile.obstacles.raycast(ray, limit, mask)
                && best.is_none_or(|b| h.toi < b.toi)
            {
                h.kind = global(h.kind, &tile.ids);
                best = Some(h);
            }
            true
        });
        best
    }

    fn sphere_contacts(&self, center: DVec3, radius: f64, margin: f64, mask: HitMask, out: &mut Vec<SurfacePoint>) {
        let r = DVec2::splat(radius + margin);
        let (min, max) = (center.truncate() - r, center.truncate() + r);
        let (tx, ty) = self.layout.tile_at(center.x, center.y);
        if self.layout.within(tx, ty, min, max, self.layout.reach) {
            let tile = self.tile(tx, ty);
            let start = out.len();
            tile.obstacles.sphere_contacts(center, radius, margin, mask, out);
            for s in &mut out[start..] {
                s.kind = global(s.kind, &tile.ids);
            }
            return;
        }
        let reach = DVec3::splat(radius + margin);
        let mut ids = Vec::new();
        self.candidates_multi(center - reach, center + reach, mask, &mut ids);
        for id in ids {
            out.extend(self.sphere_contact(id, center, radius, margin));
        }
    }

    fn nearest_distance(&self, p: DVec3, max_dist: f64, mask: HitMask) -> Option<f64> {
        let r = DVec2::splat(max_dist);
        let (min, max) = (p.truncate() - r, p.truncate() + r);
        let (tx, ty) = self.layout.tile_at(p.x, p.y);
        if self.layout.within(tx, ty, min, max, self.layout.reach) {
            return self.tile(tx, ty).obstacles.nearest_distance(p, max_dist, mask);
        }
        let (tx0, ty0, tx1, ty1) = self.layout.tiles_overlapping(min, max);
        let mut best: Option<f64> = None;
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                if let Some(d) = self.tile(tx, ty).obstacles.nearest_distance(p, max_dist, mask) {
                    best = Some(best.map_or(d, |b| b.min(d)));
                }
            }
        }
        best
    }

    fn query_candidates(&self, min: DVec3, max: DVec3, mask: HitMask, out: &mut Vec<u32>) {
        let c = 0.5 * (min + max);
        let (tx, ty) = self.layout.tile_at(c.x, c.y);
        if self.layout.within(tx, ty, min.truncate(), max.truncate(), self.layout.reach) {
            let tile = self.tile(tx, ty);
            let start = out.len();
            tile.obstacles.query_candidates(min, max, mask, out);
            // Local indices ascend with global ids (tiles store obstacles sorted by id).
            for id in &mut out[start..] {
                *id = tile.ids[*id as usize];
            }
            return;
        }
        self.candidates_multi(min, max, mask, out);
    }

    fn sphere_contact(&self, id: u32, center: DVec3, radius: f64, margin: f64) -> Option<SurfacePoint> {
        let (tile, i) = self.find(id, center);
        tile.obstacles.sphere_contact(i as u32, center, radius, margin).map(|mut s| {
            s.kind = global(s.kind, &tile.ids);
            s
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obstacles::{Obstacle, ObstacleShape};
    use autonomousim_core::math::Pose;

    /// Terrain `h(x, y)` sampled on 1 m vertices; a pillar every 10 m (anchor at the cell's
    /// centre), each with a global id from its position.
    struct Analytic {
        layout: TileLayout,
    }

    fn h(x: f64, y: f64) -> f64 {
        3.0 * (x / 17.0).sin() * (y / 23.0).cos() + 0.02 * x
    }

    const PILLAR_SPACING: f64 = 10.0;
    const PILLARS: u32 = 100;

    fn pillar(i: u32, j: u32) -> Obstacle {
        let p = DVec2::new(-500.0 + (i as f64 + 0.5) * PILLAR_SPACING, -500.0 + (j as f64 + 0.5) * PILLAR_SPACING);
        Obstacle::solid(
            ObstacleShape::Cylinder { half_height: 5.0, radius: 1.5 },
            Pose::from_translation(p.extend(h(p.x, p.y) + 4.0)),
            MaterialId::ROCK,
        )
    }

    impl TileSource for Analytic {
        fn tile(&self, tx: u32, ty: u32) -> Tile {
            let (lo, hi) = self.layout.core(tx, ty);
            let ring = self.layout.ring;
            let n = ((hi.x - lo.x + 2.0 * ring) as usize) + 1;
            let origin = lo - ring;
            let heights = (0..n * n).map(|k| h(origin.x + (k % n) as f64, origin.y + (k / n) as f64) as f32).collect();
            let grid = HeightGrid::new(origin, 1.0, n, n, heights, vec![MaterialId::GRASS; (n - 1) * (n - 1)]);
            let mut obstacles = Vec::new();
            let mut ids = Vec::new();
            for j in 0..PILLARS {
                for i in 0..PILLARS {
                    let o = pillar(i, j);
                    let (a, b) = o.aabb();
                    let r = self.layout.reach;
                    if b.x >= lo.x - r && a.x <= hi.x + r && b.y >= lo.y - r && a.y <= hi.y + r {
                        obstacles.push(o);
                        ids.push(j * PILLARS + i);
                    }
                }
            }
            Tile { grid, obstacles: ObstacleSet::new(obstacles), ids }
        }

        fn owner(&self, id: u32) -> (u32, u32) {
            let p = pillar(id % PILLARS, id / PILLARS).pose.pos;
            self.layout.tile_at(p.x, p.y)
        }
    }

    fn layout() -> TileLayout {
        TileLayout {
            origin: DVec2::splat(-500.0),
            tile_size: 100.0,
            tiles: (10, 10),
            ring: 16.0,
            reach: 4.0,
            detail_range: 300.0,
        }
    }

    fn map(capacity: usize) -> TiledMap {
        let l = layout();
        let coarse = HeightGrid::from_fn(DVec2::splat(-500.0), 10.0, 101, 101, h, |_, _| MaterialId::GRASS);
        TiledMap::new(l, coarse, 6.0, Box::new(Analytic { layout: l }), capacity)
    }

    /// The same terrain and pillars in one grid and one set.
    fn monolithic() -> (HeightGrid, ObstacleSet) {
        let grid = HeightGrid::from_fn(
            DVec2::splat(-500.0),
            1.0,
            1001,
            1001,
            |x, y| h(x, y) as f32 as f64,
            |_, _| MaterialId::GRASS,
        );
        let set = ObstacleSet::new((0..PILLARS * PILLARS).map(|k| pillar(k % PILLARS, k / PILLARS)).collect());
        (grid, set)
    }

    #[test]
    fn queries_match_a_single_grid() {
        let m = map(100);
        let (grid, set) = monolithic();
        let mut rng = autonomousim_core::rng::Seed::from_u64(3).rng();
        for _ in 0..3000 {
            let (x, y) = (rng.range(-499.0, 499.0), rng.range(-499.0, 499.0));
            // Tiles have their own grid origins: equal up to rounding.
            assert!((Terrain::height(&m, x, y) - grid.height(x, y)).abs() < 1e-9);
            let ((ha, na), (hb, nb)) = (m.height_normal(x, y), grid.height_normal(x, y));
            assert!((ha - hb).abs() < 1e-9 && (na - nb).length() < 1e-9);
            let p = DVec3::new(x, y, grid.height(x, y) + rng.range(-1.0, 6.0));
            let (a, b) = (m.closest_point(p, 5.0), grid.closest_point(p, 5.0));
            assert_eq!(a.is_some(), b.is_some(), "{p}");
            if let (Some(a), Some(b)) = (a, b) {
                assert!((a.point - b.point).length() < 1e-6 && (a.distance - b.distance).abs() < 1e-9, "{p}");
            }
            let md = rng.range(0.5, 12.0);
            let d = m.nearest_distance(p, md, HitMask::SOLID);
            let e = set.nearest_distance(p, md, HitMask::SOLID);
            assert_eq!(d.map(|d| (d * 1e9).round()), e.map(|d| (d * 1e9).round()));
            let r = rng.range(0.2, 9.0);
            let (mut u, mut v) = (Vec::new(), Vec::new());
            m.sphere_contacts(p, r, 0.1, HitMask::ALL, &mut u);
            set.sphere_contacts(p, r, 0.1, HitMask::ALL, &mut v);
            assert_eq!(u.len(), v.len());
            for (a, b) in u.iter().zip(&v) {
                assert_eq!(a.point, b.point);
                // The monolithic set's index is the global id here.
                assert_eq!(a.kind, b.kind);
            }
            let (lo, hi) = (p - rng.range(0.1, 20.0), p + rng.range(0.1, 20.0));
            let (mut u, mut v) = (Vec::new(), Vec::new());
            m.query_candidates(lo, hi, HitMask::ALL, &mut u);
            set.query_candidates(lo, hi, HitMask::ALL, &mut v);
            assert_eq!(u, v);
            // Rays of all lengths and directions, from above the ground.
            let dir = DVec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-0.4, 0.1)).normalize();
            let ray = Ray::new(p + DVec3::Z * 3.0, dir);
            let len = rng.range(1.0, 250.0);
            let (a, b) = (Terrain::raycast(&m, &ray, len, HitMask::ALL), grid.raycast(&ray, len, HitMask::ALL));
            assert_eq!(a.is_some(), b.is_some(), "{ray:?} {len}");
            if let (Some(a), Some(b)) = (a, b) {
                assert!((a.toi - b.toi).abs() < 1e-6, "{} {}", a.toi, b.toi);
            }
            let (a, b) = (StaticGeometry::raycast(&m, &ray, len, HitMask::ALL), set.raycast(&ray, len, HitMask::ALL));
            assert_eq!(a.map(|h| h.kind), b.map(|h| h.kind));
        }
        let (lo, hi) = m.height_bounds(DVec2::new(-120.0, -80.0), DVec2::new(130.0, 10.0));
        let (glo, ghi) = grid.height_bounds(DVec2::new(-120.0, -80.0), DVec2::new(130.0, 10.0));
        assert!(lo <= glo && hi >= ghi);
    }

    #[test]
    fn cache_stays_bounded_and_answers_do_not_change() {
        let m = map(3);
        let first: Vec<f64> = (0..10).map(|i| Terrain::height(&m, -450.0 + 100.0 * i as f64, 7.0)).collect();
        assert!(m.loaded() <= 3);
        assert!(m.generated() >= 10);
        let again: Vec<f64> = (0..10).rev().map(|i| Terrain::height(&m, -450.0 + 100.0 * i as f64, 7.0)).collect();
        assert_eq!(first, again.into_iter().rev().collect::<Vec<_>>());
        assert!(m.loaded() <= 3);
    }

    #[test]
    fn threads_see_the_same_tiles() {
        let m = map(4);
        let probe = |m: &TiledMap| -> Vec<f64> {
            (0..200).map(|i| Terrain::height(m, -490.0 + 4.9 * i as f64, -490.0 + 3.7 * i as f64)).collect()
        };
        let reference = probe(&map(100));
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..6).map(|_| s.spawn(|| probe(&m))).collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), reference);
            }
        });
    }

    #[test]
    fn long_rays_continue_on_the_coarse_grid() {
        let m = map(8);
        let ray = Ray::new(DVec3::new(-490.0, 0.0, 40.0), DVec3::new(1.0, 0.0, -0.05).normalize());
        let hit = Terrain::raycast(&m, &ray, 2000.0, HitMask::TERRAIN).expect("hits the ground");
        assert!(hit.toi > 300.0);
        assert!((hit.point.z - Terrain::height(&m, hit.point.x, hit.point.y)).abs() < 2.0);
    }
}
