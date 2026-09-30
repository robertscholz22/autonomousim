//! A complete static map: terrain with water, obstacles, roads, the material table and metadata.

use crate::geodesy::GeoOrigin;
use crate::heightgrid::HeightGrid;
use crate::obstacles::ObstacleSet;
use crate::roads::RoadNetwork;
use crate::sites::Sites;
use crate::tiles::TiledMap;
use autonomousim_core::geometry::{HitMask, Ray, RayHit, StaticGeometry, SurfacePoint};
use autonomousim_core::material::{Material, MaterialId, MaterialTable};
use autonomousim_core::terrain::Terrain;
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Provenance and georeference of a map.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapMeta {
    pub name: String,
    /// Generator that produced the map (`"wild"`, `"testworld/flat"`, …).
    pub generator: String,
    pub generator_version: u32,
    pub seed: u64,
    pub geo_origin: GeoOrigin,
}

impl MapMeta {
    pub fn new(name: &str, generator: &str, seed: u64) -> Self {
        Self {
            name: name.to_owned(),
            generator: generator.to_owned(),
            generator_version: 1,
            seed,
            geo_origin: GeoOrigin::default(),
        }
    }
}

/// The terrain of a map: one height grid, or tiles generated on demand.
#[derive(Clone, Debug)]
pub enum MapTerrain {
    Grid(HeightGrid),
    Tiled(Arc<TiledMap>),
}

impl MapTerrain {
    /// The height grid of a single-grid map.
    pub fn grid(&self) -> Option<&HeightGrid> {
        match self {
            Self::Grid(g) => Some(g),
            Self::Tiled(_) => None,
        }
    }

    /// The tiles of a tiled map.
    pub fn tiled(&self) -> Option<&Arc<TiledMap>> {
        match self {
            Self::Grid(_) => None,
            Self::Tiled(t) => Some(t),
        }
    }

    /// Lowest ground and highest surface over the whole map (the coarse layer's, widened by
    /// its bound on the tiles, for tiled maps).
    pub fn height_range(&self) -> (f64, f64) {
        match self {
            Self::Grid(g) => g.height_range(),
            Self::Tiled(t) => {
                let (lo, hi) = t.coarse().height_range();
                (lo - t.pad(), hi + t.pad())
            }
        }
    }
}

macro_rules! terrain {
    ($self:ident, $t:ident => $e:expr) => {
        match $self {
            MapTerrain::Grid($t) => $e,
            MapTerrain::Tiled($t) => $e,
        }
    };
}

impl Terrain for MapTerrain {
    fn extent(&self) -> (DVec2, DVec2) {
        terrain!(self, t => t.extent())
    }
    #[inline]
    fn height(&self, x: f64, y: f64) -> f64 {
        terrain!(self, t => t.height(x, y))
    }
    #[inline]
    fn height_normal(&self, x: f64, y: f64) -> (f64, DVec3) {
        terrain!(self, t => t.height_normal(x, y))
    }
    #[inline]
    fn material(&self, x: f64, y: f64) -> MaterialId {
        terrain!(self, t => t.material(x, y))
    }
    #[inline]
    fn water_level(&self, x: f64, y: f64) -> Option<f64> {
        terrain!(self, t => t.water_level(x, y))
    }
    fn height_bounds(&self, min: DVec2, max: DVec2) -> (f64, f64) {
        terrain!(self, t => t.height_bounds(min, max))
    }
    fn closest_point(&self, p: DVec3, max_dist: f64) -> Option<SurfacePoint> {
        terrain!(self, t => t.closest_point(p, max_dist))
    }
    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        match self {
            Self::Grid(g) => g.raycast(ray, max_toi, mask),
            Self::Tiled(t) => Terrain::raycast(&**t, ray, max_toi, mask),
        }
    }
}

/// The obstacles of a map: one set, or those of its tiles.
#[derive(Clone, Debug)]
pub enum MapObstacles {
    Set(ObstacleSet),
    Tiled(Arc<TiledMap>),
}

impl MapObstacles {
    /// The obstacle set of a single-grid map.
    pub fn set(&self) -> Option<&ObstacleSet> {
        match self {
            Self::Set(s) => Some(s),
            Self::Tiled(_) => None,
        }
    }
}

macro_rules! obstacles {
    ($self:ident, $o:ident => $e:expr) => {
        match $self {
            MapObstacles::Set($o) => $e,
            MapObstacles::Tiled($o) => $e,
        }
    };
}

impl StaticGeometry for MapObstacles {
    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        match self {
            Self::Set(s) => s.raycast(ray, max_toi, mask),
            Self::Tiled(t) => StaticGeometry::raycast(&**t, ray, max_toi, mask),
        }
    }
    fn sphere_contacts(&self, center: DVec3, radius: f64, margin: f64, mask: HitMask, out: &mut Vec<SurfacePoint>) {
        obstacles!(self, o => o.sphere_contacts(center, radius, margin, mask, out))
    }
    fn nearest_distance(&self, p: DVec3, max_dist: f64, mask: HitMask) -> Option<f64> {
        obstacles!(self, o => o.nearest_distance(p, max_dist, mask))
    }
    fn query_candidates(&self, min: DVec3, max: DVec3, mask: HitMask, out: &mut Vec<u32>) {
        obstacles!(self, o => o.query_candidates(min, max, mask, out))
    }
    fn sphere_contact(&self, id: u32, center: DVec3, radius: f64, margin: f64) -> Option<SurfacePoint> {
        obstacles!(self, o => o.sphere_contact(id, center, radius, margin))
    }
}

/// Immutable map shared (by `Arc`) between all environments that use it.
///
/// Contacts and sensors take the terrain ([`Terrain`]) and obstacles ([`StaticGeometry`])
/// separately; the combined queries here are for sensors and scenario sampling.
#[derive(Clone, Debug)]
pub struct StaticWorld {
    pub meta: MapMeta,
    terrain: MapTerrain,
    obstacles: MapObstacles,
    materials: MaterialTable,
    roads: RoadNetwork,
    /// Lots, buildings, pads and bays (urban maps).
    sites: Arc<Sites>,
    /// Content hash of a tiled map (fixed by its generator; grids are hashed from content).
    tiled_hash: Option<crate::MapHash>,
}

impl StaticWorld {
    pub fn new(meta: MapMeta, terrain: HeightGrid, obstacles: ObstacleSet, materials: MaterialTable) -> Self {
        Self {
            meta,
            terrain: MapTerrain::Grid(terrain),
            obstacles: MapObstacles::Set(obstacles),
            materials,
            roads: RoadNetwork::default(),
            sites: Arc::default(),
            tiled_hash: None,
        }
    }

    /// A tiled map; `hash` identifies its content (the generator's inputs).
    pub fn tiled(meta: MapMeta, tiles: Arc<TiledMap>, materials: MaterialTable, hash: crate::MapHash) -> Self {
        Self {
            meta,
            terrain: MapTerrain::Tiled(tiles.clone()),
            obstacles: MapObstacles::Tiled(tiles),
            materials,
            roads: RoadNetwork::default(),
            sites: Arc::default(),
            tiled_hash: Some(hash),
        }
    }

    /// Content hash of a tiled map (`None` for single-grid maps, see
    /// [`content_hash`](Self::content_hash)).
    pub fn tiled_hash(&self) -> Option<crate::MapHash> {
        self.tiled_hash
    }

    /// Whether the map is made of tiles.
    pub fn is_tiled(&self) -> bool {
        matches!(self.terrain, MapTerrain::Tiled(_))
    }

    /// The same map with a road network.
    pub fn with_roads(mut self, roads: RoadNetwork) -> Self {
        self.roads = roads;
        self
    }

    /// The same map with lots, buildings, pads and bays.
    pub fn with_sites(mut self, sites: Sites) -> Self {
        self.sites = Arc::new(sites);
        self
    }

    /// Lots, buildings, pads and bays (empty for maps without).
    pub fn sites(&self) -> &Sites {
        &self.sites
    }

    /// The road network (empty for maps without roads).
    pub fn roads(&self) -> &RoadNetwork {
        &self.roads
    }

    pub fn terrain(&self) -> &MapTerrain {
        &self.terrain
    }

    /// The height grid of a single-grid map.
    ///
    /// # Panics
    /// For tiled maps.
    pub fn grid(&self) -> &HeightGrid {
        self.terrain.grid().expect("a single-grid map (tiled maps have no single height grid)")
    }

    pub fn obstacles(&self) -> &MapObstacles {
        &self.obstacles
    }

    /// The obstacle set of a single-grid map.
    ///
    /// # Panics
    /// For tiled maps.
    pub fn obstacle_set(&self) -> &ObstacleSet {
        self.obstacles.set().expect("a single-grid map (tiled maps keep obstacles per tile)")
    }

    pub fn materials(&self) -> &MaterialTable {
        &self.materials
    }

    /// Properties of a material (falls back to rock for ids beyond the table).
    pub fn material(&self, id: MaterialId) -> &Material {
        self.materials.get(if (id.0 as usize) < self.materials.len() { id } else { MaterialId::ROCK })
    }

    /// Horizontal extent of the map.
    pub fn extent(&self) -> (DVec2, DVec2) {
        self.terrain.extent()
    }

    /// Whether `(x, y)` lies inside the map extent.
    pub fn contains_xy(&self, x: f64, y: f64) -> bool {
        let (lo, hi) = self.extent();
        x >= lo.x && x <= hi.x && y >= lo.y && y <= hi.y
    }

    /// First hit among terrain, water and obstacles in `mask`.
    pub fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        let ground = self.terrain.raycast(ray, max_toi, mask);
        let max_toi = ground.map_or(max_toi, |h| h.toi);
        RayHit::closest(ground, self.obstacles.raycast(ray, max_toi, mask))
    }

    /// Distance from `p` to the nearest collidable surface (terrain or solid obstacle), capped
    /// at `max_dist`; 0 when `p` is inside either.
    pub fn clearance(&self, p: DVec3, max_dist: f64) -> f64 {
        let terrain = self.terrain.closest_point(p, max_dist).map_or(max_dist, |s| s.distance.max(0.0));
        let obstacle = self.obstacles.nearest_distance(p, max_dist, HitMask::SOLID).unwrap_or(max_dist);
        terrain.min(obstacle).min(max_dist)
    }

    /// Distance from `p` to the nearest solid obstacle (terrain left out), capped at
    /// `max_dist`: the clearance of ground vehicles, which always sit on the terrain.
    pub fn obstacle_clearance(&self, p: DVec3, max_dist: f64) -> f64 {
        self.obstacles.nearest_distance(p, max_dist, HitMask::SOLID).unwrap_or(max_dist).min(max_dist)
    }

    /// Whether a sphere at `p` is free of terrain, solid obstacles and (optionally) foliage
    /// and water.
    pub fn is_free(&self, p: DVec3, radius: f64, avoid_foliage: bool, avoid_water: bool) -> bool {
        if self.clearance(p, radius) < radius {
            return false;
        }
        if avoid_foliage && self.obstacles.nearest_distance(p, radius, HitMask::FOLIAGE).is_some_and(|d| d < radius) {
            return false;
        }
        !(avoid_water && self.terrain.water_level(p.x, p.y).is_some_and(|w| p.z - radius < w))
    }

    /// Ground or water surface height (whichever is higher) at `(x, y)`.
    pub fn surface_height(&self, x: f64, y: f64) -> f64 {
        let h = self.terrain.height(x, y);
        self.terrain.water_level(x, y).map_or(h, |w| w.max(h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testworlds;
    use autonomousim_core::geometry::HitKind;

    #[test]
    fn combined_raycast_picks_nearest() {
        let w = testworlds::single_tree();
        // Downward ray through the canopy: foliage first, then ground if foliage is masked.
        let ray = Ray::new(DVec3::new(0.5, 0.0, 20.0), -DVec3::Z);
        assert!(matches!(w.raycast(&ray, 100.0, HitMask::ALL).unwrap().kind, HitKind::Foliage(_)));
        let hit = w.raycast(&ray, 100.0, HitMask::TERRAIN | HitMask::SOLID).unwrap();
        assert_eq!(hit.kind, HitKind::Terrain);
        assert!((hit.point.z).abs() < 1e-9);
        // Horizontal ray at 1 m hits the trunk.
        let ray = Ray::new(DVec3::new(-5.0, 0.0, 1.0), DVec3::X);
        assert!(matches!(w.raycast(&ray, 100.0, HitMask::ALL).unwrap().kind, HitKind::Solid(_)));
        // A ray that misses everything but meets the ground far away.
        let ray = Ray::new(DVec3::new(-5.0, 5.0, 1.0), DVec3::new(0.0, 1.0, -0.1));
        assert_eq!(w.raycast(&ray, 100.0, HitMask::ALL).unwrap().kind, HitKind::Terrain);
    }

    #[test]
    fn clearance_and_free_space() {
        let w = testworlds::single_tree();
        assert!((w.clearance(DVec3::new(10.0, 10.0, 2.0), 50.0) - 2.0).abs() < 1e-9);
        assert!((w.clearance(DVec3::new(1.0, 0.0, 1.0), 50.0) - 0.75).abs() < 1e-9);
        assert!(w.is_free(DVec3::new(10.0, 10.0, 2.0), 0.5, true, true));
        assert!(!w.is_free(DVec3::new(10.0, 10.0, 0.3), 0.5, true, true));
        // Inside the canopy: free of solids but not of foliage.
        assert!(w.is_free(DVec3::new(0.0, 0.0, 7.0), 0.3, false, true));
        assert!(!w.is_free(DVec3::new(0.0, 0.0, 7.0), 0.3, true, true));
    }

    #[test]
    fn water_surface() {
        let w = testworlds::lake(200.0, 5.0, -1.0);
        let c = w.grid().height(0.0, 0.0);
        assert!(c < -4.0);
        assert!((w.surface_height(0.0, 0.0) + 1.0).abs() < 1e-6);
        assert!(!w.is_free(DVec3::new(0.0, 0.0, -0.8), 0.5, false, true));
        assert!(w.is_free(DVec3::new(0.0, 0.0, -0.8), 0.5, false, false));
        let hit = w.raycast(&Ray::new(DVec3::new(0.0, 0.0, 10.0), -DVec3::Z), 100.0, HitMask::ALL).unwrap();
        assert_eq!(hit.kind, HitKind::Water);
        assert!((hit.point.z + 1.0).abs() < 1e-6);
    }
}
