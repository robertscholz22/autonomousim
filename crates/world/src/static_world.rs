//! A complete static map: terrain with water, obstacles, roads, the material table and metadata.

use crate::geodesy::GeoOrigin;
use crate::heightgrid::HeightGrid;
use crate::obstacles::ObstacleSet;
use crate::roads::RoadNetwork;
use autonomousim_core::geometry::{HitMask, Ray, RayHit, StaticGeometry};
use autonomousim_core::material::{Material, MaterialId, MaterialTable};
use autonomousim_core::terrain::Terrain;
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

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

/// Immutable map shared (by `Arc`) between all environments that use it.
///
/// Contacts and sensors take the terrain ([`Terrain`]) and obstacles ([`StaticGeometry`])
/// separately; the combined queries here are for sensors and scenario sampling.
#[derive(Clone, Debug)]
pub struct StaticWorld {
    pub meta: MapMeta,
    terrain: HeightGrid,
    obstacles: ObstacleSet,
    materials: MaterialTable,
    roads: RoadNetwork,
}

impl StaticWorld {
    pub fn new(meta: MapMeta, terrain: HeightGrid, obstacles: ObstacleSet, materials: MaterialTable) -> Self {
        Self { meta, terrain, obstacles, materials, roads: RoadNetwork::default() }
    }

    /// The same map with a road network.
    pub fn with_roads(mut self, roads: RoadNetwork) -> Self {
        self.roads = roads;
        self
    }

    /// The road network (empty for maps without roads).
    pub fn roads(&self) -> &RoadNetwork {
        &self.roads
    }

    pub fn terrain(&self) -> &HeightGrid {
        &self.terrain
    }

    pub fn obstacles(&self) -> &ObstacleSet {
        &self.obstacles
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
        let c = w.terrain().height(0.0, 0.0);
        assert!(c < -4.0);
        assert!((w.surface_height(0.0, 0.0) + 1.0).abs() < 1e-6);
        assert!(!w.is_free(DVec3::new(0.0, 0.0, -0.8), 0.5, false, true));
        assert!(w.is_free(DVec3::new(0.0, 0.0, -0.8), 0.5, false, false));
        let hit = w.raycast(&Ray::new(DVec3::new(0.0, 0.0, 10.0), -DVec3::Z), 100.0, HitMask::ALL).unwrap();
        assert_eq!(hit.kind, HitKind::Water);
        assert!((hit.point.z + 1.0).abs() < 1e-6);
    }
}
