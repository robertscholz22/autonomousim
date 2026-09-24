//! Air data and ground proximity as seen by the rotor model.

use super::def::SEA_LEVEL_DENSITY;
use autonomousim_core::terrain::Terrain;
use glam::DVec3;

/// Ambient air at the vehicle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AirData {
    /// Density (kg/m³).
    pub density: f64,
    /// Air velocity in the world frame (m/s).
    pub wind: DVec3,
}

impl Default for AirData {
    fn default() -> Self {
        Self { density: SEA_LEVEL_DENSITY, wind: DVec3::ZERO }
    }
}

/// Local ground (or water surface) plane below the vehicle, for ground effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundPlane {
    pub point: DVec3,
    /// Unit normal pointing up, away from the ground.
    pub normal: DVec3,
}

impl GroundPlane {
    /// Tangent plane of the terrain (or the water surface where it is higher) below `p`, if `p`
    /// is less than `range` above it.
    pub fn below(terrain: &dyn Terrain, p: DVec3, range: f64) -> Option<Self> {
        let (h, n) = terrain.height_normal(p.x, p.y);
        let plane = match terrain.water_level(p.x, p.y) {
            Some(w) if w > h => Self { point: DVec3::new(p.x, p.y, w), normal: DVec3::Z },
            _ => Self { point: DVec3::new(p.x, p.y, h), normal: n },
        };
        (p.z - plane.point.z < range).then_some(plane)
    }

    /// Distance from `hub` to the plane along the downwash direction `−axis` (world frame);
    /// infinite when the rotor points away from the ground by more than ~78°.
    #[inline]
    pub fn distance_along(&self, hub: DVec3, axis: DVec3) -> f64 {
        let c = self.normal.dot(axis);
        if c < 0.2 { f64::INFINITY } else { self.normal.dot(hub - self.point) / c }
    }
}

/// Cheeseman–Bennett in-ground-effect thrust ratio `1 / (1 − (R/4z)²)` for a rotor of radius
/// `radius` at height `z`, with `z` clamped to at least `R/2` (ratio ≤ 4/3).
#[inline]
pub fn ground_effect(z: f64, radius: f64) -> f64 {
    let x = radius / (4.0 * z.max(0.5 * radius));
    1.0 / (1.0 - x * x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::material::MaterialId;
    use autonomousim_core::terrain::{FlatTerrain, PlaneTerrain};

    #[test]
    fn cheeseman_bennett_values() {
        assert!((ground_effect(1.0, 1.0) - 16.0 / 15.0).abs() < 1e-15);
        assert!((ground_effect(0.1, 1.0) - 4.0 / 3.0).abs() < 1e-15);
        assert!((ground_effect(f64::INFINITY, 1.0) - 1.0).abs() < 1e-15);
        assert!(ground_effect(2.0, 1.0) < ground_effect(1.5, 1.0));
    }

    #[test]
    fn plane_distance_follows_tilt_and_slope() {
        let flat = FlatTerrain::new(1.0, MaterialId::GRASS);
        let g = GroundPlane::below(&flat, DVec3::new(3.0, 4.0, 1.5), 2.0).unwrap();
        assert!((g.distance_along(DVec3::new(3.0, 4.0, 1.5), DVec3::Z) - 0.5).abs() < 1e-12);
        // Tilted 60°: the downwash travels twice as far.
        let tilted = DVec3::new(60f64.to_radians().sin(), 0.0, 0.5);
        assert!((g.distance_along(DVec3::new(3.0, 4.0, 1.5), tilted) - 1.0).abs() < 1e-12);
        assert!(g.distance_along(DVec3::new(0.0, 0.0, 1.5), DVec3::X).is_infinite());
        assert!(GroundPlane::below(&flat, DVec3::new(0.0, 0.0, 3.5), 2.0).is_none());

        // On an incline the perpendicular distance counts, not the vertical one.
        let slope = PlaneTerrain::incline(0.3, MaterialId::ROCK);
        let n = DVec3::new(-0.3f64.sin(), 0.0, 0.3f64.cos());
        let hub = n * 0.4;
        let g = GroundPlane::below(&slope, hub, 2.0).unwrap();
        assert!((g.distance_along(hub, n) - 0.4).abs() < 1e-9);
    }
}
