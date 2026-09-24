//! The [`Terrain`] interface (ground height field plus water) used by contacts, tyres and
//! sensors, and two analytic implementations for tests.

use crate::geometry::{HitKind, HitMask, Ray, RayHit, SurfacePoint};
use crate::material::MaterialId;
use glam::{DVec2, DVec3};

/// Ground surface `z = h(x, y)` with optional water bodies.
pub trait Terrain: Send + Sync {
    /// Horizontal extent `(min, max)` in which the terrain is defined. Queries outside clamp
    /// to the border.
    fn extent(&self) -> (DVec2, DVec2);

    /// Ground height at `(x, y)`.
    fn height(&self, x: f64, y: f64) -> f64;

    /// Ground height and upward unit normal at `(x, y)`.
    fn height_normal(&self, x: f64, y: f64) -> (f64, DVec3);

    /// Ground material at `(x, y)`.
    fn material(&self, x: f64, y: f64) -> MaterialId;

    /// Water surface height at `(x, y)` if it lies over water.
    fn water_level(&self, x: f64, y: f64) -> Option<f64>;

    /// Conservative `(min, max)` of the surface (ground and water) over the rectangle.
    fn height_bounds(&self, min: DVec2, max: DVec2) -> (f64, f64);

    /// Closest ground point to `p` if it is within `max_dist` (or if `p` is below ground).
    fn closest_point(&self, p: DVec3, max_dist: f64) -> Option<SurfacePoint>;

    /// First ground (`HitMask::TERRAIN`) or water (`HitMask::WATER`) hit along the ray.
    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit>;
}

/// Infinite horizontal plane at `height`.
#[derive(Clone, Copy, Debug)]
pub struct FlatTerrain {
    pub height: f64,
    pub material: MaterialId,
}

impl FlatTerrain {
    pub fn new(height: f64, material: MaterialId) -> Self {
        Self { height, material }
    }
}

impl Terrain for FlatTerrain {
    fn extent(&self) -> (DVec2, DVec2) {
        (DVec2::splat(f64::NEG_INFINITY), DVec2::splat(f64::INFINITY))
    }

    fn height(&self, _: f64, _: f64) -> f64 {
        self.height
    }

    fn height_normal(&self, _: f64, _: f64) -> (f64, DVec3) {
        (self.height, DVec3::Z)
    }

    fn material(&self, _: f64, _: f64) -> MaterialId {
        self.material
    }

    fn water_level(&self, _: f64, _: f64) -> Option<f64> {
        None
    }

    fn height_bounds(&self, _: DVec2, _: DVec2) -> (f64, f64) {
        (self.height, self.height)
    }

    fn closest_point(&self, p: DVec3, max_dist: f64) -> Option<SurfacePoint> {
        PlaneTerrain::new(DVec3::new(0.0, 0.0, self.height), DVec3::Z, self.material).closest_point(p, max_dist)
    }

    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        PlaneTerrain::new(DVec3::new(0.0, 0.0, self.height), DVec3::Z, self.material).raycast(ray, max_toi, mask)
    }
}

/// Infinite inclined plane through `point` with upward unit `normal` (`normal.z > 0`).
#[derive(Clone, Copy, Debug)]
pub struct PlaneTerrain {
    pub point: DVec3,
    pub normal: DVec3,
    pub material: MaterialId,
}

impl PlaneTerrain {
    pub fn new(point: DVec3, normal: DVec3, material: MaterialId) -> Self {
        let normal = normal.normalize();
        assert!(normal.z > 1e-6, "terrain plane normal must point upwards");
        Self { point, normal, material }
    }

    /// Plane through the origin rising along +x with slope angle `angle` (radians).
    pub fn incline(angle: f64, material: MaterialId) -> Self {
        Self::new(DVec3::ZERO, DVec3::new(-angle.sin(), 0.0, angle.cos()), material)
    }
}

impl Terrain for PlaneTerrain {
    fn extent(&self) -> (DVec2, DVec2) {
        (DVec2::splat(f64::NEG_INFINITY), DVec2::splat(f64::INFINITY))
    }

    fn height(&self, x: f64, y: f64) -> f64 {
        let n = self.normal;
        self.point.z - (n.x * (x - self.point.x) + n.y * (y - self.point.y)) / n.z
    }

    fn height_normal(&self, x: f64, y: f64) -> (f64, DVec3) {
        (self.height(x, y), self.normal)
    }

    fn material(&self, _: f64, _: f64) -> MaterialId {
        self.material
    }

    fn water_level(&self, _: f64, _: f64) -> Option<f64> {
        None
    }

    fn height_bounds(&self, min: DVec2, max: DVec2) -> (f64, f64) {
        let c = [
            self.height(min.x, min.y),
            self.height(max.x, min.y),
            self.height(min.x, max.y),
            self.height(max.x, max.y),
        ];
        (c.iter().copied().fold(f64::INFINITY, f64::min), c.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    }

    fn closest_point(&self, p: DVec3, max_dist: f64) -> Option<SurfacePoint> {
        let d = self.normal.dot(p - self.point);
        (d <= max_dist).then(|| SurfacePoint {
            point: p - self.normal * d,
            normal: self.normal,
            distance: d,
            material: self.material,
            kind: HitKind::Terrain,
        })
    }

    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        if !mask.contains(HitMask::TERRAIN) {
            return None;
        }
        let d0 = self.normal.dot(ray.origin - self.point);
        let t = if d0 <= 0.0 {
            0.0
        } else {
            let denom = self.normal.dot(ray.dir);
            if denom >= 0.0 {
                return None;
            }
            -d0 / denom
        };
        (t <= max_toi).then(|| RayHit {
            toi: t,
            point: ray.at(t),
            normal: self.normal,
            material: self.material,
            kind: HitKind::Terrain,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incline_geometry() {
        let angle = 0.3;
        let t = PlaneTerrain::incline(angle, MaterialId::GRASS);
        assert!((t.height(2.0, 5.0) - 2.0 * angle.tan()).abs() < 1e-12);
        let p = DVec3::new(1.0, 0.0, 2.0);
        let s = t.closest_point(p, 10.0).unwrap();
        assert!((t.height(s.point.x, s.point.y) - s.point.z).abs() < 1e-12);
        assert!(((p - s.point).length() - s.distance).abs() < 1e-12);

        let hit = t.raycast(&Ray::new(DVec3::new(3.0, 1.0, 10.0), -DVec3::Z), 100.0, HitMask::ALL).unwrap();
        assert!((hit.point.z - 3.0 * angle.tan()).abs() < 1e-12);
        assert!(t.raycast(&Ray::new(DVec3::new(3.0, 1.0, 10.0), DVec3::Z), 100.0, HitMask::ALL).is_none());
        assert!(t.raycast(&Ray::new(DVec3::new(3.0, 1.0, 10.0), -DVec3::Z), 100.0, HitMask::WATER).is_none());
    }

    #[test]
    fn flat_below_ground_is_reported() {
        let t = FlatTerrain::new(1.0, MaterialId::ROCK);
        let s = t.closest_point(DVec3::new(0.0, 0.0, 0.8), 0.0).unwrap();
        assert!((s.distance + 0.2).abs() < 1e-12);
        let hit = t.raycast(&Ray::new(DVec3::new(0.0, 0.0, 0.5), DVec3::X), 10.0, HitMask::ALL).unwrap();
        assert_eq!(hit.toi, 0.0);
    }
}
