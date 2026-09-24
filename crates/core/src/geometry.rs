//! Query types shared by all static-world geometry (terrain, water, obstacles) and the
//! [`StaticGeometry`] interface for obstacles.

use crate::material::MaterialId;
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::ops::BitOr;

/// Half-line `origin + t · dir`, `t ≥ 0`, with unit `dir` (so `t` is a distance).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: DVec3,
    pub dir: DVec3,
}

impl Ray {
    /// Ray with a normalised direction.
    pub fn new(origin: DVec3, dir: DVec3) -> Self {
        Self { origin, dir: dir.normalize() }
    }

    #[inline]
    pub fn at(&self, t: f64) -> DVec3 {
        self.origin + self.dir * t
    }
}

/// What a ray or contact query hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HitKind {
    Terrain,
    Water,
    /// Solid obstacle with its index.
    Solid(u32),
    /// Penetrable foliage (tree canopy, bush) with its obstacle index.
    Foliage(u32),
    /// Another agent (resolved by the simulation, not by static geometry).
    Agent(u32),
}

/// Set of [`HitKind`] classes a query considers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HitMask(pub u8);

impl HitMask {
    pub const NONE: Self = Self(0);
    pub const TERRAIN: Self = Self(1);
    pub const WATER: Self = Self(2);
    pub const SOLID: Self = Self(4);
    pub const FOLIAGE: Self = Self(8);
    pub const AGENTS: Self = Self(16);
    pub const ALL: Self = Self(31);
    /// Everything a physical body can rest on or collide with.
    pub const COLLIDABLE: Self = Self(1 | 4);

    #[inline]
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// The mask bit corresponding to a hit kind.
    pub fn of(kind: HitKind) -> Self {
        match kind {
            HitKind::Terrain => Self::TERRAIN,
            HitKind::Water => Self::WATER,
            HitKind::Solid(_) => Self::SOLID,
            HitKind::Foliage(_) => Self::FOLIAGE,
            HitKind::Agent(_) => Self::AGENTS,
        }
    }
}

impl BitOr for HitMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// First intersection of a ray with the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Distance along the ray.
    pub toi: f64,
    pub point: DVec3,
    /// Unit surface normal facing the ray origin's side.
    pub normal: DVec3,
    pub material: MaterialId,
    pub kind: HitKind,
}

impl RayHit {
    /// The closer of two optional hits.
    #[inline]
    pub fn closest(a: Option<RayHit>, b: Option<RayHit>) -> Option<RayHit> {
        match (a, b) {
            (Some(a), Some(b)) => Some(if b.toi < a.toi { b } else { a }),
            (a, None) => a,
            (None, b) => b,
        }
    }
}

/// Closest surface point to a query point (or to a sphere's centre).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePoint {
    /// Point on the surface.
    pub point: DVec3,
    /// Unit normal pointing out of the solid, towards the free side of the query point.
    pub normal: DVec3,
    /// Signed distance of the query point from the surface along `normal` (negative when the
    /// point lies inside the solid).
    pub distance: f64,
    pub material: MaterialId,
    pub kind: HitKind,
}

/// Static obstacles (rocks, trunks, canopies, buildings) as seen by contacts and sensors.
pub trait StaticGeometry: Send + Sync {
    /// First hit among obstacles whose class is in `mask`.
    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit>;

    /// For every obstacle in `mask` whose surface is within `radius + margin` of `center`,
    /// append the closest surface point (`distance` is measured from the centre, so the sphere
    /// penetrates when `distance < radius`).
    fn sphere_contacts(&self, center: DVec3, radius: f64, margin: f64, mask: HitMask, out: &mut Vec<SurfacePoint>);

    /// Distance from `p` to the nearest obstacle in `mask` (0 inside), if below `max_dist`.
    fn nearest_distance(&self, p: DVec3, max_dist: f64, mask: HitMask) -> Option<f64>;

    /// Broadphase: ids of the obstacles in `mask` whose bounds overlap `[min, max]`, appended
    /// in ascending order.
    fn query_candidates(&self, min: DVec3, max: DVec3, mask: HitMask, out: &mut Vec<u32>);

    /// Narrowphase for one candidate: the closest surface point if the sphere penetrates it
    /// or comes within `margin` (same convention as [`sphere_contacts`](Self::sphere_contacts)).
    fn sphere_contact(&self, id: u32, center: DVec3, radius: f64, margin: f64) -> Option<SurfacePoint>;
}

/// A world without obstacles.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoObstacles;

impl StaticGeometry for NoObstacles {
    fn raycast(&self, _: &Ray, _: f64, _: HitMask) -> Option<RayHit> {
        None
    }

    fn sphere_contacts(&self, _: DVec3, _: f64, _: f64, _: HitMask, _: &mut Vec<SurfacePoint>) {}

    fn nearest_distance(&self, _: DVec3, _: f64, _: HitMask) -> Option<f64> {
        None
    }

    fn query_candidates(&self, _: DVec3, _: DVec3, _: HitMask, _: &mut Vec<u32>) {}

    fn sphere_contact(&self, _: u32, _: DVec3, _: f64, _: f64) -> Option<SurfacePoint> {
        None
    }
}

/// Closest point to `p` on triangle `abc` (Ericson, *Real-Time Collision Detection* §5.1.5).
pub fn closest_point_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_point_regions() {
        let (a, b, c) = (DVec3::ZERO, DVec3::X, DVec3::Y);
        let cases = [
            (DVec3::new(0.2, 0.2, 1.0), DVec3::new(0.2, 0.2, 0.0)),  // face
            (DVec3::new(-1.0, -1.0, 0.0), a),                        // vertex a
            (DVec3::new(2.0, -0.5, 0.3), b),                         // vertex b
            (DVec3::new(-0.5, 3.0, 0.0), c),                         // vertex c
            (DVec3::new(0.5, -1.0, 0.0), DVec3::new(0.5, 0.0, 0.0)), // edge ab
            (DVec3::new(1.0, 1.0, 0.0), DVec3::new(0.5, 0.5, 0.0)),  // edge bc
            (DVec3::new(-1.0, 0.5, 2.0), DVec3::new(0.0, 0.5, 0.0)), // edge ca
        ];
        for (p, want) in cases {
            let got = closest_point_on_triangle(p, a, b, c);
            assert!((got - want).length() < 1e-12, "{p} -> {got}, want {want}");
        }
    }

    #[test]
    fn masks() {
        let m = HitMask::TERRAIN | HitMask::SOLID;
        assert!(m.contains(HitMask::TERRAIN));
        assert!(!m.contains(HitMask::FOLIAGE));
        assert!(m.intersects(HitMask::of(HitKind::Solid(3))));
        assert_eq!(m, HitMask::COLLIDABLE);
    }
}
