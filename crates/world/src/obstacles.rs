//! Static obstacles (rocks, tree trunks and canopies, buildings) in a parry BVH.

use autonomousim_core::geometry::{HitKind, HitMask, Ray, RayHit, StaticGeometry, SurfacePoint};
use autonomousim_core::material::MaterialId;
use autonomousim_core::math::Pose;
use autonomousim_core::parry::bounding_volume::Aabb;
use autonomousim_core::parry::math::Pose as PPose;
use autonomousim_core::parry::partitioning::{Bvh, BvhBuildStrategy};
use autonomousim_core::parry::query::Ray as PRay;
use autonomousim_core::parry::shape::SharedShape;
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// Obstacle tags shared by the generators, the viewer and statistics.
pub mod tags {
    pub const TRUNK: u16 = 1;
    /// Conifer crown (cone).
    pub const CANOPY: u16 = 2;
    pub const ROCK: u16 = 3;
    pub const PILLAR: u16 = 4;
    pub const WALL: u16 = 5;
    /// Broadleaf crown (sphere).
    pub const CANOPY_BROADLEAF: u16 = 6;
    /// Hedgerow: a foliage cuboid around a solid woody core (both tagged).
    pub const HEDGE: u16 = 7;
    pub const FENCE: u16 = 8;
    /// Farm building (cuboid) and silo (cylinder).
    pub const BUILDING: u16 = 9;
    pub const SILO: u16 = 10;
}

/// Obstacle geometry in its local frame. Axisymmetric shapes use the local `z` axis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObstacleShape {
    Sphere {
        radius: f64,
    },
    /// Segment from `-half_height` to `+half_height` along `z`, swept by `radius`.
    Capsule {
        half_height: f64,
        radius: f64,
    },
    Cylinder {
        half_height: f64,
        radius: f64,
    },
    /// Base disc at `z = -half_height`, apex at `z = +half_height`.
    Cone {
        half_height: f64,
        radius: f64,
    },
    Cuboid {
        half_extents: DVec3,
    },
    ConvexHull {
        points: Vec<DVec3>,
    },
}

impl ObstacleShape {
    /// The parry shape and the local rotation that maps its frame into ours.
    fn to_parry(&self) -> (SharedShape, DQuat) {
        // parry's axisymmetric shapes use the y axis; rotate y → z.
        let y_to_z = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
        match self {
            Self::Sphere { radius } => (SharedShape::ball(*radius), DQuat::IDENTITY),
            Self::Capsule { half_height, radius } => (SharedShape::capsule_y(*half_height, *radius), y_to_z),
            Self::Cylinder { half_height, radius } => (SharedShape::cylinder(*half_height, *radius), y_to_z),
            Self::Cone { half_height, radius } => (SharedShape::cone(*half_height, *radius), y_to_z),
            Self::Cuboid { half_extents } => {
                (SharedShape::cuboid(half_extents.x, half_extents.y, half_extents.z), DQuat::IDENTITY)
            }
            Self::ConvexHull { points } => {
                (SharedShape::convex_hull(points).expect("degenerate convex hull"), DQuat::IDENTITY)
            }
        }
    }

    /// Closed-form ray cast in the local frame (unit `d`): the first surface point at
    /// `0 ≤ t ≤ max_toi` and its outward normal. A `solid` shape containing the origin is hit
    /// at `t = 0` with a zero normal (as in parry); a hollow one is hit where the ray leaves.
    /// Convex hulls are left to parry.
    fn ray_cast_local(&self, o: DVec3, d: DVec3, max_toi: f64, solid: bool) -> LocalCast {
        let mut best = Best { t: max_toi, n: None };
        match *self {
            Self::Sphere { radius } => {
                if solid && o.length_squared() <= radius * radius {
                    return LocalCast::Hit(0.0, DVec3::ZERO);
                }
                sphere_roots(o, d, radius, |t, p| best.offer(t, p / radius));
            }
            Self::Capsule { half_height: h, radius: r } => {
                if solid && DVec3::new(o.x, o.y, o.z - o.z.clamp(-h, h)).length_squared() <= r * r {
                    return LocalCast::Hit(0.0, DVec3::ZERO);
                }
                lateral_roots(o, d, r, |t, p| {
                    if p.z.abs() <= h {
                        best.offer(t, DVec3::new(p.x, p.y, 0.0) / r);
                    }
                });
                for c in [h, -h] {
                    let centre = DVec3::new(0.0, 0.0, c);
                    sphere_roots(o - centre, d, r, |t, q| {
                        if q.z * c >= 0.0 {
                            best.offer(t, q / r);
                        }
                    });
                }
            }
            Self::Cylinder { half_height: h, radius: r } => {
                if solid && o.x * o.x + o.y * o.y <= r * r && o.z.abs() <= h {
                    return LocalCast::Hit(0.0, DVec3::ZERO);
                }
                lateral_roots(o, d, r, |t, p| {
                    if p.z.abs() <= h {
                        best.offer(t, DVec3::new(p.x, p.y, 0.0) / r);
                    }
                });
                if d.z != 0.0 {
                    for c in [h, -h] {
                        let t = (c - o.z) / d.z;
                        let p = o + d * t;
                        if p.x * p.x + p.y * p.y <= r * r {
                            best.offer(t, DVec3::new(0.0, 0.0, c.signum()));
                        }
                    }
                }
            }
            Self::Cone { half_height: h, radius } => {
                // Lateral surface x² + y² = k²(h − z)², −h ≤ z ≤ h; base disc at z = −h.
                let k2 = (radius / (2.0 * h)).powi(2);
                let w0 = h - o.z;
                if solid && (0.0..=2.0 * h).contains(&w0) && o.x * o.x + o.y * o.y <= k2 * w0 * w0 {
                    return LocalCast::Hit(0.0, DVec3::ZERO);
                }
                let a = d.x * d.x + d.y * d.y - k2 * d.z * d.z;
                let b = o.x * d.x + o.y * d.y + k2 * w0 * d.z;
                let c = o.x * o.x + o.y * o.y - k2 * w0 * w0;
                let mut lateral = |t: f64| {
                    let p = o + d * t;
                    if p.z.abs() <= h {
                        let n = DVec3::new(p.x, p.y, k2 * (h - p.z));
                        best.offer(t, if n.length_squared() > 0.0 { n.normalize() } else { DVec3::Z });
                    }
                };
                if a.abs() > 1e-12 {
                    let disc = b * b - a * c;
                    if disc >= 0.0 {
                        let s = disc.sqrt();
                        lateral((-b - s) / a);
                        lateral((-b + s) / a);
                    }
                } else if b != 0.0 {
                    lateral(-0.5 * c / b);
                }
                if d.z != 0.0 {
                    let t = (-h - o.z) / d.z;
                    let p = o + d * t;
                    if p.x * p.x + p.y * p.y <= radius * radius {
                        best.offer(t, DVec3::NEG_Z);
                    }
                }
            }
            Self::Cuboid { half_extents: e } => {
                let (mut t_in, mut t_out) = (f64::NEG_INFINITY, f64::INFINITY);
                let (mut n_in, mut n_out) = (DVec3::ZERO, DVec3::ZERO);
                for i in 0..3 {
                    if d[i] == 0.0 {
                        if o[i].abs() > e[i] {
                            return LocalCast::Miss;
                        }
                        continue;
                    }
                    let (t1, t2) = ((-e[i] - o[i]) / d[i], (e[i] - o[i]) / d[i]);
                    let (near, far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                    let mut axis = DVec3::ZERO;
                    axis[i] = d[i].signum();
                    if near > t_in {
                        (t_in, n_in) = (near, -axis);
                    }
                    if far < t_out {
                        (t_out, n_out) = (far, axis);
                    }
                }
                if t_in > t_out || t_out < 0.0 {
                    return LocalCast::Miss;
                }
                if t_in >= 0.0 {
                    best.offer(t_in, n_in);
                } else if solid {
                    return LocalCast::Hit(0.0, DVec3::ZERO);
                } else {
                    best.offer(t_out, n_out);
                }
            }
            Self::ConvexHull { .. } => return LocalCast::Unsupported,
        }
        match best.n {
            Some(n) => LocalCast::Hit(best.t, n),
            None => LocalCast::Miss,
        }
    }

    /// Cheap lower bound on the distance from the local point `p` to the shape (the distance
    /// to a supporting half-space or slab). Exact outside spheres and capsules; values ≤ 0
    /// only mean "may touch".
    fn distance_lower_bound(&self, p: DVec3) -> f64 {
        let r = (p.x * p.x + p.y * p.y).sqrt();
        match *self {
            Self::Sphere { radius } => p.length() - radius,
            Self::Capsule { half_height, radius } => {
                DVec3::new(p.x, p.y, p.z - p.z.clamp(-half_height, half_height)).length() - radius
            }
            Self::Cylinder { half_height, radius } => (r - radius).max(p.z.abs() - half_height),
            Self::Cone { half_height, radius } => {
                // Slant half-space through the base rim and the apex, in the meridian plane of p.
                let h2 = 2.0 * half_height;
                let slant = (h2 * r + radius * p.z - radius * half_height) / (h2 * h2 + radius * radius).sqrt();
                slant.max(-half_height - p.z)
            }
            Self::Cuboid { half_extents } => (p.abs() - half_extents).max_element(),
            Self::ConvexHull { .. } => f64::NEG_INFINITY,
        }
    }
}

/// Result of [`ObstacleShape::ray_cast_local`].
enum LocalCast {
    Unsupported,
    Miss,
    /// Time of impact and outward normal (local frame).
    Hit(f64, DVec3),
}

/// Smallest admissible time of impact seen so far.
struct Best {
    t: f64,
    n: Option<DVec3>,
}

impl Best {
    #[inline]
    fn offer(&mut self, t: f64, n: DVec3) {
        if t >= 0.0 && t <= self.t {
            self.t = t;
            self.n = Some(n);
        }
    }
}

/// Both intersections of the ray with the sphere of radius `r` at the origin, as `(t, point)`.
#[inline]
fn sphere_roots(o: DVec3, d: DVec3, r: f64, mut f: impl FnMut(f64, DVec3)) {
    let b = o.dot(d);
    let disc = b * b - (o.length_squared() - r * r);
    if disc >= 0.0 {
        let s = disc.sqrt();
        for t in [-b - s, -b + s] {
            f(t, o + d * t);
        }
    }
}

/// Both intersections of the ray with the infinite cylinder of radius `r` about `z`.
#[inline]
fn lateral_roots(o: DVec3, d: DVec3, r: f64, mut f: impl FnMut(f64, DVec3)) {
    let a = d.x * d.x + d.y * d.y;
    if a <= 1e-300 {
        return;
    }
    let b = o.x * d.x + o.y * d.y;
    let disc = b * b - a * (o.x * o.x + o.y * o.y - r * r);
    if disc >= 0.0 {
        let s = disc.sqrt();
        for t in [(-b - s) / a, (-b + s) / a] {
            f(t, o + d * t);
        }
    }
}

/// Whether an obstacle blocks bodies or only slows them and triggers the foliage event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObstacleClass {
    Solid,
    Foliage,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Obstacle {
    pub shape: ObstacleShape,
    pub pose: Pose,
    pub class: ObstacleClass,
    pub material: MaterialId,
    /// Free-form type tag (tree species, rock kind, …) for rendering and statistics.
    pub tag: u16,
}

impl Obstacle {
    pub fn solid(shape: ObstacleShape, pose: Pose, material: MaterialId) -> Self {
        Self { shape, pose, class: ObstacleClass::Solid, material, tag: 0 }
    }

    pub fn foliage(shape: ObstacleShape, pose: Pose) -> Self {
        Self { shape, pose, class: ObstacleClass::Foliage, material: MaterialId::FOLIAGE, tag: 0 }
    }

    pub fn with_tag(mut self, tag: u16) -> Self {
        self.tag = tag;
        self
    }
}

/// Immutable set of obstacles with a bounding-volume hierarchy.
#[derive(Clone)]
pub struct ObstacleSet {
    obstacles: Vec<Obstacle>,
    shapes: Vec<SharedShape>,
    poses: Vec<PPose>,
    aabbs: Vec<Aabb>,
    bvh: Bvh,
}

impl std::fmt::Debug for ObstacleSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObstacleSet").field("len", &self.obstacles.len()).finish()
    }
}

impl Default for ObstacleSet {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl ObstacleSet {
    pub fn new(obstacles: Vec<Obstacle>) -> Self {
        let (shapes, poses): (Vec<_>, Vec<_>) = obstacles
            .iter()
            .map(|o| {
                let (shape, local) = o.shape.to_parry();
                (shape, PPose::from_parts(o.pose.pos, o.pose.rot * local))
            })
            .unzip();
        let aabbs: Vec<Aabb> = shapes.iter().zip(&poses).map(|(s, p)| s.compute_aabb(p)).collect();
        let bvh = Bvh::from_leaves(BvhBuildStrategy::Binned, &aabbs);
        Self { obstacles, shapes, poses, aabbs, bvh }
    }

    pub fn obstacles(&self) -> &[Obstacle] {
        &self.obstacles
    }

    pub fn len(&self) -> usize {
        self.obstacles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.obstacles.is_empty()
    }

    #[inline]
    fn mask_of(&self, i: usize) -> HitMask {
        match self.obstacles[i].class {
            ObstacleClass::Solid => HitMask::SOLID,
            ObstacleClass::Foliage => HitMask::FOLIAGE,
        }
    }

    #[inline]
    fn kind_of(&self, i: usize) -> HitKind {
        match self.obstacles[i].class {
            ObstacleClass::Solid => HitKind::Solid(i as u32),
            ObstacleClass::Foliage => HitKind::Foliage(i as u32),
        }
    }

    /// Indices of obstacles whose bounding boxes overlap `[min, max]`, in ascending order.
    pub fn query_aabb(&self, min: DVec3, max: DVec3, out: &mut Vec<usize>) {
        let aabb = Aabb::new(min, max);
        let start = out.len();
        out.extend(self.bvh.intersect_aabb(&aabb).map(|i| i as usize));
        out[start..].sort_unstable();
    }
}

impl StaticGeometry for ObstacleSet {
    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        if self.is_empty() || !mask.intersects(HitMask::SOLID | HitMask::FOLIAGE) {
            return None;
        }
        let pray = PRay::new(ray.origin, ray.dir);
        // Foliage is hollow for rays (a sensor inside a canopy sees its boundary).
        let solid = |i: usize| self.obstacles[i].class == ObstacleClass::Solid;
        // Time of impact and normal (world): closed form where possible, parry otherwise.
        let cast = |i: usize, max_toi: f64, want_normal: bool| -> Option<(f64, DVec3)> {
            let o = &self.obstacles[i];
            let (lo, ld) = (o.pose.inverse_transform_point(ray.origin), o.pose.inverse_transform_vector(ray.dir));
            match o.shape.ray_cast_local(lo, ld, max_toi, solid(i)) {
                LocalCast::Hit(t, n) => Some((t, o.pose.rot * n)),
                LocalCast::Miss => None,
                LocalCast::Unsupported if want_normal => self.shapes[i]
                    .cast_ray_and_get_normal(&self.poses[i], &pray, max_toi, solid(i))
                    .map(|h| (h.time_of_impact, h.normal)),
                LocalCast::Unsupported => {
                    self.shapes[i].cast_ray(&self.poses[i], &pray, max_toi, solid(i)).map(|t| (t, DVec3::ZERO))
                }
            }
        };
        let (leaf, _) = self.bvh.cast_ray(&pray, max_toi, |leaf, best| {
            let i = leaf as usize;
            if !mask.contains(self.mask_of(i)) {
                return None;
            }
            cast(i, best, false).map(|(t, _)| t)
        })?;
        let i = leaf as usize;
        let (toi, mut normal) = cast(i, max_toi, true)?;
        if normal.dot(ray.dir) > 0.0 {
            normal = -normal;
        }
        if normal.length_squared() < 0.5 {
            normal = -ray.dir; // origin inside a solid: no meaningful normal
        }
        Some(RayHit { toi, point: ray.at(toi), normal, material: self.obstacles[i].material, kind: self.kind_of(i) })
    }

    fn sphere_contacts(&self, center: DVec3, radius: f64, margin: f64, mask: HitMask, out: &mut Vec<SurfacePoint>) {
        if self.is_empty() {
            return;
        }
        let reach = DVec3::splat(radius + margin);
        let aabb = Aabb::new(center - reach, center + reach);
        let mut leaves: smallvec::SmallVec<[u32; 8]> = self.bvh.intersect_aabb(&aabb).collect();
        leaves.sort_unstable();
        for leaf in leaves {
            if mask.contains(self.mask_of(leaf as usize)) {
                out.extend(self.sphere_contact(leaf, center, radius, margin));
            }
        }
    }

    fn nearest_distance(&self, p: DVec3, max_dist: f64, mask: HitMask) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let (_, (d, proj)) = self.bvh.project_point(p, max_dist, |leaf, _| {
            let i = leaf as usize;
            mask.contains(self.mask_of(i)).then(|| self.shapes[i].project_point(&self.poses[i], p, true))
        })?;
        Some(if proj.is_inside { 0.0 } else { d })
    }

    fn query_candidates(&self, min: DVec3, max: DVec3, mask: HitMask, out: &mut Vec<u32>) {
        if self.is_empty() {
            return;
        }
        let start = out.len();
        out.extend(self.bvh.intersect_aabb(&Aabb::new(min, max)).filter(|&i| mask.contains(self.mask_of(i as usize))));
        out[start..].sort_unstable();
    }

    fn sphere_contact(&self, id: u32, center: DVec3, radius: f64, margin: f64) -> Option<SurfacePoint> {
        let i = id as usize;
        let reach = radius + margin;
        let b = &self.aabbs[i];
        let o = &self.obstacles[i];
        if (b.mins - center).max(center - b.maxs).max_element() > reach
            || o.shape.distance_lower_bound(o.pose.inverse_transform_point(center)) > reach
        {
            return None;
        }
        let proj = self.shapes[i].project_point(&self.poses[i], center, false);
        let delta = center - proj.point;
        let d = delta.length();
        let distance = if proj.is_inside { -d } else { d };
        if distance > reach {
            return None;
        }
        let normal = if d > 1e-12 {
            if proj.is_inside { -delta / d } else { delta / d }
        } else {
            (center - self.poses[i].translation).try_normalize().unwrap_or(DVec3::Z)
        };
        Some(SurfacePoint { point: proj.point, normal, distance, material: o.material, kind: self.kind_of(i) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(x: f64, y: f64) -> [Obstacle; 2] {
        let trunk = Obstacle::solid(
            ObstacleShape::Capsule { half_height: 2.0, radius: 0.2 },
            Pose::from_translation(DVec3::new(x, y, 2.0)),
            MaterialId::WOOD,
        );
        let canopy = Obstacle::foliage(
            ObstacleShape::Cone { half_height: 3.0, radius: 2.0 },
            Pose::from_translation(DVec3::new(x, y, 6.0)),
        );
        [trunk, canopy]
    }

    fn forest() -> ObstacleSet {
        let mut v = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                v.extend(tree(i as f64 * 10.0, j as f64 * 10.0));
            }
        }
        v.push(Obstacle::solid(
            ObstacleShape::Cuboid { half_extents: DVec3::new(1.0, 2.0, 0.5) },
            Pose::new(DVec3::new(-5.0, 0.0, 0.5), DQuat::from_rotation_z(0.3)),
            MaterialId::ROCK,
        ));
        ObstacleSet::new(v)
    }

    #[test]
    fn axisymmetric_shapes_stand_upright() {
        let set = forest();
        // A horizontal ray at trunk height hits the trunk surface at radius 0.2.
        let hit = set.raycast(&Ray::new(DVec3::new(-3.0, 0.0, 1.0), DVec3::X), 100.0, HitMask::ALL).unwrap();
        assert_eq!(hit.kind, HitKind::Solid(0));
        assert!((hit.point.x + 0.2).abs() < 1e-9, "{:?}", hit.point);
        assert!((hit.normal - -DVec3::X).length() < 1e-9);
        // The trunk top is at z = 4.2: a ray at z = 4.5 passes the trunk but hits the canopy
        // (cone base at z = 3), whose radius there is 2·(1 − 1.5/6) = 1.5.
        let hit = set.raycast(&Ray::new(DVec3::new(-3.0, 0.0, 4.5), DVec3::X), 100.0, HitMask::ALL).unwrap();
        assert_eq!(hit.kind, HitKind::Foliage(1));
        assert!((hit.point.x + 1.5).abs() < 1e-9, "{:?}", hit.point);
        // With foliage masked out the ray continues to the next trunk.
        let hit = set.raycast(&Ray::new(DVec3::new(-3.0, 0.0, 1.0), DVec3::X), 100.0, HitMask::FOLIAGE);
        assert!(hit.is_none());
    }

    #[test]
    fn ray_from_inside_canopy_sees_boundary() {
        let set = forest();
        let hit = set.raycast(&Ray::new(DVec3::new(0.0, 0.0, 4.0), DVec3::Y), 100.0, HitMask::FOLIAGE).unwrap();
        assert_eq!(hit.kind, HitKind::Foliage(1));
        let r = 2.0 * (1.0 - 1.0 / 6.0);
        assert!((hit.toi - r).abs() < 1e-9, "toi {}", hit.toi);
    }

    #[test]
    fn sphere_contacts_and_distance() {
        let set = forest();
        let mut out = Vec::new();
        // Sphere touching the trunk at x = 10 from the side, penetrating by 5 cm.
        set.sphere_contacts(DVec3::new(10.0 + 0.2 + 0.25, 10.0, 1.0), 0.3, 0.01, HitMask::COLLIDABLE, &mut out);
        assert_eq!(out.len(), 1);
        let c = out[0];
        assert!(matches!(c.kind, HitKind::Solid(_)));
        assert!((c.distance - 0.25).abs() < 1e-9);
        assert!((c.normal - DVec3::X).length() < 1e-9);
        // Centre inside the rotated box: negative distance, normal points out of the box.
        out.clear();
        set.sphere_contacts(DVec3::new(-5.0, 0.0, 0.9), 0.2, 0.0, HitMask::COLLIDABLE, &mut out);
        assert_eq!(out.len(), 1);
        assert!((out[0].distance + 0.1).abs() < 1e-9);
        assert!((out[0].normal - DVec3::Z).length() < 1e-9);
        // Foliage only reported when requested.
        out.clear();
        set.sphere_contacts(DVec3::new(0.0, 0.0, 6.0), 0.1, 0.0, HitMask::COLLIDABLE, &mut out);
        assert!(out.is_empty());
        set.sphere_contacts(DVec3::new(0.0, 0.0, 6.0), 0.1, 0.0, HitMask::FOLIAGE, &mut out);
        assert_eq!(out.len(), 1);

        let d = set.nearest_distance(DVec3::new(5.0, 0.0, 1.0), 50.0, HitMask::SOLID).unwrap();
        assert!((d - 4.8).abs() < 1e-9, "{d}");
        assert!(set.nearest_distance(DVec3::new(5.0, 5.0, 100.0), 10.0, HitMask::SOLID).is_none());
    }

    #[test]
    fn distance_lower_bounds_hold() {
        let shapes = [
            ObstacleShape::Sphere { radius: 0.7 },
            ObstacleShape::Capsule { half_height: 1.5, radius: 0.3 },
            ObstacleShape::Cylinder { half_height: 1.0, radius: 0.5 },
            ObstacleShape::Cone { half_height: 3.0, radius: 2.0 },
            ObstacleShape::Cuboid { half_extents: DVec3::new(1.0, 0.4, 0.2) },
        ];
        let mut rng = autonomousim_core::rng::Seed::from_u64(3).rng();
        for shape in shapes {
            let (parry, local) = shape.to_parry();
            let pose = PPose::from_parts(DVec3::ZERO, local);
            let mut tight = 0usize;
            for _ in 0..20_000 {
                let p = DVec3::new(rng.range(-5.0, 5.0), rng.range(-5.0, 5.0), rng.range(-5.0, 5.0));
                let exact = parry.distance_to_point(&pose, p, true);
                let bound = shape.distance_lower_bound(p);
                assert!(bound <= exact + 1e-12, "{shape:?} at {p}: bound {bound} > {exact}");
                tight += (exact > 0.0 && bound > 0.5 * exact) as usize;
            }
            // The bound must reject most far points to be worth anything.
            assert!(tight > 10_000, "{shape:?}: only {tight} tight bounds");
        }
    }

    /// The closed-form ray casts agree with parry's (GJK-based) ones for rays from outside
    /// and inside, solid and hollow.
    #[test]
    fn analytic_ray_casts_match_parry() {
        let shapes = [
            ObstacleShape::Sphere { radius: 0.7 },
            ObstacleShape::Capsule { half_height: 1.5, radius: 0.3 },
            ObstacleShape::Cylinder { half_height: 1.0, radius: 0.5 },
            ObstacleShape::Cone { half_height: 3.0, radius: 2.0 },
            ObstacleShape::Cuboid { half_extents: DVec3::new(1.0, 0.4, 0.2) },
        ];
        let mut rng = autonomousim_core::rng::Seed::from_u64(5).rng();
        for shape in shapes {
            let (parry, local) = shape.to_parry();
            let pose = PPose::from_parts(DVec3::ZERO, local);
            let (mut hits, mut inside) = (0, 0);
            for k in 0..20_000 {
                // Half the rays aim near the shape, the rest start close to or inside it.
                let o = DVec3::new(rng.range(-4.0, 4.0), rng.range(-4.0, 4.0), rng.range(-4.0, 4.0))
                    * if k % 2 == 0 { 1.0 } else { 0.4 };
                let target = DVec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-2.0, 2.0));
                let d = (target - o).try_normalize().unwrap_or(DVec3::X);
                for solid in [true, false] {
                    let want = parry.cast_ray_and_get_normal(&pose, &PRay::new(o, d), 10.0, solid);
                    let got = shape.ray_cast_local(o, d, 10.0, solid);
                    match (want, got) {
                        (None, LocalCast::Miss) => {}
                        (Some(w), LocalCast::Hit(t, n)) => {
                            // Parry's cone and cylinder casts are GJK-based and stop within
                            // ~1e-5 of the surface; the closed-form point must lie on it.
                            assert!((w.time_of_impact - t).abs() < 1e-4, "{shape:?} {o} {d} {solid}: {w:?} vs {t}");
                            if t > 0.0 {
                                let off = parry.distance_to_point(&pose, o + d * t, false);
                                assert!(off < 1e-9, "{shape:?} {o} {d}: {off} off the surface");
                                // Compared as used: facing the ray (parry points exit normals
                                // inwards). GJK normals are only accurate to ~1e-4.
                                let facing = |n: DVec3| if n.dot(d) > 0.0 { -n } else { n };
                                let (wn, gn) = (facing(w.normal), facing(n));
                                assert!((wn - gn).length() < 1e-3, "{shape:?} {o} {d}: {wn} vs {gn}");
                                hits += 1;
                            } else {
                                inside += 1;
                            }
                        }
                        // Grazing rays may differ by a hair.
                        (Some(w), LocalCast::Miss) => {
                            assert!(parry.distance_to_point(&pose, o + d * w.time_of_impact, true) < 1e-9)
                        }
                        (None, LocalCast::Hit(t, _)) => {
                            assert!(parry.distance_to_point(&pose, o + d * t, true) < 1e-6, "{shape:?} {o} {d} {t}")
                        }
                        (_, LocalCast::Unsupported) => unreachable!(),
                    }
                }
            }
            assert!(hits > 5000 && inside > 200, "{shape:?}: {hits} hits, {inside} inside");
        }
    }

    #[test]
    fn serde_round_trip_rebuilds() {
        let set = forest();
        let json = serde_json::to_string(set.obstacles()).unwrap();
        let back = ObstacleSet::new(serde_json::from_str(&json).unwrap());
        let ray = Ray::new(DVec3::new(-3.0, 0.1, 1.0), DVec3::new(1.0, 0.01, 0.0));
        assert_eq!(set.raycast(&ray, 200.0, HitMask::ALL), back.raycast(&ray, 200.0, HitMask::ALL));
    }
}
