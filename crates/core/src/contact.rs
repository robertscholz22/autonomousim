//! Penalty contacts between vehicle sphere colliders and the static world.
//!
//! * Normal force: linear spring–damper `F_n = max(0, k·δ + c·δ̇)` with `k = m·ω_c²` and
//!   `c = 2ζ·m·ω_c`. Choosing `ω_c` from the time step (`ω_c·dt = 0.2`) keeps semi-implicit
//!   Euler stable for any vehicle mass. The ground material scales `ω_c` (soft mud, snow).
//! * Friction: a tangential "bristle" spring per persistent contact, integrated from the slip
//!   velocity of the body point at the contact and capped by the Coulomb limit `μ·F_n` (the
//!   spring is reset to the limit while sliding). A body at rest on a slope keeps a constant
//!   deflection, so it sticks without creep; a rolling sphere has zero slip velocity and does
//!   not build up spring force.
//!
//! Friction states live in a per-instance [`ContactCache`] keyed by collider and world feature
//! (terrain or obstacle index), so contacts must be evaluated exactly once per physics step.

use crate::dynamics::KinCache;
use crate::geometry::{HitKind, HitMask, StaticGeometry, SurfacePoint};
use crate::material::{MaterialId, MaterialTable};
use crate::math::SpatialForce;
use crate::terrain::Terrain;
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

/// Sphere rigidly attached to a link.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SphereCollider {
    pub link: usize,
    /// Centre in link coordinates.
    pub center: DVec3,
    pub radius: f64,
    /// Free-form group for event classification (e.g. landing gear vs. airframe).
    #[serde(default)]
    pub group: u8,
    /// Multiplies the friction coefficient (0 for a frictionless skid or caster ball).
    #[serde(default = "one")]
    pub friction: f64,
}

fn one() -> f64 {
    1.0
}

impl SphereCollider {
    /// A collider with full friction.
    pub fn new(link: usize, center: DVec3, radius: f64, group: u8) -> Self {
        Self { link, center, radius, group, friction: 1.0 }
    }
}

/// Spring–damper constants of one contact class (before material scaling).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PenaltyParams {
    /// Normal stiffness (N/m).
    pub stiffness: f64,
    /// Normal damping (N·s/m).
    pub damping: f64,
    /// Bristle stiffness (N/m).
    pub tangential_stiffness: f64,
    /// Bristle damping (N·s/m).
    pub tangential_damping: f64,
    /// Multiplies the material friction coefficient.
    pub friction_scale: f64,
}

impl PenaltyParams {
    /// Constants for natural frequency `omega` (rad/s) and damping ratio `zeta` of an effective
    /// mass `mass` (kg); the bristle uses the same values.
    pub fn from_frequency(mass: f64, omega: f64, zeta: f64) -> Self {
        let k = mass * omega * omega;
        let c = 2.0 * zeta * mass * omega;
        Self { stiffness: k, damping: c, tangential_stiffness: k, tangential_damping: c, friction_scale: 1.0 }
    }

    /// Constants with the natural frequency scaled by `s` (stiffness by `s²`, damping by `s`).
    #[inline]
    fn scaled(&self, s: f64) -> Self {
        Self {
            stiffness: self.stiffness * s * s,
            damping: self.damping * s,
            tangential_stiffness: self.tangential_stiffness * s * s,
            tangential_damping: self.tangential_damping * s,
            friction_scale: self.friction_scale,
        }
    }
}

/// Contact constants of one vehicle.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactModel {
    /// Terrain and solid obstacles.
    pub solid: PenaltyParams,
    /// Foliage (canopies, bushes): soft and heavily damped.
    pub foliage: PenaltyParams,
}

impl ContactModel {
    /// Default constants for colliders that each carry about `effective_mass` (kg), e.g. a
    /// quarter of the vehicle mass for four landing feet: solid `ω_c = 0.2/dt`, `ζ = 0.8`;
    /// foliage `ω_c = 20 rad/s`, `ζ = 2`.
    pub fn for_mass(effective_mass: f64, dt: f64) -> Self {
        Self {
            solid: PenaltyParams::from_frequency(effective_mass, 0.2 / dt, 0.8),
            foliage: PenaltyParams::from_frequency(effective_mass, 20.0, 2.0),
        }
    }
}

/// The static world as seen by contacts.
#[derive(Clone, Copy)]
pub struct StaticScene<'a> {
    pub terrain: &'a dyn Terrain,
    pub obstacles: &'a dyn StaticGeometry,
    pub materials: &'a MaterialTable,
}

/// One active contact (for events, sensors and visualisation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactPoint {
    /// Index into the collider list.
    pub collider: u16,
    pub kind: HitKind,
    pub material: MaterialId,
    /// Surface point where the force acts.
    pub point: DVec3,
    /// Surface normal towards the collider.
    pub normal: DVec3,
    /// Penetration depth (m).
    pub depth: f64,
    /// Normal velocity of the body point (m/s, negative when approaching).
    pub normal_velocity: f64,
    /// Normal force magnitude (N).
    pub normal_force: f64,
    /// Friction force (N, world frame).
    pub friction: DVec3,
    pub slipping: bool,
}

#[derive(Clone, Copy, Debug)]
struct Bristle {
    collider: u16,
    kind: HitKind,
    /// Tangential spring deflection (world frame).
    deflection: DVec3,
    seen: bool,
}

/// Persistent friction states of one instance (cheap to clone for snapshots).
#[derive(Clone, Debug, Default)]
pub struct ContactCache {
    bristles: Vec<Bristle>,
}

impl ContactCache {
    pub fn clear(&mut self) {
        self.bristles.clear();
    }

    /// Number of persistent contacts.
    pub fn len(&self) -> usize {
        self.bristles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bristles.is_empty()
    }

    fn bristle(&mut self, collider: u16, kind: HitKind) -> &mut Bristle {
        let i = match self.bristles.iter().position(|b| b.collider == collider && b.kind == kind) {
            Some(i) => i,
            None => {
                self.bristles.push(Bristle { collider, kind, deflection: DVec3::ZERO, seen: false });
                self.bristles.len() - 1
            }
        };
        let b = &mut self.bristles[i];
        b.seen = true;
        b
    }
}

/// Reusable buffers for [`compute_contacts`].
#[derive(Clone, Debug, Default)]
pub struct ContactScratch {
    surface: Vec<SurfacePoint>,
    candidates: Vec<u32>,
}

/// Evaluate all collider contacts of one instance: adds the contact wrenches to `f_ext`
/// (per link, link coordinates about the link origin, as taken by the ABA), updates the
/// friction states and appends the active contacts to `out`.
pub fn compute_contacts(
    scene: &StaticScene,
    kin: &KinCache,
    colliders: &[SphereCollider],
    model: &ContactModel,
    dt: f64,
    cache: &mut ContactCache,
    scratch: &mut ContactScratch,
    f_ext: &mut [SpatialForce],
    out: &mut Vec<ContactPoint>,
) {
    for b in &mut cache.bristles {
        b.seen = false;
    }
    if colliders.is_empty() {
        cache.bristles.clear();
        return;
    }

    // Broadphase over the bounding box of all colliders.
    let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    for c in colliders {
        let p = kin.pose[c.link].transform_point(c.center);
        lo = lo.min(p - c.radius);
        hi = hi.max(p + c.radius);
    }
    let near_terrain = lo.z <= scene.terrain.height_bounds(DVec2::new(lo.x, lo.y), DVec2::new(hi.x, hi.y)).1;
    scratch.candidates.clear();
    scene.obstacles.query_candidates(lo, hi, HitMask::COLLIDABLE | HitMask::FOLIAGE, &mut scratch.candidates);

    if near_terrain || !scratch.candidates.is_empty() {
        for (ci, c) in colliders.iter().enumerate() {
            let pose = kin.pose[c.link];
            let center = pose.transform_point(c.center);
            scratch.surface.clear();
            if near_terrain && let Some(sp) = scene.terrain.closest_point(center, c.radius) {
                scratch.surface.push(sp);
            }
            for &id in &scratch.candidates {
                scratch.surface.extend(scene.obstacles.sphere_contact(id, center, c.radius, 0.0));
            }
            for sp in &scratch.surface {
                let depth = c.radius - sp.distance;
                if depth <= 0.0 {
                    continue;
                }
                let base = if matches!(sp.kind, HitKind::Foliage(_)) { &model.foliage } else { &model.solid };
                let material = scene.materials.get(sp.material);
                let params = base.scaled(material.stiffness_scale);
                let mu = material.friction * params.friction_scale * c.friction;

                let n = sp.normal;
                let p_link = pose.inverse_transform_point(sp.point);
                let v = kin.point_velocity_world(c.link, p_link);
                let vn = n.dot(v);
                let fn_ = (params.stiffness * depth - params.damping * vn).max(0.0);

                // Bristle friction.
                let b = cache.bristle(ci as u16, sp.kind);
                let vt = v - n * vn;
                let s = b.deflection - n * n.dot(b.deflection) + vt * dt;
                let trial = -params.tangential_stiffness * s - params.tangential_damping * vt;
                let limit = mu * fn_;
                let (ft, slipping) = if trial.length_squared() > limit * limit {
                    let ft = trial.normalize_or_zero() * limit;
                    b.deflection = -ft / params.tangential_stiffness;
                    (ft, true)
                } else {
                    b.deflection = s;
                    (trial, false)
                };

                let f_world = n * fn_ + ft;
                f_ext[c.link] += SpatialForce::from_force_at_point(pose.inverse_transform_vector(f_world), p_link);
                out.push(ContactPoint {
                    collider: ci as u16,
                    kind: sp.kind,
                    material: sp.material,
                    point: sp.point,
                    normal: n,
                    depth,
                    normal_velocity: vn,
                    normal_force: fn_,
                    friction: ft,
                    slipping,
                });
            }
        }
    }
    cache.bristles.retain(|b| b.seen);
}
