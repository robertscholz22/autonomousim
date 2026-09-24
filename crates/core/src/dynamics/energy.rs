//! Energy and momentum of a multibody system (for validation and diagnostics).

use super::{KinCache, MultibodyModel};
use crate::math::SpatialForce;
use glam::DVec3;

/// Total kinetic energy `Σ ½ vᵢᵀ Iᵢ vᵢ`.
pub fn kinetic_energy(model: &MultibodyModel, kin: &KinCache) -> f64 {
    model.links().iter().enumerate().map(|(i, l)| l.inertia.kinetic_energy(kin.vel[i])).sum()
}

/// Gravitational potential energy `-Σ mᵢ g · c_i` (zero at the world origin).
pub fn potential_energy(model: &MultibodyModel, kin: &KinCache, gravity: DVec3) -> f64 {
    model
        .links()
        .iter()
        .enumerate()
        .map(|(i, l)| -l.inertia.mass * gravity.dot(kin.pose[i].transform_point(l.inertia.com)))
        .sum()
}

/// Total spatial momentum in world coordinates (about the world origin).
pub fn spatial_momentum_world(model: &MultibodyModel, kin: &KinCache) -> SpatialForce {
    model
        .links()
        .iter()
        .enumerate()
        .map(|(i, l)| kin.x_world[i].inv_apply_force(l.inertia.mul_motion(kin.vel[i])))
        .fold(SpatialForce::ZERO, |a, b| a + b)
}

/// Centre of mass of the whole system in world coordinates.
pub fn center_of_mass_world(model: &MultibodyModel, kin: &KinCache) -> DVec3 {
    let m = model.total_mass();
    let s: DVec3 = model
        .links()
        .iter()
        .enumerate()
        .map(|(i, l)| l.inertia.mass * kin.pose[i].transform_point(l.inertia.com))
        .sum();
    s / m
}
