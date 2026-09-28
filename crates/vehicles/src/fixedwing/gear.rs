//! Landing gear in the manner of JSBSim's `LGear`: a massless strut spring and damper acting at
//! the wheel's contact point along the ground normal, rolling, braking and side friction in the
//! wheel's plane, and a steerable wheel. No tyre dynamics: friction is `μN·tanh(v/v_ε)`, which
//! holds the aircraft on level ground but lets it creep slowly on a slope.

use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::Terrain;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Speed over which friction reaches its full value (m/s).
const FRICTION_SPEED: f64 = 0.1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GearDef {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Wheel contact point with the strut fully extended (body frame, m).
    pub position: DVec3,
    /// Strut spring (N/m).
    pub spring: f64,
    /// Strut damping while compressing (N·s/m).
    pub damping: f64,
    /// Damping while extending (N·s/m; the compression damping if absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub damping_rebound: Option<f64>,
    #[serde(default = "default_rolling")]
    pub rolling_friction: f64,
    /// Friction across the wheel and the limit of the total friction force.
    #[serde(default = "default_side")]
    pub side_friction: f64,
    /// Friction added by a fully applied brake (0: no brake).
    #[serde(default)]
    pub brake_friction: f64,
    /// Steering angle at full rudder (rad; positive steers towards the rudder, 0: fixed).
    #[serde(default)]
    pub max_steer: f64,
    /// Wheel radius (m), for collisions with other agents and the viewer.
    #[serde(default = "default_wheel_radius")]
    pub wheel_radius: f64,
}

fn default_rolling() -> f64 {
    0.02
}
fn default_side() -> f64 {
    0.8
}
fn default_wheel_radius() -> f64 {
    0.1
}

impl GearDef {
    pub fn validate(&self) -> Result<(), String> {
        let ok = self.position.is_finite()
            && self.spring > 0.0
            && self.damping >= 0.0
            && self.damping_rebound.is_none_or(|c| c >= 0.0)
            && self.rolling_friction >= 0.0
            && self.side_friction >= 0.0
            && self.brake_friction >= 0.0
            && self.max_steer.abs() < 1.5
            && self.wheel_radius > 0.0;
        if ok { Ok(()) } else { Err(format!("invalid gear {:?}", self.name)) }
    }
}

/// One wheel's ground contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelContact {
    /// Contact point (world frame).
    pub point: DVec3,
    pub normal: DVec3,
    pub material: MaterialId,
    /// Strut compression (m).
    pub compression: f64,
    /// Normal velocity of the contact point (m/s, negative when sinking).
    pub normal_velocity: f64,
    pub normal_force: f64,
    /// Friction force (world frame).
    pub friction: DVec3,
    /// The wheel touched water (no force).
    pub water: bool,
}

/// Contact of the wheel whose contact point is at `p` moving with `v` (world frame), rolling
/// along `heading` (world frame, the wheel's forward direction), with brake `brake` in [0, 1].
pub fn contact(
    def: &GearDef,
    p: DVec3,
    v: DVec3,
    heading: DVec3,
    brake: f64,
    terrain: &dyn Terrain,
) -> Option<WheelContact> {
    let (h, n) = terrain.height_normal(p.x, p.y);
    if let Some(w) = terrain.water_level(p.x, p.y)
        && w > h
    {
        return (p.z < w).then_some(WheelContact {
            point: DVec3::new(p.x, p.y, w),
            normal: DVec3::Z,
            material: terrain.material(p.x, p.y),
            compression: w - p.z,
            normal_velocity: v.z,
            normal_force: 0.0,
            friction: DVec3::ZERO,
            water: true,
        });
    }
    let compression = (h - p.z) * n.z;
    if compression <= 0.0 {
        return None;
    }
    let vn = n.dot(v);
    let c = if vn < 0.0 { def.damping } else { def.damping_rebound.unwrap_or(def.damping) };
    let normal_force = (def.spring * compression - c * vn).max(0.0);
    let roll = (heading - n * n.dot(heading)).normalize_or_zero();
    let side = n.cross(roll);
    let mu_roll = def.rolling_friction + brake.clamp(0.0, 1.0) * def.brake_friction;
    let mut fr = -mu_roll * normal_force * (roll.dot(v) / FRICTION_SPEED).tanh();
    let mut fs = -def.side_friction * normal_force * (side.dot(v) / FRICTION_SPEED).tanh();
    let limit = def.side_friction.max(mu_roll) * normal_force;
    let total = fr.hypot(fs);
    if total > limit && total > 0.0 {
        fr *= limit / total;
        fs *= limit / total;
    }
    Some(WheelContact {
        point: p + n * compression,
        normal: n,
        material: terrain.material(p.x, p.y),
        compression,
        normal_velocity: vn,
        normal_force,
        friction: roll * fr + side * fs,
        water: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::terrain::FlatTerrain;

    #[test]
    fn strut_and_friction() {
        let def = GearDef {
            name: "main".into(),
            position: DVec3::ZERO,
            spring: 1000.0,
            damping: 100.0,
            damping_rebound: Some(200.0),
            rolling_friction: 0.02,
            side_friction: 0.8,
            brake_friction: 0.5,
            max_steer: 0.0,
            wheel_radius: 0.1,
        };
        let ground = FlatTerrain::new(0.0, MaterialId(0));
        assert!(contact(&def, DVec3::new(0.0, 0.0, 0.01), DVec3::ZERO, DVec3::X, 0.0, &ground).is_none());
        // Compressing at 0.1 m/s: k·δ + c·v.
        let c = contact(&def, DVec3::new(0.0, 0.0, -0.05), DVec3::new(0.0, 0.0, -0.1), DVec3::X, 0.0, &ground).unwrap();
        assert!((c.normal_force - 60.0).abs() < 1e-9);
        // Extending uses the rebound damping; the force never pulls.
        let c = contact(&def, DVec3::new(0.0, 0.0, -0.05), DVec3::new(0.0, 0.0, 0.1), DVec3::X, 0.0, &ground).unwrap();
        assert!((c.normal_force - 30.0).abs() < 1e-9);
        let c = contact(&def, DVec3::new(0.0, 0.0, -0.01), DVec3::new(0.0, 0.0, 1.0), DVec3::X, 0.0, &ground).unwrap();
        assert_eq!(c.normal_force, 0.0);
        // Rolling forward: rolling resistance, plus the brake; sideways: side friction.
        let at =
            |v: DVec3, brake: f64| contact(&def, DVec3::new(0.0, 0.0, -0.05), v, DVec3::X, brake, &ground).unwrap();
        let c = at(DVec3::new(5.0, 0.0, 0.0), 0.0);
        assert!((c.friction.x + 0.02 * 50.0).abs() < 1e-6 && c.friction.y.abs() < 1e-12);
        let c = at(DVec3::new(5.0, 0.0, 0.0), 1.0);
        assert!((c.friction.x + 0.52 * 50.0).abs() < 1e-6);
        let c = at(DVec3::new(0.0, 5.0, 0.0), 0.0);
        assert!((c.friction.y + 0.8 * 50.0).abs() < 1e-6);
        // Combined slip stays within the friction circle.
        let c = at(DVec3::new(5.0, 5.0, 0.0), 1.0);
        assert!(c.friction.length() <= 0.8 * 50.0 + 1e-9);
    }
}
