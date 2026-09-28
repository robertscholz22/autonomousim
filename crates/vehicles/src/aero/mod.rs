//! Shared aerodynamics: the air at the vehicle ([`AirData`]), the flow relative to the body
//! ([`AirFlow`]: airspeed, angle of attack, sideslip, Mach number), lifting surfaces
//! ([`AeroSurface`]) and ground proximity for rotors ([`GroundPlane`], [`ground_effect`]).
//!
//! Angles follow the aeronautical convention in FRD terms although the body frame is FLU:
//! `α = atan2(−v_z, v_x)` is positive with the air coming from below, `β = asin(−v_y / V)`
//! positive with the air coming from the right.

mod surface;

pub use surface::{AeroSurface, AlphaTable, Coefficients, Flap, stall_blend};

use autonomousim_core::terrain::Terrain;
use glam::{DQuat, DVec3};

/// Sea-level ISA air density (kg/m³), the default reference density of rotor coefficients.
pub const SEA_LEVEL_DENSITY: f64 = 1.225;

/// Sea-level ISA speed of sound (m/s).
pub const SEA_LEVEL_SPEED_OF_SOUND: f64 = 340.294;

/// Airspeed below which the flow angles are reported as zero (m/s).
const MIN_FLOW_SPEED: f64 = 1e-3;

/// Ambient air at the vehicle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AirData {
    /// Density (kg/m³).
    pub density: f64,
    /// Air velocity in the world frame (m/s): mean wind, gusts and turbulence.
    pub wind: DVec3,
    /// Speed of sound (m/s).
    pub speed_of_sound: f64,
    /// Rotational turbulence: the angular velocity of the air as the vehicle sees it (body
    /// frame, rad/s; zero unless the vehicle asks for rotational gusts).
    pub gust_rates: DVec3,
}

impl Default for AirData {
    fn default() -> Self {
        Self {
            density: SEA_LEVEL_DENSITY,
            wind: DVec3::ZERO,
            speed_of_sound: SEA_LEVEL_SPEED_OF_SOUND,
            gust_rates: DVec3::ZERO,
        }
    }
}

impl AirData {
    /// Velocity relative to the air of a point moving with `velocity` (world frame, m/s).
    #[inline]
    pub fn relative(&self, velocity: DVec3) -> DVec3 {
        velocity - self.wind
    }

    /// Flow over a body with the given attitude (body → world), world velocity and body rates.
    pub fn flow(&self, attitude: DQuat, velocity: DVec3, rates: DVec3) -> AirFlow {
        AirFlow::new(self, attitude.inverse() * self.relative(velocity), rates - self.gust_rates)
    }
}

/// Flow relative to a body.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AirFlow {
    /// Velocity of the body's origin relative to the air (body frame, m/s).
    pub velocity: DVec3,
    /// Angular velocity relative to the air (body frame, rad/s).
    pub rates: DVec3,
    /// True airspeed (m/s).
    pub airspeed: f64,
    /// Angle of attack and sideslip (rad).
    pub alpha: f64,
    pub beta: f64,
    pub mach: f64,
    /// ½ρV² (Pa).
    pub dynamic_pressure: f64,
    pub density: f64,
}

impl AirFlow {
    /// Flow from the air-relative velocity and rates in the body frame.
    pub fn new(air: &AirData, velocity: DVec3, rates: DVec3) -> Self {
        let airspeed = velocity.length();
        let (alpha, beta) = if airspeed > MIN_FLOW_SPEED {
            ((-velocity.z).atan2(velocity.x), (-velocity.y / airspeed).clamp(-1.0, 1.0).asin())
        } else {
            (0.0, 0.0)
        };
        Self {
            velocity,
            rates,
            airspeed,
            alpha,
            beta,
            mach: airspeed / air.speed_of_sound,
            dynamic_pressure: 0.5 * air.density * airspeed * airspeed,
            density: air.density,
        }
    }

    /// Air-relative velocity of the body point `r` (body frame).
    #[inline]
    pub fn at(&self, r: DVec3) -> DVec3 {
        self.velocity + self.rates.cross(r)
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
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn flow_angles_and_wind() {
        let air = AirData { wind: DVec3::new(-5.0, 0.0, 0.0), ..AirData::default() };
        // Level at 20 m/s ground speed into a 5 m/s headwind, climbing air from below.
        let f = air.flow(DQuat::IDENTITY, DVec3::new(20.0, 0.0, -2.0), DVec3::ZERO);
        assert!((f.velocity - DVec3::new(25.0, 0.0, -2.0)).length() < 1e-12);
        assert!((f.alpha - (2.0f64).atan2(25.0)).abs() < 1e-12 && f.beta == 0.0);
        assert!((f.dynamic_pressure - 0.5 * SEA_LEVEL_DENSITY * 629.0).abs() < 1e-9);
        assert!((f.mach - 629f64.sqrt() / SEA_LEVEL_SPEED_OF_SOUND).abs() < 1e-12);
        // Drifting right (−y in FLU): air from the right, positive sideslip.
        let f = AirData::default().flow(DQuat::IDENTITY, DVec3::new(10.0, -10.0, 0.0), DVec3::ZERO);
        assert!((f.beta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        // Yawed 90° left while flying north: the body sees the air along its x axis.
        let f = AirData::default().flow(DQuat::from_rotation_z(FRAC_PI_2), DVec3::new(0.0, 10.0, 0.0), DVec3::ZERO);
        assert!((f.velocity - DVec3::new(10.0, 0.0, 0.0)).length() < 1e-12);
        // Rolling right (+x in FLU): the left tip moves up, the right one down.
        let f = AirFlow::new(&AirData::default(), DVec3::X * 10.0, DVec3::new(1.0, 0.0, 0.0));
        assert!(f.at(DVec3::new(0.0, 2.0, 0.0)).z > 0.0 && f.at(DVec3::new(0.0, -2.0, 0.0)).z < 0.0);
        // Hovering in calm air: no angles.
        let f = AirData::default().flow(DQuat::IDENTITY, DVec3::ZERO, DVec3::ZERO);
        assert_eq!((f.alpha, f.beta, f.airspeed), (0.0, 0.0, 0.0));
    }

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
