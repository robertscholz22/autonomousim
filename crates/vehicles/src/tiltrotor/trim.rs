//! Trim of a tiltrotor in steady level flight at a given airspeed and mount tilt, and the
//! conversion corridor: the tilts at which it can fly level at each airspeed.
//!
//! The longitudinal trim solves for pitch, the speeds of the front and of the rear rotors and
//! the elevator, with aileron and rudder centred and every tilting mount at the same tilt. Four
//! unknowns need a fourth condition: the pitching moment is shared between the differential
//! rotor speed `d = (ω_f − ω_r)/ω_h` and the elevator input `e` with the least effort `d² + e²`,
//! so each carries it in proportion to its effectiveness: `d·∂M/∂e = e·∂M/∂d`. With the rotors
//! up the elevator is idle in hover, with the rotors forward the rotors turn alike. The motors'
//! throttles, currents and powers follow from the steady rotor speeds. The lateral balance holds
//! by the layout's symmetry (counter-rotating pairs).

use super::def::{MAX_ROTORS, TiltrotorDef};
use super::model::{Tiltrotor, TiltrotorInit, TiltrotorInput, TiltrotorLoads};
use crate::aero::{AirData, AirFlow};
use crate::rotorcraft::attitude;
use crate::rotorcraft::trim::newton;
use autonomousim_core::math::Pose;
use glam::{DQuat, DVec3};
use std::f64::consts::{FRAC_PI_2, PI};

/// Residual norm (forces over weight, moments over weight × rotor arm) of a trimmed state.
const TOLERANCE: f64 = 1e-8;

/// A trimmed level flight condition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TiltrotorTrim {
    /// Airspeed (m/s) and tilt of the tilting mounts (rad).
    pub speed: f64,
    pub tilt: f64,
    /// Pitch (nose up, rad): also the angle of attack at the centre of mass.
    pub pitch: f64,
    /// Throttles (unclamped: beyond [0, 1] the trim is out of reach), tilts and elevator
    /// (unclamped), aileron and rudder centred.
    pub controls: TiltrotorInput,
    /// Rotor speeds (rad/s), motor currents (A), thrusts (N).
    pub rotor_speed: [f64; MAX_ROTORS],
    pub current: [f64; MAX_ROTORS],
    pub thrust: [f64; MAX_ROTORS],
    /// Shaft power of the rotors and electric power of the motors (W).
    pub shaft_power: f64,
    pub electric_power: f64,
    /// Least margin to the stall angle over the horizontal surfaces (rad; negative stalled).
    pub stall_margin: f64,
    pub loads: TiltrotorLoads,
    /// Air-relative velocity in the body frame (m/s).
    pub velocity_body: DVec3,
    pub density: f64,
    pub gravity: f64,
}

/// Bounds of a flyable trim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrimLimits {
    /// Largest pitch either way (rad).
    pub max_pitch: f64,
}

impl Default for TrimLimits {
    /// ±15° of pitch.
    fn default() -> Self {
        Self { max_pitch: 15f64.to_radians() }
    }
}

impl TiltrotorTrim {
    /// Whether the aircraft can hold it: throttles within [0, 1], currents within the limit,
    /// elevator within its range, no stall and the pitch within the limits.
    pub fn feasible(&self, def: &TiltrotorDef, limits: &TrimLimits) -> bool {
        let n = def.rotors.len();
        let i_max = def.motor.max_current.unwrap_or(f64::INFINITY);
        self.controls.throttle[..n].iter().all(|t| (0.0..=1.0).contains(t))
            && self.current[..n].iter().all(|&i| i <= i_max)
            && self.controls.elevator.abs() <= 1.0
            && self.stall_margin > 0.0
            && self.pitch.abs() <= limits.max_pitch
    }

    /// Body → world rotation for a heading (yaw, rad, counter-clockwise from +x).
    pub fn attitude(&self, heading: f64) -> DQuat {
        attitude(heading, self.pitch, 0.0)
    }

    /// Initial conditions flying this trim at `position` on heading `heading`.
    pub fn init(&self, position: DVec3, heading: f64) -> TiltrotorInit {
        let rot = self.attitude(heading);
        TiltrotorInit {
            pose: Pose::new(position, rot),
            lin_vel_world: rot * self.velocity_body,
            ang_vel_body: DVec3::ZERO,
            controls: self.controls,
            rotor_speed: Some(self.rotor_speed),
            density: self.density,
        }
    }

    /// Total thrust of the rotors (N).
    pub fn total_thrust(&self) -> f64 {
        self.thrust.iter().sum()
    }
}

/// The tilts at which level flight at `speed` is feasible: the least and the largest (`None`
/// when there is none), and whether every tilt of the sweep between them is feasible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorridorPoint {
    pub speed: f64,
    pub tilt: Option<(f64, f64)>,
    pub contiguous: bool,
}

impl Tiltrotor {
    /// Rotor speed (rad/s) at which the rotors carry the weight in hover under gravity `g`.
    pub fn hover_rotor_speed(&self, rho: f64, g: f64) -> f64 {
        let t = self.mass() * g / self.rotor_count() as f64;
        let (mut lo, mut hi) = (0.0, 100.0);
        while self.propulsion().prop_loads(hi, 0.0, rho).0 < t && hi < 1e6 {
            hi *= 2.0;
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if self.propulsion().prop_loads(mid, 0.0, rho).0 < t { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    }

    /// Trim in level flight at `speed` (m/s, still air of density `rho`, gravity `g`) with the
    /// tilting mounts at `tilt` (rad).
    pub fn trim(&self, speed: f64, tilt: f64, rho: f64, g: f64) -> Result<TiltrotorTrim, String> {
        let w = self.hover_rotor_speed(rho, g);
        self.solve(speed, tilt, rho, g, [0.0, 1.0, 1.0, 0.0], w)
    }

    /// As [`trim`](Self::trim), starting from a nearby trim (for sweeps).
    pub fn trim_near(
        &self,
        speed: f64,
        tilt: f64,
        rho: f64,
        g: f64,
        near: &TiltrotorTrim,
    ) -> Result<TiltrotorTrim, String> {
        let w = self.hover_rotor_speed(rho, g);
        let (f, r) = self.groups();
        let mean = |set: &[usize]| set.iter().map(|&k| near.rotor_speed[k]).sum::<f64>() / (set.len() as f64 * w);
        self.solve(speed, tilt, rho, g, [near.pitch, mean(&f), mean(&r), near.controls.elevator], w)
    }

    /// Rotors ahead of and behind the centre of mass.
    fn groups(&self) -> (Vec<usize>, Vec<usize>) {
        (0..self.rotor_count()).partition(|&k| self.def().rotors[k].pivot.x > 0.0)
    }

    fn solve(
        &self,
        speed: f64,
        tilt: f64,
        rho: f64,
        g: f64,
        x0: [f64; 4],
        w_ref: f64,
    ) -> Result<TiltrotorTrim, String> {
        let def = self.def();
        let (front, rear) = self.groups();
        if front.is_empty() || rear.is_empty() {
            return Err(format!("{}: the trim needs rotors ahead of and behind the centre of mass", def.name));
        }
        let weight = self.mass() * g;
        let arm = def.rotors.iter().map(|r| r.pivot.x.abs()).fold(0.0, f64::max).max(0.1);
        let air = AirData { density: rho, ..AirData::default() };
        let mut tilts = [0.0; MAX_ROTORS];
        for (k, t) in tilts.iter_mut().enumerate().take(self.rotor_count()) {
            *t = def.mount_tilt(k, tilt);
        }
        let setup = |x: &[f64]| {
            let (s, c) = x[0].sin_cos();
            let flow = AirFlow::new(&air, DVec3::new(speed * c, 0.0, -speed * s), DVec3::ZERO);
            let mut omega = [0.0; MAX_ROTORS];
            for &k in &front {
                omega[k] = x[1] * w_ref;
            }
            for &k in &rear {
                omega[k] = x[2] * w_ref;
            }
            (flow, omega, self.channels_linear(0.0, x[3], 0.0))
        };
        let moment = |x: &[f64]| {
            let (flow, omega, channels) = setup(x);
            self.steady_wrench(&flow, &omega, &tilts, &channels).1.y / (weight * arm)
        };
        let residual = |x: &[f64]| {
            let (flow, omega, channels) = setup(x);
            let (f, m, _) = self.steady_wrench(&flow, &omega, &tilts, &channels);
            let (s, c) = x[0].sin_cos();
            let f = (f - DVec3::new(weight * s, 0.0, weight * c)) / weight;
            // Moment derivatives by central differences: the differential speed moves the front
            // rotors up and the rear ones down by the same amount.
            let h = 1e-4;
            let shifted = |df: f64, de: f64| moment(&[x[0], x[1] + df, x[2] - df, x[3] + de]);
            let m_d = (shifted(0.5 * h, 0.0) - shifted(-0.5 * h, 0.0)) / h;
            let m_e = (shifted(0.0, h) - shifted(0.0, -h)) / (2.0 * h);
            vec![f.x, f.z, m.y / (weight * arm), (x[1] - x[2]) * m_e - x[3] * m_d]
        };
        let mut x = x0.to_vec();
        let err = newton(&mut x, residual);
        if err.is_nan() || err >= TOLERANCE || x[1] <= 0.0 || x[2] <= 0.0 || x[0].abs() > FRAC_PI_2 {
            return Err(format!("{}: no trim at {speed} m/s and tilt {tilt:.3} (residual {err:.2e})", def.name));
        }
        let (flow, omega, channels) = setup(&x);
        let (_, _, loads) = self.steady_wrench(&flow, &omega, &tilts, &channels);
        let m = &def.motor;
        let k_t = 60.0 / (2.0 * PI * m.kv);
        let voltage = m.voltage.unwrap_or(0.0);
        let mut controls = TiltrotorInput { tilt: [tilt; MAX_ROTORS], elevator: x[3], ..TiltrotorInput::default() };
        let (mut current, mut shaft_power, mut electric_power) = ([0.0; MAX_ROTORS], 0.0, 0.0);
        for k in 0..self.rotor_count() {
            // Steady: the motor torque K(i − i₀) turns the propeller, i = (throttle·V − Kω)/R.
            let i = loads.prop_torque[k] / k_t + m.no_load_current;
            let throttle = (k_t * omega[k] + i * m.resistance) / voltage;
            controls.throttle[k] = throttle;
            current[k] = i;
            shaft_power += loads.prop_torque[k] * omega[k];
            electric_power += throttle * voltage * i;
        }
        let stall_margin = def
            .surfaces
            .iter()
            .filter(|s| s.roll.cos().abs() > 0.5)
            .filter_map(|s| s.angle_of_attack(&flow).map(|a| s.alpha_stall - a.abs()))
            .fold(f64::INFINITY, f64::min);
        Ok(TiltrotorTrim {
            speed,
            tilt,
            pitch: x[0],
            controls,
            rotor_speed: omega,
            current,
            thrust: loads.thrust,
            shaft_power,
            electric_power,
            stall_margin,
            loads,
            velocity_body: flow.velocity,
            density: rho,
            gravity: g,
        })
    }

    /// Conversion corridor: at each of `speeds`, the range of feasible tilts (see
    /// [`TiltrotorTrim::feasible`]) over the tilt channel's range in steps of `tilt_step`
    /// (rad), trimming by continuation along the tilt (from the last feasible trim) and from
    /// speed to speed.
    pub fn corridor(
        &self,
        speeds: &[f64],
        tilt_step: f64,
        limits: &TrimLimits,
        rho: f64,
        g: f64,
    ) -> Vec<CorridorPoint> {
        let t = &self.def().controls.tilt;
        let n = ((t.max - t.min) / tilt_step).ceil().max(1.0) as usize;
        let tilts: Vec<f64> = (0..=n).map(|i| (t.min + i as f64 * tilt_step).min(t.max)).collect();
        let mut previous: Vec<Option<TiltrotorTrim>> = vec![None; tilts.len()];
        let mut out = Vec::with_capacity(speeds.len());
        for &speed in speeds {
            let mut last: Option<TiltrotorTrim> = None;
            let mut range: Option<(f64, f64)> = None;
            let (mut contiguous, mut gap) = (true, false);
            for (i, &tilt) in tilts.iter().enumerate() {
                let guesses = [last, previous[i]];
                let trim = guesses
                    .iter()
                    .flatten()
                    .find_map(|near| self.trim_near(speed, tilt, rho, g, near).ok())
                    .or_else(|| self.trim(speed, tilt, rho, g).ok());
                previous[i] = trim;
                if trim.is_some_and(|tr| tr.feasible(self.def(), limits)) {
                    // Continue from feasible trims only, which keeps the sweep off the stalled
                    // branch.
                    last = trim;
                    contiguous &= !gap;
                    range = Some(range.map_or((tilt, tilt), |(lo, _)| (lo, tilt)));
                } else {
                    gap = range.is_some();
                }
            }
            out.push(CorridorPoint { speed, tilt: range, contiguous });
        }
        out
    }
}
