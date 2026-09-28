//! The tiltrotor controller (see the [module documentation](super)).

use super::allocation::{EFFECTORS, OBJECTIVES, allocate};
use super::schedule::TiltSchedule;
use super::{TiltrotorConfig, TiltrotorSetpoint};
use crate::ControlError;
use crate::fixedwing::euler;
use autonomousim_core::math::quat::wrap_angle;
use autonomousim_vehicles::aero::{AirData, AirFlow};
use autonomousim_vehicles::tiltrotor::{
    MAX_ROTORS, RotorLoad, TiltMount, Tiltrotor, TiltrotorDef, TiltrotorInput, TrimLimits,
};
use glam::{DQuat, DVec2, DVec3};
use std::sync::Arc;

/// Largest heading error the attitude loop acts on (rad).
const HEADING_LAG: f64 = 0.5;
/// Time (s) the loops take on the ground to lower the thrust to nothing.
const SETTLE_TIME: f64 = 1.0;
/// Effector slots after the rotor thrusts: differential tilt, then aileron, elevator, rudder.
const TILT: usize = MAX_ROTORS;
const SURFACES: usize = MAX_ROTORS + 1;
/// Least thrust the allocation leaves a rotor, as a share of the hover thrust per rotor.
const MIN_THRUST: f64 = 0.03;
/// Axial speeds of the full-throttle thrust table: first, step, count (m/s).
const AXIAL: (f64, f64, usize) = (-10.0, 1.0, 71);
/// Objectives dropped when the effectors cannot meet them all: yaw, thrust, pitch, roll.
const DROP_ORDER: [usize; 4] = [2, 3, 1, 0];

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// The aircraft as the force model sees it for one update.
struct Frame {
    /// Velocity relative to the air in the heading frame (forward, left, up; m/s).
    air: DVec3,
    rates: DVec3,
    density: f64,
    /// Thrust axis and share of the total thrust of each rotor, and the rotors' in-plane
    /// forces (body frame).
    axes: [DVec3; MAX_ROTORS],
    share: [f64; MAX_ROTORS],
    in_plane: DVec3,
    /// Total rotor thrust now (N).
    thrust: f64,
    /// Aerodynamic force on the airframe and the rotors' in-plane forces now (heading frame,
    /// N): what the airframe adds to the thrust.
    drag: DVec3,
}

/// Setpoint → tiltrotor input.
#[derive(Clone, Debug)]
pub struct TiltrotorController {
    config: TiltrotorConfig,
    dt: f64,
    schedule: Arc<TiltSchedule>,
    /// Rate, attitude, velocity and position bandwidths (rad/s).
    k_rate: f64,
    k_att: f64,
    k_vel: f64,
    k_pos: f64,
    /// Integrals: angular acceleration (rad/s²) and acceleration (m/s², heading frame).
    i_rate: DVec3,
    i_vel: DVec3,
    /// Heading reference of the attitude loop (rad); `None` until it first runs.
    heading: Option<f64>,
    /// Velocity reference (heading frame) and airspeed reference, following the commands at
    /// the configured accelerations.
    reference: Option<DVec3>,
    airspeed: Option<f64>,
    /// Last solution of the thrust and pitch (N, rad): where the next solve starts.
    solution: Option<(f64, f64)>,
    /// Tilt the schedule asked for last (rad).
    tilt: f64,
    /// Whether the thrust or the pitch met a limit in the last update.
    saturated: bool,
    /// Touching the ground in the last update, and the share (0…1) of the way the thrust has
    /// come down since.
    grounded: bool,
    settle: f64,
    mass: f64,
    /// Hover thrust per rotor (N).
    hover_thrust: f64,
    /// Steady thrust of one rotor at full throttle against axial speed (N; [`AXIAL`]).
    max_thrust: Arc<Vec<f64>>,
    /// Largest wing angle of attack the pitch may give (rad).
    alpha_limit: f64,
    /// +1 for tilting mounts on the left, −1 on the right, 0 otherwise.
    sides: [f64; MAX_ROTORS],
}

impl TiltrotorController {
    pub fn new(def: &Arc<TiltrotorDef>, dt: f64, config: &TiltrotorConfig) -> Result<Self, ControlError> {
        let fail = |m: String| ControlError::InvalidConfig(format!("{}: {m}", def.name));
        def.validate().map_err(|e| fail(e.to_string()))?;
        let valid = positive(dt)
            && config.loop_ratio >= 1.0
            && positive(config.motor_lag)
            && positive(config.max_pitch)
            && positive(config.max_bank)
            && config.accel.iter().all(|a| positive(*a))
            && config.max_differential_tilt >= 0.0
            && config.max_speed.iter().all(|s| positive(*s))
            && positive(config.design_density)
            && positive(config.gravity)
            && config.rate_bandwidth.is_none_or(positive);
        if !valid {
            return Err(fail(format!("tiltrotor controller step {dt}, config {config:?}")));
        }
        let t = Tiltrotor::new(def.clone(), dt);
        let (rho, g) = (config.design_density, config.gravity);
        let schedule = TiltSchedule::new(&t, &TrimLimits { max_pitch: config.max_pitch }, rho, g)?;
        let c = &def.controls;
        let tau_servo = [c.aileron.tau, c.elevator.tau, c.rudder.tau].into_iter().fold(0.0, f64::max);
        let k_rate = config.rate_bandwidth.unwrap_or(0.5 / (config.motor_lag + tau_servo));
        let k_att = k_rate / config.loop_ratio;
        let k_vel = k_att / config.loop_ratio;
        let k_pos = k_vel / config.loop_ratio;
        let max_thrust = (0..AXIAL.2).map(|i| t.max_thrust(AXIAL.0 + i as f64 * AXIAL.1, rho)).collect();
        let alpha_limit = def
            .surfaces
            .iter()
            .filter(|s| s.roll.cos().abs() > 0.5)
            .map(|s| s.alpha_stall - s.incidence)
            .fold(f64::INFINITY, f64::min)
            - 0.05;
        let sides = std::array::from_fn(|k| match def.rotors.get(k) {
            Some(r) if r.tilting && r.pivot.y.abs() > 1e-9 => r.pivot.y.signum(),
            _ => 0.0,
        });
        Ok(Self {
            config: config.clone(),
            dt,
            schedule: Arc::new(schedule),
            k_rate,
            k_att,
            k_vel,
            k_pos,
            i_rate: DVec3::ZERO,
            i_vel: DVec3::ZERO,
            heading: None,
            reference: None,
            airspeed: None,
            solution: None,
            tilt: 0.0,
            saturated: false,
            grounded: false,
            settle: 0.0,
            mass: t.mass(),
            hover_thrust: t.mass() * g / t.rotor_count() as f64,
            max_thrust: Arc::new(max_thrust),
            alpha_limit,
            sides,
        })
    }

    pub fn config(&self) -> &TiltrotorConfig {
        &self.config
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// The conversion schedule.
    pub fn schedule(&self) -> &TiltSchedule {
        &self.schedule
    }

    /// Rate, attitude, velocity and position bandwidths (rad/s).
    pub fn bandwidths(&self) -> [f64; 4] {
        [self.k_rate, self.k_att, self.k_vel, self.k_pos]
    }

    /// Mount tilt the schedule asked for in the last update (rad).
    pub fn tilt_command(&self) -> f64 {
        self.tilt
    }

    /// Total rotor thrust (N) and pitch (rad) the velocity loop asked for in the last update.
    pub fn thrust_pitch_command(&self) -> Option<(f64, f64)> {
        self.solution
    }

    /// Clear the integrators and references.
    pub fn reset(&mut self) {
        self.i_rate = DVec3::ZERO;
        self.i_vel = DVec3::ZERO;
        self.heading = None;
        self.reference = None;
        self.airspeed = None;
        self.solution = None;
        self.settle = 0.0;
    }

    /// Pilot input for `setpoint`.
    pub fn update(&mut self, setpoint: &TiltrotorSetpoint, t: &Tiltrotor) -> TiltrotorInput {
        // On the ground the gear holds the attitude: clear the rate and horizontal integrators
        // and hold the attitude as it stands.
        self.grounded = !t.contacts().is_empty();
        if self.grounded {
            self.i_rate = DVec3::ZERO;
            self.i_vel = DVec3::new(0.0, 0.0, self.i_vel.z);
        }
        match *setpoint {
            TiltrotorSetpoint::Raw(input) => {
                // The loops start afresh when a controlled mode takes over.
                self.reset();
                input.clamped()
            }
            TiltrotorSetpoint::Attitude { roll, pitch, yaw_rate, climb, airspeed } => {
                self.reference = None;
                self.attitude_mode(t, roll, pitch, yaw_rate, climb, airspeed)
            }
            TiltrotorSetpoint::Velocity { velocity, yaw_rate } => {
                self.airspeed = None;
                self.velocity_loop(t, velocity, yaw_rate, None)
            }
            TiltrotorSetpoint::Position { position, yaw } => {
                self.airspeed = None;
                let psi = euler(t.orientation()).2;
                let err = position - t.position();
                let [vh, vz] = self.config.max_speed;
                let horizontal = (err.truncate() * self.k_pos).clamp_length_max(vh);
                let vertical = (err.z * self.k_pos).clamp(-vz, vz);
                let v = DQuat::from_rotation_z(-psi) * horizontal.extend(vertical);
                self.velocity_loop(t, v, 0.0, Some(yaw))
            }
        }
    }

    /// Steady full-throttle thrust of one rotor at axial speed `v` (m/s) in the design air.
    fn max_thrust_at(&self, v: f64) -> f64 {
        let x = ((v - AXIAL.0) / AXIAL.1).clamp(0.0, (AXIAL.2 - 1) as f64);
        let i = (x.floor() as usize).min(AXIAL.2 - 2);
        let f = x - i as f64;
        self.max_thrust[i] + (self.max_thrust[i + 1] - self.max_thrust[i]) * f
    }

    fn frame(&self, t: &Tiltrotor, psi: f64) -> Frame {
        let flow = t.flow();
        let air = DQuat::from_rotation_z(-psi) * (t.orientation() * flow.velocity);
        let (omega, tilts) = (t.rotor_speeds(), t.tilts());
        let mut f = Frame {
            air,
            rates: flow.rates,
            density: flow.density,
            axes: [DVec3::Z; MAX_ROTORS],
            share: [0.0; MAX_ROTORS],
            in_plane: DVec3::ZERO,
            thrust: 0.0,
            drag: DVec3::ZERO,
        };
        let n = t.rotor_count();
        let mut thrusts = [0.0; MAX_ROTORS];
        for k in 0..n {
            let l = t.rotor_load(k, flow, omega[k], tilts[k], 1.0);
            f.axes[k] = TiltMount::axis(tilts[k]);
            f.in_plane += l.force - f.axes[k] * l.thrust;
            thrusts[k] = l.thrust.max(0.0);
            f.thrust += thrusts[k];
        }
        for k in 0..n {
            f.share[k] = if f.thrust > 1e-6 { thrusts[k] / f.thrust } else { 1.0 / n as f64 };
        }
        let heading = DQuat::from_rotation_z(-psi) * t.orientation();
        f.drag = heading * (t.aero_loads(flow, &t.channels()).0 + f.in_plane);
        f
    }

    /// Non-gravitational force (heading frame, N) with total rotor thrust `thrust` shared as
    /// now, at pitch `theta` and roll `phi`, in the present air-relative velocity.
    fn heading_force(&self, t: &Tiltrotor, f: &Frame, thrust: f64, theta: f64, phi: f64) -> DVec3 {
        let r = DQuat::from_rotation_y(-theta) * DQuat::from_rotation_x(phi);
        let air = AirData { density: f.density, ..AirData::default() };
        let flow = AirFlow::new(&air, r.inverse() * f.air, f.rates);
        let (aero, _) = t.aero_loads(&flow, &t.channels());
        let rotors: DVec3 = (0..t.rotor_count()).map(|k| f.axes[k] * (f.share[k] * thrust)).sum();
        r * (aero + rotors + f.in_plane)
    }

    /// Total thrust (N) and pitch (rad) that give the forward and upward components of `want`
    /// (N, heading frame) at roll `phi`, within the pitch and angle-of-attack limits.
    fn thrust_pitch(&mut self, t: &Tiltrotor, f: &Frame, want: DVec3, phi: f64, theta: f64) -> (f64, f64) {
        let weight = self.mass * self.config.gravity;
        let (mut thrust, mut pitch) = self.solution.unwrap_or((f.thrust, theta));
        let residual = |thrust: f64, pitch: f64| {
            let force = self.heading_force(t, f, thrust, pitch, phi);
            DVec2::new(force.x - want.x, force.z - want.z)
        };
        let (dt, dp) = (0.01 * weight, 1e-3);
        for _ in 0..3 {
            let r = residual(thrust, pitch);
            let jt = (residual(thrust + dt, pitch) - r) / dt;
            let jp = (residual(thrust, pitch + dp) - r) / dp;
            let det = jt.x * jp.y - jp.x * jt.y;
            if det.is_nan() || det.abs() < 1e-9 {
                break;
            }
            thrust += ((jp.x * r.y - r.x * jp.y) / det).clamp(-weight, weight);
            pitch += ((r.x * jt.y - jt.x * r.y) / det).clamp(-0.2, 0.2);
        }
        // Pitch limits, and the wing kept below the stall once it flies.
        let limit = self.config.max_pitch;
        let mut top = limit;
        if f.air.x > 0.5 * self.schedule.stall_speed() {
            top = top.min(theta + self.alpha_limit - t.flow().alpha);
        }
        let top = top.max(-limit);
        let flow = t.flow();
        let density_ratio = f.density / self.config.design_density;
        let most: f64 = (0..t.rotor_count())
            .map(|k| {
                let v = flow.at(t.def().rotors[k].hub(t.tilts()[k])).dot(f.axes[k]);
                self.max_thrust_at(v) * density_ratio
            })
            .sum();
        if !thrust.is_finite() || !pitch.is_finite() {
            (thrust, pitch) = (f.thrust, theta);
        }
        let mut saturated = false;
        if pitch < -limit || pitch > top {
            // The thrust alone does what it can, the vertical force first.
            saturated = true;
            pitch = pitch.clamp(-limit, top);
            let r = residual(thrust, pitch);
            let d = (residual(thrust + dt, pitch) - r) / dt;
            let w = DVec2::new(1.0, 16.0);
            let dd = (d * d).dot(w);
            if dd > 1e-12 {
                thrust -= (d * r).dot(w) / dd;
            }
        }
        if thrust < 0.0 || thrust > most {
            // The pitch alone holds the vertical force.
            saturated = true;
            thrust = thrust.clamp(0.0, most);
            for _ in 0..2 {
                let r = residual(thrust, pitch).y;
                let d = (residual(thrust, pitch + dp).y - r) / dp;
                if d.abs() > 1e-9 {
                    pitch = (pitch - (r / d).clamp(-0.2, 0.2)).clamp(-limit, top);
                }
            }
        }
        self.saturated = saturated;
        self.solution = Some((thrust, pitch));
        (thrust, pitch)
    }

    /// Heading-frame velocity: the specific force it asks for, from the total thrust and the
    /// pitch (forward and up) and the roll (sideways), with the mounts on the schedule at the
    /// reference airspeed.
    fn velocity_loop(&mut self, t: &Tiltrotor, velocity: DVec3, yaw_rate: f64, heading: Option<f64>) -> TiltrotorInput {
        let g = self.config.gravity;
        let (phi, theta, psi) = euler(t.orientation());
        let v = DQuat::from_rotation_z(-psi) * t.lin_vel_world();
        let prev = self.reference.unwrap_or(v);
        let [ah, av] = self.config.accel;
        let step = DVec3::new(ah, ah, av) * self.dt;
        // The reference is kept within reach: no further from the velocity than the error
        // that asks for the configured acceleration.
        let leash = DVec3::new(ah, ah, av) / self.k_vel;
        let next = (prev + (velocity - prev).clamp(-step, step)).clamp(v - leash, v + leash);
        self.reference = Some(next);
        let feed = ((next - prev) / self.dt).clamp(-DVec3::new(ah, ah, av), DVec3::new(ah, ah, av));
        let err = next - v;
        let f = self.frame(t, psi);
        // On the wing, sideways velocity is not flown: the yaw rate turns, coordinated.
        let vs = self.schedule.stall_speed();
        let wing = self.schedule.wing_share(f.air.x);
        let mut acc = err * self.k_vel + self.i_vel + feed;
        acc.y = (1.0 - wing) * acc.y + wing * f.air.x.max(0.0) * yaw_rate;
        // The schedule tilts the mounts for the airspeed the reference asks for.
        let tilt = self.schedule.tilt(f.air.x + next.x - v.x);
        let want = (acc + DVec3::Z * g) * self.mass;
        let max_bank = self.config.max_bank;
        // In hover the bank gives the sideways force the airframe does not; on the wing the
        // airframe's side force is sideslip, which the turn coordination below removes.
        let side = want.y - (1.0 - wing) * f.drag.y;
        let roll = if self.grounded { phi } else { (-side).atan2(want.z).clamp(-max_bank, max_bank) };
        let (thrust, pitch) = self.thrust_pitch(t, &f, want, roll, theta);
        let pitch = if self.grounded { theta } else { pitch };
        let thrust = self.settled(thrust, velocity.z);
        // On the wing the pitch follows the flight path: its rate, from the vertical
        // acceleration asked for, leads the attitude loop.
        let path_rate = if self.grounded { 0.0 } else { wing * acc.z / f.air.x.max(vs) };
        // Turn coordination: the nose follows the air-relative velocity on the wing.
        let yaw_rate = yaw_rate - wing * self.k_att * t.flow().beta;
        let rates = self.attitude_loop(t, [roll, pitch], path_rate, yaw_rate, heading);
        let out = self.rate_loop(t, rates, thrust, tilt);
        // Integrate (k²/4 per unit error) up to 0.3 g.
        let k2 = 0.25 * self.k_vel * self.k_vel * self.dt;
        let lim = 0.3 * g;
        let mut next = (self.i_vel + err * k2).clamp(DVec3::splat(-lim), DVec3::splat(lim));
        if self.saturated {
            // Only the climb rate keeps integrating: it comes first when a limit is met.
            next = DVec3::new(self.i_vel.x, self.i_vel.y, next.z);
        }
        // Nor while the reference is still on its way: the lag behind a ramp is not an offset.
        let moving = feed.abs().cmpgt(DVec3::splat(1e-9));
        next = DVec3::select(moving, self.i_vel, next);
        if next.is_finite() && !self.grounded {
            self.i_vel = next;
        }
        out
    }

    /// Attitude with airspeed: the pilot's roll and pitch, the mounts on the schedule at the
    /// airspeed reference, and the thrust along its own direction for the climb rate and the
    /// airspeed (the climb in hover, the airspeed on the wing).
    fn attitude_mode(
        &mut self,
        t: &Tiltrotor,
        roll: f64,
        pitch: f64,
        yaw_rate: f64,
        climb: f64,
        airspeed: f64,
    ) -> TiltrotorInput {
        let g = self.config.gravity;
        let psi = euler(t.orientation()).2;
        let f = self.frame(t, psi);
        let prev = self.airspeed.unwrap_or(f.air.x);
        let step = self.config.accel[0] * self.dt;
        let next = prev + (airspeed - prev).clamp(-step, step);
        self.airspeed = Some(next);
        let feed = (next - prev) / self.dt;
        let tilt = self.schedule.tilt(next);
        let vz = t.lin_vel_world().z;
        let acc = DVec3::new((next - f.air.x) * self.k_vel + feed, 0.0, (climb - vz) * self.k_vel);
        let want = (acc + DVec3::Z * g) * self.mass;
        let base = self.heading_force(t, &f, 0.0, pitch, roll);
        let per_newton = self.heading_force(t, &f, 1.0, pitch, roll) - base;
        let (d, r) = (DVec2::new(per_newton.x, per_newton.z), DVec2::new(want.x - base.x, want.z - base.z));
        // In hover the thrust holds the climb rate; on the wing, the airspeed.
        let wing = self.schedule.wing_share(f.air.x);
        let w = DVec2::new(wing + 1e-3, 1.0 - wing + 1e-3);
        let dd = (d * d).dot(w);
        let thrust = if dd > 1e-12 { ((d * r).dot(w) / dd).max(0.0) } else { f.thrust };
        let thrust = self.settled(thrust, climb);
        let rates = self.attitude_loop(t, [roll, pitch], 0.0, yaw_rate, None);
        self.rate_loop(t, rates, thrust, tilt)
    }

    /// `thrust` brought down to nothing over [`SETTLE_TIME`] while on the ground with no climb
    /// asked for: the rotors would otherwise keep carrying the weight and skate the gear.
    fn settled(&mut self, thrust: f64, climb: f64) -> f64 {
        self.settle = if self.grounded && climb <= 0.0 { (self.settle + self.dt / SETTLE_TIME).min(1.0) } else { 0.0 };
        thrust * (1.0 - self.settle)
    }

    /// Body rate references for Euler-angle targets (with the pitch target's rate) and a yaw
    /// rate, holding the heading the yaw rate integrates to (or `heading`).
    fn attitude_loop(
        &mut self,
        t: &Tiltrotor,
        [roll, pitch]: [f64; 2],
        pitch_rate: f64,
        yaw_rate: f64,
        heading: Option<f64>,
    ) -> DVec3 {
        let (phi, theta, psi) = euler(t.orientation());
        let (roll, pitch) = if self.grounded { (phi, theta) } else { (roll, pitch) };
        let reference = heading.unwrap_or_else(|| self.heading.unwrap_or(psi) + yaw_rate * self.dt);
        // Hold the reference within reach so that it does not wind up.
        let lag = wrap_angle(reference - psi).clamp(-HEADING_LAG, HEADING_LAG);
        self.heading = Some(wrap_angle(psi + lag));
        let yaw_rate = yaw_rate + self.k_att * lag;
        let phi_dot = self.k_att * wrap_angle(roll - phi);
        let theta_dot = self.k_att * (pitch - theta) + pitch_rate;
        // Body rates of Euler rates for R_z(ψ)·R_y(−θ)·R_x(φ).
        DVec3::new(
            phi_dot + yaw_rate * theta.sin(),
            -theta_dot * phi.cos() + yaw_rate * theta.cos() * phi.sin(),
            theta_dot * phi.sin() + yaw_rate * theta.cos() * phi.cos(),
        )
    }

    /// Incremental inversion of the moment model: the moment the rate error asks for, less the
    /// moment now, and the total thrust, allocated over the rotor thrusts, the differential
    /// tilt and the surfaces; the mounts' common tilt is `tilt`.
    fn rate_loop(&mut self, t: &Tiltrotor, rates: DVec3, thrust: f64, tilt: f64) -> TiltrotorInput {
        self.tilt = tilt;
        let def = t.def();
        let n = t.rotor_count();
        let w = t.ang_vel_body();
        let err = rates - w;
        let inertia = t.inertia();
        let wanted = inertia * (err * self.k_rate + self.i_rate) + w.cross(inertia * w + t.rotor_momentum_body());
        let flow = *t.flow();
        let rho = flow.density;
        let (omega, tilts, channels) = (t.rotor_speeds(), t.tilts(), t.channels());
        let reaction = |k: usize, tilt: f64, torque: f64| -TiltMount::axis(tilt) * (def.rotors[k].sense * torque);
        let mut b = [[0.0; EFFECTORS]; OBJECTIVES];
        let (mut lo, mut hi, mut weight) = ([0.0; EFFECTORS], [0.0; EFFECTORS], [1.0; EFFECTORS]);
        let set = |b: &mut [[f64; EFFECTORS]; OBJECTIVES], j: usize, m: DVec3, thrust: f64| {
            b[0][j] = m.x;
            b[1][j] = m.y;
            b[2][j] = m.z;
            b[3][j] = thrust;
        };
        let mut moment = t.aero_loads(&flow, &channels).1;
        let (mut thrust_now, mut loads) = (0.0, [RotorLoad::default(); MAX_ROTORS]);
        let t_ref = self.hover_thrust;
        let density_ratio = rho / self.config.design_density;
        for k in 0..n {
            let l = t.rotor_load(k, &flow, omega[k], tilts[k], 1.0);
            loads[k] = l;
            moment += l.moment + reaction(k, tilts[k], l.torque);
            thrust_now += l.thrust;
            // Moment per newton of thrust: its lever and the torque that comes with it.
            let h = 1e-3 * omega[k] + 1.0;
            let up = t.rotor_load(k, &flow, omega[k] + h, tilts[k], 1.0);
            let down = t.rotor_load(k, &flow, (omega[k] - h).max(0.0), tilts[k], 1.0);
            let dt = up.thrust - down.thrust;
            let per_newton = if dt > 1e-9 {
                (up.moment + reaction(k, tilts[k], up.torque) - down.moment - reaction(k, tilts[k], down.torque)) / dt
            } else {
                def.rotors[k].hub(tilts[k]).cross(TiltMount::axis(tilts[k]))
            };
            set(&mut b, k, per_newton, 1.0);
            weight[k] = 1.0 / (t_ref * t_ref);
            hi[k] = self.max_thrust_at(l.v_axial) * density_ratio - l.thrust;
            lo[k] = (MIN_THRUST * t_ref - l.thrust).min(hi[k]);
        }
        // Differential tilt (left mounts forward, right back), cut short near the ends of the
        // mounts' range.
        let span: f64 = self.sides.iter().map(|s| s.abs()).sum();
        let diff_now = if span > 0.0 { (0..n).map(|k| self.sides[k] * tilts[k]).sum::<f64>() / span } else { 0.0 };
        let range = &def.controls.tilt;
        let room = self.config.max_differential_tilt.min((tilt - range.min).min(range.max - tilt).max(0.0));
        if span > 0.0 && room > 1e-3 {
            let h = 0.01;
            let at = |k: usize, tau: f64| {
                let l = t.rotor_load(k, &flow, omega[k], tau, 1.0);
                l.moment + reaction(k, tau, l.torque)
            };
            let col: DVec3 = (0..n)
                .filter(|&k| self.sides[k] != 0.0)
                .map(|k| (at(k, tilts[k] + h) - at(k, tilts[k] - h)) * (self.sides[k] / (2.0 * h)))
                .sum();
            set(&mut b, TILT, col, 0.0);
            weight[TILT] = 1.0 / (0.1 * 0.1);
            lo[TILT] = (-room - diff_now).min(0.0);
            hi[TILT] = (room - diff_now).max(0.0);
        }
        // Surfaces.
        let ch = def.controls.surface_channels();
        for c in 0..3 {
            let h = 0.01;
            let (mut up, mut down) = (channels, channels);
            up[c] += h;
            down[c] -= h;
            let col = (t.aero_loads(&flow, &up).1 - t.aero_loads(&flow, &down).1) / (2.0 * h);
            set(&mut b, SURFACES + c, col, 0.0);
            let travel = 0.5 * (ch[c].max - ch[c].min);
            weight[SURFACES + c] = 1.0 / (travel * travel);
            lo[SURFACES + c] = (ch[c].min - channels[c]).min(0.0);
            hi[SURFACES + c] = (ch[c].max - channels[c]).max(0.0);
        }
        let demand = wanted - moment;
        let v = [demand.x, demand.y, demand.z, thrust - thrust_now];
        let u = allocate(&b, &weight, &v, &lo, &hi, &DROP_ORDER);
        let saturated = (0..EFFECTORS).any(|j| hi[j] > lo[j] && (u[j] <= lo[j] || u[j] >= hi[j]) && u[j] != 0.0);

        // Rotor thrusts → speeds → throttles; mounts; surfaces.
        let mut input = TiltrotorInput::default();
        for k in 0..n {
            let target = (loads[k].thrust + u[k]).max(0.0);
            let speed = t.rotor_speed_for(target, loads[k].v_axial, rho, omega[k]);
            input.throttle[k] =
                t.throttle_for(omega[k], speed, loads[k].v_axial, rho, self.config.motor_lag).clamp(0.0, 1.0);
        }
        let diff = diff_now + u[TILT];
        for k in 0..n {
            input.tilt[k] = tilt + self.sides[k] * diff;
        }
        // Surfaces without authority (below half the stall speed) are centred rather than
        // left where the last flight put them.
        let vs = self.schedule.stall_speed();
        let authority = smoothstep(flow.dynamic_pressure / (0.125 * rho * vs * vs));
        let [a, e, r] =
            std::array::from_fn(|c| ch[c].input(authority * (channels[c] + u[SURFACES + c])).clamp(-1.0, 1.0));
        (input.aileron, input.elevator, input.rudder) = (a, e, r);
        // Integrate (k²/4 per unit error: critically damped) unless an effector saturates.
        if !saturated {
            let next = self.i_rate + err * (0.25 * self.k_rate * self.k_rate * self.dt);
            if next.is_finite() {
                self.i_rate = next;
            }
        }
        input
    }
}
