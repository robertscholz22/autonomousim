//! Multirotor flight control: a PX4-style cascade in physical units.
//!
//! ```text
//! position ─P→ velocity ─P+DOB→ thrust vector → attitude ─P→ body rates ─PID→ torque ─B⁺→ rotor speeds
//! ```
//!
//! Rate and attitude loops run every control step (the physics rate); velocity and position
//! loops at [`Tuning::outer_rate`]. [`Setpoint`] enters the cascade at any level, and
//! [`ActionMap`] turns normalised policy actions into setpoints. The controller only knows the
//! nominal vehicle definition, so randomised parameters are disturbances it must reject.
//!
//! The velocity loop's disturbance observer (DOB, see [`PositionController`]) compares the
//! measured acceleration with the one the rotors produced, which the controller predicts with
//! its own model of the motor lag. It estimates wind, drag and model errors (mass, air density,
//! motor constants) and pauses while the vehicle rests on something solid.

pub mod action;
pub mod allocation;
pub mod attitude;
pub mod position;
pub mod rate;
pub mod tuning;

pub use action::{ActionLimits, ActionMap, ActionMode};
pub use allocation::Allocator;
pub use attitude::AttitudeController;
pub use position::{PositionController, PositionGains, PositionLimits, attitude_from_thrust, limit_tilt};
pub use rate::{RateController, RateGains};
pub use tuning::{ControllerConfig, Gains, Limits, Tuning};

use crate::ControlError;
use autonomousim_core::geometry::HitKind;
use autonomousim_core::math::quat::{exp_map, from_yaw, wrap_angle, yaw};
use autonomousim_vehicles::multirotor::{MAX_ROTORS, Multirotor, MultirotorDef};
use glam::{DQuat, DVec2, DVec3};
use serde::{Deserialize, Serialize};

/// What the controller knows about the vehicle (ground truth or an estimate).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateEstimate {
    /// World position (m).
    pub position: DVec3,
    /// World velocity (m/s).
    pub velocity: DVec3,
    /// Body → world rotation.
    pub attitude: DQuat,
    /// Body angular velocity (rad/s).
    pub rates: DVec3,
    /// Resting on or pushing against terrain, a solid obstacle or another agent.
    pub ground_contact: bool,
}

impl StateEstimate {
    /// Ground truth of a simulated vehicle.
    pub fn of(v: &Multirotor) -> Self {
        let attitude = v.orientation();
        Self {
            position: v.position(),
            velocity: attitude * v.lin_vel_body(),
            attitude,
            rates: v.ang_vel_body(),
            ground_contact: v.contacts().iter().any(|c| {
                c.normal_force > 0.0 && matches!(c.kind, HitKind::Terrain | HitKind::Solid(_) | HitKind::Agent(_))
            }),
        }
    }
}

/// Frame of a velocity setpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Frame {
    World,
    /// World frame rotated by the current heading: x forward, y left, z up.
    #[default]
    Heading,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum YawCommand {
    /// Heading (rad, ENU: 0 = east, counter-clockwise).
    Angle(f64),
    /// Heading rate (rad/s), integrated into a heading setpoint.
    Rate(f64),
}

/// Input to the cascade, at the level it enters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Setpoint {
    /// Rotor speeds (rad/s), passed through.
    Motors([f64; MAX_ROTORS]),
    /// Collective thrust (N) and body rates (rad/s).
    Ctbr { thrust: f64, rates: DVec3 },
    /// Tilt as a rotation vector `(x, y)` in the heading frame (rad; its length is the tilt
    /// angle), heading, and collective thrust (N).
    Attitude { tilt: DVec2, yaw: YawCommand, thrust: f64 },
    /// Velocity (m/s) and heading.
    Velocity { velocity: DVec3, frame: Frame, yaw: YawCommand },
    /// World position (m) and heading.
    Position { position: DVec3, yaw: YawCommand },
}

impl Setpoint {
    /// The action mode that enters the cascade at this level.
    pub fn mode(&self) -> ActionMode {
        match self {
            Setpoint::Motors(_) => ActionMode::Motors,
            Setpoint::Ctbr { .. } => ActionMode::Ctbr,
            Setpoint::Attitude { .. } => ActionMode::Attitude,
            Setpoint::Velocity { .. } => ActionMode::Velocity,
            Setpoint::Position { .. } => ActionMode::Position,
        }
    }
}

/// Largest lead of an integrated heading setpoint over the actual heading (rad), so that a
/// saturated yaw axis cannot wind the setpoint up.
const YAW_LEAD: f64 = 0.5;

/// Cascaded multirotor controller. Call [`update`](Self::update) once per control step.
#[derive(Clone, Debug)]
pub struct MultirotorController {
    n: usize,
    dt: f64,
    gravity: f64,
    mass: f64,
    config: ControllerConfig,
    gains: Gains,
    alloc: Allocator,
    rate: RateController,
    attitude: AttitudeController,
    position: PositionController,
    motors: MotorModel,
    // Loop state.
    mode: Option<ActionMode>,
    outer_phase: u32,
    yaw_sp: Option<f64>,
    att_sp: DQuat,
    thrust_sp: f64,
    yaw_ff: f64,
    /// Frame of the velocity reference (velocity mode).
    velocity_frame: Option<Frame>,
    sat_pos: [bool; 3],
    sat_neg: [bool; 3],
    // Outputs of the last update.
    rate_sp: DVec3,
    torque_sp: DVec3,
    thrusts: [f64; MAX_ROTORS],
}

impl MultirotorController {
    /// Controller for `def` running every `dt` seconds.
    pub fn new(def: &MultirotorDef, dt: f64, config: &ControllerConfig) -> Result<Self, ControlError> {
        config.validate()?;
        if !(dt > 0.0 && dt.is_finite()) {
            return Err(ControlError::InvalidConfig(format!("control period {dt}")));
        }
        let density = config.design_density.unwrap_or(def.rotor.reference_density);
        let alloc = Allocator::new(def, density, config.airmode)?;
        let gains = Gains::derive(def, &config.tuning, dt, alloc.torque_authority());
        let l = &config.limits;
        let t_max = alloc.max_collective();
        let limits = PositionLimits {
            vel_xy: l.vel_xy,
            vel_up: l.vel_up,
            vel_down: l.vel_down,
            tilt: l.tilt,
            thrust_min: l.thrust_min * t_max,
            thrust_max: t_max,
            xy_margin: l.xy_margin * t_max,
            accel_xy: l.accel_xy,
            accel_up: l.accel_up,
            accel_down: l.accel_down,
        };
        Ok(Self {
            n: def.rotors.len(),
            dt,
            gravity: config.gravity,
            mass: def.body.mass,
            rate: RateController::new(gains.rate, def.body.inertia, dt),
            attitude: AttitudeController::new(gains.attitude, config.tuning.yaw_weight, l.rate),
            position: PositionController::new(gains.position, limits, def.body.mass, config.gravity),
            motors: MotorModel::new(def, dt, density, config.gravity),
            config: config.clone(),
            gains,
            alloc,
            mode: None,
            outer_phase: 0,
            yaw_sp: None,
            att_sp: DQuat::IDENTITY,
            thrust_sp: 0.0,
            yaw_ff: 0.0,
            velocity_frame: None,
            sat_pos: [false; 3],
            sat_neg: [false; 3],
            rate_sp: DVec3::ZERO,
            torque_sp: DVec3::ZERO,
            thrusts: [0.0; MAX_ROTORS],
        })
    }

    pub fn config(&self) -> &ControllerConfig {
        &self.config
    }

    pub fn gains(&self) -> &Gains {
        &self.gains
    }

    pub fn allocator(&self) -> &Allocator {
        &self.alloc
    }

    pub fn num_rotors(&self) -> usize {
        self.n
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Nominal hover thrust (N).
    pub fn hover_thrust(&self) -> f64 {
        self.mass * self.gravity
    }

    /// Collective thrust at full rotor speed (N).
    pub fn max_thrust(&self) -> f64 {
        self.alloc.max_collective()
    }

    /// Clear all loop states (integrators, disturbance estimate, held setpoints); the motor
    /// model restarts at hover speed.
    pub fn reset(&mut self) {
        self.rate.reset();
        self.position.reset();
        self.motors.reset();
        self.mode = None;
        self.outer_phase = 0;
        self.yaw_sp = None;
        self.att_sp = DQuat::IDENTITY;
        self.thrust_sp = 0.0;
        self.yaw_ff = 0.0;
        self.velocity_frame = None;
        self.sat_pos = [false; 3];
        self.sat_neg = [false; 3];
        self.rate_sp = DVec3::ZERO;
        self.torque_sp = DVec3::ZERO;
        self.thrusts = [0.0; MAX_ROTORS];
    }

    /// Set the motor model to measured rotor speeds (rad/s), e.g. after a reset of the vehicle.
    pub fn sync_motors(&mut self, omega: &[f64]) {
        self.motors.omega[..self.n].copy_from_slice(&omega[..self.n]);
    }

    /// Update the feasible rotor speed range (rad/s), e.g. from
    /// [`Multirotor::speed_range`] as the battery sags.
    pub fn set_speed_limits(&mut self, omega_min: f64, omega_max: f64) {
        self.alloc.set_speed_limits(omega_min, omega_max);
        self.motors.omega_min = omega_min;
        self.motors.omega_max = omega_max;
        let t_max = self.alloc.max_collective();
        let l = &self.config.limits;
        self.position.set_thrust_limits(l.thrust_min * t_max, t_max, l.xy_margin * t_max);
    }

    /// One control step: rotor speed commands (rad/s) for setpoint `sp` given state `est`.
    pub fn update(&mut self, sp: &Setpoint, est: &StateEstimate, omega_cmd: &mut [f64]) {
        let mode = sp.mode();
        if self.mode != Some(mode) {
            // Entering a mode: outer loops start fresh; the rate integrator carries over unless
            // the rate loop was bypassed.
            if matches!(self.mode, None | Some(ActionMode::Motors)) {
                self.rate.reset();
            }
            self.position.reset();
            self.yaw_sp = None;
            self.velocity_frame = None;
            self.outer_phase = 0;
            self.mode = Some(mode);
        }
        let outer = self.outer_phase == 0;
        self.outer_phase = (self.outer_phase + 1) % self.gains.outer_every;
        let outer_dt = self.dt * f64::from(self.gains.outer_every);

        let (rate_sp, thrust) = match *sp {
            Setpoint::Motors(w) => {
                omega_cmd[..self.n].copy_from_slice(&w[..self.n]);
                self.motors.advance(&w[..self.n]);
                return;
            }
            Setpoint::Ctbr { thrust, rates } => (rates, thrust),
            Setpoint::Attitude { tilt, yaw, thrust } => {
                let (yaw_sp, ff) = self.yaw_setpoint(yaw, est, self.dt);
                let tilt = tilt.clamp_length_max(std::f64::consts::PI);
                self.att_sp = from_yaw(yaw_sp) * exp_map(tilt.extend(0.0));
                (self.attitude.update(est.attitude, self.att_sp, ff), thrust)
            }
            Setpoint::Velocity { velocity, frame, yaw } => {
                if outer {
                    // Shape the command in its own frame, so that a heading-frame reference
                    // turns with the vehicle instead of lagging behind it.
                    if self.velocity_frame != Some(frame) {
                        self.position.reset_reference();
                        self.velocity_frame = Some(frame);
                    }
                    let rot = match frame {
                        Frame::World => DQuat::IDENTITY,
                        Frame::Heading => from_yaw(self::yaw(est.attitude)),
                    };
                    let cmd = self.position.limit_velocity(velocity);
                    let (v, a) = self.position.shape(cmd, rot.inverse() * est.velocity, outer_dt);
                    let (v, mut a) = (rot * v, rot * a);
                    if frame == Frame::Heading {
                        // Centripetal acceleration of a reference fixed in the turning frame.
                        a += DVec3::Z.cross(v) * (est.attitude * est.rates).z;
                    }
                    self.outer_loop(v, a, yaw, est, outer_dt);
                }
                self.position.record_applied(self.motors.acceleration(est.attitude));
                (self.attitude.update(est.attitude, self.att_sp, self.yaw_ff), self.thrust_sp)
            }
            Setpoint::Position { position, yaw } => {
                if outer {
                    let v = self.position.velocity_setpoint(position, est.position);
                    self.outer_loop(v, DVec3::ZERO, yaw, est, outer_dt);
                }
                self.position.record_applied(self.motors.acceleration(est.attitude));
                (self.attitude.update(est.attitude, self.att_sp, self.yaw_ff), self.thrust_sp)
            }
        };

        let torque = self.rate.update(rate_sp, est.rates, self.sat_pos, self.sat_neg);
        self.alloc.allocate(torque, thrust, &mut self.thrusts);
        let (achieved, _) = self.alloc.wrench(&self.thrusts[..self.n]);
        let unallocated = (torque - achieved).to_array();
        let tol = (1e-3 * self.alloc.torque_authority()).to_array();
        for k in 0..3 {
            self.sat_pos[k] = unallocated[k] > tol[k];
            self.sat_neg[k] = unallocated[k] < -tol[k];
        }
        for (w, &f) in omega_cmd.iter_mut().zip(&self.thrusts[..self.n]) {
            *w = self.alloc.speed(f);
        }
        self.motors.advance(&omega_cmd[..self.n]);
        self.rate_sp = rate_sp;
        self.torque_sp = torque;
        self.thrust_sp = thrust;
    }

    /// Velocity loop (setpoint and feedforward acceleration, world) and heading. Velocity
    /// commands pass through the shaped reference first; the position loop's setpoints are
    /// continuous already, and the reference would add lag inside the position loop.
    fn outer_loop(&mut self, vel_sp: DVec3, acc_ff: DVec3, yaw: YawCommand, est: &StateEstimate, dt: f64) {
        let (yaw_sp, ff) = self.yaw_setpoint(yaw, est, dt);
        let thrust = self.position.update(vel_sp, acc_ff, est.velocity, dt, !est.ground_contact);
        self.att_sp = attitude_from_thrust(thrust, yaw_sp);
        self.thrust_sp = thrust.length();
        self.yaw_ff = ff;
    }

    /// Heading setpoint and feedforward rate; a rate command is integrated over `dt`.
    fn yaw_setpoint(&mut self, cmd: YawCommand, est: &StateEstimate, dt: f64) -> (f64, f64) {
        match cmd {
            YawCommand::Angle(a) => {
                self.yaw_sp = Some(a);
                (a, 0.0)
            }
            YawCommand::Rate(r) => {
                let heading = yaw(est.attitude);
                let next = self.yaw_sp.unwrap_or(heading) + r * dt;
                let sp = wrap_angle(heading + wrap_angle(next - heading).clamp(-YAW_LEAD, YAW_LEAD));
                self.yaw_sp = Some(sp);
                (sp, r)
            }
        }
    }

    // ---------------------------------------------------------------- diagnostics

    /// Body-rate setpoint of the last update (rad/s).
    pub fn rate_setpoint(&self) -> DVec3 {
        self.rate_sp
    }

    /// Attitude setpoint of the last update in the attitude, velocity and position modes.
    pub fn attitude_setpoint(&self) -> DQuat {
        self.att_sp
    }

    /// Collective thrust setpoint of the last update (N).
    pub fn thrust_setpoint(&self) -> f64 {
        self.thrust_sp
    }

    /// Torque requested from the allocation in the last update (N·m).
    pub fn torque_setpoint(&self) -> DVec3 {
        self.torque_sp
    }

    /// Allocated rotor thrusts of the last update (N).
    pub fn rotor_thrusts(&self) -> &[f64] {
        &self.thrusts[..self.n]
    }

    /// Rate-loop integrator (rad/s²).
    pub fn rate_integral(&self) -> DVec3 {
        self.rate.integral()
    }

    /// Disturbance acceleration estimated by the velocity loop (world, m/s²).
    pub fn disturbance_estimate(&self) -> DVec3 {
        self.position.disturbance()
    }

    /// Rotor speeds the motor model predicts for the next physics step (rad/s).
    pub fn motor_estimate(&self) -> &[f64] {
        &self.motors.omega[..self.n]
    }
}

/// The controller's copy of the nominal motor lag, to predict the force the rotors produce.
///
/// Mirrors [`Multirotor`]: in each step the rotors produce thrust from the current speed, and
/// the speed then moves towards the command with an exact first-order response.
#[derive(Clone, Debug)]
struct MotorModel {
    n: usize,
    omega: [f64; MAX_ROTORS],
    hover: f64,
    omega_min: f64,
    omega_max: f64,
    decay_up: f64,
    decay_down: f64,
    /// Body force per squared rotor speed (N·s²).
    force: [DVec3; MAX_ROTORS],
    inv_mass: f64,
    gravity: f64,
}

impl MotorModel {
    fn new(def: &MultirotorDef, dt: f64, density: f64, gravity: f64) -> Self {
        let r = &def.rotor;
        let k = r.k_thrust * density / r.reference_density;
        let mut force = [DVec3::ZERO; MAX_ROTORS];
        for (f, m) in force.iter_mut().zip(&def.rotors) {
            *f = m.axis * k;
        }
        let hover = def.hover_omega(gravity, density).clamp(r.omega_min, r.omega_max);
        Self {
            n: def.rotors.len(),
            omega: [hover; MAX_ROTORS],
            hover,
            omega_min: r.omega_min,
            omega_max: r.omega_max,
            decay_up: (-dt / r.tau_up).exp(),
            decay_down: (-dt / r.tau_down).exp(),
            force,
            inv_mass: 1.0 / def.body.mass,
            gravity,
        }
    }

    fn reset(&mut self) {
        self.omega = [self.hover; MAX_ROTORS];
    }

    /// World acceleration from rotor thrust and gravity in the coming physics step.
    fn acceleration(&self, attitude: DQuat) -> DVec3 {
        let f: DVec3 = (0..self.n).map(|i| self.force[i] * (self.omega[i] * self.omega[i])).sum();
        attitude * f * self.inv_mass - DVec3::new(0.0, 0.0, self.gravity)
    }

    fn advance(&mut self, cmd: &[f64]) {
        for (w, &c) in self.omega.iter_mut().zip(cmd) {
            let c = if c.is_finite() { c.clamp(self.omega_min, self.omega_max) } else { *w };
            let decay = if c > *w { self.decay_up } else { self.decay_down };
            *w = c + (*w - c) * decay;
        }
    }
}
