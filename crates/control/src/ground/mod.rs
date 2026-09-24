//! Ground vehicle control: speed, curvature and yaw-rate loops that produce driver input
//! ([`DriveInput`]) for a [`Wheeled`] vehicle, and the normalised action modes.
//!
//! ```text
//! speed ─PI→ acceleration demand ─inverse powertrain→ throttle / motor commands / brake
//! curvature ─bicycle model + ∫ curvature error→ steering
//! (v, ω) ─kinematics + ∫ yaw-rate error→ side wheel speeds ─PI→ motor commands   (side drives)
//! ```
//!
//! The speed loop commands an acceleration. The allocator inverts the nominal powertrain to
//! reach it: for an engine, the throttle that gives the required torque at the current engine
//! speed and gear (and the brake where engine braking is not enough); for electric motors, the
//! command that gives the required motor torque (for DC motors, including the back-EMF term).
//! The demand is clamped to what the powertrain and brakes can give at the current state, and
//! the integrator runs only near the target (integral separation), so it trims grades, drag
//! and model errors without winding up in large steps. Traction control fades the drive force
//! out when the driven wheels slip: without it, launches excite the lightly damped wheel-spin
//! mode of the clutchless driveline. Side drives add integrals on speed and yaw rate for the
//! slip of skid steering, held while a motor saturates. Gains follow from time constants in
//! [`GroundConfig`] and the vehicle's nominal mass and inertia, so one configuration fits cars
//! and robots.
//!
//! Automatic gear selection (combustion) engages reverse only near standstill: a negative
//! speed request brakes a vehicle still rolling forward first.

pub mod action;

pub use action::{GroundActionLimits, GroundActionMap, GroundActionMode, PerWheelChannel};

use crate::ControlError;
use autonomousim_vehicles::ground::{
    DriveInput, MAX_WHEELS, MotorSide, PowertrainDef, WheelCommands, Wheeled, WheeledDef,
};
use glam::DVec3;
use serde::{Deserialize, Serialize};

const RPM: f64 = std::f64::consts::PI / 30.0;

/// What the controller knows about the vehicle (ground truth or an estimate).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GroundEstimate {
    /// Chassis velocity in body axes (m/s).
    pub velocity_body: DVec3,
    /// Body yaw rate (rad/s).
    pub yaw_rate: f64,
    /// Wheel spin speeds (rad/s), by wheel.
    pub wheel_spin: [f64; MAX_WHEELS],
    /// Selected gear (combustion; 0 for electric drives) and engine speed (rad/s).
    pub gear: i32,
    pub engine_speed: f64,
}

impl GroundEstimate {
    /// Ground truth of a simulated vehicle.
    pub fn of(v: &Wheeled) -> Self {
        let mut wheel_spin = [0.0; MAX_WHEELS];
        for (s, w) in wheel_spin.iter_mut().zip(v.wheels()) {
            *s = w.spin;
        }
        let p = v.powertrain();
        Self {
            velocity_body: v.lin_vel_body(),
            yaw_rate: v.ang_vel_body().z,
            wheel_spin,
            gear: p.gear,
            engine_speed: p.engine_speed,
        }
    }

    /// Forward speed (m/s).
    pub fn speed(&self) -> f64 {
        self.velocity_body.x
    }
}

/// Input to the controller, at the level it enters.
#[derive(Clone, Copy, Debug, PartialEq)]
// Per-wheel input makes `Direct` large; setpoints live on the stack, and boxing would allocate
// every control step.
#[allow(clippy::large_enum_variant)]
pub enum GroundSetpoint {
    /// Driver input, passed through.
    Direct(DriveInput),
    /// One pedal axis (`[−1, 1]`: accelerate forward, or brake and then reverse) and steering
    /// (`[−1, 1]`), with automatic reverse engagement near standstill; `handbrake` applies the
    /// parking brake.
    Pedal { drive: f64, steering: f64, handbrake: bool },
    /// Left and right motor commands (`[−1, 1]`), for side drives.
    Sides { left: f64, right: f64 },
    /// Forward speed (m/s, negative reverses) and path curvature (1/m, positive turns left).
    SpeedCurvature { speed: f64, curvature: f64 },
    /// Forward speed (m/s) and yaw rate (rad/s), for side drives.
    SpeedYawRate { speed: f64, yaw_rate: f64 },
}

/// Loop tuning.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroundConfig {
    /// Speed loop time constant (s): the proportional gain is its inverse.
    pub speed_time_constant: f64,
    /// Speed integral gain as a fraction of `1/speed_time_constant²`, and the speed error
    /// (m/s) within which the integrator runs (so that it trims resistances and grades rather
    /// than winding up during large steps).
    pub speed_integral: f64,
    pub speed_integral_band: f64,
    /// Acceleration and deceleration the speed loop may ask for (m/s²).
    pub max_accel: f64,
    pub max_decel: f64,
    /// Below this speed (m/s) a zero speed request holds the vehicle on its brakes.
    pub hold_speed: f64,
    /// Traction control: a driven wheel whose slip speed exceeds this fraction of the travel
    /// speed (with `traction_slip_floor`, m/s, at least) is braked, so that its differential
    /// passes the torque on to the others; when even the least slipping wheel exceeds it, the
    /// drive force also fades out, reaching zero at twice that slip.
    pub traction_slip: f64,
    pub traction_slip_floor: f64,
    /// Integral gain on the curvature error (1/s), and the largest steering correction it may
    /// add (rad).
    pub curvature_integral: f64,
    pub curvature_correction: f64,
    /// Integral gains on the yaw-rate and speed errors of side drives (1/s), which take out
    /// the slip of skid steering.
    pub yaw_integral: f64,
    pub side_speed_integral: f64,
    /// Wheel-speed loop time constant of side drives (s).
    pub wheel_time_constant: f64,
}

impl Default for GroundConfig {
    fn default() -> Self {
        Self {
            speed_time_constant: 0.5,
            speed_integral: 0.25,
            speed_integral_band: 1.0,
            max_accel: 3.0,
            max_decel: 6.0,
            hold_speed: 0.3,
            traction_slip: 0.1,
            traction_slip_floor: 0.3,
            curvature_integral: 1.0,
            curvature_correction: 0.15,
            yaw_integral: 2.0,
            side_speed_integral: 2.0,
            wheel_time_constant: 0.05,
        }
    }
}

#[derive(Clone, Debug)]
struct Motor {
    wheels: Vec<usize>,
    side: MotorSide,
    max_torque: f64,
    ratio: f64,
    no_load_speed: Option<f64>,
    /// Inertia seen by the motor's wheels in a speed change, including their share of the
    /// vehicle mass (kg·m²).
    inertia: f64,
}

#[derive(Clone, Debug)]
enum Drive {
    Combustion(autonomousim_vehicles::ground::CombustionDef),
    Electric(Vec<Motor>),
}

/// Speed, curvature and yaw-rate control of one wheeled vehicle. Call
/// [`update`](Self::update) once per control step.
#[derive(Clone, Debug)]
pub struct GroundController {
    dt: f64,
    config: GroundConfig,
    drive: Drive,
    /// Mass including the spinning parts' equivalent mass (kg), and the mean driven-wheel
    /// radius (m).
    mass: f64,
    radius: f64,
    /// Driven wheels.
    driven: Vec<usize>,
    /// Total service brake torque (N·m).
    brake_torque: f64,
    /// Per wheel: tyre radius (m) and service brake torque (N·m).
    wheel_radius: Vec<f64>,
    wheel_brake: Vec<f64>,
    /// Bicycle model of the Ackermann-steered axle: wheelbase (m), steering share, and the
    /// steering lock (rad).
    steering: Option<(f64, f64, f64)>,
    /// Track of a side drive (m).
    track: Option<f64>,
    // Loop state.
    speed_i: f64,
    curvature_i: f64,
    yaw_i: f64,
    side_speed_i: f64,
    /// Whether a side motor saturated in the last step (holds the outer integrators).
    saturated: bool,
    motor_i: Vec<f64>,
    reverse: bool,
    /// Driver input of the last update.
    last: DriveInput,
}

impl GroundController {
    pub fn new(def: &WheeledDef, dt: f64, config: &GroundConfig) -> Result<Self, ControlError> {
        let c = config;
        let pos = |x: f64| x > 0.0 && x.is_finite();
        if !(pos(dt)
            && pos(c.speed_time_constant)
            && c.speed_integral >= 0.0
            && pos(c.speed_integral_band)
            && pos(c.max_accel)
            && pos(c.max_decel)
            && c.hold_speed >= 0.0
            && c.traction_slip >= 0.0
            && pos(c.traction_slip_floor)
            && c.curvature_integral >= 0.0
            && c.curvature_correction >= 0.0
            && c.yaw_integral >= 0.0
            && c.side_speed_integral >= 0.0
            && pos(c.wheel_time_constant))
        {
            return Err(ControlError::InvalidConfig("ground controller gains must be positive".into()));
        }
        let n = def.num_wheels();
        let inertia = def.spin_inertia();
        let radius_of = |w: usize| def.tire(w / 2).radius();
        let spinning: f64 = (0..n).map(|w| inertia[w] / radius_of(w).powi(2)).sum();
        let mass = def.total_mass() + spinning;
        let (drive, driven) = match &def.powertrain {
            PowertrainDef::Combustion(cd) => {
                let wheels: Vec<usize> = cd.driven.iter().flat_map(|&a| [2 * a, 2 * a + 1]).collect();
                (Drive::Combustion(cd.clone()), wheels)
            }
            PowertrainDef::Electric(e) => {
                let motors: Vec<Motor> = e
                    .motors
                    .iter()
                    .map(|m| {
                        let share = m.wheels.len() as f64 / n as f64;
                        let r = m.wheels.iter().map(|&w| radius_of(w)).sum::<f64>() / m.wheels.len() as f64;
                        Motor {
                            wheels: m.wheels.clone(),
                            side: m.side,
                            max_torque: m.max_torque,
                            ratio: m.ratio,
                            no_load_speed: m.no_load_speed,
                            inertia: m.wheels.iter().map(|&w| inertia[w]).sum::<f64>()
                                + share * def.total_mass() * r * r,
                        }
                    })
                    .collect();
                let wheels = motors.iter().flat_map(|m| m.wheels.clone()).collect();
                (Drive::Electric(motors), wheels)
            }
        };
        let radius = driven.iter().map(|&w| radius_of(w)).sum::<f64>() / driven.len() as f64;
        let wheel_radius: Vec<f64> = (0..n).map(radius_of).collect();
        let wheel_brake: Vec<f64> = (0..n).map(|w| def.axles[w / 2].brake.max_torque).collect();
        let brake_torque = wheel_brake.iter().sum();
        let steering = def.steering.and_then(|s| {
            let reference: Vec<f64> = def.axles.iter().filter(|a| a.steer == 0.0).map(|a| a.position.x).collect();
            let reference = if reference.is_empty() {
                def.axles.iter().map(|a| a.position.x).sum::<f64>() / def.axles.len() as f64
            } else {
                reference.iter().sum::<f64>() / reference.len() as f64
            };
            let axle =
                def.axles.iter().filter(|a| a.steer != 0.0).max_by(|a, b| a.steer.abs().total_cmp(&b.steer.abs()))?;
            let wheelbase = axle.position.x - reference;
            (wheelbase.abs() > 1e-6).then_some((wheelbase, axle.steer, s.max_angle))
        });
        let track = match &drive {
            Drive::Electric(m) if is_side_drive(m) => {
                let ys: Vec<f64> = driven.iter().map(|&w| def.wheel_position(w).y.abs()).collect();
                Some(2.0 * ys.iter().sum::<f64>() / ys.len() as f64)
            }
            _ => None,
        };
        let motors = match &drive {
            Drive::Electric(m) => m.len(),
            Drive::Combustion(_) => 0,
        };
        Ok(Self {
            dt,
            config: config.clone(),
            drive,
            mass,
            radius,
            driven,
            brake_torque,
            wheel_radius,
            wheel_brake,
            steering,
            track,
            speed_i: 0.0,
            curvature_i: 0.0,
            yaw_i: 0.0,
            side_speed_i: 0.0,
            saturated: false,
            motor_i: vec![0.0; motors],
            reverse: false,
            last: DriveInput::default(),
        })
    }

    pub fn config(&self) -> &GroundConfig {
        &self.config
    }

    /// Whether the vehicle has Ackermann steering (curvature control by steering).
    pub fn has_steering(&self) -> bool {
        self.steering.is_some()
    }

    /// Wheelbase of the bicycle model of the steered axle (m); `None` without steering.
    pub fn wheelbase(&self) -> Option<f64> {
        self.steering.map(|(l, _, _)| l)
    }

    /// Driver input of the last [`update`](Self::update).
    pub fn last_input(&self) -> &DriveInput {
        &self.last
    }

    /// Whether the vehicle is driven by left and right electric motors (skid-steer or
    /// diff-drive).
    pub fn is_side_drive(&self) -> bool {
        self.track.is_some()
    }

    /// Clear the loop state.
    pub fn reset(&mut self) {
        self.speed_i = 0.0;
        self.curvature_i = 0.0;
        self.yaw_i = 0.0;
        self.side_speed_i = 0.0;
        self.saturated = false;
        self.motor_i.fill(0.0);
        self.reverse = false;
        self.last = DriveInput::default();
    }

    /// Driver input for setpoint `sp`.
    pub fn update(&mut self, sp: &GroundSetpoint, est: &GroundEstimate) -> DriveInput {
        self.last = self.input(sp, est);
        self.last
    }

    fn input(&mut self, sp: &GroundSetpoint, est: &GroundEstimate) -> DriveInput {
        let v = est.speed();
        match *sp {
            GroundSetpoint::Direct(input) => input,
            GroundSetpoint::Sides { left, right } => {
                DriveInput { throttle: 0.5 * (left + right), yaw: 0.5 * (right - left), ..Default::default() }
            }
            GroundSetpoint::Pedal { drive, steering, handbrake } => {
                DriveInput { parking: handbrake, ..self.pedal(drive, steering, v) }
            }
            GroundSetpoint::SpeedCurvature { speed, curvature } => match self.steering {
                Some(_) => {
                    let mut input = self.longitudinal(speed, est);
                    input.steering = self.steer(curvature, est);
                    input
                }
                None => self.side_drive(speed, speed * curvature, est),
            },
            GroundSetpoint::SpeedYawRate { speed, yaw_rate } => self.side_drive(speed, yaw_rate, est),
        }
    }

    /// One pedal axis: forward throttle, or brake while rolling forward and reverse once
    /// (nearly) stopped; the other way round in reverse.
    fn pedal(&mut self, drive: f64, steering: f64, v: f64) -> DriveInput {
        let drive = finite(drive).clamp(-1.0, 1.0);
        let mut input = DriveInput { steering: finite(steering), ..Default::default() };
        match self.drive {
            Drive::Electric(_) => input.throttle = drive,
            Drive::Combustion(_) => {
                let rolling = self.config.hold_speed.max(0.5);
                let wants_reverse = drive < 0.0;
                if wants_reverse != self.reverse && v.abs() < rolling {
                    self.reverse = wants_reverse;
                }
                if wants_reverse == self.reverse {
                    input.throttle = drive.abs();
                } else {
                    input.brake = drive.abs();
                }
                input.reverse = self.reverse;
            }
        }
        input
    }

    /// Speed loop and powertrain inversion (vehicles steered by their wheels).
    fn longitudinal(&mut self, v_ref: f64, est: &GroundEstimate) -> DriveInput {
        let v = est.speed();
        let v_ref = finite(v_ref);
        let mut input = DriveInput::default();
        // Gear direction: change only near standstill.
        if (v_ref < 0.0) != self.reverse && v.abs() < self.config.hold_speed.max(0.5) {
            self.reverse = v_ref < 0.0;
        }
        input.reverse = self.reverse;
        if v_ref == 0.0 && v.abs() < self.config.hold_speed {
            self.speed_i = 0.0;
            input.brake = 1.0;
            return input;
        }
        let (lo, hi) = self.accel_bounds(est);
        let integral = self.speed_i;
        let mut accel = self.speed_loop(v_ref, v, lo, hi);
        // Traction control on the least slipping driven wheel (slip speed along the demand),
        // with the integrator held while it acts.
        let sign = accel.signum();
        let slip = |w: usize| sign * (est.wheel_spin[w] * self.wheel_radius[w] - v);
        let allowed = (self.config.traction_slip * v.abs()).max(self.config.traction_slip_floor);
        let least = self.driven.iter().map(|&w| slip(w)).fold(f64::INFINITY, f64::min);
        if accel != 0.0 && least > allowed {
            accel *= (2.0 - least / allowed).clamp(0.0, 1.0);
            self.speed_i = integral;
        }
        let force = self.mass * accel;
        // Spinning wheels under drive (not under braking) are braked in proportion to their
        // excess slip, up to their share of the drive torque, as an open differential would
        // otherwise waste that torque on them.
        if (self.reverse && accel < 0.0) || (!self.reverse && accel > 0.0) {
            let share = force.abs() * self.radius / self.driven.len() as f64;
            for &w in &self.driven {
                let excess = (slip(w) - allowed) / allowed;
                if excess > 0.0 {
                    input.wheel_brake[w] = (excess.min(1.0) * share / self.wheel_brake[w]).min(1.0);
                }
            }
        }
        let r = self.radius;
        match &self.drive {
            Drive::Combustion(c) => {
                let ratio = if self.reverse {
                    c.gearbox.reverse
                } else {
                    c.gearbox.forward[(est.gear.max(1) as usize - 1).min(c.gearbox.forward.len() - 1)]
                };
                let reduction = ratio * c.final_drive * r;
                let rpm = est.engine_speed.abs() / RPM;
                let (full, zero) = (c.engine.full_throttle.eval(rpm), c.engine.zero_throttle.eval(rpm));
                let torque = force * reduction;
                input.throttle = ((torque - zero) / (full - zero)).clamp(0.0, 1.0);
                // What the engine cannot take away at zero throttle, the brakes do.
                let residual = force - zero / reduction;
                if (self.reverse && residual > 0.0) || (!self.reverse && residual < 0.0) {
                    input.brake = (residual.abs() * r / self.brake_torque).clamp(0.0, 1.0);
                }
                if v_ref == 0.0 {
                    input.throttle = 0.0;
                }
            }
            Drive::Electric(motors) => {
                // One throttle for all motors (their wheel-mean command), so that steering
                // and brakes keep their usual channels.
                let wheels: usize = motors.iter().map(|m| m.wheels.len()).sum();
                let per_wheel = force * r / wheels as f64;
                input.throttle = motors
                    .iter()
                    .map(|m| m.wheels.len() as f64 * motor_command(m, per_wheel * m.wheels.len() as f64, est))
                    .sum::<f64>()
                    / wheels as f64;
            }
        }
        input
    }

    /// Acceleration (m/s²) the powertrain and brakes can give now: with the engine in the
    /// current gear at full throttle or the motors at full command, and full braking against
    /// the direction of travel (all on flat ground, without resistances).
    fn accel_bounds(&self, est: &GroundEstimate) -> (f64, f64) {
        let brake = self.brake_torque / self.radius / self.mass;
        let (drive_lo, drive_hi) = match &self.drive {
            Drive::Combustion(c) => {
                let ratio = if self.reverse {
                    c.gearbox.reverse
                } else {
                    c.gearbox.forward[(est.gear.max(1) as usize - 1).min(c.gearbox.forward.len() - 1)]
                };
                let rpm = est.engine_speed.abs() / RPM;
                let full = c.engine.full_throttle.eval(rpm) / (ratio * c.final_drive * self.radius) / self.mass;
                if self.reverse { (full, f64::INFINITY) } else { (f64::NEG_INFINITY, full) }
            }
            Drive::Electric(motors) => {
                let (mut lo, mut hi) = (0.0, 0.0);
                for m in motors {
                    let spin = m.wheels.iter().map(|&w| est.wheel_spin[w]).sum::<f64>() / m.wheels.len() as f64;
                    let omega = spin / m.ratio;
                    let (t_lo, t_hi) = match m.no_load_speed {
                        Some(w0) => (
                            (m.max_torque * (-1.0 - omega / w0)).max(-m.max_torque),
                            (m.max_torque * (1.0 - omega / w0)).min(m.max_torque),
                        ),
                        None => (-m.max_torque, m.max_torque),
                    };
                    lo += t_lo / m.ratio;
                    hi += t_hi / m.ratio;
                }
                (lo / self.radius / self.mass, hi / self.radius / self.mass)
            }
        };
        // Brakes act against the motion, and at standstill either way.
        let v = est.speed();
        let lo = if v >= 0.0 { drive_lo.min(-brake) } else { drive_lo };
        let hi = if v <= 0.0 { drive_hi.max(brake) } else { drive_hi };
        (lo, hi)
    }

    /// Acceleration demand (m/s²) of the speed PI, within `[lo, hi]` and the configured
    /// limits, with conditional integration.
    fn speed_loop(&mut self, v_ref: f64, v: f64, lo: f64, hi: f64) -> f64 {
        let c = &self.config;
        let kp = 1.0 / c.speed_time_constant;
        let ki = c.speed_integral * kp / c.speed_time_constant;
        let e = v_ref - v;
        let raw = kp * e + self.speed_i;
        let accel = raw.clamp(lo.max(-c.max_decel), hi.min(c.max_accel).max(lo.max(-c.max_decel)));
        if e.abs() < c.speed_integral_band && (accel == raw || (raw > accel) != (e > 0.0)) {
            self.speed_i = (self.speed_i + ki * e * self.dt).clamp(-c.max_decel, c.max_accel);
        }
        accel
    }

    /// Steering command for path curvature `curvature` (Ackermann steering).
    fn steer(&mut self, curvature: f64, est: &GroundEstimate) -> f64 {
        let Some((wheelbase, share, lock)) = self.steering else { return 0.0 };
        let curvature = finite(curvature);
        let v = est.speed();
        if v.abs() > 2.0 {
            let measured = est.yaw_rate / v;
            let c = &self.config;
            self.curvature_i = (self.curvature_i + c.curvature_integral * wheelbase * (curvature - measured) * self.dt)
                .clamp(-c.curvature_correction, c.curvature_correction);
        }
        let delta = (curvature * wheelbase).atan() / share + self.curvature_i;
        (delta / lock).clamp(-1.0, 1.0)
    }

    /// Side drives: wheel speeds from speed and yaw rate, per-motor wheel-speed PI.
    fn side_drive(&mut self, v_ref: f64, yaw_ref: f64, est: &GroundEstimate) -> DriveInput {
        let (Some(track), Drive::Electric(motors)) = (self.track, &self.drive) else {
            // Not a side drive: speed only.
            return self.longitudinal(v_ref, est);
        };
        let (v_ref, yaw_ref) = (finite(v_ref), finite(yaw_ref));
        let v = est.speed();
        let c = &self.config;
        if v_ref == 0.0 && yaw_ref == 0.0 && v.abs() < c.hold_speed && est.yaw_rate.abs() < 0.1 {
            self.yaw_i = 0.0;
            self.side_speed_i = 0.0;
            self.motor_i.fill(0.0);
            return DriveInput { brake: 1.0, ..Default::default() };
        }
        if !self.saturated {
            self.yaw_i = (self.yaw_i + c.yaw_integral * (yaw_ref - est.yaw_rate) * self.dt).clamp(-2.0, 2.0);
            self.side_speed_i = (self.side_speed_i + c.side_speed_integral * (v_ref - v) * self.dt).clamp(-1.0, 1.0);
        }
        self.saturated = false;
        let yaw = yaw_ref + self.yaw_i;
        let v_ref = v_ref + self.side_speed_i;
        let mut commands = [0.0; MAX_WHEELS];
        for (k, m) in motors.iter().enumerate() {
            let sign = match m.side {
                MotorSide::Left => -1.0,
                MotorSide::Right => 1.0,
                MotorSide::Both => 0.0,
            };
            let target = (v_ref + sign * yaw * 0.5 * track) / self.radius;
            let spin = m.wheels.iter().map(|&w| est.wheel_spin[w]).sum::<f64>() / m.wheels.len() as f64;
            let e = target - spin;
            let tau = c.wheel_time_constant;
            let kp = m.inertia / tau;
            let torque = kp * e + self.motor_i[k];
            let command = motor_command(m, torque, est);
            self.saturated |= command.abs() >= 1.0;
            // Conditional integration (ki = kp / (4τ)).
            if command.abs() < 1.0 || (command > 0.0) != (e > 0.0) {
                self.motor_i[k] += kp / (4.0 * tau) * e * self.dt;
            }
            for &w in &m.wheels {
                commands[w] = command;
            }
        }
        DriveInput { wheels: Some(WheelCommands { drive: commands, ..Default::default() }), ..Default::default() }
    }
}

/// Command of motor `m` for a torque `wheel_torque` (N·m) summed over its wheels.
fn motor_command(m: &Motor, wheel_torque: f64, est: &GroundEstimate) -> f64 {
    let torque = wheel_torque * m.ratio;
    let command = match m.no_load_speed {
        Some(w0) => {
            let spin = m.wheels.iter().map(|&w| est.wheel_spin[w]).sum::<f64>() / m.wheels.len() as f64;
            torque / m.max_torque + spin / m.ratio / w0
        }
        None => torque / m.max_torque,
    };
    command.clamp(-1.0, 1.0)
}

fn is_side_drive(motors: &[Motor]) -> bool {
    motors.iter().any(|m| m.side == MotorSide::Left)
        && motors.iter().any(|m| m.side == MotorSide::Right)
        && motors.iter().all(|m| m.side != MotorSide::Both)
}

fn finite(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}
