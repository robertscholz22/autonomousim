//! Closed-loop checks of the multirotor cascade on the full vehicle model over a flat test
//! world: step responses of every loop, saturation, and the same tuning for cf2x and iris.

use autonomousim_control::multirotor::*;
use autonomousim_core::contact::StaticScene;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::{tilt, uniform_rotation, wrap_angle, yaw};
use autonomousim_vehicles::multirotor::{
    AirData, GroundPlane, InitialState, MAX_ROTORS, MotorInit, Multirotor, MultirotorScales, StepEnv,
};
use autonomousim_vehicles::presets;
use autonomousim_world::StaticWorld;
use autonomousim_world::testworlds;
use glam::{DVec2, DVec3};
use std::sync::Arc;

const G: f64 = 9.80665;
const DT: f64 = 0.002;
const PRESETS: [&str; 2] = ["cf2x", "iris_like"];

struct Rig {
    quad: Multirotor,
    ctrl: MultirotorController,
    world: StaticWorld,
    air: AirData,
    t: f64,
    cmd: [f64; MAX_ROTORS],
}

impl Rig {
    /// Hovering at rest 10 m above flat ground, default controller.
    fn new(name: &str) -> Self {
        Self::with_config(name, &ControllerConfig::default())
    }

    fn with_config(name: &str, config: &ControllerConfig) -> Self {
        let def = Arc::new(presets::multirotor(name).unwrap());
        let mut quad = Multirotor::new(def.clone(), DT);
        let hover = def.hover_omega(G, 1.225);
        quad.reset(&InitialState::at(Pose::from_translation(DVec3::new(0.0, 0.0, 10.0)), MotorInit::Speed(hover)));
        let ctrl = MultirotorController::new(&def, DT, config).unwrap();
        Self { quad, ctrl, world: testworlds::flat(200.0), air: AirData::default(), t: 0.0, cmd: [0.0; MAX_ROTORS] }
    }

    fn est(&self) -> StateEstimate {
        StateEstimate::of(&self.quad)
    }

    fn step(&mut self, sp: &Setpoint) {
        let n = self.quad.num_rotors();
        let est = self.est();
        self.ctrl.update(sp, &est, &mut self.cmd[..n]);
        let scene = StaticScene {
            terrain: self.world.terrain(),
            obstacles: self.world.obstacles(),
            materials: self.world.materials(),
        };
        let ground = GroundPlane::below(self.world.terrain(), self.quad.position(), 2.0);
        let env = StepEnv { scene: Some(scene), gravity: DVec3::new(0.0, 0.0, -G), air: self.air, ground };
        self.quad.step(&self.cmd[..n], &env).unwrap();
        self.t += DT;
    }

    /// Run for `duration` seconds, recording `f(rig)` every step.
    fn run(&mut self, sp: &Setpoint, duration: f64, mut f: impl FnMut(&Self) -> f64) -> Vec<(f64, f64)> {
        let t0 = self.t;
        let mut out = Vec::new();
        while self.t - t0 < duration - 1e-9 {
            self.step(sp);
            out.push((self.t - t0, f(self)));
        }
        out
    }

    fn hover_ctbr(&self) -> Setpoint {
        Setpoint::Ctbr { thrust: self.ctrl.hover_thrust(), rates: DVec3::ZERO }
    }
}

/// 10–90 % rise time and overshoot (fraction) of a step response towards `target`.
fn step_metrics(trace: &[(f64, f64)], target: f64) -> (f64, f64) {
    let t10 = trace.iter().find(|(_, y)| *y >= 0.1 * target).map_or(f64::INFINITY, |p| p.0);
    let t90 = trace.iter().find(|(_, y)| *y >= 0.9 * target).map_or(f64::INFINITY, |p| p.0);
    let peak = trace.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    (t90 - t10, (peak - target).max(0.0) / target)
}

/// Time after which the response stays within `band` of `target`.
fn settling_time(trace: &[(f64, f64)], target: f64, band: f64) -> f64 {
    trace.iter().rev().find(|(_, y)| (y - target).abs() > band).map_or(0.0, |p| p.0)
}

/// 1 rad/s body-rate steps from hover: roll and pitch rise (10–90 %) within 50 ms, yaw within
/// 150 ms (small yaw authority), overshoot below 20 %.
#[test]
fn rate_steps() {
    for name in PRESETS {
        for axis in 0..3 {
            let mut rig = Rig::new(name);
            rig.run(&rig.hover_ctbr(), 0.5, |_| 0.0);
            let mut rates = DVec3::ZERO;
            rates[axis] = 1.0;
            let sp = Setpoint::Ctbr { thrust: rig.ctrl.hover_thrust(), rates };
            let tr = rig.run(&sp, 0.4, |r| r.quad.ang_vel_body()[axis]);
            let (rise, overshoot) = step_metrics(&tr, 1.0);
            let max_rise = if axis == 2 { 0.15 } else { 0.05 };
            println!("{name} axis {axis}: rise {:.0} ms, overshoot {:.1} %", rise * 1e3, overshoot * 100.0);
            assert!(rise < max_rise && overshoot < 0.2, "{name} axis {axis}: rise {rise}, overshoot {overshoot}");
        }
    }
}

/// 10° roll and pitch steps settle within 5 % in 0.3 s; a 45° heading step within 1 s.
#[test]
fn attitude_steps() {
    let step = 10f64.to_radians();
    for name in PRESETS {
        for tilt_sp in [DVec2::new(step, 0.0), DVec2::new(0.0, -step)] {
            let mut rig = Rig::new(name);
            let hover = rig.ctrl.hover_thrust();
            let level = Setpoint::Attitude { tilt: DVec2::ZERO, yaw: YawCommand::Angle(0.0), thrust: hover };
            rig.run(&level, 0.5, |_| 0.0);
            let sp = Setpoint::Attitude { tilt: tilt_sp, yaw: YawCommand::Angle(0.0), thrust: hover };
            let tr = rig.run(&sp, 1.0, |r| tilt(r.quad.orientation()));
            let settle = settling_time(&tr, step, 0.05 * step);
            println!("{name} tilt {tilt_sp}: settles in {settle:.3} s");
            assert!(settle < 0.3, "{name}: {settle}");
            // The tilt goes where it was asked to.
            let axis = rig.quad.orientation() * DVec3::Z;
            let expect = glam::DQuat::from_scaled_axis(tilt_sp.extend(0.0)) * DVec3::Z;
            assert!(axis.angle_between(expect) < 0.01 * step);
        }
        let mut rig = Rig::new(name);
        let heading = 45f64.to_radians();
        let sp =
            Setpoint::Attitude { tilt: DVec2::ZERO, yaw: YawCommand::Angle(heading), thrust: rig.ctrl.hover_thrust() };
        let tr = rig.run(&sp, 2.0, |r| yaw(r.quad.orientation()));
        let settle = settling_time(&tr, heading, 0.05 * heading);
        println!("{name} heading 45°: settles in {settle:.3} s");
        assert!(settle < 1.0, "{name}: {settle}");
    }
}

/// 1 m position steps (horizontal and vertical) in a 3.6 m/s wind settle within 2 % in 2 s,
/// and the disturbance observer removes the steady-state error (without it, the P velocity
/// loop would leave ~0.2 m).
#[test]
fn position_steps_under_wind() {
    for name in PRESETS {
        for dir in [DVec3::X, DVec3::Z] {
            let mut rig = Rig::new(name);
            rig.air.wind = DVec3::new(3.0, 2.0, 0.0);
            let p0 = rig.quad.position();
            rig.run(&Setpoint::Position { position: p0, yaw: YawCommand::Angle(0.0) }, 8.0, |_| 0.0);
            let target = p0 + dir;
            let sp = Setpoint::Position { position: target, yaw: YawCommand::Angle(0.0) };
            let tr = rig.run(&sp, 10.0, |r| (r.quad.position() - p0).dot(dir));
            let settle = settling_time(&tr, 1.0, 0.02);
            let err = (rig.quad.position() - target).length();
            println!("{name} {dir}: settles in {settle:.3} s, final error {err:.1e} m");
            assert!(settle < 2.0 && err < 1e-3, "{name} {dir}: settle {settle}, error {err}");
        }
    }
}

/// The rate integrator does not wind up while the rotors saturate: after a 30 rad/s roll
/// command the rate returns to zero without a large reverse overshoot.
#[test]
fn saturated_rate_command_recovers() {
    for name in PRESETS {
        let mut rig = Rig::new(name);
        rig.quad.reset(&InitialState::at(
            Pose::from_translation(DVec3::new(0.0, 0.0, 50.0)),
            MotorInit::Speed(rig.quad.def().hover_omega(G, 1.225)),
        ));
        let thrust = rig.ctrl.hover_thrust();
        rig.run(&Setpoint::Ctbr { thrust, rates: DVec3::new(30.0, 0.0, 0.0) }, 0.3, |_| 0.0);
        let limit = rig.ctrl.gains().rate.i_limit;
        assert!(rig.ctrl.rate_integral().abs().cmple(limit).all());
        let tr = rig.run(&Setpoint::Ctbr { thrust, rates: DVec3::ZERO }, 0.5, |r| r.quad.ang_vel_body().x);
        let reverse = -tr.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let settle = settling_time(&tr, 0.0, 0.1);
        println!(
            "{name}: peak rate before release {:.1}, reverse overshoot {reverse:.2} rad/s, settles in {settle:.3} s",
            tr[0].1
        );
        assert!(reverse < 0.05 * tr[0].1 && settle < 0.4, "{name}: {reverse} {settle}");
    }
}

/// Full collective plus a roll command: the allocation gives up thrust, not roll.
#[test]
fn full_throttle_keeps_roll_authority() {
    for name in PRESETS {
        let mut rig = Rig::new(name);
        let sp = Setpoint::Ctbr { thrust: rig.ctrl.max_thrust(), rates: DVec3::new(2.0, 0.0, 0.0) };
        let tr = rig.run(&sp, 0.3, |r| r.quad.ang_vel_body().x);
        let (rise, _) = step_metrics(&tr, 2.0);
        let (torque, thrust) = rig.ctrl.allocator().wrench(rig.ctrl.rotor_thrusts());
        println!(
            "{name}: rise {:.0} ms, collective {:.0} % of max",
            rise * 1e3,
            100.0 * thrust / rig.ctrl.max_thrust()
        );
        assert!(rise < 0.06 && thrust > 0.8 * rig.ctrl.max_thrust() && rig.quad.lin_vel_world().z > 1.0);
        assert!(torque.is_finite());
    }
}

/// A 12 m/s velocity command: tilt stays within the limit (plus attitude-loop overshoot),
/// altitude is held (vertical priority), and the speed converges to what the tilt allows.
#[test]
fn velocity_command_respects_limits() {
    for name in PRESETS {
        let mut rig = Rig::new(name);
        let z0 = rig.quad.position().z;
        let limit = rig.ctrl.config().limits.tilt;
        let sp = Setpoint::Velocity {
            velocity: DVec3::new(12.0, 0.0, 0.0),
            frame: Frame::World,
            yaw: YawCommand::Angle(0.0),
        };
        let mut worst = (0.0f64, 0.0f64);
        rig.run(&sp, 8.0, |r| {
            worst.0 = worst.0.max(tilt(r.quad.orientation()));
            worst.1 = worst.1.max((r.quad.position().z - z0).abs());
            0.0
        });
        let v = rig.quad.lin_vel_world();
        println!(
            "{name}: max tilt {:.1}°, max altitude error {:.3} m, final speed {:.2} m/s",
            worst.0.to_degrees(),
            worst.1,
            v.x
        );
        assert!(worst.0 < limit + 2f64.to_radians() && worst.1 < 0.5 && v.x > 5.0, "{name}: {worst:?} {v}");
    }
}

/// Position hold recovers from random attitudes (including inverted) and tumbling: the thrust
/// axis follows the controller's setpoint again within 1 s (after that the position loop may
/// tilt on purpose, to brake), and the vehicle returns to the setpoint.
#[test]
fn recovers_from_arbitrary_attitude() {
    for name in PRESETS {
        for k in 0..8 {
            let mut rig = Rig::new(name);
            let u = |i: u32| ((k * 7 + i) as f64 * 0.618_033_988_7).fract();
            let rot = if k == 0 {
                glam::DQuat::from_rotation_x(std::f64::consts::PI)
            } else {
                uniform_rotation(u(1), u(2), u(3))
            };
            let p0 = DVec3::new(0.0, 0.0, 30.0);
            rig.quad.reset(&InitialState {
                lin_vel_world: DVec3::new(2.0 * u(4) - 1.0, 1.0, -2.0),
                ang_vel_body: DVec3::new(4.0 * u(5) - 2.0, 3.0 * u(6) - 1.5, 2.0),
                ..InitialState::at(Pose::new(p0, rot), MotorInit::Speed(rig.quad.def().hover_omega(G, 1.225)))
            });
            let sp = Setpoint::Position { position: p0, yaw: YawCommand::Angle(0.0) };
            let mut tracking = f64::INFINITY;
            let mut lowest = f64::INFINITY;
            let t0 = rig.t;
            rig.run(&sp, 10.0, |r| {
                let z = r.quad.orientation() * DVec3::Z;
                let z_sp = r.ctrl.attitude_setpoint() * DVec3::Z;
                if tracking.is_infinite() && z.angle_between(z_sp) < 10f64.to_radians() {
                    tracking = r.t - t0;
                }
                lowest = lowest.min(r.quad.position().z);
                0.0
            });
            let err = (rig.quad.position() - p0).length();
            println!(
                "{name} #{k}: tilt {:.0}°, tracking after {tracking:.2} s, lowest {lowest:.1} m, final error {err:.1e} m",
                tilt(rot).to_degrees()
            );
            assert!(tracking < 1.0 && lowest > 20.0 && err < 0.01, "{name} #{k}");
        }
    }
}

/// Heading-rate commands are tracked and heading-frame velocities turn with the vehicle.
#[test]
fn yaw_rate_and_heading_frame() {
    for name in PRESETS {
        let mut rig = Rig::new(name);
        let sp = Setpoint::Velocity {
            velocity: DVec3::new(2.0, 0.0, 0.0),
            frame: Frame::Heading,
            yaw: YawCommand::Rate(0.5),
        };
        rig.run(&sp, 3.0, |_| 0.0);
        let y0 = yaw(rig.quad.orientation());
        rig.run(&sp, 2.0, |_| 0.0);
        let rate = wrap_angle(yaw(rig.quad.orientation()) - y0) / 2.0;
        let v = rig.quad.lin_vel_world();
        let heading = yaw(rig.quad.orientation());
        let along = v.truncate().dot(DVec2::from_angle(heading));
        println!(
            "{name}: yaw rate {rate:.4} rad/s, speed along heading {along:.3} m/s, total {:.3}",
            v.truncate().length()
        );
        assert!((rate - 0.5).abs() < 0.01 && (along - 2.0).abs() < 0.1, "{name}: {rate} {along}");
    }
}

/// Randomised mass, inertia, thrust coefficients and motor lag are rejected as disturbances.
#[test]
fn randomised_vehicle_holds_position() {
    for name in PRESETS {
        let mut rig = Rig::new(name);
        rig.quad.set_scales(&MultirotorScales {
            mass: 1.25,
            inertia: DVec3::new(0.8, 1.2, 1.1),
            k_thrust: vec![0.95, 1.05, 1.0, 0.97],
            k_torque: 1.1,
            motor_tau: 1.3,
            body_drag: 1.5,
            rotor_drag: 0.7,
        });
        let p0 = rig.quad.position();
        let target = p0 + DVec3::new(1.0, -1.0, 0.5);
        rig.run(&Setpoint::Position { position: target, yaw: YawCommand::Angle(0.3) }, 10.0, |_| 0.0);
        let err = (rig.quad.position() - target).length();
        let yaw_err = wrap_angle(yaw(rig.quad.orientation()) - 0.3);
        println!("{name}: error {err:.1e} m, heading error {yaw_err:.1e} rad");
        assert!(err < 0.01 && yaw_err.abs() < 0.01, "{name}: {err} {yaw_err}");
    }
}

/// Normalised actions drive the vehicle through every mode.
#[test]
fn action_modes_end_to_end() {
    for name in PRESETS {
        let def = presets::multirotor(name).unwrap();
        let fresh = || Rig::new(name);
        let map = |rig: &Rig, mode| ActionMap::new(mode, ActionLimits::default(), &def, rig.ctrl.max_thrust());

        // motors: passed through.
        let mut rig = fresh();
        let sp = map(&rig, ActionMode::Motors).setpoint(&[0.0, 0.2, -0.2, 1.0], &rig.est());
        let mut cmd = [0.0; 4];
        rig.ctrl.update(&sp, &rig.est(), &mut cmd);
        let w = def.rotor.omega_max;
        assert_eq!(cmd, [0.5 * w, 0.6 * w, 0.4 * w, w]);

        // ctbr: the hover action holds altitude.
        let mut rig = fresh();
        let m = map(&rig, ActionMode::Ctbr);
        let hover = m.thrust_action(rig.ctrl.hover_thrust());
        let z0 = rig.quad.position().z;
        let sp = m.setpoint(&[0.0, 0.0, 0.0, hover], &rig.est());
        rig.run(&sp, 1.0, |_| 0.0);
        assert!((rig.quad.position().z - z0).abs() < 0.02 && tilt(rig.quad.orientation()) < 1e-3);

        // attitude: full tilt action → the action limit's tilt.
        let mut rig = fresh();
        let m = map(&rig, ActionMode::Attitude);
        let sp = m.setpoint(&[1.0, 0.0, 0.0, hover], &rig.est());
        rig.run(&sp, 0.5, |_| 0.0);
        assert!((tilt(rig.quad.orientation()) - m.limits().tilt).abs() < 0.01);

        // velocity: forward in the heading frame.
        let mut rig = fresh();
        let m = map(&rig, ActionMode::Velocity);
        let sp = m.setpoint(&[0.4, 0.0, 0.25, 0.0], &rig.est());
        rig.run(&sp, 4.0, |_| 0.0);
        assert!((rig.quad.lin_vel_world() - DVec3::new(2.0, 0.0, 0.5)).length() < 0.05);

        // position: offsets anchored where the action was taken.
        let mut rig = fresh();
        let m = map(&rig, ActionMode::Position);
        let p0 = rig.quad.position();
        let sp = m.setpoint(&[0.2, -0.2, 0.5, 0.25], &rig.est());
        rig.run(&sp, 6.0, |_| 0.0);
        assert!((rig.quad.position() - p0 - DVec3::new(1.0, -1.0, 1.0)).length() < 0.01);
        assert!((yaw(rig.quad.orientation()) - std::f64::consts::FRAC_PI_4).abs() < 0.01);
    }
}

/// Take off from the ground, land by commanding a position below it, take off again: the
/// disturbance observer ignores the ground support, so the second take-off is as fast as the
/// first.
#[test]
fn takes_off_and_lands() {
    for name in PRESETS {
        let mut rig = Rig::new(name);
        rig.quad.reset(&InitialState::at(Pose::from_translation(DVec3::new(0.0, 0.0, 0.2)), MotorInit::Idle));
        rig.ctrl.sync_motors(rig.quad.motor_speeds());
        rig.run(&Setpoint::Ctbr { thrust: 0.0, rates: DVec3::ZERO }, 1.0, |_| 0.0);
        assert!(rig.est().ground_contact, "{name}: not resting after the drop");
        let ground = rig.quad.position();
        let up = Setpoint::Position { position: ground + DVec3::Z * 2.0, yaw: YawCommand::Angle(0.0) };
        let first = rig.run(&up, 5.0, |r| r.quad.position().z - ground.z);

        let down = Setpoint::Position { position: ground - DVec3::Z * 0.5, yaw: YawCommand::Angle(0.0) };
        rig.run(&down, 5.0, |_| 0.0);
        let est = rig.est();
        assert!(est.ground_contact && (est.position - ground).length() < 0.02, "{name}: landed at {}", est.position);
        assert_eq!(rig.ctrl.disturbance_estimate(), DVec3::ZERO);

        let second = rig.run(&up, 5.0, |r| r.quad.position().z - ground.z);
        let (t1, t2) = (settling_time(&first, 2.0, 0.02), settling_time(&second, 2.0, 0.02));
        println!("{name}: take-off settles in {t1:.2} s, again after landing in {t2:.2} s");
        assert!(t1 < 3.0 && (t2 - t1).abs() < 0.1, "{name}: {t1} {t2}");
        assert!((rig.quad.position() - ground - DVec3::Z * 2.0).length() < 1e-3);
    }
}
