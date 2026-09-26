//! Closed-loop checks of the ground vehicle controller and action modes on the full vehicle
//! models over flat asphalt: speed steps (accelerating, braking, holding, reversing), curvature
//! tracking on a circle, (v, ω) tracking of the robots and the tracked carrier, and the `per_wheel` channels (torque
//! vectoring, one-wheel braking, crab and counter-phase steering) on an electric four-motor,
//! four-wheel-steered variant of the sedan.

use autonomousim_control::ground::*;
use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::presets;
use glam::DVec3;
use std::sync::Arc;

const DT: f64 = 1e-3;
const G: f64 = 9.81;

/// The sedan with a 540 N·m (at the wheel) torque-commanded motor per wheel and independent
/// steering (±0.6 rad) on both axles, without the sedan's static toe-in (≈1°): turned 30°
/// for crabbing, the toe forces of a pair of wheels would form a yaw couple and curve the path
/// (κ ≈ 0.004 1/m).
fn e_sedan() -> WheeledDef {
    let mut d = presets::wheeled("sedan_like").unwrap();
    d.name = "e_sedan".into();
    d.powertrain = PowertrainDef::Electric(ElectricDef {
        motors: (0..4)
            .map(|w| MotorDef {
                wheels: vec![w],
                side: MotorSide::Both,
                max_torque: 60.0,
                max_power: 40e3,
                ratio: 1.0 / 9.0,
                no_load_speed: None,
                time_constant: 0.02,
                rotor_inertia: 0.0,
                coupling: DifferentialDef::Open,
            })
            .collect(),
    });
    for a in &mut d.axles {
        a.steer_mode = SteerMode::Independent { max_angle: 0.6, rate: 2.0 };
        let k = &mut a.suspension.as_mut().unwrap().kinematics;
        k.toe.fill(0.0);
    }
    d.finish().unwrap();
    d
}

fn def(name: &str) -> WheeledDef {
    if name == "e_sedan" { e_sedan() } else { presets::wheeled(name).unwrap() }
}

struct Rig {
    v: Wheeled,
    ctrl: GroundController,
    terrain: FlatTerrain,
    materials: MaterialTable,
}

impl Rig {
    /// At rest on flat asphalt, heading east, default controller.
    fn new(name: &str) -> Self {
        let def = Arc::new(def(name));
        let mut v = Wheeled::new(def.clone(), DT);
        let init = v.rest(DVec3::ZERO, 0.0, 0.0);
        v.reset(&init);
        let ctrl = GroundController::new(&def, DT, &GroundConfig::default()).unwrap();
        Self { v, ctrl, terrain: FlatTerrain::new(0.0, MaterialId::ASPHALT), materials: MaterialTable::standard() }
    }

    fn est(&self) -> GroundEstimate {
        GroundEstimate::of(&self.v)
    }

    fn step(&mut self, sp: &GroundSetpoint) {
        let input = self.ctrl.update(sp, &self.est());
        let env = GroundStepEnv {
            scene: StaticScene { terrain: &self.terrain, obstacles: &NoObstacles, materials: &self.materials },
            gravity: DVec3::new(0.0, 0.0, -G),
            air: AirData::default(),
        };
        self.v.step(&input, &env).unwrap();
    }

    /// Run for `duration` seconds, recording `f(rig)` every step.
    fn run(&mut self, sp: &GroundSetpoint, duration: f64, mut f: impl FnMut(&Self) -> f64) -> Vec<f64> {
        (0..(duration / DT).round() as usize)
            .map(|_| {
                self.step(sp);
                f(self)
            })
            .collect()
    }
}

fn speed_curvature(speed: f64, curvature: f64) -> GroundSetpoint {
    GroundSetpoint::SpeedCurvature { speed, curvature }
}

/// Speed steps up, down, to a standstill, into reverse and back: each settles within 5 % of
/// the step (0.15 m/s at least) without overshooting it by more than 10 %.
#[test]
fn speed_steps_settle_without_overshoot() {
    for name in ["sedan_like", "offroad_4x4", "e_sedan"] {
        let mut rig = Rig::new(name);
        let mut from = 0.0;
        for (target, duration) in [(10.0, 12.0), (20.0, 15.0), (5.0, 10.0), (0.0, 6.0), (-3.0, 10.0), (0.0, 6.0)] {
            let speeds = rig.run(&speed_curvature(target, 0.0), duration, |r| r.v.speed());
            let step: f64 = target - from;
            let overshoot = speeds.iter().map(|&v| (v - target) * step.signum()).fold(f64::MIN, f64::max);
            let settled = speeds[speeds.len() - 1000..].iter().map(|&v| (v - target).abs()).fold(0.0, f64::max);
            println!("{name}: {from} → {target} m/s: overshoot {overshoot:.3} m/s, final error {settled:.3} m/s");
            assert!(overshoot <= 0.1 * step.abs() + 0.02, "{name}: {from} → {target}: overshoot {overshoot} m/s");
            assert!(settled <= (0.05 * step.abs()).max(0.15), "{name}: {from} → {target}: error {settled} m/s");
            from = target;
        }
        // Held on the brakes.
        assert!(rig.v.speed().abs() < 0.02, "{name}: {}", rig.v.speed());
    }
}

/// On a 30 m circle at 10 m/s (0.34 g) the curvature integral takes out the understeer: the
/// driven radius is within 5 % of the commanded one, forwards and in reverse at 3 m/s.
#[test]
fn curvature_tracking_on_a_circle() {
    for name in ["sedan_like", "offroad_4x4", "e_sedan"] {
        for (speed, curvature) in [(10.0, 1.0 / 30.0), (10.0, -1.0 / 30.0), (-3.0, 1.0 / 15.0)] {
            let mut rig = Rig::new(name);
            let sp = speed_curvature(speed, curvature);
            rig.run(&sp, 20.0, |_| 0.0);
            let measured = rig.run(&sp, 5.0, |r| r.v.ang_vel_body().z / r.v.speed());
            let mean = measured.iter().sum::<f64>() / measured.len() as f64;
            println!("{name} at {speed} m/s: curvature {mean:.4} for {curvature:.4}");
            assert!((mean / curvature - 1.0).abs() < 0.05, "{name} at {speed}: {mean} for {curvature}");
            assert!((rig.v.speed() - speed).abs() < 0.2, "{name}: {}", rig.v.speed());
        }
    }
}

/// The robots follow (v, ω) commands, including turning on the spot, within 5 % (0.02 at
/// least) of the command.
#[test]
fn robots_track_speed_and_yaw_rate() {
    for name in ["rover_diff", "rover_skid", "rover_tracked"] {
        let map = GroundActionMap::new(GroundActionMode::Vw, &GroundActionLimits::default(), &def(name)).unwrap();
        let (vmax, wmax) = (map.speed(), map.yaw_rate());
        for (speed, yaw_rate) in
            [(0.6 * vmax, 0.0), (0.5 * vmax, 0.3 * wmax), (0.0, 0.5 * wmax), (-0.4 * vmax, -0.3 * wmax)]
        {
            let mut rig = Rig::new(name);
            let sp = GroundSetpoint::SpeedYawRate { speed, yaw_rate };
            rig.run(&sp, 4.0, |_| 0.0);
            let v = rig.run(&sp, 2.0, |r| r.v.speed());
            let w = rig.run(&sp, 2.0, |r| r.v.ang_vel_body().z);
            let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
            let (v, w) = (mean(&v), mean(&w));
            println!("{name}: v {v:.3} for {speed:.3}, ω {w:.3} for {yaw_rate:.3}");
            assert!((v - speed).abs() < (0.05 * speed.abs()).max(0.02), "{name}: v {v} for {speed}");
            assert!((w - yaw_rate).abs() < (0.05 * yaw_rate.abs()).max(0.02), "{name}: ω {w} for {yaw_rate}");
        }
        // vk on a side drive: ω = v·κ.
        let mut rig = Rig::new(name);
        let sp = speed_curvature(0.4 * vmax, 1.0);
        rig.run(&sp, 4.0, |_| 0.0);
        let k = rig.run(&sp, 2.0, |r| r.v.ang_vel_body().z / r.v.speed());
        let k = k.iter().sum::<f64>() / k.len() as f64;
        assert!((k - 1.0).abs() < 0.05, "{name}: curvature {k}");
    }
}

/// The tracked carrier (engine, braked differential steering) follows (v, ω) commands within
/// its means, to 5 % (0.02 at least): straight, turning, and turning in reverse; then stops and
/// holds on its brakes. Braking the inner track takes drive away, and Chrono's M113 gearing
/// (no torque converter) leaves little reserve in the forward gears, so forward turns are
/// gentle (the reverse gear is lower). Asked to turn on the spot, which it cannot, it keeps
/// rolling in a left turn instead of stalling on its brakes, and drives off straight after.
#[test]
fn tracked_carrier_tracks_speed_and_yaw_rate() {
    let name = "tracked_apc";
    let map = GroundActionMap::new(GroundActionMode::Vw, &GroundActionLimits::default(), &def(name)).unwrap();
    assert!((map.yaw_rate() - 0.8 * map.speed() / (2.0 * 1.0795)).abs() < 1e-9);
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    for (speed, yaw_rate) in [(8.0, 0.0), (4.0, 0.05), (-2.0, -0.1)] {
        let mut rig = Rig::new(name);
        let sp = GroundSetpoint::SpeedYawRate { speed, yaw_rate };
        rig.run(&sp, 20.0, |_| 0.0);
        let v = mean(&rig.run(&sp, 3.0, |r| r.v.speed()));
        let w = mean(&rig.run(&sp, 3.0, |r| r.v.ang_vel_body().z));
        println!("{name}: v {v:.3} for {speed:.3}, ω {w:.3} for {yaw_rate:.3}");
        assert!((v - speed).abs() < (0.05 * speed.abs()).max(0.02), "{name}: v {v} for {speed}");
        assert!((w - yaw_rate).abs() < (0.05 * yaw_rate.abs()).max(0.02), "{name}: ω {w} for {yaw_rate}");
        rig.run(&GroundSetpoint::SpeedYawRate { speed: 0.0, yaw_rate: 0.0 }, 10.0, |_| 0.0);
        assert!(
            rig.v.speed().abs() < 0.02 && rig.v.ang_vel_body().z.abs() < 0.01,
            "{name}: not held, v {} ω {}",
            rig.v.speed(),
            rig.v.ang_vel_body().z
        );
    }
    let mut rig = Rig::new(name);
    rig.run(&GroundSetpoint::SpeedYawRate { speed: 0.0, yaw_rate: 0.3 }, 20.0, |_| 0.0);
    assert!(rig.v.speed() > 0.02 && rig.v.ang_vel_body().z > 0.0, "{name}: stalled");
    let sp = GroundSetpoint::SpeedYawRate { speed: 4.0, yaw_rate: 0.0 };
    rig.run(&sp, 20.0, |_| 0.0);
    assert!((rig.v.speed() - 4.0).abs() < 0.1 && rig.v.ang_vel_body().z.abs() < 0.01, "{name}: not recovered");
}

// ------------------------------------------------------------------------------ per_wheel

fn per_wheel(name: &str) -> GroundActionMap {
    GroundActionMap::new(GroundActionMode::PerWheel, &GroundActionLimits::default(), &def(name)).unwrap()
}

/// Action with the named channels set, the others 0.
fn action(map: &GroundActionMap, set: &[(&str, f64)]) -> Vec<f64> {
    let names = map.names();
    let mut a = vec![0.0; names.len()];
    for (name, x) in set {
        a[names.iter().position(|n| n == name).unwrap_or_else(|| panic!("no channel {name}"))] = *x;
    }
    a
}

/// Rig rolling straight at `speed`.
fn rolling(name: &str, speed: f64) -> Rig {
    let mut rig = Rig::new(name);
    let init = rig.v.rest(DVec3::ZERO, 0.0, speed);
    rig.v.reset(&init);
    rig
}

#[test]
fn per_wheel_channels() {
    let names = per_wheel("e_sedan").names();
    assert_eq!(
        names,
        ["drive_0", "drive_1", "drive_2", "drive_3", "brake_0", "brake_1", "brake_2", "brake_3"]
            .into_iter()
            .chain(["steer_0", "steer_1", "steer_2", "steer_3"])
            .collect::<Vec<_>>()
    );
}

/// Torque vectoring: driving the right wheels forward and the left ones backward at 10 m/s
/// gives the yaw moment of the longitudinal tyre forces that statics predicts
/// (`Σ −y·T/r`) and turns the car left.
#[test]
fn torque_difference_gives_the_predicted_yaw_moment() {
    let map = per_wheel("e_sedan");
    let d = e_sedan();
    let torque = 0.15 * 60.0 * 9.0;
    let mut rig = rolling("e_sedan", 10.0);
    let sp =
        map.setpoint(&action(&map, &[("drive_0", -0.15), ("drive_1", 0.15), ("drive_2", -0.15), ("drive_3", 0.15)]));
    rig.run(&sp, 0.3, |_| 0.0);
    let mut predicted = 0.0;
    let mut moment = 0.0;
    for w in 0..4 {
        let y = d.wheel_position(w).y;
        let r = d.tire(w / 2).radius() - rig.v.wheel(w).tire.deflection;
        predicted += -y * rig.v.wheel(w).drive_torque / r;
        moment += -y * rig.v.wheel(w).tire.fx;
    }
    let expected = 2.0 * (d.axles[0].position.y + d.axles[1].position.y) * torque / d.tire(0).radius();
    println!("yaw moment {moment:.0} N·m, predicted {predicted:.0} (nominal {expected:.0})");
    assert!((moment / predicted - 1.0).abs() < 0.1, "{moment} vs {predicted}");
    assert!((predicted / expected - 1.0).abs() < 0.05, "{predicted} vs {expected}");
    let yaw = rig.run(&sp, 1.0, |r| r.v.ang_vel_body().z);
    assert!(yaw.last().unwrap() > &0.005, "yaw rate {}", yaw.last().unwrap());
}

/// Braking one wheel yaws the car towards that side.
#[test]
fn braking_one_wheel_yaws_towards_it() {
    let map = per_wheel("sedan_like");
    for (channel, sign) in [("brake_0", 1.0), ("brake_1", -1.0), ("brake_2", 1.0), ("brake_3", -1.0)] {
        let mut rig = rolling("sedan_like", 15.0);
        let sp = map.setpoint(&action(&map, &[(channel, 0.3)]));
        let yaw = rig.run(&sp, 1.5, |r| r.v.ang_vel_body().z);
        let heading = yaw.iter().sum::<f64>() * DT;
        println!("{channel}: heading change {heading:.4} rad");
        assert!(heading * sign > 0.005, "{channel}: heading change {heading}");
        assert!(rig.v.speed() < 14.5, "{channel}: not braking ({})", rig.v.speed());
    }
}

/// All four wheels at 30°: the car crabs along 30° without turning. The drive is split in
/// proportion to the static wheel loads, so that its resultant passes through the centre of
/// mass (with equal torques it acts at the wheels' centroid, behind the sedan's centre of mass,
/// and turns the car).
#[test]
fn four_wheel_steering_crabs() {
    let map = per_wheel("e_sedan");
    let lock = 0.6;
    let angle = 30f64.to_radians();
    let steer = |x: f64| [("steer_0", x), ("steer_1", x), ("steer_2", x), ("steer_3", x)];
    let mut rig = Rig::new("e_sedan");
    // Steer at standstill, then drive.
    rig.run(&map.setpoint(&action(&map, &steer(angle / lock))), 0.5, |_| 0.0);
    let load: Vec<f64> = (0..4).map(|w| rig.v.wheel(w).tire.fz).collect();
    let mean = load.iter().sum::<f64>() / 4.0;
    let drive: Vec<f64> = load.iter().map(|fz| 0.25 * fz / mean).collect();
    let mut set: Vec<(&str, f64)> = steer(angle / lock).to_vec();
    set.extend([("drive_0", drive[0]), ("drive_1", drive[1]), ("drive_2", drive[2]), ("drive_3", drive[3])]);
    let sp = map.setpoint(&action(&map, &set));
    rig.run(&sp, 3.0, |_| 0.0);
    let yaw = rig.run(&sp, 2.0, |r| r.v.ang_vel_body().z.abs()).into_iter().fold(0.0, f64::max);
    let v = rig.v.lin_vel_world();
    let travel = v.y.atan2(v.x);
    println!("crab: travel {:.2}°, yaw rate ≤ {yaw:.4} rad/s at {:.2} m/s", travel.to_degrees(), v.length());
    assert!(v.length() > 2.0, "{v}");
    assert!(yaw < 0.01, "yaw rate {yaw}");
    assert!((travel - angle).abs() < 3f64.to_radians(), "travel {}°", travel.to_degrees());
}

/// Counter-phase rear steering halves the turn radius of front steering alone, as the
/// kinematic bicycle model predicts (`κ = (tan δ_f − tan δ_r)/L`), at low speed.
#[test]
fn counter_phase_steering_turns_tighter() {
    let map = per_wheel("e_sedan");
    let d = e_sedan();
    let wheelbase = d.axles[0].position.x - d.axles[1].position.x;
    let delta: f64 = 0.15;
    let curvature = |rear: f64| {
        let mut rig = Rig::new("e_sedan");
        let steer =
            [("steer_0", delta / 0.6), ("steer_1", delta / 0.6), ("steer_2", rear / 0.6), ("steer_3", rear / 0.6)];
        let mut k = 0.0;
        for step in 0..12_000 {
            // Hold about 3 m/s.
            let drive = (0.3 * (3.0 - rig.v.speed())).clamp(-0.3, 0.3);
            let mut set = steer.to_vec();
            set.extend([("drive_0", drive), ("drive_1", drive), ("drive_2", drive), ("drive_3", drive)]);
            rig.step(&map.setpoint(&action(&map, &set)));
            if step >= 10_000 {
                k += rig.v.ang_vel_body().z / rig.v.speed() / 2000.0;
            }
        }
        k
    };
    let (front, counter) = (curvature(0.0), curvature(-delta));
    let predicted = [delta.tan() / wheelbase, 2.0 * delta.tan() / wheelbase];
    println!("curvature front {front:.4} ({:.4}), counter-phase {counter:.4} ({:.4})", predicted[0], predicted[1]);
    assert!((front / predicted[0] - 1.0).abs() < 0.1, "front {front} vs {}", predicted[0]);
    assert!((counter / front / 2.0 - 1.0).abs() < 0.15, "ratio {}", counter / front);
}

/// Raw modes: the pedal brakes before reversing, and the side commands mix into left and right.
#[test]
fn raw_pedal_brakes_then_reverses() {
    let map = GroundActionMap::new(GroundActionMode::Raw, &GroundActionLimits::default(), &def("sedan_like")).unwrap();
    let mut rig = rolling("sedan_like", 5.0);
    let back = map.setpoint(&[-0.5, 0.0]);
    let speeds = rig.run(&back, 8.0, |r| r.v.speed());
    let stop = speeds.iter().position(|&v| v < 0.1).expect("stops");
    assert!(stop < 3000, "stopped after {} s", stop as f64 * DT);
    assert!(rig.v.speed() < -0.5 && rig.v.powertrain().gear == -1, "{} m/s", rig.v.speed());
}
