//! Multirotor model checks: presets, hover equilibrium, motor response, per-motor signs, yaw
//! decoupling, ground effect, drag, gyroscopic/reaction torques, battery, contacts, scales.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::multirotor::*;
use autonomousim_vehicles::{VehicleDef, VehicleError, presets};
use glam::{DQuat, DVec3};
use std::sync::Arc;

const G: f64 = 9.80665;
const GRAVITY: DVec3 = DVec3::new(0.0, 0.0, -G);
const DT: f64 = 0.002;

fn quad(name: &str, dt: f64) -> Multirotor {
    Multirotor::new(Arc::new(presets::multirotor(name).unwrap()), dt)
}

fn free_env(air: AirData) -> StepEnv<'static> {
    StepEnv { scene: None, gravity: GRAVITY, air, ground: None }
}

fn hover_at(q: &mut Multirotor, z: f64, density: f64) -> f64 {
    let w = q.def().hover_omega(G, density);
    q.reset(&InitialState::at(Pose::from_translation(DVec3::new(0.0, 0.0, z)), MotorInit::Speed(w)));
    w
}

#[test]
fn presets_load_with_documented_numbers() {
    assert_eq!(presets::names().collect::<Vec<_>>(), ["cf2x", "iris_like"]);
    let cf = presets::multirotor("cf2x").unwrap();
    assert!((cf.hover_omega(G, 1.225) - 1515.64).abs() < 0.1);
    assert!((cf.thrust_to_weight(G) - 2.25).abs() < 1e-3);
    let iris = presets::multirotor("iris_like").unwrap();
    assert!((iris.hover_omega(G, 1.225) - 793.54).abs() < 0.1);
    assert!((iris.thrust_to_weight(G) - 1.92).abs() < 0.01);
    assert_eq!(iris.colliders.iter().filter(|c| c.part == ColliderPart::Gear).count(), 4);
    assert!(matches!(presets::get("nope"), Err(VehicleError::UnknownPreset(_))));

    // JSON round trip (recordings embed vehicle definitions).
    let def = presets::get("iris_like").unwrap();
    let json = serde_json::to_string(&def).unwrap();
    assert_eq!(serde_json::from_str::<VehicleDef>(&json).unwrap(), def);
}

#[test]
fn definitions_are_validated() {
    let src = include_str!("../../../assets/vehicles/cf2x.toml");
    assert!(VehicleDef::from_toml(&src.replace("mass = 0.027", "mass = -1.0")).is_err());
    assert!(matches!(VehicleDef::from_toml(&src.replace("tau_up", "tau_upp")), Err(VehicleError::Parse(_))));
    assert!(VehicleDef::from_toml(&src.replace("omega_max = 2273.5", "omega_max = 0.0")).is_err());
    let tilted = src.replacen("spin = \"ccw\"", "spin = \"ccw\"\naxis = [0.0, 0.2, 2.0]", 1);
    let VehicleDef::Multirotor(m) = VehicleDef::from_toml(&tilted).unwrap();
    assert!((m.rotors[0].axis.length() - 1.0).abs() < 1e-15);
}

#[test]
fn hover_is_an_exact_equilibrium() {
    for name in ["cf2x", "iris_like"] {
        let mut q = quad(name, DT);
        let air = AirData { density: 1.1, wind: DVec3::ZERO };
        let w = hover_at(&mut q, 50.0, air.density);
        let cmd = vec![w; q.num_rotors()];
        for _ in 0..1000 {
            q.step(&cmd, &free_env(air)).unwrap();
        }
        assert!((q.specific_force_body() - DVec3::Z * G).length() < 1e-9, "{name}");
        assert!((q.position() - DVec3::new(0.0, 0.0, 50.0)).length() < 1e-9, "{name}: {}", q.position());
        assert!(q.ang_vel_body().length() < 1e-12 && q.lin_vel_world().length() < 1e-9, "{name}");
    }
}

#[test]
fn motors_follow_first_order_lag() {
    for (name, dt) in [("cf2x", 0.002), ("iris_like", 0.0005)] {
        let mut q = quad(name, dt);
        q.reset(&InitialState::at(Pose::from_translation(DVec3::Z * 100.0), MotorInit::Speed(0.0)));
        let (tau_up, tau_down) = (q.def().rotor.tau_up, q.def().rotor.tau_down);
        let target = 0.8 * q.def().rotor.omega_max;
        let cmd = vec![target; q.num_rotors()];
        for _ in 0..(tau_up / dt).round() as usize {
            q.step(&cmd, &free_env(AirData::default())).unwrap();
        }
        let want = target * (1.0 - (-1f64).exp());
        assert!(q.motor_speeds().iter().all(|w| (w - want).abs() < 1e-9 * target), "{name}: {:?}", q.motor_speeds());
        // Spin down from the target with the slower constant.
        q.set_motor_speeds(&cmd);
        let zero = vec![0.0; q.num_rotors()];
        for _ in 0..(tau_down / dt).round() as usize {
            q.step(&zero, &free_env(AirData::default())).unwrap();
        }
        assert!(q.motor_speeds().iter().all(|w| (w - target * (-1f64).exp()).abs() < 1e-9 * target), "{name}");
        // Commands beyond the limits are clamped.
        q.step(&vec![1e9; q.num_rotors()], &free_env(AirData::default())).unwrap();
        assert!(q.motor_speeds()[0] <= q.def().rotor.omega_max);
    }
}

#[test]
fn per_motor_torque_signs() {
    for name in ["cf2x", "iris_like"] {
        let mut q = quad(name, DT);
        let def = q.def().clone();
        for (i, m) in def.rotors.iter().enumerate() {
            let w = hover_at(&mut q, 50.0, 1.225);
            let mut speeds = vec![w; def.rotors.len()];
            speeds[i] *= 1.05;
            q.set_motor_speeds(&speeds);
            q.step(&speeds, &free_env(AirData::default())).unwrap();
            let a = q.ang_acc_body();
            // More thrust on the right (y < 0) rolls right side up (negative roll); at the front
            // it pitches the nose up (negative pitch); a faster CCW rotor yaws the body clockwise.
            assert_eq!(a.x.signum(), m.position.y.signum(), "{name} rotor {i}: {a}");
            assert_eq!(a.y.signum(), -m.position.x.signum(), "{name} rotor {i}: {a}");
            assert_eq!(a.z.signum(), -m.spin.sign(), "{name} rotor {i}: {a}");
            assert!(q.specific_force_body().z > G);
        }
    }
}

#[test]
fn pure_yaw_leaves_roll_and_pitch_alone() {
    let mut q = quad("cf2x", DT);
    // Rotor drag would damp the yaw rate (hubs move tangentially); leave it out for the check.
    q.set_scales(&MultirotorScales { rotor_drag: 0.0, ..Default::default() });
    let w = hover_at(&mut q, 50.0, 1.225);
    let spins: Vec<f64> = q.def().rotors.iter().map(|m| m.spin.sign()).collect();
    // Same total thrust, more drag torque from the CCW rotors.
    let speeds: Vec<f64> = spins.iter().map(|s| w * (1.0 + 0.1 * s).sqrt()).collect();
    q.set_motor_speeds(&speeds);
    for _ in 0..250 {
        q.step(&speeds, &free_env(AirData::default())).unwrap();
    }
    let omega = q.ang_vel_body();
    assert!(omega.x.abs() < 1e-12 && omega.y.abs() < 1e-12, "{omega}");
    let kq = q.def().rotor.k_torque;
    let want = -4.0 * 0.1 * kq * w * w / q.def().body.inertia.z * 0.5;
    assert!((omega.z / want - 1.0).abs() < 1e-9, "yaw rate {} vs {want}", omega.z);
    assert!((q.position().z - 50.0).abs() < 1e-9);
}

#[test]
fn ground_effect_raises_thrust_near_ground() {
    let mut q = quad("iris_like", DT);
    let r = q.def().rotor.radius;
    let hub_z = q.def().rotors[0].position.z;
    let w = hover_at(&mut q, r - hub_z, 1.225);
    let ground = FlatTerrain::new(0.0, MaterialId::GRASS);
    let plane = GroundPlane::below(&ground, q.position(), 2.0);
    let env = StepEnv { scene: None, gravity: GRAVITY, air: AirData::default(), ground: plane };
    q.step(&[w; 4], &env).unwrap();
    assert!((q.specific_force_body().z / G - 16.0 / 15.0).abs() < 1e-12);
}

#[test]
fn free_fall_reaches_drag_terminal_velocity() {
    let mut q = quad("iris_like", DT);
    q.reset(&InitialState::at(Pose::from_translation(DVec3::Z * 5000.0), MotorInit::Speed(0.0)));
    let cda = q.def().body.drag_area.z;
    for _ in 0..(30.0 / DT) as usize {
        q.step(&[0.0; 4], &free_env(AirData::default())).unwrap();
    }
    let vt = (2.0 * 1.5 * G / (1.225 * cda)).sqrt();
    assert!((q.lin_vel_world().z / -vt - 1.0).abs() < 1e-6, "{} vs {vt}", q.lin_vel_world().z);
}

#[test]
fn rotor_drag_and_wind_are_relative_airspeed() {
    let mut q = quad("cf2x", DT);
    q.set_scales(&MultirotorScales { body_drag: 0.0, ..Default::default() });
    let w = hover_at(&mut q, 50.0, 1.225);
    let mut init = InitialState::at(Pose::from_translation(DVec3::Z * 50.0), MotorInit::Speed(w));
    init.lin_vel_world = DVec3::new(2.0, 0.0, 0.0);
    q.reset(&init);
    q.step(&[w; 4], &free_env(AirData::default())).unwrap();
    let moving = q.external_wrench();
    let want = -4.0 * w * q.def().rotor.drag * 2.0 / q.mass();
    assert!((q.specific_force_body().x / want - 1.0).abs() < 1e-12);

    // Hovering in a 2 m/s tail wind from −x is the same as flying at 2 m/s in still air.
    init.lin_vel_world = DVec3::ZERO;
    q.reset(&init);
    q.step(&[w; 4], &free_env(AirData { density: 1.225, wind: DVec3::new(-2.0, 0.0, 0.0) })).unwrap();
    let windy = q.external_wrench();
    assert!((moving.lin - windy.lin).length() < 1e-15 && (moving.ang - windy.ang).length() < 1e-15);
}

/// A body with one rotor at the centre of mass and no aerodynamic torque: only the rotor
/// reaction and gyroscopic torques act, so body + rotor angular momentum is conserved.
fn gyro_testbed() -> Multirotor {
    let mut def = presets::multirotor("iris_like").unwrap();
    def.rotors.truncate(1);
    def.rotors[0].position = DVec3::ZERO;
    def.rotor.k_torque = 0.0;
    def.rotor.drag = 0.0;
    def.rotor.rolling_moment = 0.0;
    def.rotor.inertia = 1e-4;
    def.body.drag_area = DVec3::ZERO;
    def.body.inertia = DVec3::new(0.03, 0.04, 0.05);
    Multirotor::new(Arc::new(def), DT)
}

fn angular_momentum_world(q: &Multirotor) -> DVec3 {
    let i = q.def().body.inertia;
    q.orientation() * (i * q.ang_vel_body() + DVec3::Z * q.def().rotors[0].spin.sign() * 1e-4 * q.motor_speeds()[0])
}

#[test]
fn spin_up_reaction_conserves_angular_momentum() {
    let mut q = gyro_testbed();
    q.reset(&InitialState::at(Pose::IDENTITY, MotorInit::Speed(0.0)));
    let zero_g = StepEnv { scene: None, gravity: DVec3::ZERO, air: AirData::default(), ground: None };
    for _ in 0..200 {
        q.step(&[900.0], &zero_g).unwrap();
    }
    assert!(q.motor_speeds()[0] > 800.0);
    let l = angular_momentum_world(&q);
    assert!(l.length() < 1e-14, "{l}");
    assert!(q.ang_vel_body().z < 0.0); // CCW rotor spun up → body turns clockwise
}

#[test]
fn rotor_gyroscopics_conserve_angular_momentum() {
    let mut q = gyro_testbed();
    let init = InitialState {
        ang_vel_body: DVec3::new(1.0, 0.5, 0.2),
        ..InitialState::at(Pose::IDENTITY, MotorInit::Speed(600.0))
    };
    q.reset(&init);
    let zero_g = StepEnv { scene: None, gravity: DVec3::ZERO, air: AirData::default(), ground: None };
    let l0 = angular_momentum_world(&q);
    let e0 = (q.def().body.inertia * q.ang_vel_body()).dot(q.ang_vel_body());
    let mut max_err: f64 = 0.0;
    for _ in 0..(10.0 / DT) as usize {
        q.step(&[600.0], &zero_g).unwrap();
        max_err = max_err.max((angular_momentum_world(&q) - l0).length() / l0.length());
    }
    // The rotor momentum (0.06) dominates the body's, so a sign error gives an O(1) error. The
    // attitude update with the end-of-step rate leaves a bounded 1.8e-3 (no drift over 100 s).
    assert!(max_err < 3e-3, "relative angular momentum error {max_err}");
    let e = (q.def().body.inertia * q.ang_vel_body()).dot(q.ang_vel_body());
    assert!((e / e0 - 1.0).abs() < 1e-10, "body kinetic energy drift {}", e / e0 - 1.0);
}

#[test]
fn battery_sags_under_load() {
    let mut def = presets::multirotor("iris_like").unwrap();
    def.battery = Some(BatteryDef {
        cells: 3,
        capacity: 5.1,
        internal_resistance: 0.03,
        efficiency: 0.7,
        idle_power: 0.0,
        cell_full: 4.2,
        cell_empty: 3.3,
    });
    let mut q = Multirotor::new(Arc::new(def), DT);
    let w = hover_at(&mut q, 50.0, 1.225);
    assert_eq!(q.speed_range().1, 1100.0);
    for _ in 0..500 {
        q.step(&[w; 4], &free_env(AirData::default())).unwrap();
    }
    let b = *q.battery().unwrap();
    // Hover shaft power 4·k_Q·ω³ at 70 % efficiency.
    let p = 4.0 * q.def().rotor.k_torque * w.powi(3) / 0.7;
    assert!((b.voltage * b.current / p - 1.0).abs() < 1e-9);
    assert!(b.voltage < 12.6 && b.soc < 1.0);
    assert!((q.speed_range().1 - 1100.0 * b.voltage / 12.6).abs() < 1e-9);
}

#[test]
fn lands_on_its_gear() {
    let materials = MaterialTable::standard();
    let ground = FlatTerrain::new(0.0, MaterialId::GRASS);
    for name in ["cf2x", "iris_like"] {
        let mut q = quad(name, DT);
        let yaw = DQuat::from_rotation_z(0.4);
        q.reset(&InitialState::at(Pose::new(DVec3::new(1.0, 2.0, 0.3), yaw), MotorInit::Speed(0.0)));
        let scene = StaticScene { terrain: &ground, obstacles: &NoObstacles, materials: &materials };
        let n = q.num_rotors();
        for _ in 0..(3.0 / DT) as usize {
            let env = StepEnv { scene: Some(scene), gravity: GRAVITY, air: AirData::default(), ground: None };
            q.step(&vec![0.0; n], &env).unwrap();
        }
        let gear: Vec<_> = q.colliders().iter().filter(|c| c.group == ColliderPart::Gear as u8).collect();
        assert_eq!(q.contacts().len(), gear.len(), "{name}");
        assert!(q.contacts().iter().all(|c| q.colliders()[c.collider as usize].group == ColliderPart::Gear as u8));
        // Resting on the feet with the static penetration g/ω_c² (ω_c scaled by the material).
        let foot = gear[0];
        let omega_c = q.def().contact.omega_dt / DT * materials.get(MaterialId::GRASS).stiffness_scale;
        let want = foot.radius - foot.center.z - G / (omega_c * omega_c);
        assert!((q.position().z - want).abs() < 1e-6, "{name}: z {} vs {want}", q.position().z);
        assert!(q.lin_vel_world().length() < 1e-6 && q.ang_vel_body().length() < 1e-6);
        assert!((q.orientation().angle_between(yaw)).abs() < 1e-6);
    }
}

#[test]
fn scales_perturb_the_model() {
    let mut q = quad("iris_like", DT);
    q.set_scales(&MultirotorScales { mass: 1.2, k_thrust: vec![1.1], ..Default::default() });
    assert!((q.mass() - 1.8).abs() < 1e-12);
    assert!((q.inertia() - DVec3::new(0.029125, 0.029125, 0.055225) * 1.2).length() < 1e-12);
    let w = q.def().hover_omega(G, 1.225);
    q.reset(&InitialState::at(Pose::from_translation(DVec3::Z * 50.0), MotorInit::Speed(w)));
    q.step(&[w; 4], &free_env(AirData::default())).unwrap();
    assert!((q.specific_force_body().z - G * 4.1 / 4.0 / 1.2).abs() < 1e-9);
    assert!(q.ang_acc_body().x < 0.0); // rotor 0 (front right) is stronger
    q.set_scales(&MultirotorScales::default());
    assert!((q.mass() - 1.5).abs() < 1e-15);
}

#[test]
fn external_force_at_point() {
    let mut q = quad("iris_like", DT);
    q.reset(&InitialState::at(
        Pose::new(DVec3::new(0.0, 0.0, 50.0), DQuat::from_rotation_z(0.5)),
        MotorInit::Speed(0.0),
    ));
    q.begin_step();
    let p = q.position() + DVec3::new(0.0, 0.0, 0.1);
    q.apply_force(DVec3::new(3.0, 0.0, 0.0), p);
    let f = q.external_wrench();
    let rot = q.orientation();
    assert!((rot * f.lin - DVec3::new(3.0, 0.0, 0.0)).length() < 1e-12);
    assert!((rot * f.ang - DVec3::new(0.0, 0.3, 0.0)).length() < 1e-12);
    q.finish_step(DVec3::ZERO).unwrap();
    // Up to the O(ω·dt) rotation during the step.
    assert!((q.lin_vel_world() - DVec3::new(3.0, 0.0, 0.0) / 1.5 * DT).length() < 1e-6);
}
