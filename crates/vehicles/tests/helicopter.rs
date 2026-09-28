//! Helicopter family: presets, trim in hover and forward flight, standing on the skids, the
//! governed rotor speed.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::multirotor::ColliderPart;
use autonomousim_vehicles::multirotor::{AirData, StepEnv};
use autonomousim_vehicles::presets;
use autonomousim_vehicles::rotorcraft::*;
use glam::{DQuat, DVec3};
use std::sync::Arc;

const DT: f64 = 0.002;
const G: f64 = 9.80665;
const GRAVITY: DVec3 = DVec3::new(0.0, 0.0, -G);
const RHO: f64 = 1.225;

fn heli(name: &str) -> Helicopter {
    Helicopter::new(Arc::new(presets::helicopter(name).unwrap()), DT)
}

/// Trim speeds (m/s) per preset.
const CASES: [(&str, &[f64]); 2] = [("bo105_like", &[0.0, 30.0, 60.0]), ("xcell60_like", &[0.0, 10.0])];

fn free_air() -> StepEnv<'static> {
    StepEnv { scene: None, gravity: GRAVITY, air: AirData::default(), ground: None }
}

fn start_at_trim(h: &mut Helicopter, t: &HelicopterTrim) {
    let rot = t.attitude(0.0);
    h.reset(&HelicopterInit {
        pose: Pose::new(DVec3::new(0.0, 0.0, 500.0), rot),
        lin_vel_world: rot * t.velocity_body,
        ang_vel_body: DVec3::ZERO,
        controls: t.controls,
        rotor_speed: Some(t.rotor_speed),
        density: RHO,
    });
}

fn flat_scene() -> (FlatTerrain, MaterialTable) {
    (FlatTerrain::new(0.0, MaterialId(0)), MaterialTable::standard())
}

#[test]
fn presets_load() {
    for (name, _) in CASES {
        let def = presets::helicopter(name).unwrap();
        def.validate().unwrap();
        assert!(def.skid_height().unwrap() > 0.0, "{name}");
        let back: HelicopterDef = toml::from_str(&toml::to_string(&def).unwrap()).unwrap();
        assert_eq!(back.main_rotor.rotor, def.main_rotor.rotor, "{name}");
    }
    assert!(presets::helicopter("iris_like").is_err());
}

/// Hover and forward trims within the control range; hover power close to momentum theory
/// with the tip-loss factor plus the profile power; the Bo105's power bucket.
#[test]
fn trims_in_hover_and_forward_flight() {
    for (name, speeds) in CASES {
        let h = heli(name);
        let def = h.def();
        let r = &def.main_rotor.rotor;
        let mut powers = Vec::new();
        for &v in speeds {
            let t = h.trim(v, RHO, G).unwrap_or_else(|e| panic!("{e}"));
            assert!(t.controls.to_array().iter().all(|u| u.abs() < 0.9), "{name} {v}: {:?}", t.controls);
            assert!(t.pitch.abs() < 0.15 && t.roll.abs() < 0.1, "{name} {v}: {} {}", t.pitch, t.roll);
            assert!((t.rotor_speed - def.engine.rated_speed).abs() < 1e-12);
            assert!(t.power > 0.0 && t.power < def.engine.max_power, "{name} {v}: {}", t.power);
            // The main rotor carries the weight and (in forward flight) the drag, plus the
            // download on the fuselage and tailplane when nose down at high speed.
            let drag = 0.5 * RHO * v * v * def.fuselage.drag_area.x;
            let need = (h.mass() * G).hypot(drag);
            let ratio = t.loads.main.force.length() / need;
            assert!(ratio > 0.99 && ratio < 1.12, "{name} {v}: {ratio}");
            powers.push(t.power);
            if v == 0.0 {
                let thrust = t.loads.main.force.z;
                let b = r.tip_loss;
                let a = r.disc_area();
                let tip = def.engine.rated_speed * r.radius;
                let induced = thrust * (thrust / (2.0 * RHO * a)).sqrt() / b;
                let profile = RHO * a * tip.powi(3) * r.solidity() * r.profile_drag / 8.0;
                let main = t.loads.main.power;
                assert!((main / (induced + profile) - 1.0).abs() < 0.15, "{name}: {main} vs {induced} + {profile}");
                // The tail rotor takes a modest share.
                assert!(t.power - main > 0.03 * main && t.power - main < 0.2 * main, "{name}: {}", t.power);
                // Nose up with the forward shaft tilt; the tail rotor thrust is balanced by
                // a small disc tilt against it.
                assert!(t.pitch > 0.0 || name != "bo105_like");
            }
        }
        if name == "bo105_like" {
            // Power bucket: cruise at 30 m/s needs much less than hover or 60 m/s.
            assert!(powers[1] < 0.7 * powers[0] && powers[1] < 0.7 * powers[2], "{powers:?}");
        }
    }
}

/// Released at a trim, the helicopter starts with (almost) no acceleration.
#[test]
fn trim_is_an_equilibrium() {
    for (name, speeds) in CASES {
        for &v in speeds {
            let mut h = heli(name);
            let t = h.trim(v, RHO, G).unwrap();
            start_at_trim(&mut h, &t);
            let (v0, w0) = (h.lin_vel_world(), h.ang_vel_body());
            h.step(&t.controls, &free_air()).unwrap();
            let acc = (h.lin_vel_world() - v0) / DT;
            let ang = (h.ang_vel_body() - w0) / DT;
            assert!(acc.length() < 1e-3, "{name} {v}: acc {acc}");
            assert!(ang.length() < 1e-3, "{name} {v}: ang {ang}");
            assert!(h.rotor_acceleration().abs() < 1e-6, "{name} {v}: {}", h.rotor_acceleration());
        }
    }
}

/// Engine running, collective down: it stands still on its skids with nothing else touching.
#[test]
fn stands_on_its_skids() {
    let (ground, materials) = flat_scene();
    let scene = StaticScene { terrain: &ground, obstacles: &NoObstacles, materials: &materials };
    for (name, _) in CASES {
        let mut h = heli(name);
        let z = h.def().skid_height().unwrap();
        h.reset(&HelicopterInit::at_rest(Pose::new(DVec3::new(0.0, 0.0, z + 0.01), DQuat::IDENTITY)));
        let controls = *h.hold_input();
        for _ in 0..(5.0 / DT) as usize {
            let env = StepEnv { scene: Some(scene), gravity: GRAVITY, air: AirData::default(), ground: None };
            h.step(&controls, &env).unwrap();
        }
        let gear = h.def().colliders.iter().filter(|c| c.part == ColliderPart::Gear).count();
        assert!(!h.contacts().is_empty(), "{name}");
        assert!(h.contacts().iter().all(|c| (c.collider as usize) < gear), "{name}: airframe contact");
        assert!(h.lin_vel_body().length() < 0.01 && h.ang_vel_body().length() < 0.01, "{name}");
        assert!(h.position().truncate().length() < 0.05, "{name}: slid to {}", h.position());
        let tilt = (h.orientation() * DVec3::Z).z.acos();
        assert!(tilt < 0.05, "{name}: tilt {tilt}");
        let omega = h.rotor_speed();
        assert!((omega / h.def().engine.rated_speed - 1.0).abs() < 1e-3, "{name}: {omega}");
    }
}

/// A collective step from hover loads the rotor; the governor restores the rotor speed.
#[test]
fn rotor_speed_recovers_from_load_steps() {
    for (name, _) in CASES {
        let mut h = heli(name);
        let t = h.trim(0.0, RHO, G).unwrap();
        start_at_trim(&mut h, &t);
        let rated = h.def().engine.rated_speed;
        let mut c = t.controls;
        c.collective += 0.3;
        let mut low = rated;
        for _ in 0..(6.0 / DT) as usize {
            h.step(&c, &free_air()).unwrap();
            low = low.min(h.rotor_speed());
        }
        let droop = 1.0 - low / rated;
        assert!(droop > 0.002 && droop < 0.1, "{name}: droop {droop}");
        assert!((h.rotor_speed() / rated - 1.0).abs() < 2e-3, "{name}: {}", h.rotor_speed());
        assert!(h.engine_power() > t.power, "{name}");
    }
}

/// Engine off: the rotor decelerates at load torque over drive inertia.
#[test]
fn engine_failure_decays_rotor_speed() {
    for (name, _) in CASES {
        let mut h = heli(name);
        let t = h.trim(0.0, RHO, G).unwrap();
        start_at_trim(&mut h, &t);
        h.set_engine_running(false);
        let def = h.def().clone();
        let torque = t.loads.main.torque + def.tail_gear_ratio * t.loads.tail.torque;
        let expect = -torque / def.drive_inertia();
        let mut omega = h.rotor_speed();
        for _ in 0..(1.0 / DT) as usize {
            h.step(&t.controls, &free_air()).unwrap();
            assert!(h.rotor_speed() < omega, "{name}");
            omega = h.rotor_speed();
        }
        assert!(h.engine_torque() < 1e-6 * torque, "{name}: {}", h.engine_torque());
        let rate = h.rotor_acceleration();
        assert!(rate < 0.3 * expect && rate > 1.5 * expect, "{name}: {rate} vs {expect}");
    }
}
