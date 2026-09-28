//! Fixed-wing model checks: presets, trim over a speed range, trimmed flight holds, control
//! signs, energy in an engine-off glide, standing on the gear, and a takeoff roll.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::fixedwing::*;
use autonomousim_vehicles::multirotor::{AirData, GroundPlane, StepEnv};
use autonomousim_vehicles::presets;
use glam::DVec3;
use std::sync::Arc;

const G: f64 = 9.80665;
const GRAVITY: DVec3 = DVec3::new(0.0, 0.0, -G);
const DT: f64 = 0.002;
const RHO: f64 = 1.225;

fn aircraft(name: &str) -> FixedWing {
    FixedWing::new(Arc::new(presets::fixed_wing(name).unwrap()), DT)
}

fn free_env() -> StepEnv<'static> {
    StepEnv { scene: None, gravity: GRAVITY, air: AirData::default(), ground: None }
}

/// Reset to trimmed flight heading east at 500 m.
fn start_trimmed(a: &mut FixedWing, t: &Trim) {
    let rot = t.attitude(0.0);
    a.reset(&FixedWingInit {
        pose: Pose::new(DVec3::new(0.0, 0.0, 500.0), rot),
        lin_vel_world: rot * t.velocity_body(),
        ang_vel_body: DVec3::ZERO,
        controls: t.controls,
        rotor_speed: Some(t.rotor_speed),
        soc: 1.0,
    });
}

#[test]
fn presets_are_consistent() {
    for name in ["aerosonde_like", "c172_like"] {
        let a = aircraft(name);
        let d = a.def();
        // Commands +1 roll right, pitch up and yaw right whatever the derivative signs.
        let signs = a.control_signs();
        assert!(signs.iter().all(|s| s.abs() == 1.0), "{name}: {signs:?}");
        let (hi, lo) = a.stall_angles();
        assert!(hi > 0.2 && hi < 0.5 && lo < 0.0, "{name}: stall {hi} {lo}");
        assert_eq!(a.colliders().len(), d.colliders.len() + d.gear.len());
    }
    // The c172's stall at the top of JSBSim's lift table (0.28 rad).
    let c = aircraft("c172_like");
    assert!((c.stall_angles().0 - 0.28).abs() < 1e-9);
    // Beard & McLain: elevator and rudder derivatives are negative.
    assert_eq!(aircraft("aerosonde_like").control_signs(), [1.0, -1.0, -1.0]);
}

#[test]
fn trims_over_the_speed_range() {
    for (name, speeds) in [("aerosonde_like", [18.0, 23.0, 28.0]), ("c172_like", [30.0, 45.0, 60.0])] {
        let a = aircraft(name);
        let mut last_alpha = f64::INFINITY;
        for v in speeds {
            let t = a.trim(v, RHO, 0.0, 0.0, G).unwrap_or_else(|e| panic!("{e}"));
            assert!(t.residual < 1e-8, "{name} at {v}: residual {}", t.residual);
            // Faster: less angle of attack; level flight: pitch = α without bank.
            assert!(t.alpha < last_alpha, "{name} at {v}: α {}", t.alpha);
            last_alpha = t.alpha;
            assert!(t.bank.abs() < 0.05, "{name} at {v}: bank {}", t.bank);
            assert!(t.controls.throttle > 0.05 && t.controls.throttle < 1.0, "{name} at {v}: {:?}", t.controls);
            // Lift ≈ weight: the vertical force balance holds with the thrust component.
            let q = 0.5 * RHO * v * v;
            let cl = a.def().body.mass * G / (q * a.def().geometry.area);
            assert!(cl > 0.1 && cl < 1.5, "{name} at {v}: CL {cl}");
        }
    }
    // A climb needs more throttle than level flight.
    let a = aircraft("c172_like");
    let level = a.trim(40.0, RHO, 0.0, 0.0, G).unwrap();
    let climb = a.trim(40.0, RHO, 0.05, 0.0, G).unwrap();
    assert!(climb.controls.throttle > level.controls.throttle + 0.05);
    // Flaps: less angle of attack at the same speed.
    let flaps = a.trim(35.0, RHO, 0.0, 1.0, G).unwrap();
    assert!(flaps.alpha < a.trim(35.0, RHO, 0.0, 0.0, G).unwrap().alpha);
    // Beyond the envelope: too slow for the elevator, too fast for the motor.
    assert!(aircraft("aerosonde_like").trim(8.0, RHO, 0.0, 0.0, G).is_err());
    assert!(aircraft("aerosonde_like").trim(40.0, RHO, 0.0, 0.0, G).is_err());
}

/// Released at trim, the aircraft holds speed, height and attitude for a few seconds (the
/// spiral and phugoid modes act slowly).
#[test]
fn trimmed_flight_holds() {
    for (name, v) in [("aerosonde_like", 25.0), ("c172_like", 50.0)] {
        let mut a = aircraft(name);
        let t = a.trim(v, RHO, 0.0, 0.0, G).unwrap();
        start_trimmed(&mut a, &t);
        let (z0, rot0) = (a.position().z, a.orientation());
        let controls = t.controls;
        for _ in 0..(5.0 / DT) as usize {
            a.step(&controls, &free_env()).unwrap();
        }
        let dz = a.position().z - z0;
        let dv = a.flow().airspeed - v;
        let angle = a.orientation().angle_between(rot0);
        assert!(dz.abs() < 1.0 && dv.abs() < 0.3 && angle < 0.03, "{name}: dz {dz} dv {dv} angle {angle}");
        assert!(a.ang_vel_body().length() < 0.01, "{name}: ω {}", a.ang_vel_body());
        assert!((a.rotor_speed() - t.rotor_speed).abs() < 1e-3 * t.rotor_speed);
        assert!(!a.stalled());
    }
}

#[test]
fn control_signs() {
    for name in ["aerosonde_like", "c172_like"] {
        let mut a = aircraft(name);
        let t = a.trim(if name == "c172_like" { 50.0 } else { 25.0 }, RHO, 0.0, 0.0, G).unwrap();
        let response = |a: &mut FixedWing, f: &dyn Fn(&mut FixedWingInput)| {
            start_trimmed(a, &t);
            let mut c = t.controls;
            f(&mut c);
            for _ in 0..(0.5 / DT) as usize {
                a.step(&c, &free_env()).unwrap();
            }
            a.ang_vel_body()
        };
        // Roll right is +x, nose up is −y, nose right is −z (FLU).
        let w = response(&mut a, &|c| c.aileron += 0.3);
        assert!(w.x > 0.1, "{name}: aileron {w}");
        let w = response(&mut a, &|c| c.elevator += 0.3);
        assert!(w.y < -0.05, "{name}: elevator {w}");
        let w = response(&mut a, &|c| c.rudder += 0.3);
        assert!(w.z < -0.02, "{name}: rudder {w}");
    }
}

/// Engine off: the mechanical energy lost equals the work of the aerodynamic forces (and the
/// windmilling propeller), and the glide settles at roughly the lift-to-drag ratio.
#[test]
fn engine_off_glide_energy() {
    for (name, v) in [("aerosonde_like", 22.0), ("c172_like", 36.0)] {
        let mut a = aircraft(name);
        let t = a.trim(v, RHO, 0.0, 0.0, G).unwrap();
        start_trimmed(&mut a, &t);
        a.set_engine_running(false);
        let controls = FixedWingInput { throttle: 0.0, ..t.controls };
        let e0 = a.energy(G);
        let mut work = 0.0;
        let (mut drag_work, p0) = (0.0, a.position());
        for _ in 0..(20.0 / DT) as usize {
            a.step(&controls, &free_env()).unwrap();
            // Rectangle rule on the power of the step's forces at the new velocities.
            work += a.external_power() * DT;
            let f = a.orientation() * a.external_wrench().lin;
            drag_work += f.dot(a.lin_vel_world()) * DT;
        }
        let loss = e0 - a.energy(G);
        assert!(loss > 0.0, "{name}: energy must decrease");
        // Semi-implicit Euler: the balance holds to first order in dt.
        assert!((loss + work).abs() < 0.02 * loss, "{name}: lost {loss} J, forces did {work} J");
        assert!(drag_work < 0.0);
        // Glide ratio: distance over energy height lost (the phugoid trades speed for height),
        // between 5 and 20 for these airframes with the windmilling propeller.
        let p1 = a.position();
        let ratio = (p1 - p0).truncate().length() * a.mass() * G / loss;
        assert!(ratio > 5.0 && ratio < 20.0, "{name}: glide ratio {ratio}");
    }
}

fn flat_scene() -> (FlatTerrain, MaterialTable) {
    (FlatTerrain::new(0.0, MaterialId(0)), MaterialTable::standard())
}

#[test]
fn stands_on_its_gear() {
    let (ground, materials) = flat_scene();
    let scene = StaticScene { terrain: &ground, obstacles: &NoObstacles, materials: &materials };
    for name in ["aerosonde_like", "c172_like"] {
        let mut a = aircraft(name);
        let (rot, h) = a.def().resting_pose(G).unwrap();
        a.reset(&FixedWingInit::at_rest(Pose::new(DVec3::new(0.0, 0.0, h + 0.02), rot)));
        let idle = a.rotor_speed();
        let controls = *a.hold_input();
        for _ in 0..(5.0 / DT) as usize {
            let env = StepEnv { scene: Some(scene), gravity: GRAVITY, air: AirData::default(), ground: None };
            a.step(&controls, &env).unwrap();
        }
        // At rest on all wheels, nothing else touching; the wheels carry the weight less the
        // idle thrust's small share.
        let wheels: Vec<_> = a.wheels().iter().map(|w| w.expect("wheel on the ground")).collect();
        let load: f64 = wheels.iter().map(|w| w.normal_force).sum();
        let weight = a.mass() * G;
        assert!((load - weight).abs() < 0.02 * weight, "{name}: load {load} vs {weight}");
        assert!(a.lin_vel_body().length() < 0.01 && a.ang_vel_body().length() < 0.01, "{name}");
        assert!(
            a.contacts().iter().all(|c| c.collider as usize >= a.def().colliders.len()),
            "{name}: airframe contact"
        );
        assert!(a.weight_on_wheels() && !a.stalled());
        // The brakes hold it against the idle thrust.
        assert!(a.position().truncate().length() < 0.05, "{name}: rolled to {}", a.position());
        let pitch = (a.orientation() * DVec3::X).z.asin();
        assert!(pitch.abs() < 0.1, "{name}: pitch {pitch}");
        if name == "c172_like" {
            assert!((idle * 60.0 / std::f64::consts::TAU - 550.0).abs() < 5.0, "idle {idle}");
        }
    }
}

/// Full throttle, brakes off, rotate at 55 kt: airborne after a ground roll of a few hundred
/// metres (the POH gives about 290 m at 2450 lb; JSBSim's c172p needs a similar distance).
#[test]
fn c172_takes_off() {
    let (ground, materials) = flat_scene();
    let scene = StaticScene { terrain: &ground, obstacles: &NoObstacles, materials: &materials };
    let mut a = aircraft("c172_like");
    let (rot, h) = a.def().resting_pose(G).unwrap();
    a.reset(&FixedWingInit::at_rest(Pose::new(DVec3::new(0.0, 0.0, h), rot)));
    let mut airborne = None;
    for i in 0..(40.0 / DT) as usize {
        // Rotate to 6° nose up at 55 kt with a pitch-attitude hold (nose-up pitch rate is −ω_y).
        let pitch = (a.orientation() * DVec3::X).z.asin();
        let elevator = if a.flow().airspeed > 55.0 * 0.5144 {
            (3.0 * (0.1 - pitch) + 0.5 * a.ang_vel_body().y).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let c = FixedWingInput { throttle: 1.0, elevator, ..Default::default() };
        let pos = a.position();
        let plane = GroundPlane { point: DVec3::new(pos.x, pos.y, 0.0), normal: DVec3::Z };
        let env = StepEnv { scene: Some(scene), gravity: GRAVITY, air: AirData::default(), ground: Some(plane) };
        a.step(&c, &env).unwrap();
        if airborne.is_none() && !a.weight_on_wheels() && i > 100 {
            airborne = Some(a.position());
        }
        if let Some(c) = a.contacts().iter().find(|c| (c.collider as usize) < a.def().colliders.len()) {
            panic!(
                "strike of collider {} at {:.2} s: {} m/s pitch {:.3} pos {}",
                c.collider,
                i as f64 * DT,
                a.flow().airspeed,
                (a.orientation() * DVec3::X).z.asin(),
                a.position()
            );
        }
    }
    let lift_off = airborne.expect("never left the ground");
    eprintln!("lift-off at {lift_off}");
    // JSBSim's c172p reaches 55 kt in 199 m; lift-off follows the rotation.
    assert!(lift_off.x > 150.0 && lift_off.x < 350.0, "ground roll {} m", lift_off.x);
    // Hands off on the runway, the nose-wheel steering and propeller torque keep it close to
    // the centre line (in the air the torque slowly turns it left).
    assert!(lift_off.y.abs() < 5.0, "tracked straight: {lift_off}");
    assert!(a.position().z > 30.0, "climbing: {}", a.position());
}
