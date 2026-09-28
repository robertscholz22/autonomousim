//! Closed-loop checks of the helicopter controller on both presets: attitude and velocity steps
//! in hover and at 20 m/s, position hold in hover in a steady wind, and the action maps.

use autonomousim_control::AgentActionMode;
use autonomousim_control::fixedwing::euler;
use autonomousim_control::rotorcraft::*;
use autonomousim_core::math::Pose;
use autonomousim_vehicles::Family;
use autonomousim_vehicles::aero::AirData;
use autonomousim_vehicles::multirotor::StepEnv;
use autonomousim_vehicles::presets;
use autonomousim_vehicles::rotorcraft::{Helicopter, HelicopterInit};
use glam::DVec3;
use std::sync::Arc;

const G: f64 = 9.80665;
const DT: f64 = 0.002;
const RHO: f64 = 1.225;
const PRESETS: [&str; 2] = ["bo105_like", "xcell60_like"];

fn setup(name: &str) -> (Helicopter, HelicopterController) {
    let def = Arc::new(presets::helicopter(name).unwrap());
    let c = HelicopterController::new(&def, DT, &HelicopterConfig::default()).unwrap();
    (Helicopter::new(def, DT), c)
}

/// Trimmed level flight at `v` heading east (yaw 0) at 500 m.
fn start(h: &mut Helicopter, c: &mut HelicopterController, v: f64) {
    let t = h.trim(v, RHO, G).unwrap();
    let rot = t.attitude(0.0);
    h.reset(&HelicopterInit {
        pose: Pose::new(DVec3::new(0.0, 0.0, 500.0), rot),
        lin_vel_world: rot * t.velocity_body,
        ang_vel_body: DVec3::ZERO,
        controls: t.controls,
        rotor_speed: Some(t.rotor_speed),
        density: RHO,
    });
    c.reset();
}

/// Fly `seconds` tracking `sp` in a steady `wind`, calling `each(t, heli)` every step.
fn fly(
    h: &mut Helicopter,
    c: &mut HelicopterController,
    seconds: f64,
    sp: HelicopterSetpoint,
    wind: DVec3,
    mut each: impl FnMut(f64, &Helicopter),
) {
    let env = StepEnv {
        scene: None,
        gravity: DVec3::new(0.0, 0.0, -G),
        air: AirData { density: RHO, wind, ..AirData::default() },
        ground: None,
    };
    for i in 0..(seconds / DT).round() as usize {
        let input = c.update(&sp, h);
        h.step(&input, &env).unwrap();
        each((i + 1) as f64 * DT, h);
    }
}

/// Heading-frame velocity (forward, left, up).
fn velocity(h: &Helicopter) -> DVec3 {
    let yaw = euler(h.orientation()).2;
    glam::DQuat::from_rotation_z(-yaw) * h.lin_vel_world()
}

#[test]
fn schedules_both_presets() {
    for name in PRESETS {
        let (_, c) = setup(name);
        let [kr, ka, kv, kp, kz] = c.bandwidths();
        assert!(kr > 2.0 && kr < 20.0 && ka < kr && kv < ka && kp < kv && kz > 0.0, "{name}: {:?}", c.bandwidths());
        assert!(c.max_speed() > 20.0, "{name}: {}", c.max_speed());
    }
}

/// 10° roll and pitch steps from hover and from 20 m/s settle within a few attitude time
/// constants with little overshoot, holding the yaw rate at zero.
#[test]
fn attitude_steps_settle() {
    for name in PRESETS {
        for v in [0.0, 20.0] {
            let (mut h, mut c) = setup(name);
            let (trim_in, pitch0, roll0) = c.trim_at(v);
            let step = 10f64.to_radians();
            for (droll, dpitch) in [(step, 0.0), (0.0, step), (-step, 0.0), (0.0, -step)] {
                start(&mut h, &mut c, v);
                let target = (roll0 + droll, pitch0 + dpitch);
                let sp = HelicopterSetpoint::Attitude {
                    roll: target.0,
                    pitch: target.1,
                    yaw_rate: 0.0,
                    collective: trim_in.collective,
                };
                let settle = 4.0 / c.bandwidths()[1];
                let mut peak: f64 = 0.0;
                fly(&mut h, &mut c, settle + 1.0, sp, DVec3::ZERO, |t, h| {
                    let (r, p, _) = euler(h.orientation());
                    let e = (r - target.0).abs().max((p - target.1).abs());
                    if t > settle {
                        peak = peak.max(e);
                    }
                });
                let (r, p, _) = euler(h.orientation());
                assert!(peak < 0.15 * step, "{name} {v} ({droll}, {dpitch}): error {peak} after {settle} s; {r} {p}");
                assert!(h.ang_vel_body().z.abs() < 0.1, "{name} {v}: yaw rate {}", h.ang_vel_body());
            }
        }
    }
}

/// Velocity steps of 2 m/s forward, sideways and up from hover and from 20 m/s settle, with
/// the rotor speed governed throughout.
#[test]
fn velocity_steps_settle() {
    for name in PRESETS {
        for v in [0.0, 20.0] {
            let (mut h, mut c) = setup(name);
            for dv in [DVec3::X, DVec3::Y, DVec3::Z, -DVec3::Y, -DVec3::Z] {
                start(&mut h, &mut c, v);
                let target = DVec3::new(v, 0.0, 0.0) + 2.0 * dv;
                let sp = HelicopterSetpoint::Velocity { velocity: target, yaw_rate: 0.0 };
                let omega = h.def().engine.rated_speed;
                let mut droop: f64 = 0.0;
                fly(&mut h, &mut c, 15.0, sp, DVec3::ZERO, |_, h| {
                    droop = droop.max((h.rotor_speed() / omega - 1.0).abs());
                });
                let err = velocity(&h) - target;
                assert!(err.length() < 0.2, "{name} {v} {dv}: {} vs {target}", velocity(&h));
                assert!(h.ang_vel_body().length() < 0.05, "{name} {v} {dv}: {}", h.ang_vel_body());
                assert!(droop < 0.05, "{name} {v} {dv}: rotor speed {droop}");
                let (r, p, _) = euler(h.orientation());
                let (_, pitch, roll) = c.trim_at(target.x);
                assert!((r - roll).abs() < 0.1 && (p - pitch).abs() < 0.1, "{name} {v} {dv}: {r} {p}");
            }
        }
    }
}

/// Hover holds its position and heading in a 5 m/s crosswind (the Bo105) or 3 m/s (the
/// X-Cell) from the side and from behind.
#[test]
fn hover_holds_position_in_wind() {
    for (name, speed) in [("bo105_like", 5.0), ("xcell60_like", 3.0)] {
        for wind in [DVec3::new(0.0, speed, 0.0), DVec3::new(-speed, 0.0, 0.0), DVec3::new(0.6, -0.8, 0.0) * speed] {
            let (mut h, mut c) = setup(name);
            start(&mut h, &mut c, 0.0);
            let home = h.position();
            let sp = HelicopterSetpoint::Position { position: home, yaw: 0.0 };
            let mut worst: f64 = 0.0;
            fly(&mut h, &mut c, 40.0, sp, wind, |t, h| {
                if t > 30.0 {
                    worst = worst.max(h.position().distance(home));
                }
            });
            assert!(worst < 0.3, "{name} {wind}: {} from {home}", h.position());
            assert!(euler(h.orientation()).2.abs() < 0.02, "{name} {wind}: heading {}", euler(h.orientation()).2);
        }
    }
}

#[test]
fn action_maps() {
    let def = Arc::new(presets::helicopter("bo105_like").unwrap());
    let c = HelicopterController::new(&def, DT, &HelicopterConfig::default()).unwrap();
    let limits = HelicopterActionLimits::default();
    for mode in HelicopterActionMode::ALL {
        assert_eq!(mode.name().parse::<HelicopterActionMode>().unwrap(), mode);
        let m = HelicopterActionMap::new(mode, &limits, &def, &c).unwrap();
        assert_eq!(m.dim(), 4);
    }
    let vel = HelicopterActionMap::new(HelicopterActionMode::Velocity, &limits, &def, &c).unwrap();
    let [fwd, side, up] = vel.speeds();
    assert!(fwd > 20.0 && side > 10.0 && side < 30.0 && up > 2.0 && up < 10.0, "{:?}", vel.speeds());
    match vel.setpoint(&[1.0, -1.0, 0.5, 1.0]) {
        HelicopterSetpoint::Velocity { velocity, yaw_rate } => {
            assert_eq!(velocity, DVec3::new(fwd, -side, 0.5 * up));
            assert_eq!(yaw_rate, limits.yaw_rate);
        }
        s => panic!("{s:?}"),
    }
    match vel.setpoint(&[-1.0, 0.0, 0.0, 0.0]) {
        HelicopterSetpoint::Velocity { velocity, .. } => assert_eq!(velocity.x, -side),
        s => panic!("{s:?}"),
    }
    // Shared names resolve to the helicopter's modes; the default is `velocity`.
    for name in ["rates", "attitude", "velocity"] {
        let m: AgentActionMode = name.parse().unwrap();
        assert_eq!(m.resolve(Family::Rotorcraft), AgentActionMode::Helicopter(name.parse().unwrap()));
    }
    assert_eq!(AgentActionMode::default_for(Family::Rotorcraft).name(), "velocity");
}
