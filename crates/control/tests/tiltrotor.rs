//! Closed-loop checks of the tiltrotor controller on the preset: the conversion schedule,
//! hover and cruise velocity steps, transitions to the wing and back holding altitude without
//! a stall, attitude mode, and the action maps.

use autonomousim_control::fixedwing::euler;
use autonomousim_control::tiltrotor::*;
use autonomousim_vehicles::aero::AirData;
use autonomousim_vehicles::multirotor::StepEnv;
use autonomousim_vehicles::presets;
use autonomousim_vehicles::tiltrotor::Tiltrotor;
use glam::DVec3;
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

const G: f64 = 9.80665;
const DT: f64 = 0.002;
const RHO: f64 = 1.225;
const ALTITUDE: f64 = 300.0;

fn setup() -> (Tiltrotor, TiltrotorController) {
    let def = Arc::new(presets::tiltrotor("quadtilt_like").unwrap());
    let c = TiltrotorController::new(&def, DT, &TiltrotorConfig::default()).unwrap();
    (Tiltrotor::new(def, DT), c)
}

/// Scheduled level flight at `v` heading east (yaw 0) at [`ALTITUDE`].
fn start(t: &mut Tiltrotor, c: &mut TiltrotorController, v: f64) {
    let p = c.schedule().points().iter().min_by(|a, b| (a.speed - v).abs().total_cmp(&(b.speed - v).abs())).unwrap();
    t.reset(&p.trim.init(DVec3::new(0.0, 0.0, ALTITUDE), 0.0));
    c.reset();
}

/// What a flight saw: the largest altitude error (m) and wing angle of attack (rad), and the
/// heading-frame velocity at the end.
#[derive(Debug)]
struct Flight {
    altitude: f64,
    alpha: f64,
    velocity: DVec3,
}

/// Fly `seconds` tracking `sp`, calling `each(time, aircraft)` every step.
fn fly(
    t: &mut Tiltrotor,
    c: &mut TiltrotorController,
    seconds: f64,
    sp: TiltrotorSetpoint,
    each: impl FnMut(f64, &Tiltrotor),
) -> Flight {
    fly_in(DVec3::ZERO, t, c, seconds, sp, each)
}

/// [`fly`] in a steady `wind` (m/s, world frame).
fn fly_in(
    wind: DVec3,
    t: &mut Tiltrotor,
    c: &mut TiltrotorController,
    seconds: f64,
    sp: TiltrotorSetpoint,
    mut each: impl FnMut(f64, &Tiltrotor),
) -> Flight {
    let env = StepEnv {
        scene: None,
        gravity: DVec3::new(0.0, 0.0, -G),
        air: AirData { density: RHO, wind, ..AirData::default() },
        ground: None,
    };
    let (mut altitude, mut alpha) = (0.0f64, f64::NEG_INFINITY);
    for i in 0..(seconds / DT).round() as usize {
        let input = c.update(&sp, t);
        t.step(&input, &env).unwrap();
        altitude = altitude.max((t.position().z - ALTITUDE).abs());
        if t.flow().airspeed > 5.0 {
            alpha = alpha.max(t.flow().alpha);
        }
        each((i + 1) as f64 * DT, t);
    }
    Flight { altitude, alpha, velocity: velocity(t) }
}

/// Heading-frame velocity (forward, left, up).
fn velocity(t: &Tiltrotor) -> DVec3 {
    let yaw = euler(t.orientation()).2;
    glam::DQuat::from_rotation_z(-yaw) * t.lin_vel_world()
}

fn level(speed: f64) -> TiltrotorSetpoint {
    TiltrotorSetpoint::Velocity { velocity: DVec3::new(speed, 0.0, 0.0), yaw_rate: 0.0 }
}

#[test]
fn schedules_the_conversion() {
    let (_, c) = setup();
    let s = c.schedule();
    let vs = s.stall_speed();
    let p = s.points();
    assert_eq!(p[0].tilt, 0.0);
    assert!(p.windows(2).all(|w| w[1].tilt >= w[0].tilt), "{:?}", p.iter().map(|p| p.tilt).collect::<Vec<_>>());
    assert!(s.tilt(1.3 * vs) > 1.5, "{}", s.tilt(1.3 * vs));
    assert!(s.max_speed() > 2.0 * vs, "{} vs {vs}", s.max_speed());
    let [kr, ka, kv, kp] = c.bandwidths();
    assert!(kr > 2.0 && ka < kr && kv < ka && kp < kv, "{:?}", c.bandwidths());
}

/// Largest stall margin angle of attack of the lifting surfaces (rad).
fn stall_alpha(t: &Tiltrotor) -> f64 {
    t.def()
        .surfaces
        .iter()
        .filter(|s| s.roll.cos().abs() > 0.5)
        .map(|s| s.alpha_stall - s.incidence)
        .fold(f64::INFINITY, f64::min)
}

/// Time (s) after which `x(t)` stays within `tol` of `target`, over a flight of `seconds`.
fn settle(
    t: &mut Tiltrotor,
    c: &mut TiltrotorController,
    seconds: f64,
    sp: TiltrotorSetpoint,
    target: f64,
    tol: f64,
    x: impl Fn(&Tiltrotor) -> f64,
) -> (f64, Flight) {
    let mut settled = 0.0;
    let f = fly(t, c, seconds, sp, |time, t| {
        if (x(t) - target).abs() > tol {
            settled = time;
        }
    });
    (settled, f)
}

/// Velocity steps in hover: forward, sideways and up each settle within a few velocity time
/// constants, with the others held.
#[test]
fn velocity_steps_in_hover() {
    let (mut t, mut c) = setup();
    let steps =
        [DVec3::new(3.0, 0.0, 0.0), DVec3::new(0.0, -3.0, 0.0), DVec3::new(0.0, 0.0, 2.0), DVec3::new(0.0, 0.0, -2.0)];
    for v in steps {
        start(&mut t, &mut c, 0.0);
        let sp = TiltrotorSetpoint::Velocity { velocity: v, yaw_rate: 0.0 };
        let mut settled = 0.0;
        let f = fly(&mut t, &mut c, 12.0, sp, |time, t| {
            if (velocity(t) - v).length() > 0.1 {
                settled = time;
            }
        });
        assert!(settled < 5.0 && (f.velocity - v).length() < 0.05, "{v}: settled {settled} s, {f:?}");
        assert!(euler(t.orientation()).2.abs() < 0.02, "{v}: yaw {}", euler(t.orientation()).2);
    }
}

/// Velocity steps on the wing at 20 m/s: speed up and down, climb and a coordinated turn.
#[test]
fn velocity_steps_in_cruise() {
    let (mut t, mut c) = setup();
    for (dv, climb) in [(4.0, 0.0), (-4.0, 0.0), (0.0, 1.5), (0.0, -1.5)] {
        start(&mut t, &mut c, 20.0);
        let v = DVec3::new(20.0 + dv, 0.0, climb);
        let sp = TiltrotorSetpoint::Velocity { velocity: v, yaw_rate: 0.0 };
        let mut settled = 0.0;
        let f = fly(&mut t, &mut c, 20.0, sp, |time, t| {
            if (velocity(t) - v).length() > 0.2 {
                settled = time;
            }
        });
        assert!(settled < 12.0 && (f.velocity - v).length() < 0.1, "{v}: settled {settled} s, {f:?}");
        assert!(f.alpha < stall_alpha(&t), "{v}: {f:?}");
    }
    // A coordinated turn at 0.15 rad/s: banked into the turn, little sideslip, speed held.
    start(&mut t, &mut c, 20.0);
    let sp = TiltrotorSetpoint::Velocity { velocity: DVec3::new(20.0, 0.0, 0.0), yaw_rate: 0.15 };
    let (mut beta, mut rate) = (0.0f64, 0.0);
    let f = fly(&mut t, &mut c, 20.0, sp, |time, t| {
        if time > 10.0 {
            beta = beta.max(t.flow().beta.abs());
            rate = (t.orientation() * t.ang_vel_body()).z;
        }
    });
    let roll = euler(t.orientation()).0;
    assert!((rate - 0.15).abs() < 0.01 && roll < -0.2 && beta < 0.03, "beta {beta} rate {rate} roll {roll} {f:?}");
    assert!((f.velocity.x - 20.0).abs() < 0.3 && f.altitude < 3.0, "{f:?}");
}

/// Scripted conversion to the wing and back in one flight: altitude held within bounds and
/// the wing kept below the stall; the mounts end forward, then up.
#[test]
fn transitions_both_ways() {
    let (mut t, mut c) = setup();
    start(&mut t, &mut c, 0.0);
    let limit = stall_alpha(&t);
    let cruise = (1.8 * c.schedule().stall_speed()).round();
    let out = fly(&mut t, &mut c, 25.0, level(cruise), |_, _| {});
    assert!((out.velocity.x - cruise).abs() < 0.1 && out.altitude < 3.0 && out.alpha < limit, "{out:?}");
    assert!(t.tilts().iter().all(|x| *x > 1.5), "{:?}", t.tilts());
    let back = fly(&mut t, &mut c, 30.0, level(0.0), |_, _| {});
    assert!(back.velocity.length() < 0.2 && back.altitude < 3.0 && back.alpha < limit, "{back:?}");
    assert!(t.tilts().iter().all(|x| x.abs() < 0.05), "{:?}", t.tilts());
}

/// Position hold: a 10 m step forward and up in hover settles with the heading held.
#[test]
fn position_step_in_hover() {
    let (mut t, mut c) = setup();
    start(&mut t, &mut c, 0.0);
    let goal = DVec3::new(10.0, 5.0, ALTITUDE + 5.0);
    let sp = TiltrotorSetpoint::Position { position: goal, yaw: 0.5 };
    let (settled, f) = settle(&mut t, &mut c, 30.0, sp, 0.0, 0.2, |t| (t.position() - goal).length());
    let yaw = euler(t.orientation()).2;
    assert!(settled < 25.0 && (yaw - 0.5).abs() < 0.02, "settled {settled}, yaw {yaw}, {f:?}");
}

/// Slow flight in the conversion band, where the wing does not carry the turns yet: sideways
/// and yaw commands keep the sideslip small (the nose weathervanes into the airflow), and in a
/// crosswind the aircraft turns and stops over the ground (a velocity trim held against the
/// wind does not cancel the command).
#[test]
fn slow_flight_in_the_conversion_band() {
    let (mut t, mut c) = setup();
    for (v0, vx, vy, r) in [
        (12.0, 5.0, -1.0, -0.2),
        (12.0, 5.0, -1.0, -0.5),
        (12.0, 5.0, 0.0, -0.5),
        (14.0, 8.0, -1.0, -0.3),
        (10.0, 5.0, -1.0, -0.5),
    ] {
        start(&mut t, &mut c, v0);
        let sp = TiltrotorSetpoint::Velocity { velocity: DVec3::new(vx, vy, 0.0), yaw_rate: r };
        let mut beta = 0.0f64;
        // Sideslip where the fin's weathercock moment counts (from ~0.7·V_s).
        let f = fly(&mut t, &mut c, 15.0, sp, |_, t| {
            if t.flow().airspeed > 8.0 {
                beta = beta.max(t.flow().beta.abs());
            }
        });
        // Sideways is flown up to ~5 m/s; faster, the nose follows the airflow instead.
        let (want, got) = (DVec3::new(vx, vy, 0.0), f.velocity);
        let side = if vx < 6.0 { (got.y - vy).abs() } else { 0.0 };
        assert!(
            beta < 0.45 && f.altitude < 2.0 && (got.x - vx).abs() < 0.6 && side < 0.2,
            "{v0} {want} {r}: β {beta}, {f:?}"
        );
    }
    // Crabbing across a crosswind in the hover regime, ~11 m/s through the air.
    let wind = DVec3::new(-2.5, 6.7, 0.0);
    let crab = |vx, vy, r| TiltrotorSetpoint::Velocity { velocity: DVec3::new(vx, vy, 0.0), yaw_rate: r };
    for turn in [false, true] {
        start(&mut t, &mut c, 0.0);
        fly_in(wind, &mut t, &mut c, 25.0, crab(8.5, 6.7, 0.0), |_, _| {});
        let yaw0 = euler(t.orientation()).2;
        let mut yaw = 0.0;
        let sp = if turn { crab(8.0, -4.0, -0.5) } else { crab(0.0, 0.0, 0.0) };
        let f = fly_in(wind, &mut t, &mut c, 15.0, sp, |_, t| {
            // Unwrapped heading change.
            let d = (euler(t.orientation()).2 - yaw0 - yaw + PI).rem_euclid(TAU) - PI;
            yaw += d;
        });
        if turn {
            assert!(yaw < -3.0, "turned {yaw}, {f:?}");
        } else {
            assert!(t.lin_vel_world().length() < 1.0 && f.altitude < 3.0, "{f:?}");
        }
    }
}

/// Attitude mode: a bank and a pitch held in hover, and a climb rate; then the airspeed
/// reference carries the aircraft onto the wing.
#[test]
fn attitude_mode() {
    let (mut t, mut c) = setup();
    start(&mut t, &mut c, 0.0);
    let sp = TiltrotorSetpoint::Attitude { roll: 0.2, pitch: 0.1, yaw_rate: 0.0, climb: 1.0, airspeed: 0.0 };
    let f = fly(&mut t, &mut c, 3.0, sp, |_, _| {});
    let (r, p, _) = euler(t.orientation());
    assert!((r - 0.2).abs() < 0.01 && (p - 0.1).abs() < 0.01 && (f.velocity.z - 1.0).abs() < 0.1, "{r} {p} {f:?}");
    start(&mut t, &mut c, 0.0);
    let sp = TiltrotorSetpoint::Attitude { roll: 0.0, pitch: 0.0, yaw_rate: 0.0, climb: 0.0, airspeed: 20.0 };
    let f = fly(&mut t, &mut c, 30.0, sp, |_, _| {});
    assert!((t.flow().airspeed - 20.0).abs() < 0.5 && f.altitude < 5.0 && f.alpha < stall_alpha(&t), "{f:?}");
}
