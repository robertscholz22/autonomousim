//! Closed-loop checks of the fixed-wing controller on both presets: rate and attitude steps
//! across the speed range, coordinated turns, altitude and airspeed hold in turbulence, and L1
//! path following.

use autonomousim_control::fixedwing::*;
use autonomousim_core::math::Pose;
use autonomousim_core::rng::Seed;
use autonomousim_vehicles::aero::AirData;
use autonomousim_vehicles::fixedwing::{FixedWing, FixedWingInit};
use autonomousim_vehicles::multirotor::StepEnv;
use autonomousim_vehicles::presets;
use autonomousim_world::environment::wind::{Dryden, DrydenScales};
use glam::{DVec2, DVec3};
use std::sync::Arc;

const G: f64 = 9.80665;
const DT: f64 = 0.002;
const RHO: f64 = 1.225;
const PRESETS: [&str; 2] = ["aerosonde_like", "c172_like"];

fn setup(name: &str) -> (FixedWing, FixedWingController) {
    let def = Arc::new(presets::fixed_wing(name).unwrap());
    let c = FixedWingController::new(&def, DT, &FixedWingConfig::default()).unwrap();
    (FixedWing::new(def, DT), c)
}

/// Trimmed level flight at `v` heading east at 500 m.
fn start(a: &mut FixedWing, c: &mut FixedWingController, v: f64) {
    let t = a.trim(v, RHO, 0.0, 0.0, G).unwrap();
    let rot = t.attitude(0.0);
    a.reset(&FixedWingInit {
        pose: Pose::new(DVec3::new(0.0, 0.0, 500.0), rot),
        lin_vel_world: rot * t.velocity_body(),
        ang_vel_body: DVec3::ZERO,
        controls: t.controls,
        rotor_speed: Some(t.rotor_speed),
        soc: 1.0,
    });
    c.reset();
}

/// Fly `seconds` tracking `sp(t)` in wind `wind(t)`, calling `each(t, aircraft)` every step.
fn fly(
    a: &mut FixedWing,
    c: &mut FixedWingController,
    seconds: f64,
    mut sp: impl FnMut(f64, &FixedWing) -> FixedWingSetpoint,
    mut wind: impl FnMut(f64, &FixedWing) -> DVec3,
    mut each: impl FnMut(f64, &FixedWing, &FixedWingController),
) {
    for i in 0..(seconds / DT).round() as usize {
        let t = (i + 1) as f64 * DT;
        let input = c.update(&sp(t, a), a);
        let env = StepEnv {
            scene: None,
            gravity: DVec3::new(0.0, 0.0, -G),
            air: AirData { density: RHO, wind: wind(t, a), ..AirData::default() },
            ground: None,
        };
        a.step(&input, &env).unwrap();
        each(t, a, c);
    }
}

fn calm(_: f64, _: &FixedWing) -> DVec3 {
    DVec3::ZERO
}

/// Speeds across the envelope: both ends of the normal range and the design speed.
fn speeds(c: &FixedWingController) -> [f64; 3] {
    let m = c.model();
    let [lo, hi] = m.normal_speeds();
    [lo, m.design_airspeed(), hi]
}

/// 10–90 % rise time (s) and overshoot (fraction) of a step to `target` in `samples`.
fn step_response(samples: &[(f64, f64)], target: f64) -> (f64, f64) {
    let t10 = samples.iter().find(|s| s.1 / target >= 0.1).map_or(f64::INFINITY, |s| s.0);
    let t90 = samples.iter().find(|s| s.1 / target >= 0.9).map_or(f64::INFINITY, |s| s.0);
    let peak = samples.iter().map(|s| s.1 / target).fold(f64::MIN, f64::max);
    (t90 - t10, (peak - 1.0).max(0.0))
}

#[test]
fn rate_steps() {
    for name in PRESETS {
        let (mut a, mut c) = setup(name);
        for v in speeds(&c) {
            // Roll 0.5 rad/s; pitch the rate of a 0.25 g pull-up.
            for (axis, step) in [(0, 0.5), (1, 0.25 * G / v)] {
                start(&mut a, &mut c, v);
                let throttle = a.input().throttle;
                let mut rates = DVec3::ZERO;
                rates[axis] = step;
                let mut samples = Vec::new();
                fly(
                    &mut a,
                    &mut c,
                    2.0,
                    |_, _| FixedWingSetpoint::Rates { rates, throttle },
                    calm,
                    |t, a, c| {
                        // Against the setpoint with its coordination part, which is what the
                        // loop tracks.
                        let sp = c.rate_setpoint()[axis];
                        samples.push((t, to_pilot(a.ang_vel_body())[axis] * step / sp));
                    },
                );
                let (rise, overshoot) = step_response(&samples, step);
                eprintln!("{name} {v:.1} m/s axis {axis}: rise {rise:.3} s overshoot {overshoot:.3}");
                // First order at the loop bandwidth rises in 2.2/ω_b; allow the servo lag.
                let bound = 1.5 * 2.2 / c.rate_gains()[axis];
                assert!(rise < bound && overshoot < 0.2, "{name} {v} axis {axis}: rise {rise} overshoot {overshoot}");
            }
        }
    }
}

#[test]
fn attitude_steps() {
    for name in PRESETS {
        let (mut a, mut c) = setup(name);
        for v in speeds(&c) {
            // Bank 30° at constant pitch.
            start(&mut a, &mut c, v);
            let pitch0 = euler(a.orientation()).1;
            let target = 30f64.to_radians();
            let mut samples = Vec::new();
            fly(
                &mut a,
                &mut c,
                4.0,
                |_, _| FixedWingSetpoint::Attitude { roll: target, pitch: pitch0, airspeed: v },
                calm,
                |t, a, _| samples.push((t, euler(a.orientation()).0)),
            );
            let (rise, overshoot) = step_response(&samples, target);
            let last = samples.last().unwrap().1;
            eprintln!("{name} {v:.1} m/s bank: rise {rise:.3} s overshoot {overshoot:.3} final {last:.4}");
            let k = c.rate_gains() / c.config().tuning.attitude_ratio;
            assert!(rise < 1.5 * 2.2 / k.x && overshoot < 0.1 && (last - target).abs() < 0.02, "{name} {v}: bank");
            // Pitch 5° up from trim, wings level.
            start(&mut a, &mut c, v);
            let target = 5f64.to_radians();
            let mut samples = Vec::new();
            fly(
                &mut a,
                &mut c,
                4.0,
                |_, _| FixedWingSetpoint::Attitude { roll: 0.0, pitch: pitch0 + target, airspeed: v },
                calm,
                |t, a, _| samples.push((t, euler(a.orientation()).1 - pitch0)),
            );
            let (rise, overshoot) = step_response(&samples, target);
            let last = samples.last().unwrap().1;
            eprintln!("{name} {v:.1} m/s pitch: rise {rise:.3} s overshoot {overshoot:.3} final {last:.4}");
            assert!(rise < 1.5 * 2.2 / k.y && overshoot < 0.2 && (last - target).abs() < 0.01, "{name} {v}: pitch");
        }
    }
}

/// A 30° banked turn holds altitude and airspeed with the sideslip near zero.
#[test]
fn coordinated_turn() {
    for name in PRESETS {
        let (mut a, mut c) = setup(name);
        let v = c.model().design_airspeed();
        start(&mut a, &mut c, v);
        let rate = G * 30f64.to_radians().tan() / v;
        let (mut beta_max, mut dh_max, mut dv_max, mut roll_sum, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0, 0);
        fly(
            &mut a,
            &mut c,
            30.0,
            |_, _| FixedWingSetpoint::Guidance {
                lateral: Lateral::CourseRate(-rate),
                vertical: Vertical::Altitude(500.0),
                airspeed: v,
            },
            calm,
            |t, a, _| {
                if t > 5.0 {
                    beta_max = beta_max.max(a.flow().beta.abs());
                    dh_max = dh_max.max((a.position().z - 500.0).abs());
                    dv_max = dv_max.max((a.flow().airspeed - v).abs());
                    roll_sum += euler(a.orientation()).0;
                    n += 1;
                }
            },
        );
        let roll = roll_sum / n as f64;
        eprintln!("{name}: beta {beta_max:.4} dh {dh_max:.2} dv {dv_max:.2} roll {roll:.3}");
        assert!(beta_max < 2f64.to_radians(), "{name}: sideslip {beta_max}");
        assert!(dh_max < 5.0 && dv_max < 1.0, "{name}: dh {dh_max} dv {dv_max}");
        assert!((roll - 30f64.to_radians()).abs() < 0.05, "{name}: bank {roll}");
    }
}

/// Moderate turbulence (W20 = 15 m/s) over a 6 m/s crosswind: altitude, airspeed and course
/// hold. The C172 stays within 20 m (its worst over seeds 1–4 and 7 is about 16 m: the vertical gusts
/// of W20 = 15 move a slow, lightly loaded wing a lot), the Aerosonde within 10 m.
#[test]
fn hold_in_turbulence() {
    for name in PRESETS {
        let (mut a, mut c) = setup(name);
        let v = c.model().design_airspeed();
        start(&mut a, &mut c, v);
        let mut rng = Seed::from_u64(7).child("turbulence").rng();
        let mut dryden = Dryden::stationary(&mut rng);
        let scales = DrydenScales::at(500.0, 15.0);
        let mean = DVec3::new(0.0, 6.0, 0.0);
        let (mut sh, mut sv, mut sc, mut n) = (0.0, 0.0, 0.0, 0);
        let mut dh_max = 0.0f64;
        fly(
            &mut a,
            &mut c,
            90.0,
            |_, _| FixedWingSetpoint::Guidance {
                lateral: Lateral::Course(0.0),
                vertical: Vertical::Altitude(500.0),
                airspeed: v,
            },
            |_, a| {
                dryden.step(DT, a.flow().airspeed, &scales, &mut rng);
                mean + dryden.velocity(&scales, DVec2::Y)
            },
            |t, a, _| {
                if t > 10.0 {
                    let dh = a.position().z - 500.0;
                    let vg = a.lin_vel_world();
                    sh += dh * dh;
                    sv += (a.flow().airspeed - v).powi(2);
                    sc += vg.y.atan2(vg.x).powi(2);
                    dh_max = dh_max.max(dh.abs());
                    n += 1;
                }
            },
        );
        let rms = |s: f64| (s / n as f64).sqrt();
        let (h, sp, course) = (rms(sh), rms(sv), rms(sc));
        eprintln!("{name}: rms altitude {h:.2} m airspeed {sp:.2} m/s course {course:.3} rad; max dh {dh_max:.2}");
        let bound = if name == "c172_like" { 20.0 } else { 10.0 };
        assert!(h < 0.35 * bound && dh_max < bound, "{name}: altitude");
        assert!(sp < 1.5, "{name}: airspeed");
        assert!(course < 0.1, "{name}: course");
    }
}

/// L1 captures a straight line from 100 m off and 40° off course in a crosswind, and a circle
/// of three minimum turn radii.
#[test]
fn l1_paths() {
    for name in PRESETS {
        let (mut a, mut c) = setup(name);
        let v = c.model().design_airspeed();
        let wind = |_: f64, _: &FixedWing| DVec3::new(-2.0, 4.0, 0.0);
        let line = Path::Line { start: [0.0, 100.0], end: [1000.0, 100.0 + 1000.0 * 0.7f64.tan()] };
        let circle_radius = 3.0 * v * v / (G * 45f64.to_radians().tan());
        let circle = Path::Circle { center: [200.0, 0.0], radius: circle_radius, clockwise: false };
        for (path, settle, seconds) in [(line, 50.0, 90.0), (circle, 60.0, 120.0)] {
            start(&mut a, &mut c, v);
            let mut worst = 0.0f64;
            fly(
                &mut a,
                &mut c,
                seconds,
                |_, _| FixedWingSetpoint::Guidance {
                    lateral: Lateral::Path(path),
                    vertical: Vertical::Altitude(500.0),
                    airspeed: v,
                },
                wind,
                |t, a, _| {
                    if t > settle {
                        worst = worst.max(path.cross_track(a.position().truncate()).abs());
                    }
                },
            );
            eprintln!("{name} {path:?}: worst cross-track {worst:.2} m");
            assert!(worst < 5.0, "{name} {path:?}: {worst}");
        }
    }
}
