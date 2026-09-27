//! The two-wheeler presets: statics, straight-line performance, steady turning and the
//! linearised modes of the full models.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::{DenseMatrix, Pose};
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::single_track::WhippleParams;
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::presets;
use glam::{DQuat, DVec3};
use std::sync::Arc;

const DT: f64 = 1e-3;
const G: f64 = STANDARD_GRAVITY;

fn preset(name: &str) -> WheeledDef {
    presets::wheeled(name).unwrap()
}

/// Lean of the chassis (rad, positive to the right): the roll of its yaw–pitch–roll angles.
fn lean(v: &Wheeled) -> f64 {
    v.orientation().to_euler(glam::EulerRot::ZYX).2
}

/// Yaw rate about the world vertical (rad/s, positive to the left).
fn yaw_rate(v: &Wheeled) -> f64 {
    (v.orientation() * v.ang_vel_body()).z
}

/// Ride on flat asphalt for `seconds`: `control` gives the input before every step, `each`
/// sees the state after it.
fn ride(
    v: &mut Wheeled,
    seconds: f64,
    mut control: impl FnMut(&Wheeled) -> DriveInput,
    mut each: impl FnMut(&Wheeled),
) {
    let terrain = FlatTerrain::new(0.0, MaterialId::ASPHALT);
    let materials = MaterialTable::standard();
    let env = GroundStepEnv {
        scene: StaticScene { terrain: &terrain, obstacles: &NoObstacles, materials: &materials },
        gravity: DVec3::new(0.0, 0.0, -G),
        air: AirData::default(),
    };
    for _ in 0..(seconds / v.dt()).round() as usize {
        let input = control(v);
        v.step(&input, &env).unwrap();
        each(v);
    }
}

/// Started upright on its static loads at `speed`.
fn start(def: &WheeledDef, speed: f64) -> Wheeled {
    let mut v = Wheeled::new(Arc::new(def.clone()), DT);
    let init = v.rest(DVec3::ZERO, 0.0, speed);
    v.reset(&init);
    v
}

/// A rider's balance: steering torque into the fall and towards the lean `target`, as a
/// fraction of the steering's largest torque, with integral action for the steady torque of a
/// turn.
struct Balance {
    target: f64,
    kp: f64,
    kd: f64,
    ki: f64,
    max_torque: f64,
    integral: f64,
}

impl Balance {
    fn new(def: &WheeledDef, kp: f64, kd: f64, ki: f64) -> Self {
        let max_torque = def.steering_head().unwrap().0.max_torque;
        Self { target: 0.0, kp, kd, ki, max_torque, integral: 0.0 }
    }

    /// Steering input, positive turning left.
    fn steering(&mut self, v: &Wheeled) -> f64 {
        let e = lean(v) - self.target;
        self.integral += e * v.dt();
        let right = self.kp * e + self.kd * v.ang_vel_body().x + self.ki * self.integral;
        (-right / self.max_torque).clamp(-1.0, 1.0)
    }
}

// ------------------------------------------------------------------------------------ statics

#[test]
fn presets_rest_on_their_static_loads() {
    for (name, mass, rear_share) in [("bicycle_city", 93.0, 0.66), ("motorcycle_sport", 277.0, 0.516)] {
        let d = preset(name);
        let st = d.rest_state().expect("statics").clone();
        println!("{name}: {:.1} kg, com {:.3}, loads {:?}", d.total_mass(), d.total_com(), st.loads);
        assert!((d.total_mass() - mass).abs() < 0.5, "{name}: mass {}", d.total_mass());
        let share = st.loads[0] / (st.loads[0] + st.loads[1]);
        assert!((share - rear_share).abs() < 0.01, "{name}: rear share {share}");
        // Upright and still on those loads after two seconds.
        let mut v = start(&d, 0.0);
        ride(&mut v, 2.0, |_| DriveInput::default(), |_| {});
        assert!(lean(&v).abs() < 1e-9, "{name}: lean {}", lean(&v));
        assert!(v.lin_vel_world().length() < 1e-3, "{name}: moving {}", v.lin_vel_world());
        for (w, s) in v.wheels().enumerate() {
            assert!(
                (s.tire.fz - st.loads[w]).abs() < 2e-3 * st.loads[w],
                "{name} wheel {w}: {} vs {}",
                s.tire.fz,
                st.loads[w]
            );
            assert!(s.travel.abs() < 1e-3, "{name} wheel {w}: travel {}", s.travel);
        }
    }
}

// -------------------------------------------------------------------------- straight line

/// Full throttle from standstill, the rider holding the front wheel down (throttle on its
/// load, proportional–integral) and balancing: the time to 100 km/h and the top speed.
fn motorcycle_run(d: &WheeledDef) -> (f64, f64, f64) {
    let front_static = d.rest_state().unwrap().loads[1];
    let mut v = start(d, 0.0);
    let mut bal = Balance::new(d, 100.0, 10.0, 0.0);
    let mut held = 1.0f64;
    let (mut t, mut t100, mut worst_lean) = (0.0, f64::NAN, 0.0f64);
    ride(
        &mut v,
        60.0,
        |v| {
            let e = (v.wheel(1).tire.fz - 0.12 * front_static) / front_static;
            held = (held + 10.0 * e * v.dt()).clamp(0.0, 1.0);
            DriveInput { throttle: (held + 3.0 * e).clamp(0.0, 1.0), steering: bal.steering(v), ..Default::default() }
        },
        |v| {
            t += v.dt();
            if t100.is_nan() && v.speed() >= 100.0 / 3.6 {
                t100 = t;
            }
            worst_lean = worst_lean.max(lean(v).abs());
        },
    );
    (t100, 3.6 * v.speed(), worst_lean)
}

#[test]
fn motorcycle_acceleration_and_top_speed() {
    // Published for the GSX-R1000 K1–K4: 0–100 km/h in about 3 s (wheelie-limited),
    // about 285–295 km/h. The drive torque reacts on the swing arm, which jacks the tail up
    // (anti-squat about 120 %), so the front lifts at about 0.9 g.
    let (t100, top, worst_lean) = motorcycle_run(&preset("motorcycle_sport"));
    println!("0-100 km/h in {t100:.2} s, top speed {top:.1} km/h");
    assert!((2.8..3.8).contains(&t100), "0-100 km/h in {t100} s");
    assert!((270.0..300.0).contains(&top), "top speed {top} km/h");
    assert!(worst_lean < 1e-6, "lean {worst_lean}");
}

/// Stopping from `speed` with the brake levels `[rear, front]` squeezed over 0.5 s, each
/// released as its wheel starts to lock: the stopping time, distance, lowest rear-wheel load
/// after the squeeze and largest pitch.
fn stop(d: &WheeledDef, speed: f64, level: [f64; 2]) -> (f64, f64, f64, f64) {
    let mut v = start(d, speed);
    let mut bal = Balance::new(d, 100.0, 10.0, 0.0);
    let (mut k, mut j) = (0usize, 0usize);
    let (mut dist, mut stopped, mut min_rear, mut max_pitch) = (0.0, f64::NAN, f64::INFINITY, 0.0f64);
    ride(
        &mut v,
        8.0,
        |v| {
            k += 1;
            let ramp = (k as f64 * v.dt() / 0.5).min(1.0);
            let mut wheel_brake = [0.0; MAX_WHEELS];
            for w in 0..2 {
                let lock = -v.wheel(w).tire.kappa;
                wheel_brake[w] = ramp * level[w] * (1.0 - (lock - 0.08) / 0.05).clamp(0.0, 1.0);
            }
            DriveInput { wheel_brake, steering: bal.steering(v), ..Default::default() }
        },
        |v| {
            j += 1;
            let t = j as f64 * v.dt();
            if stopped.is_nan() {
                dist += v.speed() * v.dt();
                if v.lin_vel_body().x < 0.05 {
                    stopped = t;
                }
                if t > 0.5 {
                    min_rear = min_rear.min(v.wheel(0).tire.fz);
                }
            }
            max_pitch = max_pitch.max(v.orientation().to_euler(glam::EulerRot::ZYX).1);
        },
    );
    (stopped, dist, min_rear, max_pitch)
}

#[test]
fn braking() {
    // Motorcycle from 100 km/h: sport bikes stop in about 36–45 m (1.0 g, the rear wheel
    // skimming the road, as here). The fork runs onto its bump stop near 1 g, so the rider squeezes to 0.6 of the
    // front brake.
    let d = preset("motorcycle_sport");
    let (t, dist, min_rear, pitch) = stop(&d, 100.0 / 3.6, [0.2, 0.6]);
    println!("motorcycle: stops in {t:.2} s over {dist:.1} m, lowest rear load {min_rear:.0} N, pitch {pitch:.3}");
    assert!((36.0..46.0).contains(&dist), "stopping distance {dist} m");
    assert!(pitch < 0.15, "pitching over: {pitch}");
    // Bicycle from 25 km/h on rim brakes: the front brake's 180 N·m limit (0.57 g) plus the
    // unloaded rear, below the 0.75 g at which it would pitch over; the 0.5 s squeeze takes a
    // good part of the 1.3 s stop.
    let d = preset("bicycle_city");
    let speed = 25.0 / 3.6;
    let (t, dist, min_rear, pitch) = stop(&d, speed, [0.5, 1.0]);
    let decel = speed * speed / (2.0 * dist) / G;
    println!("bicycle: stops in {t:.2} s over {dist:.1} m ({decel:.2} g), lowest rear load {min_rear:.0} N");
    assert!((0.45..0.7).contains(&decel), "deceleration {decel} g");
    assert!(min_rear > 0.0 && pitch < 0.1);
}

#[test]
fn bicycle_top_speed_by_power() {
    // A cyclist's 250 W gives about 8 m/s on a city bicycle, 600 W about 11 m/s.
    for (power, range) in [(250.0, 7.3..8.3), (600.0, 10.6..11.8)] {
        let mut d = preset("bicycle_city");
        let PowertrainDef::Electric(e) = &mut d.powertrain else { panic!("electric drive") };
        e.motors[0].max_power = power;
        d.finish().unwrap();
        let mut v = start(&d, 0.0);
        let mut bal = Balance::new(&d, 5.0, 1.0, 0.0);
        ride(&mut v, 60.0, |v| DriveInput { throttle: 1.0, steering: bal.steering(v), ..Default::default() }, |_| {});
        println!("{power} W: {:.2} m/s", v.speed());
        assert!(range.contains(&v.speed()), "{power} W: {} m/s", v.speed());
    }
}

// ----------------------------------------------------------------------------- steady turns

#[test]
fn steady_turning_roll_angle() {
    // The rider holds a lean (steering torque, proportional–integral–derivative) at constant
    // speed; the roll angle against the lateral acceleration a, from the moment balance about
    // the contact line: tan θ = a/g (1 + Σ I_spin/(r m h)) with the wheels' gyroscopic moment,
    // and φ = θ + asin(ρ sin θ/(h − ρ)) as the contact points move around the tyre crowns of
    // (load-weighted) radius ρ. The bicycle's larger steer angles in its tighter turns take it
    // out of this small-steer formula beyond about 0.3 rad (1.1° off at 0.4 rad, 0.45 g).
    for (name, speed, leans, gains) in [
        ("motorcycle_sport", 20.0, &[0.2, 0.4, 0.6, 0.8][..], (100.0, 10.0, 100.0)),
        ("bicycle_city", 8.0, &[0.1, 0.2, 0.3][..], (5.0, 1.0, 5.0)),
    ] {
        let d = preset(name);
        let (m, h) = (d.total_mass(), d.total_com().z);
        let loads = d.rest_state().unwrap().loads.clone();
        let rho = (d.tire(0).crown_radius * loads[0] + d.tire(1).crown_radius * loads[1]) / (loads[0] + loads[1]);
        let spin: f64 = (0..2).map(|a| d.axles[a].wheel.inertia.y / d.tire(a).radius()).sum();
        for &target in leans {
            let mut v = start(&d, speed);
            let mut bal = Balance::new(&d, gains.0, gains.1, gains.2);
            let mut hold = SpeedHold::new(speed);
            let (mut phi, mut a, mut n, mut k, mut j) = (0.0, 0.0, 0.0, 0usize, 0usize);
            ride(
                &mut v,
                12.0,
                |v| {
                    k += 1;
                    bal.target = target * (k as f64 * v.dt() / 3.0).min(1.0);
                    DriveInput { throttle: hold.throttle(v), steering: bal.steering(v), ..Default::default() }
                },
                |v| {
                    j += 1;
                    if j as f64 * v.dt() > 9.0 {
                        phi += lean(v);
                        a -= v.lin_vel_world().length() * yaw_rate(v);
                        n += 1.0;
                    }
                },
            );
            let (phi, a) = (phi / n, a / n);
            let theta = (a / G * (1.0 + spin / (m * h))).atan();
            let want = theta + (rho * theta.sin() / (h - rho)).asin();
            let err = (phi - want).to_degrees();
            println!("{name}: lean {phi:.4} rad at {:.3} g, analytic {want:.4} rad: {err:+.2}°", a / G);
            assert!((v.speed() - speed).abs() < 0.1, "{name}: speed {}", v.speed());
            assert!((phi - target).abs() < 0.03, "{name}: lean {phi} for {target}");
            assert!(err.abs() < 1.0, "{name}: roll angle {phi} rad at {a} m/s² vs {want}");
        }
    }
}

// ------------------------------------------------------------------------------------- modes

/// The lateral states: chassis lean, steer angle, their rates, yaw rate, lateral velocity,
/// the rider's lean and rate and both tyres' lateral deflections, scaled to similar sizes.
fn lateral(v: &Wheeled) -> [f64; 10] {
    let (delta, rate) = v.steering_head().unwrap();
    let (rider, rider_rate) = v.rider_lean();
    [
        lean(v),
        delta,
        v.ang_vel_body().x,
        rate,
        v.ang_vel_body().z,
        v.lin_vel_body().y,
        rider,
        rider_rate,
        100.0 * v.tire_state(0).v,
        100.0 * v.tire_state(1).v,
    ]
}

/// Eigenvalues `(re, im)` of the full model's lateral dynamics riding straight at `speed`
/// (held by the throttle), by dynamic mode decomposition of eight runs perturbed in each
/// mechanical state (the tyres' deflections follow), sampled as 5 ms averages over 1 s after
/// 20 ms of transients.
fn identify_lateral(def: &WheeledDef, speed: f64) -> Vec<(f64, f64)> {
    const N: usize = 10;
    let (sample, skip, length, eps) = (0.005, 0.02, 1.0, 1e-5);
    let every = (sample / DT).round() as usize;
    let mut xx = DenseMatrix::zeros(N, N);
    let mut yx = DenseMatrix::zeros(N, N);
    for k in 0..8 {
        let mut v = Wheeled::new(Arc::new(def.clone()), DT);
        let mut init = v.rest(DVec3::ZERO, 0.0, speed);
        match k {
            0 => {
                let rot = DQuat::from_rotation_x(eps);
                init.pose = Pose::new(rot * init.pose.pos, rot * init.pose.rot);
            }
            2 => init.ang_vel_body.x = eps,
            4 => init.ang_vel_body.z = eps,
            5 => init.lin_vel_world.y = eps,
            _ => {}
        }
        v.reset(&init);
        let names: Vec<&str> = v.model().links().iter().map(|l| l.name.as_str()).collect();
        let head = names.iter().position(|n| *n == "head_1").unwrap();
        let rider = names.iter().position(|n| *n == "rider").unwrap();
        let (hq, hv) = (v.model().q_offset(head), v.model().v_offset(head));
        let (rq, rv) = (v.model().q_offset(rider), v.model().v_offset(rider));
        match k {
            1 => v.state.q[hq] = eps,
            3 => v.state.v[hv] = 10.0 * eps,
            6 => v.state.q[rq] = eps,
            7 => v.state.v[rv] = 10.0 * eps,
            _ => {}
        }
        let mut hold = SpeedHold::new(speed);
        let mut prev: Option<[f64; N]> = None;
        let mut mean = [0.0; N];
        let mut i = 0usize;
        ride(
            &mut v,
            skip + length,
            |v| DriveInput { throttle: hold.throttle(v), ..Default::default() },
            |v| {
                i += 1;
                for (m, x) in mean.iter_mut().zip(lateral(v)) {
                    *m += x / (eps * every as f64);
                }
                if !i.is_multiple_of(every) {
                    return;
                }
                let x = std::mem::take(&mut mean);
                if (i as f64) * DT < skip {
                    return;
                }
                if let Some(p) = prev {
                    for r in 0..N {
                        for c in 0..N {
                            xx[(r, c)] += p[r] * p[c];
                            yx[(r, c)] += x[r] * p[c];
                        }
                    }
                }
                prev = Some(x);
            },
        );
    }
    let f = yx.mul(&xx.inverse().expect("rich enough data"));
    let mut out: Vec<(f64, f64)> = f
        .eigenvalues()
        .expect("QR converges")
        .into_iter()
        .map(|(re, im)| {
            let r = (re * re + im * im).sqrt();
            (r.ln() / sample, im.atan2(re) / sample)
        })
        .collect();
    out.sort_by(|a, b| a.1.abs().total_cmp(&b.1.abs()).then(a.0.total_cmp(&b.0)));
    out
}

/// Holds the speed with the throttle (proportional–integral).
struct SpeedHold {
    target: f64,
    integral: f64,
}

impl SpeedHold {
    fn new(target: f64) -> Self {
        Self { target, integral: 0.0 }
    }

    fn throttle(&mut self, v: &Wheeled) -> f64 {
        let e = self.target - v.lin_vel_body().x;
        self.integral = (self.integral + e * v.dt()).clamp(-5.0, 5.0);
        (0.5 * e + 0.5 * self.integral).clamp(0.0, 1.0)
    }
}

/// The least damped mode with a frequency in `band` (Hz): `(growth rate 1/s, frequency Hz)`.
fn mode(e: &[(f64, f64)], band: std::ops::Range<f64>) -> (f64, f64) {
    e.iter()
        .map(|&(re, im)| (re, im.abs() / std::f64::consts::TAU))
        .filter(|(_, f)| band.contains(f))
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap_or_else(|| panic!("no mode in {band:?} Hz: {e:?}"))
}

#[test]
fn motorcycle_weave_and_wobble() {
    // Straight running, lateral modes of the full model. Sharp, Evangelou & Limebeer (2004)
    // for this machine (Evangelou 2004, fig. 8.1): weave rising to about 3.5 Hz at 40 m/s and
    // lightly damped at high speed; wobble about 7.6 Hz, least damped near 13 m/s. Their model
    // twists at the steering head (frame compliance, which lowers the wobble frequency and
    // damps it at speed); this one has a rigid frame, whose wobble (as in Sharp 1971) sits
    // near 10 Hz and loses damping with speed, the steering damper keeping it stable.
    let d = preset("motorcycle_sport");
    let speeds = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let modes: Vec<_> = speeds
        .iter()
        .map(|&s| {
            let e = identify_lateral(&d, s);
            let (weave, wobble) = (mode(&e, 0.3..5.0), mode(&e, 7.0..13.0));
            println!(
                "{s:4.0} m/s: weave {:6.2}/s {:.2} Hz, wobble {:6.2}/s {:.2} Hz",
                weave.0, weave.1, wobble.0, wobble.1
            );
            (weave, wobble)
        })
        .collect();
    for (k, &(weave, wobble)) in modes.iter().enumerate() {
        assert!(weave.0 < 0.0 && wobble.0 < 0.0, "unstable at {} m/s", speeds[k]);
        assert!((8.0..12.0).contains(&wobble.1), "wobble at {} Hz", wobble.1);
        if k > 0 {
            // From 20 m/s on, the weave's frequency rises and its damping falls.
            let (prev_weave, prev_wobble) = modes[k - 1];
            if speeds[k] > 20.0 {
                assert!(weave.1 > prev_weave.1 && weave.0 > prev_weave.0, "weave at {} m/s", speeds[k]);
            }
            assert!(wobble.0 > prev_wobble.0, "wobble damping at {} m/s", speeds[k]);
        }
    }
    let (weave40, _) = modes[3];
    assert!((2.8..4.0).contains(&weave40.1), "weave at 40 m/s: {} Hz", weave40.1);
    // Without its steering damper, the wobble goes unstable at speed.
    let mut free = d.clone();
    free.axles[1].steering_head.as_mut().unwrap().damping = 0.0;
    free.finish().unwrap();
    let e = identify_lateral(&free, 50.0);
    let wobble = mode(&e, 7.0..13.0);
    println!("no damper, 50 m/s: wobble {:.2}/s {:.2} Hz", wobble.0, wobble.1);
    assert!(wobble.0 > 0.0, "wobble without the damper {wobble:?}");
}

#[test]
fn bicycle_self_stable_range() {
    // The full model (tyres with slip, crowns and relaxation, a leaning rider) has a
    // self-stable range starting near the Whipple model's for the same bicycle (5.00–7.53 m/s):
    // weave stable above about 4.5 m/s; the capsize mode, with the tyres' slip and camber
    // forces and the rider's sway, turns (slowly) unstable earlier, at about 5.5 m/s.
    let d = preset("bicycle_city");
    let (weave, capsize) = WhippleParams::from_def(&d).unwrap().matrices().stable_speeds(G, 20.0).unwrap();
    println!("Whipple: stable from {weave:.2} to {capsize:.2} m/s");
    // Largest growth rate of the slow modes: weave (below 3 Hz) and capsize (real).
    let growth = |s: f64| {
        let e = identify_lateral(&d, s);
        let capsize = e.iter().filter(|x| x.1 == 0.0 && x.0.abs() < 5.0).map(|x| x.0).fold(f64::MIN, f64::max);
        let g = mode(&e, 0.0..3.0).0.max(capsize);
        println!("{s} m/s: {g:.3}/s");
        g
    };
    let (mut lo, mut hi) = (3.0, 5.0);
    assert!(growth(lo) > 0.0 && growth(hi) < 0.0);
    for _ in 0..8 {
        let mid = 0.5 * (lo + hi);
        if growth(mid) > 0.0 { lo = mid } else { hi = mid }
    }
    let full = 0.5 * (lo + hi);
    println!("full model: weave stable from {full:.2} m/s");
    assert!((full / weave - 1.0).abs() < 0.15, "weave speed {full} vs Whipple {weave}");
    // Capsize: slow, well below the rider's bandwidth.
    let g = growth(8.0);
    assert!(g > 0.0 && g < 0.5, "capsize at 8 m/s: {g}/s");
}
