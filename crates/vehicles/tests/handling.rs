//! Handling validation of the Sedan (M2 step 7) against Chrono::Vehicle
//! (`fixtures/chrono/handling_sedan.json`, from `tools/gen_chrono_handling_fixtures.py`) and
//! against the linear bicycle model:
//! * ISO 4138 steady-state circular driving (constant steering-wheel angle, rising speed): the
//!   understeer gradient;
//! * ISO 7401 step steer at 80 km/h: the yaw-rate response;
//! * straight braking from 100 km/h;
//! * ISO 3888-1 double lane change at 80 km/h with a pure-pursuit driver.
//!
//! Both simulators are driven by the same speed controller and driver (re-implemented here
//! from the generator). The Sedan's steering maps its input to road-wheel angles through a
//! linkage in Chrono and linearly here, so inputs are matched by the road-wheel angle.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::tire::{MfInput, TireModel};
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::presets;
use glam::DVec3;
use serde_json::Value;
use std::sync::Arc;

const DT: f64 = 1e-3;
const G: f64 = 9.81;

fn fixture() -> Value {
    let path = format!("{}/../../fixtures/chrono/handling_sedan.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

/// Flat ground with the tyres' reference friction and no rolling resistance, as in Chrono.
struct Ground {
    terrain: FlatTerrain,
    materials: MaterialTable,
}

impl Ground {
    fn new() -> Self {
        let mut materials = MaterialTable::standard();
        let mut m = materials.get(MaterialId::ASPHALT).clone();
        m.name = "asphalt_no_rolling".into();
        m.rolling_resistance = 0.0;
        let id = materials.push(m);
        Self { terrain: FlatTerrain::new(0.0, id), materials }
    }

    fn step(&self, v: &mut Wheeled, input: &DriveInput) {
        let env = GroundStepEnv {
            scene: StaticScene { terrain: &self.terrain, obstacles: &NoObstacles, materials: &self.materials },
            gravity: DVec3::new(0.0, 0.0, -G),
            air: AirData::default(),
        };
        v.step(input, &env).unwrap();
    }
}

/// Magic Formula coefficients through which the inclination acts in MF 5.2.
const CAMBER_COEFFICIENTS: [&str; 22] = [
    "PDX3", "PDY3", "PEY4", "PKY3", "PHY3", "PVY3", "PVY4", "RVY3", "QBZ4", "QBZ5", "QDZ3", "QDZ4", "QDZ8", "QDZ9",
    "QEZ5", "QHZ3", "QHZ4", "SSZ3", "SSZ4", "QSX1", "QSX2", "QSY5",
];

/// The Sedan's tyre file with every camber coefficient zeroed: Chrono's `ChPac02Tire` never
/// passes the inclination to its formulas, so its tyres have no camber thrust.
fn tir_without_camber() -> String {
    static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PATH.get_or_init(write_tir_without_camber).clone()
}

fn write_tir_without_camber() -> String {
    let src = format!("{}/../../assets/tires/Sedan_Pac02Tire.tir", env!("CARGO_MANIFEST_DIR"));
    let text: String = std::fs::read_to_string(src)
        .unwrap()
        .lines()
        .map(|line| {
            let key = line.split_whitespace().next().unwrap_or("");
            if CAMBER_COEFFICIENTS.contains(&key) { format!("{key} = 0.0\n") } else { format!("{line}\n") }
        })
        .collect();
    let path = format!("{}/Sedan_Pac02Tire_no_camber.tir", env!("CARGO_TARGET_TMPDIR"));
    std::fs::write(&path, text).unwrap();
    path
}

/// The Sedan without aerodynamic drag (Chrono models none), at rest pose and rolling at
/// `speed`; with tyres as Chrono evaluates them (no camber) unless `camber`.
fn sedan_with(speed: f64, camber: bool) -> Wheeled {
    let mut d = presets::wheeled("sedan_like").unwrap();
    d.chassis.drag_area = DVec3::ZERO;
    if !camber {
        let file = tir_without_camber();
        for a in &mut d.axles {
            a.tire = TireSpec::Tir { file: file.clone(), pressure: None };
        }
        d.finish().unwrap();
    }
    let mut v = Wheeled::new(Arc::new(d), DT);
    let init = v.rest(DVec3::ZERO, 0.0, speed);
    v.reset(&init);
    v
}

/// The Sedan as Chrono models it.
fn sedan(speed: f64) -> Wheeled {
    sedan_with(speed, false)
}

/// Speed PI of the generator: throttle for positive output, brake for negative.
#[derive(Default)]
struct SpeedPi {
    i: f64,
    kp: f64,
    ki: f64,
}

impl SpeedPi {
    fn new(fx: &Value) -> Self {
        Self { i: 0.0, kp: num(&fx["speed_pi"][0]), ki: num(&fx["speed_pi"][1]) }
    }

    fn update(&mut self, v_ref: f64, v: f64, dt: f64) -> (f64, f64) {
        let e = v_ref - v;
        self.i = (self.i + e * dt).clamp(-5.0, 5.0);
        let u = self.kp * e + self.ki * self.i;
        (u.clamp(0.0, 1.0), (-u).clamp(0.0, 1.0))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Sample {
    t: f64,
    x: f64,
    y: f64,
    yaw: f64,
    vx: f64,
    yaw_rate: f64,
    roll: f64,
    /// Mean road-wheel angle of the front axle relative to the chassis.
    delta: f64,
}

impl Sample {
    fn of(v: &Wheeled, t: f64) -> Self {
        let rot = v.orientation();
        let fwd = rot * DVec3::X;
        let vel = v.lin_vel_body();
        let angle = |w: usize| {
            let axis = rot.inverse() * (v.wheel_pose(w).rot * DVec3::Y);
            (-axis.x).atan2(axis.y)
        };
        let (_, _, roll) = rot.to_euler(glam::EulerRot::ZYX);
        Self {
            t,
            x: v.position().x,
            y: v.position().y,
            yaw: fwd.y.atan2(fwd.x),
            vx: vel.x,
            yaw_rate: v.ang_vel_body().z,
            roll,
            delta: 0.5 * (angle(0) + angle(1)),
        }
    }

    fn from_json(s: &Value) -> Self {
        Self {
            t: num(&s["t"]),
            x: num(&s["x"]),
            y: num(&s["y"]),
            yaw: num(&s["yaw"]),
            vx: num(&s["vx"]),
            yaw_rate: num(&s["yaw_rate"]),
            roll: num(&s["roll"]),
            delta: num(&s["delta"]),
        }
    }

    fn lateral_accel(&self) -> f64 {
        self.vx * self.yaw_rate
    }
}

fn samples(v: &Value) -> Vec<Sample> {
    v.as_array().unwrap().iter().map(Sample::from_json).collect()
}

/// Our steering input for road-wheel angle `delta` (parallel steer, linear map).
fn steering_for(v: &Wheeled, delta: f64) -> f64 {
    delta / v.def().steering.unwrap().max_angle
}

/// Chrono's road-wheel angle at zero roll: its front wheels steer out of the turn as the body
/// rolls (roll steer, the same gradient as ours), and its steering lags the input by about
/// 20 ms, so a single sample does not give the angle its input commands. Fitted as
/// `δ = δ₀ + g·φ` over samples from `t0` on.
fn zero_roll_delta(s: &[Sample], t0: f64) -> f64 {
    fit(s.iter().filter(|q| q.t >= t0).map(|q| (q.roll, q.delta))).1
}

/// Least-squares slope and intercept of `y` over `x`.
fn fit(points: impl Iterator<Item = (f64, f64)>) -> (f64, f64) {
    let p: Vec<(f64, f64)> = points.collect();
    let n = p.len() as f64;
    let (mx, my) = (p.iter().map(|q| q.0).sum::<f64>() / n, p.iter().map(|q| q.1).sum::<f64>() / n);
    let sxy: f64 = p.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum();
    let sxx: f64 = p.iter().map(|q| (q.0 - mx).powi(2)).sum();
    (sxy / sxx, my - sxy / sxx * mx)
}

// ------------------------------------------------------------------ ISO 4138 constant steer

fn run_constant_steer(fx: &Value) -> Vec<Sample> {
    let cs = &fx["constant_steer"];
    let chrono = samples(&cs["samples"]);
    let ramp = num(&cs["ramp"]);
    let (v0, v1) = (5.0, 16.0);
    let mut v = sedan(v0);
    let steer = steering_for(&v, zero_roll_delta(&chrono, 2.0));
    let ground = Ground::new();
    let mut speed = SpeedPi::new(fx);
    let mut out = Vec::new();
    let mut k = 0usize;
    loop {
        let t = k as f64 * DT;
        let v_ref = (v0 + ramp * (t - 4.0).max(0.0)).min(v1);
        let (throttle, brake) = speed.update(v_ref, v.lin_vel_body().x, DT);
        let input = DriveInput { steering: steer * (t / 1.0).min(1.0), throttle, brake, ..Default::default() };
        ground.step(&mut v, &input);
        k += 1;
        if k.is_multiple_of(250) {
            out.push(Sample::of(&v, k as f64 * DT));
        }
        if v_ref >= v1 && t > 4.0 + (v1 - v0) / ramp + 3.0 {
            break;
        }
    }
    out
}

/// Understeer gradient (rad per m/s²) over lateral accelerations `range`, from the path
/// curvature at the fixed steering input: `K = −d(L/R)/da_y` (ISO 4138).
fn understeer(s: &[Sample], wheelbase: f64, range: (f64, f64)) -> f64 {
    let pts = s.iter().filter(|q| (range.0..=range.1).contains(&q.lateral_accel()));
    fit(pts.map(|q| (q.lateral_accel(), -wheelbase * q.yaw_rate / q.vx))).0
}

/// The Chrono Sedan is close to neutral steer (K ≈ 0.0005 rad/(m/s²) up to 3 m/s², 0.3°/g), so
/// a relative bound on K is ill-conditioned; K must agree within 10 % of a typical passenger
/// car's gradient (3°/g ≈ 0.0053 rad/(m/s²)), and the path curvature within 5 % throughout (up to 6 m/s²).
#[test]
fn constant_steer_understeer_gradient() {
    let fx = fixture();
    let l = num(&fx["wheelbase"]);
    let chrono = samples(&fx["constant_steer"]["samples"]);
    let ours = run_constant_steer(&fx);
    for range in [(0.5, 3.0), (3.0, 5.5)] {
        let (k, kc) = (understeer(&ours, l, range), understeer(&chrono, l, range));
        println!("a_y {range:?}: K {k:.6} (Chrono {kc:.6}) rad/(m/s²)");
        assert!((k - kc).abs() < 5.3e-4, "understeer gradient over a_y {range:?}: {k:.6} vs Chrono {kc:.6}");
    }
    assert_eq!(ours.len(), chrono.len());
    let mut worst: f64 = 0.0;
    for (o, c) in ours.iter().zip(&chrono).filter(|(_, c)| c.t >= 2.0) {
        let (ko, kc) = (o.yaw_rate / o.vx, c.yaw_rate / c.vx);
        worst = worst.max((ko / kc - 1.0).abs());
    }
    println!("largest curvature difference {:.2} %", 100.0 * worst);
    assert!(worst < 0.05, "path curvature differs from Chrono by {:.2} %", 100.0 * worst);
}

// ------------------------------------------------------------------ ISO 7401 step steer

fn approach(fx: &Value, v_ref: f64, seconds: f64, camber: bool) -> (Wheeled, SpeedPi, (f64, f64)) {
    let mut v = sedan_with(v_ref, camber);
    let ground = Ground::new();
    let mut speed = SpeedPi::new(fx);
    let mut pedals = (0.0, 0.0);
    for _ in 0..(seconds / DT).round() as usize {
        pedals = speed.update(v_ref, v.lin_vel_body().x, DT);
        ground.step(&mut v, &DriveInput { throttle: pedals.0, brake: pedals.1, ..Default::default() });
    }
    (v, speed, pedals)
}

/// Road-wheel angle relative to the chassis and inclination (the tyre's convention,
/// `sin γ = axis·n`) of every wheel.
fn wheel_angles(v: &Wheeled) -> [[f64; 2]; 4] {
    let rot = v.orientation();
    std::array::from_fn(|w| {
        let axis = v.wheel_pose(w).rot * DVec3::Y;
        let local = rot.inverse() * axis;
        [(-local.x).atan2(local.y), axis.z.asin()]
    })
}

struct StepSteer {
    samples: Vec<Sample>,
    /// Wheel angles before the step and averaged over the last second.
    wheels_before: [[f64; 2]; 4],
    wheels_steady: [[f64; 2]; 4],
    vehicle: Wheeled,
}

fn run_step_steer(fx: &Value, run: &Value, camber: bool) -> StepSteer {
    let chrono = samples(&run["samples"]);
    let delta = zero_roll_delta(&chrono, 0.15);
    let (mut v, _, (throttle, _)) = approach(fx, 80.0 / 3.6, 5.0, camber);
    let steer = steering_for(&v, delta);
    let ground = Ground::new();
    let mut samples = vec![Sample::of(&v, 0.0)];
    let wheels_before = wheel_angles(&v);
    let mut wheels_steady = [[0.0; 2]; 4];
    for k in 1..=4000usize {
        let t = (k - 1) as f64 * DT;
        let input = DriveInput { steering: steer * (t / 0.1).min(1.0), throttle, ..Default::default() };
        ground.step(&mut v, &input);
        if k.is_multiple_of(10) {
            samples.push(Sample::of(&v, k as f64 * DT));
        }
        if k > 3000 {
            for (acc, a) in wheels_steady.iter_mut().zip(wheel_angles(&v)) {
                acc[0] += a[0] / 1000.0;
                acc[1] += a[1] / 1000.0;
            }
        }
    }
    StepSteer { samples, wheels_before, wheels_steady, vehicle: v }
}

/// Steady yaw rate (mean over the last second), peak yaw rate, and the response time (from 50 %
/// of the steering ramp, 0.05 s, to 90 % of the steady yaw rate).
fn yaw_response(s: &[Sample]) -> (f64, f64, f64) {
    let tail: Vec<f64> = s.iter().filter(|q| q.t >= 3.0).map(|q| q.yaw_rate).collect();
    let steady = tail.iter().sum::<f64>() / tail.len() as f64;
    let peak = s.iter().map(|q| q.yaw_rate).fold(f64::MIN, f64::max);
    let t90 = s.iter().find(|q| q.yaw_rate >= 0.9 * steady).unwrap().t - 0.05;
    (steady, peak, t90)
}

#[test]
fn step_steer_yaw_rate_response() {
    let fx = fixture();
    for run in fx["step_steer"].as_array().unwrap() {
        let chrono = samples(&run["samples"]);
        let ours = run_step_steer(&fx, run, false).samples;
        let (sc, pc, tc) = yaw_response(&chrono);
        let (s, p, t) = yaw_response(&ours);
        let (ay, ayc) = (ours.last().unwrap().lateral_accel(), chrono.last().unwrap().lateral_accel());
        println!(
            "step {}: yaw rate {s:.4} (Chrono {sc:.4}), peak {p:.4} ({pc:.4}), t90 {t:.3} ({tc:.3}) s, a_y {ay:.2} ({ayc:.2})",
            num(&run["steering"])
        );
        assert!((s / sc - 1.0).abs() < 0.1, "steady yaw rate {s:.4} vs Chrono {sc:.4}");
        assert!((p / pc - 1.0).abs() < 0.1, "peak yaw rate {p:.4} vs Chrono {pc:.4}");
        assert!((t - tc).abs() < 0.03, "response time {t:.3} vs Chrono {tc:.3} s");
    }
}

/// Steady yaw rate of the linear single-track model with per-wheel kinematics: every wheel's
/// lateral force is linear in its slip angle and inclination, `F_w = C_α·((v_y + x_w·r)/u − Δδ_w)
/// + C_γ·Δγ_w`, with the cornering and camber stiffnesses of the Magic Formula at the static
/// load and straight-running inclination, and the changes `Δδ_w`, `Δγ_w` of road-wheel angle
/// and inclination from straight running (steering, roll and compliance steer, roll camber) as
/// the vehicle model produced them. Solves `ΣF_w = m·u·r` and `Σx_w·F_w = 0` for `(v_y, r)`.
fn bicycle_yaw_rate(v: &Wheeled, u: f64, before: &[[f64; 2]; 4], steady: &[[f64; 2]; 4]) -> f64 {
    let d = v.def();
    let st = d.static_state(G).unwrap();
    let (m, cg) = (d.total_mass(), d.total_com());
    let (mut a, mut b) = ([[0.0; 2]; 2], [0.0; 2]);
    a[0][1] -= m * u;
    for w in 0..4 {
        let TireModel::MagicFormula(p) = &d.tire(w / 2).model else { panic!("MF tyres expected") };
        let fy = |alpha: f64, gamma: f64| p.eval(&MfInput::new(st.loads[w], 0.0, alpha, gamma, u)).fy;
        let (h, g0) = (1e-4, before[w][1]);
        let c_alpha = (fy(h, g0) - fy(-h, g0)) / (2.0 * h);
        let c_gamma = (fy(0.0, g0 + h) - fy(0.0, g0 - h)) / (2.0 * h);
        let x = d.wheel_position(w).x - cg.x;
        let rhs = c_alpha * (steady[w][0] - before[w][0]) - c_gamma * (steady[w][1] - before[w][1]);
        a[0][0] += c_alpha / u;
        a[0][1] += c_alpha * x / u;
        a[1][0] += c_alpha * x / u;
        a[1][1] += c_alpha * x * x / u;
        b[0] += rhs;
        b[1] += x * rhs;
    }
    (a[0][0] * b[1] - a[1][0] * b[0]) / (a[0][0] * a[1][1] - a[0][1] * a[1][0])
}

/// At low lateral acceleration (the small step, about 1 m/s²) the steady yaw rate agrees
/// with the linear model within 10 %, with the tyres as Chrono evaluates them and with camber.
#[test]
fn step_steer_matches_linear_bicycle_model() {
    let fx = fixture();
    let run = &fx["step_steer"][0];
    for camber in [false, true] {
        let s = run_step_steer(&fx, run, camber);
        let (steady, _, _) = yaw_response(&s.samples);
        let u = s.samples.last().unwrap().vx;
        let r = bicycle_yaw_rate(&s.vehicle, u, &s.wheels_before, &s.wheels_steady);
        println!("camber {camber}: yaw rate {steady:.5}, linear model {r:.5}");
        assert!((steady / r - 1.0).abs() < 0.1, "yaw rate {steady:.5} vs linear model {r:.5} (camber {camber})");
    }
}

// ------------------------------------------------------------------ braking

/// Straight braking from 100 km/h: stopping distance within 5 % of Chrono. Without ABS, the
/// Chrono Sedan's rear wheels lock from pedal 0.7 and it spins (yaw rate 1.4–1.75 rad/s, in
/// the fixture); it stays straight at the lower pedals, which are compared.
#[test]
fn braking_from_100_kmh() {
    let fx = fixture();
    for run in fx["brake"].as_array().unwrap().iter().filter(|r| num(&r["pedal"]) <= 0.5) {
        let pedal = num(&run["pedal"]);
        let chrono = samples(&run["samples"]);
        let stop_c = chrono.iter().find(|q| q.vx < 0.1).unwrap();
        let (mut v, _, _) = approach(&fx, 100.0 / 3.6, 3.0, false);
        let ground = Ground::new();
        let x0 = v.position().x;
        let mut t = 0.0;
        while v.lin_vel_body().x > 0.1 && t < 8.0 {
            ground.step(&mut v, &DriveInput { brake: pedal, ..Default::default() });
            t += DT;
        }
        let d = v.position().x - x0;
        println!("pedal {pedal}: {d:.2} m in {t:.2} s (Chrono {:.2} m in {:.2} s)", stop_c.x, stop_c.t);
        assert!((d / stop_c.x - 1.0).abs() < 0.05, "stopping distance {d:.2} vs Chrono {:.2} m", stop_c.x);
    }
}

// ------------------------------------------------------------------ ISO 3888-1 lane change

struct Course {
    sections: Vec<[f64; 4]>,
    offset: f64,
    lookahead: (f64, f64),
    wheelbase: f64,
}

impl Course {
    fn new(fx: &Value) -> Self {
        let c = &fx["course"];
        Self {
            sections: c["sections"].as_array().unwrap().iter().map(|s| [0, 1, 2, 3].map(|i| num(&s[i]))).collect(),
            offset: num(&c["offset"]),
            lookahead: (num(&c["lookahead"][0]), num(&c["lookahead"][1])),
            wheelbase: num(&fx["wheelbase"]),
        }
    }

    fn path_y(&self, x: f64) -> f64 {
        let blend = |a: f64, b: f64| 0.5 - 0.5 * (std::f64::consts::PI * ((x - a) / (b - a)).clamp(0.0, 1.0)).cos();
        self.offset * (blend(15.0, 45.0) - blend(70.0, 95.0))
    }

    /// Pure pursuit of the centreline point one lookahead ahead in x.
    fn pursuit(&self, x: f64, y: f64, yaw: f64, v: f64) -> f64 {
        let ld = self.lookahead.0.max(self.lookahead.1 * v);
        let (dx, dy) = (ld, self.path_y(x + ld) - y);
        let alpha = dy.atan2(dx) - yaw;
        (2.0 * self.wheelbase * alpha.sin()).atan2(dx.hypot(dy))
    }
}

fn run_lane_change(fx: &Value, course: &Course) -> Vec<Sample> {
    let v_ref = num(&fx["lane_change"]["speed"]);
    let (mut v, mut speed, _) = approach(fx, v_ref, 3.0, false);
    let ground = Ground::new();
    let x0 = v.position().x + 10.0;
    let mut out = Vec::new();
    let mut k = 0usize;
    loop {
        let s = Sample::of(&v, 0.0);
        let x = s.x - x0;
        let delta = course.pursuit(x, s.y, s.yaw, s.vx);
        let (throttle, brake) = speed.update(v_ref, s.vx, DT);
        let input =
            DriveInput { steering: steering_for(&v, delta).clamp(-1.0, 1.0), throttle, brake, ..Default::default() };
        ground.step(&mut v, &input);
        k += 1;
        if k.is_multiple_of(20) {
            let mut s = Sample::of(&v, k as f64 * DT);
            s.x -= x0;
            out.push(s);
        }
        if x > 140.0 {
            break;
        }
    }
    out
}

/// Both cars, steered by the same pure-pursuit driver, keep their centre inside every lane of
/// the course, and ours follows Chrono's path within 0.3 m with peak yaw rate and lateral
/// acceleration within 10 %.
#[test]
fn double_lane_change() {
    let fx = fixture();
    let course = Course::new(&fx);
    let chrono = samples(&fx["lane_change"]["samples"]);
    let ours = run_lane_change(&fx, &course);
    let peak = |s: &[Sample], f: fn(&Sample) -> f64| s.iter().map(|q| f(q).abs()).fold(0.0, f64::max);
    for (name, f) in [("yaw rate", (|q: &Sample| q.yaw_rate) as fn(&Sample) -> f64), ("a_y", Sample::lateral_accel)] {
        let (o, c) = (peak(&ours, f), peak(&chrono, f));
        println!("peak {name} {o:.3} (Chrono {c:.3})");
        assert!((o / c - 1.0).abs() < 0.1, "peak {name} {o:.3} vs Chrono {c:.3}");
    }
    for sec in &course.sections {
        for (name, s) in [("ours", &ours), ("Chrono", &chrono)] {
            let ys: Vec<f64> = s.iter().filter(|q| (sec[0]..=sec[1]).contains(&q.x)).map(|q| q.y).collect();
            let (lo, hi) = (ys.iter().cloned().fold(f64::MAX, f64::min), ys.iter().cloned().fold(f64::MIN, f64::max));
            println!(
                "section {:.0}–{:.0} m, lane {:.2}..{:.2}: {name} {lo:.2}..{hi:.2}",
                sec[0], sec[1], sec[2], sec[3]
            );
            assert!(
                !ys.is_empty() && lo > sec[2] && hi < sec[3],
                "{name} leaves the lane at x {:.0}–{:.0} m",
                sec[0],
                sec[1]
            );
        }
    }
    // Lateral position against x, where both have samples.
    let mut worst: f64 = 0.0;
    for o in &ours {
        if let Some(w) = chrono.windows(2).find(|w| w[0].x <= o.x && o.x < w[1].x) {
            let f = (o.x - w[0].x) / (w[1].x - w[0].x);
            worst = worst.max((o.y - (w[0].y + f * (w[1].y - w[0].y))).abs());
        }
    }
    println!("largest lateral difference {worst:.3} m");
    assert!(worst < 0.3, "path differs from Chrono by {worst:.3} m");
}
