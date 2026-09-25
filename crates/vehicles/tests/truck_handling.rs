//! Truck handling against Chrono::Vehicle (fixtures/chrono/truck_handling.json from
//! tools/gen_chrono_truck_handling_fixtures.py): the MAN 10t 8×8 in a constant-steer run, and
//! the Kraz tractor with its semitrailer in a step steer and a single sine steer at 60 km/h;
//! and low-speed offtracking of the rigs against the kinematic steady state.
//!
//! Both sides run the same driver laws (a speed PI, and pure pursuit on the straight
//! approach). Chrono's steering linkages are compliant (the MAN's road-wheel angle at a fixed
//! input falls by two thirds between 4 and 9 m/s), so our runs replay Chrono's road-wheel
//! angles rather than its inputs. As in `handling.rs`, our tyres run without camber
//! coefficients (Chrono's PAC2002 ignores the inclination) and without drag.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::presets;
use glam::{DVec2, DVec3};
use serde_json::Value;
use std::sync::Arc;

const DT: f64 = 1e-3;
const G: f64 = 9.81;

fn fixture() -> &'static Value {
    static FX: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    FX.get_or_init(|| {
        let path = format!("{}/../../fixtures/chrono/truck_handling.json", env!("CARGO_MANIFEST_DIR"));
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    })
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

/// Magic Formula coefficients through which the inclination acts in MF 5.2 (see
/// `handling.rs`).
const CAMBER_COEFFICIENTS: [&str; 22] = [
    "PDX3", "PDY3", "PEY4", "PKY3", "PHY3", "PVY3", "PVY4", "RVY3", "QBZ4", "QBZ5", "QDZ3", "QDZ4", "QDZ8", "QDZ9",
    "QEZ5", "QHZ3", "QHZ4", "SSZ3", "SSZ4", "QSX1", "QSX2", "QSY5",
];

fn tir_without_camber() -> String {
    static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        let src = format!("{}/../../assets/tires/Truck_Pac02Tire.tir", env!("CARGO_MANIFEST_DIR"));
        let text: String = std::fs::read_to_string(src)
            .unwrap()
            .lines()
            .map(|line| {
                let key = line.split_whitespace().next().unwrap_or("");
                if CAMBER_COEFFICIENTS.contains(&key) { format!("{key} = 0.0\n") } else { format!("{line}\n") }
            })
            .collect();
        let path = format!("{}/Truck_Pac02Tire_no_camber.tir", env!("CARGO_TARGET_TMPDIR"));
        std::fs::write(&path, text).unwrap();
        path
    })
    .clone()
}

/// The rig as Chrono models it: no drag, tyres without camber.
fn chrono_like(mut d: WheeledDef) -> WheeledDef {
    d.chassis.drag_area = DVec3::ZERO;
    for u in &mut d.units {
        u.chassis.drag_area = DVec3::ZERO;
    }
    let file = tir_without_camber();
    for a in &mut d.axles {
        a.tire = TireSpec::Tir { file: file.clone(), pressure: None };
    }
    d.finish().unwrap();
    d
}

fn kraz() -> WheeledDef {
    let d = presets::wheeled("truck_6x4").unwrap();
    chrono_like(d.with_trailers(&[presets::trailer("semitrailer_3axle").unwrap()]).unwrap())
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

/// Speed PI of the generator: throttle for positive output, brake for negative.
struct SpeedPi {
    i: f64,
    kp: f64,
    ki: f64,
}

impl SpeedPi {
    fn new() -> Self {
        let fx = fixture();
        Self { i: 0.0, kp: num(&fx["speed_pi"][0]), ki: num(&fx["speed_pi"][1]) }
    }

    fn update(&mut self, v_ref: f64, v: f64) -> (f64, f64) {
        let e = v_ref - v;
        self.i = (self.i + e * DT).clamp(-5.0, 5.0);
        let u = self.kp * e + self.ki * self.i;
        (u.clamp(0.0, 1.0), (-u).clamp(0.0, 1.0))
    }
}

/// The generator's straight driver: bicycle angle (rad) toward y = 0 one lookahead ahead.
fn pursuit(y: f64, yaw: f64, v: f64, wheelbase: f64) -> f64 {
    let fx = fixture();
    let ld = num(&fx["lookahead"][0]).max(num(&fx["lookahead"][1]) * v);
    let alpha = (-y).atan2(ld) - yaw;
    (2.0 * wheelbase * alpha.sin()).atan2(ld.hypot(y))
}

/// Per-unit state at one instant.
#[derive(Clone, Copy, Debug, Default)]
struct Unit {
    yaw: f64,
    yaw_rate: f64,
    com_vel: DVec2,
}

#[derive(Clone, Debug, Default)]
struct Sample {
    t: f64,
    speed: f64,
    /// Mean road-wheel angle of each steered axle.
    delta: Vec<f64>,
    units: Vec<Unit>,
}

impl Sample {
    fn from_json(s: &Value) -> Self {
        let units = s["units"]
            .as_array()
            .unwrap()
            .iter()
            .map(|u| Unit {
                yaw: num(&u["yaw"]),
                yaw_rate: num(&u["yaw_rate"]),
                com_vel: DVec2::new(num(&u["com_vel"][0]), num(&u["com_vel"][1])),
            })
            .collect();
        let delta = s["delta"].as_array().unwrap().iter().map(num).collect();
        Self { t: num(&s["t"]), speed: num(&s["speed"]), delta, units }
    }

    fn articulation(&self) -> f64 {
        let a = self.units[1].yaw - self.units[0].yaw;
        (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
    }
}

fn samples(v: &Value) -> Vec<Sample> {
    v.as_array().unwrap().iter().map(Sample::from_json).collect()
}

/// Linear interpolation of `f` over the samples at time `t`.
fn interp(s: &[Sample], t: f64, f: impl Fn(&Sample) -> f64) -> f64 {
    let k = s.partition_point(|q| q.t <= t);
    if k == 0 {
        return f(&s[0]);
    }
    if k == s.len() {
        return f(&s[k - 1]);
    }
    let (a, b) = (&s[k - 1], &s[k]);
    f(a) + (f(b) - f(a)) * (t - a.t) / (b.t - a.t)
}

/// Lateral acceleration of unit `u`'s centre of mass in its heading frame, by central
/// differences of the sampled velocities (the same operator on both sides).
fn lateral_accel(s: &[Sample], u: usize) -> Vec<f64> {
    (1..s.len() - 1)
        .map(|k| {
            let a = (s[k + 1].units[u].com_vel - s[k - 1].units[u].com_vel) / (s[k + 1].t - s[k - 1].t);
            let yaw = s[k].units[u].yaw;
            a.dot(DVec2::new(-yaw.sin(), yaw.cos()))
        })
        .collect()
}

/// Our rig with a recorder of the per-unit centre of mass and yaw at every step (for the
/// velocities and yaw rates of the trailing units).
struct Run {
    v: Wheeled,
    ground: Ground,
    speed: SpeedPi,
    t: f64,
    /// Per step: each unit's centre of mass and yaw.
    history: Vec<Vec<(DVec2, f64)>>,
}

impl Run {
    fn new(d: WheeledDef, speed: f64) -> Self {
        let mut v = Wheeled::new(Arc::new(d), DT);
        let init = v.rest(DVec3::ZERO, 0.0, speed);
        v.reset(&init);
        let mut run = Self { v, ground: Ground::new(), speed: SpeedPi::new(), t: 0.0, history: vec![] };
        run.history.push(run.units());
        run
    }

    fn units(&self) -> Vec<(DVec2, f64)> {
        let d = self.v.def();
        (0..self.v.num_units())
            .map(|u| {
                let pose = self.v.unit_pose(u);
                let com = if u == 0 { d.chassis.com } else { d.units[u - 1].chassis.com };
                let fwd = pose.rot * DVec3::X;
                (pose.transform_point(com).truncate(), fwd.y.atan2(fwd.x))
            })
            .collect()
    }

    /// One step at speed `v_ref` with steering input `steering`, or the straight driver's.
    fn drive(&mut self, v_ref: f64, steering: Option<f64>, gain: f64, wheelbase: f64) {
        let steering = steering.unwrap_or_else(|| {
            let p = self.v.position();
            let fwd = self.v.orientation() * DVec3::X;
            (pursuit(p.y, fwd.y.atan2(fwd.x), self.v.speed(), wheelbase) / gain).clamp(-1.0, 1.0)
        });
        let (throttle, brake) = self.speed.update(v_ref, self.v.speed());
        let input = DriveInput { steering, throttle, brake, ..Default::default() };
        self.ground.step(&mut self.v, &input);
        self.t += DT;
        self.history.push(self.units());
    }

    /// The state one step ago (velocities by central differences over the recorded steps).
    fn sample(&self) -> Sample {
        let n = self.history.len();
        assert!(n >= 3);
        let (prev, now, next) = (&self.history[n - 3], &self.history[n - 2], &self.history[n - 1]);
        let units = (0..now.len())
            .map(|u| {
                let dyaw = (next[u].1 - prev[u].1 + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI;
                Unit { yaw: now[u].1, yaw_rate: dyaw / (2.0 * DT), com_vel: (next[u].0 - prev[u].0) / (2.0 * DT) }
            })
            .collect();
        let d = self.v.def();
        let steered: Vec<usize> =
            (0..d.axles.len()).filter(|&a| d.axles[a].unit == 0 && d.axles[a].is_steered()).collect();
        let delta =
            steered.iter().map(|&a| 0.5 * (self.v.wheel(2 * a).steer + self.v.wheel(2 * a + 1).steer)).collect();
        Sample { t: self.t - DT, speed: self.v.speed(), delta, units }
    }
}

// ------------------------------------------------------------------ MAN 10t constant steer

/// The MAN's second axle steers by its own linkage ratio in Chrono (about 0.85 of the first at
/// these angles, 1.04 at full lock), not geometrically: the run takes Chrono's ratio. Our
/// steering replays Chrono's lead-axle angle; the path curvature must follow Chrono's.
#[test]
fn man_constant_steer() {
    let fx = &fixture()["man_constant_steer"];
    let chrono = samples(&fx["samples"]);
    let ramp = num(&fx["ramp"]);
    let (v0, v1) = (num(&fx["speed"][0]), num(&fx["speed"][1]));
    let steady: Vec<&Sample> = chrono.iter().filter(|s| s.t >= 5.0).collect();
    let ratio = steady.iter().map(|s| s.delta[1] / s.delta[0]).sum::<f64>() / steady.len() as f64;
    let mut d = presets::wheeled("truck_8x8").unwrap();
    let lead = d.axles[0].share();
    d.axles[1].steer = Steer::Share(ratio * lead);
    let d = chrono_like(d);
    let gain = d.steering.unwrap().max_angle * lead;
    let wheelbase = num(&fixture()["wheelbase"]["man_10t"]);
    let mut run = Run::new(d, 0.0);
    let mut ours = vec![];
    let t_end = chrono.last().unwrap().t;
    while run.t < t_end - 1e-9 {
        let t = run.t;
        let v_ref = (v0 + ramp * (t - 5.0).max(0.0)).min(v1);
        let delta = interp(&chrono, t, |s| s.delta[0]);
        run.drive(v_ref, Some(delta / gain), gain, wheelbase);
        let k = ((run.t - DT) / 0.25).round();
        if (run.t - DT - 0.25 * k).abs() < 1e-6 && run.history.len() >= 3 {
            ours.push(run.sample());
        }
    }
    let mut worst: f64 = 0.0;
    println!("   t    v    δ1 ours/Chrono   δ2 ours/Chrono   curvature ours/Chrono   a_y");
    for c in steady.iter().step_by(8) {
        let o = ours.iter().min_by(|a, b| (a.t - c.t).abs().total_cmp(&(b.t - c.t).abs())).unwrap();
        let (ko, kc) = (o.units[0].yaw_rate / o.speed, c.units[0].yaw_rate / c.speed);
        println!(
            "{:5.1} {:4.1} {:.4}/{:.4} {:.4}/{:.4} {:.5}/{:.5} {:.2}",
            c.t,
            c.speed,
            o.delta[0],
            c.delta[0],
            o.delta[1],
            c.delta[1],
            ko,
            kc,
            c.speed * c.units[0].yaw_rate
        );
    }
    for c in &steady {
        let o = ours.iter().min_by(|a, b| (a.t - c.t).abs().total_cmp(&(b.t - c.t).abs())).unwrap();
        let (ko, kc) = (o.units[0].yaw_rate / o.speed, c.units[0].yaw_rate / c.speed);
        worst = worst.max((ko / kc - 1.0).abs());
    }
    println!("largest curvature difference {:.2} %", 100.0 * worst);
    assert!(worst < 0.05, "path curvature differs from Chrono by {:.2} %", 100.0 * worst);
}

// ------------------------------------------------------------------ Kraz manoeuvres

/// The Kraz rig on the straight driver for 15 s at the manoeuvre speed, then the steering
/// input at that moment plus Chrono's change of lead-axle angle, for as long as Chrono's run.
fn kraz_run(fx: &Value) -> (Vec<Sample>, Vec<Sample>) {
    let chrono = samples(&fx["samples"]);
    let speed = num(&fx["speed"]);
    let d = kraz();
    let gain = d.steering.unwrap().max_angle;
    let wheelbase = num(&fixture()["wheelbase"]["kraz_rig"]);
    let mut run = Run::new(d, speed);
    let mut s0 = 0.0;
    while run.t < 15.0 - 1e-9 {
        run.drive(speed, None, gain, wheelbase);
        let p = run.v.position();
        let fwd = run.v.orientation() * DVec3::X;
        s0 = (pursuit(p.y, fwd.y.atan2(fwd.x), run.v.speed(), wheelbase) / gain).clamp(-1.0, 1.0);
    }
    let (t0, delta0) = (run.t, chrono[0].delta[0]);
    let mut ours = vec![];
    let t_end = chrono.last().unwrap().t;
    while run.t - t0 < t_end + 2.0 * DT - 1e-9 {
        let t = run.t - t0;
        run.drive(speed, Some(s0 + (interp(&chrono, t, |s| s.delta[0]) - delta0) / gain), gain, wheelbase);
        let k = ((run.t - DT - t0) / 0.01).round();
        if ((run.t - DT - t0) - 0.01 * k).abs() < 1e-6 && run.history.len() >= 3 {
            let mut s = run.sample();
            s.t -= t0;
            ours.push(s);
        }
    }
    (chrono, ours)
}

fn peak(s: &[Sample], f: impl Fn(&Sample) -> f64) -> (f64, f64) {
    // The sample of largest magnitude, with its time.
    s.iter().map(|q| (q.t, f(q))).max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap()
}

fn mean(s: &[Sample], range: (f64, f64), f: impl Fn(&Sample) -> f64) -> f64 {
    let v: Vec<f64> = s.iter().filter(|q| (range.0..=range.1).contains(&q.t)).map(f).collect();
    v.iter().sum::<f64>() / v.len() as f64
}

fn rel(a: f64, b: f64) -> f64 {
    (a / b - 1.0).abs()
}

/// Step steer at 60 km/h (about 1.9 m/s² steady): the tractor's and trailer's yaw rates and
/// the articulation, peaks and steady state.
#[test]
fn kraz_step_steer() {
    let (chrono, ours) = kraz_run(&fixture()["kraz_step_steer"]);
    println!("  t   δ ours/Chrono   r tractor ours/Chrono   r trailer ours/Chrono   articulation ours/Chrono");
    for (o, c) in ours.iter().zip(&chrono).step_by(25) {
        println!(
            "{:4.2} {:.4}/{:.4} {:.4}/{:.4} {:.4}/{:.4} {:.4}/{:.4}",
            c.t,
            o.delta[0],
            c.delta[0],
            o.units[0].yaw_rate,
            c.units[0].yaw_rate,
            o.units[1].yaw_rate,
            c.units[1].yaw_rate,
            o.articulation(),
            c.articulation()
        );
    }
    for (label, f, tol_peak, tol_steady) in [
        ("tractor yaw rate", Box::new(|s: &Sample| s.units[0].yaw_rate) as Box<dyn Fn(&Sample) -> f64>, 0.05, 0.05),
        ("trailer yaw rate", Box::new(|s: &Sample| s.units[1].yaw_rate), 0.1, 0.05),
        ("articulation", Box::new(|s: &Sample| s.articulation()), 0.1, 0.1),
    ] {
        let ((to, po), (tc, pc)) = (peak(&ours, &f), peak(&chrono, &f));
        let (so, sc) = (mean(&ours, (6.0, 8.0), &f), mean(&chrono, (6.0, 8.0), &f));
        println!("{label}: peak {po:.4} at {to:.2} s (Chrono {pc:.4} at {tc:.2} s), steady {so:.4} ({sc:.4})");
        assert!(rel(po, pc) < tol_peak, "{label}: peak {po} vs {pc}");
        assert!((to - tc).abs() < 0.25, "{label}: peak at {to} vs {tc}");
        assert!(rel(so, sc) < tol_steady, "{label}: steady {so} vs {sc}");
    }
}

/// Single sine steer (0.4 Hz) at 60 km/h: peaks of the yaw rates and articulation, and the
/// rearward amplification (the trailer's peak lateral acceleration over the tractor's).
#[test]
fn kraz_sine_steer() {
    let (chrono, ours) = kraz_run(&fixture()["kraz_sine_steer"]);
    for (label, f) in [
        ("tractor yaw rate", Box::new(|s: &Sample| s.units[0].yaw_rate) as Box<dyn Fn(&Sample) -> f64>),
        ("trailer yaw rate", Box::new(|s: &Sample| s.units[1].yaw_rate)),
        ("articulation", Box::new(|s: &Sample| s.articulation())),
    ] {
        // Both lobes: the positive and the negative extreme.
        for sign in [1.0, -1.0] {
            let g = |s: &Sample| (sign * f(s)).max(0.0);
            let ((to, po), (tc, pc)) = (peak(&ours, g), peak(&chrono, g));
            println!("{label}: {sign:+} peak {po:.4} at {to:.2} s (Chrono {pc:.4} at {tc:.2} s)");
            assert!(rel(po, pc) < 0.1, "{label}: peak {po} vs {pc}");
            assert!((to - tc).abs() < 0.25, "{label}: peak at {to} vs {tc}");
        }
    }
    let amplification = |s: &[Sample]| {
        let [tractor, trailer] = [0, 1].map(|u| lateral_accel(s, u).iter().fold(0.0f64, |m, a| m.max(a.abs())));
        (tractor, trailer, trailer / tractor)
    };
    let ((ao, bo, ro), (ac, bc, rc)) = (amplification(&ours), amplification(&chrono));
    println!(
        "peak a_y: tractor {ao:.3} ({ac:.3}), trailer {bo:.3} ({bc:.3}); rearward amplification {ro:.3} ({rc:.3})"
    );
    assert!(rel(ao, ac) < 0.1 && rel(bo, bc) < 0.1, "peak lateral accelerations");
    assert!((ro - rc).abs() < 0.1, "rearward amplification {ro} vs {rc}");
}
