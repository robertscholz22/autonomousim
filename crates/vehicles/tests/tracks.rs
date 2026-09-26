//! Tracked running gear on rigid ground: the static solution, straight running, holding and
//! sliding on slopes around `atan μ`, skid steering, and the track patches' steady shear
//! against Janosi–Hanamoto's integral; the analytic checks, gradeability and Wong and Chiang's
//! pivot turn; and the `tracked_apc` against Chrono's M113 (fixtures/chrono/tracked_m113.json:
//! static pose and loads, launch, crawling up 15 %, holding on 30 %, brake steering).

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use autonomousim_core::terrain::{FlatTerrain, PlaneTerrain, Terrain};
use autonomousim_vehicles::ground::tire::{TRACK_CELLS, TireModel};
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::presets;
use glam::{DQuat, DVec3};
use std::sync::Arc;

const DT: f64 = 1e-3;
const G: f64 = 9.81;

struct World {
    terrain: Box<dyn Terrain>,
    materials: MaterialTable,
}

impl World {
    fn new(terrain: impl Terrain + 'static) -> Self {
        Self { terrain: Box::new(terrain), materials: MaterialTable::standard() }
    }

    fn flat() -> Self {
        Self::new(FlatTerrain::new(0.0, MaterialId::ASPHALT))
    }

    fn run(&self, v: &mut Wheeled, input: &DriveInput, seconds: f64) {
        let env = GroundStepEnv {
            scene: StaticScene { terrain: self.terrain.as_ref(), obstacles: &NoObstacles, materials: &self.materials },
            gravity: DVec3::new(0.0, 0.0, -G),
            air: AirData::default(),
        };
        for _ in 0..(seconds / DT).round() as usize {
            v.step(input, &env).unwrap();
        }
    }
}

fn rover(drag: bool) -> Wheeled {
    let mut d = presets::wheeled("rover_tracked").unwrap();
    if !drag {
        d.chassis.drag_area = DVec3::ZERO;
    }
    let mut v = Wheeled::new(Arc::new(d), DT);
    let init = v.rest(DVec3::ZERO, 0.0, 0.0);
    v.reset(&init);
    v
}

fn patch(v: &Wheeled) -> tire::TrackPatch {
    let TireModel::Track(p) = &v.def().tire(0).model else { panic!("track expected") };
    p.clone()
}

/// Band speed of a side (0 left, 1 right) from its road wheels' spin (m/s).
fn band(v: &Wheeled, side: usize) -> f64 {
    let ws: Vec<_> = v.wheels().enumerate().filter(|(w, _)| w % 2 == side).map(|(_, s)| s).collect();
    ws.iter().map(|s| s.spin * (patch(v).radius - s.tire.deflection)).sum::<f64>() / ws.len() as f64
}

#[test]
fn rover_loads_and_rests_at_the_static_solution() {
    let d = presets::wheeled("rover_tracked").unwrap();
    assert!((d.total_mass() - 56.0).abs() < 1e-9);
    assert_eq!(d.num_wheels(), 8);
    assert!(d.tire(0).is_track() && d.track.is_some());
    // Road wheels on each side are neighbours along the track, front to rear.
    let nb = d.track_neighbours();
    assert_eq!(nb[0], [None, Some(2)]);
    assert_eq!(nb[3], [Some(1), Some(5)]);
    assert_eq!(nb[7], [Some(5), None]);
    // Sprocket and idler on both sides are running-gear colliders.
    let cs = d.sphere_colliders();
    assert_eq!(cs.len(), 2 + 4);
    // The trailing arm swings the wheel back as it rises.
    let table = d.axles[0].suspension.as_ref().unwrap().table(0).unwrap();
    assert!(table.eval(0.03).position.x < table.eval(0.0).position.x);

    let st = d.static_state(G).unwrap();
    let mut v = rover(true);
    World::flat().run(&mut v, &DriveInput { parking: true, ..Default::default() }, 3.0);
    assert!(v.lin_vel_world().length() < 1e-3 && v.contacts().is_empty());
    assert!((v.position().z - st.height).abs() < 5e-4, "height {} vs {}", v.position().z, st.height);
    for (k, wh) in v.wheels().enumerate() {
        assert!((wh.tire.fz / st.loads[k] - 1.0).abs() < 5e-3, "wheel {k}: {} vs {}", wh.tire.fz, st.loads[k]);
        assert!(wh.travel.abs() < 5e-4, "wheel {k}: travel {}", wh.travel);
    }
    let total: f64 = st.loads.iter().sum();
    assert!((total / (v.mass() * G) - 1.0).abs() < 1e-9);
}

/// On the level at constant speed the motors carry the internal resistance and the tracks run
/// without slip, straight. Climbing a 15° grade at constant speed, the tracks' traction
/// carries the grade's pull, at the slip where the patches' steady shear (Janosi–Hanamoto
/// over each patch under its load, shear rising from the front of the track) gives it; the
/// cells' shear grows from front to rear. Braked, the tracks stop without creep.
#[test]
fn drives_straight_and_climbs_with_the_slip_of_its_traction() {
    let input = DriveInput { throttle: 0.7, ..Default::default() };
    let w = World::flat();
    let mut v = rover(false);
    w.run(&mut v, &input, 4.0);
    let u = v.lin_vel_body().x;
    assert!(u > 0.8, "speed {u}");
    assert!(v.position().y.abs() < 1e-6 && v.ang_vel_body().z.abs() < 1e-9, "not straight");
    for side in 0..2 {
        let slip = 1.0 - u / band(&v, side);
        assert!(slip.abs() < 1e-3, "side {side}: slip {slip}");
    }

    let angle = f64::to_radians(15.0);
    let w = World::new(PlaneTerrain::incline(angle, MaterialId::ASPHALT));
    let tilt = DQuat::from_rotation_y(-angle);
    let mut v = rover(false);
    let rest = v.rest(DVec3::ZERO, 0.0, 0.0);
    v.reset(&WheeledInit { pose: Pose::new(tilt * rest.pose.pos, tilt * rest.pose.rot), ..rest });
    w.run(&mut v, &input, 4.0);
    let u0 = v.lin_vel_body().x;
    w.run(&mut v, &input, 0.2);
    let u = v.lin_vel_body().x;
    assert!(u > 0.5 && ((u - u0) / 0.2).abs() < 0.01, "speed {u0} → {u}");
    let fx: f64 = v.wheels().map(|s| s.tire.fx).sum();
    let pull = v.mass() * G * angle.sin();
    assert!((fx / pull - 1.0).abs() < 0.01, "traction {fx} vs grade {pull}");
    let p = patch(&v);
    let mu = p.mu * MaterialTable::standard().get(MaterialId::ASPHALT).friction / 0.8;
    for side in 0..2 {
        let slip = 1.0 - u / band(&v, side);
        // Steady traction at slip i of the side's patches (front to rear), under their loads.
        let loads: Vec<f64> = (0..4).map(|k| v.wheel(2 * k + side).tire.fz).collect();
        let traction = |i: f64| -> f64 {
            let (k, l) = (p.shear_modulus, p.length);
            let at = |x: f64| if i * x < 1e-12 { 0.0 } else { 1.0 - k / (i * x) * (1.0 - (-i * x / k).exp()) };
            // Patch n spans [n·l, (n+1)·l] from the front: the mean stress over it.
            loads
                .iter()
                .enumerate()
                .map(|(n, fz)| mu * fz * ((n + 1) as f64 * at((n + 1) as f64 * l) - n as f64 * at(n as f64 * l)))
                .sum()
        };
        let side_fx: f64 = (0..4).map(|k| v.wheel(2 * k + side).tire.fx).sum();
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if traction(mid) < side_fx { lo = mid } else { hi = mid }
        }
        assert!(slip > 0.0 && (slip / lo - 1.0).abs() < 0.05, "side {side}: slip {slip} vs {lo}");
    }
    let shear: Vec<f64> = (0..4).flat_map(|k| v.tire_state(2 * k).shear.map(|c| c[0])).collect();
    assert_eq!(shear.len(), 4 * TRACK_CELLS);
    assert!(shear.windows(2).all(|p| p[1] > p[0]), "{shear:?}");

    let braked = DriveInput { brake: 1.0, ..Default::default() };
    w.run(&mut v, &braked, 5.0);
    let x = v.position();
    w.run(&mut v, &braked, 3.0);
    assert!((v.position() - x).length() < 1e-4, "crept {}", (v.position() - x).length());
}

/// Braked, the tracks hold on a slope below `atan μ` without creep and slide down one above
/// it at `g(sin θ − μ cos θ)`.
#[test]
fn holds_below_atan_mu_and_slides_above() {
    let mu = 0.9 * MaterialTable::standard().get(MaterialId::ASPHALT).friction / 0.8;
    let parked = DriveInput { parking: true, brake: 1.0, ..Default::default() };
    for (deg, holds) in [(38.0, true), (50.0, false)] {
        let angle = f64::to_radians(deg);
        assert_eq!(angle.tan() < mu, holds);
        let w = World::new(PlaneTerrain::incline(angle, MaterialId::ASPHALT));
        let tilt = DQuat::from_rotation_y(-angle);
        for yaw in [0.0, std::f64::consts::PI] {
            let mut v = rover(false);
            let rest = v.rest(DVec3::ZERO, yaw, 0.0);
            v.reset(&WheeledInit { pose: Pose::new(tilt * rest.pose.pos, tilt * rest.pose.rot), ..rest });
            w.run(&mut v, &parked, if holds { 5.0 } else { 2.0 });
            let p = v.position();
            let vel = v.lin_vel_world();
            w.run(&mut v, &parked, 2.0);
            assert!(v.contacts().is_empty(), "{deg}° yaw {yaw}: hull on the ground");
            if holds {
                let drift = (v.position() - p).length();
                assert!(drift < 1e-4, "{deg}° yaw {yaw}: crept {drift} m in 2 s");
            } else {
                let a = (v.lin_vel_world() - vel).length() / 2.0;
                let expected = G * (angle.sin() - mu * angle.cos());
                assert!((a / expected - 1.0).abs() < 0.05, "{deg}° yaw {yaw}: slides at {a} vs {expected}");
                assert!(v.lin_vel_world().x < 0.0, "slides downhill");
            }
        }
    }
}

/// Skid steering: with the right track faster the rover circles left at a steady rate, on a
/// radius wider than the tracks' speeds alone give (the tracks slip); the outer track drives
/// and the inner one brakes. With opposite tracks it turns on the spot.
#[test]
fn skid_steers_in_a_circle_and_on_the_spot() {
    let w = World::flat();
    let mut v = rover(false);
    let input = DriveInput { throttle: 0.6, yaw: 0.3, ..Default::default() };
    w.run(&mut v, &input, 5.0);
    let r0 = v.ang_vel_body().z;
    w.run(&mut v, &input, 2.0);
    let (r, u) = (v.ang_vel_body().z, v.lin_vel_body().x);
    assert!(r > 0.3 && (r / r0 - 1.0).abs() < 0.01, "yaw rate {r0} → {r}");
    let (left, right) = (band(&v, 0), band(&v, 1));
    let tread = 2.0 * v.def().axles[0].position.y;
    let kinematic = 0.5 * tread * (right + left) / (right - left);
    let radius = u / r;
    assert!(radius > kinematic && radius < 3.0 * kinematic, "radius {radius} vs kinematic {kinematic}");
    let fx = |side: usize| v.wheels().enumerate().filter(|(w, _)| w % 2 == side).map(|(_, s)| s.tire.fx).sum::<f64>();
    assert!(fx(1) > 0.0 && fx(0) < 0.0, "outer {} and inner {} track forces", fx(1), fx(0));

    let spot = DriveInput { yaw: 1.0, ..Default::default() };
    w.run(&mut v, &spot, 3.0);
    let p = v.position();
    w.run(&mut v, &spot, 2.0);
    assert!(v.ang_vel_body().z > 0.5, "spot turn at {}", v.ang_vel_body().z);
    assert!((v.position() - p).length() < 0.05, "drifted {} m while turning", (v.position() - p).length());
}

/// A side's patches in steady slip `i` (band speed against ground speed) sum to
/// Janosi–Hanamoto's integral over the track's contact length `ℓ`,
/// `μF_z·(1 − K/(iℓ)·(1 − e^(−iℓ/K)))`, driving and braking.
#[test]
fn steady_shear_matches_janosi_hanamoto() {
    let d = presets::wheeled("rover_tracked").unwrap();
    let tire = d.tire(0).clone();
    let p = patch(&rover(false));
    let terrain = FlatTerrain::new(0.0, MaterialId::ASPHALT);
    let (fz, n, vx) = (100.0, 4, 2.0);
    let rho = fz / p.vertical_stiffness;
    let l = n as f64 * p.length;
    for slip in [0.002, 0.01, 0.05, 0.2, -0.01, -0.1] {
        let band = vx / (1.0 - slip);
        let mut states = vec![tire.initial_state(); n];
        let mut total = 0.0;
        for _ in 0..(3.0 / DT) as usize {
            for k in 0..n {
                states[k].inflow =
                    [k.checked_sub(1).map(|f| states[f].shear[TRACK_CELLS - 1]), states.get(k + 1).map(|r| r.shear[0])];
            }
            total = 0.0;
            for (k, state) in states.iter_mut().enumerate() {
                let motion = tire::WheelMotion {
                    center: DVec3::new(-(k as f64) * p.length, 0.0, p.radius - rho),
                    axis: DVec3::Y,
                    velocity: DVec3::new(vx, 0.0, 0.0),
                    carrier_angvel: DVec3::ZERO,
                    spin: band / (p.radius - rho),
                };
                let contact = tire.contact(&terrain, &motion);
                let f = tire.step(state, contact.as_ref(), &motion, tire::Surface::REFERENCE, DT);
                assert!((f.fz - fz).abs() < 1e-6);
                total += f.fx;
            }
        }
        let k = p.shear_modulus;
        let i = slip.abs();
        let integral = slip.signum() * p.mu * n as f64 * fz * (1.0 - k / (i * l) * (1.0 - (-i * l / k).exp()));
        assert!((total / integral - 1.0).abs() < 0.01, "slip {slip}: {total} vs {integral}");
    }
}

fn m113() -> serde_json::Value {
    let path = format!("{}/../../fixtures/chrono/tracked_m113.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The M113-based preset has Chrono's mass, and its static solution Chrono's pose and ground
/// loads (fixtures/chrono/tracked_m113.json): the band lifts the end road wheels clear of the
/// ground, the middle three carry the weight in Chrono's shares. Parked, it settles there.
#[test]
fn apc_rests_at_chronos_static_pose_and_loads() {
    let f = m113();
    let d = presets::wheeled("tracked_apc").unwrap();
    assert!((d.total_mass() / f["design"]["total_mass"].as_f64().unwrap() - 1.0).abs() < 1e-4);
    let st = d.static_state(G).unwrap();
    let s = &f["static"];
    let (height, pitch) = (s["height"].as_f64().unwrap(), s["pitch"].as_f64().unwrap());
    assert!((st.height - height).abs() < 0.01, "height {} vs Chrono {height}", st.height);
    assert!((st.pitch - pitch).abs() < 0.005, "pitch {} vs Chrono {pitch}", st.pitch);
    let ground: Vec<f64> = s["wheels"].as_array().unwrap().iter().map(|w| w["ground_load"].as_f64().unwrap()).collect();
    let chrono_total: f64 = ground.iter().sum();
    for a in 0..5 {
        let ours = (st.loads[2 * a] + st.loads[2 * a + 1]) / (d.total_mass() * G);
        let chrono = (ground[a] + ground[5 + a]) / chrono_total;
        assert!((ours - chrono).abs() < 0.03, "road wheel {}: load share {ours:.3} vs Chrono {chrono:.3}", a + 1);
    }

    let mut v = Wheeled::new(Arc::new(d), DT);
    World::flat().run(&mut v, &DriveInput { parking: true, ..Default::default() }, 3.0);
    assert!(v.lin_vel_world().length() < 1e-3 && v.contacts().is_empty());
    assert!((v.position().z - st.height).abs() < 1e-3, "height {} vs {}", v.position().z, st.height);
    for (k, wh) in v.wheels().enumerate() {
        assert!((wh.tire.fz - st.loads[k]).abs() < 0.01 * 18500.0, "wheel {k}: {} vs {}", wh.tire.fz, st.loads[k]);
    }
}

/// A sample of an APC run, as the Chrono fixture's.
#[derive(Debug)]
struct Sample {
    t: f64,
    speed: f64,
    yaw_rate: f64,
    gear: i32,
    /// Band speeds, left and right (m/s).
    bands: [f64; 2],
    /// Along the slope, uphill positive (m).
    displacement: f64,
}

/// The APC on a grade (uphill along +x), starting at `speed`: settled braked for `settle` s,
/// then driven by `input(t, vehicle)` for `seconds`, sampled every `every` s.
fn apc_run(
    grade: f64,
    speed: f64,
    settle: f64,
    seconds: f64,
    every: f64,
    mut input: impl FnMut(f64, &Wheeled) -> DriveInput,
) -> Vec<Sample> {
    let angle = grade.atan();
    let w = World::new(PlaneTerrain::incline(angle, MaterialId::ASPHALT));
    let mut v = Wheeled::new(Arc::new(presets::wheeled("tracked_apc").unwrap()), DT);
    let rest = v.rest(DVec3::ZERO, 0.0, speed);
    let tilt = DQuat::from_rotation_y(-angle);
    v.reset(&WheeledInit { pose: Pose::new(tilt * rest.pose.pos, tilt * rest.pose.rot), ..rest });
    w.run(&mut v, &DriveInput { brake: 1.0, ..Default::default() }, settle);
    let x0 = v.position();
    let slope = tilt * DVec3::X;
    let mut out = Vec::new();
    let (n, k) = ((seconds / DT).round() as usize, (every / DT).round() as usize);
    for i in 1..=n {
        let u = input((i - 1) as f64 * DT, &v);
        w.run(&mut v, &u, DT);
        if i % k == 0 {
            out.push(Sample {
                t: i as f64 * DT,
                speed: v.lin_vel_body().x,
                yaw_rate: v.ang_vel_body().z,
                gear: v.powertrain().gear,
                bands: [band(&v, 0), band(&v, 1)],
                displacement: (v.position() - x0).dot(slope),
            });
        }
    }
    out
}

/// A Chrono run's samples.
fn chrono_run(name: &str) -> (serde_json::Value, Vec<Sample>) {
    let f = m113()[name].clone();
    let samples = f["samples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let x = |k: &str| s[k].as_f64().unwrap();
            let sp = s["sprockets"].as_array().unwrap();
            let d: Vec<f64> = s["displacement"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            Sample {
                t: x("t"),
                speed: x("speed"),
                yaw_rate: x("yaw_rate"),
                gear: s["gear"].as_i64().unwrap() as i32,
                // Sprocket pitch radius.
                bands: [sp[0].as_f64().unwrap() * 0.245, sp[1].as_f64().unwrap() * 0.245],
                // Chrono tilts gravity, not the ground: x is along the slope.
                displacement: d[0],
            }
        })
        .collect();
    (f, samples)
}

/// Our run of the Chrono fixture's run `name`, with the same inputs, and Chrono's.
fn same_run(
    name: &str,
    input: impl FnMut(f64, &Wheeled) -> DriveInput,
) -> (serde_json::Value, Vec<Sample>, Vec<Sample>) {
    let (f, chrono) = chrono_run(name);
    let x = |k: &str| f[k].as_f64().unwrap();
    // Chrono's turns start faster (its tracks start at rest, and drag the hull down), and
    // settle at the held speed before the steering.
    let speed = if f.get("launch").is_some() { x("speed") } else { 0.0 };
    let grade = f["grade"].as_f64().unwrap_or(0.0);
    let ours = apc_run(grade, speed, x("settle"), chrono.last().unwrap().t, x("sample"), input);
    assert_eq!(ours.len(), chrono.len());
    (f, ours, chrono)
}

/// Mean of `f` over the samples in `[from, to]` s.
fn mean(s: &[Sample], from: f64, to: f64, f: impl Fn(&Sample) -> f64) -> f64 {
    let v: Vec<f64> = s.iter().filter(|s| s.t >= from - 1e-9 && s.t <= to + 1e-9).map(f).collect();
    v.iter().sum::<f64>() / v.len() as f64
}

/// The Chrono runs are at 600 solver iterations: at Chrono's usual 150, its loose contact and
/// brake constraints show as extra running resistance, brake creep and slip (see the fixture
/// generator), and 600 only reduce these.
///
/// Full throttle from rest on flat ground against Chrono's M113 with our engine map: the same
/// launch (the first 0.5 s, in which Chrono's running gear resists with `f ≈ 0.045`, the
/// preset's). Beyond it Chrono's resistance grows with speed, to about 0.115·W at 2 m/s
/// (0.14·W at 150 iterations), far above measured tracked vehicles' (Wong, §6.3: the running
/// gear's `(133 + 2.5·V[km/h])` N/t, ≈ 0.015–0.03·W): its rigid shoes' impacts on the NSC
/// contacts dissipate at every road wheel and sprocket tooth. After 4 s it runs at 2.8 m/s,
/// ours at 4.5 m/s.
#[test]
fn apc_launches_like_chronos_m113() {
    let full = DriveInput { throttle: 1.0, ..Default::default() };
    let (_, ours, chrono) = same_run("accel", |_, _| full);
    let a = |s: &[Sample]| s[1].speed / s[1].t;
    assert!((a(&ours) / a(&chrono) - 1.0).abs() < 0.15, "launch {} vs Chrono {} m/s²", a(&ours), a(&chrono));
    assert!(ours.iter().zip(&chrono).all(|(o, c)| o.speed > c.speed - 0.05));
}

/// At full throttle on a 15 % grade both crawl at the limit of their gradeability: the launch
/// thrust is 0.186·W against `sin θ + f·cos θ = 0.19`. Ours creeps at 7 cm/s, where the
/// internal resistance fades towards standstill; Chrono's at 15 cm/s (at 150 iterations it
/// stalled), its resistance at a crawl a little under the preset's 0.045.
#[test]
fn apc_crawls_up_15_percent_like_chronos_m113() {
    let full = DriveInput { throttle: 1.0, ..Default::default() };
    let (f, ours, chrono) = same_run("climb", |_, _| full);
    assert_eq!(f["grade"].as_f64().unwrap(), 0.15);
    for (s, who) in [(&ours, "ours"), (&chrono, "Chrono")] {
        let v = mean(s, 4.0, 6.0, |s| s.speed);
        assert!(v > 0.0 && v < 0.25, "{who} climbs at {v} m/s");
        assert!(s.iter().all(|s| s.displacement > -0.05), "{who} rolls back");
    }
}

/// Braked on a 30 % grade: the brakes' 10 kN·m per sprocket are 2.5 times the grade's torque,
/// and the tracks' friction 0.8 far above the grade. Ours holds; Chrono's creeps down at
/// 2 cm/s (5 cm/s at 150 iterations), its sprockets turning with it: its iterative solver
/// leaves the brakes' constraints loose under the shoes' chatter.
#[test]
fn apc_holds_on_30_percent_like_chronos_m113() {
    let (f, ours, chrono) = same_run("hold", |_, _| DriveInput { brake: 1.0, ..Default::default() });
    assert_eq!(f["grade"].as_f64().unwrap(), 0.3);
    let crept = |s: &[Sample]| s.last().unwrap().displacement - s[3].displacement;
    assert!(crept(&ours).abs() < 0.005, "ours crept {} m", crept(&ours));
    let creep = crept(&chrono) / (chrono.last().unwrap().t - chrono[3].t);
    let sprockets = mean(&chrono, 1.0, 4.0, |s| 0.5 * (s.bands[0] + s.bands[1]));
    assert!(
        creep < 0.0 && creep > -0.1 && (sprockets / creep - 1.0).abs() < 0.5,
        "Chrono creep {creep}, sprockets {sprockets}"
    );
}

/// The Chrono fixture's turn: speed held by its PI throttle, steering from `turn_at`.
fn held_turn(f: &serde_json::Value) -> impl FnMut(f64, &Wheeled) -> DriveInput {
    let x = |k: &str| f[k].as_f64().unwrap();
    let (speed, kp, ki, turn_at, steering) = (x("speed"), x("kp"), x("ki"), x("turn_at"), x("steering"));
    let (mut i, mut last) = (0.0, 0.0);
    move |t, v| {
        let e = speed - v.lin_vel_body().x;
        let raw = kp * e + i;
        if (0.0..1.0).contains(&raw) && raw > 0.0 || (raw >= 1.0) != (e > 0.0) {
            i = f64::clamp(i + ki * e * (t - last), 0.0, 1.0);
        }
        last = t;
        let steering = if t >= turn_at { steering } else { 0.0 };
        DriveInput { throttle: raw.clamp(0.0, 1.0), steering, ..Default::default() }
    }
}

/// Brake steering against Chrono's M113: from 1.2 m/s, the speed held by the same PI throttle,
/// steering from 1 s. Both lose most of their speed to the braked inner track at once. With
/// steering 0.3 they then turn at the same order of yaw rate (ours 0.008, Chrono 0.010 rad/s),
/// and with 0.6 both stall into a slow pivot about the braked track. Closer agreement is not
/// to be had from Chrono here: its turning moves with the solver's convergence (the pivot's
/// yaw rate halves from 150 to 600 iterations, the lighter turn's early response too), and
/// its speed with its running resistance. The turning physics is checked against Wong and
/// Chiang's theory instead (`pivot_turn_follows_wong_and_chiang`).
#[test]
fn apc_brake_steers_like_chronos_m113() {
    let light = m113()["turn"].clone();
    let (_, ours, chrono) = same_run("turn", held_turn(&light));
    let (a, b) = (mean(&ours, 2.5, 5.0, |s| s.yaw_rate), mean(&chrono, 2.5, 5.0, |s| s.yaw_rate));
    assert!(a > 0.0 && b > 0.0 && a / b > 0.5 && a / b < 2.0, "turning at {a} vs Chrono {b} rad/s");

    let hard = m113()["turn_hard"].clone();
    let (_, ours, chrono) = same_run("turn_hard", held_turn(&hard));
    for (s, who) in [(&ours, "ours"), (&chrono, "Chrono")] {
        let (v, w) = (mean(s, 2.5, 5.0, |s| s.speed), mean(s, 2.5, 5.0, |s| s.yaw_rate));
        assert!(v.abs() < 0.15 && w > 0.0 && w < 0.05, "{who} at {v} m/s, {w} rad/s");
    }
}

/// Gradeability on rigid ground: at full throttle in first gear, the APC's thrust is the
/// engine's launch torque through the gearbox, `F = T/(g₁·i_f·r)`, against the grade and the
/// running gear's internal resistance `f·W·cos θ`, accelerating the vehicle and its spinning
/// parts `m_eff = m + ΣJ/r²`. It climbs grades up to `sin θ + f cos θ = F/W`, and stalls above
/// (creeping at a few cm/s, where the internal resistance fades towards standstill).
#[test]
fn apc_climbs_up_to_its_gradeability() {
    let d = presets::wheeled("tracked_apc").unwrap();
    let TireModel::Track(p) = &d.tire(0).model else { panic!("track expected") };
    let PowertrainDef::Combustion(c) = &d.powertrain else { panic!("combustion expected") };
    let r = p.radius - p.nominal_load / p.vertical_stiffness;
    let thrust = c.engine.full_throttle.eval(0.0) / (c.gearbox.forward[0] * c.final_drive * r);
    let weight = d.total_mass() * G;
    let spin: f64 = d.axles.iter().map(|a| 2.0 * a.wheel.inertia.y).sum::<f64>() + c.driveline_inertia;
    let m_eff = d.total_mass() + spin / (r * r);
    let f = p.rolling_resistance;
    // sin θ + f cos θ = F/W.
    let limit = (thrust / weight / f.hypot(1.0)).asin() - f.atan();
    let full = DriveInput { throttle: 1.0, ..Default::default() };
    for angle in [0.0, 0.5 * limit, 0.9 * limit] {
        let s = apc_run(angle.tan(), 0.0, 1.0, 3.0, 0.5, |_, _| full);
        assert!(s.iter().all(|s| s.gear == 1));
        let a = (s[5].speed - s[0].speed) / 2.5;
        let expected = (thrust - weight * (angle.sin() + f * angle.cos())) / m_eff;
        assert!((a - expected).abs() < 0.05 * expected + 0.005, "grade {:.3}: {a} vs {expected} m/s²", angle.tan());
    }
    let s = apc_run((1.1 * limit).tan(), 0.0, 1.0, 3.0, 0.5, |_, _| full);
    assert!(s.iter().all(|s| s.speed < 0.1), "climbs {:.3} beyond its gradeability {:.3}", s[5].speed, limit.tan());
}

/// Wong and Chiang's theory of skid steering on firm ground (Wong, *Theory of Ground
/// Vehicles*, §7.3.3): each track element carries `μp` against its sliding velocity, the
/// shear being well developed. In a pivot turn at yaw rate `ω` with band speeds `±V`, the
/// elements at `(x, ±B/2)` slide at `(ω·B/2 − V, ω·x)` (outer track); per track of length `ℓ`
/// and load `W/2` the thrust is `μp·∫ s/√(s² + ω²x²) dx` and the turning resistance moment
/// `μp·∫ ωx²/√(s² + ω²x²) dx`, `s = V − ω·B/2`, `p = W/(2ℓ)`: less than the classic `μWℓ/4`
/// (no longitudinal slip), which the rover's motors cannot reach. At steady yaw the tracks'
/// thrusts balance the resistance.
#[test]
fn pivot_turn_follows_wong_and_chiang() {
    let w = World::flat();
    let mut v = rover(false);
    w.run(&mut v, &DriveInput { yaw: 1.0, ..Default::default() }, 4.0);
    let r = v.ang_vel_body().z;
    let fx = |side: usize| v.wheels().enumerate().filter(|(w, _)| w % 2 == side).map(|(_, s)| s.tire.fx).sum::<f64>();
    let (tread, p) = (2.0 * v.def().axles[0].position.y, patch(&v));
    let l = 4.0 * p.length;
    let weight = v.def().total_mass() * G;
    let mu = p.mu * MaterialTable::standard().get(MaterialId::ASPHALT).friction / 0.8;
    let slip = 0.5 * (band(&v, 1) - band(&v, 0)) - 0.5 * r * tread;
    assert!(slip > 0.0);
    let mup = mu * weight / (2.0 * l);
    let n = 10_000;
    let (mut thrust, mut moment) = (0.0, 0.0);
    for i in 0..n {
        let x = -0.5 * l + (i as f64 + 0.5) * l / n as f64;
        let speed = slip.hypot(r * x);
        thrust += mup * slip / speed * l / n as f64;
        moment += 2.0 * mup * r * x * x / speed * l / n as f64;
    }
    assert!((fx(1) / thrust - 1.0).abs() < 0.03, "outer thrust {} vs {thrust}", fx(1));
    assert!((-fx(0) / thrust - 1.0).abs() < 0.03, "inner thrust {} vs {thrust}", -fx(0));
    let resistance = 0.5 * tread * (fx(1) - fx(0));
    assert!((resistance / moment - 1.0).abs() < 0.05, "turning moment {resistance} vs {moment}");
    assert!(moment < mu * weight * l / 4.0);
}
