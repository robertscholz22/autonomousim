//! The `c172_like` preset against JSBSim's c172p (fixtures/jsbsim/c172.json, from
//! tools/gen_jsbsim_fixtures.py): trim over the airspeed sweep, the linear modes at 90 kt, an
//! elevator doublet and the takeoff roll.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::aero::AirData;
use autonomousim_vehicles::fixedwing::*;
use autonomousim_vehicles::multirotor::{GroundPlane, StepEnv};
use autonomousim_vehicles::presets;
use glam::{DQuat, DVec3};
use serde_json::Value;
use std::sync::Arc;

const G: f64 = 9.80665;
const GRAVITY: DVec3 = DVec3::new(0.0, 0.0, -G);
const KT: f64 = 0.514444;

fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/jsbsim/c172.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn c172(dt: f64) -> FixedWing {
    FixedWing::new(Arc::new(presets::fixed_wing("c172_like").unwrap()), dt)
}

fn air(density: f64) -> AirData {
    AirData { density, ..AirData::default() }
}

fn free_env(density: f64) -> StepEnv<'static> {
    StepEnv { scene: None, gravity: GRAVITY, air: air(density), ground: None }
}

/// Trim at the fixture's 90 kt row (airspeed, density).
fn trim_90kt(f: &Value, a: &FixedWing) -> (Trim, f64) {
    let m = &f["modes"];
    let v = m["airspeed"].as_f64().unwrap();
    let rho = f["trim"].as_array().unwrap().iter().find(|r| (r["airspeed"].as_f64().unwrap() - v).abs() < 0.1).unwrap()
        ["density"]
        .as_f64()
        .unwrap();
    (a.trim(v, rho, 0.0, 0.0, G).unwrap(), rho)
}

fn start(a: &mut FixedWing, t: &Trim, rot: DQuat, v_body: DVec3, w_body: DVec3, rotor: f64) {
    a.reset(&FixedWingInit {
        pose: Pose::new(DVec3::new(0.0, 0.0, 300.0), rot),
        lin_vel_world: rot * v_body,
        ang_vel_body: w_body,
        controls: t.controls,
        rotor_speed: Some(rotor),
        soc: 1.0,
    });
}

/// Trim α and elevator within 0.5° (or 5 %) and throttle within 5 % of JSBSim; the engine speed
/// within 2 %.
#[test]
fn trim_sweep() {
    let f = fixture();
    let a = c172(0.002);
    for row in f["trim"].as_array().unwrap() {
        let v = |k: &str| row[k].as_f64().unwrap();
        let t = a.trim(v("airspeed"), v("density"), 0.0, 0.0, G).unwrap();
        let rpm = t.rotor_speed * 60.0 / std::f64::consts::TAU;
        let msg = format!(
            "V {:.1}: alpha {:.4} vs {:.4}, elevator {:.4} vs {:.4}, throttle {:.3} vs {:.3}, rpm {:.0} vs {:.0}",
            v("airspeed"),
            t.alpha,
            v("alpha"),
            t.surfaces[1],
            v("elevator"),
            t.controls.throttle,
            v("throttle"),
            rpm,
            v("rpm")
        );
        eprintln!("{msg}");
        let close = |ours: f64, theirs: f64| (ours - theirs).abs() < 0.5f64.to_radians().max(0.05 * theirs.abs());
        assert!(close(t.alpha, v("alpha")) && close(t.surfaces[1], v("elevator")), "{msg}");
        assert!((t.controls.throttle - v("throttle")).abs() < 0.05 * v("throttle"), "{msg}");
        assert!((rpm - v("rpm")).abs() < 0.02 * v("rpm"), "{msg}");
    }
}

// ------------------------------------------------------------------ linear modes

#[derive(Clone, Copy, Debug)]
struct C(f64, f64);

impl C {
    fn add(self, o: C) -> C {
        C(self.0 + o.0, self.1 + o.1)
    }
    fn sub(self, o: C) -> C {
        C(self.0 - o.0, self.1 - o.1)
    }
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    fn div(self, o: C) -> C {
        let d = o.0 * o.0 + o.1 * o.1;
        C((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d)
    }
    fn abs(self) -> f64 {
        self.0.hypot(self.1)
    }
}

/// Eigenvalues of a small real matrix: the characteristic polynomial (Faddeev–LeVerrier), then
/// its roots (Durand–Kerner).
fn eigenvalues(a: &[Vec<f64>]) -> Vec<C> {
    let n = a.len();
    let matmul = |x: &[Vec<f64>], y: &[Vec<f64>]| -> Vec<Vec<f64>> {
        (0..n).map(|i| (0..n).map(|j| (0..n).map(|k| x[i][k] * y[k][j]).sum()).collect()).collect()
    };
    // p(λ) = λⁿ + c[1] λⁿ⁻¹ + … + c[n].
    let mut c = vec![1.0; n + 1];
    let mut m = vec![vec![0.0; n]; n];
    for k in 1..=n {
        let mut am = matmul(a, &m);
        for (i, row) in am.iter_mut().enumerate() {
            row[i] += c[k - 1];
        }
        m = am;
        let trace: f64 = (0..n).map(|i| matmul(a, &m)[i][i]).sum();
        c[k] = -trace / k as f64;
    }
    let p = |z: C| c.iter().fold(C(0.0, 0.0), |acc, &ck| acc.mul(z).add(C(ck, 0.0)));
    let scale = 1.0 + c.iter().skip(1).fold(0.0f64, |m, x| m.max(x.abs()));
    let mut roots: Vec<C> = (0..n)
        .map(|k| {
            let phi = 0.4 + std::f64::consts::TAU * k as f64 / n as f64;
            C(scale * phi.cos(), scale * phi.sin())
        })
        .collect();
    for _ in 0..2000 {
        for i in 0..n {
            let den = (0..n).filter(|&j| j != i).fold(C(1.0, 0.0), |d, j| d.mul(roots[i].sub(roots[j])));
            roots[i] = roots[i].sub(p(roots[i]).div(den));
        }
    }
    roots
}

/// Oscillatory modes (ω_n, ζ) by increasing frequency, and the real roots by magnitude.
fn modes(eig: &[C]) -> (Vec<(f64, f64)>, Vec<f64>) {
    let mut osc: Vec<(f64, f64)> = eig.iter().filter(|e| e.1 > 1e-6).map(|e| (e.abs(), -e.0 / e.abs())).collect();
    osc.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut real: Vec<f64> = eig.iter().filter(|e| e.1.abs() <= 1e-6).map(|e| e.0).collect();
    real.sort_by(|x, y| x.abs().total_cmp(&y.abs()));
    (osc, real)
}

/// Longitudinal and lateral Jacobians about the trim, by central differences of the simulated
/// state derivative. The state is (body velocity, body rates, attitude error δ with
/// rot = rot₀·exp(δ), rotor speed); δ̇ = ω about the trim. Each derivative comes from the
/// second of two tiny steps, so the α̇ terms see the actual rate of change of α.
fn jacobians(t: &Trim, rho: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    const DT: f64 = 1e-5;
    let mut a = c172(DT);
    let rot0 = t.attitude(0.0);
    let deriv = |a: &mut FixedWing, x: &[f64; 10]| -> [f64; 10] {
        let (v, w, d) = (DVec3::new(x[0], x[1], x[2]), DVec3::new(x[3], x[4], x[5]), DVec3::new(x[6], x[7], x[8]));
        start(a, t, rot0 * DQuat::from_scaled_axis(d), v, w, x[9]);
        a.step(&t.controls, &free_env(rho)).unwrap();
        let (v1, w1, r1) = (a.lin_vel_body(), a.ang_vel_body(), a.rotor_speed());
        a.step(&t.controls, &free_env(rho)).unwrap();
        let dv = (a.lin_vel_body() - v1) / DT;
        let dw = (a.ang_vel_body() - w1) / DT;
        [dv.x, dv.y, dv.z, dw.x, dw.y, dw.z, w.x, w.y, w.z, (a.rotor_speed() - r1) / DT]
    };
    let v0 = t.velocity_body();
    let x0 = [v0.x, v0.y, v0.z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, t.rotor_speed];
    let steps = [0.05, 0.05, 0.05, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0.5];
    let mut jac = [[0.0; 10]; 10];
    for j in 0..10 {
        let (mut xp, mut xm) = (x0, x0);
        xp[j] += steps[j];
        xm[j] -= steps[j];
        let (fp, fm) = (deriv(&mut a, &xp), deriv(&mut a, &xm));
        for i in 0..10 {
            jac[i][j] = (fp[i] - fm[i]) / (2.0 * steps[j]);
        }
    }
    let block = |idx: &[usize]| idx.iter().map(|&i| idx.iter().map(|&j| jac[i][j]).collect()).collect();
    // FLU: u, w, pitch rate, pitch error, rotor | v, roll rate, yaw rate, roll and yaw error.
    (block(&[0, 2, 4, 7, 9]), block(&[1, 3, 5, 6, 8]))
}

/// Phugoid, short period and Dutch roll within 10 % in frequency and damping ratio of JSBSim's
/// linear model at 90 kt; the roll subsidence within 10 % and a slow spiral mode.
#[test]
fn linear_modes() {
    let f = fixture();
    let m = &f["modes"];
    let (t, rho) = trim_90kt(&f, &c172(0.002));
    let (lon, lat) = jacobians(&t, rho);
    let (lon_osc, _) = modes(&eigenvalues(&lon));
    let (lat_osc, lat_real) = modes(&eigenvalues(&lat));
    eprintln!("lon {lon_osc:?}\nlat {lat_osc:?} {lat_real:?}");
    let check = |name: &str, (wn, zeta): (f64, f64)| {
        let (wn_j, zeta_j) = (m[name]["wn"].as_f64().unwrap(), m[name]["zeta"].as_f64().unwrap());
        eprintln!("{name}: wn {wn:.4} vs {wn_j:.4}, zeta {zeta:.4} vs {zeta_j:.4}");
        assert!((wn - wn_j).abs() < 0.1 * wn_j && (zeta - zeta_j).abs() < 0.1 * zeta_j, "{name}");
    };
    assert_eq!((lon_osc.len(), lat_osc.len()), (2, 1));
    check("phugoid", lon_osc[0]);
    check("short_period", lon_osc[1]);
    check("dutch_roll", lat_osc[0]);
    // Real lateral roots: heading (neutral), spiral, roll subsidence.
    assert_eq!(lat_real.len(), 3);
    assert!(lat_real[0].abs() < 1e-4, "heading {}", lat_real[0]);
    let (spiral, roll) = (lat_real[1], lat_real[2]);
    let (spiral_j, roll_j) = (m["spiral"]["lambda"].as_f64().unwrap(), m["roll"]["lambda"].as_f64().unwrap());
    eprintln!("spiral {spiral:.4} vs {spiral_j:.4}, roll {roll:.3} vs {roll_j:.3}");
    assert!((roll - roll_j).abs() < 0.1 * roll_j.abs(), "roll");
    // The spiral root is a small difference of large terms: only its size is compared.
    assert!(spiral.abs() < 0.1 && (spiral - spiral_j).abs() < 0.05, "spiral");
}

// ------------------------------------------------------------------ doublet

/// JSBSim's elevator doublet, replayed as deflections about our 90 kt trim: pitch rate,
/// pitch and α follow within 5 % RMS (20 % at worst) of the peak response.
#[test]
fn elevator_doublet() {
    const DT: f64 = 0.002;
    let f = fixture();
    let d = &f["doublet"];
    let cols: Vec<&str> = d["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let col = |name: &str| cols.iter().position(|c| *c == name).unwrap();
    let rows: Vec<Vec<f64>> = d["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())
        .collect();
    // JSBSim's c172p moves its surfaces instantly: replay the deflections without our servo.
    let mut def = presets::fixed_wing("c172_like").unwrap();
    def.controls.elevator.rate = 0.0;
    def.controls.elevator.tau = 0.0;
    let mut a = FixedWing::new(Arc::new(def), DT);
    let (t, rho) = trim_90kt(&f, &a);
    start(&mut a, &t, t.attitude(0.0), t.velocity_body(), DVec3::ZERO, t.rotor_speed);
    // The normalised elevator giving a deflection (bisection; the map is monotonic).
    let command = |a: &FixedWing, deflection: f64| {
        let mut c = t.controls;
        let sign = if a.targets(&FixedWingInput { elevator: 1.0, ..c })[1] > a.targets(&c)[1] { 1.0 } else { -1.0 };
        let (mut lo, mut hi) = (-1.0, 1.0);
        for _ in 0..50 {
            c.elevator = 0.5 * (lo + hi);
            if sign * (a.targets(&c)[1] - deflection) > 0.0 {
                hi = c.elevator;
            } else {
                lo = c.elevator;
            }
        }
        c
    };
    let (theta0, alpha0) = (t.pitch, t.alpha);
    let (theta0_j, alpha0_j, elevator0_j) = (rows[0][col("theta")], rows[0][col("alpha")], rows[0][col("elevator")]);
    let (mut sim_t, mut k) = (0.0, 0);
    // Per signal: maximum and RMS error, and JSBSim's peak.
    let mut stats = [[0.0f64; 3]; 3];
    for row in &rows {
        // Hold JSBSim's deflection over the interval up to this sample.
        let c = command(&a, t.surfaces[1] + row[col("elevator")] - elevator0_j);
        while sim_t < row[col("time")] - 1e-9 {
            a.step(&c, &free_env(rho)).unwrap();
            k += 1;
            sim_t = k as f64 * DT;
        }
        let q = -a.ang_vel_body().y;
        let theta = (a.orientation() * DVec3::X).z.asin() - theta0;
        let alpha = a.flow().alpha - alpha0;
        let (q_j, theta_j, alpha_j) = (row[col("q")], row[col("theta")] - theta0_j, row[col("alpha")] - alpha0_j);
        for (s, (ours, theirs)) in stats.iter_mut().zip([(q, q_j), (theta, theta_j), (alpha, alpha_j)]) {
            s[0] = s[0].max((ours - theirs).abs());
            s[1] += (ours - theirs).powi(2) / rows.len() as f64;
            s[2] = s[2].max(theirs.abs());
        }
    }
    for (name, [max, ms, peak]) in ["q", "theta", "alpha"].into_iter().zip(stats) {
        let rms = ms.sqrt();
        eprintln!("{name}: max error {max:.4}, rms {rms:.4}, peak {peak:.4}");
        // The largest errors sit at the steps, where the sample phases differ.
        assert!(max < 0.2 * peak && rms < 0.05 * peak, "{name}");
    }
}

// ------------------------------------------------------------------ takeoff

/// Full throttle against the brakes, then the ground roll to 55 kt at sea level: static engine
/// speed and thrust, distance and time within 10 % of JSBSim.
#[test]
fn takeoff_roll() {
    const DT: f64 = 0.002;
    let f = fixture();
    let to = &f["takeoff"];
    let ground = FlatTerrain::new(0.0, MaterialId(0));
    let materials = MaterialTable::standard();
    let scene = StaticScene { terrain: &ground, obstacles: &NoObstacles, materials: &materials };
    let mut a = c172(DT);
    let (rot, h) = a.def().resting_pose(G).unwrap();
    a.reset(&FixedWingInit::at_rest(Pose::new(DVec3::new(0.0, 0.0, h), rot)));
    let step = |a: &mut FixedWing, brake: f64| {
        let c = FixedWingInput { throttle: 1.0, brake, ..Default::default() };
        let pos = a.position();
        let plane = GroundPlane { point: DVec3::new(pos.x, pos.y, 0.0), normal: DVec3::Z };
        let env = StepEnv { scene: Some(scene), gravity: GRAVITY, air: AirData::default(), ground: Some(plane) };
        a.step(&c, &env).unwrap();
    };
    for _ in 0..(10.0 / DT) as usize {
        step(&mut a, 1.0);
    }
    let rpm = a.rotor_speed() * 60.0 / std::f64::consts::TAU;
    let thrust = a.propulsion_output().thrust;
    let (rpm_j, thrust_j) = (to["static"]["rpm"].as_f64().unwrap(), to["static"]["thrust"].as_f64().unwrap());
    eprintln!("static: rpm {rpm:.0} vs {rpm_j:.0}, thrust {thrust:.0} vs {thrust_j:.0}");
    assert!((rpm - rpm_j).abs() < 0.03 * rpm_j && (thrust - thrust_j).abs() < 0.05 * thrust_j);
    let x0 = a.position();
    let mut time = 0.0;
    while a.flow().airspeed < 55.0 * KT {
        step(&mut a, 0.0);
        time += DT;
        assert!(time < 60.0, "too slow");
    }
    let distance = (a.position() - x0).truncate().length();
    let (d_j, t_j) = (to["distance_to_55kt"].as_f64().unwrap(), to["time_to_55kt"].as_f64().unwrap());
    eprintln!("to 55 kt: {distance:.1} m vs {d_j:.1} m, {time:.2} s vs {t_j:.2} s");
    assert!((distance - d_j).abs() < 0.1 * d_j && (time - t_j).abs() < 0.1 * t_j);
    assert!(a.weight_on_wheels());
}
