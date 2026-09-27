//! The Whipple bicycle benchmark (Meijaard, Papadopoulos, Ruina & Schwab 2007): the linearised
//! model's matrices, eigenvalues and self-stable speed range, and the full multibody model's
//! lean and steer dynamics against them.

use autonomousim_vehicles::ground::single_track::{Complex, WhippleMatrices, WhippleParams, eigenvalues4};
use autonomousim_vehicles::presets;
use glam::{DMat4, DVec4};

/// Gravity in the benchmark.
const G: f64 = 9.81;

/// The published matrices (Meijaard et al. 2007, eq. 5.4, to 1e-14).
fn published() -> WhippleMatrices {
    WhippleMatrices {
        m: [[80.81722, 2.31941332208709], [2.31941332208709, 0.29784188199686]],
        c1: [[0.0, 33.86641391492494], [-0.85035641456978, 1.68540397397560]],
        k0: [[-80.95, -2.59951685249872], [-2.59951685249872, -0.80329488458618]],
        k2: [[0.0, 76.59734589573222], [0.0, 2.65431523794604]],
    }
}

fn assert_matrices(got: &WhippleMatrices, want: &WhippleMatrices, tol: f64) {
    for (name, a, b) in
        [("M", got.m, want.m), ("C1", got.c1, want.c1), ("K0", got.k0, want.k0), ("K2", got.k2, want.k2)]
    {
        for i in 0..2 {
            for j in 0..2 {
                let err = (a[i][j] - b[i][j]).abs();
                assert!(err <= tol * b[i][j].abs().max(1.0), "{name}[{i}][{j}]: {} vs {}", a[i][j], b[i][j]);
            }
        }
    }
}

#[test]
fn benchmark_matrices_match_the_paper() {
    assert_matrices(&WhippleParams::benchmark().matrices(), &published(), 1e-12);
    // The preset, converted from the chassis frame.
    let def = presets::wheeled("bicycle_benchmark").unwrap();
    let params = WhippleParams::from_def(&def).unwrap();
    assert!((params.trail - 0.08).abs() < 1e-9, "trail {}", params.trail);
    assert_matrices(&params.matrices(), &published(), 1e-9);
}

#[test]
fn eigenvalues_and_the_stable_speed_range() {
    let m = published();
    // Weave and capsize speeds (Meijaard et al. 2007, §5).
    let (weave, capsize) = m.stable_speeds(G, 10.0).expect("a self-stable range");
    println!("weave {weave:.8} m/s, capsize {capsize:.8} m/s");
    assert!((weave - 4.29238253634111).abs() < 1e-8, "weave speed {weave}");
    assert!((capsize - 6.02426201538837).abs() < 1e-8, "capsize speed {capsize}");
    // Standing still: the inverted pendulum pair ±√(...), all real.
    let still = m.eigenvalues(0.0, G);
    println!("v = 0: {still:?}");
    for (got, want) in still.iter().zip([-5.53094371765393, -3.13164324790656, 3.13164324790656, 5.53094371765393]) {
        assert!((got.re - want).abs() < 1e-9 && got.im == 0.0, "{got:?} vs {want}");
    }
    // At 5 m/s (Table 2): castering, capsize and the weave pair.
    let five = m.eigenvalues(5.0, G);
    println!("v = 5: {five:?}");
    let want = [
        Complex::new(-14.07838969279822, 0.0),
        Complex::new(-0.77534188219585, -4.46486771378823),
        Complex::new(-0.77534188219585, 4.46486771378823),
        Complex::new(-0.32286642900409, 0.0),
    ];
    for (got, want) in five.iter().zip(want) {
        assert!((*got - want).abs() < 1e-9, "{got:?} vs {want:?}");
    }
}

#[test]
fn eigenvalues_of_a_general_matrix() {
    // A block-diagonal rotation-plus-scaling and two real roots.
    let a = DMat4::from_cols(
        DVec4::new(-1.0, 2.0, 0.0, 0.0),
        DVec4::new(-2.0, -1.0, 0.0, 0.0),
        DVec4::new(0.0, 0.0, 3.0, 0.0),
        DVec4::new(0.0, 0.0, 1.0, -7.0),
    );
    let e = eigenvalues4(&a);
    let want = [Complex::new(-7.0, 0.0), Complex::new(-1.0, -2.0), Complex::new(-1.0, 2.0), Complex::new(3.0, 0.0)];
    for (got, want) in e.iter().zip(want) {
        assert!((*got - want).abs() < 1e-12, "{e:?}");
    }
}

// ------------------------------------------------------------------ full multibody model

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::{DriveInput, GroundStepEnv, TireSpec, Wheeled, WheeledDef};
use autonomousim_vehicles::multirotor::AirData;
use glam::{DQuat, DVec3};
use std::sync::Arc;

/// The benchmark preset with its tyres' slip stiffnesses and relaxation lengths replaced.
fn with_tyres(stiffness: [f64; 2], relaxation: [f64; 2]) -> WheeledDef {
    let mut def = presets::wheeled("bicycle_benchmark").unwrap();
    for a in &mut def.axles {
        let Some(TireSpec::Fiala(p)) = &mut a.tire else { panic!("Fiala tyres") };
        [p.slip_stiffness, p.cornering_stiffness] = stiffness;
        [p.relaxation_x, p.relaxation_y] = relaxation;
    }
    def.finish().unwrap();
    def
}

/// Lean, steer angle (both positive to the right, as in the paper) and their rates.
fn lean_steer(v: &Wheeled) -> DVec4 {
    let z = v.orientation() * DVec3::Z;
    let (delta, rate) = v.steering_head().unwrap();
    DVec4::new((-z.y).atan2(z.z), -delta, v.ang_vel_body().x, -rate)
}

/// Eigenvalues of the lean and steer dynamics of the full model at speed `speed`, from four
/// runs perturbed in each coordinate: dynamic mode decomposition of the states averaged over
/// 5 ms intervals (which filters the tyres' ringing but keeps the slow modes' eigenvalues),
/// the first 20 ms of tyre transients skipped.
fn identify(def: &WheeledDef, speed: f64, dt: f64) -> [Complex; 4] {
    let terrain = FlatTerrain::new(0.0, MaterialId::ASPHALT);
    let materials = MaterialTable::standard();
    let env = GroundStepEnv {
        scene: StaticScene { terrain: &terrain, obstacles: &NoObstacles, materials: &materials },
        gravity: DVec3::new(0.0, 0.0, -G),
        air: AirData::default(),
    };
    let def = Arc::new(def.clone());
    let (sample, skip, length) = (0.005, 0.02, 1.0);
    let every = (sample / dt).round() as usize;
    let (mut xx, mut yx) = (DMat4::ZERO, DMat4::ZERO);
    let eps = 1e-5;
    for k in 0..4 {
        let mut v = Wheeled::new(def.clone(), dt);
        let mut init = v.rest(DVec3::ZERO, 0.0, speed);
        match k {
            0 => {
                let rot = DQuat::from_rotation_x(eps);
                init.pose = Pose::new(rot * init.pose.pos, rot * init.pose.rot);
            }
            2 => init.ang_vel_body.x = eps,
            _ => {}
        }
        v.reset(&init);
        let names: Vec<&str> = v.model().links().iter().map(|l| l.name.as_str()).collect();
        let head = names.iter().position(|n| *n == "head_1").unwrap();
        let (hq, hv) = (v.model().q_offset(head), v.model().v_offset(head));
        match k {
            1 => v.state.q[hq] = eps,
            3 => v.state.v[hv] = eps * 10.0,
            _ => {}
        }
        let mut prev: Option<DVec4> = None;
        let mut mean = DVec4::ZERO;
        let steps = ((skip + length) / dt).round() as usize;
        for i in 1..=steps {
            v.step(&DriveInput::default(), &env).unwrap();
            mean += lean_steer(&v) / (eps * every as f64);
            if i % every != 0 {
                continue;
            }
            let x = std::mem::take(&mut mean);
            if (i as f64) * dt < skip {
                continue;
            }
            if let Some(p) = prev {
                for c in 0..4 {
                    *xx.col_mut(c) += p * p[c];
                    *yx.col_mut(c) += x * p[c];
                }
            }
            prev = Some(x);
        }
    }
    let f = yx * xx.inverse();
    eigenvalues4(&f).map(|mu| {
        let l = mu.ln();
        Complex::new(l.re / sample, l.im / sample)
    })
}

/// Worst eigenvalue error of the full model over `speeds` against the linearised one, relative
/// to the eigenvalue's size (at least 1/s), and the speed where it occurs.
fn worst_error(def: &WheeledDef, dt: f64, speeds: &[f64]) -> (f64, f64) {
    let reference = WhippleParams::from_def(def).unwrap().matrices();
    let mut worst = (0.0f64, 0.0);
    for &speed in speeds {
        let got = identify(def, speed, dt);
        let want = reference.eigenvalues(speed, G);
        let show = |e: &[Complex; 4]| e.iter().map(|c| format!("{:8.3}{:+7.3}i", c.re, c.im)).collect::<String>();
        println!("{speed:4.1} m/s: {}\n      want {}", show(&got), show(&want));
        for w in want {
            let e = got.iter().map(|g| (*g - w).abs()).fold(f64::INFINITY, f64::min) / w.abs().max(1.0);
            if e > worst.0 {
                worst = (e, speed);
            }
        }
    }
    worst
}

/// Where the full model's largest real part changes sign between `a` and `b`.
fn crossing(def: &WheeledDef, dt: f64, mut a: f64, mut b: f64) -> f64 {
    let growth = |v: f64| identify(def, v, dt)[3].re;
    let rising = growth(a) < 0.0;
    assert_ne!(rising, growth(b) < 0.0, "no crossing in [{a}, {b}]");
    for _ in 0..10 {
        let m = 0.5 * (a + b);
        if (growth(m) < 0.0) == rising { a = m } else { b = m }
    }
    0.5 * (a + b)
}

#[test]
fn knife_edge_limit_matches_the_benchmark() {
    // Stiffer tyres with shorter lateral relaxation approach knife edges; the explicit
    // tyre springs then need 0.1 ms steps.
    let def = with_tyres([1e5, 1e6], [0.01, 0.002]);
    let dt = 1e-4;
    let speeds: Vec<f64> = (0..=20).map(|k| 0.5 * k as f64).collect();
    let (worst, at) = worst_error(&def, dt, &speeds);
    println!("worst eigenvalue error {:.2} % at {at} m/s", 100.0 * worst);
    assert!(worst < 0.01, "eigenvalues off by {:.2} % at {at} m/s", 100.0 * worst);
    let weave = crossing(&def, dt, 4.0, 4.6);
    let capsize = crossing(&def, dt, 5.6, 6.6);
    println!("weave speed {weave:.3} m/s, capsize speed {capsize:.3} m/s");
    assert!((weave / 4.29238253634111 - 1.0).abs() < 0.01, "weave speed {weave}");
    assert!((capsize / 6.02426201538837 - 1.0).abs() < 0.01, "capsize speed {capsize}");
}

#[test]
fn preset_at_one_kilohertz_is_close_to_the_benchmark() {
    // The preset's softer tyres run at the ground vehicles' 1 kHz; the castering mode is the
    // most affected, the more so at low speed, where the relaxation length is no longer short
    // against the travel in a castering time.
    let def = presets::wheeled("bicycle_benchmark").unwrap();
    let speeds: Vec<f64> = (1..=10).map(f64::from).collect();
    let (worst, at) = worst_error(&def, 1e-3, &speeds);
    println!("worst eigenvalue error {:.2} % at {at} m/s", 100.0 * worst);
    assert!(worst < 0.03, "eigenvalues off by {:.2} % at {at} m/s", 100.0 * worst);
    let weave = crossing(&def, 1e-3, 4.0, 4.6);
    let capsize = crossing(&def, 1e-3, 5.6, 6.6);
    println!("weave speed {weave:.3} m/s, capsize speed {capsize:.3} m/s");
    assert!((weave / 4.29238253634111 - 1.0).abs() < 0.01, "weave speed {weave}");
    assert!((capsize / 6.02426201538837 - 1.0).abs() < 0.01, "capsize speed {capsize}");
}
