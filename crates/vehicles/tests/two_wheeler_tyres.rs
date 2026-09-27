//! Tyres for two-wheelers: the motorcycle Magic Formula sets against Evangelou's (2004) data
//! and the physics of camber, the toroidal wheel in the force element, turn slip through the
//! wheel's yaw rate, and the rigid-rolling approximation of a knife edge.

use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::TireSpec;
use autonomousim_vehicles::ground::tire::{McParams, Surface, Tire, TireForces, TireModel, WheelMotion};
use glam::{DQuat, DVec3};
use std::path::PathBuf;

fn load(file: &str) -> Tire {
    TireSpec::Mc { file: file.into() }.load().unwrap()
}

fn mc(tire: &Tire) -> &McParams {
    let TireModel::Motorcycle(p) = &tire.model else { panic!("MF-MC expected") };
    p
}

#[test]
fn relaxation_matches_de_vries_and_pacejka() {
    // Table 9.5: σ/K_yα (m/N) at 20, 59, 100, 140, 200 and 251 km/h; the thesis's quadratic
    // fits (which the files carry) stay within 7 % of the measurements.
    let table = [
        ("Evangelou_120_70_ZR17", [0.91e-5, 0.90e-5, 1.04e-5, 1.16e-5, 1.32e-5, 1.53e-5]),
        ("Evangelou_180_55_ZR17", [0.97e-5, 0.99e-5, 1.09e-5, 1.20e-5, 1.48e-5, 1.80e-5]),
    ];
    for (file, ratios) in table {
        let tire = load(file);
        let p = mc(&tire);
        for (kmh, want) in [20.0, 59.0, 100.0, 140.0, 200.0, 251.0].into_iter().zip(ratios) {
            let o = p.eval(p.fzo, 0.0, 0.0, 0.0, kmh / 3.6, 1.0);
            let got = o.sigma_y / o.kya.abs();
            assert!((got / want - 1.0).abs() < 0.07, "{file} at {kmh} km/h: {got:.3e} vs {want:.3e}");
        }
        // Relaxation lengths of 0.1–0.3 m.
        let s = p.eval(p.fzo, 0.0, 0.0, 0.0, 20.0, 1.0).sigma_y;
        assert!(s > 0.1 && s < 0.3, "{file}: σ = {s}");
    }
}

#[test]
fn camber_and_slip_act_the_right_way() {
    for file in ["Evangelou_120_70_ZR17", "Evangelou_180_55_ZR17", "Bicycle_37_622"] {
        let tire = load(file);
        let p = mc(&tire);
        let fz = p.fzo;
        let eval = |kappa: f64, alpha: f64, gamma: f64| p.eval(fz, kappa, f64::tan(alpha), gamma, 20.0, 1.0);
        // Leaning right (γ > 0): thrust and twisting moment to the right (ISO-W: −y, −z).
        let o = eval(0.0, 0.0, 0.3);
        assert!(o.fy < 0.0 && o.mz < 0.0, "{file}: {o:?}");
        // Sliding left (α > 0): the force to the right, the moment aligning the wheel.
        let o = eval(0.0, 0.03, 0.0);
        assert!(o.fy < 0.0 && o.mz > 0.0 && o.trail > 0.0, "{file}: {o:?}");
        // Symmetric without the asymmetric fits.
        let (l, r) = (eval(0.0, 0.0, 0.4), eval(0.0, 0.0, -0.4));
        assert!((l.fy + r.fy).abs() < 1e-9 * fz && (l.mz + r.mz).abs() < 1e-9 * fz, "{file}");
        // Camber stiffness and cornering stiffness at the nominal load.
        let kyg = (eval(0.0, 0.0, 1e-4).fy - eval(0.0, 0.0, -1e-4).fy) / 2e-4;
        let kya = (eval(0.0, 1e-5, 0.0).fy - eval(0.0, -1e-5, 0.0).fy) / 2e-5;
        assert!((kyg + p.pky6 * fz).abs() < 1e-3 * kyg.abs(), "{file}: K_yγ {kyg}");
        let want = p.pky1 * fz * (p.pky2 * (1.0 / p.pky3).atan()).sin();
        assert!((kya + want).abs() < 1e-3 * want, "{file}: K_yα {kya} vs {want}");
        println!(
            "{file}: K_yα/F_z {:.2} /rad, K_yγ/F_z {:.2} /rad, trail {:.1} mm, σ_y {:.3} m",
            -kya / fz,
            -kyg / fz,
            1e3 * eval(0.0, 1e-4, 0.0).trail,
            eval(0.0, 0.0, 0.0).sigma_y
        );
        // Driving and braking.
        assert!(eval(0.05, 0.0, 0.0).fx > 0.0 && eval(-0.05, 0.0, 0.0).fx < 0.0);
        // Side slip takes away drive force, drive takes away side force.
        assert!(eval(0.05, 0.1, 0.0).fx < eval(0.05, 0.0, 0.0).fx);
        assert!(eval(0.1, 0.05, 0.0).fy.abs() < eval(0.0, 0.05, 0.0).fy.abs());
    }
}

#[test]
fn motorcycle_tyres_grip_like_the_thesis_says() {
    for file in ["Evangelou_120_70_ZR17", "Evangelou_180_55_ZR17"] {
        let tire = load(file);
        let p = mc(&tire);
        for fz in [1000.0, 2000.0, 3000.0] {
            // §9.3.7: longitudinal force peaks about 1.33 times the load.
            let peak = (0..200).map(|k| p.eval(fz, k as f64 * 0.002, 0.0, 0.0, 20.0, 1.0).fx).fold(0.0, f64::max);
            assert!((peak / fz - 1.35).abs() < 0.06, "{file} at {fz} N: F_x peak {:.3} F_z", peak / fz);
            // Leaning 45° the tyre holds the load with a few degrees of side slip (a lean angle
            // of about 45° needs F_y ≈ F_z), and the side force can exceed it.
            let fy = |alpha: f64| -p.eval(fz, 0.0, f64::tan(alpha), 45f64.to_radians(), 20.0, 1.0).fy;
            let camber_only = fy(0.0);
            assert!(camber_only > 0.4 * fz && camber_only < fz, "{file} at {fz} N: {camber_only}");
            let needed = (0..100).map(|k| k as f64 * 0.1f64.to_radians()).find(|&a| fy(a) >= fz);
            assert!(needed.is_some_and(|a| a < 6f64.to_radians()), "{file} at {fz} N: {needed:?}");
        }
    }
}

/// A single wheel held at a lean, rolling straight on flat ground without slip.
fn roll_leaning(tire: &Tire, lean: f64, yaw_rate: f64) -> TireForces {
    let flat = FlatTerrain::new(0.0, MaterialId::ASPHALT);
    let r = tire.radius();
    let rc = tire.crown_radius;
    let q = DQuat::from_rotation_x(lean);
    let axis = q * DVec3::Y;
    let deflection = 0.01;
    let center = DVec3::new(0.0, 0.0, (r - rc) * lean.cos() + rc - deflection);
    let v = 15.0;
    let mut state = tire.initial_state();
    let mut f = TireForces::default();
    for _ in 0..2000 {
        let motion = WheelMotion {
            center,
            axis,
            velocity: DVec3::new(v, 0.0, 0.0),
            carrier_angvel: DVec3::new(0.0, 0.0, yaw_rate),
            spin: v / (r - deflection),
        };
        let contact = tire.contact(&flat, &motion);
        f = tire.step(&mut state, contact.as_ref(), &motion, Surface::REFERENCE, 1e-3);
    }
    f
}

#[test]
fn a_leaning_toroidal_wheel_carries_its_load_on_the_crown() {
    let tire = load("Evangelou_180_55_ZR17");
    let (r, rc) = (tire.radius(), tire.crown_radius);
    for deg in [0.0f64, 20.0, 40.0, 55.0] {
        let lean = deg.to_radians();
        let f = roll_leaning(&tire, lean, 0.0);
        // The load from the deflection along the normal, at the point below the crown centre.
        assert!((f.fz - 1.4e5 * 0.01).abs() < 1e-6 * f.fz, "{deg}°: F_z {}", f.fz);
        assert!((f.deflection - 0.01).abs() < 1e-12);
        // Leaning right, the contact point lies left of the centre, but less so than a thin
        // disc's would (it has moved around the crown towards the lean).
        let offset = (r - rc) * lean.sin();
        assert!((f.point.y - offset).abs() < 1e-12 && f.point.z.abs() < 1e-12, "{deg}°: {}", f.point);
        let disc = (r - 0.01) * lean.sin();
        assert!(deg == 0.0 || f.point.y < disc);
        // No side slip at the crown centre: the side force is the camber thrust (to the right).
        let p = mc(&tire);
        let want = p.eval(f.fz, f.kappa, 0.0, lean, 15.0, 1.0);
        assert!(f.tan_alpha.abs() < 1e-9 && (f.fy - want.fy).abs() < 1e-6 * f.fz, "{deg}°: {f:?}");
    }
    // A thin disc under the same lean would carry the load in the wheel plane.
    let disc = TireSpec::Tir { file: "Sedan_Pac02Tire".into(), pressure: None, crown_radius: 0.0, turn_slip: false }
        .load()
        .unwrap();
    assert_eq!(disc.crown_radius, 0.0);
}

#[test]
fn turn_slip_through_the_wheel() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let file = root.join("fixtures/tir/MagicFormula62_Parameters.tir").to_string_lossy().into_owned();
    let spec = |turn_slip: bool| TireSpec::Tir { file: file.clone(), pressure: None, crown_radius: 0.05, turn_slip };
    let (plain, spinning) = (spec(false).load().unwrap(), spec(true).load().unwrap());
    // Leaning straight: through the camber spin, about the same camber thrust.
    let (a, b) = (roll_leaning(&plain, 0.05, 0.0), roll_leaning(&spinning, 0.05, 0.0));
    assert!(a.fy < 0.0 && (b.fy / a.fy - 1.0).abs() < 0.02, "{} vs {}", b.fy, a.fy);
    // Upright and yawing left: pushed outwards (right), turned back against the yaw (beyond
    // the sample tyre's conicity and ply steer).
    let (a, b) = (roll_leaning(&plain, 0.0, 0.5), roll_leaning(&spinning, 0.0, 0.5));
    assert!(b.fy - a.fy < -0.02 * a.fz && b.mz - a.mz < 0.0, "{a:?} {b:?}");
    // Turn slip needs MF 6.x.
    let old = TireSpec::Tir { file: "Sedan_Pac02Tire".into(), pressure: None, crown_radius: 0.0, turn_slip: true };
    assert!(old.load().is_err());
}

#[test]
fn rigid_rolling_is_linear_without_limit() {
    let spec: TireSpec = toml::from_str(
        "[fiala]\nradius = 0.35\nwidth = 0.03\nvertical_stiffness = 2e5\nslip_stiffness = 1e5\n\
         cornering_stiffness = 1e5\nmu = 0.9\nnominal_load = 350\nrelaxation_x = 0.01\nrelaxation_y = 0.01\n\
         rigid_rolling = true\n",
    )
    .unwrap();
    let tire = spec.load().unwrap();
    let flat = FlatTerrain::new(0.0, MaterialId::ASPHALT);
    let mut state = tire.initial_state();
    let (v, vy) = (5.0, 0.5);
    let mut f = TireForces::default();
    for _ in 0..500 {
        let motion = WheelMotion {
            center: DVec3::new(0.0, 0.0, 0.349),
            axis: DVec3::Y,
            velocity: DVec3::new(v, vy, 0.0),
            carrier_angvel: DVec3::ZERO,
            spin: v / 0.35,
        };
        let contact = tire.contact(&flat, &motion);
        f = tire.step(&mut state, contact.as_ref(), &motion, Surface::REFERENCE, 1e-3);
    }
    // Far beyond friction (μ F_z = 180 N), no aligning or rolling moment.
    assert!((f.fy + 1e5 * vy / v).abs() < 1e-6 * 1e5, "{f:?}");
    assert!(f.mz == 0.0 && f.my == 0.0);
}
