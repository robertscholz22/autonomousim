//! Helicopter presets against published data and first principles: the Bo105's rotor
//! parameters, power curve, trim trends, damping and autorotation (Padfield, *Helicopter Flight
//! Dynamics*, 2nd ed., Appendix 4B), the X-Cell's hover torque, flapping time constant and
//! forward-flight attitude (Gavrilets, Mettler & Feron, "Dynamic model for a miniature
//! aerobatic helicopter"), and the linearisation.

use autonomousim_vehicles::presets;
use autonomousim_vehicles::rotorcraft::*;
use std::sync::Arc;

const G: f64 = 9.80665;
const RHO: f64 = 1.225;
type L = HelicopterLinear;

fn heli(name: &str) -> Helicopter {
    Helicopter::new(Arc::new(presets::helicopter(name).unwrap()), 0.002)
}

fn trim(h: &Helicopter, v: f64) -> HelicopterTrim {
    h.trim(v, RHO, G).unwrap_or_else(|e| panic!("{e}"))
}

/// Padfield's Bo105: Lock number 5.087 and flap frequency ratio λ_β² = 1.248 (hingeless
/// rotor as a centre spring).
#[test]
fn bo105_rotor_matches_padfield() {
    let h = heli("bo105_like");
    let r = &h.def().main_rotor.rotor;
    let omega = h.def().engine.rated_speed;
    let lock = r.lock_number(RHO);
    let nu2 = 1.0 + r.flap_stiffness / (r.flap_inertia * omega * omega);
    assert!((lock / 5.087 - 1.0).abs() < 0.01, "Lock number {lock}");
    assert!((nu2 / 1.248 - 1.0).abs() < 0.005, "flap frequency ratio² {nu2}");
    // The hover trim uses the rotor's own flapping lag.
    let t = trim(&h, 0.0);
    assert!((t.loads.main.time_constant - 16.0 / (lock * omega)).abs() < 1e-3 * t.loads.main.time_constant);
}

/// Hover: a figure of merit typical of a full-size rotor; the Bo105's power bucket (minimum
/// power near 60 kt at about 55–65 % of hover power) and trim trends: the nose drops and the
/// stick moves forward with speed, collective has its own bucket, and the tail rotor needs less
/// pedal in cruise (the fin unloads it).
#[test]
fn bo105_power_curve_and_trim_trends() {
    let h = heli("bo105_like");
    let r = &h.def().main_rotor.rotor;
    let hover = trim(&h, 0.0);
    let thrust = hover.loads.main.force.z;
    let ideal = thrust * (thrust / (2.0 * RHO * r.disc_area())).sqrt();
    let fm = ideal / hover.loads.main.power;
    assert!((0.65..0.8).contains(&fm), "figure of merit {fm}");
    let speeds: Vec<f64> = (0..=12).map(|i| 5.0 * f64::from(i)).collect();
    let trims: Vec<_> = speeds.iter().map(|&v| trim(&h, v)).collect();
    let (i_min, min) = trims.iter().enumerate().min_by(|a, b| a.1.power.total_cmp(&b.1.power)).unwrap();
    assert!((25.0..=40.0).contains(&speeds[i_min]), "minimum power at {} m/s", speeds[i_min]);
    let ratio = min.power / hover.power;
    assert!((0.5..0.7).contains(&ratio), "bucket {ratio}");
    for w in trims.windows(2).skip(3) {
        assert!(w[1].pitch < w[0].pitch && w[1].controls.longitudinal > w[0].controls.longitudinal);
    }
    let coll: Vec<f64> = trims.iter().map(|t| t.controls.collective).collect();
    assert!(coll[i_min] < coll[0] - 0.1 && coll[i_min] < coll[12] - 0.1, "{coll:?}");
    assert!(trims[i_min].loads.tail.force.length() < 0.8 * hover.loads.tail.force.length());
    // Hover attitude: nose up with the forward shaft tilt, left side down against the tail
    // rotor thrust (a few degrees each).
    assert!((0.02..0.08).contains(&hover.pitch) && (-0.08..-0.02).contains(&hover.roll), "{hover:?}");
}

/// Steady autorotation at the governed rotor speed: collective within range, and the rate of
/// descent set by energy — the weight's descent power equals the power level flight needs at
/// the same speed (within the difference of the flow conditions).
#[test]
fn autorotation_descent_matches_power_required() {
    let h = heli("bo105_like");
    let weight = h.mass() * G;
    for v in [20.0, 30.0, 40.0, 50.0] {
        let a = h.trim_autorotation(v, RHO, G).unwrap_or_else(|e| panic!("{e}"));
        assert!(a.power.abs() < 1e-6 * weight * v, "{v}: {}", a.power);
        assert!(a.controls.collective > -1.0 && a.controls.collective < -0.3, "{v}: {:?}", a.controls);
        let level = trim(&h, v);
        let ratio = -a.climb_rate * weight / level.power;
        assert!((ratio - 1.0).abs() < 0.1, "{v}: {} m/s descent vs {} W: {ratio}", a.climb_rate, level.power);
        if v == 30.0 {
            // About 1700 ft/min near the minimum-power speed.
            assert!((7.0..10.0).contains(&-a.climb_rate), "{}", a.climb_rate);
        }
    }
    // The X-Cell's small, draggy rotor autorotates only in steep descents: in the windmill
    // brake state in vertical descent at about the hover power over weight.
    let x = heli("xcell60_like");
    let a = x.trim_autorotation(0.0, RHO, G).unwrap();
    let level = trim(&x, 0.0);
    let ratio = -a.climb_rate * x.mass() * G / level.power;
    assert!((ratio - 1.0).abs() < 0.15, "{ratio}");
}

/// Gavrilets et al.: hover main rotor torque about 6.3 N·m at 1600 rpm; the stabiliser bar's
/// effective flapping time constant about 0.1 s; −10° pitch attitude at 14.5 m/s.
#[test]
fn xcell_matches_gavrilets() {
    let h = heli("xcell60_like");
    let hover = trim(&h, 0.0);
    let q = hover.loads.main.torque;
    assert!((q / 6.3 - 1.0).abs() < 0.1, "hover torque {q}");
    let tau = hover.loads.main.time_constant;
    assert!((tau / 0.1 - 1.0).abs() < 0.1, "flapping time constant {tau}");
    let pitch = trim(&h, 14.5).pitch.to_degrees();
    assert!((pitch / -10.0 - 1.0).abs() < 0.25, "pitch at 14.5 m/s {pitch}°");
    // Trim cyclic and pedal stay well inside the range up to 20 m/s.
    for v in [0.0, 5.0, 10.0, 15.0, 20.0] {
        let c = trim(&h, v).controls.to_array();
        assert!(c.iter().all(|u| u.abs() < 0.8), "{v}: {c:?}");
    }
}

/// Hover pitch and roll damping from the rotor: the disc lags a body rate by `16/(γΩ)` rad per
/// rad/s, tilting the thrust (moment arm h) and loading the hub spring
/// (`N_b·K_β/2`): `M_q ≈ −(N_b K_β/2 + T h)·16/(γΩ)`, both axes (plus a little from the tail).
#[test]
fn hover_damping_matches_the_flapping_lag() {
    for name in ["bo105_like", "xcell60_like"] {
        let h = heli(name);
        let d = h.def();
        let r = &d.main_rotor.rotor;
        let t = trim(&h, 0.0);
        let lin = h.linearize(&t);
        let tau = 16.0 / (r.lock_number(RHO) * d.engine.rated_speed);
        let thrust = t.loads.main.force.z;
        let expect = -(f64::from(r.blades) * r.flap_stiffness / 2.0 + thrust * d.main_rotor.hub.z) * tau;
        // Moment derivatives: inertia times the angular-acceleration columns.
        let i = h.inertia();
        let col = |j: usize| i * glam::DVec3::new(lin.a[L::P][j], lin.a[L::Q][j], lin.a[L::R][j]);
        let (lp, mq) = (col(L::P).x, col(L::Q).y);
        assert!((lp / expect - 1.0).abs() < 0.15, "{name}: L_p {lp} vs {expect}");
        assert!((mq / expect - 1.0).abs() < 0.15, "{name}: M_q {mq} vs {expect}");
    }
}

/// Signs and consistency of the linear model: collective climbs, forward stick pitches the nose
/// down, right stick rolls right, right pedal yaws right; speed and heave are damped; the linear
/// model predicts a small perturbation of the nonlinear one.
#[test]
fn linear_model_signs_and_consistency() {
    for (name, v) in [("bo105_like", 0.0), ("bo105_like", 40.0), ("xcell60_like", 0.0), ("xcell60_like", 10.0)] {
        let h = heli(name);
        let t = trim(&h, v);
        let lin = h.linearize(&t);
        let b = &lin.b;
        assert!(b[L::W][0] > 0.0, "{name} {v}: collective {}", b[L::W][0]);
        // Nose-down pitch rate is +q in FLU; right roll +p; right yaw −r.
        assert!(b[L::Q][1] > 0.0 && b[L::P][2] > 0.0 && b[L::R][3] < 0.0, "{name} {v}: {b:?}");
        let a = &lin.a;
        assert!(a[L::U][L::U] < 0.0 && a[L::W][L::W] < 0.0 && a[L::R][L::R] < 0.0, "{name} {v}");
        assert!(a[L::P][L::P] < 0.0 && a[L::Q][L::Q] < 0.0, "{name} {v}");
        // Gravity couples the attitude into the velocities: pitch nose-down accelerates forward.
        assert!(a[L::U][L::THETA] < 0.0 && a[L::V][L::PHI] < 0.0, "{name} {v}");
        let dx = [0.1, -0.05, 0.08, 0.002, -0.003, 0.004, 0.001, -0.002];
        let du = [0.001, -0.002, 0.001, 0.002];
        let vb = t.velocity_body;
        let mut x = [vb.x, vb.y, vb.z, 0.0, 0.0, 0.0, t.roll, t.pitch];
        let mut u = t.controls.to_array();
        x.iter_mut().zip(dx).for_each(|(x, d)| *x += d);
        u.iter_mut().zip(du).for_each(|(u, d)| *u += d);
        let f = h.trim_derivative(&t, &x, &u);
        for (i, fi) in f.iter().enumerate() {
            let lin_i: f64 =
                (0..8).map(|j| a[i][j] * dx[j]).sum::<f64>() + (0..4).map(|j| b[i][j] * du[j]).sum::<f64>();
            assert!((fi - lin_i).abs() < 0.02 * lin_i.abs() + 2e-4, "{name} {v}: state {i}: {fi} vs {lin_i}");
        }
    }
}
