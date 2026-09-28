//! Rotor model against closed-form blade-element and momentum results (Johnson, *Helicopter
//! Theory*, ch. 2 and 5) and a brute-force numerical blade-element integration.

use autonomousim_vehicles::rotorcraft::{Rotor, RotorDef, RotorInput, RotorLoads, RotorState, Spin};
use glam::{DMat3, DVec3};
use std::f64::consts::{PI, TAU};

const RHO: f64 = 1.225;
const OMEGA: f64 = 40.0;

/// A Bo105-sized rotor with ideal tips and no cut-out, Lock number 8.
fn ideal() -> RotorDef {
    let (radius, chord, a) = (5.0, 0.3, 5.73);
    RotorDef {
        radius,
        blades: 4,
        chord,
        lift_slope: a,
        twist: -0.14,
        root_cutout: 0.0,
        tip_loss: 1.0,
        profile_drag: 0.008,
        profile_drag_thrust: 0.0,
        flap_inertia: RHO * a * chord * radius.powi(4) / 8.0,
        flap_stiffness: 0.0,
        polar_inertia: None,
        spin: Spin::Ccw,
        flap_limit: 0.35,
    }
}

fn input(velocity: DVec3, collective: f64, cyclic: [f64; 2]) -> RotorInput {
    RotorInput { velocity, collective, cyclic, ..RotorInput::still(RHO) }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1e-3)
}

#[test]
fn hover_matches_momentum_and_blade_element_theory() {
    let def = ideal();
    let rotor = Rotor::new(def.clone()).unwrap();
    let (sigma, a, delta) = (def.solidity(), def.lift_slope, def.profile_drag);
    let th75 = 0.12;
    let l = rotor.loads(&RotorState::spinning(OMEGA), &input(DVec3::ZERO, th75, [0.0; 2]));

    // λ = (σa/16)(√(1 + 64θ₇₅/(3σa)) − 1), C_T = (σa/2)(θ₇₅/3 − λ/2).
    let lambda = sigma * a / 16.0 * ((1.0 + 64.0 * th75 / (3.0 * sigma * a)).sqrt() - 1.0);
    let ct = 0.5 * sigma * a * (th75 / 3.0 - lambda / 2.0);
    assert!(close(l.inflow_ratio, lambda, 1e-10), "{} vs {lambda}", l.inflow_ratio);
    assert!(close(l.ct, ct, 1e-10), "{} vs {ct}", l.ct);
    assert!(close(l.induced, (ct / 2.0).sqrt(), 1e-10));
    // C_Q = λC_T + σδ/8; figure of merit C_T^{3/2}/√2 / C_Q.
    let cq = lambda * ct + sigma * delta / 8.0;
    assert!(close(l.cq, cq, 1e-10), "{} vs {cq}", l.cq);
    let fm = ct.powf(1.5) / 2f64.sqrt() / l.cq;
    assert!((0.6..0.8).contains(&fm), "figure of merit {fm}");

    // Dimensional thrust, torque and power; no in-plane force or tilt in hover.
    let tip = OMEGA * def.radius;
    let area = PI * def.radius * def.radius;
    assert!(close(l.force.z, ct * RHO * area * tip * tip, 1e-10));
    assert!(close(l.torque, cq * RHO * area * tip * tip * def.radius, 1e-10));
    assert!(close(l.power, l.torque * OMEGA, 1e-14));
    assert!(l.force.truncate().length() < 1e-9 * l.force.z && l.moment == DVec3::ZERO);
    assert!(l.flap_steady[0].abs() < 1e-12 && l.flap_steady[1].abs() < 1e-12);

    // Coning β₀ = γ(θ₀/8 + θ_tw/10 − λ/6) with θ₀ the pitch at the centre.
    let th0 = th75 - 0.75 * def.twist;
    let beta0 = 8.0 * (th0 / 8.0 + def.twist / 10.0 - lambda / 6.0);
    assert!(close(l.coning, beta0, 1e-10), "{} vs {beta0}", l.coning);
    assert!(close(l.lock_number, 8.0, 1e-12) && close(l.time_constant, 16.0 / (8.0 * OMEGA), 1e-12));
    assert!(!l.vortex_ring);
}

#[test]
fn forward_flight_matches_closed_form_blade_element_theory() {
    let def = ideal();
    let rotor = Rotor::new(def.clone()).unwrap();
    let (sigma, a, tw) = (def.solidity(), def.lift_slope, def.twist);
    let mu = 0.25;
    let (th75, th1c, th1s) = (0.10, 0.02, -0.04);
    let v = DVec3::new(mu * OMEGA * def.radius, 0.0, 0.0);
    let l = rotor.loads(&RotorState::spinning(OMEGA), &input(v, th75, [th1c, th1s]));
    let th0 = th75 - 0.75 * tw;
    let lambda = l.inflow_ratio;

    // Glauert: λ_i = C_T/(2√(μ² + λ²)) with an untilted tip-path plane.
    assert!(close(l.induced, l.ct / (2.0 * (mu * mu + lambda * lambda).sqrt()), 1e-10));
    // C_T = (σa/2)[θ₀(⅓ + μ²/2) + θ_tw(1 + μ²)/4 + μθ₁s/2 − λ/2].
    let ct = 0.5
        * sigma
        * a
        * (th0 * (1.0 / 3.0 + mu * mu / 2.0) + tw * (1.0 + mu * mu) / 4.0 + mu * th1s / 2.0 - lambda / 2.0);
    assert!(close(l.ct, ct, 1e-10), "{} vs {ct}", l.ct);
    assert!(close(l.advance_ratio, mu, 1e-14));

    // Flapping of a centrally hinged blade:
    // β₀ = γ[θ₀(1 + μ²)/8 + θ_tw(1/10 + μ²/12) + μθ₁s/6 − λ/6],
    // β₁c = −[2μ(4θ₀/3 + θ_tw − λ) + θ₁s(1 + 3μ²/2)]/(1 − μ²/2),
    // β₁s = θ₁c − (4/3)μβ₀/(1 + μ²/2).
    let gamma = 8.0;
    let b0 = gamma * (th0 * (1.0 + mu * mu) / 8.0 + tw * (0.1 + mu * mu / 12.0) + mu * th1s / 6.0 - lambda / 6.0);
    let b1c = -(2.0 * mu * (4.0 * th0 / 3.0 + tw - lambda) + th1s * (1.0 + 1.5 * mu * mu)) / (1.0 - mu * mu / 2.0);
    let b1s = th1c - 4.0 / 3.0 * mu * b0 / (1.0 + mu * mu / 2.0);
    assert!(close(l.coning, b0, 1e-10), "{} vs {b0}", l.coning);
    assert!(close(l.flap_steady[0], b1c, 1e-10), "{} vs {b1c}", l.flap_steady[0]);
    assert!(close(l.flap_steady[1], b1s, 1e-10), "{} vs {b1s}", l.flap_steady[1]);

    // Without tilt the disc drags aft and needs more power than in hover at the same thrust.
    assert!(l.force.x < 0.0);
}

/// Brute-force blade-element loads, integrated numerically in vector form in the geometric frame
/// with the blade turning either way: (hub force, aerodynamic torque).
fn brute_loads(def: &RotorDef, omega: f64, inp: &RotorInput, l: &RotorLoads, beta: [f64; 3]) -> (DVec3, f64) {
    let s = def.spin_sign();
    let w = inp.rates;
    let omega_eff = omega + s * w.z;
    let v_i = l.induced * omega_eff * def.radius;
    let delta = def.profile_drag + def.profile_drag_thrust * l.ct * l.ct;
    let (nr, nphi) = (4000, 64);
    let mut force = DVec3::ZERO;
    let mut torque = 0.0;
    for j in 0..nphi {
        let phi = TAU * (j as f64 + 0.5) / nphi as f64;
        let (c, sn) = (phi.cos(), phi.sin());
        let e_r = DVec3::new(-c, -sn, 0.0);
        let t = DVec3::new(sn, -c, 0.0) * s;
        let b = beta[0] + beta[1] * c + beta[2] * sn;
        let db_dt = s * omega_eff * (-beta[1] * sn + beta[2] * c);
        let n = DVec3::Z - e_r * b;
        for (lo, hi, lift) in [(def.root_cutout, def.tip_loss, true), (def.root_cutout, 1.0, false)] {
            let dr = (hi - lo) / nr as f64 * def.radius;
            for i in 0..nr {
                let rb = lo + (hi - lo) * (i as f64 + 0.5) / nr as f64;
                let r = rb * def.radius;
                let vel = inp.velocity
                    + DVec3::Z * v_i
                    + (DVec3::Z * (s * omega) + w).cross(e_r * r)
                    + DVec3::Z * (r * db_dt);
                let (ut, up) = (vel.dot(t), vel.dot(n));
                let q = 0.5 * inp.density * def.chord * dr;
                if lift {
                    let th = inp.collective + def.twist * (rb - 0.75) + inp.cyclic[0] * c + inp.cyclic[1] * sn;
                    let f_n = def.lift_slope * (th * ut * ut - up * ut);
                    let f_t = def.lift_slope * (th * ut * up - up * up);
                    force += (n * f_n - t * f_t) * q;
                    torque += r * f_t * q;
                } else {
                    force -= t * (delta * ut * ut * q);
                    torque += r * delta * ut * ut * q;
                }
            }
        }
    }
    let k = f64::from(def.blades) / nphi as f64;
    (force * k, torque * k)
}

/// Brute-force harmonic balance of the flap equation `I_β β̈ + (I_βΩ² + K_β)β = M +
/// 2sΩI_β(p cos φ + q sin φ)`, with the aerodynamic flap moment integrated numerically:
/// `[β₀, β₁c, β₁s]`.
fn brute_flapping(def: &RotorDef, omega: f64, inp: &RotorInput, l: &RotorLoads) -> [f64; 3] {
    let s = def.spin_sign();
    let w = inp.rates;
    let om = omega + s * w.z;
    let ib = def.flap_inertia;
    let v_i = l.induced * om * def.radius;
    let (nr, nphi) = (4000, 64);
    let residual = |beta: [f64; 3]| {
        let mut res = DVec3::ZERO;
        for j in 0..nphi {
            let phi = TAU * (j as f64 + 0.5) / nphi as f64;
            let (c, sn) = (phi.cos(), phi.sin());
            let e_r = DVec3::new(-c, -sn, 0.0);
            let t = DVec3::new(sn, -c, 0.0) * s;
            let b = beta[0] + beta[1] * c + beta[2] * sn;
            let db_dt = s * om * (-beta[1] * sn + beta[2] * c);
            let n = DVec3::Z - e_r * b;
            let (lo, hi) = (def.root_cutout, def.tip_loss);
            let dr = (hi - lo) / nr as f64 * def.radius;
            let mut moment = 0.0;
            for i in 0..nr {
                let rb = lo + (hi - lo) * (i as f64 + 0.5) / nr as f64;
                let r = rb * def.radius;
                let vel = inp.velocity
                    + DVec3::Z * v_i
                    + (DVec3::Z * (s * omega) + w).cross(e_r * r)
                    + DVec3::Z * (r * db_dt);
                let (ut, up) = (vel.dot(t), vel.dot(n));
                let th = inp.collective + def.twist * (rb - 0.75) + inp.cyclic[0] * c + inp.cyclic[1] * sn;
                moment += r * 0.5 * inp.density * def.chord * def.lift_slope * (th * ut * ut - up * ut) * dr;
            }
            let accel = -ib * om * om * (beta[1] * c + beta[2] * sn);
            let gyro = 2.0 * s * om * ib * (w.x * c + w.y * sn);
            let r = accel + (ib * om * om + def.flap_stiffness) * b - moment - gyro;
            res += DVec3::new(1.0, 2.0 * c, 2.0 * sn) * (r / nphi as f64);
        }
        res
    };
    let r0 = residual([0.0; 3]);
    let cols = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].map(|e| residual(e) - r0);
    let x = DMat3::from_cols(cols[0], cols[1], cols[2]).inverse() * -r0;
    [x.x, x.y, x.z]
}

#[test]
fn general_flight_matches_brute_force_blade_elements() {
    for spin in [Spin::Ccw, Spin::Cw] {
        for ground in [None, Some(4.0)] {
            let def = RotorDef {
                root_cutout: 0.15,
                tip_loss: 0.96,
                profile_drag_thrust: 30.0,
                flap_stiffness: 40_000.0,
                spin,
                ..ideal()
            };
            let rotor = Rotor::new(def.clone()).unwrap();
            let state = RotorState { omega: OMEGA, flap: [0.03, -0.02], inflow: 0.03 };
            let inp = RotorInput {
                velocity: DVec3::new(30.0, -12.0, -3.0),
                rates: DVec3::new(0.3, -0.2, 0.1),
                density: 1.1,
                collective: 0.11,
                cyclic: [0.03, -0.05],
                ground_distance: ground,
            };
            let l = rotor.loads(&state, &inp);
            let (force, torque) = brute_loads(&def, OMEGA, &inp, &l, [l.coning, state.flap[0], state.flap[1]]);
            let tol = 1e-6;
            for k in 0..3 {
                assert!(
                    (l.force[k] - force[k]).abs() < tol * force.length(),
                    "{spin:?} {ground:?} force {} vs {force}",
                    l.force
                );
            }
            assert!(close(l.torque, torque, tol), "{spin:?} torque {} vs {torque}", l.torque);
            let beta = brute_flapping(&def, OMEGA, &inp, &l);
            let flap = [l.coning, l.flap_steady[0], l.flap_steady[1]];
            for k in 0..3 {
                assert!((flap[k] - beta[k]).abs() < 1e-7, "{spin:?} {ground:?} flapping {flap:?} vs {beta:?}");
            }
            // The thrust coefficient and hub force agree, and the spring moment follows the tilt.
            let tip = (OMEGA + def.spin_sign() * inp.rates.z) * def.radius;
            assert!(close(l.force.z, l.ct * 1.1 * def.disc_area() * tip * tip, 1e-12));
            let k_s = 2.0 * def.flap_stiffness;
            assert!((l.moment - DVec3::new(-k_s * state.flap[1], k_s * state.flap[0], 0.0)).length() < 1e-9);
        }
    }
}

#[test]
fn flapping_lags_body_rates_and_cyclic_by_16_over_gamma_omega() {
    // Steady flapping in a hovering pitch and roll: the disc lags the shaft by τ·rate, with the
    // gyroscopic cross-coupling −p̂, −q̂ (its sign follows the direction of rotation).
    for spin in [Spin::Ccw, Spin::Cw] {
        let def = RotorDef { spin, ..ideal() };
        let rotor = Rotor::new(def.clone()).unwrap();
        let (p, q) = (0.4, -0.3);
        let inp = RotorInput { rates: DVec3::new(p, q, 0.0), ..input(DVec3::ZERO, 0.1, [0.0; 2]) };
        let l = rotor.loads(&RotorState::spinning(OMEGA), &inp);
        let (tau, s) = (16.0 / (8.0 * OMEGA), def.spin_sign());
        assert!(close(l.flap_steady[0], -tau * q - s * p / OMEGA, 1e-10), "{spin:?} {:?}", l.flap_steady);
        assert!(close(l.flap_steady[1], tau * p - s * q / OMEGA, 1e-10), "{spin:?} {:?}", l.flap_steady);
    }

    // A cyclic step: β₁c = −θ₁s in hover, reached with the time constant 16/(γΩ).
    let rotor = Rotor::new(ideal()).unwrap();
    let inp = input(DVec3::ZERO, 0.1, [0.0, 0.05]);
    let mut state = RotorState::spinning(OMEGA);
    let tau = 16.0 / (8.0 * OMEGA);
    let n = 1000;
    for _ in 0..n {
        let l = rotor.loads(&state, &inp);
        assert!(close(l.flap_steady[0], -0.05, 1e-10) && l.flap_steady[1].abs() < 1e-12);
        // A perfect governor holds Ω.
        rotor.advance(&mut state, &l, l.torque, tau / n as f64);
    }
    let expect = -0.05 * (1.0 - (-1.0f64).exp());
    assert!(close(state.flap[0], expect, 1e-6), "{} vs {expect}", state.flap[0]);
    // Once settled, the thrust tilts with the disc (toward −x) and a hub spring pitches the shaft.
    state.flap = [-0.05, 0.0];
    let l = rotor.loads(&state, &inp);
    assert!(close(l.force.x / l.force.z, -0.05, 1e-2), "{}", l.force.x / l.force.z);
    assert!(l.force.y.abs() < 1e-3 * l.force.z);
    let stiff = Rotor::new(RotorDef { flap_stiffness: 1e5, ..ideal() }).unwrap();
    assert!(stiff.loads(&state, &inp).moment.y < 0.0);
}

#[test]
fn ground_effect_reduces_induced_inflow() {
    let def = ideal();
    let rotor = Rotor::new(def.clone()).unwrap();
    let state = RotorState::spinning(OMEGA);
    let oge = rotor.loads(&state, &input(DVec3::ZERO, 0.1, [0.0; 2]));
    let ige =
        rotor.loads(&state, &RotorInput { ground_distance: Some(def.radius), ..input(DVec3::ZERO, 0.1, [0.0; 2]) });
    // Cheeseman–Bennett at z = R: λ_i = (1 − 1/16)·√(C_T/2), so more thrust for the collective
    // and less torque per thrust.
    assert!(close(ige.induced, (15.0 / 16.0) * (ige.ct / 2.0).sqrt(), 1e-10));
    assert!(ige.ct > oge.ct * 1.02);
    assert!(ige.cq / ige.ct < oge.cq / oge.ct);
    // Far away it does nothing.
    let far = rotor.loads(&state, &RotorInput { ground_distance: Some(1e6), ..input(DVec3::ZERO, 0.1, [0.0; 2]) });
    assert!(close(far.ct, oge.ct, 1e-9));
}

#[test]
fn rotor_speed_follows_the_torque_balance() {
    let def = ideal();
    let rotor = Rotor::new(def.clone()).unwrap();
    let inp = input(DVec3::ZERO, 0.1, [0.0; 2]);
    let mut state = RotorState::spinning(OMEGA);
    let l = rotor.loads(&state, &inp);
    rotor.advance(&mut state, &l, l.torque + 1000.0, 0.01);
    assert!(close(state.omega - OMEGA, 1000.0 * 0.01 / (4.0 * def.flap_inertia), 1e-12));
    assert!(close(rotor.angular_momentum(&state).z, 4.0 * def.flap_inertia * state.omega, 1e-14));
    // A constant drive torque settles where the aerodynamic torque matches it (Q ∝ Ω²).
    let drive = 1.2 * l.torque;
    for _ in 0..20_000 {
        let l = rotor.loads(&state, &inp);
        rotor.advance(&mut state, &l, drive, 0.002);
    }
    assert!(close(rotor.loads(&state, &inp).torque, drive, 1e-6));
    assert!(close(state.omega, OMEGA * 1.2f64.sqrt(), 2e-2), "{}", state.omega);
}

#[test]
fn a_descending_rotor_windmills_and_hovering_descent_warns_of_the_vortex_ring() {
    let rotor = Rotor::new(ideal()).unwrap();
    let state = RotorState::spinning(OMEGA);
    // Autorotation: forward flight with air flowing up through the disc at low collective.
    let l = rotor.loads(&state, &input(DVec3::new(40.0, 0.0, -12.0), 0.02, [0.0; 2]));
    assert!(l.torque < 0.0 && l.force.z > 0.0, "{l:?}");
    assert!(l.inflow_ratio < 0.0);
    // A slow vertical descent at the hover induced velocity is in the vortex ring.
    let hover = rotor.loads(&state, &input(DVec3::ZERO, 0.1, [0.0; 2]));
    let v_h = hover.induced * OMEGA * 5.0;
    assert!(!hover.vortex_ring);
    let mut state = state;
    state.inflow = hover.momentum_inflow;
    let l = rotor.loads(&state, &input(DVec3::new(0.0, 0.0, -v_h), 0.1, [0.0; 2]));
    assert!(l.vortex_ring && l.induced > hover.induced);
    assert!(!rotor.loads(&state, &input(DVec3::new(0.0, 0.0, -v_h * 3.0), 0.1, [0.0; 2])).vortex_ring);
    // A stopped rotor makes nothing.
    let l = rotor.loads(&RotorState::default(), &input(DVec3::new(10.0, 0.0, 0.0), 0.1, [0.0; 2]));
    assert_eq!((l.force, l.torque), (DVec3::ZERO, 0.0));
}

#[test]
fn rotor_definitions_validate_and_round_trip() {
    let toml = r#"
        radius = 0.775
        blades = 2
        chord = 0.058
        twist = 0.0
        flap_inertia = 0.038
        flap_stiffness = 54.0
        spin = "cw"
    "#;
    let def: RotorDef = toml::from_str(toml).unwrap();
    assert_eq!((def.lift_slope, def.tip_loss, def.profile_drag, def.spin), (5.73, 0.97, 0.008, Spin::Cw));
    assert!(Rotor::new(def.clone()).is_ok());
    let back: RotorDef = toml::from_str(&toml::to_string(&def).unwrap()).unwrap();
    assert_eq!(back, def);
    for bad in [
        RotorDef { radius: 0.0, ..def.clone() },
        RotorDef { blades: 0, ..def.clone() },
        RotorDef { root_cutout: 0.98, ..def.clone() },
        RotorDef { flap_stiffness: -1.0, ..def.clone() },
        RotorDef { polar_inertia: Some(0.0), ..def.clone() },
    ] {
        assert!(Rotor::new(bad).is_err());
    }
}
