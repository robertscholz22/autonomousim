//! Validation of the multibody dynamics against analytic solutions and conservation laws.

use autonomousim_core::dynamics::*;
use autonomousim_core::math::{Pose, RigidInertia, SpatialForce};
use glam::{DQuat, DVec3};

const G: DVec3 = DVec3::new(0.0, 0.0, -9.80665);

fn single_body(inertia: RigidInertia) -> MultibodyModel {
    let mut m = MultibodyModel::new();
    m.add_link("body", None, JointType::Free, Pose::IDENTITY, inertia);
    m
}

/// Integrate a torque-free/gravity-free system with RK4, calling `each` after every step.
fn run_rk4(model: &MultibodyModel, state: &mut MbState, dt: f64, steps: usize, mut each: impl FnMut(usize, &MbState)) {
    let mut ws = AbaWorkspace::new(model);
    let mut rk = Rk4Workspace::new(model);
    let tau = vec![0.0; model.nv()];
    for k in 0..steps {
        rk4(model, state, dt, &mut rk, |q, v, qdd| {
            aba(model, q, v, &tau, &[], DVec3::ZERO, &mut ws)?;
            qdd.copy_from_slice(&ws.qdd);
            Ok::<_, DynamicsError>(())
        })
        .unwrap();
        each(k, state);
    }
}

fn energy_momentum(model: &MultibodyModel, s: &MbState) -> (f64, SpatialForce) {
    let mut kin = KinCache::new(model);
    forward_kinematics(model, &s.q, &s.v, &mut kin);
    (kinetic_energy(model, &kin), spatial_momentum_world(model, &kin))
}

#[test]
fn torque_free_body_conserves_energy_and_momentum_rk4() {
    let model = single_body(RigidInertia::diag(2.0, DVec3::new(0.3, 0.5, 0.9)).with_com(DVec3::new(0.1, -0.2, 0.05)));
    let mut s = model.neutral_state();
    s.v.copy_from_slice(&[1.0, 2.5, -0.7, 0.3, -0.4, 1.2]);
    let (e0, h0) = energy_momentum(&model, &s);
    run_rk4(&model, &mut s, 1e-4, 100_000, |_, _| {});
    let (e1, h1) = energy_momentum(&model, &s);
    assert!(((e1 - e0) / e0).abs() < 1e-9, "energy drift {}", (e1 - e0) / e0);
    let dh = (h1 - h0).ang.length().max((h1 - h0).lin.length()) / h0.ang.length();
    assert!(dh < 1e-9, "momentum drift {dh}");
}

#[test]
fn torque_free_body_semi_implicit_drift_is_bounded() {
    let model = single_body(RigidInertia::diag(1.5, DVec3::new(0.02, 0.03, 0.05)));
    let mut s = model.neutral_state();
    s.v.copy_from_slice(&[3.0, -2.0, 5.0, 0.0, 0.0, 0.0]);
    let (e0, _) = energy_momentum(&model, &s);
    let mut ws = AbaWorkspace::new(&model);
    let tau = vec![0.0; 6];
    let mut max_rel = 0.0f64;
    for _ in 0..5_000 {
        aba(&model, &s.q, &s.v, &tau, &[], DVec3::ZERO, &mut ws).unwrap();
        let qdd = ws.qdd.clone();
        semi_implicit_euler(&model, &mut s, &qdd, 0.002);
        let (e, _) = energy_momentum(&model, &s);
        max_rel = max_rel.max(((e - e0) / e0).abs());
    }
    // The midpoint gyroscopic update conserves kinetic energy of torque-free motion.
    assert!(max_rel < 1e-10, "semi-implicit energy error {max_rel}");
}

/// Complete elliptic integral of the first kind via the arithmetic–geometric mean.
fn ellip_k(k2: f64) -> f64 {
    let (mut a, mut b) = (1.0f64, (1.0 - k2).sqrt());
    for _ in 0..30 {
        let (an, bn) = (0.5 * (a + b), (a * b).sqrt());
        a = an;
        b = bn;
    }
    std::f64::consts::PI / (2.0 * a)
}

/// Rotation near the intermediate principal axis flips periodically (Dzhanibekov effect).
/// The flip period follows from the Jacobi-elliptic solution of Euler's equations
/// (Landau & Lifshitz §37).
#[test]
fn dzhanibekov_flip_period_matches_analytic() {
    let (i1, i2, i3) = (1.0, 2.0, 3.0);
    let model = single_body(RigidInertia::diag(1.0, DVec3::new(i1, i2, i3)));
    let mut s = model.neutral_state();
    let (w1, w2, w3) = (0.0, 2.0, 0.02);
    s.v.copy_from_slice(&[w1, w2, w3, 0.0, 0.0, 0.0]);
    let e2 = i1 * w1 * w1 + i2 * w2 * w2 + i3 * w3 * w3; // 2E
    let m2 = (i1 * w1).powi(2) + (i2 * w2).powi(2) + (i3 * w3).powi(2); // M²
    assert!(m2 > e2 * i2);
    let rate = ((i3 - i2) * (m2 - e2 * i1) / (i1 * i2 * i3)).sqrt();
    let k2 = (i2 - i1) * (e2 * i3 - m2) / ((i3 - i2) * (m2 - e2 * i1));
    let period = 4.0 * ellip_k(k2) / rate;

    let dt = 1e-3;
    let mut crossings = Vec::new();
    let mut prev = s.v[1];
    run_rk4(&model, &mut s, dt, (3.5 * period / dt) as usize, |k, st| {
        let w = st.v[1];
        if prev > 0.0 && w <= 0.0 {
            // Linear interpolation of the downward zero crossing.
            crossings.push((k as f64 + prev / (prev - w)) * dt);
        }
        prev = w;
    });
    assert!(crossings.len() >= 3, "expected several flips, got {crossings:?}");
    let measured = crossings[2] - crossings[1];
    assert!(((measured - period) / period).abs() < 0.01, "measured {measured} vs analytic {period}");
}

/// Symmetric top without torque: body-frame ω precesses about the symmetry axis at
/// `Ω = (I3 - I1)/I1 · ω3`.
#[test]
fn symmetric_top_body_precession() {
    let (i1, i3) = (0.4, 0.9);
    let model = single_body(RigidInertia::diag(1.0, DVec3::new(i1, i1, i3)));
    let mut s = model.neutral_state();
    let (a, w3) = (0.5, 4.0);
    s.v.copy_from_slice(&[a, 0.0, w3, 0.0, 0.0, 0.0]);
    let omega = (i3 - i1) / i1 * w3;
    let dt = 1e-3;
    let steps = 3000;
    let mut max_err = 0.0f64;
    run_rk4(&model, &mut s, dt, steps, |k, st| {
        let t = (k + 1) as f64 * dt;
        let (ex, ey) = (a * (omega * t).cos(), a * (omega * t).sin());
        max_err = max_err.max((st.v[0] - ex).abs()).max((st.v[1] - ey).abs()).max((st.v[2] - w3).abs());
    });
    assert!(max_err < 1e-9, "max error {max_err}");
}

/// Fast heavy top on a spherical joint: slow precession rate `Ω_p ≈ m g l / (I3 ω3)`.
#[test]
fn heavy_top_gyroscopic_precession() {
    let (m, l, i1, i3, spin) = (1.0, 0.1, 0.02, 0.01, 300.0);
    let mut model = MultibodyModel::new();
    // Top pivots at the world origin; COM at distance l along the body z-axis.
    let inertia = RigidInertia::diag(m, DVec3::new(i1, i1, i3)).with_com(DVec3::new(0.0, 0.0, l));
    model.add_link("top", None, JointType::Spherical, Pose::IDENTITY, inertia);
    let mut s = model.neutral_state();
    let tilt = 0.3;
    let rot = DQuat::from_rotation_x(tilt);
    s.q.copy_from_slice(&[rot.x, rot.y, rot.z, rot.w]);
    let wp = m * 9.80665 * l / (i3 * spin);
    // Start on the steady-precession solution to suppress nutation: spin + precession about world z.
    let w_body = DVec3::new(0.0, 0.0, spin) + rot.inverse() * DVec3::new(0.0, 0.0, wp);
    s.v.copy_from_slice(&w_body.to_array());

    let mut ws = AbaWorkspace::new(&model);
    let mut rk = Rk4Workspace::new(&model);
    let tau = [0.0; 3];
    let dt = 2e-4;
    let t_end = 2.0;
    let azimuth = |s: &MbState| {
        let q = DQuat::from_xyzw(s.q[0], s.q[1], s.q[2], s.q[3]);
        let axis = q * DVec3::Z;
        axis.y.atan2(axis.x)
    };
    let az0 = azimuth(&s);
    let mut unwrapped = 0.0;
    let mut last = az0;
    for _ in 0..(t_end / dt) as usize {
        rk4(&model, &mut s, dt, &mut rk, |q, v, qdd| {
            aba(&model, q, v, &tau, &[], G, &mut ws)?;
            qdd.copy_from_slice(&ws.qdd);
            Ok::<_, DynamicsError>(())
        })
        .unwrap();
        let az = azimuth(&s);
        let mut d = az - last;
        if d > std::f64::consts::PI {
            d -= std::f64::consts::TAU;
        } else if d < -std::f64::consts::PI {
            d += std::f64::consts::TAU;
        }
        unwrapped += d;
        last = az;
    }
    let measured = unwrapped / t_end;
    assert!(((measured - wp) / wp).abs() < 0.02, "precession {measured} vs {wp}");
}

/// Planar double pendulum with distributed-mass links compared with its closed-form
/// Lagrangian equations of motion.
#[test]
fn double_pendulum_matches_closed_form() {
    let (m1, l1, c1, j1) = (1.3, 0.8, 0.35, 0.07);
    let (m2, c2, j2) = (0.9, 0.3, 0.05);
    let mut model = MultibodyModel::new();
    let axis = DVec3::Y;
    let link_inertia = |m: f64, c: f64, j: f64| {
        RigidInertia::new(m, DVec3::new(0.0, 0.0, -c), glam::DMat3::from_diagonal(DVec3::new(j, j, 0.01)))
    };
    let a = model.add_link("l1", None, JointType::revolute(axis), Pose::IDENTITY, link_inertia(m1, c1, j1));
    model.add_link(
        "l2",
        Some(a),
        JointType::revolute(axis),
        Pose::from_translation(DVec3::new(0.0, 0.0, -l1)),
        link_inertia(m2, c2, j2),
    );
    let g = 9.80665;
    let mut ws = AbaWorkspace::new(&model);
    let mut kin = KinCache::new(&model);
    let mut rng = autonomousim_core::rng::Seed::from_u64(11).rng();
    for _ in 0..200 {
        let q = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
        let v = [rng.range(-4.0, 4.0), rng.range(-4.0, 4.0)];
        let tau = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
        // Closed form (θ about +y; hanging along -z at θ = 0).
        let (t1, t2) = (q[0], q[1]);
        let m11 = j1 + m1 * c1 * c1 + j2 + m2 * (l1 * l1 + c2 * c2 + 2.0 * l1 * c2 * t2.cos());
        let m12 = j2 + m2 * (c2 * c2 + l1 * c2 * t2.cos());
        let m22 = j2 + m2 * c2 * c2;
        let h = m2 * l1 * c2 * t2.sin();
        let g1 = m1 * g * c1 * t1.sin() + m2 * g * (l1 * t1.sin() + c2 * (t1 + t2).sin());
        let g2 = m2 * g * c2 * (t1 + t2).sin();
        let b1 = tau[0] + h * (2.0 * v[0] * v[1] + v[1] * v[1]) - g1;
        let b2 = tau[1] - h * v[0] * v[0] - g2;
        let det = m11 * m22 - m12 * m12;
        let qdd_ref = [(m22 * b1 - m12 * b2) / det, (m11 * b2 - m12 * b1) / det];

        aba(&model, &q, &v, &tau, &[], G, &mut ws).unwrap();
        for k in 0..2 {
            assert!((ws.qdd[k] - qdd_ref[k]).abs() < 1e-10, "qdd {:?} vs {:?}", ws.qdd, qdd_ref);
        }
        let tau_back = rnea(&model, &q, &v, &qdd_ref, &[], G, &mut kin);
        for k in 0..2 {
            assert!((tau_back[k] - tau[k]).abs() < 1e-10);
        }
        forward_kinematics(&model, &q, &v, &mut kin);
        let mm = crba(&model, &kin);
        assert!(
            (mm[(0, 0)] - m11).abs() < 1e-12 && (mm[(0, 1)] - m12).abs() < 1e-12 && (mm[(1, 1)] - m22).abs() < 1e-12
        );
    }
}

/// Deterministic pseudo-random kinematic trees mixing all joint types.
fn random_tree(seed: u64, n: usize, floating: bool) -> MultibodyModel {
    let mut rng = autonomousim_core::rng::Seed::from_u64(seed).rng();
    let mut model = MultibodyModel::new();
    for i in 0..n {
        let parent = if i == 0 { None } else { Some(rng.below(i as u64) as usize) };
        let joint = if i == 0 {
            if floating { JointType::Free } else { JointType::revolute(rng.unit_vector()) }
        } else {
            match rng.below(5) {
                0 | 1 => JointType::revolute(rng.unit_vector()),
                2 => JointType::prismatic(rng.unit_vector()),
                3 => JointType::Spherical,
                _ => JointType::Fixed,
            }
        };
        let frame = Pose::new(rng.normal3() * 0.4, rng.rotation());
        let moments = DVec3::new(rng.range(0.1, 1.0), rng.range(0.1, 1.0), rng.range(0.1, 1.0));
        let d = DVec3::new(moments.y + moments.z, moments.x + moments.z, moments.x + moments.y) * 0.1;
        let r = glam::DMat3::from_quat(rng.rotation());
        let mass = rng.range(0.2, 3.0);
        let inertia =
            RigidInertia::new(mass, rng.normal3() * 0.2, r * glam::DMat3::from_diagonal(d * mass) * r.transpose());
        model.add_link(format!("l{i}"), parent, joint, frame, inertia);
    }
    model
}

fn random_state(model: &MultibodyModel, seed: u64) -> (MbState, Vec<f64>) {
    let mut rng = autonomousim_core::rng::Seed::from_u64(seed).rng();
    let mut s = model.neutral_state();
    let dq: Vec<f64> = (0..model.nv()).map(|_| rng.range(-1.5, 1.5)).collect();
    semi_implicit_euler(model, &mut s, &vec![0.0; model.nv()], 0.0);
    for (i, link) in model.links().iter().enumerate() {
        let (qo, vo) = (model.q_offset(i), model.v_offset(i));
        link.joint.integrate(&mut s.q[qo..qo + link.joint.nq()], &dq[vo..vo + link.joint.nv()], 1.0);
    }
    for v in s.v.iter_mut() {
        *v = rng.range(-2.0, 2.0);
    }
    let tau = (0..model.nv()).map(|_| rng.range(-3.0, 3.0)).collect();
    (s, tau)
}

/// ABA ≡ M⁻¹ (τ - C) and RNEA(ABA(τ)) = τ on random trees with all joint types.
#[test]
fn aba_crba_rnea_consistency_on_random_trees() {
    for seed in 0..60 {
        let floating = seed % 2 == 0;
        let model = random_tree(seed, 2 + (seed as usize % 9), floating);
        let (s, tau) = random_state(&model, 1000 + seed);
        let mut rng = autonomousim_core::rng::Seed::from_u64(5000 + seed).rng();
        let f_ext: Vec<SpatialForce> =
            (0..model.num_links()).map(|_| SpatialForce::new(rng.normal3(), rng.normal3())).collect();
        let mut ws = AbaWorkspace::new(&model);
        aba(&model, &s.q, &s.v, &tau, &f_ext, G, &mut ws).unwrap();

        let mut kin = KinCache::new(&model);
        let bias = rnea(&model, &s.q, &s.v, &vec![0.0; model.nv()], &f_ext, G, &mut kin);
        let mm = crba(&model, &kin);
        let rhs: Vec<f64> = tau.iter().zip(&bias).map(|(t, b)| t - b).collect();
        let qdd_ref = mm.solve_spd(&rhs).unwrap();
        for k in 0..model.nv() {
            assert!((ws.qdd[k] - qdd_ref[k]).abs() < 1e-8 * (1.0 + qdd_ref[k].abs()), "seed {seed} dof {k}");
        }
        let tau_back = rnea(&model, &s.q, &s.v, &ws.qdd, &f_ext, G, &mut kin);
        for k in 0..model.nv() {
            assert!((tau_back[k] - tau[k]).abs() < 1e-8 * (1.0 + tau[k].abs()), "seed {seed} dof {k}");
        }
        // Mass matrix symmetry.
        for r in 0..model.nv() {
            for c in 0..model.nv() {
                assert!((mm[(r, c)] - mm[(c, r)]).abs() < 1e-12);
            }
        }
    }
}

/// Hybrid dynamics: prescribing some joint accelerations yields the joint forces that
/// inverse dynamics would require, and leaves the free joints consistent.
#[test]
fn prescribed_joints_hybrid_dynamics() {
    for seed in 0..30 {
        let mut model = random_tree(100 + seed, 6, true);
        let prescribed: Vec<usize> =
            (1..model.num_links()).filter(|&i| model.link(i).joint.nv() > 0 && i % 2 == 1).collect();
        for &i in &prescribed {
            model.set_prescribed(i, true);
        }
        let (s, tau) = random_state(&model, 200 + seed);
        let mut ws = AbaWorkspace::new(&model);
        let mut rng = autonomousim_core::rng::Seed::from_u64(300 + seed).rng();
        for &i in &prescribed {
            let vo = model.v_offset(i);
            for k in 0..model.link(i).joint.nv() {
                ws.qdd[vo + k] = rng.range(-2.0, 2.0);
            }
        }
        aba(&model, &s.q, &s.v, &tau, &[], G, &mut ws).unwrap();
        let mut kin = KinCache::new(&model);
        let tau_id = rnea(&model, &s.q, &s.v, &ws.qdd, &[], G, &mut kin);
        for (i, link) in model.links().iter().enumerate() {
            let vo = model.v_offset(i);
            for k in 0..link.joint.nv() {
                let expected = if link.prescribed { ws.tau_prescribed[vo + k] } else { tau[vo + k] };
                assert!((tau_id[vo + k] - expected).abs() < 1e-8 * (1.0 + expected.abs()), "seed {seed} link {i}");
            }
        }
    }
}

/// A free-floating chain driven only by internal joint torques (no gravity) conserves its
/// total spatial momentum.
#[test]
fn internal_torques_conserve_momentum() {
    let model = random_tree(7, 6, true);
    let (mut s, _) = random_state(&model, 8);
    let (_, h0) = energy_momentum(&model, &s);
    let mut ws = AbaWorkspace::new(&model);
    let mut rk = Rk4Workspace::new(&model);
    let mut tau = vec![0.0; model.nv()];
    let dt = 1e-3;
    for step in 0..2000 {
        // Internal torques only: the floating base (first 6 DoF) is unactuated.
        for (k, t) in tau.iter_mut().enumerate().skip(6) {
            *t = (0.01 * step as f64 + k as f64).sin();
        }
        rk4(&model, &mut s, dt, &mut rk, |q, v, qdd| {
            aba(&model, q, v, &tau, &[], DVec3::ZERO, &mut ws)?;
            qdd.copy_from_slice(&ws.qdd);
            Ok::<_, DynamicsError>(())
        })
        .unwrap();
    }
    let (_, h1) = energy_momentum(&model, &s);
    let d = (h1 - h0).ang.length().max((h1 - h0).lin.length());
    assert!(d < 1e-6 * (1.0 + h0.ang.length()), "momentum drift {d}");
}

/// Free fall: a free body under gravity accelerates at g and its proper acceleration is zero.
#[test]
fn free_fall_proper_acceleration_is_zero() {
    let model = single_body(RigidInertia::cuboid(2.0, DVec3::new(0.3, 0.2, 0.1)));
    let mut s = model.neutral_state();
    let rot = DQuat::from_euler(glam::EulerRot::XYZ, 0.3, -0.5, 1.1);
    s.q[3..7].copy_from_slice(&[rot.x, rot.y, rot.z, rot.w]);
    let mut ws = AbaWorkspace::new(&model);
    aba(&model, &s.q, &s.v, &[0.0; 6], &[], G, &mut ws).unwrap();
    // Proper acceleration (what an accelerometer measures) is zero in free fall.
    assert!(ws.acc[0].lin.length() < 1e-12 && ws.acc[0].ang.length() < 1e-12);
    // Generalised acceleration = body-frame gravity.
    let g_body = rot.inverse() * G;
    assert!((DVec3::new(ws.qdd[3], ws.qdd[4], ws.qdd[5]) - g_body).length() < 1e-12);
}
