//! Baseline timings of the multibody core (targets in docs/PLAN.md: ABA 20-DoF ≤ 3 µs).

use autonomousim_core::dynamics::*;
use autonomousim_core::math::{Pose, RigidInertia};
use criterion::{Criterion, criterion_group, criterion_main};
use glam::DVec3;
use std::hint::black_box;

const G: DVec3 = DVec3::new(0.0, 0.0, -9.80665);

fn quad() -> (MultibodyModel, MbState) {
    let mut m = MultibodyModel::new();
    m.add_link("body", None, JointType::Free, Pose::IDENTITY, RigidInertia::diag(1.5, DVec3::new(0.029, 0.029, 0.055)));
    let mut s = m.neutral_state();
    s.v.copy_from_slice(&[0.3, -0.2, 0.5, 1.0, 0.5, -0.2]);
    (m, s)
}

/// Serial chain of `n` revolute links with alternating axes.
fn chain(n: usize) -> (MultibodyModel, MbState) {
    let mut m = MultibodyModel::new();
    let mut parent = None;
    for i in 0..n {
        let axis = [DVec3::X, DVec3::Y, DVec3::Z][i % 3];
        let frame = if parent.is_some() { Pose::from_translation(DVec3::new(0.0, 0.0, 0.3)) } else { Pose::IDENTITY };
        let inertia = RigidInertia::cuboid(1.0, DVec3::new(0.05, 0.05, 0.3)).with_com(DVec3::new(0.0, 0.0, 0.15));
        parent = Some(m.add_link(format!("l{i}"), parent, JointType::revolute(axis), frame, inertia));
    }
    let mut s = m.neutral_state();
    for i in 0..n {
        s.q[i] = 0.1 * (i as f64 + 1.0).sin();
        s.v[i] = 0.2 * (i as f64 * 0.7).cos();
    }
    (m, s)
}

fn bench_single_body(c: &mut Criterion) {
    let (m, s) = quad();
    let tau = vec![0.0; 6];
    let mut ws = AbaWorkspace::new(&m);
    c.bench_function("aba/free_body", |b| {
        b.iter(|| aba(&m, black_box(&s.q), black_box(&s.v), &tau, &[], G, &mut ws).unwrap())
    });
    c.bench_function("step/free_body_semi_implicit", |b| {
        let mut st = s.clone();
        b.iter(|| {
            aba(&m, &st.q, &st.v, &tau, &[], G, &mut ws).unwrap();
            semi_implicit_euler(&m, &mut st, &ws.qdd, 0.002);
            black_box(&st);
        })
    });
}

fn bench_chain(c: &mut Criterion) {
    for n in [6, 20] {
        let (m, s) = chain(n);
        let tau = vec![0.1; n];
        let mut ws = AbaWorkspace::new(&m);
        let mut kin = KinCache::new(&m);
        c.bench_function(&format!("aba/chain{n}"), |b| {
            b.iter(|| aba(&m, black_box(&s.q), black_box(&s.v), &tau, &[], G, &mut ws).unwrap())
        });
        c.bench_function(&format!("rnea/chain{n}"), |b| {
            b.iter(|| black_box(rnea(&m, black_box(&s.q), &s.v, &tau, &[], G, &mut kin)))
        });
        c.bench_function(&format!("crba/chain{n}"), |b| {
            forward_kinematics(&m, &s.q, &s.v, &mut kin);
            b.iter(|| black_box(crba(&m, black_box(&kin))))
        });
    }
}

criterion_group!(benches, bench_single_body, bench_chain);
criterion_main!(benches);
