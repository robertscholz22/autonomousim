//! Golden comparison against Pinocchio (fixtures from `tools/gen_pinocchio_fixtures.py`):
//! ABA with and without external forces, RNEA, CRBA, kinetic energy and centre of mass.

use autonomousim_core::dynamics::*;
use autonomousim_core::math::{Pose, RigidInertia, SpatialForce};
use glam::{DMat3, DQuat, DVec3};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fixture {
    gravity: [f64; 3],
    links: Vec<FLink>,
    samples: Vec<Sample>,
}

#[derive(Deserialize)]
struct FLink {
    parent: Option<usize>,
    joint: FJoint,
    frame: FFrame,
    inertia: FInertia,
}

#[derive(Deserialize)]
struct FJoint {
    #[serde(rename = "type")]
    kind: String,
    axis: Option<[f64; 3]>,
}

#[derive(Deserialize)]
struct FFrame {
    pos: [f64; 3],
    rot: [f64; 4],
}

#[derive(Deserialize)]
struct FInertia {
    mass: f64,
    com: [f64; 3],
    i_com: [[f64; 3]; 3],
}

#[derive(Deserialize)]
struct Sample {
    q: Vec<f64>,
    v: Vec<f64>,
    tau: Vec<f64>,
    qdd: Vec<f64>,
    f_ext: Vec<[f64; 6]>,
    aba_qdd: Vec<f64>,
    aba_qdd_no_fext: Vec<f64>,
    rnea_tau: Vec<f64>,
    mass_matrix: Vec<Vec<f64>>,
    kinetic_energy: f64,
    com: [f64; 3],
}

fn build(f: &Fixture) -> MultibodyModel {
    let mut m = MultibodyModel::new();
    for (i, l) in f.links.iter().enumerate() {
        let axis = || DVec3::from_array(l.joint.axis.expect("axis"));
        let joint = match l.joint.kind.as_str() {
            "free" => JointType::Free,
            "spherical" => JointType::Spherical,
            "fixed" => JointType::Fixed,
            "revolute" => JointType::revolute(axis()),
            "prismatic" => JointType::prismatic(axis()),
            other => panic!("unknown joint type {other}"),
        };
        let r = l.frame.rot;
        let frame = Pose::new(DVec3::from_array(l.frame.pos), DQuat::from_xyzw(r[0], r[1], r[2], r[3]).normalize());
        let ic = l.inertia.i_com;
        // Row-major in the fixture; glam matrices are column-major.
        let i_com = DMat3::from_cols_array_2d(&ic).transpose();
        let inertia = RigidInertia::new(l.inertia.mass, DVec3::from_array(l.inertia.com), i_com);
        m.add_link(format!("l{i}"), l.parent, joint, frame, inertia);
    }
    m
}

fn assert_close(what: &str, name: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: {what} length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{name}: {what}[{k}] = {g}, Pinocchio {w} (Δ {:e})", g - w);
    }
}

#[test]
fn matches_pinocchio() {
    const TOL: f64 = 1e-10;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/pinocchio");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("fixtures/pinocchio missing; run tools/gen_pinocchio_fixtures.py")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert!(files.len() >= 5, "expected Pinocchio fixtures in {}", dir.display());

    for path in files {
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let fixture: Fixture = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let model = build(&fixture);
        let g = DVec3::from_array(fixture.gravity);
        let mut ws = AbaWorkspace::new(&model);
        let mut kin = KinCache::new(&model);
        for (si, s) in fixture.samples.iter().enumerate() {
            let name = format!("{name}#{si}");
            assert_eq!((s.q.len(), s.v.len()), (model.nq(), model.nv()), "{name}: dimensions");
            let f_ext: Vec<_> = s.f_ext.iter().map(|f| SpatialForce::from_array(*f)).collect();

            aba(&model, &s.q, &s.v, &s.tau, &f_ext, g, &mut ws).unwrap();
            assert_close("aba qdd", &name, &ws.qdd, &s.aba_qdd, TOL);
            aba(&model, &s.q, &s.v, &s.tau, &[], g, &mut ws).unwrap();
            assert_close("aba qdd (no f_ext)", &name, &ws.qdd, &s.aba_qdd_no_fext, TOL);

            let tau = rnea(&model, &s.q, &s.v, &s.qdd, &f_ext, g, &mut kin);
            assert_close("rnea tau", &name, &tau, &s.rnea_tau, TOL);

            forward_kinematics(&model, &s.q, &s.v, &mut kin);
            let m = crba(&model, &kin);
            for (r, row) in s.mass_matrix.iter().enumerate() {
                let got: Vec<f64> = (0..model.nv()).map(|c| m[(r, c)]).collect();
                assert_close(&format!("M row {r}"), &name, &got, row, TOL);
            }
            assert_close("kinetic energy", &name, &[kinetic_energy(&model, &kin)], &[s.kinetic_energy], TOL);
            assert_close("com", &name, &center_of_mass_world(&model, &kin).to_array(), &s.com, TOL);
        }
    }
}
