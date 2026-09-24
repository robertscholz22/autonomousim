//! Suspension kinematics as a one-DoF joint: the wheel carrier's pose as a function of the
//! wheel travel `s` (a kinematics-and-compliance table), so a suspension linkage need not be
//! modelled link by link.
//!
//! The carrier (child) frame sits at `p(s) = (x(s), y(s), z(s))` in the joint (predecessor)
//! frame, rotated by `R(s) = R_z(toe(s)) · R_x(camber(s))`. Each curve is a natural cubic
//! spline over the knots `travel`. The motion subspace in child coordinates and its derivative
//! follow analytically:
//!
//! ```text
//! S(s)   = (ω, v)  with  ω = (c′, t′ sin c, t′ cos c),  v = Rᵀ p′
//! dS/ds  = (c″, t″ sin c + t′c′ cos c, t″ cos c − t′c′ sin c,
//!           Rᵀ p″ − c′ R_xᵀ (e_x × R_zᵀ p′) − t′ Rᵀ (e_z × p′))
//! ```
//!
//! and the joint's bias acceleration is `c_J = dS/ds · ṡ²`.

use crate::math::spline::{CubicSpline, TableError};
use crate::math::{SpatialMotion, Xform};
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// The tabulated curves, as written in vehicle files. Empty curves are zero, except `z`, which
/// defaults to the travel itself (a straight vertical guide).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KcTableSpec {
    /// Travel knots (m), strictly increasing; positive is bump (wheel up relative to the body).
    pub travel: Vec<f64>,
    /// Carrier position in the joint frame (m).
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    /// Rotation about the joint frame's z axis, then about the rotated x axis (rad).
    pub toe: Vec<f64>,
    pub camber: Vec<f64>,
}

/// Interpolated suspension kinematics; see the module docs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "KcTableSpec", into = "KcTableSpec")]
pub struct KcTable {
    spec: KcTableSpec,
    x: CubicSpline,
    y: CubicSpline,
    z: CubicSpline,
    toe: CubicSpline,
    camber: CubicSpline,
}

/// Carrier pose, motion subspace and its travel derivative at one travel.
#[derive(Clone, Copy, Debug)]
pub struct KcPoint {
    pub position: DVec3,
    pub rotation: DQuat,
    /// Motion subspace `S(s)` in child coordinates.
    pub s: SpatialMotion,
    /// `dS/ds` in child coordinates.
    pub ds: SpatialMotion,
}

impl KcTable {
    pub fn new(spec: KcTableSpec) -> Result<Self, TableError> {
        let curve = |values: &[f64], default: Option<&[f64]>| -> Result<CubicSpline, TableError> {
            match (values.is_empty(), default) {
                (false, _) => CubicSpline::new(spec.travel.clone(), values.to_vec()),
                (true, Some(d)) => CubicSpline::new(spec.travel.clone(), d.to_vec()),
                (true, None) => {
                    CubicSpline::new(spec.travel.clone(), vec![0.0; spec.travel.len()])?;
                    Ok(CubicSpline::constant(0.0))
                }
            }
        };
        Ok(Self {
            x: curve(&spec.x, None)?,
            y: curve(&spec.y, None)?,
            z: curve(&spec.z, Some(&spec.travel))?,
            toe: curve(&spec.toe, None)?,
            camber: curve(&spec.camber, None)?,
            spec,
        })
    }

    /// A straight guide along the joint frame's z axis (equivalent to a prismatic joint).
    pub fn vertical(travel: [f64; 2]) -> Self {
        Self::new(KcTableSpec { travel: travel.to_vec(), ..Default::default() }).expect("valid range")
    }

    pub fn spec(&self) -> &KcTableSpec {
        &self.spec
    }

    /// Travel range covered by the table (m).
    pub fn range(&self) -> (f64, f64) {
        (self.spec.travel[0], self.spec.travel[self.spec.travel.len() - 1])
    }

    #[inline]
    pub fn eval(&self, s: f64) -> KcPoint {
        let (x, dx, ddx) = self.x.eval(s);
        let (y, dy, ddy) = self.y.eval(s);
        let (z, dz, ddz) = self.z.eval(s);
        let (t, dt, ddt) = self.toe.eval(s);
        let (c, dc, ddc) = self.camber.eval(s);
        let (st, ct) = t.sin_cos();
        let (sc, cc) = c.sin_cos();
        let (dp, ddp) = (DVec3::new(dx, dy, dz), DVec3::new(ddx, ddy, ddz));
        // Rᵀ = R_xᵀ(c) R_zᵀ(t).
        let rz_t = |v: DVec3| DVec3::new(ct * v.x + st * v.y, -st * v.x + ct * v.y, v.z);
        let rx_t = |v: DVec3| DVec3::new(v.x, cc * v.y + sc * v.z, -sc * v.y + cc * v.z);
        let r_t = |v: DVec3| rx_t(rz_t(v));
        let omega = DVec3::new(dc, dt * sc, dt * cc);
        let lin = r_t(dp);
        let d_omega = DVec3::new(ddc, ddt * sc + dt * dc * cc, ddt * cc - dt * dc * sc);
        let d_lin = r_t(ddp) - dc * rx_t(DVec3::X.cross(rz_t(dp))) - dt * r_t(DVec3::Z.cross(dp));
        KcPoint {
            position: DVec3::new(x, y, z),
            rotation: DQuat::from_rotation_z(t) * DQuat::from_rotation_x(c),
            s: SpatialMotion::new(omega, lin),
            ds: SpatialMotion::new(d_omega, d_lin),
        }
    }

    /// Joint transform `X_J(s)` (predecessor → carrier coordinates).
    #[inline]
    pub fn transform(&self, s: f64) -> Xform {
        let p = self.eval(s);
        Xform::from_pose(p.position, p.rotation)
    }
}

impl TryFrom<KcTableSpec> for KcTable {
    type Error = TableError;

    fn try_from(spec: KcTableSpec) -> Result<Self, TableError> {
        Self::new(spec)
    }
}

impl From<KcTable> for KcTableSpec {
    fn from(t: KcTable) -> Self {
        t.spec
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A table with every curve non-trivial (lateral scrub, toe and camber change with travel).
    pub(crate) fn curved() -> KcTable {
        KcTable::new(KcTableSpec {
            travel: vec![-0.1, -0.03, 0.0, 0.05, 0.1],
            x: vec![0.004, 0.001, 0.0, -0.002, -0.006],
            y: vec![-0.01, -0.002, 0.0, 0.001, -0.003],
            z: vec![-0.098, -0.03, 0.0, 0.049, 0.095],
            toe: vec![0.01, 0.003, 0.0, -0.004, -0.012],
            camber: vec![0.03, 0.008, 0.0, -0.015, -0.035],
        })
        .unwrap()
    }

    /// `S` and `dS/ds` against central differences of the transform and of `S`.
    #[test]
    fn subspace_and_its_derivative_match_finite_differences() {
        let table = curved();
        let h = 1e-6;
        for s in [-0.12, -0.08, -0.02, 0.0, 0.031, 0.07, 0.1, 0.15] {
            let p = table.eval(s);
            // Child velocity for ṡ = 1: rel = C(s+h) X C(s) over h.
            let (x0, x1) = (table.transform(s), table.transform(s + h));
            let rel = x1 * x0.inverse();
            let rot_err = rel.rot.transpose() - glam::DMat3::IDENTITY;
            let omega = DVec3::new(rot_err.y_axis.z, rot_err.z_axis.x, rot_err.x_axis.y) / h;
            assert!((omega - p.s.ang).length() < 1e-5, "{s}: ω {omega} vs {}", p.s.ang);
            assert!((rel.pos / h - p.s.lin).length() < 1e-5, "{s}: v {} vs {}", rel.pos / h, p.s.lin);
            let fd = (table.eval(s + h).s - table.eval(s - h).s) * (0.5 / h);
            assert!((fd.ang - p.ds.ang).length() < 1e-4 && (fd.lin - p.ds.lin).length() < 1e-4, "{s}: dS");
        }
        let v = KcTable::vertical([-0.1, 0.1]).eval(0.03);
        assert_eq!((v.position, v.s.lin, v.s.ang), (DVec3::new(0.0, 0.0, 0.03), DVec3::Z, DVec3::ZERO));
    }

    #[test]
    fn serde_round_trip_and_validation() {
        let table = curved();
        let json = serde_json::to_string(&table).unwrap();
        assert_eq!(serde_json::from_str::<KcTable>(&json).unwrap(), table);
        let bad = r#"{ "travel": [0.0, 0.1], "toe": [0.0] }"#;
        assert!(serde_json::from_str::<KcTable>(bad).is_err());
    }
}
