//! Whole-aircraft aerodynamic models: stability and control derivatives (Beard & McLain,
//! *Small Unmanned Aircraft*, 2012, §4.2–4.4) or JSBSim-style coefficient build-ups (sums of
//! products of variables and tables per axis).
//!
//! Coefficients follow the aeronautical conventions in FRD terms: lift up and drag aft of the
//! flow (stability axes for derivatives, wind axes for tables, as JSBSim), side force to the
//! right, and roll, pitch and yaw moments positive right wing down, nose up and nose right.
//! Control deflections are positive trailing edge down (elevator, flap), and as each model's
//! derivatives define them for aileron and rudder; [`AeroModel::control_signs`] recovers the
//! directions that roll right, pitch up and yaw right. Results are returned in the FLU body
//! frame about the centre of mass.

use super::table::{Lookup, Table};
use crate::aero::stall_blend;
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Speed below which rate terms use this speed instead (m/s), keeping `b/2V` finite.
const MIN_RATE_SPEED: f64 = 0.5;

/// Everything the coefficients depend on.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AeroInput {
    pub alpha: f64,
    pub beta: f64,
    pub airspeed: f64,
    pub mach: f64,
    /// Air-relative body rates in FRD terms: roll, pitch (nose up), yaw (nose right) (rad/s).
    pub p: f64,
    pub q: f64,
    pub r: f64,
    /// Rate of change of α (rad/s).
    pub alpha_dot: f64,
    /// Surface deflections (rad).
    pub aileron: f64,
    pub elevator: f64,
    pub rudder: f64,
    pub flap: f64,
    /// Height of the aerodynamic reference point above the ground over the span.
    pub height_over_span: f64,
    /// Stall hysteresis state (0 attached, 1 stalled).
    pub stall: f64,
    /// Free-stream and propeller-slipstream dynamic pressure (Pa).
    pub dynamic_pressure: f64,
    pub wash_pressure: f64,
}

/// Aircraft reference geometry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Geometry {
    /// Wing area (m²).
    pub area: f64,
    /// Wing span (m).
    pub span: f64,
    /// Mean aerodynamic chord (m).
    pub chord: f64,
    /// Point the moment coefficients refer to (body frame, relative to the centre of mass, m).
    #[serde(default)]
    pub aero_reference: DVec3,
}

impl Geometry {
    pub fn aspect_ratio(&self) -> f64 {
        self.span * self.span / self.area
    }
}

/// Force and moment coefficients of an evaluation (FRD axes as documented above; lift and drag
/// in the model's own axes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AeroCoefficients {
    pub lift: f64,
    pub drag: f64,
    pub side: f64,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
}

/// Aerodynamic force and moment on the aircraft (FLU body frame, about the centre of mass).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AeroForces {
    pub force: DVec3,
    pub moment: DVec3,
    pub coefficients: AeroCoefficients,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum AeroModel {
    Derivatives(Derivatives),
    Tables(TableModel),
}

impl AeroModel {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            AeroModel::Derivatives(d) => d.validate(),
            AeroModel::Tables(t) => t.validate(),
        }
    }

    /// Stall hysteresis limits `[α_off, α_on]`, if the model has a stall state.
    pub fn stall_hysteresis(&self) -> Option<[f64; 2]> {
        match self {
            AeroModel::Tables(t) => t.stall_hysteresis,
            AeroModel::Derivatives(_) => None,
        }
    }

    pub fn coefficients(&self, geo: &Geometry, x: &AeroInput) -> AeroCoefficients {
        match self {
            AeroModel::Derivatives(d) => d.coefficients(geo, x),
            AeroModel::Tables(t) => t.coefficients(geo, x),
        }
    }

    /// Force and moment for input `x`.
    pub fn forces(&self, geo: &Geometry, x: &AeroInput) -> AeroForces {
        let c = self.coefficients(geo, x);
        let qs = x.dynamic_pressure * geo.area;
        // Coefficients are already multiplied by their own pressure ratio (see TableModel).
        let (sa, ca) = x.alpha.sin_cos();
        let (lift, drag, side) = (qs * c.lift, qs * c.drag, qs * c.side);
        let f_frd = match self {
            AeroModel::Derivatives(_) => DVec3::new(-drag * ca + lift * sa, side, -drag * sa - lift * ca),
            AeroModel::Tables(_) => {
                let (sb, cb) = x.beta.sin_cos();
                let xw = DVec3::new(ca * cb, sb, sa * cb);
                let yw = DVec3::new(-ca * sb, cb, -sa * sb);
                let zw = DVec3::new(-sa, 0.0, ca);
                -drag * xw + side * yw - lift * zw
            }
        };
        let m_frd = DVec3::new(qs * geo.span * c.roll, qs * geo.chord * c.pitch, qs * geo.span * c.yaw);
        let force = DVec3::new(f_frd.x, -f_frd.y, -f_frd.z);
        let moment = DVec3::new(m_frd.x, -m_frd.y, -m_frd.z) + geo.aero_reference.cross(force);
        AeroForces { force, moment, coefficients: c }
    }

    /// Lift coefficient at `alpha` with everything else neutral.
    pub fn lift_at(&self, geo: &Geometry, alpha: f64) -> f64 {
        let x = AeroInput {
            alpha,
            airspeed: 30.0,
            dynamic_pressure: 1.0,
            wash_pressure: 1.0,
            height_over_span: 10.0,
            ..Default::default()
        };
        self.coefficients(geo, &x).lift
    }

    /// Angles of attack of maximum and minimum lift (the stall angles), searched over ±0.6 rad
    /// in 0.005 rad steps with everything else neutral.
    pub fn stall_angles(&self, geo: &Geometry) -> (f64, f64) {
        let search = |sign: f64| {
            let (mut best, mut at) = (f64::NEG_INFINITY, 0.0);
            for i in 0..=120 {
                let a = sign * 0.005 * i as f64;
                let cl = sign * self.lift_at(geo, a);
                if cl > best + 1e-12 {
                    (best, at) = (cl, a);
                }
            }
            at
        };
        (search(1.0), search(-1.0))
    }

    /// Signs that turn positive normalised commands into deflections that roll right, pitch
    /// nose up and yaw nose right (aileron, elevator, rudder), from the sign of each control's
    /// moment derivative at a small angle of attack. A control without effect gets `+1`.
    pub fn control_signs(&self, geo: &Geometry) -> [f64; 3] {
        let base = AeroInput {
            alpha: 0.05,
            airspeed: 30.0,
            dynamic_pressure: 1.0,
            wash_pressure: 1.0,
            height_over_span: 10.0,
            ..Default::default()
        };
        let d = 0.01;
        let c0 = self.coefficients(geo, &base);
        let roll = self.coefficients(geo, &AeroInput { aileron: d, ..base }).roll - c0.roll;
        let pitch = self.coefficients(geo, &AeroInput { elevator: d, ..base }).pitch - c0.pitch;
        let yaw = self.coefficients(geo, &AeroInput { rudder: d, ..base }).yaw - c0.yaw;
        [roll, pitch, yaw].map(|x| if x < 0.0 { -1.0 } else { 1.0 })
    }
}

/// Stability and control derivatives (per radian; rates normalised as `p̂ = pb/2V`,
/// `q̂ = qc/2V`, `r̂ = rb/2V`), Beard & McLain's names: lift `cl`, drag `cd`, side force `cy`,
/// roll `croll`, pitch `cm`, yaw `cn`.
///
/// Lift blends the linear curve into flat-plate lift `2·sgn(α)·sin²α·cosα` with the sigmoid
/// `σ(α; alpha0, stall_sharpness)`; drag is the parasitic drag plus the induced polar
/// `(cl0 + cl_alpha·α)²/(π·e·AR)`, blended into `cd90·sin²α` the same way.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Derivatives {
    pub cl0: f64,
    pub cl_alpha: f64,
    #[serde(default)]
    pub cl_q: f64,
    #[serde(default)]
    pub cl_de: f64,
    #[serde(default)]
    pub cl_flap: f64,
    /// Parasitic drag.
    pub cd0: f64,
    /// Linear drag in α (Beard & McLain's `C_Dα`; 0 with the polar).
    #[serde(default)]
    pub cd_alpha: f64,
    #[serde(default)]
    pub cd_q: f64,
    #[serde(default)]
    pub cd_de: f64,
    #[serde(default)]
    pub cd_flap: f64,
    /// Oswald efficiency of the induced polar (0: no induced drag).
    #[serde(default)]
    pub oswald: f64,
    /// Flat-plate drag at 90°.
    #[serde(default = "default_cd90")]
    pub cd90: f64,
    pub cm0: f64,
    pub cm_alpha: f64,
    #[serde(default)]
    pub cm_q: f64,
    #[serde(default)]
    pub cm_de: f64,
    #[serde(default)]
    pub cm_flap: f64,
    #[serde(default)]
    pub cy0: f64,
    pub cy_beta: f64,
    #[serde(default)]
    pub cy_p: f64,
    #[serde(default)]
    pub cy_r: f64,
    #[serde(default)]
    pub cy_da: f64,
    #[serde(default)]
    pub cy_dr: f64,
    #[serde(default)]
    pub croll0: f64,
    pub croll_beta: f64,
    pub croll_p: f64,
    #[serde(default)]
    pub croll_r: f64,
    pub croll_da: f64,
    #[serde(default)]
    pub croll_dr: f64,
    #[serde(default)]
    pub cn0: f64,
    pub cn_beta: f64,
    #[serde(default)]
    pub cn_p: f64,
    pub cn_r: f64,
    #[serde(default)]
    pub cn_da: f64,
    #[serde(default)]
    pub cn_dr: f64,
    /// Centre of the stall blend (rad).
    pub alpha0: f64,
    /// Sharpness `M` of the stall blend (1/rad).
    #[serde(default = "default_sharpness")]
    pub stall_sharpness: f64,
}

fn default_cd90() -> f64 {
    2.0
}
fn default_sharpness() -> f64 {
    50.0
}

impl Derivatives {
    fn validate(&self) -> Result<(), String> {
        let all = [
            self.cl0,
            self.cl_alpha,
            self.cl_q,
            self.cl_de,
            self.cl_flap,
            self.cd0,
            self.cd_alpha,
            self.cd_q,
            self.cd_de,
            self.cd_flap,
            self.oswald,
            self.cd90,
            self.cm0,
            self.cm_alpha,
            self.cm_q,
            self.cm_de,
            self.cm_flap,
            self.cy0,
            self.cy_beta,
            self.cy_p,
            self.cy_r,
            self.cy_da,
            self.cy_dr,
            self.croll0,
            self.croll_beta,
            self.croll_p,
            self.croll_r,
            self.croll_da,
            self.croll_dr,
            self.cn0,
            self.cn_beta,
            self.cn_p,
            self.cn_r,
            self.cn_da,
            self.cn_dr,
            self.alpha0,
            self.stall_sharpness,
        ];
        if all.iter().any(|x| !x.is_finite()) {
            return Err("derivatives must be finite".into());
        }
        if !(self.cl_alpha > 0.0 && self.cd0 >= 0.0 && self.oswald >= 0.0 && self.cd90 >= 0.0) {
            return Err("need cl_alpha > 0 and non-negative drag terms".into());
        }
        if !(self.alpha0 > 0.0 && self.stall_sharpness > 0.0) {
            return Err("stall alpha0 and sharpness must be positive".into());
        }
        Ok(())
    }

    fn coefficients(&self, geo: &Geometry, x: &AeroInput) -> AeroCoefficients {
        let v = x.airspeed.max(MIN_RATE_SPEED);
        let (ph, qh, rh) = (x.p * geo.span / (2.0 * v), x.q * geo.chord / (2.0 * v), x.r * geo.span / (2.0 * v));
        let a = x.alpha;
        let sigma = stall_blend(a, self.alpha0, self.stall_sharpness);
        let (sa, ca) = a.sin_cos();
        let cl_lin = self.cl0 + self.cl_alpha * a;
        let cl_alpha = (1.0 - sigma) * cl_lin + sigma * 2.0 * a.signum() * sa * sa * ca;
        let induced = if self.oswald > 0.0 { cl_lin * cl_lin / (PI * self.oswald * geo.aspect_ratio()) } else { 0.0 };
        let cd_alpha =
            (1.0 - sigma) * (self.cd0 + self.cd_alpha * a + induced) + sigma * (self.cd0 + self.cd90 * sa * sa);
        AeroCoefficients {
            lift: cl_alpha + self.cl_q * qh + self.cl_de * x.elevator + self.cl_flap * x.flap,
            drag: cd_alpha + self.cd_q * qh + self.cd_de * x.elevator + self.cd_flap * x.flap,
            side: self.cy0
                + self.cy_beta * x.beta
                + self.cy_p * ph
                + self.cy_r * rh
                + self.cy_da * x.aileron
                + self.cy_dr * x.rudder,
            roll: self.croll0
                + self.croll_beta * x.beta
                + self.croll_p * ph
                + self.croll_r * rh
                + self.croll_da * x.aileron
                + self.croll_dr * x.rudder,
            pitch: self.cm0 + self.cm_alpha * a + self.cm_q * qh + self.cm_de * x.elevator + self.cm_flap * x.flap,
            yaw: self.cn0
                + self.cn_beta * x.beta
                + self.cn_p * ph
                + self.cn_r * rh
                + self.cn_da * x.aileron
                + self.cn_dr * x.rudder,
        }
    }
}

/// Variables of coefficient terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Var {
    Alpha,
    Beta,
    AbsBeta,
    /// `pb/2V`, `qc/2V`, `rb/2V`, `α̇c/2V`.
    PHat,
    QHat,
    RHat,
    AlphaDotHat,
    Aileron,
    Elevator,
    AbsElevator,
    Rudder,
    Flap,
    Mach,
    /// Height of the aerodynamic reference point above the ground over the span.
    HeightOverSpan,
    /// Stall hysteresis state (0 or 1).
    Stall,
}

struct Vars<'a> {
    x: &'a AeroInput,
    ph: f64,
    qh: f64,
    rh: f64,
    adh: f64,
}

impl Lookup<Var> for Vars<'_> {
    fn value(&self, v: Var) -> f64 {
        let x = self.x;
        match v {
            Var::Alpha => x.alpha,
            Var::Beta => x.beta,
            Var::AbsBeta => x.beta.abs(),
            Var::PHat => self.ph,
            Var::QHat => self.qh,
            Var::RHat => self.rh,
            Var::AlphaDotHat => self.adh,
            Var::Aileron => x.aileron,
            Var::Elevator => x.elevator,
            Var::AbsElevator => x.elevator.abs(),
            Var::Rudder => x.rudder,
            Var::Flap => x.flap,
            Var::Mach => x.mach,
            Var::HeightOverSpan => x.height_over_span,
            Var::Stall => x.stall,
        }
    }
}

/// Dynamic pressure a term is multiplied by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pressure {
    /// Free stream.
    #[default]
    Free,
    /// Free stream plus the propeller slipstream (JSBSim's `qbar-induced`), for surfaces in the
    /// slipstream such as the elevator and rudder.
    Slipstream,
}

/// One coefficient term: `scale × Π vars × Π tables`, times its dynamic pressure, the wing
/// area and (for moments) the span or chord.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Term {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default = "one")]
    pub scale: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vars: Vec<Var>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<Table<Var>>,
    #[serde(default)]
    pub pressure: Pressure,
}

fn one() -> f64 {
    1.0
}

impl Term {
    fn eval(&self, v: &Vars) -> f64 {
        let mut c = self.scale;
        for &var in &self.vars {
            c *= v.value(var);
        }
        for t in &self.tables {
            c *= t.eval(v);
        }
        match self.pressure {
            Pressure::Free => c,
            // Relative to the free stream, since the caller multiplies by it.
            Pressure::Slipstream if v.x.dynamic_pressure > 0.0 => c * v.x.wash_pressure / v.x.dynamic_pressure,
            Pressure::Slipstream => 0.0,
        }
    }
}

/// JSBSim-style build-up: each axis sums its terms. Lift, drag and side force act in wind axes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableModel {
    /// Stall hysteresis `[α_off, α_on]` (rad): the `stall` variable becomes 1 above `α_on` and
    /// 0 again below `α_off`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stall_hysteresis: Option<[f64; 2]>,
    #[serde(default)]
    pub lift: Vec<Term>,
    #[serde(default)]
    pub drag: Vec<Term>,
    #[serde(default)]
    pub side: Vec<Term>,
    #[serde(default)]
    pub roll: Vec<Term>,
    #[serde(default)]
    pub pitch: Vec<Term>,
    #[serde(default)]
    pub yaw: Vec<Term>,
}

impl TableModel {
    fn axes(&self) -> [&[Term]; 6] {
        [&self.lift, &self.drag, &self.side, &self.roll, &self.pitch, &self.yaw]
    }

    fn validate(&self) -> Result<(), String> {
        for t in self.axes().into_iter().flatten() {
            if !t.scale.is_finite() {
                return Err(format!("term {:?}: scale must be finite", t.name));
            }
            for table in &t.tables {
                table.validate().map_err(|e| format!("term {:?}: {e}", t.name))?;
            }
        }
        if let Some([off, on]) = self.stall_hysteresis
            && !(off.is_finite() && on > off)
        {
            return Err("stall hysteresis needs off < on".into());
        }
        Ok(())
    }

    fn coefficients(&self, geo: &Geometry, x: &AeroInput) -> AeroCoefficients {
        let k = 1.0 / (2.0 * x.airspeed.max(MIN_RATE_SPEED));
        let vars = Vars {
            x,
            ph: x.p * geo.span * k,
            qh: x.q * geo.chord * k,
            rh: x.r * geo.span * k,
            adh: x.alpha_dot * geo.chord * k,
        };
        let sum = |terms: &[Term]| terms.iter().map(|t| t.eval(&vars)).sum::<f64>();
        AeroCoefficients {
            lift: sum(&self.lift),
            drag: sum(&self.drag),
            side: sum(&self.side),
            roll: sum(&self.roll),
            pitch: sum(&self.pitch),
            yaw: sum(&self.yaw),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo() -> Geometry {
        Geometry { area: 0.55, span: 2.8956, chord: 0.18994, aero_reference: DVec3::ZERO }
    }

    fn derivs() -> Derivatives {
        Derivatives {
            cl0: 0.23,
            cl_alpha: 5.61,
            cl_q: 7.95,
            cl_de: 0.13,
            cl_flap: 0.0,
            cd0: 0.043,
            cd_alpha: 0.0,
            cd_q: 0.0,
            cd_de: 0.0135,
            cd_flap: 0.0,
            oswald: 0.9,
            cd90: 2.0,
            cm0: 0.0135,
            cm_alpha: -2.74,
            cm_q: -38.21,
            cm_de: -0.99,
            cm_flap: 0.0,
            cy0: 0.0,
            cy_beta: -0.98,
            cy_p: 0.0,
            cy_r: 0.0,
            cy_da: 0.075,
            cy_dr: 0.19,
            croll0: 0.0,
            croll_beta: -0.13,
            croll_p: -0.51,
            croll_r: 0.25,
            croll_da: 0.17,
            croll_dr: 0.0024,
            cn0: 0.0,
            cn_beta: 0.073,
            cn_p: -0.069,
            cn_r: -0.095,
            cn_da: -0.011,
            cn_dr: -0.069,
            alpha0: 0.47,
            stall_sharpness: 50.0,
        }
    }

    fn input(alpha: f64, beta: f64) -> AeroInput {
        AeroInput { alpha, beta, airspeed: 25.0, dynamic_pressure: 382.8, wash_pressure: 382.8, ..Default::default() }
    }

    /// Stability- and wind-axis resolution agree at zero sideslip, and the FLU conversion puts
    /// lift up, drag aft and a right side force on −y.
    #[test]
    fn axes() {
        let d = AeroModel::Derivatives(derivs());
        let x = input(0.1, 0.0);
        let f = d.forces(&geo(), &x);
        let c = f.coefficients;
        let qs = 382.8 * 0.55;
        let (sa, ca) = 0.1f64.sin_cos();
        assert!((f.force.x - qs * (c.lift * sa - c.drag * ca)).abs() < 1e-9);
        assert!((f.force.z - qs * (c.lift * ca + c.drag * sa)).abs() < 1e-9);
        assert!(f.force.z > 0.0 && f.moment.y.abs() > 0.0);

        // The same coefficients through the table path.
        let term =
            |v: f64| Term { name: String::new(), scale: v, vars: vec![], tables: vec![], pressure: Pressure::Free };
        let t = AeroModel::Tables(TableModel {
            stall_hysteresis: None,
            lift: vec![term(c.lift)],
            drag: vec![term(c.drag)],
            side: vec![term(0.3)],
            roll: vec![],
            pitch: vec![term(c.pitch)],
            yaw: vec![],
        });
        let g = t.forces(&geo(), &x);
        assert!((g.force - f.force - DVec3::new(0.0, -qs * 0.3, 0.0)).length() < 1e-9);
        assert!((g.moment - f.moment).length() < 1e-9);
        // Sideslip: wind-axis drag gains a body side component, air from the right (β > 0)
        // gives drag towards +y (left, FLU).
        let g = AeroModel::Tables(TableModel {
            side: vec![],
            ..match t {
                AeroModel::Tables(m) => m,
                _ => unreachable!(),
            }
        })
        .forces(&geo(), &input(0.0, 0.2));
        assert!(g.force.y > 0.0);
    }

    #[test]
    fn signs_and_stall() {
        let d = AeroModel::Derivatives(derivs());
        // Positive aileron rolls right already; elevator and rudder derivatives are negative.
        assert_eq!(d.control_signs(&geo()), [1.0, -1.0, -1.0]);
        let (hi, lo) = d.stall_angles(&geo());
        assert!(hi > 0.3 && hi < 0.5, "{hi}");
        assert!(lo < -0.3 && lo > -0.5, "{lo}");
        // Linear region matches cl0 + cl_alpha·α.
        assert!((d.lift_at(&geo(), 0.05) - (0.23 + 5.61 * 0.05)).abs() < 1e-3);
        // Moment transfer: a reference point ahead of the CG adds r × F.
        let mut g = geo();
        g.aero_reference = DVec3::new(0.1, 0.0, 0.0);
        let x = input(0.1, 0.0);
        let (f0, f1) = (d.forces(&geo(), &x), d.forces(&g, &x));
        assert!((f1.moment - f0.moment - DVec3::new(0.1, 0.0, 0.0).cross(f0.force)).length() < 1e-9);
        // Lift ahead of the CG pitches the nose up (−y moment in FLU).
        assert!(f1.moment.y < f0.moment.y);
    }

    #[test]
    fn slipstream_terms() {
        let t = AeroModel::Tables(TableModel {
            stall_hysteresis: Some([0.1, 0.3]),
            lift: vec![],
            drag: vec![],
            side: vec![],
            roll: vec![],
            pitch: vec![Term {
                name: "cm_de".into(),
                scale: -1.0,
                vars: vec![Var::Elevator],
                tables: vec![],
                pressure: Pressure::Slipstream,
            }],
            yaw: vec![],
        });
        let mut x = input(0.0, 0.0);
        x.elevator = 0.1;
        let c0 = t.coefficients(&geo(), &x).pitch;
        x.wash_pressure = 2.0 * x.dynamic_pressure;
        let c1 = t.coefficients(&geo(), &x).pitch;
        assert!((c0 + 0.1).abs() < 1e-12 && (c1 - 2.0 * c0).abs() < 1e-12);
        assert_eq!(t.control_signs(&geo())[1], -1.0);
    }
}
