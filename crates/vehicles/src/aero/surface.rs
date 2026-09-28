//! Lifting surface (wing panel, tailplane, fin): lift curve with stall, drag polar and pitching
//! moment about the quarter chord, blended into flat-plate behaviour past the stall with Beard &
//! McLain's sigmoid (*Small Unmanned Aircraft*, 2012, §4.4), and an optional trailing-edge
//! control surface whose effectiveness τ and moment follow thin-aerofoil theory.
//!
//! The surface frame is the body frame rolled by `roll` about x and pitched up by `incidence`:
//! x along the chord (forward), z the lift normal. Only the flow in the chord plane counts
//! (no sweep or spanwise flow); the force acts at `position`, the aerodynamic centre.

use super::AirFlow;
use crate::VehicleError;
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};
use std::f64::consts::{FRAC_PI_2, PI};

/// Largest exponent in the stall blend (keeps it finite for any angle).
const MAX_EXPONENT: f64 = 60.0;

/// Squared chord-plane speed below which a surface produces no force ((m/s)²).
const MIN_SPEED_SQ: f64 = 1e-6;

/// Beard & McLain's stall blend `σ(α)`: the weight of flat-plate flow at angle of attack
/// `alpha` (0 attached, 1 fully stalled) for stall angle `alpha0` and sharpness `m` (1/rad).
pub fn stall_blend(alpha: f64, alpha0: f64, m: f64) -> f64 {
    let e1 = (-m * (alpha - alpha0)).clamp(-MAX_EXPONENT, MAX_EXPONENT).exp();
    let e2 = (m * (alpha + alpha0)).clamp(-MAX_EXPONENT, MAX_EXPONENT).exp();
    (1.0 + e1 + e2) / ((1.0 + e1) * (1.0 + e2))
}

/// Trailing-edge control surface (aileron, elevator, rudder, flap).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Flap {
    /// Flap chord over surface chord (0–1].
    pub chord_fraction: f64,
    /// Largest deflection either way (rad); positive is trailing edge down.
    pub max_deflection: f64,
    /// Lift effectiveness τ (dα_eff/dδ); thin-aerofoil value from the chord fraction when
    /// absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effectiveness: Option<f64>,
}

impl Flap {
    /// Hinge angle θ_h of thin-aerofoil theory: `cos θ_h = 2·c_f/c − 1`.
    fn hinge_angle(&self) -> f64 {
        (2.0 * self.chord_fraction - 1.0).clamp(-1.0, 1.0).acos()
    }

    /// Lift effectiveness `τ = 1 − (θ_h − sin θ_h)/π`, or the given value.
    pub fn tau(&self) -> f64 {
        self.effectiveness.unwrap_or_else(|| {
            let t = self.hinge_angle();
            1.0 - (t - t.sin()) / PI
        })
    }

    /// Thin-aerofoil change of the quarter-chord moment coefficient per radian of deflection:
    /// `−½·sin θ_h·(1 − cos θ_h)`.
    pub fn moment_slope(&self) -> f64 {
        let t = self.hinge_angle();
        -0.5 * t.sin() * (1.0 - t.cos())
    }
}

/// Section coefficients against angle of attack (radians, increasing), linearly interpolated
/// and held at the ends; replaces the parametric lift curve and polar.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlphaTable {
    pub alpha: Vec<f64>,
    pub cl: Vec<f64>,
    pub cd: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cm: Vec<f64>,
}

impl AlphaTable {
    fn validate(&self) -> Result<(), String> {
        let n = self.alpha.len();
        if n < 2 || self.cl.len() != n || self.cd.len() != n || !(self.cm.is_empty() || self.cm.len() == n) {
            return Err("table needs at least two angles and equally long cl, cd (and cm)".into());
        }
        if !self.alpha.windows(2).all(|w| w[1] > w[0]) {
            return Err("table angles must increase".into());
        }
        Ok(())
    }

    fn lookup(ys: &[f64], xs: &[f64], x: f64) -> f64 {
        if ys.is_empty() {
            return 0.0;
        }
        let i = xs.partition_point(|&a| a <= x).clamp(1, xs.len() - 1);
        let (x0, x1) = (xs[i - 1], xs[i]);
        let s = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0);
        ys[i - 1] + s * (ys[i] - ys[i - 1])
    }
}

/// Force and moment coefficients (moment about the quarter chord, positive nose up).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Coefficients {
    pub cl: f64,
    pub cd: f64,
    pub cm: f64,
}

/// A lifting surface of an aircraft.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AeroSurface {
    #[serde(default)]
    pub name: String,
    /// Planform area (m²), span (m) and mean aerodynamic chord (m).
    pub area: f64,
    pub span: f64,
    pub chord: f64,
    /// Aspect ratio for the lift-curve slope and induced drag (m²/m²): that of the whole wing
    /// for one panel of it; `span²/area` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<f64>,
    /// Aerodynamic centre in the body frame (m).
    pub position: DVec3,
    /// Rotation about the body x axis (rad): 0 for a wing (lift up), ±π/2 for a fin.
    #[serde(default)]
    pub roll: f64,
    /// Incidence relative to the body x axis (rad, leading edge up).
    #[serde(default)]
    pub incidence: f64,
    /// Lift at zero angle of attack and lift-curve slope (1/rad; from the aspect ratio,
    /// `πA/(1 + √(1 + (A/2)²))`, when absent).
    #[serde(default)]
    pub cl0: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cl_alpha: Option<f64>,
    /// Stall angle (rad) and sharpness of the blend into flat-plate flow (1/rad).
    #[serde(default = "default_alpha_stall")]
    pub alpha_stall: f64,
    #[serde(default = "default_stall_sharpness")]
    pub stall_sharpness: f64,
    /// Zero-lift drag, Oswald efficiency and flat-plate normal-force coefficient at 90°.
    #[serde(default = "default_cd0")]
    pub cd0: f64,
    #[serde(default = "default_oswald")]
    pub oswald: f64,
    #[serde(default = "default_cd90")]
    pub cd90: f64,
    /// Moment about the quarter chord at zero lift (positive nose up).
    #[serde(default)]
    pub cm0: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flap: Option<Flap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table: Option<AlphaTable>,
}

fn default_alpha_stall() -> f64 {
    15f64.to_radians()
}
fn default_stall_sharpness() -> f64 {
    50.0
}
fn default_cd0() -> f64 {
    0.01
}
fn default_oswald() -> f64 {
    0.8
}
fn default_cd90() -> f64 {
    1.98
}

impl AeroSurface {
    /// A surface with default aerofoil data.
    pub fn new(area: f64, span: f64, chord: f64, position: DVec3) -> Self {
        Self {
            name: String::new(),
            area,
            span,
            chord,
            aspect_ratio: None,
            position,
            roll: 0.0,
            incidence: 0.0,
            cl0: 0.0,
            cl_alpha: None,
            alpha_stall: default_alpha_stall(),
            stall_sharpness: default_stall_sharpness(),
            cd0: default_cd0(),
            oswald: default_oswald(),
            cd90: default_cd90(),
            cm0: 0.0,
            flap: None,
            table: None,
        }
    }

    pub fn validate(&self) -> Result<(), VehicleError> {
        let bad = |m: &str| Err(VehicleError::Invalid(format!("aero surface {:?}: {m}", self.name)));
        let positive = [self.area, self.span, self.chord, self.oswald, self.stall_sharpness];
        if !positive.iter().all(|x| x.is_finite() && *x > 0.0)
            || self.aspect_ratio.is_some_and(|a| !(a.is_finite() && a > 0.0))
        {
            return bad("area, span, chord, aspect_ratio, oswald and stall_sharpness must be positive");
        }
        if !(self.alpha_stall > 0.0 && self.alpha_stall < FRAC_PI_2) || !(self.cd0 >= 0.0 && self.cd90 >= 0.0) {
            return bad("alpha_stall must be in (0, π/2), cd0 and cd90 non-negative");
        }
        if let Some(f) = &self.flap
            && !(f.chord_fraction > 0.0 && f.chord_fraction <= 1.0 && f.max_deflection >= 0.0)
        {
            return bad("flap chord_fraction must be in (0, 1] and max_deflection non-negative");
        }
        if let Some(t) = &self.table
            && let Err(m) = t.validate()
        {
            return bad(&m);
        }
        Ok(())
    }

    pub fn aspect_ratio(&self) -> f64 {
        self.aspect_ratio.unwrap_or(self.span * self.span / self.area)
    }

    /// Lift-curve slope (1/rad).
    pub fn lift_slope(&self) -> f64 {
        self.cl_alpha.unwrap_or_else(|| {
            let a = self.aspect_ratio();
            PI * a / (1.0 + (1.0 + 0.25 * a * a).sqrt())
        })
    }

    /// Surface → body rotation.
    pub fn rotation(&self) -> DQuat {
        DQuat::from_rotation_x(self.roll) * DQuat::from_rotation_y(-self.incidence)
    }

    /// Weight of flat-plate flow at angle of attack `alpha` (0 attached, 1 fully stalled).
    pub fn stall_blend(&self, alpha: f64) -> f64 {
        stall_blend(alpha, self.alpha_stall, self.stall_sharpness)
    }

    /// Coefficients at angle of attack `alpha` (rad, any value) with the control surface
    /// deflected by `deflection` (rad, clamped to its range; ignored without one).
    pub fn coefficients(&self, alpha: f64, deflection: f64) -> Coefficients {
        let alpha = (alpha + PI).rem_euclid(2.0 * PI) - PI;
        let (shift, dcm) = match &self.flap {
            Some(f) => {
                let d = deflection.clamp(-f.max_deflection, f.max_deflection);
                (f.tau() * d, f.moment_slope() * d * self.lift_slope() / (2.0 * PI))
            }
            None => (0.0, 0.0),
        };
        let slope = self.lift_slope();
        let sigma = self.stall_blend(alpha);
        if let Some(t) = &self.table {
            return Coefficients {
                cl: AlphaTable::lookup(&t.cl, &t.alpha, alpha) + (1.0 - sigma) * slope * shift,
                cd: AlphaTable::lookup(&t.cd, &t.alpha, alpha),
                cm: AlphaTable::lookup(&t.cm, &t.alpha, alpha) + (1.0 - sigma) * dcm,
            };
        }
        let (s, c) = alpha.sin_cos();
        let cl_attached = self.cl0 + slope * (alpha + shift);
        // Flat plate: normal force cd90·sin α; the centre of pressure moves from the quarter
        // chord towards mid-chord as the flow turns broadside.
        let cn = self.cd90 * s;
        let folded = if alpha.abs() > FRAC_PI_2 { PI - alpha.abs() } else { alpha.abs() };
        let arm = 0.25 * folded / FRAC_PI_2;
        let induced = cl_attached * cl_attached / (PI * self.oswald * self.aspect_ratio());
        Coefficients {
            cl: (1.0 - sigma) * cl_attached + sigma * cn * c,
            cd: self.cd0 + (1.0 - sigma) * induced + sigma * cn * s,
            cm: (1.0 - sigma) * (self.cm0 + dcm) - sigma * cn * arm,
        }
    }

    /// Angle of attack (rad) of the flow in the chord plane; `None` without flow.
    pub fn angle_of_attack(&self, flow: &AirFlow) -> Option<f64> {
        let v = self.rotation().inverse() * flow.at(self.position);
        (v.x * v.x + v.z * v.z >= MIN_SPEED_SQ).then(|| (-v.z).atan2(v.x))
    }

    /// Force and moment about the body origin (body frame; N, N·m) in the flow `flow` with
    /// the control surface at `deflection`.
    pub fn wrench(&self, flow: &AirFlow, deflection: f64) -> (DVec3, DVec3) {
        let rot = self.rotation();
        let v = rot.inverse() * flow.at(self.position);
        let speed_sq = v.x * v.x + v.z * v.z;
        if speed_sq < MIN_SPEED_SQ {
            return (DVec3::ZERO, DVec3::ZERO);
        }
        let alpha = (-v.z).atan2(v.x);
        let k = self.coefficients(alpha, deflection);
        let qs = 0.5 * flow.density * speed_sq * self.area;
        let (s, c) = alpha.sin_cos();
        // Lift normal to the flow in the chord plane, drag along it; a nose-up moment is
        // negative about the surface's y axis (FLU).
        let force = rot * (DVec3::new(s, 0.0, c) * (qs * k.cl) + DVec3::new(-c, 0.0, s) * (qs * k.cd));
        let moment = rot * DVec3::new(0.0, -qs * self.chord * k.cm, 0.0);
        (force, self.position.cross(force) + moment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aero::{AirData, SEA_LEVEL_DENSITY};

    fn flow(v: DVec3) -> AirFlow {
        AirFlow::new(&AirData::default(), v, DVec3::ZERO)
    }

    /// A 2-D section: aspect ratio 10⁶.
    fn section() -> AeroSurface {
        AeroSurface { cd0: 0.0, ..AeroSurface::new(1.0, 1000.0, 1e-3, DVec3::ZERO) }
    }

    #[test]
    fn thin_aerofoil_section() {
        let s = section();
        assert!((s.lift_slope() / (2.0 * PI) - 1.0).abs() < 1e-5);
        let a = 4f64.to_radians();
        // The stall blend leaks ~10⁻⁴ of flat-plate flow at 4°.
        let k = s.coefficients(a, 0.0);
        assert!((k.cl / (2.0 * PI * a) - 1.0).abs() < 2e-4, "{k:?}");
        assert!(k.cd.abs() < 1e-6 && k.cm.abs() < 1e-6, "{k:?}");
        // Quarter-chord flap: τ = 1 − (2π/3 − sin 2π/3)/π, Δc_m = −½·sin θ_h·(1 − cos θ_h)·δ.
        let flap = Flap { chord_fraction: 0.25, max_deflection: 0.4, effectiveness: None };
        let t = 2.0 * PI / 3.0;
        assert!((flap.tau() - (1.0 - (t - t.sin()) / PI)).abs() < 1e-12);
        assert!((flap.tau() - 0.609).abs() < 1e-3);
        let s = AeroSurface { flap: Some(flap), ..section() };
        let d = 5f64.to_radians();
        let k = s.coefficients(0.0, d);
        assert!((k.cl - 2.0 * PI * 0.609 * d).abs() < 1e-3 * k.cl, "{k:?}");
        assert!((k.cm / d + 0.5 * t.sin() * (1.0 - t.cos())).abs() < 1e-4, "{k:?}");
        // Deflection is clamped, and a full-chord flap is a rotation of the section.
        assert_eq!(s.coefficients(0.0, 1.0), s.coefficients(0.0, 0.4));
        let full = Flap { chord_fraction: 1.0, max_deflection: 1.0, effectiveness: None };
        assert!((full.tau() - 1.0).abs() < 1e-12 && full.moment_slope().abs() < 1e-12);
    }

    #[test]
    fn finite_wing_lift_and_induced_drag() {
        let w = AeroSurface::new(6.0, 6.0, 1.0, DVec3::ZERO);
        // Helmbold: 2πA/(2 + √(A² + 4)) for A = 6.
        assert!((w.lift_slope() - 2.0 * PI * 6.0 / (2.0 + 40f64.sqrt())).abs() < 1e-12);
        let a = 5f64.to_radians();
        let k = w.coefficients(a, 0.0);
        let cl = w.lift_slope() * a;
        assert!((k.cl / cl - 1.0).abs() < 1e-4);
        assert!((k.cd - 0.01 - cl * cl / (PI * 0.8 * 6.0)).abs() < 1e-5);
    }

    #[test]
    fn flat_plate_past_the_stall() {
        let w = AeroSurface { cd0: 0.0, cl0: 0.3, ..AeroSurface::new(6.0, 6.0, 1.0, DVec3::ZERO) };
        for deg in [45.0f64, 60.0, 90.0, 135.0, -60.0, -120.0, 180.0] {
            let a = deg.to_radians();
            let k = w.coefficients(a, 0.0);
            assert!((k.cl - 1.98 * a.sin() * a.cos()).abs() < 1e-6, "{deg}: {k:?}");
            assert!((k.cd - 1.98 * a.sin().powi(2)).abs() < 1e-6, "{deg}: {k:?}");
        }
        // Broadside: all drag, centre of pressure at mid-chord.
        let k = w.coefficients(FRAC_PI_2, 0.0);
        assert!(k.cl.abs() < 1e-9 && (k.cd - 1.98).abs() < 1e-9 && (k.cm + 0.25 * 1.98).abs() < 1e-9);
        // The blend is 0 attached, ½ at the stall and 1 beyond; the lift curve stays bounded
        // and continuous through it.
        assert!(w.stall_blend(0.0) < 1e-5 && (w.stall_blend(w.alpha_stall) - 0.5).abs() < 1e-6);
        assert!(w.stall_blend(0.5) > 0.999 && w.stall_blend(-0.5) > 0.999);
        let mut prev = w.coefficients(-PI, 0.0).cl;
        for i in 1..=3600 {
            let cl = w.coefficients(-PI + i as f64 * PI / 1800.0, 0.0).cl;
            assert!(cl.is_finite() && cl.abs() < 2.0 && (cl - prev).abs() < 0.05, "{i}: {cl}");
            prev = cl;
        }
    }

    #[test]
    fn forces_act_at_the_aerodynamic_centre() {
        // Tailplane 3 m behind the centre of mass, 4° angle of attack at 20 m/s.
        let tail = AeroSurface::new(1.0, 2.0, 0.5, DVec3::new(-3.0, 0.0, 0.0));
        let a = 4f64.to_radians();
        let f = flow(DVec3::new(20.0 * a.cos(), 0.0, -20.0 * a.sin()));
        let (force, torque) = tail.wrench(&f, 0.0);
        let k = tail.coefficients(a, 0.0);
        let q = 0.5 * SEA_LEVEL_DENSITY * 400.0;
        let lift = DVec3::new(a.sin(), 0.0, a.cos()) * q * k.cl;
        let drag = DVec3::new(-a.cos(), 0.0, a.sin()) * q * k.cd;
        assert!((force - lift - drag).length() < 1e-9);
        // Lift behind the centre of mass pitches the nose down (+y in FLU).
        assert!(torque.y > 0.0 && (torque.y / (3.0 * force.z) - 1.0).abs() < 1e-3);
        // Incidence adds to the angle of attack; the flow along the span does not count.
        let set = AeroSurface { incidence: a, ..tail.clone() };
        let (f2, _) = set.wrench(&flow(DVec3::new(20.0, 7.0, 0.0)), 0.0);
        assert!((f2 - DVec3::new(-q * k.cd, 0.0, q * k.cl)).length() < 1e-9, "{f2}");
        // Pitching up moves the tail down, raising its angle of attack (pitch damping).
        let pitching = AirFlow::new(&AirData::default(), DVec3::new(20.0, 0.0, 0.0), DVec3::new(0.0, -0.2, 0.0));
        assert!(tail.wrench(&pitching, 0.0).1.y > 0.0);
    }

    #[test]
    fn vertical_fin_weathercocks() {
        let fin = AeroSurface { roll: FRAC_PI_2, ..AeroSurface::new(0.5, 1.0, 0.5, DVec3::new(-3.0, 0.0, 0.5)) };
        // Slipping left (air from the left): the fin's side force pushes the tail right,
        // yawing the nose left into the flow.
        let (force, torque) = fin.wrench(&flow(DVec3::new(20.0, 2.0, 0.0)), 0.0);
        assert!(force.y < 0.0 && torque.z > 0.0, "{force} {torque}");
        let (force, torque) = fin.wrench(&flow(DVec3::new(20.0, -2.0, 0.0)), 0.0);
        assert!(force.y > 0.0 && torque.z < 0.0, "{force} {torque}");
        assert_eq!(fin.wrench(&flow(DVec3::ZERO), 0.0), (DVec3::ZERO, DVec3::ZERO));
    }

    #[test]
    fn tables_and_validation() {
        let t = AlphaTable {
            alpha: vec![-0.2, 0.0, 0.2],
            cl: vec![-1.0, 0.2, 1.4],
            cd: vec![0.05, 0.02, 0.05],
            cm: vec![],
        };
        let w = AeroSurface { table: Some(t), ..AeroSurface::new(6.0, 6.0, 1.0, DVec3::ZERO) };
        w.validate().unwrap();
        let k = w.coefficients(0.1, 0.0);
        assert!((k.cl - 0.8).abs() < 1e-12 && (k.cd - 0.035).abs() < 1e-12 && k.cm == 0.0);
        assert_eq!(w.coefficients(1.0, 0.0).cl, 1.4);
        let bad = AeroSurface { alpha_stall: 2.0, ..w.clone() };
        assert!(bad.validate().is_err());
        let bad = AeroSurface {
            table: Some(AlphaTable { alpha: vec![0.1, 0.0], cl: vec![0.0; 2], cd: vec![0.0; 2], cm: vec![] }),
            ..w
        };
        assert!(bad.validate().is_err());
        let s: AeroSurface = toml::from_str(
            "area = 0.5\nspan = 1.0\nchord = 0.5\nposition = [-1.0, 0.0, 0.0]\nflap = { chord_fraction = 0.3, max_deflection = 0.4 }",
        )
        .unwrap();
        assert_eq!((s.alpha_stall, s.cd90, s.flap.unwrap().chord_fraction), (15f64.to_radians(), 1.98, 0.3));
    }
}
