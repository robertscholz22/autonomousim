//! Fiala-type brush tyre for small tyres without Magic Formula data (robots): an isotropic
//! brush with a parabolic pressure distribution (Pacejka 2012, §3.2) under combined slip.
//!
//! With the slip demand `s = (C_κ κ, −C_α tan α)` and `z = |s| / (3 μ F_z)`, the force has
//! magnitude `μ F_z (3z − 3z² + z³)` along `s` while `z < 1` and `μ F_z` (full sliding)
//! beyond; the pneumatic trail is `(a/3)(1 − z)³ / (1 − z + z²/3)` for patch half length `a`.
//! Signs follow the Magic Formula convention (`κ = −V_sx/|V_cx|`, `tan α = V_sy/|V_cx|`).

use serde::{Deserialize, Serialize};

/// Parameters of a [`fiala`](self) tyre.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FialaParams {
    /// Unloaded radius and section width (m).
    pub radius: f64,
    pub width: f64,
    /// Vertical stiffness (N/m) and damping (N s/m).
    pub vertical_stiffness: f64,
    #[serde(default)]
    pub vertical_damping: f64,
    /// Longitudinal slip stiffness `C_κ` (N per unit slip) and cornering stiffness `C_α`
    /// (N/rad, positive).
    pub slip_stiffness: f64,
    pub cornering_stiffness: f64,
    /// Friction coefficient on the reference surface (asphalt).
    pub mu: f64,
    /// Rolling-resistance coefficient on the reference surface.
    #[serde(default = "default_rolling_resistance")]
    pub rolling_resistance: f64,
    /// Design load (N): sets the default low-speed damping.
    pub nominal_load: f64,
    /// Relaxation lengths (m) of the longitudinal and lateral slip.
    pub relaxation_x: f64,
    pub relaxation_y: f64,
    /// Speed below which the low-speed slip damping acts (m/s).
    #[serde(default = "default_vxlow")]
    pub vxlow: f64,
}

fn default_rolling_resistance() -> f64 {
    super::REFERENCE_ROLLING_RESISTANCE
}

fn default_vxlow() -> f64 {
    1.0
}

/// Forces of the brush model in the contact frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FialaOutput {
    pub fx: f64,
    pub fy: f64,
    pub mz: f64,
    pub trail: f64,
}

impl FialaParams {
    /// Check that the parameters are physical.
    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            ("radius", self.radius),
            ("width", self.width),
            ("vertical_stiffness", self.vertical_stiffness),
            ("slip_stiffness", self.slip_stiffness),
            ("cornering_stiffness", self.cornering_stiffness),
            ("mu", self.mu),
            ("nominal_load", self.nominal_load),
            ("relaxation_x", self.relaxation_x),
            ("relaxation_y", self.relaxation_y),
            ("vxlow", self.vxlow),
        ];
        for (name, v) in positive {
            if !(v > 0.0 && v.is_finite()) {
                return Err(format!("{name} must be positive, got {v}"));
            }
        }
        for (name, v) in [("vertical_damping", self.vertical_damping), ("rolling_resistance", self.rolling_resistance)]
        {
            if !(v >= 0.0 && v.is_finite()) {
                return Err(format!("{name} must be non-negative, got {v}"));
            }
        }
        Ok(())
    }

    /// Contact-patch half length (m) at deflection `rho`: `√(R ρ)`, about 0.7 of the chord
    /// where the undeformed tyre would cut the road.
    pub fn half_length(&self, rho: f64) -> f64 {
        (self.radius * rho.max(0.0)).sqrt()
    }

    /// Steady-state forces at load `fz`, slip `kappa`, `tan_alpha`, patch half length `a` and
    /// friction scale `mu_scale`.
    pub fn eval(&self, fz: f64, kappa: f64, tan_alpha: f64, a: f64, mu_scale: f64) -> FialaOutput {
        if fz <= 0.0 {
            return FialaOutput::default();
        }
        let sx = self.slip_stiffness * kappa;
        let sy = -self.cornering_stiffness * tan_alpha;
        let s = sx.hypot(sy);
        if s == 0.0 {
            return FialaOutput { trail: a / 3.0, ..Default::default() };
        }
        let limit = self.mu * mu_scale * fz;
        let z = s / (3.0 * limit);
        let (f, trail) = if z < 1.0 {
            let w = 1.0 - z;
            (limit * z * (3.0 - 3.0 * z + z * z), a / 3.0 * w * w * w / (1.0 - z + z * z / 3.0))
        } else {
            (limit, 0.0)
        };
        let (fx, fy) = (f * sx / s, f * sy / s);
        FialaOutput { fx, fy, mz: -trail * fy, trail }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tyre() -> FialaParams {
        toml::from_str(
            "radius = 0.1\nwidth = 0.05\nvertical_stiffness = 4e4\nslip_stiffness = 800\n\
             cornering_stiffness = 600\nmu = 0.9\nnominal_load = 50\nrelaxation_x = 0.05\nrelaxation_y = 0.06\n",
        )
        .unwrap()
    }

    #[test]
    fn stiffness_saturation_trail_and_friction_circle() {
        let t = tyre();
        t.validate().unwrap();
        let (fz, a) = (50.0, 0.02);
        let h = 1e-7;
        // Initial slopes are the slip and cornering stiffnesses; the trail starts at a/3.
        assert!((t.eval(fz, h, 0.0, a, 1.0).fx / h - 800.0).abs() < 1e-3);
        let lat = t.eval(fz, 0.0, h, a, 1.0);
        assert!((lat.fy / h + 600.0).abs() < 1e-3);
        assert!((lat.trail - a / 3.0).abs() < 1e-8 && lat.mz > 0.0);
        // Full sliding at z = 1 (|s| = 3 μ F_z) and beyond, without aligning moment.
        let k_slide = 3.0 * 0.9 * fz / 800.0;
        for k in [k_slide, 2.0 * k_slide] {
            let o = t.eval(fz, k, 0.0, a, 1.0);
            assert!((o.fx - 0.9 * fz).abs() < 1e-9 && o.mz == 0.0);
        }
        // The friction scale moves the limit.
        assert!((t.eval(fz, 1.0, 0.0, a, 0.5).fx - 0.45 * fz).abs() < 1e-9);
        // Combined slip: the force follows the slip direction with the pure-slip magnitude.
        let (k, ta) = (0.02, -0.03);
        let o = t.eval(fz, k, ta, a, 1.0);
        let (sx, sy) = (800.0 * k, 600.0 * 0.03);
        let z = sx.hypot(sy) / (3.0 * 0.9 * fz);
        let f = 0.9 * fz * (3.0 * z - 3.0 * z * z + z * z * z);
        assert!((o.fx.hypot(o.fy) - f).abs() < 1e-9);
        assert!((o.fy / o.fx - sy / sx).abs() < 1e-12);
        assert!(o.fx.hypot(o.fy) <= 0.9 * fz);
        // Trail against the brush-model moment μ F_z a z (1 − z)³ in pure side slip.
        let o = t.eval(fz, 0.0, 0.05, a, 1.0);
        let z = 600.0 * 0.05 / (3.0 * 0.9 * fz);
        assert!((o.mz - 0.9 * fz * a * z * (1.0 - z).powi(3)).abs() < 1e-9);
    }

    #[test]
    fn rejects_unphysical_parameters() {
        let mut t = tyre();
        t.mu = 0.0;
        assert!(t.validate().is_err());
        assert!(toml::from_str::<FialaParams>("radius = 0.1\nbogus = 1\n").is_err());
    }
}
