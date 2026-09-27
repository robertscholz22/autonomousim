//! Motorcycle Magic Formula (MF-MC, Pacejka 2002 §11.5 as fitted and completed by Evangelou,
//! *Control and stability analysis of two-wheeled road vehicles*, PhD thesis, Imperial College
//! London, 2004, ch. 9 and appendix C): a wide tyre with a circular crown, where camber
//! produces thrust through its own sine term (`C_γ`) instead of MF 6.x's shifts, and the
//! overturning moment follows from the contact point moving around the crown (no `M_x`).
//!
//! The equations use the thesis's conventions: side slip `β = −V_y/|V_x|` with forces and
//! moments in SAE axes (y right, z down), camber `γ` positive leaning right (as ISO-W). Here
//! they are evaluated with `β = tan α` (the ISO-W lateral slip, whose sign is that of the
//! thesis's β in SAE axes) and `F_y`, `M_z` change sign to ISO-W (y left, z up).
//! Longitudinal slip `κ` and `F_x` are as in the Magic Formula.
//!
//! Only side slip relaxes, with `σ = K_yα0 (c₀ + c₁ V + c₂ V²)` (thesis §9.3.10; camber acts
//! instantaneously); the longitudinal relaxation length is a parameter.

use serde::{Deserialize, Serialize};
use std::f64::consts::FRAC_2_PI;

/// Coefficients of an MF-MC tyre (the thesis's names in lower case; see appendix C).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McParams {
    /// Where the coefficients come from.
    #[serde(default)]
    pub source: String,
    /// Unloaded radius, section width and crown radius (m): the crown radius is also `R_o` in
    /// the aligning-moment equations.
    pub radius: f64,
    pub width: f64,
    pub crown_radius: f64,
    /// Radial stiffness (N/m) and damping (N s/m).
    pub vertical_stiffness: f64,
    #[serde(default)]
    pub vertical_damping: f64,
    /// Nominal load `F_zo` (N).
    pub fzo: f64,
    /// Rolling-resistance coefficient on the reference surface.
    #[serde(default = "default_rolling_resistance")]
    pub rolling_resistance: f64,
    /// Side-slip relaxation length over cornering stiffness, `σ/K_yα0 = c₀ + c₁ V + c₂ V²`
    /// (m/N, V in m/s), and the longitudinal relaxation length (m).
    pub relaxation: [f64; 3],
    pub relaxation_x: f64,
    /// Speed below which the low-speed slip damping acts (m/s).
    #[serde(default = "default_vxlow")]
    pub vxlow: f64,
    // Longitudinal force (C.2.1).
    pub pcx1: f64,
    pub pdx1: f64,
    #[serde(default)]
    pub pdx2: f64,
    #[serde(default)]
    pub pex1: f64,
    #[serde(default)]
    pub pex2: f64,
    #[serde(default)]
    pub pex3: f64,
    #[serde(default)]
    pub pex4: f64,
    pub pkx1: f64,
    #[serde(default)]
    pub pkx2: f64,
    #[serde(default)]
    pub pkx3: f64,
    // Lateral force (C.2.2).
    pub pcy1: f64,
    pub pdy1: f64,
    #[serde(default)]
    pub pdy2: f64,
    #[serde(default)]
    pub pdy3: f64,
    #[serde(default)]
    pub pey1: f64,
    #[serde(default)]
    pub pey2: f64,
    #[serde(default)]
    pub pey3: f64,
    #[serde(default)]
    pub pey4: f64,
    pub pky1: f64,
    pub pky2: f64,
    pub pky3: f64,
    #[serde(default)]
    pub pky4: f64,
    #[serde(default)]
    pub pky5: f64,
    pub pcy2: f64,
    pub pky6: f64,
    #[serde(default)]
    pub pky7: f64,
    #[serde(default)]
    pub pey5: f64,
    // Aligning moment (C.2.3).
    #[serde(default)]
    pub qhz3: f64,
    #[serde(default)]
    pub qhz4: f64,
    pub qbz1: f64,
    #[serde(default)]
    pub qbz2: f64,
    #[serde(default)]
    pub qbz3: f64,
    #[serde(default)]
    pub qbz5: f64,
    #[serde(default)]
    pub qbz6: f64,
    pub qcz1: f64,
    pub qdz1: f64,
    #[serde(default)]
    pub qdz2: f64,
    #[serde(default)]
    pub qdz3: f64,
    #[serde(default)]
    pub qdz4: f64,
    #[serde(default)]
    pub qez1: f64,
    #[serde(default)]
    pub qez2: f64,
    #[serde(default)]
    pub qez3: f64,
    #[serde(default)]
    pub qez5: f64,
    #[serde(default)]
    pub qbz9: f64,
    #[serde(default)]
    pub qbz10: f64,
    #[serde(default)]
    pub qdz8: f64,
    #[serde(default)]
    pub qdz9: f64,
    #[serde(default)]
    pub qdz10: f64,
    #[serde(default)]
    pub qdz11: f64,
    // Combined slip (C.2.4, §9.3.6).
    pub rbx1: f64,
    #[serde(default)]
    pub rbx2: f64,
    pub rcx1: f64,
    pub rby1: f64,
    #[serde(default)]
    pub rby2: f64,
    #[serde(default)]
    pub rby3: f64,
    pub rcy1: f64,
}

fn default_rolling_resistance() -> f64 {
    super::REFERENCE_ROLLING_RESISTANCE
}

fn default_vxlow() -> f64 {
    1.0
}

/// Forces and moments in the contact frame (ISO-W) and characteristic quantities.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct McOutput {
    pub fx: f64,
    pub fy: f64,
    pub mz: f64,
    /// Slip stiffness `K_xκ` (N) and cornering stiffness (N/rad, ISO-W: negative).
    pub kxk: f64,
    pub kya: f64,
    /// Peak friction coefficients, pneumatic trail (m) and residual (twisting) moment (N m,
    /// ISO-W).
    pub mux: f64,
    pub muy: f64,
    pub trail: f64,
    pub mzr: f64,
    /// Relaxation lengths (m).
    pub sigma_x: f64,
    pub sigma_y: f64,
}

const EPS: f64 = 1e-6;

#[inline]
fn sgn(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { -1.0 }
}

/// `B x − E (B x − atan(B x))`.
#[inline]
fn shape(b: f64, e: f64, x: f64) -> f64 {
    let bx = b * x;
    bx - e * (bx - bx.atan())
}

impl McParams {
    /// Check that the parameters are physical.
    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            ("radius", self.radius),
            ("width", self.width),
            ("crown_radius", self.crown_radius),
            ("vertical_stiffness", self.vertical_stiffness),
            ("fzo", self.fzo),
            ("relaxation_x", self.relaxation_x),
            ("vxlow", self.vxlow),
            ("pcx1", self.pcx1),
            ("pdx1", self.pdx1),
            ("pkx1", self.pkx1),
            ("pcy1", self.pcy1),
            ("pdy1", self.pdy1),
            ("pky1", self.pky1),
            ("pky3", self.pky3),
            ("pcy2", self.pcy2),
            ("qbz1", self.qbz1),
            ("qcz1", self.qcz1),
            ("rbx1", self.rbx1),
            ("rcx1", self.rcx1),
            ("rby1", self.rby1),
            ("rcy1", self.rcy1),
        ];
        for (name, v) in positive {
            if !(v > 0.0 && v.is_finite()) {
                return Err(format!("{name} must be positive, got {v}"));
            }
        }
        if self.crown_radius > 0.5 * self.width || self.crown_radius >= self.radius {
            return Err(format!("crown_radius {} is larger than the tyre", self.crown_radius));
        }
        if self.pcy1 + self.pcy2 >= 2.0 {
            return Err(format!("pcy1 + pcy2 = {} must be below 2", self.pcy1 + self.pcy2));
        }
        for (name, v) in [("vertical_damping", self.vertical_damping), ("rolling_resistance", self.rolling_resistance)]
        {
            if !(v >= 0.0 && v.is_finite()) {
                return Err(format!("{name} must be non-negative, got {v}"));
            }
        }
        let sigma = self.eval(self.fzo, 0.0, 0.0, 0.0, 0.0, 1.0).sigma_y;
        if !(sigma > 0.0 && sigma.is_finite()) {
            return Err(format!("relaxation length {sigma} at nominal load is not positive"));
        }
        Ok(())
    }

    /// Cornering stiffness `K_yα` (N/rad, thesis sign: positive) at load `fz` and camber `g`.
    fn cornering_stiffness(&self, fz: f64, g: f64) -> f64 {
        let fzo = self.fzo;
        let kyao = self.pky1 * fzo * (self.pky2 * (fz / ((self.pky3 + self.pky4 * g * g) * fzo)).atan()).sin();
        kyao / (1.0 + self.pky5 * g * g)
    }

    /// Steady-state forces at load `fz` (N), longitudinal slip `kappa`, lateral slip
    /// `tan_alpha` (ISO-W), camber `gamma` (rad) and rolling speed `vx` (m/s, for the
    /// relaxation length), on a road with friction `mu_scale` times the reference.
    pub fn eval(&self, fz: f64, kappa: f64, tan_alpha: f64, gamma: f64, vx: f64, mu_scale: f64) -> McOutput {
        if fz <= 0.0 {
            return McOutput::default();
        }
        let (b, g) = (tan_alpha, gamma);
        let fzo = self.fzo;
        let dfz = (fz - fzo) / fzo;

        // Pure longitudinal slip.
        let cx = self.pcx1;
        let mux = (self.pdx1 + self.pdx2 * dfz) * mu_scale;
        let dx = mux * fz;
        let kxk = fz * (self.pkx1 + self.pkx2 * dfz) * (self.pkx3 * dfz).exp();
        let bx = kxk / (cx * dx + EPS);
        let ex = ((self.pex1 + self.pex2 * dfz + self.pex3 * dfz * dfz) * (1.0 - self.pex4 * sgn(kappa))).min(1.0);
        let fxo = dx * (cx * shape(bx, ex, kappa).atan()).sin();

        // Pure side slip and camber.
        let cy = self.pcy1;
        let lateral = |g: f64| {
            let muy = self.pdy1 * (self.pdy2 * dfz).exp() / (1.0 + self.pdy3 * g * g) * mu_scale;
            let dy = muy * fz;
            let kya = self.cornering_stiffness(fz, g);
            let by = kya / (cy * dy + EPS);
            let ey = (self.pey1 + self.pey2 * g * g + (self.pey3 + self.pey4 * g) * sgn(b)).min(1.0);
            let cg = self.pcy2;
            let kyg = (self.pky6 + self.pky7 * dfz) * fz;
            let bg = kyg / (cg * dy + EPS);
            let eg = self.pey5.min(1.0);
            let fyo = dy * (cy * shape(by, ey, b).atan() + cg * shape(bg, eg, g).atan()).sin();
            (fyo, muy, kya, by)
        };
        let (fyo, muy, kya, by) = lateral(g);
        let (fyoo, kyaoo) = if g == 0.0 { (fyo, kya) } else { (lateral(0.0).0, self.cornering_stiffness(fz, 0.0)) };

        // Combined slip; the loss functions stop at zero (the thesis fits hold to β ≈ 23°).
        let bxa = self.rbx1 * (self.rbx2 * kappa).atan().cos();
        let gxa = (self.rcx1 * (bxa * b).atan()).cos().max(0.0);
        let byk = self.rby1 * (self.rby2 * (b - self.rby3)).atan().cos();
        let gyk = (self.rcy1 * (byk * kappa).atan()).cos().max(0.0);
        let fx = gxa * fxo;
        let fy = gyk * fyo;

        // Aligning moment: pneumatic trail on the camber-free side force, residual (twisting)
        // moment from camber.
        let ro = self.crown_radius;
        let cos_a = 1.0 / (1.0 + b * b).sqrt();
        let bt =
            (self.qbz1 + self.qbz2 * dfz + self.qbz3 * dfz * dfz) * (1.0 + self.qbz5 * g.abs() + self.qbz6 * g * g);
        let ct = self.qcz1;
        let dt = fz * (ro / fzo) * (self.qdz1 + self.qdz2 * dfz) * (1.0 + self.qdz3 * g.abs() + self.qdz4 * g * g);
        let et = ((self.qez1 + self.qez2 * dfz + self.qez3 * dfz * dfz)
            * (1.0 + self.qez5 * g * FRAC_2_PI * (bt * ct * b).atan()))
        .min(1.0);
        let ar = b + (self.qhz3 + self.qhz4 * dfz) * g;
        let br = self.qbz9 + self.qbz10 * by * cy;
        let dr = fz * ro * ((self.qdz8 + self.qdz9 * dfz) * g + (self.qdz10 + self.qdz11 * dfz) * g * g.abs()) * cos_a;
        let k2 = (kxk / (kyaoo + EPS)).powi(2) * kappa * kappa;
        let lt = (b * b + k2).sqrt() * sgn(b);
        let lr = (ar * ar + k2).sqrt() * sgn(ar);
        let trail = dt * (ct * shape(bt, et, lt).atan()).cos() * cos_a;
        let mzr = dr / (1.0 + (br * lr).powi(2)).sqrt();
        let mz = -trail * gyk * fyoo + mzr;

        let v = vx.abs();
        let [c0, c1, c2] = self.relaxation;
        let sigma_y = kyaoo * (c0 + c1 * v + c2 * v * v);
        // ISO-W: y left, z up.
        McOutput {
            fx,
            fy: -fy,
            mz: -mz,
            kxk,
            kya: -kya,
            mux,
            muy,
            trail,
            mzr: -mzr,
            sigma_x: self.relaxation_x,
            sigma_y,
        }
    }
}
