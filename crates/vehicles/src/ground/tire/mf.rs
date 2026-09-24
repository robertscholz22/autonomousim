//! Pacejka Magic Formula, steady state: MF 5.2 and MF 6.1/6.2 behind a version switch, as in
//! MFeval (Furlan) and its equation references — Pacejka, *Tire and Vehicle Dynamics*, 3rd ed.
//! (2012), eqs. (4.E1–4.E78), Besselink et al. (2010) and the MF-Tyre 5.2 manual. Turn slip
//! is not modelled (ζ = 1); the input limits and low-speed reductions of MFeval are left to
//! the caller (the transient slip model handles standstill).
//!
//! Sign conventions are the TNO/ISO-W ones used by `.tir` files: `κ = −V_sx/|V_cx|`,
//! `tan α = V_sy/|V_cx|` (so the cornering stiffness `K_yα` is negative), forces and moments
//! in the contact frame (x forward, z up).

use super::tir::{TirError, TirFile};
use std::f64::consts::FRAC_2_PI;

/// Magic Formula version (the equations that differ between them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MfVersion {
    /// MF 5.2 / PAC2002 (FITTYP 5, 6 or 21).
    V52,
    /// MF 6.1 (FITTYP 61), and MF 6.2 (62), which differs only in the loaded radius.
    V61,
}

macro_rules! mf_params {
    ($($field:ident = $default:expr),* $(,)?) => {
        /// Coefficients of a `.tir` file (the names are the file's keys in lower case).
        /// Missing coefficients default to values that switch their term off (scaling
        /// factors to 1); see [`REQUIRED`] for the ones that must be given.
        #[derive(Clone, Debug, PartialEq)]
        pub struct MfParams {
            pub version: MfVersion,
            $(pub $field: f64,)*
        }

        impl MfParams {
            /// Every coefficient as `(file key, value)`.
            pub fn values(&self) -> Vec<(String, f64)> {
                vec![$((stringify!($field).to_ascii_uppercase(), self.$field),)*]
            }

            fn read_values(version: MfVersion, file: &TirFile) -> Result<Self, TirError> {
                Ok(Self {
                    version,
                    $($field: file.number(&stringify!($field).to_ascii_uppercase())?.unwrap_or($default),)*
                })
            }
        }
    };
}

/// Coefficients without which the model is meaningless.
pub const REQUIRED: &[&str] = &[
    "UNLOADED_RADIUS",
    "FNOMIN",
    "VERTICAL_STIFFNESS",
    "PCX1",
    "PDX1",
    "PKX1",
    "PCY1",
    "PDY1",
    "PKY1",
    "PKY2",
    "QBZ1",
    "QCZ1",
    "QDZ1",
];

mf_params! {
    // [MODEL]
    longvl = 16.7, vxlow = 1.0,
    // [DIMENSION]
    unloaded_radius = 0.0, width = 0.2, aspect_ratio = 0.6, rim_radius = 0.0, rim_width = 0.0,
    // [OPERATING_CONDITIONS] (MF 6.x)
    inflpres = 0.0, nompres = 0.0,
    // [INERTIA]
    mass = 0.0, ixx = 0.0, iyy = 0.0,
    // [VERTICAL]
    fnomin = 0.0, vertical_stiffness = 0.0, vertical_damping = 0.0,
    breff = 0.0, dreff = 0.0, freff = 0.0, q_re0 = 1.0, q_v1 = 0.0, q_v2 = 0.0,
    q_fz2 = 0.0, q_fcx = 0.0, q_fcy = 0.0, pfz1 = 0.0, bottom_offst = 0.0, bottom_stiff = 0.0,
    // [STRUCTURAL] (MF 6.x)
    longitudinal_stiffness = 0.0, lateral_stiffness = 0.0,
    pcfx1 = 0.0, pcfx2 = 0.0, pcfx3 = 0.0, pcfy1 = 0.0, pcfy2 = 0.0, pcfy3 = 0.0,
    // [CONTACT_PATCH]
    q_ra1 = 0.0, q_ra2 = 0.0, q_rb1 = 0.0, q_rb2 = 0.0, q_a1 = 0.0, q_a2 = 0.0,
    // [VERTICAL_FORCE_RANGE]
    fzmin = 0.0,
    // [SCALING_COEFFICIENTS]
    lfzo = 1.0, lcx = 1.0, lmux = 1.0, lex = 1.0, lkx = 1.0, lhx = 1.0, lvx = 1.0,
    lcy = 1.0, lmuy = 1.0, ley = 1.0, lky = 1.0, lhy = 1.0, lvy = 1.0, ltr = 1.0, lres = 1.0,
    lxal = 1.0, lyka = 1.0, lvyka = 1.0, ls = 1.0, lkyc = 1.0, lkzc = 1.0, lvmx = 1.0,
    lmx = 1.0, lmy = 1.0, lsgkp = 1.0, lsgal = 1.0, lcz = 1.0,
    // [LONGITUDINAL_COEFFICIENTS]
    pcx1 = 0.0, pdx1 = 0.0, pdx2 = 0.0, pdx3 = 0.0, pex1 = 0.0, pex2 = 0.0, pex3 = 0.0,
    pex4 = 0.0, pkx1 = 0.0, pkx2 = 0.0, pkx3 = 0.0, phx1 = 0.0, phx2 = 0.0, pvx1 = 0.0,
    pvx2 = 0.0, ppx1 = 0.0, ppx2 = 0.0, ppx3 = 0.0, ppx4 = 0.0, rbx1 = 0.0, rbx2 = 0.0,
    rbx3 = 0.0, rcx1 = 1.0, rex1 = 0.0, rex2 = 0.0, rhx1 = 0.0,
    ptx1 = 0.0, ptx2 = 0.0, ptx3 = 0.0,
    // [OVERTURNING_COEFFICIENTS]
    qsx1 = 0.0, qsx2 = 0.0, qsx3 = 0.0, qsx4 = 0.0, qsx5 = 0.0, qsx6 = 0.0, qsx7 = 0.0,
    qsx8 = 0.0, qsx9 = 0.0, qsx10 = 0.0, qsx11 = 0.0, qsx12 = 0.0, qsx13 = 0.0, qsx14 = 0.0,
    ppmx1 = 0.0,
    // [LATERAL_COEFFICIENTS]
    pcy1 = 0.0, pdy1 = 0.0, pdy2 = 0.0, pdy3 = 0.0, pey1 = 0.0, pey2 = 0.0, pey3 = 0.0,
    pey4 = 0.0, pey5 = 0.0, pky1 = 0.0, pky2 = 0.0, pky3 = 0.0, pky4 = 2.0, pky5 = 0.0,
    pky6 = 0.0, pky7 = 0.0, phy1 = 0.0, phy2 = 0.0, phy3 = 0.0, pvy1 = 0.0, pvy2 = 0.0,
    pvy3 = 0.0, pvy4 = 0.0, ppy1 = 0.0, ppy2 = 0.0, ppy3 = 0.0, ppy4 = 0.0, ppy5 = 0.0,
    rby1 = 0.0, rby2 = 0.0, rby3 = 0.0, rby4 = 0.0, rcy1 = 1.0, rey1 = 0.0, rey2 = 0.0,
    rhy1 = 0.0, rhy2 = 0.0, rvy1 = 0.0, rvy2 = 0.0, rvy3 = 0.0, rvy4 = 0.0, rvy5 = 0.0,
    rvy6 = 0.0, pty1 = 0.0, pty2 = 0.0,
    // [ROLLING_COEFFICIENTS]
    qsy1 = 0.0, qsy2 = 0.0, qsy3 = 0.0, qsy4 = 0.0, qsy5 = 0.0, qsy6 = 0.0, qsy7 = 0.0,
    qsy8 = 0.0,
    // [ALIGNING_COEFFICIENTS]
    qbz1 = 0.0, qbz2 = 0.0, qbz3 = 0.0, qbz4 = 0.0, qbz5 = 0.0, qbz9 = 0.0, qbz10 = 0.0,
    qcz1 = 0.0, qdz1 = 0.0, qdz2 = 0.0, qdz3 = 0.0, qdz4 = 0.0, qdz6 = 0.0, qdz7 = 0.0,
    qdz8 = 0.0, qdz9 = 0.0, qdz10 = 0.0, qdz11 = 0.0, qez1 = 0.0, qez2 = 0.0, qez3 = 0.0,
    qez4 = 0.0, qez5 = 0.0, qhz1 = 0.0, qhz2 = 0.0, qhz3 = 0.0, qhz4 = 0.0, ppz1 = 0.0,
    ppz2 = 0.0, ssz1 = 0.0, ssz2 = 0.0, ssz3 = 0.0, ssz4 = 0.0,
}

/// MF 5.2 camber scaling factors that MF 6.x (and this implementation) dropped; files must
/// leave them at 1.
const UNSUPPORTED_SCALINGS: [&str; 4] = ["LGAX", "LGAY", "LGAZ", "LKG"];

impl MfParams {
    pub fn from_tir(file: &TirFile) -> Result<Self, TirError> {
        let version = match file.number("FITTYP")? {
            Some(v) if [5.0, 6.0, 21.0].contains(&v) => MfVersion::V52,
            Some(v) if v == 61.0 || v == 62.0 => MfVersion::V61,
            Some(v) => return Err(TirError::Unsupported(format!("FITTYP {v}"))),
            None => match file.text("PROPERTY_FILE_FORMAT").map(str::to_ascii_uppercase).as_deref() {
                Some("PAC2002" | "MF_05") => MfVersion::V52,
                other => return Err(TirError::Unsupported(format!("no FITTYP, property file format {other:?}"))),
            },
        };
        for key in REQUIRED {
            if file.number(key)?.is_none() {
                return Err(TirError::Missing(key));
            }
        }
        for key in UNSUPPORTED_SCALINGS {
            if let Some(v) = file.number(key)?
                && v != 1.0
            {
                return Err(TirError::Invalid { key: key.into(), msg: format!("{v} (only 1 is supported)") });
            }
        }
        let mut p = Self::read_values(version, file)?;
        if version == MfVersion::V52 {
            // Pressure terms are MF 6.x; PAC2002 files name the pressures IP/IP_NOM.
            p.nompres = file.number("IP_NOM")?.unwrap_or(1.0);
            p.inflpres = file.number("IP")?.unwrap_or(p.nompres);
        }
        if p.nompres <= 0.0 {
            return Err(TirError::Invalid { key: "NOMPRES".into(), msg: "must be positive".into() });
        }
        if p.inflpres <= 0.0 {
            p.inflpres = p.nompres;
        }
        for (key, v) in [("UNLOADED_RADIUS", p.unloaded_radius), ("FNOMIN", p.fnomin), ("LONGVL", p.longvl)] {
            if v <= 0.0 {
                return Err(TirError::Invalid { key: key.into(), msg: "must be positive".into() });
            }
        }
        Ok(p)
    }

    pub fn read(path: impl AsRef<std::path::Path>) -> Result<Self, TirError> {
        Self::from_tir(&TirFile::read(path)?)
    }

    /// A complete `.tir` file with every coefficient (for oracles with other defaults).
    pub fn to_tir(&self) -> String {
        let fittyp = match self.version {
            MfVersion::V52 => 6,
            MfVersion::V61 => 61,
        };
        let mut s = format!(
            "[UNITS]\nLENGTH = 'meter'\nFORCE = 'newton'\nANGLE = 'radians'\nMASS = 'kg'\nTIME = 'second'\n\
             [MODEL]\nFITTYP = {fittyp}\n[COEFFICIENTS]\n"
        );
        for (key, v) in self.values() {
            s += &format!("{key} = {v:e}\n");
        }
        s
    }
}

/// Operating point of one evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MfInput {
    /// Vertical load (N).
    pub fz: f64,
    /// Longitudinal slip `κ`.
    pub kappa: f64,
    /// Lateral slip `α*`: `tan α` (MF 6.x convention) or `α`.
    pub alpha: f64,
    /// `cos α'` of the contact-point velocity's angle to the wheel plane.
    pub cos_alpha: f64,
    /// Inclination `γ` and its measure `γ*` (`sin γ` or `γ`, matching `alpha`).
    pub gamma: f64,
    pub gamma_star: f64,
    /// Longitudinal speed of the contact centre `|V_cx|` (m/s; rolling resistance).
    pub vx: f64,
    /// Inflation pressure (Pa; MF 6.x), or `None` for the file's.
    pub pressure: Option<f64>,
    /// Road friction relative to the tyre's test surface (scales λ_μx, λ_μy).
    pub mu_scale: f64,
}

impl MfInput {
    /// The operating point for slip angle `alpha` and inclination `gamma` in the MF 6.x
    /// convention (`α* = tan α`, `γ* = sin γ`).
    pub fn new(fz: f64, kappa: f64, alpha: f64, gamma: f64, vx: f64) -> Self {
        Self {
            fz,
            kappa,
            alpha: alpha.tan(),
            cos_alpha: alpha.cos(),
            gamma,
            gamma_star: gamma.sin(),
            vx,
            pressure: None,
            mu_scale: 1.0,
        }
    }
}

/// Forces and moments in the contact frame, and characteristic quantities.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MfOutput {
    pub fx: f64,
    pub fy: f64,
    pub mx: f64,
    /// Rolling resistance moment for forward rolling (≤ 0 for positive coefficients).
    pub my: f64,
    pub mz: f64,
    /// Longitudinal slip stiffness `K_xκ` (N) and cornering stiffness `K_yα` (N/rad).
    pub kxk: f64,
    pub kya: f64,
    /// Peak friction coefficients.
    pub mux: f64,
    pub muy: f64,
    /// Pneumatic trail (m) and residual aligning moment (N m).
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

/// `D sin(C atan(B x − E (B x − atan(B x))))`.
#[inline]
fn magic(b: f64, c: f64, d: f64, e: f64, x: f64) -> f64 {
    let bx = b * x;
    d * (c * (bx - e * (bx - bx.atan())).atan()).sin()
}

/// `cos(C atan(B x − E (B x − atan(B x))))` (the combined-slip weighting shape).
#[inline]
fn weight(b: f64, c: f64, e: f64, x: f64) -> f64 {
    let bx = b * x;
    (c * (bx - e * (bx - bx.atan())).atan()).cos()
}

/// Pure lateral-force curve (4.E19–4.E30).
struct LateralCurve {
    kya: f64,
    shy: f64,
    svy: f64,
    by: f64,
    cy: f64,
    dy: f64,
    muy: f64,
    ey_base: f64,
    ey_sym: f64,
    ey_asym: f64,
}

impl LateralCurve {
    fn eval(&self, alpha: f64) -> f64 {
        let ay = alpha + self.shy;
        let ey = (self.ey_base * (self.ey_sym - self.ey_asym * sgn(ay))).min(1.0);
        magic(self.by, self.cy, self.dy, ey, ay) + self.svy
    }
}

impl MfParams {
    #[inline]
    fn dpi(&self, pressure: Option<f64>) -> f64 {
        match self.version {
            MfVersion::V52 => 0.0,
            MfVersion::V61 => (pressure.unwrap_or(self.inflpres) - self.nompres) / self.nompres,
        }
    }

    fn lateral(&self, fz: f64, dfz: f64, dpi: f64, gs: f64, lmuy: f64) -> LateralCurve {
        let fz0p = self.lfzo * self.fnomin;
        let kya = self.pky1
            * fz0p
            * (1.0 + self.ppy1 * dpi)
            * (1.0 - self.pky3 * gs.abs())
            * (self.pky4 * ((fz / fz0p) / ((self.pky2 + self.pky5 * gs * gs) * (1.0 + self.ppy2 * dpi))).atan()).sin()
            * self.lky;
        let svyg = fz * (self.pvy3 + self.pvy4 * dfz) * gs * self.lkyc * lmuy;
        let shy = match self.version {
            MfVersion::V52 => (self.phy1 + self.phy2 * dfz) * self.lhy + self.phy3 * gs * self.lkyc,
            MfVersion::V61 => {
                let kyg0 = fz * (self.pky6 + self.pky7 * dfz) * (1.0 + self.ppy5 * dpi) * self.lkyc;
                (self.phy1 + self.phy2 * dfz) * self.lhy + (kyg0 * gs - svyg) / (kya + EPS * sgn(kya))
            }
        };
        let svy = fz * (self.pvy1 + self.pvy2 * dfz) * self.lvy * lmuy + svyg;
        let cy = self.pcy1 * self.lcy;
        let muy = if fz == 0.0 {
            0.0
        } else {
            (self.pdy1 + self.pdy2 * dfz)
                * (1.0 + self.ppy3 * dpi + self.ppy4 * dpi * dpi)
                * (1.0 - self.pdy3 * gs * gs)
                * lmuy
        };
        let dy = muy * fz;
        LateralCurve {
            kya,
            shy,
            svy,
            by: kya / (cy * dy + EPS * sgn(dy)),
            cy,
            dy,
            muy,
            ey_base: (self.pey1 + self.pey2 * dfz) * self.ley,
            ey_sym: 1.0 + self.pey5 * gs * gs,
            ey_asym: self.pey3 + self.pey4 * gs,
        }
    }

    /// Steady-state forces and moments (combined slip).
    pub fn eval(&self, inp: &MfInput) -> MfOutput {
        let fz = inp.fz;
        if fz <= 0.0 {
            return MfOutput::default();
        }
        let (kappa, a, g, gs) = (inp.kappa, inp.alpha, inp.gamma, inp.gamma_star);
        let (r0, fz0) = (self.unloaded_radius, self.fnomin);
        let fz0p = self.lfzo * fz0;
        let dfz = (fz - fz0p) / fz0p;
        let dpi = self.dpi(inp.pressure);
        let lmux = self.lmux * inp.mu_scale;
        let lmuy = self.lmuy * inp.mu_scale;

        // Pure longitudinal slip (4.E9–4.E18).
        let cx = self.pcx1 * self.lcx;
        let mux = (self.pdx1 + self.pdx2 * dfz)
            * (1.0 + self.ppx3 * dpi + self.ppx4 * dpi * dpi)
            * (1.0 - self.pdx3 * g * g)
            * lmux;
        let dx = mux * fz;
        let kxk = fz
            * (self.pkx1 + self.pkx2 * dfz)
            * (self.pkx3 * dfz).exp()
            * (1.0 + self.ppx1 * dpi + self.ppx2 * dpi * dpi)
            * self.lkx;
        let bx = kxk / (cx * dx + EPS * sgn(dx));
        let shx = (self.phx1 + self.phx2 * dfz) * self.lhx;
        let svx = fz * (self.pvx1 + self.pvx2 * dfz) * self.lvx * lmux;
        let kx = kappa + shx;
        let ex =
            ((self.pex1 + self.pex2 * dfz + self.pex3 * dfz * dfz) * (1.0 - self.pex4 * sgn(kx)) * self.lex).min(1.0);
        let fx0 = magic(bx, cx, dx, ex, kx) + svx;

        // Pure lateral slip.
        let lat = self.lateral(fz, dfz, dpi, gs, lmuy);
        let fy0 = lat.eval(a);

        // Combined slip (4.E50–4.E67).
        let exa = (self.rex1 + self.rex2 * dfz).min(1.0);
        let bxa = (self.rbx1 + self.rbx3 * gs * gs) * (self.rbx2 * kappa).atan().cos() * self.lxal;
        let gxa = weight(bxa, self.rcx1, exa, a + self.rhx1) / weight(bxa, self.rcx1, exa, self.rhx1);
        let fx = gxa * fx0;

        let dvyk = lat.muy * fz * (self.rvy1 + self.rvy2 * dfz + self.rvy3 * gs) * (self.rvy4 * a).atan().cos();
        let svyk = dvyk * (self.rvy5 * (self.rvy6 * kappa).atan()).sin() * self.lvyka;
        let shyk = self.rhy1 + self.rhy2 * dfz;
        let eyk = (self.rey1 + self.rey2 * dfz).min(1.0);
        let byk = (self.rby1 + self.rby4 * gs * gs) * (self.rby2 * (a - self.rby3)).atan().cos() * self.lyka;
        let gyk = weight(byk, self.rcy1, eyk, kappa + shyk) / weight(byk, self.rcy1, eyk, shyk);
        let fy = gyk * fy0 + svyk;

        // Aligning moment (4.E31–4.E49, 4.E71–4.E78 with MFeval's corrections).
        let kya_p = lat.kya + EPS * sgn(lat.kya);
        let alpha_r = a + lat.shy + lat.svy / kya_p;
        let alpha_t = a + self.qhz1 + self.qhz2 * dfz + (self.qhz3 + self.qhz4 * dfz) * gs;
        let bt = (self.qbz1 + self.qbz2 * dfz + self.qbz3 * dfz * dfz)
            * (1.0 + self.qbz4 * g + self.qbz5 * g.abs())
            * self.lky
            / lmuy;
        let ct = self.qcz1;
        let dt = (self.qdz1 + self.qdz2 * dfz)
            * (1.0 - self.ppz1 * dpi)
            * (1.0 + self.qdz3 * g + self.qdz4 * g * g)
            * fz
            * (r0 / fz0p)
            * self.ltr;
        let et = ((self.qez1 + self.qez2 * dfz + self.qez3 * dfz * dfz)
            * (1.0 + (self.qez4 + self.qez5 * gs) * FRAC_2_PI * (bt * ct * alpha_t).atan()))
        .min(1.0);
        let br = self.qbz9 * self.lky / lmuy + self.qbz10 * lat.by * lat.cy;
        let dr = fz
            * r0
            * ((self.qdz6 + self.qdz7 * dfz) * self.lres
                + ((self.qdz8 + self.qdz9 * dfz) * (1.0 + self.ppz2 * dpi)
                    + (self.qdz10 + self.qdz11 * dfz) * gs.abs())
                    * gs
                    * self.lkzc)
            * lmuy
            * a.cos();
        let k2 = (kxk / kya_p).powi(2) * kappa * kappa;
        let equivalent = |x: f64| (x.tan().powi(2) + k2).sqrt().atan() * sgn(x);
        let (ar_eq, at_eq) = (equivalent(alpha_r), equivalent(alpha_t));
        let s = r0 * (self.ssz1 + self.ssz2 * (fy / fz0) + (self.ssz3 + self.ssz4 * dfz) * g) * self.ls;
        let mzr = dr * (br * ar_eq).atan().cos();
        let trail = dt * weight(bt, ct, et, at_eq) * inp.cos_alpha * self.lfzo;
        let mz = match self.version {
            MfVersion::V52 => -trail * (fy - svyk) + mzr + s * fx,
            MfVersion::V61 => {
                // Fy' = Gyk · Fy0(γ = 0).
                let fy0_g0 = if gs == 0.0 { fy0 } else { self.lateral(fz, dfz, dpi, 0.0, lmuy).eval(a) };
                -trail * gyk * fy0_g0 + mzr + s * fx
            }
        };

        // Overturning and rolling resistance moments.
        let fzmin = self.fzmin;
        let fz_mx = if fz < fzmin { fz * (fz / fzmin).powi(2) } else { fz };
        let mx = r0
            * fz_mx
            * self.lmx
            * (self.qsx1 * self.lvmx - self.qsx2 * g * (1.0 + self.ppmx1 * dpi)
                + self.qsx3 * (fy / fz0)
                + self.qsx4
                    * (self.qsx5 * (self.qsx6 * fz_mx / fz0).powi(2).atan()).cos()
                    * (self.qsx7 * g + self.qsx8 * (self.qsx9 * fy / fz0).atan()).sin()
                + self.qsx10 * (self.qsx11 * fz_mx / fz0).atan() * g)
            + r0 * self.lmx * (fy * (self.qsx13 + self.qsx14 * g.abs()) - fz_mx * self.qsx12 * g * g.abs());
        let fz_my = if fz < fzmin { fz * (fz / fzmin) } else { fz };
        let v = inp.vx.abs() / self.longvl;
        let rolling = self.qsy1 + self.qsy2 * (fx / fz0) + self.qsy3 * v + self.qsy4 * v.powi(4);
        let my = match self.version {
            MfVersion::V52 => -r0 * fz_my * self.lmy * rolling,
            MfVersion::V61 => {
                let p = inp.pressure.unwrap_or(self.inflpres);
                -r0 * fz0
                    * self.lmy
                    * (rolling + (self.qsy5 + self.qsy6 * fz_my / fz0) * g * g)
                    * ((fz_my / fz0).powf(self.qsy7) * (p / self.nompres).powf(self.qsy8))
            }
        };

        let (sigma_x, sigma_y) = self.relaxation_lengths(fz, g, dpi, kxk, lat.kya);
        MfOutput { fx, fy, mx, my, mz, kxk, kya: lat.kya, mux, muy: lat.muy, trail, mzr, sigma_x, sigma_y }
    }

    fn relaxation_lengths(&self, fz: f64, gamma: f64, dpi: f64, kxk: f64, kya: f64) -> (f64, f64) {
        let fz0p = self.lfzo * self.fnomin;
        let dfz = (fz - fz0p) / fz0p;
        let r0 = self.unloaded_radius;
        match self.version {
            MfVersion::V52 => (
                (self.ptx1 + self.ptx2 * dfz) * (-self.ptx3 * dfz).exp() * self.lsgkp * r0 * fz / self.fnomin,
                self.pty1
                    * (2.0 * (fz / (self.pty2 * fz0p)).atan()).sin()
                    * (1.0 - self.pky3 * gamma.abs())
                    * r0
                    * self.lfzo
                    * self.lsgal,
            ),
            MfVersion::V61 => {
                let cx = self.longitudinal_stiffness
                    * (1.0 + self.pcfx1 * dfz + self.pcfx2 * dfz * dfz)
                    * (1.0 + self.pcfx3 * dpi);
                let cy = self.lateral_stiffness
                    * (1.0 + self.pcfy1 * dfz + self.pcfy2 * dfz * dfz)
                    * (1.0 + self.pcfy3 * dpi);
                ((kxk / cx).abs(), (kya / cy).abs())
            }
        }
    }

    /// Effective rolling radius `R_e` at wheel speed `omega` (rad/s) and load `fz`
    /// (Besselink eqs. 1 and 7).
    pub fn effective_radius(&self, fz: f64, omega: f64, pressure: Option<f64>) -> f64 {
        let r0 = self.unloaded_radius;
        let fz0 = self.fnomin;
        let cz = self.vertical_stiffness * self.lcz * (1.0 + self.pfz1 * self.dpi(pressure));
        let r_omega = r0 * (self.q_re0 + self.q_v1 * (omega * r0 / self.longvl).powi(2));
        let fz = fz.max(0.0);
        r_omega - (fz0 / cz) * (self.dreff * (self.breff * fz / fz0).atan() + self.freff * fz / fz0)
    }

    /// Vertical load for tyre deflection `rho` (m) (Besselink eq. 4, without the speed and
    /// in-plane force corrections of MF 6.x, which the contact model adds).
    pub fn vertical_force(&self, rho: f64, pressure: Option<f64>) -> f64 {
        if rho <= 0.0 {
            return 0.0;
        }
        let (r0, fz0) = (self.unloaded_radius, self.fnomin);
        let cz = self.vertical_stiffness * self.lcz;
        let q_fz1 = ((cz * r0 / fz0).powi(2) - 4.0 * self.q_fz2).max(0.0).sqrt();
        let x = rho / r0;
        (1.0 + self.pfz1 * self.dpi(pressure)) * (q_fz1 * x + self.q_fz2 * x * x) * fz0
    }

    /// Half length of the contact patch (m) at load `fz`.
    pub fn contact_half_length(&self, fz: f64, pressure: Option<f64>) -> f64 {
        let (r0, fz0) = (self.unloaded_radius, self.fnomin);
        let cz = self.vertical_stiffness * self.lcz * (1.0 + self.pfz1 * self.dpi(pressure));
        // Beyond bottoming, the patch stops growing.
        let bottom = r0 - self.rim_radius - self.bottom_offst;
        let fz = fz.max(0.0).min(if bottom > 0.0 { bottom * cz } else { f64::INFINITY });
        if self.version == MfVersion::V61 {
            let n = fz / (cz * r0);
            return r0 * (self.q_ra2 * n + self.q_ra1 * n.sqrt());
        }
        let (qa1, qa2) = if self.q_a1 != 0.0 || self.q_a2 != 0.0 {
            (self.q_a1, self.q_a2)
        } else {
            // Default fit to R0 C_z / F_z0 (MF-Tyre 5.2).
            let y = (r0 * self.vertical_stiffness * self.lcz / fz0).log10();
            let qa1 = -0.0388 * y.powi(3) + 0.2509 * y * y - 0.6283 * y + 0.6279;
            (qa1, 1.693 * qa1 * qa1)
        };
        r0 * (qa2 * (fz / fz0) + qa1 * (fz / fz0).sqrt())
    }
}
