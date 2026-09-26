//! A tyre as a force element: vertical load from the deflection on the road plane, transient
//! slip, the steady-state model (Magic Formula or Fiala) and the resulting wrench on the wheel.
//!
//! **Transient slip** (Pacejka 2012, §7.2): carcass deflections `u` and `v` lag the slip
//! with relaxation lengths `σ_κ` and `σ_α`,
//!
//! ```text
//! du/dt = −V_sx − |V_cx| u / σ_κ        κ' = u / σ_κ
//! dv/dt =  V_sy − |V_cx| v / σ_α        tan α' = v / σ_α
//! ```
//!
//! and the steady-state model is evaluated at `(κ', tan α')`. The decay term is integrated
//! implicitly, so short relaxation times at speed are stable at any step. At standstill the
//! deflections act as springs (a parked vehicle holds without creep). Below `VXLOW`, the
//! MF 6.1 low-speed damping `k_Vlow = ½ k_Vlow0 (1 + cos(π |V_cx| / VXLOW))` adds
//! `−k_Vlow V_sx / K_xκ` to `κ'` (and the lateral equivalent), a damping force `−k_Vlow V_s`
//! that still saturates with friction. The relaxation lengths and slip stiffnesses are those
//! of the previous step (one tick of lag).

use super::fiala::FialaParams;
use super::mf::{MfInput, MfParams};
use super::road::{RoadContact, road_contact};
use super::track::{TRACK_CELLS, TrackPatch};
use autonomousim_core::material::Material;
use autonomousim_core::terrain::Terrain;
use glam::DVec3;

/// Friction of the reference surface (asphalt in the standard material table): the road
/// friction a tyre's data describe.
pub const REFERENCE_FRICTION: f64 = 0.8;
/// Rolling-resistance coefficient of the reference surface.
pub const REFERENCE_ROLLING_RESISTANCE: f64 = 0.013;
/// Rolling speed (m/s) over which the rolling-resistance moment changes sign.
const ROLLING_SMOOTHING: f64 = 0.05;
/// Damping ratio of a corner's mass on the carcass spring that sets the default `k_Vlow0`.
const LOW_SPEED_DAMPING_RATIO: f64 = 0.25;
const GRAVITY: f64 = 9.80665;

/// The steady-state tyre model.
#[derive(Clone, Debug, PartialEq)]
pub enum TireModel {
    MagicFormula(Box<MfParams>),
    Fiala(FialaParams),
    /// A track patch under a road wheel (see [`super::track`]).
    Track(TrackPatch),
}

/// A tyre and its fixed operating conditions.
#[derive(Clone, Debug, PartialEq)]
pub struct Tire {
    pub model: TireModel,
    /// Inflation pressure (Pa; MF 6.x), or `None` for the file's.
    pub pressure: Option<f64>,
    /// Low-speed damping `k_Vlow0` (N s/m), longitudinal and lateral.
    pub low_speed_damping: [f64; 2],
    /// MF files without rolling-resistance coefficients use the reference coefficient.
    rolling_from_surface: bool,
    /// Twin tyres (dual wheels) side by side at this centre distance (m): one wheel with two
    /// identical tyres sharing the load, each evaluated at half of it.
    pub dual: Option<f64>,
}

/// Road friction and rolling resistance relative to the reference surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    pub mu_scale: f64,
    pub rolling_scale: f64,
}

impl Surface {
    pub const REFERENCE: Self = Self { mu_scale: 1.0, rolling_scale: 1.0 };

    pub fn of(material: &Material) -> Self {
        Self {
            mu_scale: material.friction / REFERENCE_FRICTION,
            rolling_scale: material.rolling_resistance / REFERENCE_ROLLING_RESISTANCE,
        }
    }
}

/// Transient state of a tyre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TireState {
    /// Longitudinal and lateral carcass deflections (m).
    pub u: f64,
    pub v: f64,
    /// Relaxation lengths (m) and slip stiffnesses `K_xκ`, `|K_yα|` (N) of the last step.
    pub sigma: [f64; 2],
    pub stiffness: [f64; 2],
    /// Track patches: the shear vectors of the cells, front to rear (m), and the shear flowing
    /// in from the neighbouring patches (the front one's rear cell, the rear one's front cell;
    /// `None` at the track's ends), set by the vehicle before each step.
    pub shear: [[f64; 2]; TRACK_CELLS],
    pub inflow: [Option<[f64; 2]>; 2],
}

/// Motion of a wheel, all in world coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelMotion {
    pub center: DVec3,
    /// Spin axis (unit), pointing to the wheel's left.
    pub axis: DVec3,
    /// Velocity of the wheel centre.
    pub velocity: DVec3,
    /// Angular velocity of the (non-spinning) wheel carrier.
    pub carrier_angvel: DVec3,
    /// Spin rate of the wheel relative to its carrier, about `axis` (rad/s).
    pub spin: f64,
}

/// Result of one tyre step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TireForces {
    /// Force on the wheel and torque about the wheel centre (world).
    pub force: DVec3,
    pub torque: DVec3,
    /// Contact point (world).
    pub point: DVec3,
    /// Forces and moments in the contact frame (ISO-W: x forward, y left, z up).
    pub fx: f64,
    pub fy: f64,
    pub fz: f64,
    pub mx: f64,
    pub my: f64,
    pub mz: f64,
    /// Transient slip the forces were evaluated at, and the tyre deflection (m).
    pub kappa: f64,
    pub tan_alpha: f64,
    pub deflection: f64,
    /// Contact-point velocity along the contact x and y axes, and slip velocity `V_sx` (m/s).
    pub vx: f64,
    pub vy: f64,
    pub vsx: f64,
}

impl Tire {
    /// A Magic Formula tyre; fails if the file gives no positive relaxation lengths (MF 5.2:
    /// `PTX1`, `PTY1`; MF 6.1: `LONGITUDINAL_STIFFNESS`, `LATERAL_STIFFNESS`).
    pub fn magic_formula(params: MfParams, pressure: Option<f64>) -> Result<Self, String> {
        let rolling_from_surface = [params.qsy1, params.qsy2, params.qsy3, params.qsy4].iter().all(|&q| q == 0.0);
        let mut tire = Self {
            model: TireModel::MagicFormula(Box::new(params)),
            pressure,
            low_speed_damping: [0.0; 2],
            rolling_from_surface,
            dual: None,
        };
        let sigma = tire.initial_state().sigma;
        if !sigma.iter().all(|s| s.is_finite() && *s > 0.0) {
            return Err(format!("relaxation lengths {sigma:?} at nominal load are not positive"));
        }
        tire.low_speed_damping = tire.default_low_speed_damping();
        Ok(tire)
    }

    pub fn fiala(params: FialaParams) -> Result<Self, String> {
        params.validate()?;
        let mut tire = Self {
            model: TireModel::Fiala(params),
            pressure: None,
            low_speed_damping: [0.0; 2],
            rolling_from_surface: false,
            dual: None,
        };
        tire.low_speed_damping = tire.default_low_speed_damping();
        Ok(tire)
    }

    /// A track patch (its own low-speed damping; no tyre-style damping).
    pub fn track(params: TrackPatch) -> Result<Self, String> {
        params.validate()?;
        Ok(Self {
            model: TireModel::Track(params),
            pressure: None,
            low_speed_damping: [0.0; 2],
            rolling_from_surface: false,
            dual: None,
        })
    }

    /// Whether this is a track patch rather than a tyre.
    pub fn is_track(&self) -> bool {
        matches!(self.model, TireModel::Track(_))
    }

    /// The tyre as mounted on the vehicle's right (or left) side: Magic Formula tyres measured
    /// on the other side are mirrored.
    pub fn on_side(&self, right: bool) -> Self {
        let mut tire = self.clone();
        if let TireModel::MagicFormula(p) = &mut tire.model
            && p.measured_right != right
        {
            **p = p.mirrored();
        }
        tire
    }

    /// The same tyre as a dual pair at centre distance `spacing` (m).
    pub fn with_dual(mut self, spacing: f64) -> Self {
        self.dual = Some(spacing);
        self
    }

    /// Number of tyres on the wheel (2 for duals).
    pub fn count(&self) -> f64 {
        if self.dual.is_some() { 2.0 } else { 1.0 }
    }

    /// `k_Vlow0` that damps a corner mass (nominal load over g) on the standstill carcass
    /// spring `K/σ` with ratio 0.25: enough to settle a parked vehicle in a few cycles, small
    /// enough for the wheel's spin mode to stay stable with explicit steps of 1 ms.
    fn default_low_speed_damping(&self) -> [f64; 2] {
        if self.is_track() {
            return [0.0; 2];
        }
        let state = self.initial_state();
        let mass = self.single_nominal_load() / GRAVITY;
        [0, 1].map(|i| 2.0 * LOW_SPEED_DAMPING_RATIO * (state.stiffness[i] / state.sigma[i] * mass).sqrt())
    }

    /// Nominal load of the wheel (N; of both tyres for duals).
    pub fn nominal_load(&self) -> f64 {
        self.count() * self.single_nominal_load()
    }

    fn single_nominal_load(&self) -> f64 {
        match &self.model {
            TireModel::MagicFormula(p) => p.fnomin * p.lfzo,
            TireModel::Fiala(p) => p.nominal_load,
            TireModel::Track(p) => p.nominal_load,
        }
    }

    pub fn radius(&self) -> f64 {
        match &self.model {
            TireModel::MagicFormula(p) => p.unloaded_radius,
            TireModel::Fiala(p) => p.radius,
            TireModel::Track(p) => p.radius,
        }
    }

    /// Section width (m); overall width of both tyres for duals.
    pub fn width(&self) -> f64 {
        self.section_width() + self.dual.unwrap_or(0.0)
    }

    /// Section width of one tyre (m).
    pub fn section_width(&self) -> f64 {
        match &self.model {
            TireModel::MagicFormula(p) => p.width,
            TireModel::Fiala(p) => p.width,
            TireModel::Track(p) => p.width,
        }
    }

    fn vxlow(&self) -> f64 {
        match &self.model {
            TireModel::MagicFormula(p) => p.vxlow,
            TireModel::Fiala(p) => p.vxlow,
            TireModel::Track(p) => p.vxlow,
        }
    }

    /// Undeflected carcass with the relaxation lengths and stiffnesses at nominal load.
    pub fn initial_state(&self) -> TireState {
        let (sigma, stiffness) = match &self.model {
            TireModel::MagicFormula(p) => {
                let mut input = MfInput::new(self.single_nominal_load(), 0.0, 0.0, 0.0, 0.0);
                input.pressure = self.pressure;
                let o = p.eval(&input);
                ([o.sigma_x, o.sigma_y], [o.kxk, o.kya.abs()])
            }
            TireModel::Fiala(p) => ([p.relaxation_x, p.relaxation_y], [p.slip_stiffness, p.cornering_stiffness]),
            // Shear stiffness per wheel at nominal load, μF_z/K per metre over half the patch.
            TireModel::Track(p) => ([0.5 * p.length; 2], [0.5 * p.length * p.mu * p.nominal_load / p.shear_modulus; 2]),
        };
        TireState { u: 0.0, v: 0.0, sigma, stiffness, shear: [[0.0; 2]; TRACK_CELLS], inflow: [None; 2] }
    }

    /// The road plane below the wheel, sampled over ±0.3 R ahead and behind (a track patch:
    /// its half length) and the tyre's half width to the sides; `None` when the wheel is more
    /// than 2 R above the ground.
    pub fn contact<T: Terrain + ?Sized>(&self, terrain: &T, motion: &WheelMotion) -> Option<RoadContact> {
        let r = self.radius();
        let half_length = match &self.model {
            TireModel::Track(p) => 0.5 * p.length,
            _ => 0.3 * r,
        };
        road_contact(terrain, motion.center, motion.axis, half_length, 0.5 * self.width(), 2.0 * r)
    }

    /// Vertical force (N; of both tyres for duals) at deflection `rho` (m), without damping.
    pub fn vertical_force(&self, rho: f64) -> f64 {
        self.count() * self.single_vertical_force(rho)
    }

    fn single_vertical_force(&self, rho: f64) -> f64 {
        match &self.model {
            TireModel::MagicFormula(p) => {
                let bottom = p.unloaded_radius - p.rim_radius - p.bottom_offst;
                let bottoming =
                    if p.bottom_stiff > 0.0 && rho > bottom { p.bottom_stiff * (rho - bottom) } else { 0.0 };
                p.vertical_force(rho, self.pressure) + bottoming
            }
            TireModel::Fiala(p) => p.vertical_stiffness * rho.max(0.0),
            TireModel::Track(p) => p.vertical_stiffness * rho.max(0.0),
        }
    }

    /// Advance the transient state by `dt` and return the tyre's wrench on the wheel (both
    /// tyres' for duals: each tyre sees the same slip and half the deflection force).
    pub fn step(
        &self,
        state: &mut TireState,
        contact: Option<&RoadContact>,
        motion: &WheelMotion,
        surface: Surface,
        dt: f64,
    ) -> TireForces {
        let r0 = self.radius();
        let Some(c) = contact.filter(|c| c.loaded_radius < r0) else {
            state.u = 0.0;
            state.v = 0.0;
            state.shear = [[0.0; 2]; TRACK_CELLS];
            return TireForces::default();
        };
        if let TireModel::Track(p) = &self.model {
            return p.step(state, c, motion, surface, dt);
        }
        let rho = r0 - c.loaded_radius;
        let damping = match &self.model {
            TireModel::MagicFormula(p) => p.vertical_damping,
            TireModel::Fiala(p) => p.vertical_damping,
            TireModel::Track(_) => unreachable!("track patches step on their own"),
        };
        let fz = (self.single_vertical_force(rho) - damping * motion.velocity.dot(c.normal)).max(0.0);
        let arm = c.point - motion.center;
        let vc = motion.velocity + motion.carrier_angvel.cross(arm);
        let (vx, vy) = (vc.dot(c.x), vc.dot(c.y));
        let re = match &self.model {
            TireModel::MagicFormula(p) => p.effective_radius(fz, motion.spin, self.pressure),
            _ => r0 - rho / 3.0,
        };
        let vsx = vx - motion.spin * re;
        let avx = vx.abs();

        let [sx, sy] = state.sigma;
        state.u = ((state.u - dt * vsx) / (1.0 + dt * avx / sx)).clamp(-sx, sx);
        state.v = ((state.v + dt * vy) / (1.0 + dt * avx / sy)).clamp(-sy, sy);
        let vxlow = self.vxlow();
        let k_low = if avx < vxlow { 0.5 * (1.0 + (std::f64::consts::PI * avx / vxlow).cos()) } else { 0.0 };
        let kappa = state.u / sx - k_low * self.low_speed_damping[0] * vsx / state.stiffness[0];
        let tan_alpha = state.v / sy + k_low * self.low_speed_damping[1] * vy / state.stiffness[1];

        let rolling = (motion.spin * re / ROLLING_SMOOTHING).tanh() * surface.rolling_scale;
        let (fx, fy, mx, my, mz) = match &self.model {
            TireModel::MagicFormula(p) => {
                let gamma = c.sin_gamma.asin();
                let input = MfInput {
                    fz,
                    kappa,
                    alpha: tan_alpha,
                    cos_alpha: 1.0 / (1.0 + tan_alpha * tan_alpha).sqrt(),
                    gamma,
                    gamma_star: c.sin_gamma,
                    vx: avx,
                    pressure: self.pressure,
                    mu_scale: surface.mu_scale,
                };
                let o = p.eval(&input);
                if fz > 0.0 && o.sigma_x > 0.0 && o.sigma_y > 0.0 {
                    state.sigma = [o.sigma_x, o.sigma_y];
                    state.stiffness = [o.kxk, o.kya.abs()];
                }
                let my = if self.rolling_from_surface { -REFERENCE_ROLLING_RESISTANCE * fz * r0 } else { o.my };
                (o.fx, o.fy, o.mx, my * rolling, o.mz)
            }
            TireModel::Fiala(p) => {
                let o = p.eval(fz, kappa, tan_alpha, p.half_length(rho), surface.mu_scale);
                (o.fx, o.fy, 0.0, -p.rolling_resistance * fz * r0 * rolling, o.mz)
            }
            TireModel::Track(_) => unreachable!("track patches step on their own"),
        };
        let n = self.count();
        let (fx, fy, fz, mx, my, mz) = (n * fx, n * fy, n * fz, n * mx, n * my, n * mz);
        let force = c.x * fx + c.y * fy + c.normal * fz;
        let moment = c.x * mx + c.y * my + c.normal * mz;
        TireForces {
            force,
            torque: moment + arm.cross(force),
            point: c.point,
            fx,
            fy,
            fz,
            mx,
            my,
            mz,
            kappa,
            tan_alpha,
            deflection: rho,
            vx,
            vy,
            vsx,
        }
    }
}
