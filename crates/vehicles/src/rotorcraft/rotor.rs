//! Helicopter rotor: blade-element theory with uniform momentum inflow in forward flight
//! (Glauert), rigid flapping blades on a centre-spring hinge with first-order tip-path-plane
//! dynamics, hub moments, ground effect and the rotor speed as a state.
//!
//! **Frame.** All inputs and outputs are in the rotor's shaft frame: z along the shaft in the
//! direction of positive thrust, x a reference direction in the disc (forward for a main rotor),
//! y = z × x. Blade azimuths `ψ` are measured from −x (the blade pointing aft), counter-clockwise
//! about +z; for a clockwise rotor the model mirrors y internally, so its inputs and outputs keep
//! these geometric meanings.
//!
//! **Blade element.** Blades are rigid with linear twist, pitch `θ = θ₇₅ + θ_tw(r̄ − ¾) +
//! θ₁c cos ψ + θ₁s sin ψ` and flapping `β = β₀ + β₁c cos ψ + β₁s sin ψ`. With velocities
//! normalized by the tip speed `ΩR`, a section sees `u_T = r̄ + μ_x sin ψ − μ_y cos ψ` and `u_P =
//! λ + r̄β′ + β(μ_x cos ψ + μ_y sin ψ) + r̄(q̂ cos ψ − p̂ sin ψ)` (positive down through the disc);
//! it produces the small-angle normal force `a(θu_T² − u_P u_T)` and in-plane force `a(θu_T u_P −
//! u_P²) + δu_T²` (per `½ρc(ΩR)²`), lift over the root cut-out to the tip-loss radius and profile
//! drag to the tip. Reverse flow and blade stall are not modelled (valid to μ ≈ 0.5).
//! These integrands are polynomials of degree ≤ 4 in r̄ and trigonometric polynomials of degree ≤
//! 5 in ψ, so three Gauss–Legendre points along the blade and eight azimuths integrate them
//! exactly: the loads are the closed-form blade-element results (Johnson, *Helicopter Theory*,
//! ch. 5; Padfield, *Helicopter Flight Dynamics*, ch. 3), without transcribing their many terms.
//! Profile drag follows Padfield's `δ = δ₀ + δ₂C_T²`.
//!
//! **Inflow.** Uniform, from Glauert's momentum theory through the tip-path plane: `λ_o =
//! C_T / (2√(μ² + (λ_o + λ_c)²))` with `λ_c` the hub's velocity along the tip-path-plane normal,
//! solved by Newton's method (bisection as the fallback) warm-started from the last step. In
//! ground effect the blades see `λ_i = k_G·λ_o` with Cheeseman–Bennett's `k_G = 1 − (R/4z)²/(1 +
//! (μ/λ)²)`. Momentum theory has no valid solution in the vortex-ring state; the model keeps the
//! normal working state there and raises [`RotorLoads::vortex_ring`].
//!
//! **Flapping.** The flap equation of a blade on a centre spring (Padfield's equivalent of hinge
//! offset and hub stiffness), `β″ + ν²β = γM + 2(p̂ cos ψ + q̂ sin ψ)` with `ν² = 1 +
//! K_β/(I_βΩ²)` and Lock number `γ = ρacR⁴/I_β`, is balanced harmonically for the steady
//! coning and tip-path-plane tilt. Coning is quasi-steady; the tilt `(β₁c, β₁s)` follows its
//! steady value with the first-order lag `τ = 16/(γΩ)` of the tip-path-plane models (Chen;
//! Mettler, Tischler & Kanade), which also gives the flapping's lag behind body rates. The spring
//! passes the hub moment `(N_b/2)K_β(−β₁s, β₁c, 0)` to the shaft.
//!
//! **Rotor speed.** `Ω` is a state: `I_Ω·Ω̇ = Q_drive − Q`, with `Q` the aerodynamic torque, so a
//! drive train with a governor (or none, in autorotation) sets it.

use crate::VehicleError;
use glam::{DMat3, DVec3};
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

/// Azimuths of the quadrature (exact for trigonometric polynomials of degree < 8).
const AZIMUTHS: usize = 8;

/// Three-point Gauss–Legendre nodes and weights on [−1, 1] (exact to degree 5).
const GAUSS: [(f64, f64); 3] =
    [(-0.774_596_669_241_483_4, 5.0 / 9.0), (0.0, 8.0 / 9.0), (0.774_596_669_241_483_4, 5.0 / 9.0)];

/// Tip speed below which the rotor produces no loads (m/s).
const MIN_TIP_SPEED: f64 = 1.0;

/// Newton iterations and tolerance of the inflow solution.
const MAX_NEWTON: usize = 30;
const INFLOW_TOL: f64 = 1e-13;

/// Bisection bracket and iterations of the inflow fallback.
const MAX_INFLOW: f64 = 2.0;
const BISECTIONS: usize = 80;

/// Floor inside the momentum-theory root `√(μ² + λ² + ε²)`.
const MOMENTUM_EPS: f64 = 1e-9;

/// Determinant below which the flapping balance is treated as singular.
const MIN_FLAP_DET: f64 = 1e-12;

/// Direction of rotation seen from above (from +z of the shaft frame).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Spin {
    #[default]
    Ccw,
    Cw,
}

fn default_lift_slope() -> f64 {
    5.73
}

fn default_tip_loss() -> f64 {
    0.97
}

fn default_profile_drag() -> f64 {
    0.008
}

fn default_flap_limit() -> f64 {
    0.35
}

/// Rotor geometry, aerodynamics and blade dynamics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotorDef {
    /// Radius R (m).
    pub radius: f64,
    pub blades: u32,
    /// Blade chord c (m), constant along the blade.
    pub chord: f64,
    /// Section lift slope a (1/rad).
    #[serde(default = "default_lift_slope")]
    pub lift_slope: f64,
    /// Linear twist θ_tw, tip minus centre (rad; negative is washout).
    #[serde(default)]
    pub twist: f64,
    /// Root cut-out as a fraction of R: no lift or drag inboard.
    #[serde(default)]
    pub root_cutout: f64,
    /// Tip-loss factor B: no lift outboard of B·R.
    #[serde(default = "default_tip_loss")]
    pub tip_loss: f64,
    /// Profile drag coefficient δ₀ and its growth with thrust δ₂ (`δ = δ₀ + δ₂C_T²`).
    #[serde(default = "default_profile_drag")]
    pub profile_drag: f64,
    #[serde(default)]
    pub profile_drag_thrust: f64,
    /// Flap moment of inertia of one blade about the hinge, I_β (kg·m²).
    pub flap_inertia: f64,
    /// Centre-spring flap stiffness K_β (N·m/rad): the hub stiffness, or the equivalent of a hinge
    /// offset; zero for a centrally hinged or teetering rotor.
    #[serde(default)]
    pub flap_stiffness: f64,
    /// Polar moment of inertia of the rotor about the shaft (kg·m²); `blades·I_β` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polar_inertia: Option<f64>,
    #[serde(default)]
    pub spin: Spin,
    /// Largest tip-path-plane tilt and coning (rad).
    #[serde(default = "default_flap_limit")]
    pub flap_limit: f64,
}

impl RotorDef {
    pub fn validate(&self) -> Result<(), VehicleError> {
        let bad = |m: &str| Err(VehicleError::Invalid(format!("rotor: {m}")));
        let positive = [
            ("radius", self.radius),
            ("chord", self.chord),
            ("lift_slope", self.lift_slope),
            ("flap_inertia", self.flap_inertia),
            ("flap_limit", self.flap_limit),
        ];
        for (name, v) in positive {
            if !(v.is_finite() && v > 0.0) {
                return bad(&format!("`{name}` must be positive"));
            }
        }
        if self.blades == 0 {
            return bad("`blades` must be at least 1");
        }
        if !(self.root_cutout >= 0.0 && self.root_cutout < self.tip_loss && self.tip_loss <= 1.0) {
            return bad("needs 0 ≤ `root_cutout` < `tip_loss` ≤ 1");
        }
        for (name, v) in [
            ("profile_drag", self.profile_drag),
            ("profile_drag_thrust", self.profile_drag_thrust),
            ("flap_stiffness", self.flap_stiffness),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return bad(&format!("`{name}` must be non-negative"));
            }
        }
        if !self.twist.is_finite() {
            return bad("`twist` must be finite");
        }
        if let Some(i) = self.polar_inertia
            && !(i.is_finite() && i > 0.0)
        {
            return bad("`polar_inertia` must be positive");
        }
        Ok(())
    }

    /// Solidity σ = N_b·c/(πR).
    pub fn solidity(&self) -> f64 {
        f64::from(self.blades) * self.chord / (PI * self.radius)
    }

    /// Disc area πR² (m²).
    pub fn disc_area(&self) -> f64 {
        PI * self.radius * self.radius
    }

    /// Lock number γ = ρacR⁴/I_β at the given air density.
    pub fn lock_number(&self, density: f64) -> f64 {
        density * self.lift_slope * self.chord * self.radius.powi(4) / self.flap_inertia
    }

    pub fn polar_inertia(&self) -> f64 {
        self.polar_inertia.unwrap_or(f64::from(self.blades) * self.flap_inertia)
    }

    /// Sign of the rotation about +z: +1 counter-clockwise, −1 clockwise.
    pub fn spin_sign(&self) -> f64 {
        match self.spin {
            Spin::Ccw => 1.0,
            Spin::Cw => -1.0,
        }
    }
}

/// Per-instance rotor state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RotorState {
    /// Rotor speed Ω (rad/s, ≥ 0).
    pub omega: f64,
    /// Tip-path-plane tilt `(β₁c, β₁s)` (rad): β₁c > 0 tilts the disc toward +x, β₁s > 0 toward +y.
    pub flap: [f64; 2],
    /// Momentum-theory induced inflow ratio λ_o (out of ground effect), the warm start of the
    /// next inflow solution.
    pub inflow: f64,
}

impl RotorState {
    /// A rotor spinning at `omega` with an untilted disc.
    pub fn spinning(omega: f64) -> Self {
        Self { omega, ..Self::default() }
    }
}

/// Flow and controls at the rotor, in the shaft frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotorInput {
    /// Velocity of the hub relative to the air (m/s).
    pub velocity: DVec3,
    /// Angular velocity of the shaft relative to the air (rad/s).
    pub rates: DVec3,
    /// Air density (kg/m³).
    pub density: f64,
    /// Collective pitch at ¾ radius θ₇₅ (rad).
    pub collective: f64,
    /// Cyclic pitch `(θ₁c, θ₁s)` (rad), geometric: the blade pitch gains `θ₁c cos ψ + θ₁s sin ψ`
    /// with ψ counter-clockwise from −x, whichever way the rotor turns.
    pub cyclic: [f64; 2],
    /// Distance from the hub to the ground along the downwash (m), if near it.
    pub ground_distance: Option<f64>,
}

impl RotorInput {
    /// Still air at the given density, no controls.
    pub fn still(density: f64) -> Self {
        Self {
            velocity: DVec3::ZERO,
            rates: DVec3::ZERO,
            density,
            collective: 0.0,
            cyclic: [0.0; 2],
            ground_distance: None,
        }
    }
}

/// Rotor loads and flow quantities for one step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RotorLoads {
    /// Aerodynamic force on the hub (shaft frame, N): thrust along z, H- and Y-forces in the disc.
    pub force: DVec3,
    /// Hub moment from the flap springs (shaft frame, N·m). The aerodynamic torque acts on the
    /// rotor, not the shaft; see [`RotorLoads::torque`].
    pub moment: DVec3,
    /// Aerodynamic torque opposing the rotation (N·m); negative when the air drives the rotor.
    pub torque: f64,
    /// Shaft power `Q·Ω` (W).
    pub power: f64,
    /// Thrust, hub force and torque coefficients (normalized by `ρπR²(ΩR)²` and `…R`).
    pub ct: f64,
    pub cq: f64,
    /// Advance ratio in the disc and total inflow ratio through it (λ = λ_i + μ_z).
    pub advance_ratio: f64,
    pub inflow_ratio: f64,
    /// Induced inflow ratio seen by the blades, λ_i = k_G·λ_o.
    pub induced: f64,
    /// Momentum-theory induced inflow λ_o, the next warm start.
    pub momentum_inflow: f64,
    /// Coning β₀ (rad, quasi-steady) and the steady tip-path-plane tilt the flapping lags toward.
    pub coning: f64,
    pub flap_steady: [f64; 2],
    pub lock_number: f64,
    /// Flapping time constant 16/(γΩ) (s).
    pub time_constant: f64,
    /// The flow is in the vortex-ring region, where momentum theory is not valid.
    pub vortex_ring: bool,
}

/// Quadrature node along the blade: radius r̄ and weight (including the azimuth average).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Node {
    r: f64,
    w: f64,
}

fn nodes(a: f64, b: f64) -> [Node; 3] {
    let (m, h) = (0.5 * (a + b), 0.5 * (b - a));
    GAUSS.map(|(x, w)| Node { r: m + h * x, w: h * w / AZIMUTHS as f64 })
}

/// A rotor: its definition and precomputed quadrature. Immutable; the state is [`RotorState`].
#[derive(Clone, Debug, PartialEq)]
pub struct Rotor {
    def: RotorDef,
    solidity: f64,
    lift: [Node; 3],
    drag: [Node; 3],
    /// (cos ψ, sin ψ) of the azimuths.
    trig: [(f64, f64); AZIMUTHS],
    /// ∫ r̄ dr̄ over the lifting span: `∂C_T/∂λ = −(σa/2)·lift_moment`.
    lift_moment: f64,
}

/// Blade section kinematics at one quadrature point, in the canonical (counter-clockwise) frame
/// and normalized by the tip speed.
struct Flow {
    mu: DVec3,
    p: f64,
    q: f64,
    lambda: f64,
    th75: f64,
    twist: f64,
    cyclic: [f64; 2],
}

impl Flow {
    fn theta(&self, r: f64, c: f64, s: f64) -> f64 {
        self.th75 + self.twist * (r - 0.75) + self.cyclic[0] * c + self.cyclic[1] * s
    }

    fn ut(&self, r: f64, c: f64, s: f64) -> f64 {
        r + self.mu.x * s - self.mu.y * c
    }

    /// Radial velocity of the hub outward along the blade, the flapping's share of `u_P` per β.
    fn radial(&self, c: f64, s: f64) -> f64 {
        self.mu.x * c + self.mu.y * s
    }

    /// `u_P` without flapping.
    fn up0(&self, r: f64, c: f64, s: f64) -> f64 {
        self.lambda + r * (self.q * c - self.p * s)
    }

    /// `u_P` with flapping `[β₀, β₁c, β₁s]`.
    fn up(&self, r: f64, c: f64, s: f64, beta: [f64; 3]) -> f64 {
        let b = beta[0] + beta[1] * c + beta[2] * s;
        let db = -beta[1] * s + beta[2] * c;
        self.up0(r, c, s) + r * db + b * self.radial(c, s)
    }
}

impl Rotor {
    pub fn new(def: RotorDef) -> Result<Self, VehicleError> {
        def.validate()?;
        let (r0, b) = (def.root_cutout, def.tip_loss);
        Ok(Self {
            solidity: def.solidity(),
            lift: nodes(r0, b),
            drag: nodes(r0, 1.0),
            trig: std::array::from_fn(|k| {
                let psi = TAU * k as f64 / AZIMUTHS as f64;
                (psi.cos(), psi.sin())
            }),
            lift_moment: 0.5 * (b * b - r0 * r0),
            def,
        })
    }

    pub fn def(&self) -> &RotorDef {
        &self.def
    }

    /// Loads for the current state and flow. Pure: advance the state with [`Rotor::advance`].
    pub fn loads(&self, state: &RotorState, input: &RotorInput) -> RotorLoads {
        let d = &self.def;
        // Mirror y for a clockwise rotor: vectors flip y, pseudovectors x and z.
        let m = d.spin_sign();
        let velocity = DVec3::new(input.velocity.x, m * input.velocity.y, input.velocity.z);
        let rates = DVec3::new(m * input.rates.x, input.rates.y, m * input.rates.z);
        let omega = state.omega + rates.z;
        let tip = omega * d.radius;
        let gamma = d.lock_number(input.density);
        let live = tip > MIN_TIP_SPEED && input.density > 0.0;
        if !live {
            return RotorLoads {
                flap_steady: state.flap,
                momentum_inflow: state.inflow,
                lock_number: gamma,
                time_constant: f64::INFINITY,
                ..RotorLoads::default()
            };
        }
        let mu = velocity / tip;
        let mu_xy = mu.x.hypot(mu.y);
        let flap = [state.flap[0], m * state.flap[1]];
        let mut flow = Flow {
            mu,
            p: rates.x / omega,
            q: rates.y / omega,
            lambda: mu.z,
            th75: input.collective,
            twist: d.twist,
            cyclic: [input.cyclic[0], m * input.cyclic[1]],
        };
        let half_sa = 0.5 * self.solidity * d.lift_slope;

        // Thrust with no induced flow; it is affine in λ and independent of the flapping.
        let mut ct0 = 0.0;
        for &(c, s) in &self.trig {
            for n in &self.lift {
                let ut = flow.ut(n.r, c, s);
                ct0 += n.w * (flow.theta(n.r, c, s) * ut * ut - flow.up0(n.r, c, s) * ut);
            }
        }
        ct0 *= half_sa;
        let slope = half_sa * self.lift_moment;

        // Momentum inflow through the tip-path plane, and the ground-effect factor.
        let lambda_c = mu.z + mu.x * flap[0] + mu.y * flap[1];
        let k_g = input.ground_distance.map_or(1.0, |z| {
            let x = d.radius / (4.0 * z.max(0.5 * d.radius));
            let ratio = mu_xy / state.inflow.abs().max(1e-3);
            1.0 - x * x / (1.0 + ratio * ratio)
        });
        let lambda_o = solve_inflow(ct0, slope * k_g, lambda_c, mu_xy, state.inflow);
        let induced = k_g * lambda_o;
        let ct = ct0 - slope * induced;
        flow.lambda = mu.z + induced;

        // Harmonic balance of the flap equation: M(β) = M⁰ + Jβ in [β₀, β₁c, β₁s].
        let mut m0 = DVec3::ZERO;
        let mut jac = [DVec3::ZERO; 3];
        for &(c, s) in &self.trig {
            let h = DVec3::new(1.0, 2.0 * c, 2.0 * s);
            for n in &self.lift {
                let ut = flow.ut(n.r, c, s);
                let base = 0.5 * n.r * (flow.theta(n.r, c, s) * ut * ut - flow.up0(n.r, c, s) * ut);
                m0 += h * (n.w * base);
                let w = flow.radial(c, s);
                let g = [w, -n.r * s + c * w, n.r * c + s * w];
                for (col, gk) in jac.iter_mut().zip(g) {
                    *col += h * (n.w * -0.5 * n.r * ut * gk);
                }
            }
        }
        let nu2 = 1.0 + d.flap_stiffness / (d.flap_inertia * omega * omega);
        let a = DMat3::from_diagonal(DVec3::new(nu2, nu2 - 1.0, nu2 - 1.0))
            - DMat3::from_cols(jac[0], jac[1], jac[2]) * gamma;
        let b = m0 * gamma + DVec3::new(0.0, 2.0 * flow.p, 2.0 * flow.q);
        let lim = d.flap_limit;
        let (coning, steady) = if a.determinant().abs() > MIN_FLAP_DET {
            let x = a.inverse() * b;
            (x.x.clamp(-lim, lim), [x.y.clamp(-lim, lim), x.z.clamp(-lim, lim)])
        } else {
            (0.0, flap)
        };

        // Loads with the current (lagging) tip-path plane.
        let beta = [coning, flap[0], flap[1]];
        let delta = d.profile_drag + d.profile_drag_thrust * ct * ct;
        let mut force = DVec3::ZERO;
        let mut cq = 0.0;
        for &(c, s) in &self.trig {
            let e_r = DVec3::new(-c, -s, 0.0);
            let t = DVec3::new(s, -c, 0.0);
            let bpsi = beta[0] + beta[1] * c + beta[2] * s;
            let normal = DVec3::Z - e_r * bpsi;
            for n in &self.lift {
                let (ut, up) = (flow.ut(n.r, c, s), flow.up(n.r, c, s, beta));
                let th = flow.theta(n.r, c, s);
                let f_n = d.lift_slope * (th * ut * ut - up * ut);
                let f_t = d.lift_slope * (th * ut * up - up * up);
                force += (normal * f_n - t * f_t) * n.w;
                cq += n.w * n.r * f_t;
            }
            for n in &self.drag {
                let ut = flow.ut(n.r, c, s);
                let f_t = delta * ut * ut;
                force -= t * (f_t * n.w);
                cq += n.w * n.r * f_t;
            }
        }
        let half_s = 0.5 * self.solidity;
        let coeff = force * half_s;
        let cq = cq * half_s;
        let scale = input.density * d.disc_area() * tip * tip;
        let spring = 0.5 * f64::from(d.blades) * d.flap_stiffness;
        let moment = DVec3::new(-spring * flap[1], spring * flap[0], 0.0);
        let torque = cq * scale * d.radius;

        let lambda_h = (0.5 * ct.abs()).sqrt();
        let descent = -lambda_c / lambda_h.max(1e-9);
        RotorLoads {
            force: DVec3::new(coeff.x, m * coeff.y, coeff.z) * scale,
            moment: DVec3::new(m * moment.x, moment.y, m * moment.z),
            torque,
            power: torque * state.omega,
            ct: coeff.z,
            cq,
            advance_ratio: mu_xy,
            inflow_ratio: flow.lambda,
            induced,
            momentum_inflow: lambda_o,
            coning,
            flap_steady: [steady[0], m * steady[1]],
            lock_number: gamma,
            time_constant: 16.0 / (gamma * omega),
            vortex_ring: ct > 0.0 && mu_xy < lambda_h && (0.28..2.0).contains(&descent),
        }
    }

    /// Advance the state by `dt`: the tip-path plane lags toward its steady tilt (exactly, for
    /// a first-order lag held over the step), the rotor speed follows `drive_torque − Q`, and the
    /// inflow warm start is kept.
    pub fn advance(&self, state: &mut RotorState, loads: &RotorLoads, drive_torque: f64, dt: f64) {
        let k = if loads.time_constant.is_finite() { 1.0 - (-dt / loads.time_constant).exp() } else { 0.0 };
        for (b, target) in state.flap.iter_mut().zip(loads.flap_steady) {
            *b += (target - *b) * k;
        }
        state.inflow = loads.momentum_inflow;
        let omega = state.omega + dt * (drive_torque - loads.torque) / self.def.polar_inertia();
        state.omega = omega.max(0.0);
    }

    /// Angular momentum of the spinning rotor (shaft frame, kg·m²/s).
    pub fn angular_momentum(&self, state: &RotorState) -> DVec3 {
        DVec3::Z * (self.def.spin_sign() * self.def.polar_inertia() * state.omega)
    }
}

/// Solve `f(λ) = λ − (c₀ − kλ)/(2√(μ² + (λ + λ_c)²)) = 0` for the momentum inflow λ, by Newton's
/// method from `guess`, falling back to bisection on the working-state bracket (λ of the sign
/// of the thrust).
fn solve_inflow(c0: f64, k: f64, lambda_c: f64, mu: f64, guess: f64) -> f64 {
    let f = |x: f64| {
        let root = (mu * mu + (x + lambda_c).powi(2) + MOMENTUM_EPS * MOMENTUM_EPS).sqrt();
        let ct = c0 - k * x;
        (x - ct / (2.0 * root), 1.0 + k / (2.0 * root) + ct * (x + lambda_c) / (2.0 * root.powi(3)))
    };
    let mut x = guess;
    for _ in 0..MAX_NEWTON {
        let (fx, dfx) = f(x);
        if dfx.is_nan() || dfx.abs() <= 1e-12 {
            break;
        }
        let step = fx / dfx;
        x -= step;
        if !x.is_finite() || x.abs() > MAX_INFLOW {
            break;
        }
        if step.abs() < INFLOW_TOL {
            // The root must have the sign of the thrust it makes (a working state).
            if x * (c0 - k * x) >= 0.0 {
                return x;
            }
            break;
        }
    }
    let (mut lo, mut hi) = if c0 >= 0.0 { (0.0, MAX_INFLOW) } else { (-MAX_INFLOW, 0.0) };
    for _ in 0..BISECTIONS {
        let mid = 0.5 * (lo + hi);
        if f(mid).0 > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}
