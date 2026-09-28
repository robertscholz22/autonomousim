//! Propeller on an electric motor or a piston engine.
//!
//! The propeller follows `T = ρn²D⁴·C_T(J)`, `Q = ρn²D⁵·C_Q(J)` with `n` in rev/s and
//! `J = V/(nD)` (`C_Q = C_P/2π`). A polynomial of degree ≤ 2 in `J` is expanded in `n` and `V`,
//! so it stays finite for a stopped propeller; a table gives no load below 10⁻³ rev/s.
//!
//! - **Electric motor** (Beard & McLain, 2nd ed. supplement): `Q = K(i − i₀)` with
//!   `i = (V_in − KΩ)/R` clamped to `[0, i_max]` (the controller neither brakes nor
//!   regenerates) and `V_in = throttle·V_supply`; `K = 60/(2π·Kv)`.
//! - **Piston engine**: torque `T_r(1 + f)·(t_idle + (1 − t_idle)·throttle)·φ(σ) − f·T_r·Ω/Ω_r`
//!   with `T_r = P_max/Ω_r`, friction fraction `f`, Gagg–Ferrar altitude factor
//!   `φ = 1.132σ − 0.132`, and the idle fraction `t_idle` set so the static propeller idles at
//!   `idle_rpm`. Mixture, fuel burn and the magnetos are not modelled.
//!
//! The rotor speed is integrated semi-implicitly: with the motor torque linear in the new speed
//! (`a − bΩ'`) and the propeller torque as `cΩΩ'`,
//! `Ω' = (JΩ/dt + a)/(J/dt + b + cΩ)`.

use super::table::Curve;
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Propeller speed (rev/s) below which a tabulated propeller carries no load.
const MIN_TABLE_REV: f64 = 1e-3;
/// Motor friction ramps in linearly below this speed (rad/s), so it stops without reversing.
const FRICTION_RAMP: f64 = 10.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropulsionDef {
    /// Propeller hub (body frame, m).
    pub position: DVec3,
    /// Thrust direction (body frame; normalised on load).
    #[serde(default = "x_axis")]
    pub axis: DVec3,
    /// `+1` when the propeller's angular velocity points along `axis` (clockwise seen from
    /// behind for a tractor propeller, JSBSim's `sense = 1`), `−1` otherwise.
    #[serde(default = "one")]
    pub sense: f64,
    pub propeller: PropellerDef,
    pub engine: EngineDef,
}

fn x_axis() -> DVec3 {
    DVec3::X
}
fn one() -> f64 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropellerDef {
    /// Diameter (m).
    pub diameter: f64,
    /// Inertia of the propeller and the motor's or engine's rotating parts (kg·m²).
    pub inertia: f64,
    /// Thrust coefficient against advance ratio.
    pub ct: Curve,
    /// Torque coefficient `C_Q`, or power coefficient `C_P = 2π·C_Q` (exactly one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cq: Option<Curve>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp: Option<Curve>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineDef {
    Electric(ElectricMotorDef),
    Piston(PistonEngineDef),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricMotorDef {
    /// Speed constant (rpm/V).
    pub kv: f64,
    /// Winding resistance (Ω).
    pub resistance: f64,
    /// No-load current (A).
    #[serde(default)]
    pub no_load_current: f64,
    /// Current limit of the speed controller (A; none if absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_current: Option<f64>,
    /// Supply voltage without a battery model (V).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voltage: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PistonEngineDef {
    /// Sea-level power at the rated speed and full throttle (W).
    pub max_power: f64,
    pub rated_rpm: f64,
    #[serde(default = "default_idle_rpm")]
    pub idle_rpm: f64,
    /// Friction and pumping torque at the rated speed as a fraction of the rated torque.
    #[serde(default = "default_friction")]
    pub friction: f64,
}

fn default_idle_rpm() -> f64 {
    600.0
}
fn default_friction() -> f64 {
    0.2
}

const RPM: f64 = 2.0 * PI / 60.0;

impl PropulsionDef {
    pub fn validate(&self) -> Result<(), String> {
        let p = &self.propeller;
        if !(self.position.is_finite() && self.axis.length_squared() > 0.5 && (self.sense.abs() - 1.0).abs() < 1e-12) {
            return Err("propeller position must be finite, the axis non-zero and the sense ±1".into());
        }
        if !(p.diameter > 0.0 && p.inertia > 0.0) {
            return Err("propeller diameter and inertia must be positive".into());
        }
        let quadratic = |c: &Curve| match c {
            Curve::Poly { poly } if poly.len() > 3 => Err("propeller polynomials are at most quadratic".to_string()),
            c => c.validate(),
        };
        quadratic(&p.ct)?;
        match (&p.cq, &p.cp) {
            (Some(c), None) | (None, Some(c)) => quadratic(c)?,
            _ => return Err("give exactly one of cq and cp".into()),
        }
        match &self.engine {
            EngineDef::Electric(e) => {
                let ok = e.kv > 0.0
                    && e.resistance > 0.0
                    && e.no_load_current >= 0.0
                    && e.max_current.is_none_or(|i| i > 0.0)
                    && e.voltage.is_none_or(|v| v > 0.0);
                if !ok {
                    return Err("electric motor needs positive kv, resistance, current limit and voltage".into());
                }
            }
            EngineDef::Piston(e) => {
                if !(e.max_power > 0.0 && e.rated_rpm > e.idle_rpm && e.idle_rpm > 0.0 && e.friction >= 0.0) {
                    return Err("piston engine needs positive power and 0 < idle_rpm < rated_rpm".into());
                }
            }
        }
        Ok(())
    }

    /// Torque coefficient curve (`cq`, or `cp/2π`).
    pub fn cq(&self) -> Curve {
        match (&self.propeller.cq, &self.propeller.cp) {
            (Some(c), _) => c.clone(),
            (None, Some(c)) => c.scaled(1.0 / (2.0 * PI)),
            (None, None) => Curve::Poly { poly: vec![0.0] },
        }
    }
}

/// `n²·C(J)` with `J = V/(nD)`, finite at `n = 0` for polynomials.
fn n2_coefficient(c: &Curve, n: f64, v: f64, d: f64) -> f64 {
    match c {
        // Degree ≤ 2 (validated), so every power of n is non-negative.
        Curve::Poly { poly } => {
            poly.iter().enumerate().map(|(k, c)| c * n.powi(2 - k as i32) * (v / d).powi(k as i32)).sum()
        }
        Curve::Table { .. } if n < MIN_TABLE_REV => 0.0,
        Curve::Table { .. } => n * n * c.eval(v / (n * d)),
    }
}

/// Runtime propulsion constants.
#[derive(Clone, Debug)]
pub struct Propulsion {
    pub position: DVec3,
    pub axis: DVec3,
    pub sense: f64,
    pub diameter: f64,
    pub inertia: f64,
    ct: Curve,
    cq: Curve,
    engine: Engine,
}

#[derive(Clone, Debug)]
enum Engine {
    Electric { k: f64, resistance: f64, i0: f64, i_max: f64, voltage: f64 },
    Piston { rated_torque: f64, rated_omega: f64, friction: f64, idle_fraction: f64 },
}

/// Loads and state of the propulsion after a step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PropulsionOutput {
    /// Thrust along the axis (N).
    pub thrust: f64,
    /// Aerodynamic torque on the propeller, opposing its rotation (N·m).
    pub prop_torque: f64,
    /// Motor or engine torque driving it (N·m).
    pub engine_torque: f64,
    /// Rotor speed after the step (rad/s).
    pub omega: f64,
    /// Momentum-theory induced velocity at the disc (m/s).
    pub induced_velocity: f64,
    /// Electrical power drawn (W; electric motors) and current (A).
    pub electric_power: f64,
    pub current: f64,
}

impl Propulsion {
    pub fn new(def: &PropulsionDef) -> Self {
        let p = &def.propeller;
        let engine = match &def.engine {
            EngineDef::Electric(e) => Engine::Electric {
                k: 1.0 / (e.kv * RPM),
                resistance: e.resistance,
                i0: e.no_load_current,
                i_max: e.max_current.unwrap_or(f64::INFINITY),
                voltage: e.voltage.unwrap_or(0.0),
            },
            EngineDef::Piston(e) => {
                let rated_omega = e.rated_rpm * RPM;
                let rated_torque = e.max_power / rated_omega;
                Engine::Piston { rated_torque, rated_omega, friction: e.friction, idle_fraction: 0.0 }
            }
        };
        let mut s = Self {
            position: def.position,
            axis: def.axis.normalize(),
            sense: def.sense,
            diameter: p.diameter,
            inertia: p.inertia,
            ct: p.ct.clone(),
            cq: def.cq(),
            engine,
        };
        if let (EngineDef::Piston(e), Engine::Piston { rated_torque, rated_omega, friction, .. }) =
            (&def.engine, &s.engine)
        {
            let omega = e.idle_rpm * RPM;
            let load = s.prop_loads(omega, 0.0, crate::aero::SEA_LEVEL_DENSITY).1;
            let fraction = (load + friction * rated_torque * omega / rated_omega) / (rated_torque * (1.0 + friction));
            if let Engine::Piston { idle_fraction, .. } = &mut s.engine {
                *idle_fraction = fraction.clamp(0.0, 1.0);
            }
        }
        s
    }

    /// Thrust and propeller torque (N, N·m) at rotor speed `omega` (rad/s), axial air speed
    /// `v` (m/s, positive into the disc from the front) and density `rho`.
    pub fn prop_loads(&self, omega: f64, v: f64, rho: f64) -> (f64, f64) {
        let n = omega.max(0.0) / (2.0 * PI);
        let d = self.diameter;
        let d4 = d.powi(4);
        (rho * d4 * n2_coefficient(&self.ct, n, v, d), rho * d4 * d * n2_coefficient(&self.cq, n, v, d))
    }

    /// Motor or engine torque as `a − b·Ω'` (N·m) in the new speed `Ω'`.
    pub(super) fn engine_torque(
        &self,
        throttle: f64,
        omega: f64,
        density_ratio: f64,
        running: bool,
        supply: f64,
    ) -> (f64, f64) {
        match self.engine {
            Engine::Electric { k, resistance, i0, i_max, voltage } => {
                let v_in = if running { throttle * if supply > 0.0 { supply } else { voltage } } else { 0.0 };
                let i = (v_in - k * omega) / resistance;
                let (mut a, mut b) = if (0.0..=i_max).contains(&i) {
                    (k * v_in / resistance, k * k / resistance)
                } else {
                    (k * i.clamp(0.0, i_max), 0.0)
                };
                if omega < FRICTION_RAMP {
                    b += k * i0 / FRICTION_RAMP;
                } else {
                    a -= k * i0;
                }
                (a, b)
            }
            Engine::Piston { rated_torque, rated_omega, friction, idle_fraction } => {
                let power_factor = (1.132 * density_ratio - 0.132).max(0.0);
                let fraction = idle_fraction + (1.0 - idle_fraction) * throttle;
                let a = if running { rated_torque * (1.0 + friction) * fraction * power_factor } else { 0.0 };
                (a, friction * rated_torque / rated_omega)
            }
        }
    }

    /// Advance the rotor by `dt` from speed `omega`; `supply` is the battery voltage (0: the
    /// motor's own supply voltage).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        omega: f64,
        throttle: f64,
        v_axial: f64,
        rho: f64,
        running: bool,
        supply: f64,
        dt: f64,
    ) -> PropulsionOutput {
        let throttle = throttle.clamp(0.0, 1.0);
        let (thrust, q_prop) = self.prop_loads(omega, v_axial, rho);
        let (a, b) = self.engine_torque(throttle, omega, rho / crate::aero::SEA_LEVEL_DENSITY, running, supply);
        let j = self.inertia / dt;
        let next = if q_prop > 0.0 && omega > 1.0 {
            let c = q_prop / (omega * omega);
            (j * omega + a) / (j + b + c * omega)
        } else {
            (j * omega + a - q_prop) / (j + b)
        }
        .max(0.0);
        let engine_torque = a - b * next;
        let induced_velocity = self.induced_velocity(thrust, v_axial, rho);
        let (electric_power, current) = match self.engine {
            Engine::Electric { k, resistance, i_max, voltage, .. } if running => {
                let v_in = throttle * if supply > 0.0 { supply } else { voltage };
                let i = ((v_in - k * next) / resistance).clamp(0.0, i_max);
                (v_in * i, throttle * i)
            }
            _ => (0.0, 0.0),
        };
        PropulsionOutput {
            thrust,
            prop_torque: q_prop,
            engine_torque,
            omega: next,
            induced_velocity,
            electric_power,
            current,
        }
    }

    /// Momentum-theory induced velocity at the disc for `thrust` at axial speed `v` (m/s).
    pub fn induced_velocity(&self, thrust: f64, v: f64, rho: f64) -> f64 {
        let area = PI * self.diameter * self.diameter / 4.0;
        let v = v.max(0.0);
        if thrust > 0.0 { 0.5 * (-v + (v * v + 2.0 * thrust / (rho * area)).sqrt()) } else { 0.0 }
    }

    /// Steady rotor speed at `throttle` (engine torque = propeller torque), by bisection.
    pub fn steady_omega(&self, throttle: f64, v_axial: f64, rho: f64, supply: f64) -> f64 {
        let net = |w: f64| {
            let (a, b) = self.engine_torque(throttle, w, rho / crate::aero::SEA_LEVEL_DENSITY, true, supply);
            a - b * w - self.prop_loads(w, v_axial, rho).1
        };
        let (mut lo, mut hi) = (0.0, 100.0);
        while net(hi) > 0.0 && hi < 1e5 {
            hi *= 2.0;
        }
        if net(lo) <= 0.0 {
            return 0.0;
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if net(mid) > 0.0 { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    }

    pub fn is_electric(&self) -> bool {
        matches!(self.engine, Engine::Electric { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aerosonde() -> PropulsionDef {
        PropulsionDef {
            position: DVec3::ZERO,
            axis: DVec3::X,
            sense: 1.0,
            propeller: PropellerDef {
                diameter: 0.508,
                inertia: 1e-3,
                ct: Curve::Poly { poly: vec![0.09357, -0.06044, -0.1079] },
                cq: Some(Curve::Poly { poly: vec![0.005230, 0.004970, -0.01664] }),
                cp: None,
            },
            engine: EngineDef::Electric(ElectricMotorDef {
                kv: 145.0,
                resistance: 0.042,
                no_load_current: 1.5,
                max_current: None,
                voltage: Some(44.4),
            }),
        }
    }

    #[test]
    fn polynomial_expansion() {
        let c = Curve::Poly { poly: vec![0.09, -0.06, -0.1] };
        let (n, v, d) = (50.0, 20.0, 0.5);
        let j = v / (n * d);
        assert!((n2_coefficient(&c, n, v, d) - n * n * c.eval(j)).abs() < 1e-9);
        // Stopped: only the J² term survives, −0.1·(V/D)².
        assert!((n2_coefficient(&c, 0.0, v, d) + 0.1 * 1600.0).abs() < 1e-9);
    }

    #[test]
    fn electric_steady_state() {
        let def = aerosonde();
        def.validate().unwrap();
        let p = Propulsion::new(&def);
        let rho = 1.225;
        let w = p.steady_omega(1.0, 25.0, rho, 0.0);
        // Integrating from rest reaches the same speed; torques balance there.
        let mut omega = 0.0;
        let mut out = PropulsionOutput::default();
        for _ in 0..5000 {
            out = p.step(omega, 1.0, 25.0, rho, true, 0.0, 0.002);
            omega = out.omega;
        }
        assert!((omega - w).abs() < 1e-3 * w, "{omega} vs {w}");
        assert!((out.engine_torque - out.prop_torque).abs() < 1e-3 * out.prop_torque);
        assert!(out.thrust > 20.0 && out.induced_velocity > 0.0);
        // Current = (V − Kω)/R and the torque constant equals the back-EMF constant.
        let k = 60.0 / (2.0 * PI * 145.0);
        let i = (44.4 - k * omega) / 0.042;
        assert!((out.engine_torque - k * (i - 1.5)).abs() < 1e-6 * out.engine_torque.abs().max(1.0));
        assert!((out.electric_power - 44.4 * i).abs() < 1e-6 * out.electric_power);
        // Motor off: the propeller windmills near C_Q = 0 and makes little drag.
        let mut omega = w;
        for _ in 0..20000 {
            out = p.step(omega, 0.0, 25.0, rho, true, 0.0, 0.002);
            omega = out.omega;
        }
        let j = 25.0 / (omega / (2.0 * PI) * 0.508);
        assert!(j > 0.65 && j < 0.8, "windmilling J {j}");
        // C_T(0.73) ≈ −0.008: a few newtons of drag against 13.5 kg.
        assert!(out.thrust < 0.0 && out.thrust > -8.0, "{}", out.thrust);
        assert_eq!(out.electric_power, 0.0);
    }

    #[test]
    fn piston_idle_and_power() {
        let def = PropulsionDef {
            position: DVec3::ZERO,
            axis: DVec3::X,
            sense: 1.0,
            propeller: PropellerDef {
                diameter: 1.905,
                inertia: 2.26,
                ct: Curve::Table { x: vec![0.0, 1.0, 2.0], y: vec![0.073, 0.024, -0.069] },
                cq: None,
                cp: Some(Curve::Table { x: vec![0.0, 1.0, 2.0], y: vec![0.066, 0.0282, 0.0525] }),
            },
            engine: EngineDef::Piston(PistonEngineDef {
                max_power: 119_312.0,
                rated_rpm: 2700.0,
                idle_rpm: 600.0,
                friction: 0.2,
            }),
        };
        def.validate().unwrap();
        let p = Propulsion::new(&def);
        let idle = p.steady_omega(0.0, 0.0, 1.225, 0.0) / RPM;
        assert!((idle - 600.0).abs() < 1.0, "{idle}");
        // Full throttle static: rpm between idle and rated, power at most the rating.
        let w = p.steady_omega(1.0, 0.0, 1.225, 0.0);
        let q = p.prop_loads(w, 0.0, 1.225).1;
        assert!(w / RPM > 1500.0 && w / RPM < 2700.0, "{}", w / RPM);
        assert!(q * w < 119_312.0);
        // Less power at altitude (σ = 0.74 at 10 000 ft).
        assert!(p.steady_omega(1.0, 0.0, 0.905, 0.0) < w);
    }
}
