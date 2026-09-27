//! The rider of a single-track vehicle (bicycle, motorcycle): balance and steering by the
//! steering torque, from a path curvature.
//!
//! ```text
//! curvature ─steady turn + ∫ curvature error→ lean φ_ref ─linear model→ (δ_ref, T_ref)
//! (φ, δ, φ̇, δ̇) − (φ_ref, δ_ref, 0, 0) ─LQR gain K(v)→ steering torque T_ref − K·e
//! ```
//!
//! The gains are a discrete linear–quadratic regulator of the Whipple model at the
//! controller's rate (Meijaard et al. 2007; see
//! [`single_track`](autonomousim_vehicles::ground::single_track)), with the steering damper
//! added, the rider locked upright, and scheduled over the forward speed; above about 25 m/s
//! the rider's authority fades out (see [`LOOSE_SPEED`]), leaving the vehicle's own weave
//! damping, and near its own high-speed instability (for `motorcycle_sport` about 65 m/s)
//! no rider holds it. The lean of a steady
//! turn follows from the lateral acceleration `v²κ` with the wheels' gyroscopic moment and the
//! contact points moving around the tyre crowns: `tan θ = v²κ/g·(1 + Σ I_spin/(r m h))`,
//! `φ = θ + asin(ρ sin θ/(h − ρ))`. An integral on the curvature error trims the lean for what
//! the model misses (tyre slip and relaxation, suspension, the rider's own lean), and the lean
//! is capped at `max_lean`.
//!
//! Below `balance_speed` the rider crawls on their feet: a steering-angle servo to the
//! kinematic angle `atan(κ w)/cos λ`, the feet holding the vehicle up. The rider's upper body
//! stays neutral.

use super::GroundEstimate;
use crate::ControlError;
use autonomousim_vehicles::ground::single_track::{WhippleMatrices, WhippleParams};
use autonomousim_vehicles::ground::{STANDARD_GRAVITY, WheeledDef};
use glam::{DMat4, DVec4};

/// Balance and steering of one single-track vehicle.
#[derive(Clone, Debug)]
pub(super) struct Rider {
    /// Scheduling speeds (m/s, increasing) and the regulator gains there, for the state
    /// `(φ, δ, φ̇, δ̇)` in the paper's frame (lean and steer positive to the right) and the
    /// steering torque positive to the right.
    speeds: Vec<f64>,
    gains: Vec<DVec4>,
    model: WhippleMatrices,
    /// Largest steering torque (N·m), wheelbase (m), head angle (rad).
    max_torque: f64,
    wheelbase: f64,
    head_angle: f64,
    /// Steering-angle servo of the crawl: stiffness (N·m/rad) and damping (N·m·s/rad).
    crawl_kp: f64,
    crawl_kd: f64,
    /// Steady-turn lean: gyroscopic factor `Σ I_spin/(r m h)`, centre-of-mass height `h` and
    /// crown radius `ρ` (m).
    gyro: f64,
    height: f64,
    crown: f64,
    /// Lean correction of the curvature integral (rad).
    lean_i: f64,
}

/// Weights of the regulator: the state errors and steering torque (as a fraction of the
/// largest) that cost the same.
const LEAN_SCALE: f64 = 0.05;
const STEER_SCALE: f64 = 0.05;
const LEAN_RATE_SCALE: f64 = 0.5;
const STEER_RATE_SCALE: f64 = 2.0;
const TORQUE_SCALE: f64 = 0.2;
/// Above about this speed (m/s) the rider holds the bars ever more loosely: the torque's
/// weight grows as `(v/LOOSE_SPEED)⁸`. The knife-edge model misses the tyres' lag, which
/// shifts the phase of the weave; with full gains there, the regulator destabilises a weave
/// the vehicle itself damps.
const LOOSE_SPEED: f64 = 25.0;

impl Rider {
    /// The rider of `def`, or `None` unless it is a single-track vehicle with a steering head.
    pub(super) fn new(def: &WheeledDef, dt: f64) -> Result<Option<Self>, ControlError> {
        let Some((head, _)) = def.steering_head() else { return Ok(None) };
        if !def.is_single_track() {
            return Ok(None);
        }
        let params = WhippleParams::from_def(def).map_err(ControlError::InvalidConfig)?;
        let model = params.matrices();
        let g = STANDARD_GRAVITY;
        let q = DMat4::from_diagonal(DVec4::new(
            LEAN_SCALE.powi(-2),
            STEER_SCALE.powi(-2),
            LEAN_RATE_SCALE.powi(-2),
            STEER_RATE_SCALE.powi(-2),
        ));
        let r = (TORQUE_SCALE * head.max_torque).powi(-2);
        // 0.5 to 120 m/s in steps of 8 %.
        let mut speeds = Vec::new();
        let mut v = 0.5;
        while v < 120.0 {
            speeds.push(v);
            v *= 1.08;
        }
        let gains = speeds
            .iter()
            .map(|&v| {
                let (a, b) = linear_model(&model, head.damping, v, g);
                let (phi, gamma) = discretize(a, b, dt);
                let loose = 1.0 + (v / LOOSE_SPEED).powi(8);
                lqr(phi, gamma, q * dt, r * loose * dt)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ControlError::InvalidConfig(format!("{}: no stabilising rider gains", def.name)))?;
        // Crawl: the steering at 5 Hz, critically damped, on the front assembly's inertia
        // about the axis.
        let inertia = model.m[1][1];
        let omega = 2.0 * std::f64::consts::PI * 5.0;
        let (m, h) = (def.total_mass(), def.total_com().z);
        let loads = def.rest_state().map_or(vec![1.0, 1.0], |s| s.loads.clone());
        let crown = (0..2).map(|a| def.tire(a).crown_radius * loads[a]).sum::<f64>() / (loads[0] + loads[1]);
        let spin: f64 = (0..2).map(|a| def.axles[a].wheel.inertia.y / def.tire(a).radius()).sum();
        Ok(Some(Self {
            speeds,
            gains,
            model,
            max_torque: head.max_torque,
            wheelbase: params.wheelbase,
            head_angle: head.angle,
            crawl_kp: inertia * omega * omega,
            crawl_kd: 2.0 * inertia * omega,
            gyro: spin / (m * h),
            height: h,
            crown,
            lean_i: 0.0,
        }))
    }

    pub(super) fn reset(&mut self) {
        self.lean_i = 0.0;
    }

    /// Regulator gain at speed `v`, interpolated in the schedule.
    fn gain(&self, v: f64) -> DVec4 {
        let s = &self.speeds;
        let k = s.partition_point(|&x| x < v);
        if k == 0 {
            return self.gains[0];
        }
        if k == s.len() {
            return self.gains[k - 1];
        }
        let t = (v - s[k - 1]) / (s[k] - s[k - 1]);
        self.gains[k - 1].lerp(self.gains[k], t)
    }

    /// Lean of a steady turn of curvature `curvature` (1/m, positive left) at speed `v` (rad,
    /// positive right).
    pub(super) fn turn_lean(&self, v: f64, curvature: f64) -> f64 {
        let theta = -(v * v * curvature / STANDARD_GRAVITY * (1.0 + self.gyro)).atan();
        theta + (self.crown * theta.sin() / (self.height - self.crown)).asin()
    }

    /// Steering input (`[−1, 1]`, positive turning left) for path curvature `curvature` at the
    /// estimated state, riding at `balance_speed` or faster, crawling below it.
    pub(super) fn steering(
        &mut self,
        curvature: f64,
        est: &GroundEstimate,
        dt: f64,
        config: &super::GroundConfig,
    ) -> f64 {
        let v = est.speed();
        // Paper frame: steer positive to the right.
        let (delta, delta_rate) = (-est.steer_angle, -est.steer_rate);
        if v.abs() < config.balance_speed {
            self.lean_i = 0.0;
            let target = -(curvature * self.wheelbase).atan() / self.head_angle.cos();
            let torque = -self.crawl_kp * (delta - target) - self.crawl_kd * delta_rate;
            return (-torque / self.max_torque).clamp(-1.0, 1.0);
        }
        let measured = est.yaw_rate / (v * est.roll.cos());
        let max = config.max_lean;
        let base = self.turn_lean(v, curvature);
        // The integral runs while the lean is within its cap, and nudges it by the lean a
        // steady turn needs per curvature.
        if base.abs() < max {
            let per_curvature = v * v / STANDARD_GRAVITY;
            self.lean_i = (self.lean_i - config.curvature_integral * per_curvature * (curvature - measured) * dt)
                .clamp(-config.curvature_correction, config.curvature_correction);
        }
        let phi_ref = (base + self.lean_i).clamp(-max, max);
        // The linear model's steady turn at that lean: no lean torque, the steering torque.
        let g = STANDARD_GRAVITY;
        let m = &self.model;
        let k = |i: usize, j: usize| g * m.k0[i][j] + v * v * m.k2[i][j];
        let delta_ref = -k(0, 0) / k(0, 1) * phi_ref;
        let torque_ref = k(1, 0) * phi_ref + k(1, 1) * delta_ref;
        let e = DVec4::new(est.roll - phi_ref, delta - delta_ref, est.roll_rate, delta_rate);
        let torque = torque_ref - self.gain(v).dot(e);
        (-torque / self.max_torque).clamp(-1.0, 1.0)
    }
}

/// `ẋ = A x + b T` of the Whipple model at speed `v` with a steering damper `damping`.
fn linear_model(m: &WhippleMatrices, damping: f64, v: f64, g: f64) -> (DMat4, DVec4) {
    let det = m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0];
    let inv = [[m.m[1][1] / det, -m.m[0][1] / det], [-m.m[1][0] / det, m.m[0][0] / det]];
    let k = |i: usize, j: usize| g * m.k0[i][j] + v * v * m.k2[i][j];
    let c = |i: usize, j: usize| v * m.c1[i][j] + if i == 1 && j == 1 { damping } else { 0.0 };
    let row = |i: usize, f: &dyn Fn(usize, usize) -> f64| {
        [-(inv[i][0] * f(0, 0) + inv[i][1] * f(1, 0)), -(inv[i][0] * f(0, 1) + inv[i][1] * f(1, 1))]
    };
    let (k0, k1) = (row(0, &k), row(1, &k));
    let (c0, c1) = (row(0, &c), row(1, &c));
    let a = DMat4::from_cols(
        DVec4::new(0.0, 0.0, k0[0], k1[0]),
        DVec4::new(0.0, 0.0, k0[1], k1[1]),
        DVec4::new(1.0, 0.0, c0[0], c1[0]),
        DVec4::new(0.0, 1.0, c0[1], c1[1]),
    );
    (a, DVec4::new(0.0, 0.0, inv[0][1], inv[1][1]))
}

/// Zero-order-hold discretisation over `dt`: `(e^{A dt}, ∫₀^dt e^{A s} ds · b)`, by a Taylor
/// series on a step halved until it is small, then doubled back.
fn discretize(a: DMat4, b: DVec4, dt: f64) -> (DMat4, DVec4) {
    let norm = [a.x_axis, a.y_axis, a.z_axis, a.w_axis].iter().map(|c| c.abs().element_sum()).fold(0.0, f64::max);
    let halvings = (norm * dt / 0.1).log2().ceil().max(0.0) as i32;
    let h = dt / 2f64.powi(halvings);
    let (mut phi, mut gamma, mut term) = (DMat4::IDENTITY, DMat4::IDENTITY * h, DMat4::IDENTITY);
    for k in 1..=12 {
        term = term * a * (h / k as f64);
        phi += term;
        gamma += term * (h / (k + 1) as f64);
    }
    for _ in 0..halvings {
        gamma = phi * gamma + gamma;
        phi = phi * phi;
    }
    (phi, gamma * b)
}

/// Gain `k` of the discrete regulator `u = −k·x` of `x⁺ = Φ x + γ u` for the cost
/// `Σ xᵀQx + r u²`, from the Riccati equation solved by the structure-preserving doubling
/// algorithm (Chu, Fan & Lin 2005); `None` if it does not converge.
fn lqr(phi: DMat4, gamma: DVec4, q: DMat4, r: f64) -> Option<DVec4> {
    let outer = DMat4::from_cols(gamma * gamma.x, gamma * gamma.y, gamma * gamma.z, gamma * gamma.w);
    let (mut a, mut g, mut h) = (phi, outer / r, q);
    for _ in 0..200 {
        let w = (DMat4::IDENTITY + g * h).inverse();
        let next = h + a.transpose() * h * w * a;
        g += a * w * g * a.transpose();
        a = a * w * a;
        let change = (next - h).abs_diff_eq(DMat4::ZERO, 1e-12 * next.x_axis.x.abs().max(1.0));
        h = next;
        if !h.is_finite() {
            return None;
        }
        if change {
            let pg = h * gamma;
            return Some(phi.transpose() * pg / (r + gamma.dot(pg)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regulator_stabilises_the_benchmark_bicycle() {
        // Closed-loop eigenvalues of the benchmark bicycle inside the unit circle at every
        // speed, including below the weave speed and above the capsize speed.
        let m = WhippleParams::benchmark().matrices();
        let dt = 0.01;
        for v in [0.5, 2.0, 4.0, 5.0, 8.0, 20.0] {
            let (a, b) = linear_model(&m, 0.0, v, 9.81);
            let (phi, gamma) = discretize(a, b, dt);
            let k = lqr(phi, gamma, DMat4::IDENTITY, 1.0).unwrap();
            let closed = phi - DMat4::from_cols(gamma * k.x, gamma * k.y, gamma * k.z, gamma * k.w);
            let e = autonomousim_vehicles::ground::single_track::eigenvalues4(&closed);
            let largest = e.iter().map(|z| z.abs()).fold(0.0, f64::max);
            assert!(largest < 1.0, "{v} m/s: spectral radius {largest}");
        }
    }

    #[test]
    fn discretisation_matches_a_known_exponential() {
        // Double integrator: Φ = [[1, dt], [0, 1]], γ = (dt²/2, dt).
        let a = DMat4::from_cols(DVec4::ZERO, DVec4::new(1.0, 0.0, 0.0, 0.0), DVec4::ZERO, DVec4::ZERO);
        let (phi, gamma) = discretize(a * 40.0, DVec4::new(0.0, 1.0, 0.0, 0.0), 0.1);
        assert!((phi.y_axis.x - 4.0).abs() < 1e-12 && (phi.x_axis.x - 1.0).abs() < 1e-12);
        assert!((gamma.x - 0.2).abs() < 1e-12 && (gamma.y - 0.1).abs() < 1e-12);
    }
}
