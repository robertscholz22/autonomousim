//! Body-rate PID with derivative on the measurement and gyroscopic feedforward.
//!
//! The loop works in angular acceleration: `α = K_p·e + ∫K_i·e − K_d·ω̇`, then
//! `τ = J·α + ω × Jω`. Gains in acceleration units carry over between vehicles of any size.

use glam::DVec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateGains {
    /// Proportional gain (1/s).
    pub p: DVec3,
    /// Integral gain (1/s²).
    pub i: DVec3,
    /// Derivative gain on the measured angular acceleration (dimensionless).
    pub d: DVec3,
    /// Integrator limit (rad/s²).
    pub i_limit: DVec3,
    /// Derivative low-pass cutoff (rad/s); 0 disables the filter.
    pub d_cutoff: f64,
}

/// Rate error (rad/s) at which the integrator stops accumulating (PX4 `i_factor`, 400°/s).
const I_FADE_RATE: f64 = 400.0 * std::f64::consts::PI / 180.0;

#[derive(Clone, Debug)]
pub struct RateController {
    gains: RateGains,
    inertia: DVec3,
    dt: f64,
    d_alpha: f64,
    integral: DVec3,
    prev_rate: Option<DVec3>,
    rate_dot: DVec3,
}

impl RateController {
    /// Controller for principal inertia `inertia` (kg·m²), called every `dt` seconds.
    pub fn new(gains: RateGains, inertia: DVec3, dt: f64) -> Self {
        let d_alpha = if gains.d_cutoff > 0.0 { 1.0 - (-gains.d_cutoff * dt).exp() } else { 1.0 };
        Self { gains, inertia, dt, d_alpha, integral: DVec3::ZERO, prev_rate: None, rate_dot: DVec3::ZERO }
    }

    pub fn gains(&self) -> &RateGains {
        &self.gains
    }

    pub fn reset(&mut self) {
        self.integral = DVec3::ZERO;
        self.prev_rate = None;
        self.rate_dot = DVec3::ZERO;
    }

    /// Integrator state (rad/s²).
    pub fn integral(&self) -> DVec3 {
        self.integral
    }

    /// Body torque (N·m) for rate setpoint `rate_sp` and measured body rate `rate` (rad/s).
    /// `sat_pos`/`sat_neg` flag axes where the allocation could not deliver more positive or
    /// negative torque last time; the integrator does not grow further into them.
    pub fn update(&mut self, rate_sp: DVec3, rate: DVec3, sat_pos: [bool; 3], sat_neg: [bool; 3]) -> DVec3 {
        let g = &self.gains;
        if let Some(prev) = self.prev_rate {
            self.rate_dot += ((rate - prev) / self.dt - self.rate_dot) * self.d_alpha;
        }
        self.prev_rate = Some(rate);
        let err = rate_sp - rate;
        let alpha = g.p * err + self.integral - g.d * self.rate_dot;
        let torque = self.inertia * alpha + rate.cross(self.inertia * rate);

        let mut e = err.to_array();
        for k in 0..3 {
            if sat_pos[k] {
                e[k] = e[k].min(0.0);
            }
            if sat_neg[k] {
                e[k] = e[k].max(0.0);
            }
            e[k] *= (1.0 - (e[k] / I_FADE_RATE).powi(2)).max(0.0);
        }
        let next = self.integral + g.i * DVec3::from_array(e) * self.dt;
        if next.is_finite() {
            self.integral = next.clamp(-g.i_limit, g.i_limit);
        }
        torque
    }
}
