//! Outer loops: total energy control (TECS) for throttle and pitch, and L1 lateral guidance.
//!
//! **TECS** (Lambregts; PX4 `TECS`), in specific energy rates normalised by `g·V`:
//!
//! ```text
//! total  Ė_T = V̇/g + ḣ/V      (the throttle's job)
//! balance Ė_B = ḣ/V − V̇/g     (the pitch's job)
//! ```
//!
//! The throttle is the level trim plus the thrust over weight the total-rate demand needs (the
//! demand, a proportional term and an integrator); the pitch is the level trim α plus the
//! balance-rate demand with its own proportional and integral terms. With a pitch setpoint
//! (the `attitude` mode) only the throttle part runs, on the climb actually flown.
//!
//! **L1** (Park, Deyst & How 2004; PX4 `ECL_L1_Pos_Controller`): the reference point lies on the
//! path at distance `L1 = ζ·T·V_g/π` from the aircraft; the lateral acceleration
//! `2·V_g²·sin η / L1` turns the ground velocity towards it. On a circle of radius `R` the
//! aircraft settles on the circle itself.

use autonomousim_core::math::quat::wrap_angle;
use glam::DVec2;

/// A horizontal path for L1 guidance (ENU, m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Path {
    /// The line through `start` and `end`, flown from `start` towards `end` (and beyond).
    Line { start: [f64; 2], end: [f64; 2] },
    /// A circle about `center`, counter-clockwise unless `clockwise`.
    Circle { center: [f64; 2], radius: f64, clockwise: bool },
}

impl Path {
    /// Signed distance from the path (m, positive to the left of the direction of travel).
    pub fn cross_track(&self, p: DVec2) -> f64 {
        match *self {
            Path::Line { start, end } => {
                let (a, b) = (DVec2::from(start), DVec2::from(end));
                let d = (b - a).normalize_or(DVec2::X);
                d.perp_dot(p - a)
            }
            Path::Circle { center, radius, clockwise } => {
                let r = (p - DVec2::from(center)).length() - radius;
                if clockwise { r } else { -r }
            }
        }
    }

    /// The L1 reference point for an aircraft at `p` with look-ahead `l1`.
    fn reference(&self, p: DVec2, l1: f64) -> DVec2 {
        match *self {
            Path::Line { start, end } => {
                let a = DVec2::from(start);
                let d = (DVec2::from(end) - a).normalize_or(DVec2::X);
                let along = d.dot(p - a);
                let cross = d.perp_dot(p - a);
                a + d * (along + (l1 * l1 - cross * cross).max(0.0).sqrt())
            }
            Path::Circle { center, radius, clockwise } => {
                let c = DVec2::from(center);
                let r = p - c;
                let dist = r.length();
                let u = if dist > 1e-9 { r / dist } else { DVec2::X };
                let sense = if clockwise { -1.0 } else { 1.0 };
                // Intersections of the path with the circle of radius L1 about the aircraft;
                // the one ahead in the direction of travel.
                let a = (radius * radius - l1 * l1 + dist * dist) / (2.0 * dist.max(1e-9));
                let h2 = radius * radius - a * a;
                if h2 >= 0.0 && dist > 1e-9 {
                    c + u * a + u.perp() * (sense * h2.sqrt())
                } else {
                    // Too far out or in: head for the nearest point, a little ahead.
                    let ahead = (l1 / radius).min(std::f64::consts::FRAC_PI_2) * sense;
                    c + DVec2::from_angle(ahead).rotate(u) * radius
                }
            }
        }
    }
}

/// L1 lateral acceleration (m/s², positive to the left) for ground velocity `v` at `p`.
pub fn l1_acceleration(path: &Path, p: DVec2, v: DVec2, period: f64, damping: f64) -> f64 {
    let speed = v.length().max(1.0);
    let l1 = damping * period * speed / std::f64::consts::PI;
    let to_ref = path.reference(p, l1) - p;
    let dir = if v.length() > 0.5 { v } else { to_ref };
    let eta = wrap_angle(dir.angle_to(to_ref)).clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
    2.0 * speed * speed / l1 * eta.sin()
}

/// TECS state: integrators and the filtered airspeed rate.
#[derive(Clone, Debug, Default)]
pub struct Tecs {
    pub throttle_integral: f64,
    pub pitch_integral: f64,
    speed_rate: f64,
    prev_speed: Option<f64>,
}

/// One TECS evaluation.
#[derive(Clone, Copy, Debug)]
pub struct TecsInput {
    pub dt: f64,
    pub airspeed: f64,
    /// Climb rate (m/s).
    pub climb: f64,
    pub airspeed_sp: f64,
    /// Climb-rate demand; `None`: pitch is set elsewhere, the throttle follows the climb flown.
    pub climb_sp: Option<f64>,
    pub gravity: f64,
    pub speed_gain: f64,
    pub accel_limit: f64,
    pub cutoff: f64,
    pub energy_p: f64,
    pub energy_i: f64,
    pub balance_p: f64,
    pub balance_i: f64,
    /// Thrust-to-weight change per unit throttle.
    pub tw_per_throttle: f64,
    pub trim_throttle: f64,
    pub trim_alpha: f64,
    pub pitch_range: (f64, f64),
    /// Bank angle (rad), for the load-factor feedforward.
    pub roll: f64,
}

impl Tecs {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Filtered airspeed rate (m/s²).
    pub fn speed_rate(&self) -> f64 {
        self.speed_rate
    }

    /// Throttle and, with a climb demand, the pitch setpoint.
    pub fn update(&mut self, x: &TecsInput) -> (f64, Option<f64>) {
        // No air data yet (before the aircraft's first step): no rate either.
        if let Some(prev) = self.prev_speed.filter(|_| x.airspeed > 0.0) {
            let raw = (x.airspeed - prev) / x.dt;
            self.speed_rate += (raw - self.speed_rate) * (1.0 - (-x.cutoff * x.dt).exp());
        }
        self.prev_speed = (x.airspeed > 0.0).then_some(x.airspeed);
        let v = x.airspeed.max(1.0);
        let g = x.gravity;
        // No more acceleration than 80 % of the thrust between the trim throttle and its limits
        // gives: the pitch loop would take the rest from the height, nosing over.
        let up = (0.8 * g * x.tw_per_throttle * (1.0 - x.trim_throttle)).clamp(0.1, x.accel_limit);
        let down = (0.8 * g * x.tw_per_throttle * x.trim_throttle).clamp(0.1, x.accel_limit);
        let accel_sp = (x.speed_gain * (x.airspeed_sp - x.airspeed)).clamp(-down, up);
        let gamma = (x.climb / v).clamp(-1.0, 1.0);
        let gamma_sp = x.climb_sp.map_or(gamma, |c| (c / v).clamp(-1.0, 1.0));
        let total_sp = accel_sp / g + gamma_sp;
        let total = self.speed_rate / g + gamma;
        let total_err = total_sp - total;
        let k = 1.0 / x.tw_per_throttle.max(1e-6);
        let unclamped = x.trim_throttle + k * (total_sp + x.energy_p * total_err) + self.throttle_integral;
        let throttle = unclamped.clamp(0.0, 1.0);
        // The integrator stops where the throttle saturates in the same direction.
        let di = k * x.energy_i * total_err * x.dt;
        if !((unclamped >= 1.0 && di > 0.0) || (unclamped <= 0.0 && di < 0.0)) {
            self.throttle_integral = (self.throttle_integral + di).clamp(-1.0, 1.0);
        }
        let pitch = x.climb_sp.map(|_| {
            let balance_sp = gamma_sp - accel_sp / g;
            let balance = gamma - self.speed_rate / g;
            let err = balance_sp - balance;
            // Level-flight α grows with the load factor 1/cos φ in a turn.
            let alpha = x.trim_alpha / x.roll.cos().max(0.5);
            let unclamped = alpha + balance_sp + x.balance_p * err + self.pitch_integral;
            let (lo, hi) = x.pitch_range;
            let di = x.balance_i * err * x.dt;
            if !((unclamped >= hi && di > 0.0) || (unclamped <= lo && di < 0.0)) {
                self.pitch_integral = (self.pitch_integral + di).clamp(-0.3, 0.3);
            }
            unclamped.clamp(lo, hi)
        });
        (throttle, pitch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l1_on_the_path() {
        // On a straight line flying along it: no acceleration; offset to the right: turn left.
        let line = Path::Line { start: [0.0, 0.0], end: [100.0, 0.0] };
        let v = DVec2::new(20.0, 0.0);
        assert!(l1_acceleration(&line, DVec2::new(50.0, 0.0), v, 20.0, 0.75).abs() < 1e-9);
        assert!(l1_acceleration(&line, DVec2::new(50.0, -10.0), v, 20.0, 0.75) > 0.0);
        assert!((line.cross_track(DVec2::new(5.0, 3.0)) - 3.0).abs() < 1e-12);
        // On a counter-clockwise circle flying tangentially: the centripetal V²/R.
        let circle = Path::Circle { center: [0.0, 0.0], radius: 200.0, clockwise: false };
        let a = l1_acceleration(&circle, DVec2::new(200.0, 0.0), DVec2::new(0.0, 20.0), 20.0, 0.75);
        assert!((a - 400.0 / 200.0).abs() < 1e-9, "{a}");
        let cw = Path::Circle { center: [0.0, 0.0], radius: 200.0, clockwise: true };
        let a = l1_acceleration(&cw, DVec2::new(200.0, 0.0), DVec2::new(0.0, -20.0), 20.0, 0.75);
        assert!((a + 2.0).abs() < 1e-9, "{a}");
        assert!(circle.cross_track(DVec2::new(190.0, 0.0)) > 0.0 && cw.cross_track(DVec2::new(190.0, 0.0)) < 0.0);
    }
}
