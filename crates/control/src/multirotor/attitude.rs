//! Tilt-prioritised quaternion attitude control (Brescianini & D'Andrea 2018, as in PX4
//! `AttitudeControl`).
//!
//! The thrust axis is corrected first; only a fraction `yaw_weight` of the remaining yaw error
//! enters the reference, and the yaw gain is divided by the same weight so that small yaw
//! errors still converge at the nominal rate.

use glam::{DQuat, DVec3};

#[derive(Clone, Debug)]
pub struct AttitudeController {
    gain: DVec3,
    yaw_weight: f64,
    rate_limit: DVec3,
}

impl AttitudeController {
    /// `gain`: rate per unit attitude error (1/s) per body axis; `rate_limit`: rate setpoint
    /// bounds (rad/s).
    pub fn new(gain: DVec3, yaw_weight: f64, rate_limit: DVec3) -> Self {
        let yaw_weight = yaw_weight.clamp(0.0, 1.0);
        let mut gain = gain;
        if yaw_weight > 1e-4 {
            gain.z /= yaw_weight;
        }
        Self { gain, yaw_weight, rate_limit }
    }

    /// Effective gains (the yaw entry already divided by the yaw weight).
    pub fn gain(&self) -> DVec3 {
        self.gain
    }

    /// Body-rate setpoint (rad/s) that turns attitude `q` towards `q_sp` (both body → world),
    /// plus `yaw_rate` (rad/s) about world z as feedforward.
    pub fn update(&self, q: DQuat, q_sp: DQuat, yaw_rate: f64) -> DVec3 {
        let e_z = q * DVec3::Z;
        let e_z_d = q_sp * DVec3::Z;
        // Reduced setpoint: current attitude with its thrust axis rotated onto the desired one.
        let q_red = if e_z.dot(e_z_d) < -1.0 + 1e-10 {
            // Thrust axis exactly reversed: any axis works, the full attitude decides.
            q_sp
        } else {
            DQuat::from_rotation_arc(e_z, e_z_d) * q
        };
        // Remaining rotation about the (shared) thrust axis, scaled by the yaw weight.
        let mut q_mix = q_red.inverse() * q_sp;
        if q_mix.w < 0.0 {
            q_mix = -q_mix;
        }
        let w = self.yaw_weight;
        let yaw_part = DQuat::from_xyzw(
            0.0,
            0.0,
            (w * q_mix.z.clamp(-1.0, 1.0).asin()).sin(),
            (w * q_mix.w.clamp(-1.0, 1.0).acos()).cos(),
        );
        let q_d = q_red * yaw_part;

        let mut q_e = q.inverse() * q_d;
        if q_e.w < 0.0 {
            q_e = -q_e;
        }
        let e = 2.0 * DVec3::new(q_e.x, q_e.y, q_e.z);
        let rate_sp = e * self.gain + q.inverse() * DVec3::Z * yaw_rate;
        rate_sp.clamp(-self.rate_limit, self.rate_limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::math::quat::exp_map;

    #[test]
    fn small_errors_are_proportional_and_reduced_tilt_comes_first() {
        let c = AttitudeController::new(DVec3::new(10.0, 10.0, 4.0), 0.4, DVec3::splat(100.0));
        let q = DQuat::from_rotation_z(0.3);
        // Small roll error: rate about body x.
        let r = c.update(q, q * exp_map(DVec3::new(0.01, 0.0, 0.0)), 0.0);
        assert!((r - DVec3::new(0.1, 0.0, 0.0)).length() < 1e-5, "{r}");
        // Small yaw error: the weight is compensated by the gain.
        let r = c.update(q, q * exp_map(DVec3::new(0.0, 0.0, 0.01)), 0.0);
        assert!((r - DVec3::new(0.0, 0.0, 0.04)).length() < 1e-5, "{r}");
        // Large yaw and tilt error: the full tilt error, but only 0.4 of the 2.5 rad yaw error.
        let q_sp = DQuat::from_rotation_z(2.5) * exp_map(DVec3::new(0.3, 0.0, 0.0));
        let r = c.update(DQuat::IDENTITY, q_sp, 0.0);
        assert!((r.x.hypot(r.y) - 20.0 * 0.15f64.sin()).abs() < 1e-9, "{r}");
        // Yaw: 2·sin(0.4·2.5/2) about the tilted axis, times the compensated gain 4/0.4.
        assert!((r.z - 20.0 * 0.15f64.cos() * 0.5f64.sin()).abs() < 1e-9, "{r}");
        // Feedforward: level vehicle, world yaw rate maps to body z.
        let r = c.update(q, q, 0.7);
        assert!((r - DVec3::new(0.0, 0.0, 0.7)).length() < 1e-12);
        // Upside down: finite command, limited.
        let r = c.update(DQuat::from_rotation_x(std::f64::consts::PI), DQuat::IDENTITY, 0.0);
        assert!(r.is_finite() && r.length() > 1.0);
    }
}
