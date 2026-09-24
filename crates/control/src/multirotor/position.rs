//! Velocity and position control (after PX4 `PositionControl`, in ENU and physical units).
//!
//! Position P → velocity P → acceleration → thrust vector. The tilt is limited and the vertical
//! thrust has priority over the horizontal thrust (minus a horizontal margin), as in PX4.
//!
//! **Setpoint shaping** ([`PositionController::shape`]): the cascade does not feed velocity
//! setpoints to the loop directly but through a critically damped second-order reference with
//! limited acceleration, and feeds the reference acceleration forward (in the spirit of PX4's
//! velocity smoothing). The acceleration, and with it the tilt setpoint, stays continuous, so
//! commands that jump, such as a policy's exploration noise, no longer ask for bang-bang
//! attitude changes that saturate the rotors and cost lift. Being linear below the limits,
//! the reference keeps the mean of a noisy command.
//!
//! Instead of PX4's velocity integrator, a disturbance observer supplies the zero steady-state
//! error: it low-pass filters the difference between the measured acceleration and the one the
//! thrust produced (from the actual attitude and a motor-lag thrust model), and the loop
//! subtracts it. Unlike an integrator of the velocity error it does not respond to setpoint
//! changes, so steps settle without the slow tail of an integrator pole–zero pair, and it
//! cannot wind up while the thrust saturates.

use glam::{DMat3, DQuat, DVec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionGains {
    /// Position error → velocity (1/s).
    pub pos_p: DVec3,
    /// Velocity error → acceleration (1/s).
    pub vel_p: DVec3,
    /// Disturbance-observer bandwidth (rad/s).
    pub disturbance: f64,
    /// Natural frequency of the velocity reference (rad/s); 0: setpoints pass unshaped.
    pub reference: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionLimits {
    /// Horizontal speed (m/s).
    pub vel_xy: f64,
    /// Climb and descent speed (m/s).
    pub vel_up: f64,
    pub vel_down: f64,
    /// Largest angle between thrust and world z (rad).
    pub tilt: f64,
    /// Collective thrust range (N).
    pub thrust_min: f64,
    pub thrust_max: f64,
    /// Horizontal thrust kept available when the vertical demand saturates (N).
    pub xy_margin: f64,
    /// Acceleration limits of the velocity reference (m/s²).
    pub accel_xy: f64,
    pub accel_up: f64,
    pub accel_down: f64,
}

#[derive(Clone, Debug)]
pub struct PositionController {
    gains: PositionGains,
    limits: PositionLimits,
    mass: f64,
    gravity: f64,
    disturbance: DVec3,
    prev_vel: Option<DVec3>,
    applied_sum: DVec3,
    applied_steps: u32,
    /// Shaped velocity reference and its acceleration (world).
    vel_ref: Option<DVec3>,
    acc_ref: DVec3,
}

impl PositionController {
    /// Controller for a vehicle of nominal `mass` (kg) under gravity `gravity` (m/s²).
    pub fn new(gains: PositionGains, limits: PositionLimits, mass: f64, gravity: f64) -> Self {
        Self {
            gains,
            limits,
            mass,
            gravity,
            disturbance: DVec3::ZERO,
            prev_vel: None,
            applied_sum: DVec3::ZERO,
            applied_steps: 0,
            vel_ref: None,
            acc_ref: DVec3::ZERO,
        }
    }

    pub fn gains(&self) -> &PositionGains {
        &self.gains
    }

    pub fn limits(&self) -> &PositionLimits {
        &self.limits
    }

    pub fn set_thrust_limits(&mut self, thrust_min: f64, thrust_max: f64, xy_margin: f64) {
        self.limits.thrust_min = thrust_min;
        self.limits.thrust_max = thrust_max;
        self.limits.xy_margin = xy_margin;
    }

    pub fn reset(&mut self) {
        self.disturbance = DVec3::ZERO;
        self.prev_vel = None;
        self.applied_sum = DVec3::ZERO;
        self.applied_steps = 0;
        self.vel_ref = None;
        self.acc_ref = DVec3::ZERO;
    }

    /// Restart the velocity reference from the measured velocity at the next
    /// [`shape`](Self::shape).
    pub fn reset_reference(&mut self) {
        self.vel_ref = None;
        self.acc_ref = DVec3::ZERO;
    }

    /// Estimated disturbance acceleration (world, m/s²).
    pub fn disturbance(&self) -> DVec3 {
        self.disturbance
    }

    /// Record the acceleration the thrust and gravity produced during one control step (world,
    /// m/s²); [`update`](Self::update) compares their mean with the measured velocity change.
    pub fn record_applied(&mut self, acc: DVec3) {
        self.applied_sum += acc;
        self.applied_steps += 1;
    }

    /// Clamp a velocity setpoint to the speed limits.
    pub fn limit_velocity(&self, v: DVec3) -> DVec3 {
        let l = &self.limits;
        let xy = v.truncate().clamp_length_max(l.vel_xy);
        DVec3::new(xy.x, xy.y, v.z.clamp(-l.vel_down, l.vel_up))
    }

    /// Velocity setpoint (world, m/s) towards position setpoint `pos_sp`.
    pub fn velocity_setpoint(&self, pos_sp: DVec3, pos: DVec3) -> DVec3 {
        self.limit_velocity(self.gains.pos_p * (pos_sp - pos))
    }

    /// Advance the velocity reference by `dt` towards the command `vel_cmd` and return it with
    /// its acceleration (the feedforward for [`update`](Self::update)). The reference starts
    /// at the measured velocity `vel` after a reset and follows `v̈ = ω²(v_cmd − v) − 2ω·v̇`
    /// (`ω`: [`PositionGains::reference`]), with the acceleration limited to `accel_xy`
    /// horizontally and `accel_up` / `accel_down` vertically, and towards the command to
    /// `ω·|v_cmd − v|` so that it does not overshoot after running at a limit.
    pub fn shape(&mut self, vel_cmd: DVec3, vel: DVec3, dt: f64) -> (DVec3, DVec3) {
        let (w, l) = (self.gains.reference, &self.limits);
        if w <= 0.0 {
            return (vel_cmd, DVec3::ZERO);
        }
        let v = *self.vel_ref.get_or_insert(vel);
        let a = self.acc_ref + (w * w * (vel_cmd - v) - 2.0 * w * self.acc_ref) * dt;
        let e = vel_cmd - v;
        // Horizontal: limit the length, then the component along the error.
        let mut a_xy = a.truncate().clamp_length_max(l.accel_xy);
        let (e_len, e_dir) = (e.truncate().length(), e.truncate().normalize_or_zero());
        let excess = a_xy.dot(e_dir) - w * e_len;
        if excess > 0.0 {
            a_xy -= e_dir * excess;
        }
        let mut a_z = a.z.clamp(-l.accel_down, l.accel_up);
        if a_z * e.z > 0.0 && a_z.abs() > w * e.z.abs() {
            a_z = w * e.z;
        }
        self.acc_ref = a_xy.extend(a_z);
        let v = v + self.acc_ref * dt;
        self.vel_ref = Some(v);
        (v, self.acc_ref)
    }

    /// Thrust vector (world, N) that drives the velocity `vel` towards `vel_sp`, with the
    /// feedforward acceleration `acc_ff` (world, m/s²). `dt` is the time since the previous
    /// update. With `observe` false (e.g. while resting on the ground, whose support is not a
    /// disturbance to fight) the observer is cleared.
    pub fn update(&mut self, vel_sp: DVec3, acc_ff: DVec3, vel: DVec3, dt: f64, observe: bool) -> DVec3 {
        let (g, l) = (self.gravity, &self.limits);
        match self.prev_vel {
            Some(prev) if observe && self.applied_steps > 0 => {
                let residual = (vel - prev) / dt - self.applied_sum / f64::from(self.applied_steps);
                let alpha = 1.0 - (-self.gains.disturbance * dt).exp();
                let d = self.disturbance + (residual - self.disturbance) * alpha;
                if d.is_finite() {
                    self.disturbance = d.clamp_length_max(g);
                }
            }
            _ if !observe => self.disturbance = DVec3::ZERO,
            _ => {}
        }
        self.prev_vel = Some(vel);
        self.applied_sum = DVec3::ZERO;
        self.applied_steps = 0;

        let acc_sp = acc_ff + self.gains.vel_p * (vel_sp - vel) - self.disturbance;

        // Acceleration → thrust direction (tilt-limited) and the collective that meets the
        // vertical demand.
        let body_z = limit_tilt(DVec3::new(acc_sp.x, acc_sp.y, g + acc_sp.z).normalize_or(DVec3::Z), l.tilt);
        let collective = (self.mass * (g + acc_sp.z) / body_z.z).max(l.thrust_min);
        let mut thr = body_z * collective;

        // Vertical priority with a horizontal margin.
        let t2 = l.thrust_max * l.thrust_max;
        let xy = thr.truncate();
        let margin = xy.length().min(l.xy_margin);
        thr.z = thr.z.min((t2 - margin * margin).max(0.0).sqrt());
        let xy = xy.clamp_length_max((t2 - thr.z * thr.z).max(0.0).sqrt());
        thr.x = xy.x;
        thr.y = xy.y;
        thr
    }
}

/// Rotate the unit vector `v` towards world z so that it makes at most `max` (rad) with it.
pub fn limit_tilt(v: DVec3, max: f64) -> DVec3 {
    if v.z.clamp(-1.0, 1.0).acos() <= max {
        return v;
    }
    let dir = DVec3::new(v.x, v.y, 0.0).normalize_or(DVec3::X);
    let (s, c) = max.sin_cos();
    DVec3::Z * c + dir * s
}

/// Attitude (body → world) with the body z-axis along `thrust` and heading `yaw` (PX4
/// `bodyzToAttitude`).
pub fn attitude_from_thrust(thrust: DVec3, yaw: f64) -> DQuat {
    let z = thrust.normalize_or(DVec3::Z);
    let y_c = DVec3::new(-yaw.sin(), yaw.cos(), 0.0);
    let mut x = y_c.cross(z);
    if z.z < 0.0 {
        // Keep the nose on the heading when inverted.
        x = -x;
    }
    let x = x.try_normalize().unwrap_or_else(|| z.any_orthonormal_vector());
    DQuat::from_mat3(&DMat3::from_cols(x, z.cross(x), z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::math::quat::{tilt, yaw};

    const G: f64 = 9.80665;

    fn ctrl() -> PositionController {
        let gains =
            PositionGains { pos_p: DVec3::splat(1.5), vel_p: DVec3::splat(4.0), disturbance: 3.0, reference: 3.0 };
        let limits = PositionLimits {
            vel_xy: 5.0,
            vel_up: 3.0,
            vel_down: 1.5,
            tilt: 0.6,
            thrust_min: 2.0,
            thrust_max: 20.0,
            xy_margin: 6.0,
            accel_xy: 4.0,
            accel_up: 3.0,
            accel_down: 2.0,
        };
        PositionController::new(gains, limits, 1.0, G)
    }

    #[test]
    fn thrust_vector_geometry() {
        let q = attitude_from_thrust(DVec3::Z, 0.7);
        assert!((yaw(q) - 0.7).abs() < 1e-12 && tilt(q) < 1e-12);
        let t = DVec3::new(1.0, -2.0, 5.0);
        let q = attitude_from_thrust(t, -2.0);
        assert!((q * DVec3::Z - t.normalize()).length() < 1e-12);
        // The body x-axis stays in the vertical plane of the heading.
        assert!((q * DVec3::X).dot(DVec3::new(-(-2.0f64).sin(), (-2.0f64).cos(), 0.0)).abs() < 1e-12);
        let v = limit_tilt(DVec3::new(1.0, 1.0, 0.1).normalize(), 0.6);
        assert!((v.z.acos() - 0.6).abs() < 1e-12 && (v.x - v.y).abs() < 1e-12);
    }

    #[test]
    fn hover_and_limits() {
        let mut c = ctrl();
        assert!(
            (c.update(DVec3::ZERO, DVec3::ZERO, DVec3::ZERO, 0.01, true) - DVec3::new(0.0, 0.0, G)).length() < 1e-12
        );
        // Large horizontal demand: tilt limited, vertical demand met.
        let thr = c.update(DVec3::new(50.0, 0.0, 0.0), DVec3::ZERO, DVec3::ZERO, 0.01, true);
        assert!((thr.z - G).abs() < 1e-9 && (thr.x / thr.z - 0.6f64.tan()).abs() < 1e-9);
        // Climb beyond the thrust limit: vertical capped so that the margin stays available.
        let thr = c.update(DVec3::new(50.0, 0.0, 50.0), DVec3::ZERO, DVec3::ZERO, 0.01, true);
        assert!((thr.length() - 20.0).abs() < 1e-9 && (thr.x - 6.0).abs() < 1e-9, "{thr}");
        // Velocity limits.
        let v = c.velocity_setpoint(DVec3::new(10.0, 10.0, -10.0), DVec3::ZERO);
        assert!((v.truncate().length() - 5.0).abs() < 1e-12 && v.z == -1.5);
    }

    /// A constant unmodelled acceleration is estimated at the observer bandwidth and the error
    /// vanishes; a setpoint step leaves the estimate alone.
    #[test]
    fn disturbance_observer() {
        let mut c = ctrl();
        let d = DVec3::new(0.5, -0.2, -1.0);
        let (dt, mut v, mut err_at_step) = (0.01, DVec3::ZERO, 0.0);
        for k in 0..800 {
            let vel_sp = if k < 400 { DVec3::ZERO } else { DVec3::new(2.0, 0.0, 0.0) };
            let thr = c.update(vel_sp, DVec3::ZERO, v, dt, true);
            // Plant: the commanded thrust acts at once, plus the disturbance.
            let applied = thr / 1.0 - DVec3::new(0.0, 0.0, G);
            for _ in 0..5 {
                c.record_applied(applied);
            }
            v += (applied + d) * dt;
            let err = (c.disturbance() - d).length();
            if k == 399 {
                assert!(err < 1e-3 && v.length() < 1e-3, "{err} {v}");
                err_at_step = err;
            } else if k >= 400 {
                assert!(err <= err_at_step);
            }
        }
        assert!((c.disturbance() - d).length() < 1e-6 && (v.x - 2.0).abs() < 1e-3, "{v}");
        c.update(DVec3::ZERO, DVec3::ZERO, v, dt, false);
        assert_eq!(c.disturbance(), DVec3::ZERO);
    }

    /// The reference reaches a step command within the acceleration limits, with continuous
    /// acceleration and without overshoot; alternating commands move it little and keep their
    /// mean.
    #[test]
    fn setpoint_shaping() {
        let mut c = ctrl();
        let (dt, cmd) = (0.01, DVec3::new(3.0, -4.0, 2.0));
        let (mut prev_a, mut t_done) = (DVec3::ZERO, None);
        for k in 0..600 {
            let (v, a) = c.shape(cmd, DVec3::ZERO, dt);
            assert!(a.truncate().length() <= 4.0 + 1e-9 && (-2.0 - 1e-9..=3.0 + 1e-9).contains(&a.z), "{a}");
            // Continuous: at most ω²·|e|·dt per step.
            assert!((a - prev_a).length() <= 9.0 * 6.0 * dt, "{k} {a} {prev_a}");
            // No overshoot along the command direction or vertically.
            assert!(v.truncate().dot(cmd.truncate().normalize()) <= 5.0 + 1e-6 && v.z <= 2.0 + 1e-6, "{v}");
            prev_a = a;
            if t_done.is_none() && (v - cmd).length() < 0.1 {
                t_done = Some(k as f64 * dt);
            }
        }
        // Within 2 % of the 5 m/s step in about the time a 4 m/s² ramp takes plus the tail.
        let t = t_done.expect("reference reached the command");
        assert!((1.3..2.5).contains(&t), "{t}");
        // A ±3 m/s square wave switching every 40 ms: small and zero-mean.
        let mut c = ctrl();
        let (mut peak_a, mut peak_v, mut sum) = (0.0f64, 0.0f64, 0.0);
        for k in 0..800 {
            let x = if (k / 4) % 2 == 0 { 3.0 } else { -3.0 };
            let (v, a) = c.shape(DVec3::new(x, 0.0, 0.0), DVec3::ZERO, dt);
            peak_a = peak_a.max(a.length());
            peak_v = peak_v.max(v.length());
            sum += v.x;
        }
        assert!(peak_a < 1.2 && peak_v < 0.3 && (sum / 800.0).abs() < 0.05, "{peak_a} {peak_v} {sum}");
        // Without a reference frequency the command passes through.
        let mut c = ctrl();
        c.gains.reference = 0.0;
        assert_eq!(c.shape(cmd, DVec3::ZERO, dt), (cmd, DVec3::ZERO));
    }
}
