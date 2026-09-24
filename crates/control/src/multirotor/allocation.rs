//! Control allocation: body torque and collective thrust → rotor thrusts → rotor speeds.
//!
//! The effectiveness matrix `B` maps rotor thrusts (N) to `[τ_x, τ_y, τ_z, T]`: body torque
//! (N·m) and thrust along body z (N). Its pseudo-inverse `B⁺ = Bᵀ(BBᵀ)⁻¹` distributes a request.
//! When rotors saturate, PX4's sequential desaturation decides what gives way: collective thrust
//! first (it is only ever reduced, except in airmode), then roll and pitch, and yaw last. Yaw
//! may use 15 % headroom above the top thrust.

use crate::ControlError;
use autonomousim_vehicles::multirotor::{MAX_ROTORS, MultirotorDef};
use glam::{DMat4, DVec3};

type PerRotor = [f64; MAX_ROTORS];

/// Headroom above the top thrust that yaw may use, as a fraction of the thrust range
/// (PX4 `MINIMUM_YAW_MARGIN`).
const YAW_MARGIN: f64 = 0.15;

#[derive(Clone, Debug)]
pub struct Allocator {
    n: usize,
    /// Rows of `B`: roll, pitch and yaw torque and thrust per newton of each rotor's thrust.
    b: [PerRotor; 4],
    /// Columns of `B⁺`: rotor thrusts per unit of roll, pitch and yaw torque and of thrust.
    pinv: [PerRotor; 4],
    /// Thrust per (rad/s)² at the design air density.
    k_thrust: f64,
    f_min: f64,
    f_max: f64,
    airmode: bool,
}

impl Allocator {
    /// Allocation for `def` at air density `density` (kg/m³). With `airmode`, the collective may
    /// also rise to keep roll and pitch authority at low thrust.
    pub fn new(def: &MultirotorDef, density: f64, airmode: bool) -> Result<Self, ControlError> {
        let n = def.rotors.len();
        let kappa = def.rotor.k_torque / def.rotor.k_thrust;
        let mut b = [[0.0; MAX_ROTORS]; 4];
        for (i, m) in def.rotors.iter().enumerate() {
            let t = m.position.cross(m.axis) - m.axis * (m.spin.sign() * kappa);
            [b[0][i], b[1][i], b[2][i], b[3][i]] = [t.x, t.y, t.z, m.axis.z];
        }
        let dot = |r: usize, c: usize| (0..n).map(|i| b[r][i] * b[c][i]).sum::<f64>();
        let norm: [f64; 4] = std::array::from_fn(|r| dot(r, r).sqrt());
        // Rank test on the Gram matrix of the unit-scaled rows (independent of units and size).
        let gram = |scaled: bool| {
            let s = |r: usize| if scaled { norm[r] } else { 1.0 };
            DMat4::from_cols_array_2d(&std::array::from_fn(|c| std::array::from_fn(|r| dot(r, c) / (s(r) * s(c)))))
        };
        let det = gram(true).determinant();
        if det.is_nan() || det <= 1e-6 {
            return Err(ControlError::Unallocatable(format!(
                "{}: rotors cannot produce roll, pitch, yaw and thrust independently (det {det:.2e})",
                def.name
            )));
        }
        let ginv = gram(false).inverse();
        let mut pinv = [[0.0; MAX_ROTORS]; 4];
        for (k, col) in pinv.iter_mut().enumerate() {
            for (i, x) in col.iter_mut().enumerate().take(n) {
                *x = (0..4).map(|j| b[j][i] * ginv.col(k)[j]).sum();
            }
        }
        let k_thrust = def.rotor.k_thrust * density / def.rotor.reference_density;
        let mut a = Self { n, b, pinv, k_thrust, f_min: 0.0, f_max: 0.0, airmode };
        a.set_speed_limits(def.rotor.omega_min, def.rotor.omega_max);
        Ok(a)
    }

    pub fn num_rotors(&self) -> usize {
        self.n
    }

    /// Entry `(row, rotor)` of `B`; rows are roll, pitch, yaw torque and thrust.
    pub fn effectiveness(&self, row: usize, rotor: usize) -> f64 {
        self.b[row][rotor]
    }

    /// Entry `(rotor, column)` of `B⁺`.
    pub fn pseudo_inverse(&self, rotor: usize, column: usize) -> f64 {
        self.pinv[column][rotor]
    }

    /// Update the feasible rotor speed range (rad/s), e.g. as the battery sags.
    pub fn set_speed_limits(&mut self, omega_min: f64, omega_max: f64) {
        self.f_min = self.k_thrust * omega_min * omega_min;
        self.f_max = self.k_thrust * omega_max * omega_max;
    }

    /// Per-rotor thrust range (N).
    pub fn thrust_limits(&self) -> (f64, f64) {
        (self.f_min, self.f_max)
    }

    /// Collective thrust with every rotor at full thrust (N).
    pub fn max_collective(&self) -> f64 {
        self.b[3][..self.n].iter().sum::<f64>() * self.f_max
    }

    /// Torque each axis gains from driving the rotors from mid-range to their limits (N·m).
    pub fn torque_authority(&self) -> DVec3 {
        let half = 0.5 * (self.f_max - self.f_min);
        DVec3::from_array(std::array::from_fn(|r| self.b[r][..self.n].iter().map(|x| x.abs()).sum::<f64>() * half))
    }

    /// Torque and thrust produced by rotor thrusts `f`.
    pub fn wrench(&self, f: &[f64]) -> (DVec3, f64) {
        let row = |r: usize| self.b[r][..self.n].iter().zip(f).map(|(b, f)| b * f).sum::<f64>();
        (DVec3::new(row(0), row(1), row(2)), row(3))
    }

    /// Rotor speed (rad/s) that produces thrust `f` (N) at the design density.
    pub fn speed(&self, f: f64) -> f64 {
        (f.max(0.0) / self.k_thrust).sqrt()
    }

    /// Rotor thrusts (N, within the limits) for body torque `torque` (N·m) and collective
    /// `thrust` (N), with sequential desaturation.
    pub fn allocate(&self, torque: DVec3, thrust: f64, f: &mut [f64]) {
        let f = &mut f[..self.n];
        let [roll, pitch, yaw, thr] = &self.pinv;
        for (i, x) in f.iter_mut().enumerate() {
            *x = roll[i] * torque.x + pitch[i] * torque.y + thr[i] * thrust;
        }
        let (lo, hi) = (self.f_min, self.f_max);
        if self.airmode {
            desaturate(f, thr, lo, hi, false);
        } else {
            desaturate(f, thr, lo, hi, true);
            desaturate(f, roll, lo, hi, false);
            desaturate(f, pitch, lo, hi, false);
        }
        for (x, y) in f.iter_mut().zip(yaw) {
            *x += y * torque.z;
        }
        desaturate(f, yaw, lo, hi + YAW_MARGIN * (hi - lo), false);
        desaturate(f, thr, lo, hi, true);
        for x in f.iter_mut() {
            *x = x.clamp(lo, hi);
        }
    }
}

/// Shift `f` along `v` to reduce the violation of `[lo, hi]` (PX4 `desaturateActuators`): one
/// full step, then half of the remaining correction to balance violations on both sides.
/// With `reduce_only`, a first step that would move along `+v` is skipped.
fn desaturate(f: &mut [f64], v: &PerRotor, lo: f64, hi: f64, reduce_only: bool) {
    let gain = desaturation_gain(f, v, lo, hi);
    if reduce_only && gain > 0.0 {
        return;
    }
    for (x, y) in f.iter_mut().zip(v) {
        *x += gain * y;
    }
    let gain = 0.5 * desaturation_gain(f, v, lo, hi);
    for (x, y) in f.iter_mut().zip(v) {
        *x += gain * y;
    }
}

/// Sum of the most negative and most positive shifts along `v` that bring a saturated entry
/// back to its bound.
fn desaturation_gain(f: &[f64], v: &PerRotor, lo: f64, hi: f64) -> f64 {
    let eps = 1e-9 * v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let (mut k_min, mut k_max) = (0.0f64, 0.0f64);
    for (&x, &y) in f.iter().zip(v) {
        if y.abs() <= eps {
            continue;
        }
        let bound = if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            continue;
        };
        let k = (bound - x) / y;
        k_min = k_min.min(k);
        k_max = k_max.max(k);
    }
    k_min + k_max
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_vehicles::multirotor::{RotorMount, Spin};
    use autonomousim_vehicles::presets;

    fn hexa() -> MultirotorDef {
        let mut def = presets::multirotor("iris_like").unwrap();
        def.rotors = (0..6)
            .map(|i| {
                let a = std::f64::consts::FRAC_PI_3 * i as f64;
                let spin = if i % 2 == 0 { Spin::Ccw } else { Spin::Cw };
                RotorMount { position: DVec3::new(a.cos(), a.sin(), 0.0) * 0.25, axis: DVec3::Z, spin }
            })
            .collect();
        def
    }

    fn defs() -> Vec<MultirotorDef> {
        vec![presets::multirotor("cf2x").unwrap(), presets::multirotor("iris_like").unwrap(), hexa()]
    }

    #[test]
    fn pseudo_inverse_is_a_right_inverse() {
        for def in defs() {
            let a = Allocator::new(&def, 1.225, false).unwrap();
            for r in 0..4 {
                for c in 0..4 {
                    let bb: f64 = (0..a.n).map(|i| a.effectiveness(r, i) * a.pseudo_inverse(i, c)).sum();
                    let expect = if r == c { 1.0 } else { 0.0 };
                    assert!((bb - expect).abs() < 1e-12, "{} B·B⁺[{r}][{c}] = {bb}", def.name);
                }
            }
        }
    }

    #[test]
    fn unsaturated_requests_are_met_exactly() {
        for def in defs() {
            let a = Allocator::new(&def, 1.225, false).unwrap();
            let hover = def.body.mass * 9.80665;
            let auth = a.torque_authority();
            let torque = auth * DVec3::new(0.2, -0.15, 0.1);
            let mut f = [0.0; MAX_ROTORS];
            a.allocate(torque, hover, &mut f);
            let (t, thrust) = a.wrench(&f[..a.n]);
            assert!((t - torque).length() < 1e-12 * auth.length() && (thrust - hover).abs() < 1e-12 * hover);
        }
    }

    #[test]
    fn saturation_sacrifices_thrust_then_yaw_before_roll_and_pitch() {
        for def in defs() {
            let a = Allocator::new(&def, 1.225, false).unwrap();
            let (lo, hi) = a.thrust_limits();
            let auth = a.torque_authority();
            let mut f = [0.0; MAX_ROTORS];
            let check = |f: &[f64]| assert!(f[..a.n].iter().all(|&x| (lo..=hi).contains(&x)));

            // Full collective plus roll: thrust gives way, roll is met exactly.
            let roll = DVec3::new(0.3 * auth.x, 0.0, 0.0);
            a.allocate(roll, a.max_collective(), &mut f);
            check(&f);
            let (t, thrust) = a.wrench(&f[..a.n]);
            assert!((t - roll).length() < 1e-9 * auth.x, "{}: {t}", def.name);
            assert!(thrust < a.max_collective());

            // Roll and yaw beyond authority at hover: yaw is cut, roll kept. Yaw may push rotors
            // 15 % past the top before the final clamp, which can cost a little roll.
            let hover = def.body.mass * 9.80665;
            let req = DVec3::new(0.5 * auth.x, 0.0, 2.0 * auth.z);
            a.allocate(req, hover, &mut f);
            check(&f);
            let (t, _) = a.wrench(&f[..a.n]);
            assert!((t.x - req.x).abs() < 0.05 * req.x, "{}: {t}", def.name);
            assert!(t.z > 0.0 && t.z < 0.5 * req.z);

            // Airmode disabled: zero collective never raises thrust, so no torque at all.
            a.allocate(roll, 0.0, &mut f);
            check(&f);
            assert!(f[..a.n].iter().all(|&x| x == lo));
            // Airmode keeps roll authority by raising the collective.
            let air = Allocator::new(&def, 1.225, true).unwrap();
            air.allocate(roll, 0.0, &mut f);
            let (t, thrust) = air.wrench(&f[..a.n]);
            assert!((t - roll).length() < 1e-9 * auth.x && thrust > 0.0);
        }
    }

    #[test]
    fn degenerate_layouts_are_rejected() {
        let mut def = presets::multirotor("cf2x").unwrap();
        for r in &mut def.rotors {
            r.spin = Spin::Ccw;
        }
        assert!(matches!(Allocator::new(&def, 1.225, false), Err(ControlError::Unallocatable(_))));
    }
}
