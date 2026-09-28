//! Steady straight flight: Newton's method on the full six-degree-of-freedom equilibrium.
//!
//! Unknowns: angle of attack α, pitch θ, bank φ, aileron, elevator and rudder deflections,
//! throttle and rotor speed; sideslip is zero and the heading north-east irrelevant. Equations:
//! the six body accelerations, the rotor torque balance and the flight-path angle. The bank and
//! the aileron and rudder absorb the propeller's torque and slipstream asymmetries.

use super::model::FixedWing;
use crate::VehicleError;
use glam::{DQuat, DVec3};

/// Trimmed straight flight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trim {
    /// Airspeed (m/s), density (kg/m³) and flight-path angle (rad) trimmed for.
    pub airspeed: f64,
    pub density: f64,
    pub flight_path: f64,
    pub alpha: f64,
    /// Pitch and bank (rad; pitch positive nose up, bank positive right wing down).
    pub pitch: f64,
    pub bank: f64,
    /// Aileron, elevator, rudder and flap deflections (rad).
    pub surfaces: [f64; 4],
    /// The normalised controls that command them (brake released).
    pub controls: super::FixedWingInput,
    pub rotor_speed: f64,
    /// Largest scaled residual left.
    pub residual: f64,
}

impl Trim {
    /// Body → world rotation for heading `yaw` (rad, counter-clockwise from east).
    pub fn attitude(&self, yaw: f64) -> DQuat {
        DQuat::from_rotation_z(yaw) * DQuat::from_rotation_y(-self.pitch) * DQuat::from_rotation_x(self.bank)
    }

    /// Air-relative velocity in the body frame.
    pub fn velocity_body(&self) -> DVec3 {
        let (s, c) = self.alpha.sin_cos();
        DVec3::new(c, 0.0, -s) * self.airspeed
    }
}

const N: usize = 8;

/// Solve `a·x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for k in 0..N {
        let p = (k..N).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[p][k].abs() < 1e-300 {
            return None;
        }
        a.swap(k, p);
        b.swap(k, p);
        for i in k + 1..N {
            let f = a[i][k] / a[k][k];
            for j in k..N {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = [0.0; N];
    for k in (0..N).rev() {
        let s: f64 = (k + 1..N).map(|j| a[k][j] * x[j]).sum();
        x[k] = (b[k] - s) / a[k][k];
    }
    Some(x)
}

/// Levenberg–Marquardt step `(JᵀJ + μI)·dx = −Jᵀr`.
fn damped_step(jac: &[[f64; N]; N], r: &[f64; N]) -> Option<[f64; N]> {
    let mut a = [[0.0; N]; N];
    let mut b = [0.0; N];
    for i in 0..N {
        for j in 0..N {
            a[i][j] = (0..N).map(|k| jac[k][i] * jac[k][j]).sum();
        }
        b[i] = -(0..N).map(|k| jac[k][i] * r[k]).sum::<f64>();
    }
    let mu = 1e-6 * (0..N).map(|i| a[i][i]).fold(0.0, f64::max).max(1e-12);
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += mu;
    }
    solve(a, b)
}

impl FixedWing {
    /// Scaled equilibrium residuals at unknowns `x` = (α, θ, φ, δa, δe, δr, throttle, Ω).
    fn trim_residual(&self, x: &[f64; N], airspeed: f64, rho: f64, gamma: f64, flap: f64, g: f64) -> [f64; N] {
        let [alpha, theta, phi, da, de, dr, throttle, omega] = *x;
        let rot = DQuat::from_rotation_y(-theta) * DQuat::from_rotation_x(phi);
        let (sa, ca) = alpha.sin_cos();
        let v_body = DVec3::new(ca, 0.0, -sa) * airspeed;
        let (f, m, rotor) = self.static_loads(v_body, rho, [da, de, dr, flap], throttle, omega);
        let lin = f / self.mass() + rot.inverse() * DVec3::new(0.0, 0.0, -g);
        let ang = self.inertia().inverse() * m;
        let climb = (rot * v_body).z / airspeed - gamma.sin();
        let rotor_acc = rotor / self.propulsion().inertia;
        [lin.x / g, lin.y / g, lin.z / g, ang.x, ang.y, ang.z, rotor_acc * 1e-3, climb]
    }

    /// Trim for straight flight at `airspeed` (m/s) in air of density `rho`, on flight-path
    /// angle `gamma` (rad, positive climbing), with flaps at `flap` (0–1) and gravity `g`.
    /// Fails when the controls or throttle would leave their ranges or Newton stalls.
    pub fn trim(&self, airspeed: f64, rho: f64, gamma: f64, flap: f64, g: f64) -> Result<Trim, VehicleError> {
        let fail =
            |m: String| Err(VehicleError::Invalid(format!("{}: cannot trim at {airspeed} m/s: {m}", self.def().name)));
        if !(airspeed > 0.0 && rho > 0.0 && g > 0.0) {
            return fail("need positive airspeed, density and gravity".into());
        }
        let flap_deflection = self.def().controls.flap.deflection(flap.clamp(0.0, 1.0));
        let supply = self.def().battery.as_ref().map_or(0.0, |b| b.full_voltage());
        let omega0 = self.propulsion().steady_omega(0.8, airspeed, rho, supply).max(10.0);
        let mut x = [0.05, 0.05 + gamma, 0.0, 0.0, 0.0, 0.0, 0.8, omega0];
        let res = |x: &[f64; N]| self.trim_residual(x, airspeed, rho, gamma, flap_deflection, g);
        let norm = |r: &[f64; N]| r.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut r = res(&x);
        for _ in 0..60 {
            if norm(&r) < 1e-11 {
                break;
            }
            let mut jac = [[0.0; N]; N];
            for j in 0..N {
                let h = 1e-6 * x[j].abs().max(1.0);
                let (mut xp, mut xm) = (x, x);
                xp[j] += h;
                xm[j] -= h;
                let (rp, rm) = (res(&xp), res(&xm));
                for i in 0..N {
                    jac[i][j] = (rp[i] - rm[i]) / (2.0 * h);
                }
            }
            // A clamped motor current can zero the throttle column: fall back to a damped
            // least-squares step.
            let Some(dx) = solve(jac, r.map(|v| -v)).or_else(|| damped_step(&jac, &r)) else {
                return fail("singular Jacobian".into());
            };
            // Backtracking line search on the residual norm.
            let mut step = 1.0;
            loop {
                let mut xn = x;
                for i in 0..N {
                    xn[i] += step * dx[i];
                }
                xn[7] = xn[7].max(1.0);
                let rn = res(&xn);
                if norm(&rn) < norm(&r) || step < 1e-4 {
                    (x, r) = (xn, rn);
                    break;
                }
                step *= 0.5;
            }
        }
        let residual = r.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if residual > 1e-6 {
            return fail(format!("no convergence (residual {residual:.2e})"));
        }
        let [alpha, pitch, bank, da, de, dr, throttle, omega] = x;
        // Deflections back to normalised commands.
        let k = &self.def().controls;
        let signs = self.control_signs();
        let command = |d: f64, s: &super::SurfaceDef, sign: f64| {
            let w = if d >= 0.0 { d / s.max } else { d / -s.min.unwrap_or(-s.max) };
            sign * w
        };
        let controls = super::FixedWingInput {
            aileron: command(da, &k.aileron, signs[0]),
            elevator: command(de, &k.elevator, signs[1]),
            rudder: command(dr, &k.rudder, signs[2]),
            throttle,
            flap: flap.clamp(0.0, 1.0),
            brake: 0.0,
        };
        for (name, c) in [("aileron", controls.aileron), ("elevator", controls.elevator), ("rudder", controls.rudder)] {
            if !c.is_finite() || c.abs() > 1.0 + 1e-9 {
                return fail(format!("{name} {c:.3} out of range"));
            }
        }
        if !(-1e-9..=1.0 + 1e-9).contains(&throttle) {
            return fail(format!("throttle {throttle:.3} out of range"));
        }
        Ok(Trim {
            airspeed,
            density: rho,
            flight_path: gamma,
            alpha,
            pitch,
            bank,
            surfaces: [da, de, dr, flap_deflection],
            controls: super::FixedWingInput { throttle: throttle.clamp(0.0, 1.0), ..controls.clamped() },
            rotor_speed: omega,
            residual,
        })
    }
}
