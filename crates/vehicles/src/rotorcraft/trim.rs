//! Trim of a helicopter in steady level flight: the controls, pitch and roll that balance
//! forces and moments with the rotors settled and the rotor speed governed.

use super::model::{Helicopter, HelicopterInput, HelicopterLoads};
use super::rotor::RotorState;
use crate::aero::{AirData, AirFlow};
use glam::{DQuat, DVec3};

const MAX_ITERATIONS: usize = 60;
/// Residual (forces over weight, moments over weight × rotor radius) of a trimmed state.
const TOLERANCE: f64 = 1e-10;
const STEP: f64 = 1e-7;

/// A trimmed flight condition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HelicopterTrim {
    /// Pilot inputs (within [−1, 1]).
    pub controls: HelicopterInput,
    /// Pitch (nose up) and roll (right wing down) angles (rad).
    pub pitch: f64,
    pub roll: f64,
    /// Air-relative velocity in the body frame (m/s).
    pub velocity_body: DVec3,
    /// Main rotor speed (rad/s) and the settled rotor states.
    pub rotor_speed: f64,
    pub main: RotorState,
    pub tail: RotorState,
    pub loads: HelicopterLoads,
    /// Engine power (W).
    pub power: f64,
}

impl HelicopterTrim {
    /// Body → world rotation for a heading (yaw, rad, counter-clockwise from +x).
    pub fn attitude(&self, heading: f64) -> DQuat {
        attitude(heading, self.pitch, self.roll)
    }
}

/// `R_z(ψ)·R_y(−θ)·R_x(φ)`: FLU yaw, nose-up pitch, right-wing-down roll.
pub fn attitude(heading: f64, pitch: f64, roll: f64) -> DQuat {
    DQuat::from_rotation_z(heading) * DQuat::from_rotation_y(-pitch) * DQuat::from_rotation_x(roll)
}

impl Helicopter {
    /// Trim in straight level flight at `speed` (m/s, along the heading, no sideslip in the
    /// world) in still air of density `rho` under gravity `g`, at the governed rotor speed.
    pub fn trim(&self, speed: f64, rho: f64, g: f64) -> Result<HelicopterTrim, String> {
        let def = self.def();
        let omega = def.engine.rated_speed;
        let weight = self.mass() * g;
        let scale = [1.0 / weight, 1.0 / (weight * def.main_rotor.rotor.radius)];
        let air = AirData { density: rho, ..AirData::default() };
        // Unknowns: the four inputs (unclamped), pitch and roll.
        let setup = |x: &[f64; 6]| {
            let q = attitude(0.0, x[4], x[5]);
            let v_body = q.inverse() * DVec3::new(speed, 0.0, 0.0);
            let flow = AirFlow::new(&air, v_body, DVec3::ZERO);
            let input = HelicopterInput::from_array([x[0], x[1], x[2], x[3]]);
            let pitches = self.pitches_linear(&input);
            (q, flow, pitches)
        };
        let residual = |x: &[f64; 6]| {
            let (q, flow, pitches) = setup(x);
            let (f, m, _) = self.steady_wrench(&flow, omega, &pitches);
            let f = f + q.inverse() * DVec3::new(0.0, 0.0, -weight);
            let (f, m) = (f * scale[0], m * scale[1]);
            [f.x, f.y, f.z, m.x, m.y, m.z]
        };
        let mut x = [0.0; 6];
        let mut r = residual(&x);
        for _ in 0..MAX_ITERATIONS {
            if norm(&r) < TOLERANCE {
                break;
            }
            let mut jac = [[0.0; 6]; 6];
            for j in 0..6 {
                let mut xp = x;
                xp[j] += STEP;
                let rp = residual(&xp);
                for i in 0..6 {
                    jac[i][j] = (rp[i] - r[i]) / STEP;
                }
            }
            let dx = solve(jac, r).ok_or("singular trim Jacobian")?;
            // Damped step: at most 0.5 in any input or angle.
            let big = dx.iter().fold(0.0f64, |a, d| a.max(d.abs()));
            let k = if big > 0.5 { 0.5 / big } else { 1.0 };
            for (xi, d) in x.iter_mut().zip(dx) {
                *xi -= k * d;
            }
            r = residual(&x);
        }
        let residual = norm(&r);
        if residual.is_nan() || residual >= TOLERANCE * 100.0 {
            return Err(format!("{}: no trim at {speed} m/s (residual {residual:.2e})", def.name));
        }
        if x[..4].iter().any(|u| u.abs() > 1.0) {
            return Err(format!("{}: trim at {speed} m/s needs controls beyond their range: {:?}", def.name, &x[..4]));
        }
        let (_, flow, pitches) = setup(&x);
        let (loads, main, tail) = self.settled(&flow, omega, &pitches);
        Ok(HelicopterTrim {
            controls: HelicopterInput::from_array([x[0], x[1], x[2], x[3]]),
            pitch: x[4],
            roll: x[5],
            velocity_body: flow.velocity,
            rotor_speed: omega,
            main,
            tail,
            power: (loads.main.torque + def.tail_gear_ratio * loads.tail.torque) * omega,
            loads,
        })
    }
}

fn norm(r: &[f64; 6]) -> f64 {
    r.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Solve `A·x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for c in 0..6 {
        let p = (c..6).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-14 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..6 {
            let f = a[r][c] / a[c][c];
            for k in c..6 {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; 6];
    for c in (0..6).rev() {
        let s: f64 = (c + 1..6).map(|k| a[c][k] * x[k]).sum();
        x[c] = (b[c] - s) / a[c][c];
    }
    Some(x)
}
