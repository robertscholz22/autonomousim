//! Trim of a helicopter in steady straight flight (level, climbing or descending, or in
//! autorotation): the controls, pitch and roll that balance forces and moments with the rotors
//! settled at the governed rotor speed; and the linearisation about a trim.

use super::model::{Helicopter, HelicopterInput, HelicopterLoads};
use super::rotor::RotorState;
use crate::aero::{AirData, AirFlow};
use glam::{DQuat, DVec3};

const MAX_ITERATIONS: usize = 60;
/// Residual (forces over weight, moments and torques over weight × rotor radius) of a trimmed
/// state.
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
    /// Climb rate (m/s; negative descending).
    pub climb_rate: f64,
    /// Air-relative velocity in the body frame (m/s).
    pub velocity_body: DVec3,
    /// Main rotor speed (rad/s) and the settled rotor states.
    pub rotor_speed: f64,
    pub main: RotorState,
    pub tail: RotorState,
    pub loads: HelicopterLoads,
    /// Shaft power the rotors take (W; zero in autorotation).
    pub power: f64,
    /// Air density (kg/m³) and gravity (m/s²) of the trim.
    pub density: f64,
    pub gravity: f64,
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

/// Linear model about a trim, `ẋ = A·x + B·u`, with the state `x = [u, v, w, p, q, r, φ, θ]`
/// (body-frame velocity and rates, FLU; roll and pitch angles) and the input
/// `u = [collective, longitudinal, lateral, pedal]` (normalised). The rotors are
/// quasi-steady (settled flapping and inflow) at the trim's rotor speed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HelicopterLinear {
    pub a: [[f64; 8]; 8],
    pub b: [[f64; 4]; 8],
}

impl HelicopterLinear {
    /// Index of each state in `x`.
    pub const U: usize = 0;
    pub const V: usize = 1;
    pub const W: usize = 2;
    pub const P: usize = 3;
    pub const Q: usize = 4;
    pub const R: usize = 5;
    pub const PHI: usize = 6;
    pub const THETA: usize = 7;
}

/// What a trim holds fixed.
#[derive(Clone, Copy)]
enum Flight {
    /// Powered, at this climb rate.
    Powered(f64),
    /// Engine off: the descent rate is free and the rotors turn with no net torque.
    Autorotation,
}

impl Helicopter {
    /// Trim in straight level flight at `speed` (m/s, along the heading, no sideslip in the
    /// world) in still air of density `rho` under gravity `g`, at the governed rotor speed.
    pub fn trim(&self, speed: f64, rho: f64, g: f64) -> Result<HelicopterTrim, String> {
        self.solve_trim(speed, Flight::Powered(0.0), rho, g)
    }

    /// As [`trim`](Self::trim), climbing at `climb_rate` (m/s; negative descends).
    pub fn trim_climb(&self, speed: f64, climb_rate: f64, rho: f64, g: f64) -> Result<HelicopterTrim, String> {
        self.solve_trim(speed, Flight::Powered(climb_rate), rho, g)
    }

    /// Steady autorotation at horizontal `speed`, the rotors at the governed speed with no
    /// engine torque: the collective that holds the rotor speed, and the descent rate.
    pub fn trim_autorotation(&self, speed: f64, rho: f64, g: f64) -> Result<HelicopterTrim, String> {
        self.solve_trim(speed, Flight::Autorotation, rho, g)
    }

    fn solve_trim(&self, speed: f64, flight: Flight, rho: f64, g: f64) -> Result<HelicopterTrim, String> {
        let def = self.def();
        let omega = def.engine.rated_speed;
        let n = def.tail_gear_ratio;
        let weight = self.mass() * g;
        let radius = def.main_rotor.rotor.radius;
        let air = AirData { density: rho, ..AirData::default() };
        // Unknowns: the four inputs (unclamped), pitch and roll; in autorotation also the
        // climb rate.
        let setup = |x: &[f64]| {
            let climb = match flight {
                Flight::Powered(c) => c,
                Flight::Autorotation => x[6],
            };
            let q = attitude(0.0, x[4], x[5]);
            let v_body = q.inverse() * DVec3::new(speed, 0.0, climb);
            let flow = AirFlow::new(&air, v_body, DVec3::ZERO);
            let pitches = self.pitches_linear(&HelicopterInput::from_array([x[0], x[1], x[2], x[3]]));
            (flow, pitches, climb)
        };
        let residual = |x: &[f64]| {
            let (flow, pitches, _) = setup(x);
            let (f, m, loads) = self.steady_wrench(&flow, omega, &pitches);
            let q = attitude(0.0, x[4], x[5]);
            let f = (f + q.inverse() * DVec3::new(0.0, 0.0, -weight)) / weight;
            let m = m / (weight * radius);
            let mut r = vec![f.x, f.y, f.z, m.x, m.y, m.z];
            if let Flight::Autorotation = flight {
                r.push((loads.main.torque + n * loads.tail.torque) / (weight * radius));
            }
            r
        };
        let mut x = vec![0.0; 6];
        if let Flight::Autorotation = flight {
            // Start from a powered descent at the level-flight power over weight, which needs
            // little power: close to the autorotation.
            let level = self.trim(speed, rho, g)?;
            let sink = -level.power / weight;
            let near = self.trim_climb(speed, sink, rho, g).unwrap_or(level);
            let c = near.controls;
            x.copy_from_slice(&[c.collective, c.longitudinal, c.lateral, c.pedal, near.pitch, near.roll]);
            x.push(sink);
        }
        let err = newton(&mut x, residual);
        if err >= TOLERANCE * 100.0 {
            return Err(format!("{}: no trim at {speed} m/s (residual {err:.2e})", def.name));
        }
        if x[..4].iter().any(|u| u.abs() > 1.0) {
            return Err(format!("{}: trim at {speed} m/s needs controls beyond their range: {:?}", def.name, &x[..4]));
        }
        let (flow, pitches, climb_rate) = setup(&x);
        let (loads, main, tail) = self.settled(&flow, omega, &pitches);
        Ok(HelicopterTrim {
            controls: HelicopterInput::from_array([x[0], x[1], x[2], x[3]]),
            pitch: x[4],
            roll: x[5],
            climb_rate,
            velocity_body: flow.velocity,
            rotor_speed: omega,
            main,
            tail,
            power: (loads.main.torque + n * loads.tail.torque) * omega,
            loads,
            density: rho,
            gravity: g,
        })
    }

    /// Linear model about `trim` by central differences of the rigid-body equations with
    /// quasi-steady rotors (see [`HelicopterLinear`]).
    pub fn linearize(&self, trim: &HelicopterTrim) -> HelicopterLinear {
        let v = trim.velocity_body;
        let x0 = [v.x, v.y, v.z, 0.0, 0.0, 0.0, trim.roll, trim.pitch];
        let u0 = trim.controls.to_array();
        let mut lin = HelicopterLinear { a: [[0.0; 8]; 8], b: [[0.0; 4]; 8] };
        for j in 0..8 {
            let h = 1e-5 * x0[j].abs().max(1.0);
            let (mut xp, mut xm) = (x0, x0);
            xp[j] += h;
            xm[j] -= h;
            let (fp, fm) = (self.trim_derivative(trim, &xp, &u0), self.trim_derivative(trim, &xm, &u0));
            for i in 0..8 {
                lin.a[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }
        for j in 0..4 {
            let h = 1e-5;
            let (mut up, mut um) = (u0, u0);
            up[j] += h;
            um[j] -= h;
            let (fp, fm) = (self.trim_derivative(trim, &x0, &up), self.trim_derivative(trim, &x0, &um));
            for i in 0..8 {
                lin.b[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }
        lin
    }

    /// State derivative `[u̇, v̇, ẇ, ṗ, q̇, ṙ, φ̇, θ̇]` in still air with quasi-steady rotors
    /// at the trim's rotor speed (heading zero).
    pub fn trim_derivative(&self, trim: &HelicopterTrim, x: &[f64; 8], u: &[f64; 4]) -> [f64; 8] {
        let v = DVec3::new(x[0], x[1], x[2]);
        let w = DVec3::new(x[3], x[4], x[5]);
        let (phi, theta) = (x[6], x[7]);
        let air = AirData { density: trim.density, ..AirData::default() };
        let flow = AirFlow::new(&air, v, w);
        let pitches = self.pitches_linear(&HelicopterInput::from_array(*u));
        let (f, m, _) = self.steady_wrench(&flow, trim.rotor_speed, &pitches);
        let gravity = attitude(0.0, theta, phi).inverse() * DVec3::new(0.0, 0.0, -trim.gravity);
        let v_dot = f / self.mass() + gravity - w.cross(v);
        let inertia = self.inertia();
        let w_dot = inertia.inverse() * (m - w.cross(inertia * w));
        // Euler rates of nose-up θ and right-wing-down φ from FLU body rates.
        let phi_dot = w.x - (w.y * phi.sin() + w.z * phi.cos()) * theta.tan();
        let theta_dot = -w.y * phi.cos() + w.z * phi.sin();
        [v_dot.x, v_dot.y, v_dot.z, w_dot.x, w_dot.y, w_dot.z, phi_dot, theta_dot]
    }
}

/// Newton's method on `x` with a finite-difference Jacobian and a damped step (at most 0.5 in
/// any unknown); returns the final residual norm (infinite when it breaks down).
fn newton(x: &mut [f64], residual: impl Fn(&[f64]) -> Vec<f64>) -> f64 {
    let n = x.len();
    let mut r = residual(x);
    for _ in 0..MAX_ITERATIONS {
        if norm(&r) < TOLERANCE {
            break;
        }
        let mut jac = vec![vec![0.0; n]; n];
        for j in 0..n {
            let mut xp = x.to_vec();
            xp[j] += STEP;
            let rp = residual(&xp);
            for i in 0..n {
                jac[i][j] = (rp[i] - r[i]) / STEP;
            }
        }
        let Some(dx) = solve(jac, r.clone()) else { return f64::INFINITY };
        let big = dx.iter().fold(0.0f64, |a, d| a.max(d.abs()));
        let k = if big > 0.5 { 0.5 / big } else { 1.0 };
        for (xi, d) in x.iter_mut().zip(dx) {
            *xi -= k * d;
        }
        r = residual(x);
    }
    let e = norm(&r);
    if e.is_nan() { f64::INFINITY } else { e }
}

fn norm(r: &[f64]) -> f64 {
    r.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Solve `A·x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-14 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for c in (0..n).rev() {
        let s: f64 = (c + 1..n).map(|k| a[c][k] * x[k]).sum();
        x[c] = (b[c] - s) / a[c][c];
    }
    Some(x)
}
