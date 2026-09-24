//! Wind: logarithmic mean-wind profile, Dryden turbulence (MIL-F-8785C low-altitude model) and
//! discrete 1−cos gusts.
//!
//! Turbulence is filtered white noise advected past the vehicle at its airspeed (Taylor's
//! frozen-field hypothesis), so every agent carries its own [`Dryden`] state; agents do not see
//! a shared turbulence field. The filters are discretised exactly (matrix exponential and
//! integrated noise covariance), so the statistics do not depend on the step size.

use autonomousim_core::rng::SimRng;
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

const FT: f64 = 0.3048;
const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// Airspeed floor for the turbulence time scales: without it a hovering vehicle in calm air
/// would see a frozen field.
pub const MIN_TURBULENCE_AIRSPEED: f64 = 2.0;

/// Discrete gust with a 1−cos velocity profile peaking at `amplitude` half-way through.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gust {
    /// Start time (s).
    pub start: f64,
    /// Duration (s).
    pub duration: f64,
    /// Peak wind velocity added (ENU, m/s).
    pub amplitude: DVec3,
}

impl Gust {
    pub fn velocity(&self, t: f64) -> DVec3 {
        let s = (t - self.start) / self.duration;
        if !(0.0..=1.0).contains(&s) {
            return DVec3::ZERO;
        }
        self.amplitude * (0.5 * (1.0 - (std::f64::consts::TAU * s).cos()))
    }
}

/// Wind configuration of an episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindConfig {
    /// Mean horizontal wind velocity (ENU, direction the air moves) at `reference_height`
    /// above ground (m/s).
    pub mean: DVec2,
    /// Height of the mean-wind specification above ground (m); 20 ft by default.
    pub reference_height: f64,
    /// Aerodynamic roughness length `z0` of the surface (m): about 0.03 for open grass,
    /// 0.1–0.25 for crops, 0.5–1 for forest and suburbs.
    pub roughness: f64,
    /// Wind speed at 20 ft that sets the turbulence intensity (`σ_w = 0.1·W20`, m/s):
    /// MIL-F-8785C light 7.7, moderate 15.4, severe 23.2. Zero disables turbulence.
    pub turbulence_w20: f64,
    pub gusts: Vec<Gust>,
}

impl Default for WindConfig {
    fn default() -> Self {
        Self { mean: DVec2::ZERO, reference_height: 20.0 * FT, roughness: 0.03, turbulence_w20: 0.0, gusts: Vec::new() }
    }
}

impl WindConfig {
    /// No wind at all.
    pub fn calm() -> Self {
        Self::default()
    }

    /// Mean wind at height `agl` above ground (log profile; heights below `2·z0` use `2·z0`).
    pub fn mean_at(&self, agl: f64) -> DVec3 {
        let z0 = self.roughness;
        let h = agl.max(2.0 * z0);
        (self.mean * ((h / z0).ln() / (self.reference_height / z0).ln())).extend(0.0)
    }

    /// Sum of the active gusts at time `t`.
    pub fn gust_at(&self, t: f64) -> DVec3 {
        self.gusts.iter().map(|g| g.velocity(t)).sum()
    }

    /// Deterministic wind (mean profile + gusts).
    pub fn steady_at(&self, agl: f64, t: f64) -> DVec3 {
        self.mean_at(agl) + self.gust_at(t)
    }

    pub fn has_turbulence(&self) -> bool {
        self.turbulence_w20 > 0.0
    }

    /// Horizontal unit vector of the turbulence `u` axis: along the mean wind, or east in calm air.
    pub fn turbulence_axis(&self) -> DVec2 {
        self.mean.try_normalize().unwrap_or(DVec2::X)
    }

    /// Turbulence scales at height `agl`.
    pub fn turbulence_scales(&self, agl: f64) -> DrydenScales {
        DrydenScales::low_altitude(agl, self.turbulence_w20)
    }
}

/// Dryden length scales (m) and intensities (m/s) of the `(u, v, w)` components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrydenScales {
    pub length: DVec3,
    pub sigma: DVec3,
}

impl DrydenScales {
    /// MIL-F-8785C low-altitude scales. The height is clamped to 10–1000 ft; the model is
    /// isotropic at 1000 ft and the scales are held there above (drones stay low).
    pub fn low_altitude(agl: f64, w20: f64) -> Self {
        let h = (agl / FT).clamp(10.0, 1000.0);
        let k = 0.177 + 0.000_823 * h;
        let l_uv = h / k.powf(1.2) * FT;
        let sigma_w = 0.1 * w20;
        let sigma_uv = sigma_w / k.powf(0.4);
        Self { length: DVec3::new(l_uv, l_uv, h * FT), sigma: DVec3::new(sigma_uv, sigma_uv, sigma_w) }
    }
}

/// Dryden turbulence filter state of one vehicle.
///
/// The states are normalised so that their stationary distribution does not depend on the
/// scales: `u ~ N(0, 1)` and the second-order `v`, `w` filters have stationary covariance
/// `I/4` with output `z₀ + √3·z₁` of unit variance. Changing altitude or airspeed therefore
/// never produces transients in the turbulence intensity.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dryden {
    u: f64,
    v: [f64; 2],
    w: [f64; 2],
}

impl Dryden {
    /// State drawn from the stationary distribution (turbulence is fully developed at `t = 0`).
    pub fn stationary(rng: &mut SimRng) -> Self {
        Self {
            u: rng.normal(),
            v: [0.5 * rng.normal(), 0.5 * rng.normal()],
            w: [0.5 * rng.normal(), 0.5 * rng.normal()],
        }
    }

    /// Advance the filters by `dt` at the given airspeed.
    pub fn step(&mut self, dt: f64, airspeed: f64, scales: &DrydenScales, rng: &mut SimRng) {
        let v = airspeed.max(MIN_TURBULENCE_AIRSPEED);
        let h = dt * v / scales.length.x;
        self.u = (-h).exp() * self.u + (-(-2.0 * h).exp_m1()).sqrt() * rng.normal();
        step_second_order(&mut self.v, dt * v / scales.length.y, rng);
        step_second_order(&mut self.w, dt * v / scales.length.z, rng);
    }

    /// Turbulence velocity in ENU with the `u` component along the horizontal unit vector `axis`
    /// and `v` to its left.
    pub fn velocity(&self, scales: &DrydenScales, axis: DVec2) -> DVec3 {
        let s = scales.sigma;
        let u = s.x * self.u;
        let v = s.y * (self.v[0] + SQRT_3 * self.v[1]);
        let w = s.z * (self.w[0] + SQRT_3 * self.w[1]);
        (axis * u + axis.perp() * v).extend(w)
    }
}

/// Exact step of the normalised transverse Dryden filter `(1 + √3·Ts)/(1 + Ts)²` over
/// `H = dt/T` time constants.
fn step_second_order(z: &mut [f64; 2], h: f64, rng: &mut SimRng) {
    let e = (-h).exp();
    let z0 = e * ((1.0 + h) * z[0] + h * z[1]);
    let z1 = e * (-h * z[0] + (1.0 - h) * z[1]);
    // Integrated noise covariance and its Cholesky factor.
    let [i0, i1, i2] = exp_moments(h);
    let (q11, q12, q22) = (i2, i1 - i2, i0 - 2.0 * i1 + i2);
    let l11 = q11.sqrt();
    let l21 = if l11 > 0.0 { q12 / l11 } else { 0.0 };
    let l22 = (q22 - l21 * l21).max(0.0).sqrt();
    let (n0, n1) = (rng.normal(), rng.normal());
    z[0] = z0 + l11 * n0;
    z[1] = z1 + l21 * n0 + l22 * n1;
}

/// `I_k(H) = ∫₀ᴴ uᵏ e^{−2u} du` for `k = 0, 1, 2`.
fn exp_moments(h: f64) -> [f64; 3] {
    if h < 0.5 {
        // Power series; the closed forms cancel catastrophically for small H.
        let mut out = [0.0; 3];
        let mut term = 1.0; // (−2H)ⁿ / n!
        for n in 0..30 {
            let mut hp = h;
            for (k, o) in out.iter_mut().enumerate() {
                *o += term * hp / (n + k + 1) as f64;
                hp *= h;
            }
            term *= -2.0 * h / (n + 1) as f64;
            if term.abs() < 1e-18 {
                break;
            }
        }
        out
    } else {
        let e = (-2.0 * h).exp();
        [0.5 * (1.0 - e), 0.25 * (1.0 - e * (1.0 + 2.0 * h)), 0.25 * (1.0 - e * (1.0 + 2.0 * h + 2.0 * h * h))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::rng::Seed;

    #[test]
    fn mean_profile_and_gusts() {
        let w = WindConfig { mean: DVec2::new(3.0, 4.0), ..WindConfig::default() };
        assert!((w.mean_at(20.0 * FT) - DVec3::new(3.0, 4.0, 0.0)).length() < 1e-12);
        let hi = w.mean_at(50.0).length();
        let lo = w.mean_at(1.0).length();
        assert!(hi > 5.0 && lo < 5.0 && lo > 0.0);
        assert_eq!(w.mean_at(0.0), w.mean_at(0.06));
        assert!((w.turbulence_axis() - DVec2::new(0.6, 0.8)).length() < 1e-12);

        let g = Gust { start: 1.0, duration: 2.0, amplitude: DVec3::new(0.0, 0.0, -4.0) };
        assert_eq!(g.velocity(0.9), DVec3::ZERO);
        assert!((g.velocity(2.0).z + 4.0).abs() < 1e-12);
        assert!((g.velocity(1.5).z + 2.0).abs() < 1e-12);
        assert_eq!(g.velocity(3.1), DVec3::ZERO);
    }

    #[test]
    fn mil_spec_scales() {
        let s = DrydenScales::low_altitude(50.0 * FT, 15.0);
        assert!((s.length.z - 50.0 * FT).abs() < 1e-12);
        assert!((s.length.x / FT - 310.8).abs() < 0.1, "{}", s.length.x / FT);
        assert!((s.sigma.z - 1.5).abs() < 1e-12);
        assert!((s.sigma.x / s.sigma.z - 1.8387).abs() < 1e-3);
        let s = DrydenScales::low_altitude(2000.0, 15.0);
        assert!((s.length - DVec3::splat(1000.0 * FT)).length() < 1e-9);
        assert!((s.sigma - DVec3::splat(1.5)).length() < 1e-12);
        assert_eq!(DrydenScales::low_altitude(0.0, 15.0), DrydenScales::low_altitude(10.0 * FT, 15.0));
    }

    #[test]
    fn exp_moments_are_continuous() {
        let below = exp_moments(0.5 - 1e-12);
        let e = (-1.0f64).exp();
        let exact = [0.5 * (1.0 - e), 0.25 * (1.0 - 2.0 * e), 0.25 * (1.0 - 2.5 * e)];
        for k in 0..3 {
            assert!((below[k] - exact[k]).abs() < 1e-12);
        }
        // Leading-order behaviour for tiny steps.
        let m = exp_moments(1e-6);
        assert!((m[0] / 1e-6 - 1.0).abs() < 1e-5 && (m[2] / (1e-18 / 3.0) - 1.0).abs() < 1e-5);
    }

    /// Unit-intensity samples of the three channels at 20 m, 10 m/s airspeed.
    fn simulate(dt: f64, n: usize) -> (DrydenScales, [Vec<f64>; 3]) {
        let scales = DrydenScales { sigma: DVec3::ONE, ..DrydenScales::low_altitude(20.0, 1.0) };
        let mut rng = Seed::from_u64(3).child("dryden").rng();
        let mut d = Dryden::stationary(&mut rng);
        let mut out = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
        for _ in 0..n {
            d.step(dt, 10.0, &scales, &mut rng);
            let v = d.velocity(&scales, DVec2::X);
            out[0].push(v.x);
            out[1].push(v.y);
            out[2].push(v.z);
        }
        (scales, out)
    }

    #[test]
    fn dryden_autocorrelation() {
        let dt = 0.1;
        let (scales, x) = simulate(dt, 2_000_000);
        for (c, xs) in x.iter().enumerate() {
            let t = scales.length[c] / 10.0;
            for tau_t in [0.0, 0.5, 1.0, 2.0] {
                let lag = (tau_t * t / dt).round() as usize;
                let s = lag as f64 * dt / t;
                let r: f64 = xs.iter().zip(&xs[lag..]).map(|(a, b)| a * b).sum::<f64>() / (xs.len() - lag) as f64;
                let want = if c == 0 { (-s).exp() } else { (-s).exp() * (1.0 - 0.5 * s) };
                // Standard error ≈ √(2T / T_total): 0.011 for u (T = 11.6 s), 0.0045 for w.
                assert!((r - want).abs() < 0.05, "channel {c}, τ/T = {s:.2}: R = {r:.4}, want {want:.4}");
            }
        }
    }

    #[test]
    fn dryden_psd() {
        // Welch estimate (Hann, 204.8 s segments) of the one-sided PSD in Hz against
        // G(f) = 2σ²T(1 + 3(2πfT)²)/(1 + (2πfT)²)² for the transverse channels.
        let (dt, n_seg) = (0.1, 2048);
        let (scales, x) = simulate(dt, n_seg * 400);
        let win: Vec<f64> =
            (0..n_seg).map(|i| 0.5 * (1.0 - (std::f64::consts::TAU * i as f64 / n_seg as f64).cos())).collect();
        let w2: f64 = win.iter().map(|w| w * w).sum();
        for c in [1, 2] {
            let t = scales.length[c] / 10.0;
            for f_target in [0.02, 0.08, 0.3, 1.0] {
                let k0 = (f_target * n_seg as f64 * dt).round() as usize;
                let (mut est, mut want) = (0.0, 0.0);
                for k in k0 - 3..=k0 + 3 {
                    let f = k as f64 / (n_seg as f64 * dt);
                    let (cs, sn): (Vec<f64>, Vec<f64>) = (0..n_seg)
                        .map(|i| (std::f64::consts::TAU * (k * i % n_seg) as f64 / n_seg as f64).sin_cos())
                        .map(|(s, c)| (c, s))
                        .unzip();
                    for seg in x[c].chunks_exact(n_seg) {
                        let (mut re, mut im) = (0.0, 0.0);
                        for i in 0..n_seg {
                            let y = seg[i] * win[i];
                            re += y * cs[i];
                            im -= y * sn[i];
                        }
                        est += 2.0 * dt * (re * re + im * im) / w2;
                    }
                    let a = (std::f64::consts::TAU * f * t).powi(2);
                    want += 2.0 * t * (1.0 + 3.0 * a) / (1.0 + a).powi(2);
                }
                est /= (x[c].len() / n_seg) as f64 * 7.0;
                want /= 7.0;
                assert!((est / want - 1.0).abs() < 0.12, "channel {c}, f = {f_target}: {est:.4e} vs {want:.4e}");
            }
        }
    }
}
