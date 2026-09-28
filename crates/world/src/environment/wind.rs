//! Wind: logarithmic mean-wind profile, Dryden turbulence (MIL-F-8785C low-altitude model below
//! 1000 ft, the medium/high-altitude model above 2000 ft, interpolated between) with optional
//! rotational gusts, and discrete 1−cos gusts.
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
        DrydenScales::at(agl, self.turbulence_w20)
    }
}

/// Dryden length scales (m) and intensities (m/s) of the `(u, v, w)` components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrydenScales {
    pub length: DVec3,
    pub sigma: DVec3,
}

/// Altitudes (ft) of the medium/high-altitude intensity table.
const HIGH_ALTITUDES: [f64; 12] =
    [500.0, 1750.0, 3750.0, 7500.0, 15000.0, 25000.0, 35000.0, 45000.0, 55000.0, 65000.0, 75000.0, 80000.0];

/// Medium/high-altitude turbulence intensity σ (ft/s) exceeded with probability 10⁻², 10⁻³ and
/// 10⁻⁵ (MIL-HDBK-1797 / MIL-F-8785C figure, as tabulated in the MATLAB Aerospace Blockset):
/// the light, moderate and severe levels.
const HIGH_SIGMA: [[f64; 12]; 3] = [
    [4.2, 3.6, 3.3, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [6.6, 6.9, 7.4, 6.7, 4.6, 2.7, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    [11.8, 13.0, 16.0, 15.1, 11.6, 9.7, 8.1, 8.2, 7.9, 4.9, 3.2, 2.1],
];

/// Wind speeds at 20 ft (m/s) of the light, moderate and severe levels.
const W20_LEVELS: [f64; 3] = [15.0 * KNOT, 30.0 * KNOT, 45.0 * KNOT];
const KNOT: f64 = 1852.0 / 3600.0;

/// Medium/high-altitude length scale, all components (ft).
const HIGH_LENGTH_FT: f64 = 1750.0;

impl DrydenScales {
    /// MIL-F-8785C scales at height `agl`: the low-altitude model up to 1000 ft, the
    /// medium/high-altitude model from 2000 ft (the height above ground stands for the
    /// altitude), linearly interpolated between.
    pub fn at(agl: f64, w20: f64) -> Self {
        let h = agl / FT;
        if h <= 1000.0 {
            Self::low_altitude(agl, w20)
        } else if h >= 2000.0 {
            Self::high_altitude(agl, w20)
        } else {
            let (a, b) = (Self::low_altitude(1000.0 * FT, w20), Self::high_altitude(2000.0 * FT, w20));
            let s = (h - 1000.0) / 1000.0;
            Self { length: a.length.lerp(b.length, s), sigma: a.sigma.lerp(b.sigma, s) }
        }
    }

    /// Medium/high-altitude scales: isotropic, length 1750 ft, intensity from the table at
    /// the light/moderate/severe level of `w20`, interpolated linearly in `w20` between the
    /// levels (and proportionally below light and above severe).
    pub fn high_altitude(agl: f64, w20: f64) -> Self {
        let h = (agl / FT).clamp(HIGH_ALTITUDES[0], HIGH_ALTITUDES[11]);
        let i = HIGH_ALTITUDES.partition_point(|&a| a <= h).clamp(1, 11);
        let s = (h - HIGH_ALTITUDES[i - 1]) / (HIGH_ALTITUDES[i] - HIGH_ALTITUDES[i - 1]);
        let level = |k: usize| HIGH_SIGMA[k][i - 1] + s * (HIGH_SIGMA[k][i] - HIGH_SIGMA[k][i - 1]);
        let sigma_ft = if w20 <= W20_LEVELS[0] {
            level(0) * w20 / W20_LEVELS[0]
        } else if w20 >= W20_LEVELS[2] {
            level(2) * w20 / W20_LEVELS[2]
        } else {
            let k = if w20 < W20_LEVELS[1] { 0 } else { 1 };
            let t = (w20 - W20_LEVELS[k]) / (W20_LEVELS[k + 1] - W20_LEVELS[k]);
            level(k) + t * (level(k + 1) - level(k))
        };
        Self { length: DVec3::splat(HIGH_LENGTH_FT * FT), sigma: DVec3::splat(sigma_ft * FT) }
    }

    /// MIL-F-8785C low-altitude scales. The height is clamped to 10–1000 ft; the model is
    /// isotropic at 1000 ft.
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
    rotational: Option<Rotational>,
}

/// Rotational gust state: the normalised roll gust; the normalised `v`, `w` at the last step
/// and their high-passed parts `x − lag(x)` (∝ `r_g`, `q_g`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Rotational {
    p: f64,
    v: f64,
    w: f64,
    hv: f64,
    hw: f64,
}

impl Dryden {
    /// State drawn from the stationary distribution (turbulence is fully developed at `t = 0`).
    pub fn stationary(rng: &mut SimRng) -> Self {
        Self {
            u: rng.normal(),
            v: [0.5 * rng.normal(), 0.5 * rng.normal()],
            w: [0.5 * rng.normal(), 0.5 * rng.normal()],
            rotational: None,
        }
    }

    /// Also produce the rotational gusts `p_g`, `q_g`, `r_g` (see [`rates`](Self::rates)); the
    /// roll gust starts from its stationary distribution, the pitch and yaw gusts from zero.
    pub fn with_rotational(mut self, rng: &mut SimRng) -> Self {
        let (v, w) = self.normalised();
        self.rotational = Some(Rotational { p: rng.normal(), v, w, hv: 0.0, hw: 0.0 });
        self
    }

    /// Normalised (unit-variance) transverse outputs.
    fn normalised(&self) -> (f64, f64) {
        (self.v[0] + SQRT_3 * self.v[1], self.w[0] + SQRT_3 * self.w[1])
    }

    /// Advance the filters by `dt` at the given airspeed.
    pub fn step(&mut self, dt: f64, airspeed: f64, scales: &DrydenScales, rng: &mut SimRng) {
        let v = airspeed.max(MIN_TURBULENCE_AIRSPEED);
        let h = dt * v / scales.length.x;
        self.u = (-h).exp() * self.u + (-(-2.0 * h).exp_m1()).sqrt() * rng.normal();
        step_second_order(&mut self.v, dt * v / scales.length.y, rng);
        step_second_order(&mut self.w, dt * v / scales.length.z, rng);
    }

    /// Advance the rotational gust filters of an aircraft of wing span `span` (after
    /// [`step`](Self::step)); nothing without [`with_rotational`](Self::with_rotational).
    pub fn step_rotational(&mut self, dt: f64, airspeed: f64, span: f64, rng: &mut SimRng) {
        let (vn, wn) = self.normalised();
        let Some(r) = &mut self.rotational else { return };
        let v = airspeed.max(MIN_TURBULENCE_AIRSPEED);
        let pi = std::f64::consts::PI;
        // p_g and the q_g lag: time constant 4b/(πV); the r_g lag: 3b/(πV). The high-passed
        // parts `e = x − lag(x)` obey `ė = ẋ − e/τ`; with `x` linear over the step
        // (first-order hold) `e ← e·e^{−H} + Δx·(1 − e^{−H})/H`.
        let h = dt * pi * v / (4.0 * span);
        r.p = (-h).exp() * r.p + (-(-2.0 * h).exp_m1()).sqrt() * rng.normal();
        let high_pass = |e: f64, dx: f64, h: f64| e * (-h).exp() - dx * (-h).exp_m1() / h;
        r.hw = high_pass(r.hw, wn - r.w, h);
        r.hv = high_pass(r.hv, vn - r.v, dt * pi * v / (3.0 * span));
        (r.v, r.w) = (vn, wn);
    }

    /// Rotational gusts (body frame FLU, rad/s; zero without rotational gusts): MIL-F-8785C
    /// `p_g` (first-order, variance `0.1·π²·σ_w²·(π/4b)^⅓ / (b·L_w^⅔)`), `q_g = ∂w_g/∂x` and
    /// `r_g = −∂v_g/∂x` through their lags, with the turbulence axes standing for the body axes.
    pub fn rates(&self, scales: &DrydenScales, span: f64) -> DVec3 {
        let Some(r) = &self.rotational else { return DVec3::ZERO };
        let pi = std::f64::consts::PI;
        let s = scales.sigma;
        let sigma_p =
            s.z * (0.1 * pi * pi * (pi / (4.0 * span)).cbrt() / (span * scales.length.z.powf(2.0 / 3.0))).sqrt();
        // FLU: q = −q_FRD = ∂w_up/∂x, r = −r_FRD = ∂v_right/∂x = −∂v_left/∂x.
        DVec3::new(sigma_p * r.p, s.z * r.hw * pi / (4.0 * span), -s.y * r.hv * pi / (3.0 * span))
    }

    /// Turbulence velocity in ENU with the `u` component along the horizontal unit vector `axis`
    /// and `v` to its left.
    pub fn velocity(&self, scales: &DrydenScales, axis: DVec2) -> DVec3 {
        let s = scales.sigma;
        let u = s.x * self.u;
        let (vn, wn) = self.normalised();
        let (v, w) = (s.y * vn, s.z * wn);
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
    fn altitude_scales() {
        let kt = KNOT / FT;
        // Table values at the light, moderate and severe levels, isotropic, L = 1750 ft.
        let s = DrydenScales::at(7500.0 * FT, 30.0 * KNOT);
        assert!(
            (s.sigma / FT - DVec3::splat(6.7)).length() < 1e-9
                && (s.length / FT - DVec3::splat(1750.0)).length() < 1e-9
        );
        assert!((DrydenScales::high_altitude(1750.0 * FT, 15.0 * KNOT).sigma.x / FT - 3.6).abs() < 1e-9);
        assert!((DrydenScales::at(55000.0 * FT, 45.0 * KNOT).sigma.z / FT - 7.9).abs() < 1e-9);
        // Between altitudes and levels: linear; light turbulence dies out by 15000 ft.
        let s = DrydenScales::at(5625.0 * FT, 37.5 * KNOT).sigma.x / FT;
        assert!((s - 0.5 * (0.5 * (7.4 + 6.7) + 0.5 * (16.0 + 15.1))).abs() < 1e-9, "{s} ({kt})");
        assert_eq!(DrydenScales::at(20000.0 * FT, 15.0 * KNOT).sigma, DVec3::ZERO);
        assert_eq!(DrydenScales::at(3000.0, 0.0).sigma, DVec3::ZERO);
        // Low altitude unchanged below 1000 ft, continuous into the transition and out of it.
        for w20 in [7.7, 15.4, 23.2] {
            assert_eq!(DrydenScales::at(100.0, w20), DrydenScales::low_altitude(100.0, w20));
            for h in [1000.0, 2000.0] {
                let (a, b) = (DrydenScales::at(h * FT - 1e-9, w20), DrydenScales::at(h * FT + 1e-9, w20));
                assert!((a.sigma - b.sigma).length() < 1e-9 && (a.length - b.length).length() < 1e-6, "{h} ft");
            }
        }
        let mid = DrydenScales::at(1500.0 * FT, 15.4);
        assert!((mid.length.x / FT - 0.5 * (1000.0 + 1750.0)).abs() < 1e-9);
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

    /// Samples of the three channels at the given scales and airspeed, and of the rotational
    /// gusts for span `span`.
    fn simulate(dt: f64, n: usize, scales: &DrydenScales, airspeed: f64, span: f64) -> [Vec<f64>; 6] {
        let mut rng = Seed::from_u64(3).child("dryden").rng();
        let mut d = Dryden::stationary(&mut rng).with_rotational(&mut rng);
        let mut out: [Vec<f64>; 6] = Default::default();
        for _ in 0..n {
            d.step(dt, airspeed, scales, &mut rng);
            d.step_rotational(dt, airspeed, span, &mut rng);
            let v = d.velocity(scales, DVec2::X);
            let r = d.rates(scales, span);
            for (o, x) in out.iter_mut().zip([v.x, v.y, v.z, r.x, r.y, r.z]) {
                o.push(x);
            }
        }
        out
    }

    /// Unit-intensity samples at 20 m, 10 m/s airspeed.
    fn simulate_low(dt: f64, n: usize) -> (DrydenScales, [Vec<f64>; 6]) {
        let scales = DrydenScales { sigma: DVec3::ONE, ..DrydenScales::low_altitude(20.0, 1.0) };
        let x = simulate(dt, n, &scales, 10.0, 10.0);
        (scales, x)
    }

    /// Welch estimate (Hann window, `n_seg`-sample segments) of the one-sided PSD in Hz around
    /// `f_target`, averaged over 7 bins, with the model PSD `want(f)` averaged over the same bins.
    fn welch(x: &[f64], dt: f64, n_seg: usize, f_target: f64, want: impl Fn(f64) -> f64) -> (f64, f64) {
        let tau = std::f64::consts::TAU;
        let win: Vec<f64> = (0..n_seg).map(|i| 0.5 * (1.0 - (tau * i as f64 / n_seg as f64).cos())).collect();
        let w2: f64 = win.iter().map(|w| w * w).sum();
        let k0 = (f_target * n_seg as f64 * dt).round() as usize;
        let (mut est, mut model) = (0.0, 0.0);
        for k in k0 - 3..=k0 + 3 {
            let f = k as f64 / (n_seg as f64 * dt);
            let (cs, sn): (Vec<f64>, Vec<f64>) = (0..n_seg)
                .map(|i| (tau * (k * i % n_seg) as f64 / n_seg as f64).sin_cos())
                .map(|(s, c)| (c, s))
                .unzip();
            for seg in x.chunks_exact(n_seg) {
                let (mut re, mut im) = (0.0, 0.0);
                for i in 0..n_seg {
                    let y = seg[i] * win[i];
                    re += y * cs[i];
                    im -= y * sn[i];
                }
                est += 2.0 * dt * (re * re + im * im) / w2;
            }
            model += want(f);
        }
        (est / ((x.len() / n_seg) as f64 * 7.0), model / 7.0)
    }

    /// One-sided Dryden PSDs in Hz: longitudinal `4σ²T/(1 + a)` and transverse
    /// `2σ²T(1 + 3a)/(1 + a)²` with `a = (2πfT)²`, `T = L/V`.
    fn dryden_long(sigma: f64, t: f64, f: f64) -> f64 {
        let a = (std::f64::consts::TAU * f * t).powi(2);
        4.0 * sigma * sigma * t / (1.0 + a)
    }
    fn dryden_trans(sigma: f64, t: f64, f: f64) -> f64 {
        let a = (std::f64::consts::TAU * f * t).powi(2);
        2.0 * sigma * sigma * t * (1.0 + 3.0 * a) / (1.0 + a).powi(2)
    }

    #[test]
    fn dryden_autocorrelation() {
        let dt = 0.1;
        let (scales, x) = simulate_low(dt, 2_000_000);
        for (c, xs) in x[..3].iter().enumerate() {
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
        let (dt, n_seg) = (0.1, 2048);
        let (scales, x) = simulate_low(dt, n_seg * 400);
        for c in [1, 2] {
            let t = scales.length[c] / 10.0;
            for f in [0.02, 0.08, 0.3, 1.0] {
                let (est, want) = welch(&x[c], dt, n_seg, f, |f| dryden_trans(1.0, t, f));
                assert!((est / want - 1.0).abs() < 0.12, "channel {c}, f = {f}: {est:.4e} vs {want:.4e}");
            }
        }
    }

    #[test]
    fn dryden_psd_at_altitude() {
        // Moderate turbulence at 3000 m, 60 m/s: L = 1750 ft, σ ≈ 6.8 ft/s on all axes; an
        // aircraft of 10 m span sees the MIL-F-8785C rotational spectra
        // Φ_p = σ_w²·0.8·(πL/4b)^⅓ / (L·V·(1 + (4bω/πV)²)), Φ_q = (ω/V)²·Φ_w / (1 + (4bω/πV)²),
        // Φ_r = (ω/V)²·Φ_v / (1 + (3bω/πV)²).
        let (dt, n_seg, v, b) = (0.02, 16384, 60.0, 10.0);
        let scales = DrydenScales::at(3000.0, 30.0 * KNOT);
        let x = simulate(dt, n_seg * 300, &scales, v, b);
        let (l, sigma) = (scales.length.x, scales.sigma.x);
        let t = l / v;
        let pi = std::f64::consts::PI;
        let var: f64 = x[2].iter().map(|w| w * w).sum::<f64>() / x[2].len() as f64;
        assert!((var.sqrt() / sigma - 1.0).abs() < 0.05, "σ_w {} vs {sigma}", var.sqrt());
        let tp = 4.0 * b / (pi * v);
        let tr = 3.0 * b / (pi * v);
        let hz = |f: f64| std::f64::consts::TAU * f;
        type Psd<'a> = Box<dyn Fn(f64) -> f64 + 'a>;
        let cases: [(usize, Psd, &[f64]); 5] = [
            (0, Box::new(|f| dryden_long(sigma, t, f)), &[0.05, 0.1, 0.5]),
            (2, Box::new(|f| dryden_trans(sigma, t, f)), &[0.05, 0.1, 0.5]),
            (
                3,
                Box::new(|f| {
                    let k = 0.8 * (pi * l / (4.0 * b)).cbrt() / (l * v);
                    2.0 * pi * sigma * sigma * k / (1.0 + (hz(f) * tp).powi(2))
                }),
                &[0.1, 0.5, 2.0],
            ),
            (
                4,
                Box::new(|f| (hz(f) / v).powi(2) / (1.0 + (hz(f) * tp).powi(2)) * dryden_trans(sigma, t, f)),
                &[0.1, 0.5, 2.0],
            ),
            (
                5,
                Box::new(|f| (hz(f) / v).powi(2) / (1.0 + (hz(f) * tr).powi(2)) * dryden_trans(sigma, t, f)),
                &[0.1, 0.5, 2.0],
            ),
        ];
        for (c, psd, freqs) in cases {
            for &f in freqs {
                let (est, want) = welch(&x[c], dt, n_seg, f, &psd);
                assert!((est / want - 1.0).abs() < 0.12, "channel {c}, f = {f}: {est:.4e} vs {want:.4e}");
            }
        }
        // Without rotational gusts the rates are zero.
        let d = Dryden::stationary(&mut Seed::from_u64(1).rng());
        assert_eq!(d.rates(&scales, b), DVec3::ZERO);
    }
}
