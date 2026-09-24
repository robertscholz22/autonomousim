//! Noise primitives: white noise from a spectral density, first-order Gauss–Markov processes
//! (random walk as the limit τ → ∞), quantisation, and the Allan deviation to verify them.

use autonomousim_core::rng::SimRng;
use glam::DVec3;

/// Standard deviation of discrete white noise with density `density` (units/√Hz) sampled
/// every `dt` seconds.
#[inline]
pub fn white_sigma(density: f64, dt: f64) -> f64 {
    density / dt.sqrt()
}

/// Round to multiples of `resolution` (no-op for `resolution ≤ 0`).
#[inline]
pub fn quantize(x: f64, resolution: f64) -> f64 {
    if resolution > 0.0 { (x / resolution).round() * resolution } else { x }
}

/// Component-wise [`quantize`].
#[inline]
pub fn quantize3(v: DVec3, resolution: f64) -> DVec3 {
    if resolution > 0.0 { (v / resolution).round() * resolution } else { v }
}

/// First-order Gauss–Markov process `ẋ = −x/τ + q·w` (w: unit white noise), discretised
/// exactly for a fixed step: `x ← φ·x + σ_step·n` with `φ = e^{−dt/τ}`. Without a correlation
/// time it is the random walk `x ← x + q·√dt·n`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussMarkov {
    phi: f64,
    step_sigma: f64,
    stationary_sigma: f64,
}

impl GaussMarkov {
    /// From the driving-noise density `q` (units/s/√Hz) and correlation time `tau` (s; `None`:
    /// random walk).
    pub fn from_density(q: f64, tau: Option<f64>, dt: f64) -> Self {
        match tau {
            Some(tau) if tau.is_finite() => {
                let phi = (-dt / tau).exp();
                let stationary = q * (0.5 * tau).sqrt();
                Self { phi, step_sigma: stationary * (1.0 - phi * phi).sqrt(), stationary_sigma: stationary }
            }
            _ => Self { phi: 1.0, step_sigma: q * dt.sqrt(), stationary_sigma: f64::INFINITY },
        }
    }

    /// From the stationary standard deviation and correlation time (s).
    pub fn from_sigma(sigma: f64, tau: f64, dt: f64) -> Self {
        Self::from_density(sigma * (2.0 / tau).sqrt(), Some(tau), dt)
    }

    /// Standard deviation in steady state (infinite for a random walk).
    pub fn stationary_sigma(&self) -> f64 {
        self.stationary_sigma
    }

    /// Whether the process is identically zero.
    pub fn is_zero(&self) -> bool {
        self.step_sigma == 0.0
    }

    #[inline]
    pub fn step(&self, x: f64, rng: &mut SimRng) -> f64 {
        if self.step_sigma == 0.0 { self.phi * x } else { self.phi * x + self.step_sigma * rng.normal() }
    }

    #[inline]
    pub fn step3(&self, x: DVec3, rng: &mut SimRng) -> DVec3 {
        if self.step_sigma == 0.0 { self.phi * x } else { self.phi * x + self.step_sigma * rng.normal3() }
    }

    /// A draw from the stationary distribution (zero for a random walk).
    pub fn sample_stationary3(&self, rng: &mut SimRng) -> DVec3 {
        if self.stationary_sigma.is_finite() && self.stationary_sigma > 0.0 {
            self.stationary_sigma * rng.normal3()
        } else {
            DVec3::ZERO
        }
    }
}

/// Overlapping Allan deviation of rate samples `y` taken every `dt`, at cluster time
/// `m·dt`. White noise of density N gives `N/√τ`; a random walk of density K gives `K·√(τ/3)`.
pub fn allan_deviation(y: &[f64], dt: f64, m: usize) -> f64 {
    let n = y.len();
    assert!(m >= 1 && 2 * m < n, "cluster size {m} for {n} samples");
    // θ_k = dt·Σ_{i<k} y_i
    let mut theta = Vec::with_capacity(n + 1);
    let mut acc = 0.0;
    theta.push(0.0);
    for &v in y {
        acc += v * dt;
        theta.push(acc);
    }
    let tau = m as f64 * dt;
    let terms = n + 1 - 2 * m;
    let sum: f64 = (0..terms)
        .map(|k| {
            let d = theta[k + 2 * m] - 2.0 * theta[k + m] + theta[k];
            d * d
        })
        .sum();
    (sum / (2.0 * tau * tau * terms as f64)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::rng::Seed;

    #[test]
    fn gauss_markov_is_stationary_and_correlated() {
        let dt = 0.01;
        let gm = GaussMarkov::from_sigma(2.0, 0.5, dt);
        assert!((gm.stationary_sigma() - 2.0).abs() < 1e-12);
        let mut rng = Seed::from_u64(1).rng();
        let (mut x, mut sum2, mut lag) = (gm.sample_stationary3(&mut rng).x, 0.0, 0.0);
        let n = 400_000;
        for _ in 0..n {
            let next = gm.step(x, &mut rng);
            sum2 += x * x;
            lag += x * next;
            x = next;
        }
        let var = sum2 / n as f64;
        assert!((var.sqrt() - 2.0).abs() < 0.05, "{}", var.sqrt());
        assert!((lag / sum2 - (-dt / 0.5f64).exp()).abs() < 2e-3);
        // Random walk variance grows as q²·t.
        let rw = GaussMarkov::from_density(0.3, None, dt);
        let mut end = 0.0;
        for _ in 0..2000 {
            let mut x = 0.0;
            for _ in 0..100 {
                x = rw.step(x, &mut rng);
            }
            end += x * x;
        }
        assert!(((end / 2000.0).sqrt() - 0.3).abs() < 0.02);
    }

    #[test]
    fn quantisation() {
        assert_eq!(quantize(0.26, 0.1), 0.30000000000000004);
        assert_eq!(quantize(0.26, 0.0), 0.26);
        assert_eq!(quantize3(DVec3::new(1.4, -1.6, 0.0), 1.0), DVec3::new(1.0, -2.0, 0.0));
    }
}
