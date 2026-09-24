//! Deterministic, hierarchical random-number streams.
//!
//! Every stochastic component draws from its own stream derived from a [`Seed`] by label,
//! e.g. `root.child("map").child("wild").child_index(tile)`. Streams are independent of how
//! many environments or threads exist. Sampling functions are implemented here (not taken
//! from `rand`/`rand_distr`) so that a dependency upgrade can never change sequences.

use glam::{DQuat, DVec3};
use rand_chacha::ChaCha8Rng;
use rand_core::{Rng as _, SeedableRng};
use serde::{Deserialize, Serialize};

/// 256-bit seed node in the seed tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Seed(pub [u8; 32]);

impl Seed {
    /// Root seed from a user-facing integer seed.
    pub fn from_u64(seed: u64) -> Self {
        let mut h = blake3::Hasher::new_derive_key("autonomousim seed v1");
        h.update(&seed.to_le_bytes());
        Seed(*h.finalize().as_bytes())
    }

    /// Child seed `blake3_keyed(self, label)`.
    pub fn child(&self, label: &str) -> Seed {
        let mut h = blake3::Hasher::new_keyed(&self.0);
        h.update(b"label:");
        h.update(label.as_bytes());
        Seed(*h.finalize().as_bytes())
    }

    /// Child seed for an integer index (tiles, environments, episodes, agents).
    pub fn child_index(&self, index: u64) -> Seed {
        let mut h = blake3::Hasher::new_keyed(&self.0);
        h.update(b"index:");
        h.update(&index.to_le_bytes());
        Seed(*h.finalize().as_bytes())
    }

    /// A random stream seeded from this node.
    pub fn rng(&self) -> SimRng {
        SimRng(ChaCha8Rng::from_seed(self.0))
    }

    /// First 8 bytes as an integer (for display / hashing).
    pub fn short(&self) -> u64 {
        u64::from_le_bytes(self.0[..8].try_into().unwrap())
    }
}

const ZIG_LAYERS: usize = 128;
const ZIG_R: f64 = 3.442619855899;
const ZIG_V: f64 = 9.91256303526217e-3;

/// Ziggurat layer edges `x` and the inner-rectangle ratios `x[i+1]/x[i]`.
struct Ziggurat {
    x: [f64; ZIG_LAYERS + 1],
    ratio: [f64; ZIG_LAYERS],
}

#[inline]
fn ziggurat() -> &'static Ziggurat {
    static TABLE: std::sync::OnceLock<Ziggurat> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut x = [0.0; ZIG_LAYERS + 1];
        let mut f = libm::exp(-0.5 * ZIG_R * ZIG_R);
        // x[0]: width of the base strip (rectangle plus tail) of area V.
        x[0] = ZIG_V / f;
        x[1] = ZIG_R;
        for i in 2..ZIG_LAYERS {
            x[i] = libm::sqrt(-2.0 * libm::log(ZIG_V / x[i - 1] + f));
            f = libm::exp(-0.5 * x[i] * x[i]);
        }
        let mut ratio = [0.0; ZIG_LAYERS];
        for i in 0..ZIG_LAYERS {
            ratio[i] = x[i + 1] / x[i];
        }
        Ziggurat { x, ratio }
    })
}

/// The simulator's random stream (ChaCha8) with self-implemented sampling functions.
#[derive(Clone, Debug)]
pub struct SimRng(ChaCha8Rng);

impl SimRng {
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    /// Uniform in `[0, 1)` with 53 bits of precision.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniform integer in `[0, n)` (Lemire's nearly-divisionless method, unbiased).
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        let mut m = (self.next_u64() as u128) * (n as u128);
        if (m as u64) < n {
            let t = n.wrapping_neg() % n;
            while (m as u64) < t {
                m = (self.next_u64() as u128) * (n as u128);
            }
        }
        (m >> 64) as u64
    }

    /// Bernoulli trial with probability `p`.
    #[inline]
    pub fn chance(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    /// Standard normal by the ziggurat method (Marsaglia & Tsang 2000, in Doornik's 2005
    /// ZIGNOR form with 128 layers). One 64-bit draw suffices ~99 % of the time; the rare
    /// wedge and tail samples use `libm`, so results are machine-independent.
    pub fn normal(&mut self) -> f64 {
        let z = ziggurat();
        loop {
            let b = self.next_u64();
            // Disjoint bits: 7 for the layer, 53 for a uniform in (−1, 1).
            let i = (b & 0x7f) as usize;
            let u = ((b >> 11) as f64 + 0.5) * (2.0 / (1u64 << 53) as f64) - 1.0;
            if u.abs() < z.ratio[i] {
                return u * z.x[i];
            }
            if i == 0 {
                return self.normal_tail(ZIG_R, u < 0.0);
            }
            let x = u * z.x[i];
            let f0 = libm::exp(-0.5 * (z.x[i] * z.x[i] - x * x));
            let f1 = libm::exp(-0.5 * (z.x[i + 1] * z.x[i + 1] - x * x));
            if f1 + self.uniform() * (f0 - f1) < 1.0 {
                return x;
            }
        }
    }

    /// Normal sample beyond `r` (Marsaglia 1964).
    #[cold]
    fn normal_tail(&mut self, r: f64, negative: bool) -> f64 {
        loop {
            // 1 − u ∈ (0, 1] avoids ln(0).
            let x = libm::log(1.0 - self.uniform()) / r;
            let y = libm::log(1.0 - self.uniform());
            if -2.0 * y >= x * x {
                return if negative { x - r } else { r - x };
            }
        }
    }

    /// Normal with mean `mu` and standard deviation `sigma`.
    #[inline]
    pub fn gaussian(&mut self, mu: f64, sigma: f64) -> f64 {
        mu + sigma * self.normal()
    }

    /// Vector of independent standard normals.
    #[inline]
    pub fn normal3(&mut self) -> DVec3 {
        DVec3::new(self.normal(), self.normal(), self.normal())
    }

    /// Uniformly distributed unit vector.
    pub fn unit_vector(&mut self) -> DVec3 {
        let z = self.range(-1.0, 1.0);
        let phi = self.range(0.0, std::f64::consts::TAU);
        let r = libm::sqrt(1.0 - z * z);
        DVec3::new(r * libm::cos(phi), r * libm::sin(phi), z)
    }

    /// Uniformly distributed rotation.
    pub fn rotation(&mut self) -> DQuat {
        let (a, b, c) = (self.uniform(), self.uniform(), self.uniform());
        crate::math::quat::uniform_rotation(a, b, c)
    }

    /// Log-normal sample with the given median and log-space standard deviation.
    pub fn log_normal(&mut self, median: f64, sigma_ln: f64) -> f64 {
        median * libm::exp(sigma_ln * self.normal())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_deterministic_and_independent() {
        let root = Seed::from_u64(42);
        assert_eq!(root, Seed::from_u64(42));
        assert_ne!(root, Seed::from_u64(43));
        let a = root.child("map");
        let b = root.child("env");
        assert_ne!(a, b);
        assert_ne!(root.child_index(0), root.child_index(1));
        let mut r1 = a.rng();
        let mut r2 = a.rng();
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    /// Golden values: if this fails, every recorded seed changes meaning (bump the map
    /// generator / recording versions deliberately before updating).
    #[test]
    fn golden_sequence() {
        let mut r = Seed::from_u64(0).child("golden").rng();
        let v: Vec<u64> = (0..3).map(|_| r.next_u64()).collect();
        assert_eq!(v, [11976232826472119315, 16308739944226381452, 8629888788220003697]);
        let g: Vec<f64> = (0..3).map(|_| r.normal()).collect();
        assert_eq!(g, [-1.4580511184362677, 0.01958514432056501, -0.6415467078733578]);
    }

    #[test]
    fn uniform_and_normal_moments() {
        let mut r = Seed::from_u64(7).rng();
        let n = 200_000;
        let (mut s, mut s2, mut g, mut g2) = (0.0, 0.0, 0.0, 0.0);
        for _ in 0..n {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
            s += u;
            s2 += u * u;
            let x = r.normal();
            g += x;
            g2 += x * x;
        }
        let nf = n as f64;
        assert!((s / nf - 0.5).abs() < 0.005);
        assert!((s2 / nf - 1.0 / 3.0).abs() < 0.005);
        assert!((g / nf).abs() < 0.01);
        assert!((g2 / nf - 1.0).abs() < 0.01);
    }

    /// The ziggurat's layers, wedges and tail reproduce the normal CDF, including far out.
    #[test]
    fn normal_matches_the_cdf() {
        let mut r = Seed::from_u64(11).rng();
        let n = 4_000_000;
        let edges = [-4.0, -3.442619855899, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 3.442619855899, 4.0];
        let mut below = vec![0u64; edges.len()];
        let (mut m3, mut m4) = (0.0, 0.0);
        for _ in 0..n {
            let x = r.normal();
            for (c, e) in below.iter_mut().zip(edges) {
                *c += u64::from(x < e);
            }
            m3 += x * x * x;
            m4 += x * x * x * x;
        }
        let phi = |x: f64| 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
        for (c, e) in below.iter().zip(edges) {
            let p = phi(e);
            let sigma = (p * (1.0 - p) / n as f64).sqrt();
            let got = *c as f64 / n as f64;
            assert!((got - p).abs() < 4.0 * sigma + 1e-9, "P(x < {e}) = {got}, expected {p}");
        }
        assert!((m3 / n as f64).abs() < 0.01 && (m4 / n as f64 - 3.0).abs() < 0.03);
    }

    #[test]
    fn below_is_in_range_and_unbiased() {
        let mut r = Seed::from_u64(3).rng();
        let mut counts = [0u32; 7];
        for _ in 0..70_000 {
            counts[r.below(7) as usize] += 1;
        }
        for c in counts {
            assert!((c as i64 - 10_000).abs() < 500, "{counts:?}");
        }
    }

    #[test]
    fn unit_vectors_are_unit() {
        let mut r = Seed::from_u64(1).rng();
        let mut mean = DVec3::ZERO;
        for _ in 0..10_000 {
            let v = r.unit_vector();
            assert!((v.length() - 1.0).abs() < 1e-12);
            mean += v;
        }
        assert!((mean / 10_000.0).length() < 0.03);
    }
}
