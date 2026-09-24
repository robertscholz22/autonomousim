//! Seeded 2-D gradient noise on the simplex lattice (the OpenSimplex2 kernel: radius² 0.5,
//! quartic falloff, 24 gradient directions) and fractal sums of it.
//!
//! Only `+`, `−`, `×`, `floor` and integer hashing are used, so values are bit-identical on
//! every IEEE-754 machine and do not depend on a platform `libm`.

use serde::{Deserialize, Serialize};

/// Skew and unskew factors of the 2-D simplex lattice, `(√3 − 1)/2` and `(3 − √3)/6`.
const F2: f64 = 0.366_025_403_784_438_6;
const G2: f64 = 0.211_324_865_405_187_1;

/// Scales the kernel sum to [−1, 1] (1 / OpenSimplex2's `NORMALIZER_2D`).
const NORM: f64 = 99.836_854_463_036_47;

/// Unit gradients at 7.5° + k·15°.
const A: f64 = 0.991_444_861_373_810_4;
const B: f64 = 0.130_526_192_220_051_57;
const C: f64 = 0.923_879_532_511_286_7;
const D: f64 = 0.382_683_432_365_089_8;
const E: f64 = 0.793_353_340_291_235_2;
const F: f64 = 0.608_761_429_008_720_7;
const GRADIENTS: [[f64; 2]; 24] = [
    [A, B],
    [C, D],
    [E, F],
    [F, E],
    [D, C],
    [B, A],
    [-B, A],
    [-D, C],
    [-F, E],
    [-E, F],
    [-C, D],
    [-A, B],
    [-A, -B],
    [-C, -D],
    [-E, -F],
    [-F, -E],
    [-D, -C],
    [-B, -A],
    [B, -A],
    [D, -C],
    [F, -E],
    [E, -F],
    [C, -D],
    [A, -B],
];

#[inline]
fn gradient(seed: u64, i: i64, j: i64) -> [f64; 2] {
    let mut h = seed ^ (i as u64).wrapping_mul(0x5205_402B_9270_C86F) ^ (j as u64).wrapping_mul(0x598C_D327_0038_17B5);
    h = h.wrapping_mul(0x53A3_F72D_EEC5_46F5);
    h ^= h >> 32;
    GRADIENTS[(((h & 0xFFFF_FFFF) * 24) >> 32) as usize]
}

#[inline]
fn corner(seed: u64, i: i64, j: i64, x: f64, y: f64) -> f64 {
    let t = 0.5 - x * x - y * y;
    if t <= 0.0 {
        return 0.0;
    }
    let g = gradient(seed, i, j);
    let t2 = t * t;
    t2 * t2 * (g[0] * x + g[1] * y)
}

/// Gradient noise in [−1, 1] with features about one unit apart.
pub fn simplex2(seed: u64, x: f64, y: f64) -> f64 {
    let s = (x + y) * F2;
    let (i, j) = ((x + s).floor(), (y + s).floor());
    let t = (i + j) * G2;
    let (x0, y0) = (x - (i - t), y - (j - t));
    let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };
    let (x1, y1) = (x0 - i1 as f64 + G2, y0 - j1 as f64 + G2);
    let (x2, y2) = (x0 - 1.0 + 2.0 * G2, y0 - 1.0 + 2.0 * G2);
    let (i, j) = (i as i64, j as i64);
    let n = corner(seed, i, j, x0, y0) + corner(seed, i + i1, j + j1, x1, y1) + corner(seed, i + 1, j + 1, x2, y2);
    (n * NORM).clamp(-1.0, 1.0)
}

/// Parameters of a fractal sum: `octaves` layers, each `lacunarity` times finer and `gain`
/// times weaker than the previous one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Fractal {
    pub octaves: u32,
    pub lacunarity: f64,
    pub gain: f64,
}

impl Default for Fractal {
    fn default() -> Self {
        Self { octaves: 5, lacunarity: 2.0, gain: 0.5 }
    }
}

impl Fractal {
    pub fn new(octaves: u32) -> Self {
        Self { octaves, ..Self::default() }
    }

    pub fn is_valid(&self) -> bool {
        (1..=16).contains(&self.octaves) && self.lacunarity > 1.0 && self.gain > 0.0 && self.gain < 1.0
    }
}

/// Seed of octave `o`; octaves are also rotated against each other (by atan(3/4) each) so that
/// lattice artefacts do not line up.
#[inline]
fn octave(seed: u64, o: u32, x: f64, y: f64) -> (u64, f64, f64) {
    let s = seed.wrapping_add(u64::from(o).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (s, 0.8 * x - 0.6 * y, 0.6 * x + 0.8 * y)
}

/// Fractal Brownian motion in [−1, 1] (normalised by the sum of the amplitudes).
pub fn fbm(seed: u64, x: f64, y: f64, f: &Fractal) -> f64 {
    let (mut sum, mut norm, mut amp) = (0.0, 0.0, 1.0);
    let (mut x, mut y) = (x, y);
    for o in 0..f.octaves {
        let (s, rx, ry) = octave(seed, o, x, y);
        sum += amp * simplex2(s, rx, ry);
        norm += amp;
        amp *= f.gain;
        x = rx * f.lacunarity;
        y = ry * f.lacunarity;
    }
    sum / norm
}

/// Ridged multifractal (Musgrave) in [0, 1]: sharp crests where the noise crosses zero, with
/// finer octaves weighted by the coarser ones so that valleys stay smooth.
pub fn ridged(seed: u64, x: f64, y: f64, f: &Fractal) -> f64 {
    let (mut sum, mut norm, mut amp, mut weight) = (0.0, 0.0, 1.0, 1.0);
    let (mut x, mut y) = (x, y);
    for o in 0..f.octaves {
        let (s, rx, ry) = octave(seed, o, x, y);
        let r = 1.0 - simplex2(s, rx, ry).abs();
        let signal = r * r * weight;
        weight = (2.0 * signal).clamp(0.0, 1.0);
        sum += amp * signal;
        norm += amp;
        amp *= f.gain;
        x = rx * f.lacunarity;
        y = ry * f.lacunarity;
    }
    sum / norm
}

/// Hermite step from 0 at `lo` to 1 at `hi`.
#[inline]
pub fn smoothstep(lo: f64, hi: f64, x: f64) -> f64 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_range_mean_and_continuity() {
        let (mut lo, mut hi, mut sum, mut sq) = (0.0f64, 0.0f64, 0.0, 0.0);
        let n = 400_000;
        for k in 0..n {
            let x = (k % 1000) as f64 * 0.137 - 50.0;
            let y = (k / 1000) as f64 * 0.173 + 20.0;
            let v = simplex2(7, x, y);
            lo = lo.min(v);
            hi = hi.max(v);
            sum += v;
            sq += v * v;
            // Lipschitz: the kernel's gradient is bounded.
            let d = (simplex2(7, x + 1e-4, y) - v).abs();
            assert!(d < 1e-4 * 20.0, "{x} {y}: {d}");
        }
        let mean = sum / n as f64;
        let std = (sq / n as f64 - mean * mean).sqrt();
        assert!(lo > -1.0 && hi < 1.0 && lo < -0.6 && hi > 0.6, "{lo} {hi}");
        // The quartic kernel gives a broad, nearly flat distribution over (−0.95, 0.95).
        assert!(mean.abs() < 0.01 && (0.45..0.6).contains(&std), "{mean} {std}");
        // Seeds give different fields; lattice points are zero.
        assert_ne!(simplex2(1, 0.3, 0.7), simplex2(2, 0.3, 0.7));
        assert_eq!(simplex2(3, 0.0, 0.0), 0.0);
    }

    #[test]
    fn fractals_are_bounded() {
        let f = Fractal::new(6);
        for k in 0..20_000 {
            let (x, y) = ((k % 200) as f64 * 0.31, (k / 200) as f64 * 0.29);
            let v = fbm(5, x, y, &f);
            let r = ridged(5, x, y, &f);
            assert!((-1.0..=1.0).contains(&v) && (0.0..=1.0).contains(&r), "{v} {r}");
        }
        assert_eq!(smoothstep(0.0, 1.0, 0.5), 0.5);
        assert_eq!(smoothstep(0.0, 1.0, -3.0), 0.0);
    }

    #[test]
    fn golden_values() {
        // Fixed values guard the noise against accidental changes (they feed the map hashes).
        let v = [simplex2(0, 0.25, 0.5), simplex2(42, -13.7, 8.1), fbm(9, 3.3, -1.2, &Fractal::new(4))];
        let bits = v.map(f64::to_bits);
        assert_eq!(bits, [4603638946825985774, 13810383131601483048, 13822188526255662516], "{v:?}");
    }
}
