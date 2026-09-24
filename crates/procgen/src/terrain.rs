//! Height-field stages: noise terrain, particle hydraulic erosion, thermal erosion and 2×
//! upsampling. Parallel stages compute every sample independently and write disjoint rows, so
//! results do not depend on the thread count.

use crate::noise::{Fractal, fbm, ridged, smoothstep};
use autonomousim_core::rng::SimRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Row-major `w × h` samples.
#[derive(Clone, Debug)]
pub(crate) struct Field {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f64>,
}

impl Field {
    /// Field filled row by row (in parallel) by `row(iy, samples)`.
    pub fn from_rows(w: usize, h: usize, row: impl Fn(usize, &mut [f64]) + Sync) -> Self {
        let mut data = vec![0.0; w * h];
        data.par_chunks_mut(w).enumerate().for_each(|(iy, r)| row(iy, r));
        Self { w, h, data }
    }

    #[inline]
    pub fn at(&self, ix: usize, iy: usize) -> f64 {
        self.data[iy * self.w + ix]
    }

    /// Sample with indices clamped to the border.
    #[inline]
    fn at_clamped(&self, ix: isize, iy: isize) -> f64 {
        let x = ix.clamp(0, self.w as isize - 1) as usize;
        let y = iy.clamp(0, self.h as isize - 1) as usize;
        self.data[y * self.w + x]
    }
}

// ------------------------------------------------------------------------------ base terrain

/// Large-scale shape: rolling hills, ridged mountains where a low-frequency mask says so, and
/// two levels of domain warping (Quilez) that bend both into less regular forms. The warp uses
/// two octaves only: finer warp octaves fold the coordinates and create cliffs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TerrainConfig {
    /// Height scale (m): hills reach about `hills_amplitude · relief`, peaks about `relief`.
    pub relief: f64,
    /// Noise periods (m); the resulting features are about half as wide. Octaves with a gain
    /// below 0.5 keep fine octaves from steepening the slopes.
    pub hills_wavelength: f64,
    pub hills_amplitude: f64,
    pub hills: Fractal,
    pub mountains_wavelength: f64,
    pub mountains_amplitude: f64,
    pub mountains: Fractal,
    /// Wavelength of the mask that chooses between hills and mountains (m).
    pub mask_wavelength: f64,
    /// Mask noise value above which mountains dominate (in [−1, 1]; lower = more mountains).
    pub mask_threshold: f64,
    pub warp_wavelength: f64,
    /// Largest displacement of the warp (m).
    pub warp_strength: f64,
    /// Fine noise added after upsampling to the final resolution.
    pub detail_amplitude: f64,
    pub detail_wavelength: f64,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            relief: 300.0,
            hills_wavelength: 800.0,
            hills_amplitude: 0.25,
            hills: Fractal { octaves: 5, lacunarity: 2.0, gain: 0.4 },
            mountains_wavelength: 1600.0,
            mountains_amplitude: 0.9,
            mountains: Fractal { octaves: 6, lacunarity: 2.0, gain: 0.4 },
            mask_wavelength: 1600.0,
            mask_threshold: 0.0,
            warp_wavelength: 800.0,
            warp_strength: 80.0,
            detail_amplitude: 0.25,
            detail_wavelength: 6.0,
        }
    }
}

impl TerrainConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let positive = [
            ("relief", self.relief),
            ("hills_wavelength", self.hills_wavelength),
            ("mountains_wavelength", self.mountains_wavelength),
            ("mask_wavelength", self.mask_wavelength),
            ("warp_wavelength", self.warp_wavelength),
            ("detail_wavelength", self.detail_wavelength),
        ];
        for (name, v) in positive {
            if !(v > 0.0 && v.is_finite()) {
                return Err(format!("terrain.{name} must be positive"));
            }
        }
        for (name, v) in [
            ("hills_amplitude", self.hills_amplitude),
            ("mountains_amplitude", self.mountains_amplitude),
            ("warp_strength", self.warp_strength),
            ("detail_amplitude", self.detail_amplitude),
        ] {
            if !(v >= 0.0 && v.is_finite()) {
                return Err(format!("terrain.{name} must be non-negative"));
            }
        }
        if !self.hills.is_valid() || !self.mountains.is_valid() {
            return Err("terrain fractals need 1–16 octaves, lacunarity > 1 and 0 < gain < 1".into());
        }
        Ok(())
    }
}

/// Noise seeds of the terrain layers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TerrainSeeds {
    pub warp: [u64; 4],
    pub mask: u64,
    pub hills: u64,
    pub mountains: u64,
    pub detail: u64,
}

const WARP_FRACTAL: Fractal = Fractal { octaves: 2, lacunarity: 2.0, gain: 0.5 };
const MASK_FRACTAL: Fractal = Fractal { octaves: 3, lacunarity: 2.0, gain: 0.5 };
const DETAIL_FRACTAL: Fractal = Fractal { octaves: 3, lacunarity: 2.0, gain: 0.5 };

/// Height (m) of the unwarped-noise terrain at world position `(x, y)`.
pub(crate) fn base_height(c: &TerrainConfig, s: &TerrainSeeds, x: f64, y: f64) -> f64 {
    let k = 1.0 / c.warp_wavelength;
    let (px, py) = (x * k, y * k);
    let qx = fbm(s.warp[0], px, py, &WARP_FRACTAL);
    let qy = fbm(s.warp[1], px + 5.2, py + 1.3, &WARP_FRACTAL);
    let rx = fbm(s.warp[2], px + qx + 1.7, py + qy + 9.2, &WARP_FRACTAL);
    let ry = fbm(s.warp[3], px + qx + 8.3, py + qy + 2.8, &WARP_FRACTAL);
    let (wx, wy) = (x + c.warp_strength * rx, y + c.warp_strength * ry);

    let m = fbm(s.mask, x / c.mask_wavelength, y / c.mask_wavelength, &MASK_FRACTAL);
    let mask = smoothstep(c.mask_threshold - 0.2, c.mask_threshold + 0.2, m);
    let hills = 0.5 + 0.5 * fbm(s.hills, wx / c.hills_wavelength, wy / c.hills_wavelength, &c.hills);
    let hills = c.hills_amplitude * hills;
    let mountains = if mask > 0.0 {
        c.mountains_amplitude
            * ridged(s.mountains, wx / c.mountains_wavelength, wy / c.mountains_wavelength, &c.mountains)
    } else {
        0.0
    };
    // Mountains rise out of the (halved) hills.
    c.relief * ((1.0 - 0.5 * mask) * hills + mask * mountains)
}

/// Fine detail (m) added at the final resolution.
pub(crate) fn detail_height(c: &TerrainConfig, s: &TerrainSeeds, x: f64, y: f64) -> f64 {
    if c.detail_amplitude == 0.0 {
        return 0.0;
    }
    c.detail_amplitude * fbm(s.detail, x / c.detail_wavelength, y / c.detail_wavelength, &DETAIL_FRACTAL)
}

// ------------------------------------------------------------------------------ erosion

/// Particle hydraulic erosion (Beyer 2015, as popularised by S. Lague) followed by thermal
/// erosion. Droplet parameters are in units of the relief and the erosion grid cell, so the
/// same values work for any map scale.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ErosionConfig {
    pub enabled: bool,
    /// Droplets per erosion-grid cell.
    pub droplets_per_cell: f64,
    /// Steps per droplet.
    pub lifetime: u32,
    /// How much a droplet keeps its direction (0 = follows the gradient).
    pub inertia: f64,
    pub capacity: f64,
    pub min_capacity: f64,
    pub deposit_rate: f64,
    pub erode_rate: f64,
    pub evaporation: f64,
    pub gravity: f64,
    /// Radius (cells) over which a droplet erodes.
    pub radius: u32,
    /// Jacobi iterations of thermal erosion (0 = off).
    pub thermal_iterations: u32,
    /// Material steeper than this slides downhill.
    pub talus_deg: f64,
    /// Fraction of the excess moved per neighbour and iteration (≤ 1/16 for stability).
    pub thermal_rate: f64,
}

impl Default for ErosionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            droplets_per_cell: 0.6,
            lifetime: 30,
            inertia: 0.05,
            capacity: 4.0,
            min_capacity: 0.01,
            deposit_rate: 0.3,
            erode_rate: 0.3,
            evaporation: 0.01,
            gravity: 4.0,
            radius: 3,
            thermal_iterations: 60,
            talus_deg: 38.0,
            thermal_rate: 0.0625,
        }
    }
}

impl ErosionConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let unit = [
            ("inertia", self.inertia),
            ("deposit_rate", self.deposit_rate),
            ("erode_rate", self.erode_rate),
            ("evaporation", self.evaporation),
        ];
        for (name, v) in unit {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("erosion.{name} must be in [0, 1]"));
            }
        }
        if !(self.droplets_per_cell >= 0.0 && self.droplets_per_cell <= 100.0) {
            return Err("erosion.droplets_per_cell must be in [0, 100]".into());
        }
        if !(1..=16).contains(&self.radius) {
            return Err("erosion.radius must be in 1..=16".into());
        }
        if !(self.thermal_rate > 0.0 && self.thermal_rate <= 1.0 / 16.0) {
            return Err("erosion.thermal_rate must be in (0, 1/16]".into());
        }
        if !(self.talus_deg > 0.0 && self.talus_deg < 90.0) {
            return Err("erosion.talus_deg must be in (0, 90)".into());
        }
        if !(self.capacity >= 0.0 && self.min_capacity >= 0.0 && self.gravity >= 0.0) {
            return Err("erosion.capacity, min_capacity and gravity must be non-negative".into());
        }
        Ok(())
    }
}

/// Bilinear height and gradient of `f` at `(x, y)` (cell units, inside the grid).
#[inline]
fn height_gradient(f: &Field, x: f64, y: f64) -> (f64, f64, f64) {
    let (ix, iy) = (x as usize, y as usize);
    let (u, v) = (x - ix as f64, y - iy as f64);
    let i = iy * f.w + ix;
    let (a, b, c, d) = (f.data[i], f.data[i + 1], f.data[i + f.w], f.data[i + f.w + 1]);
    let gx = (b - a) * (1.0 - v) + (d - c) * v;
    let gy = (c - a) * (1.0 - u) + (d - b) * u;
    let h = a * (1.0 - u) * (1.0 - v) + b * u * (1.0 - v) + c * (1.0 - u) * v + d * u * v;
    (h, gx, gy)
}

/// Erode `f` (heights divided by the relief) in place with droplets drawn from `rng`.
/// Sequential by design: droplets interact through the height field.
pub(crate) fn hydraulic_erosion(f: &mut Field, c: &ErosionConfig, rng: &mut SimRng) -> u64 {
    let r = c.radius as i64;
    // Brush: offsets within the radius, weights falling off linearly and summing to one.
    let mut brush: Vec<(i64, i64, f64)> = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            let d = ((dx * dx + dy * dy) as f64).sqrt();
            if d < r as f64 {
                brush.push((dx, dy, r as f64 - d));
            }
        }
    }
    let total: f64 = brush.iter().map(|b| b.2).sum();
    for b in &mut brush {
        b.2 /= total;
    }
    let (w, h) = (f.w as i64, f.h as i64);
    let (xmax, ymax) = ((f.w - 1) as f64, (f.h - 1) as f64);
    let droplets = (c.droplets_per_cell * ((f.w - 1) * (f.h - 1)) as f64).round() as u64;
    for _ in 0..droplets {
        let (mut x, mut y) = (rng.range(0.0, xmax), rng.range(0.0, ymax));
        let (mut dx, mut dy) = (0.0, 0.0);
        let (mut speed, mut water, mut sediment) = (1.0, 1.0, 0.0);
        for _ in 0..c.lifetime {
            let (ix, iy) = (x as usize, y as usize);
            let (u, v) = (x - ix as f64, y - iy as f64);
            let (height, gx, gy) = height_gradient(f, x, y);
            dx = dx * c.inertia - gx * (1.0 - c.inertia);
            dy = dy * c.inertia - gy * (1.0 - c.inertia);
            let len = (dx * dx + dy * dy).sqrt();
            if len == 0.0 {
                break;
            }
            dx /= len;
            dy /= len;
            x += dx;
            y += dy;
            if !(0.0..xmax).contains(&x) || !(0.0..ymax).contains(&y) {
                break;
            }
            let dh = height_gradient(f, x, y).0 - height;
            let capacity = (-dh * speed * water * c.capacity).max(c.min_capacity);
            if sediment > capacity || dh > 0.0 {
                // Fill the pit behind (uphill) or drop the excess, bilinearly at the old position.
                let amount = if dh > 0.0 { dh.min(sediment) } else { (sediment - capacity) * c.deposit_rate };
                sediment -= amount;
                let i = iy * f.w + ix;
                f.data[i] += amount * (1.0 - u) * (1.0 - v);
                f.data[i + 1] += amount * u * (1.0 - v);
                f.data[i + f.w] += amount * (1.0 - u) * v;
                f.data[i + f.w + 1] += amount * u * v;
            } else {
                let amount = ((capacity - sediment) * c.erode_rate).min(-dh);
                for &(bx, by, bw) in &brush {
                    let (px, py) = (ix as i64 + bx, iy as i64 + by);
                    if px < 0 || py < 0 || px >= w || py >= h {
                        continue;
                    }
                    let e = amount * bw;
                    f.data[(py * w + px) as usize] -= e;
                    sediment += e;
                }
            }
            speed = (speed * speed - dh * c.gravity).max(0.0).sqrt();
            water *= 1.0 - c.evaporation;
        }
    }
    droplets
}

/// Thermal erosion: wherever the height difference to a neighbour exceeds the talus
/// difference, a fraction of the excess moves downhill. Each iteration is a Jacobi update in
/// which every cell gathers its exchanges with its eight neighbours from the previous heights;
/// the exchange between two cells is computed identically from both sides, so mass is
/// conserved and rows can be processed in parallel.
pub(crate) fn thermal_erosion(f: &mut Field, cell: f64, c: &ErosionConfig) {
    let talus = libm::tan(c.talus_deg.to_radians()) * cell;
    let talus_diag = talus * std::f64::consts::SQRT_2;
    let (w, h) = (f.w, f.h);
    let mut next = vec![0.0; w * h];
    for _ in 0..c.thermal_iterations {
        let cur = &f.data;
        next.par_chunks_mut(w).enumerate().for_each(|(iy, row)| {
            for (ix, out) in row.iter_mut().enumerate() {
                let hi = cur[iy * w + ix];
                let mut delta = 0.0;
                for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let (nx, ny) = (ix as isize + dx, iy as isize + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let hj = cur[ny as usize * w + nx as usize];
                    let t = if dx != 0 && dy != 0 { talus_diag } else { talus };
                    if hi - hj > t {
                        delta -= c.thermal_rate * (hi - hj - t);
                    } else if hj - hi > t {
                        delta += c.thermal_rate * (hj - hi - t);
                    }
                }
                *out = hi + delta;
            }
        });
        std::mem::swap(&mut f.data, &mut next);
    }
}

// ------------------------------------------------------------------------------ upsampling

/// Catmull–Rom weights at the midpoint.
#[inline]
fn midpoint(a: f64, b: f64, c: f64, d: f64) -> f64 {
    (9.0 * (b + c) - (a + d)) * (1.0 / 16.0)
}

/// Twice the resolution (`2w − 1 × 2h − 1` samples): coarse samples are kept, the new ones
/// are Catmull–Rom interpolated (separably, borders clamped).
pub(crate) fn upsample2(f: &Field) -> Field {
    let (w2, h2) = (2 * f.w - 1, 2 * f.h - 1);
    let rows = Field::from_rows(w2, f.h, |iy, row| {
        let y = iy as isize;
        for (ix, out) in row.iter_mut().enumerate() {
            let k = (ix / 2) as isize;
            *out = if ix % 2 == 0 {
                f.at(ix / 2, iy)
            } else {
                midpoint(f.at_clamped(k - 1, y), f.at_clamped(k, y), f.at_clamped(k + 1, y), f.at_clamped(k + 2, y))
            };
        }
    });
    Field::from_rows(w2, h2, |iy, row| {
        let k = (iy / 2) as isize;
        for (ix, out) in row.iter_mut().enumerate() {
            let x = ix as isize;
            *out = if iy % 2 == 0 {
                rows.at(ix, iy / 2)
            } else {
                midpoint(
                    rows.at_clamped(x, k - 1),
                    rows.at_clamped(x, k),
                    rows.at_clamped(x, k + 1),
                    rows.at_clamped(x, k + 2),
                )
            };
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::rng::Seed;

    fn cone(n: usize) -> Field {
        let c = (n - 1) as f64 / 2.0;
        Field::from_rows(n, n, |iy, row| {
            for (ix, v) in row.iter_mut().enumerate() {
                let r = ((ix as f64 - c).powi(2) + (iy as f64 - c).powi(2)).sqrt();
                *v = (1.0 - r / c).max(0.0);
            }
        })
    }

    #[test]
    fn thermal_erosion_conserves_mass_and_flattens() {
        let mut f = cone(33);
        f.data[16 * 33 + 16] += 20.0; // a spike far above the talus slope
        let before: f64 = f.data.iter().sum();
        let c = ErosionConfig { thermal_iterations: 200, ..Default::default() };
        thermal_erosion(&mut f, 1.0, &c);
        let after: f64 = f.data.iter().sum();
        assert!((before - after).abs() < 1e-9, "{before} {after}");
        let talus = (c.talus_deg.to_radians()).tan();
        let steepest =
            (0..32).flat_map(|y| (0..32).map(move |x| (x, y))).map(|(x, y)| (f.at(x + 1, y) - f.at(x, y)).abs());
        assert!(steepest.fold(0.0, f64::max) < 1.05 * talus);
    }

    #[test]
    fn hydraulic_erosion_moves_material_downhill() {
        let mut f = cone(65);
        let before = f.clone();
        let mut rng = Seed::from_u64(1).rng();
        let c = ErosionConfig { droplets_per_cell: 2.0, ..Default::default() };
        // The cone slope is 1/32 per cell: similar to real terrain in relief units.
        hydraulic_erosion(&mut f, &c, &mut rng);
        let centre = |g: &Field| g.at(32, 32);
        let rim: f64 = (0..65).map(|i| f.at(i, 2) - before.at(i, 2)).sum();
        assert!(centre(&f) <= centre(&before) && f.data.iter().all(|v| v.is_finite()));
        let changed = f.data.iter().zip(&before.data).filter(|(a, b)| a != b).count();
        assert!(changed > 1000, "{changed}");
        assert!(rim.is_finite());
    }

    #[test]
    fn upsample_keeps_coarse_samples_and_is_exact_for_cubics() {
        let f = Field::from_rows(9, 7, |iy, row| {
            for (ix, v) in row.iter_mut().enumerate() {
                let (x, y) = (ix as f64, iy as f64);
                *v = 0.3 * x * x - 0.1 * x * y + 2.0 * y + 0.01 * y * y * y;
            }
        });
        let g = upsample2(&f);
        assert_eq!((g.w, g.h), (17, 13));
        for iy in 0..7 {
            for ix in 0..9 {
                assert_eq!(g.at(2 * ix, 2 * iy), f.at(ix, iy));
            }
        }
        // Interior midpoints of a cubic polynomial are reproduced exactly (Catmull–Rom is
        // exact for quadratics; the cubic term is separable here).
        let (x, y) = (7.0 / 2.0, 7.0 / 2.0);
        let expect = 0.3 * x * x - 0.1 * x * y + 2.0 * y + 0.01 * y * y * y;
        assert!((g.at(7, 7) - expect).abs() < 0.01, "{} {expect}", g.at(7, 7));
    }
}
