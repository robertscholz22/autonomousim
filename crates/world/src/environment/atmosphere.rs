//! International Standard Atmosphere (US Standard Atmosphere 1976 layers up to 86 km) with a
//! temperature offset and a sea-level pressure setting.

use serde::{Deserialize, Serialize};

/// Sea-level standard temperature (K).
pub const T0: f64 = 288.15;
/// Sea-level standard pressure (Pa).
pub const P0: f64 = 101_325.0;
/// Specific gas constant of dry air (J/(kg·K)).
pub const R_AIR: f64 = 287.052_87;
/// Ratio of specific heats of air.
pub const GAMMA: f64 = 1.4;
const G0: f64 = autonomousim_core::math::frames::STANDARD_GRAVITY;
/// Effective Earth radius for the geometric → geopotential conversion (m).
const R_EARTH: f64 = 6_356_766.0;

/// Layer base geopotential altitude (m) and temperature lapse rate (K/m).
const LAYERS: [(f64, f64); 7] = [
    (0.0, -0.0065),
    (11_000.0, 0.0),
    (20_000.0, 0.001),
    (32_000.0, 0.0028),
    (47_000.0, 0.0),
    (51_000.0, -0.0028),
    (71_000.0, -0.002),
];

/// Air properties at a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AirState {
    /// Static temperature (K).
    pub temperature: f64,
    /// Static pressure (Pa).
    pub pressure: f64,
    /// Density (kg/m³).
    pub density: f64,
    /// Speed of sound (m/s).
    pub speed_of_sound: f64,
}

/// Atmosphere model: ISA shifted by `temperature_offset` (ISA+ΔT) with the pressure profile
/// scaled to `sea_level_pressure` (QNH). Density follows from the ideal gas law, so a hot day
/// or low pressure reduces rotor thrust as expected.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Atmosphere {
    /// ΔT added to the ISA temperature at every altitude (K).
    pub temperature_offset: f64,
    /// Mean-sea-level pressure (Pa).
    pub sea_level_pressure: f64,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self { temperature_offset: 0.0, sea_level_pressure: P0 }
    }
}

/// Geopotential altitude of a geometric altitude above mean sea level.
#[inline]
pub fn geopotential(h: f64) -> f64 {
    R_EARTH * h / (R_EARTH + h)
}

/// Geometric altitude of a geopotential altitude.
#[inline]
pub fn geometric(h: f64) -> f64 {
    R_EARTH * h / (R_EARTH - h)
}

/// Temperature and pressure after climbing `dh` through a layer with the given lapse rate.
#[inline]
fn climb(t: f64, p: f64, lapse: f64, dh: f64) -> (f64, f64) {
    if lapse == 0.0 {
        (t, p * (-G0 * dh / (R_AIR * t)).exp())
    } else {
        let t1 = t + lapse * dh;
        (t1, p * (t1 / t).powf(-G0 / (lapse * R_AIR)))
    }
}

/// Standard temperature and pressure at geopotential altitude `h` (m); the first and last
/// layers are extended below sea level and above 84.85 km.
pub fn isa(h: f64) -> (f64, f64) {
    let (mut t, mut p) = (T0, P0);
    for (i, &(base, lapse)) in LAYERS.iter().enumerate() {
        let top = LAYERS.get(i + 1).map_or(f64::INFINITY, |l| l.0);
        if h <= top {
            return climb(t, p, lapse, h - base);
        }
        (t, p) = climb(t, p, lapse, top - base);
    }
    unreachable!()
}

/// Geopotential altitude at which the standard pressure equals `p` (inverse of [`isa`]).
pub fn isa_altitude(p: f64) -> f64 {
    let (mut t, mut pb) = (T0, P0);
    for (i, &(base, lapse)) in LAYERS.iter().enumerate() {
        let top = LAYERS.get(i + 1).map_or(f64::INFINITY, |l| l.0);
        let (t_top, p_top) = if top.is_finite() { climb(t, pb, lapse, top - base) } else { (t, 0.0) };
        if p >= p_top {
            return if lapse == 0.0 {
                base - R_AIR * t / G0 * (p / pb).ln()
            } else {
                base + t / lapse * ((p / pb).powf(-lapse * R_AIR / G0) - 1.0)
            };
        }
        (t, pb) = (t_top, p_top);
    }
    unreachable!()
}

impl Atmosphere {
    /// Air properties at geometric altitude `h` above mean sea level (m).
    pub fn at_altitude(&self, h: f64) -> AirState {
        self.at_geopotential(geopotential(h))
    }

    /// Air properties at geopotential altitude `h` (m).
    pub fn at_geopotential(&self, h: f64) -> AirState {
        let (t_std, p_std) = isa(h);
        let temperature = t_std + self.temperature_offset;
        let pressure = p_std * (self.sea_level_pressure / P0);
        AirState {
            temperature,
            pressure,
            density: pressure / (R_AIR * temperature),
            speed_of_sound: (GAMMA * R_AIR * temperature).sqrt(),
        }
    }

    /// Air density at geometric altitude `h` (m).
    #[inline]
    pub fn density(&self, h: f64) -> f64 {
        self.at_altitude(h).density
    }

    /// Altitude a barometric altimeter set to this atmosphere's sea-level pressure reads for a
    /// static pressure `p` (geometric metres; the altimeter assumes ISA temperatures).
    pub fn baro_altitude(&self, p: f64) -> f64 {
        geometric(isa_altitude(p * P0 / self.sea_level_pressure))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    #[test]
    fn standard_table_values() {
        // (geopotential altitude, T, p, ρ) from the US Standard Atmosphere 1976 tables.
        let table = [
            (0.0, 288.15, 101_325.0, 1.2250),
            (1_000.0, 281.65, 89_874.6, 1.1116),
            (5_000.0, 255.65, 54_019.9, 0.73612),
            (11_000.0, 216.65, 22_632.1, 0.36392),
            (20_000.0, 216.65, 5_474.89, 0.088035),
            (32_000.0, 228.65, 868.019, 0.013225),
            (47_000.0, 270.65, 110.906, 0.0014275),
        ];
        let atm = Atmosphere::default();
        for (h, t, p, rho) in table {
            let a = atm.at_geopotential(h);
            assert!((a.temperature - t).abs() < 1e-9, "T at {h}");
            assert!(rel(a.pressure, p) < 2e-5, "p at {h}: {}", a.pressure);
            assert!(rel(a.density, rho) < 2e-4, "rho at {h}: {}", a.density);
        }
        assert!((atm.at_geopotential(0.0).speed_of_sound - 340.294).abs() < 1e-3);
        // Geometric 1000 m is geopotential 999.84 m: slightly higher pressure.
        assert!(rel(atm.at_altitude(1000.0).pressure, 89_876.3) < 1e-5);
    }

    #[test]
    fn offsets_and_inverse() {
        let hot = Atmosphere { temperature_offset: 20.0, sea_level_pressure: 100_000.0 };
        let a = hot.at_altitude(500.0);
        let s = Atmosphere::default().at_altitude(500.0);
        assert!((a.temperature - s.temperature - 20.0).abs() < 1e-12);
        assert!(rel(a.pressure, s.pressure * 100_000.0 / P0) < 1e-12);
        assert!(a.density < s.density * 0.93);
        for h in [-300.0, 0.0, 150.0, 3000.0, 11_000.0, 15_000.0, 25_000.0, 40_000.0, 50_000.0, 60_000.0] {
            assert!((isa_altitude(isa(h).1) - h).abs() < 1e-6, "{h}");
            assert!((hot.baro_altitude(hot.at_altitude(h).pressure) - h).abs() < 1e-5, "{h}");
        }
    }
}
