//! Earth magnetic field, constant over a map (maps are at most tens of kilometres wide).

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// IGRF-13 (epoch 2020) degree-1 Gauss coefficients in nT.
const G10: f64 = -29_404.8;
const G11: f64 = -1_450.9;
const H11: f64 = 4_652.5;

/// Magnetic flux density in the world ENU frame (tesla).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MagneticField {
    pub enu: DVec3,
}

impl MagneticField {
    /// Field from declination (east of true north), inclination (dip, positive downwards) and
    /// total intensity, as published by NOAA/BGS calculators. Angles in radians, intensity in T.
    pub fn from_components(declination: f64, inclination: f64, intensity: f64) -> Self {
        let horizontal = intensity * inclination.cos();
        Self {
            enu: DVec3::new(
                horizontal * declination.sin(),
                horizontal * declination.cos(),
                -intensity * inclination.sin(),
            ),
        }
    }

    /// Centred-dipole approximation of IGRF-13 at the Earth's surface. Intensity and dip are
    /// within ~10 %, declination can be off by 20° (use [`from_components`](Self::from_components)
    /// for a specific site).
    pub fn dipole(lat_deg: f64, lon_deg: f64) -> Self {
        let (st, ct) = (90.0 - lat_deg).to_radians().sin_cos();
        let (sp, cp) = lon_deg.to_radians().sin_cos();
        let g = G11 * cp + H11 * sp;
        let b_r = 2.0 * (G10 * ct + g * st);
        let b_theta = G10 * st - g * ct;
        let b_phi = G11 * sp - H11 * cp;
        // North = −B_θ, East = B_φ, Up = B_r.
        Self { enu: DVec3::new(b_phi, -b_theta, b_r) * 1e-9 }
    }

    /// Declination: angle of the horizontal field east of true north (rad).
    pub fn declination(&self) -> f64 {
        self.enu.x.atan2(self.enu.y)
    }

    /// Inclination: angle of the field below the horizontal (rad).
    pub fn inclination(&self) -> f64 {
        (-self.enu.z).atan2(self.enu.truncate().length())
    }

    /// Total intensity (T).
    pub fn intensity(&self) -> f64 {
        self.enu.length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_round_trip() {
        let f = MagneticField::from_components(0.07, 1.12, 48.5e-6);
        assert!((f.declination() - 0.07).abs() < 1e-12);
        assert!((f.inclination() - 1.12).abs() < 1e-12);
        assert!((f.intensity() - 48.5e-6).abs() < 1e-18);
        assert!(f.enu.z < 0.0 && f.enu.y > 0.0);
    }

    #[test]
    fn dipole_is_plausible() {
        // Munich (IGRF: F ≈ 48.5 µT, I ≈ 64°).
        let f = MagneticField::dipole(48.1, 11.6);
        assert!((f.intensity() - 48.5e-6).abs() < 3e-6, "{}", f.intensity());
        assert!((f.inclination().to_degrees() - 64.0).abs() < 4.0);
        assert!(f.declination().to_degrees().abs() < 20.0);
        // Sydney: field points up (southern hemisphere), I ≈ −64°.
        let f = MagneticField::dipole(-33.9, 151.2);
        assert!((f.inclination().to_degrees() + 64.0).abs() < 8.0);
        // Geomagnetic north pole of the 2020 dipole: field is vertical, strongest.
        let f = MagneticField::dipole(80.65, -72.68);
        assert!(f.inclination().to_degrees() > 89.9);
        for lat in [-60.0, -20.0, 0.0, 30.0, 70.0] {
            let i = MagneticField::dipole(lat, 30.0).intensity();
            assert!((25e-6..70e-6).contains(&i), "{lat}: {i}");
        }
    }
}
