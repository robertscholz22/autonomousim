//! Environment models: atmosphere, wind and the Earth magnetic field.

pub mod atmosphere;
pub mod magnetic;
pub mod wind;

pub use atmosphere::{AirState, Atmosphere};
pub use magnetic::MagneticField;
pub use wind::{Dryden, DrydenScales, Gust, WindConfig};

use crate::geodesy::GeoOrigin;
use autonomousim_core::math::frames::STANDARD_GRAVITY;
use serde::{Deserialize, Serialize};

/// Environment of one episode (may be randomised per episode by the scenario).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EnvironmentConfig {
    /// Gravitational acceleration (m/s²).
    pub gravity: f64,
    pub atmosphere: Atmosphere,
    pub wind: WindConfig,
    /// Magnetic field override; `None` uses the dipole model at the map's geodetic origin.
    pub magnetic: Option<MagneticField>,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self { gravity: STANDARD_GRAVITY, atmosphere: Atmosphere::default(), wind: WindConfig::calm(), magnetic: None }
    }
}

impl EnvironmentConfig {
    /// The magnetic field on a map anchored at `origin`.
    pub fn magnetic_field(&self, origin: &GeoOrigin) -> MagneticField {
        self.magnetic.unwrap_or_else(|| MagneticField::dipole(origin.origin.lat_deg, origin.origin.lon_deg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_from_partial_toml_like_json() {
        let c: EnvironmentConfig =
            serde_json::from_str(r#"{"wind": {"mean": [2.0, 0.0], "turbulence_w20": 7.7}}"#).unwrap();
        assert_eq!(c.gravity, STANDARD_GRAVITY);
        assert_eq!(c.wind.reference_height, WindConfig::default().reference_height);
        assert!(c.wind.has_turbulence());
        let m = c.magnetic_field(&GeoOrigin::default());
        assert!(m.intensity() > 40e-6);
    }
}
