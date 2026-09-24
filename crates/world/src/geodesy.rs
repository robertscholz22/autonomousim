//! WGS-84 geodesy: conversion between the map's local ENU frame and geodetic coordinates.

use glam::DVec3;
use serde::{Deserialize, Serialize};

const A: f64 = 6_378_137.0;
const F: f64 = 1.0 / 298.257_223_563;
const E2: f64 = F * (2.0 - F);

/// Geodetic position (WGS-84 ellipsoid; altitude above the ellipsoid in metres).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Geodetic {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt: f64,
}

impl Geodetic {
    pub fn new(lat_deg: f64, lon_deg: f64, alt: f64) -> Self {
        Self { lat_deg, lon_deg, alt }
    }

    /// Earth-centred, Earth-fixed coordinates.
    pub fn to_ecef(&self) -> DVec3 {
        let (sl, cl) = self.lat_deg.to_radians().sin_cos();
        let (so, co) = self.lon_deg.to_radians().sin_cos();
        let n = A / (1.0 - E2 * sl * sl).sqrt();
        DVec3::new((n + self.alt) * cl * co, (n + self.alt) * cl * so, (n * (1.0 - E2) + self.alt) * sl)
    }

    /// Inverse of [`to_ecef`](Self::to_ecef) (fixed-point iteration; converges to below 1e-9 m
    /// for altitudes within ±100 km of the ellipsoid).
    pub fn from_ecef(p: DVec3) -> Self {
        let lon = p.y.atan2(p.x);
        let r = p.x.hypot(p.y);
        let mut lat = p.z.atan2(r * (1.0 - E2));
        let mut alt = 0.0;
        for _ in 0..6 {
            let s = lat.sin();
            let n = A / (1.0 - E2 * s * s).sqrt();
            alt = r / lat.cos() - n;
            lat = p.z.atan2(r * (1.0 - E2 * n / (n + alt)));
        }
        Self { lat_deg: lat.to_degrees(), lon_deg: lon.to_degrees(), alt }
    }
}

/// Geodetic anchor of a map: the ENU origin `(0, 0, 0)` sits at this position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeoOrigin {
    pub origin: Geodetic,
}

impl Default for GeoOrigin {
    /// Somewhere in the Bavarian Alps foothills.
    fn default() -> Self {
        Self { origin: Geodetic::new(47.6, 11.0, 600.0) }
    }
}

impl GeoOrigin {
    pub fn new(lat_deg: f64, lon_deg: f64, alt: f64) -> Self {
        Self { origin: Geodetic::new(lat_deg, lon_deg, alt) }
    }

    /// Rows of the ECEF → ENU rotation at the origin.
    fn basis(&self) -> [DVec3; 3] {
        let (sl, cl) = self.origin.lat_deg.to_radians().sin_cos();
        let (so, co) = self.origin.lon_deg.to_radians().sin_cos();
        [DVec3::new(-so, co, 0.0), DVec3::new(-sl * co, -sl * so, cl), DVec3::new(cl * co, cl * so, sl)]
    }

    pub fn enu_to_geodetic(&self, p: DVec3) -> Geodetic {
        let [e, n, u] = self.basis();
        Geodetic::from_ecef(self.origin.to_ecef() + e * p.x + n * p.y + u * p.z)
    }

    pub fn geodetic_to_enu(&self, g: Geodetic) -> DVec3 {
        let [e, n, u] = self.basis();
        let d = g.to_ecef() - self.origin.to_ecef();
        DVec3::new(e.dot(d), n.dot(d), u.dot(d))
    }

    /// Altitude above the ellipsoid of a local point (includes Earth curvature).
    pub fn altitude(&self, p: DVec3) -> f64 {
        self.enu_to_geodetic(p).alt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecef_round_trip() {
        for g in [
            Geodetic::new(0.0, 0.0, 0.0),
            Geodetic::new(47.6, 11.0, 600.0),
            Geodetic::new(-33.9, 151.2, -20.0),
            Geodetic::new(89.0, -120.0, 9000.0),
        ] {
            let back = Geodetic::from_ecef(g.to_ecef());
            assert!((back.lat_deg - g.lat_deg).abs() < 1e-10);
            assert!((back.lon_deg - g.lon_deg).abs() < 1e-10);
            assert!((back.alt - g.alt).abs() < 1e-6);
        }
        // Equatorial radius and polar semi-axis.
        assert!((Geodetic::new(0.0, 0.0, 0.0).to_ecef().x - A).abs() < 1e-6);
        assert!((Geodetic::new(90.0, 0.0, 0.0).to_ecef().z - 6_356_752.314_245).abs() < 1e-3);
    }

    #[test]
    fn enu_round_trip_and_scale() {
        let o = GeoOrigin::default();
        for p in [DVec3::ZERO, DVec3::new(1000.0, -2000.0, 150.0), DVec3::new(-5000.0, 5000.0, -50.0)] {
            let back = o.geodetic_to_enu(o.enu_to_geodetic(p));
            assert!((back - p).length() < 1e-6, "{p} -> {back}");
        }
        // One kilometre north changes latitude by ~0.009°, east changes longitude by 0.009°/cos(lat).
        let g = o.enu_to_geodetic(DVec3::new(0.0, 1000.0, 0.0));
        assert!(((g.lat_deg - 47.6) - 0.008_99).abs() < 5e-5);
        let g = o.enu_to_geodetic(DVec3::new(1000.0, 0.0, 0.0));
        assert!(((g.lon_deg - 11.0) - 0.008_99 / 47.6f64.to_radians().cos()).abs() < 1e-4);
        // Earth curvature: 1 km away the tangent plane is ~7.8 cm above the ellipsoid.
        assert!((o.altitude(DVec3::new(1000.0, 0.0, 0.0)) - 600.0 - 0.0785).abs() < 2e-3);
    }
}
