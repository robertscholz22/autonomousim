//! Raycast LiDAR: a fixed set of beams cast against terrain, water, obstacles and agents.
//!
//! A scan is taken at one instant (no motion distortion) and is available in the tick it is
//! taken. Ranges are `f32`; a beam without a return (nothing within range, absorbed by water,
//! dropped out) reads `+∞`.

use crate::{BodyKinematics, Mount, SensorEnv, SensorError, Targets, Timing};
use autonomousim_core::geometry::{HitKind, Ray};
use autonomousim_core::math::Pose;
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::time::Clock;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Beam layout in the sensor frame (x forward, z up).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum BeamPattern {
    /// Rings at the given elevations (degrees, positive up), each with `azimuths` beams spread
    /// over `azimuth_fov` degrees centred on +x (360: a full turn starting at +x, going
    /// counter-clockwise). Beams are ordered ring by ring, lowest elevation first.
    Rings { elevations: Vec<f64>, azimuths: u32, azimuth_fov: f64 },
    /// Explicit beam directions.
    Beams { directions: Vec<DVec3> },
}

impl BeamPattern {
    /// `n` rings evenly spaced from `lo` to `hi` degrees.
    pub fn rings(n: usize, lo: f64, hi: f64, azimuths: u32, azimuth_fov: f64) -> Self {
        let elevations =
            (0..n).map(|i| if n == 1 { 0.5 * (lo + hi) } else { lo + (hi - lo) * i as f64 / (n - 1) as f64 }).collect();
        Self::Rings { elevations, azimuths, azimuth_fov }
    }

    /// Unit beam directions in the sensor frame.
    pub fn directions(&self) -> Vec<DVec3> {
        match self {
            BeamPattern::Rings { elevations, azimuths, azimuth_fov } => {
                let n = *azimuths as usize;
                let full = (*azimuth_fov - 360.0).abs() < 1e-9;
                let step = if full || n == 1 { azimuth_fov / n as f64 } else { azimuth_fov / (n - 1) as f64 };
                let start = if full || n == 1 { 0.0 } else { -0.5 * azimuth_fov };
                let mut out = Vec::with_capacity(elevations.len() * n);
                for &el in elevations {
                    let (se, ce) = el.to_radians().sin_cos();
                    for k in 0..n {
                        let (sa, ca) = (start + step * k as f64).to_radians().sin_cos();
                        out.push(DVec3::new(ce * ca, ce * sa, se));
                    }
                }
                out
            }
            BeamPattern::Beams { directions } => directions.iter().map(|d| d.normalize()).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LidarConfig {
    pub rate_hz: u32,
    pub mount: Mount,
    pub pattern: BeamPattern,
    /// Returns closer than `min_range` are discarded; beams reach `max_range` (m).
    pub min_range: f64,
    pub max_range: f64,
    /// Range noise standard deviation (m).
    pub noise: f64,
    /// Probability that a return is lost.
    pub dropout: f64,
    pub targets: Targets,
}

impl Default for LidarConfig {
    fn default() -> Self {
        Self::rl64()
    }
}

impl LidarConfig {
    /// Sparse 64-beam layout for learning: 4 rings (−24°, −8°, 8°, 24°) × 16 azimuths, 40 m,
    /// 10 Hz.
    pub fn rl64() -> Self {
        Self {
            rate_hz: 10,
            mount: Mount::default(),
            pattern: BeamPattern::Rings { elevations: vec![-24.0, -8.0, 8.0, 24.0], azimuths: 16, azimuth_fov: 360.0 },
            min_range: 0.1,
            max_range: 40.0,
            noise: 0.02,
            dropout: 0.0,
            targets: Targets::default(),
        }
    }

    /// Velodyne VLP-16-like layout: 16 rings from −15° to +15°, 0.2° azimuth steps, 100 m,
    /// 10 Hz (28 800 beams; for the viewer, not for training).
    pub fn vlp16_like() -> Self {
        Self {
            rate_hz: 10,
            mount: Mount::default(),
            pattern: BeamPattern::rings(16, -15.0, 15.0, 1800, 360.0),
            min_range: 0.5,
            max_range: 100.0,
            noise: 0.03,
            dropout: 0.0,
            targets: Targets::default(),
        }
    }

    pub fn validate(&self) -> Result<(), SensorError> {
        let pattern_ok = match &self.pattern {
            BeamPattern::Rings { elevations, azimuths, azimuth_fov } => {
                !elevations.is_empty()
                    && elevations.iter().all(|e| e.abs() <= 90.0)
                    && *azimuths > 0
                    && *azimuth_fov > 0.0
                    && *azimuth_fov <= 360.0
            }
            BeamPattern::Beams { directions } => {
                !directions.is_empty() && directions.iter().all(|d| d.is_finite() && d.length() > 1e-9)
            }
        };
        let ok = pattern_ok
            && self.min_range >= 0.0
            && self.max_range > self.min_range
            && self.max_range.is_finite()
            && self.noise.is_finite()
            && self.noise >= 0.0
            && (0.0..=1.0).contains(&self.dropout);
        if ok { Ok(()) } else { Err(SensorError::InvalidConfig(format!("{self:?}"))) }
    }
}

/// What a beam returned from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ReturnKind {
    #[default]
    None = 0,
    Terrain = 1,
    Water = 2,
    Solid = 3,
    Foliage = 4,
    Agent = 5,
}

impl From<HitKind> for ReturnKind {
    fn from(k: HitKind) -> Self {
        match k {
            HitKind::Terrain => ReturnKind::Terrain,
            HitKind::Water => ReturnKind::Water,
            HitKind::Solid(_) => ReturnKind::Solid,
            HitKind::Foliage(_) => ReturnKind::Foliage,
            HitKind::Agent(_) => ReturnKind::Agent,
        }
    }
}

/// One scan, beam order as in [`BeamPattern::directions`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LidarScan {
    pub tick: u64,
    pub time: f64,
    /// World pose of the sensor frame at the scan.
    pub pose: Pose,
    /// Range per beam (m; `+∞`: no return).
    pub ranges: Vec<f32>,
    pub kinds: Vec<ReturnKind>,
}

impl LidarScan {
    /// World points of the returns.
    pub fn points<'a>(&'a self, directions: &'a [DVec3]) -> impl Iterator<Item = DVec3> + 'a {
        self.ranges.iter().zip(directions).filter(|(r, _)| r.is_finite()).map(|(&r, d)| {
            let local = *d * f64::from(r);
            self.pose.transform_point(local)
        })
    }
}

#[derive(Clone, Debug)]
pub struct Lidar {
    config: LidarConfig,
    timing: Timing,
    directions: Vec<DVec3>,
    rng: SimRng,
    scan: LidarScan,
    valid: bool,
}

impl Lidar {
    pub fn new(config: LidarConfig, clock: &Clock, seed: Seed) -> Result<Self, SensorError> {
        config.validate()?;
        let timing = Timing::new("lidar", clock, config.rate_hz, 0.0)?;
        let directions = config.pattern.directions();
        let n = directions.len();
        Ok(Self {
            timing,
            directions,
            rng: seed.rng(),
            scan: LidarScan {
                ranges: vec![f32::INFINITY; n],
                kinds: vec![ReturnKind::None; n],
                ..LidarScan::default()
            },
            valid: false,
            config,
        })
    }

    pub fn config(&self) -> &LidarConfig {
        &self.config
    }

    pub fn num_beams(&self) -> usize {
        self.directions.len()
    }

    /// Beam directions in the sensor frame.
    pub fn directions(&self) -> &[DVec3] {
        &self.directions
    }

    pub fn reset(&mut self, seed: Seed) {
        self.rng = seed.rng();
        self.valid = false;
    }

    /// Cast all beams now and store the scan.
    pub fn scan(&mut self, tick: u64, time: f64, kin: &BodyKinematics, env: &SensorEnv) {
        let c = &self.config;
        let pose = c.mount.world_pose(kin);
        let mask = c.targets.query_mask();
        for (i, d) in self.directions.iter().enumerate() {
            let ray = Ray { origin: pose.pos, dir: pose.rot * *d };
            let hit = env.rays.raycast(&ray, c.max_range, mask).filter(|h| c.targets.returns(h.kind));
            let (mut range, mut kind) = match hit {
                Some(h) if h.toi >= c.min_range => (h.toi, ReturnKind::from(h.kind)),
                _ => (f64::INFINITY, ReturnKind::None),
            };
            if kind != ReturnKind::None {
                if c.dropout > 0.0 && self.rng.chance(c.dropout) {
                    (range, kind) = (f64::INFINITY, ReturnKind::None);
                } else if c.noise > 0.0 {
                    range = (range + c.noise * self.rng.normal()).clamp(c.min_range, c.max_range);
                }
            }
            self.scan.ranges[i] = range as f32;
            self.scan.kinds[i] = kind;
        }
        self.scan.tick = tick;
        self.scan.time = time;
        self.scan.pose = pose;
        self.valid = true;
    }

    /// Advance one physics tick; true if a new scan was taken.
    pub fn update(&mut self, tick: u64, time: f64, kin: &BodyKinematics, env: &SensorEnv) -> bool {
        if self.timing.is_due(tick) {
            self.scan(tick, time, kin, env);
            true
        } else {
            false
        }
    }

    pub fn latest(&self) -> Option<&LidarScan> {
        self.valid.then_some(&self.scan)
    }
}
