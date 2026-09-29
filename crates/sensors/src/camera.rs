//! Pinhole camera: RGB, depth and semantic images.
//!
//! The images are rendered outside the sensor (by the `render` crate, batched over worlds);
//! the sensor holds the configuration, decides when a frame is due, adds noise from its own
//! random stream and delays frames by a whole number of camera periods.
//!
//! Camera frame: FLU with the optical axis along +x (as the body frame), so a mount without
//! rotation looks forward; a rotation of +90° about y looks down.

use crate::latency::Stamped;
use crate::{Mount, SensorError};
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::time::Clock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraNoise {
    /// Standard deviation of the pixel noise, as a fraction of full scale (added to the
    /// encoded 8-bit values).
    pub pixel: f64,
    /// Depth noise: standard deviation `depth · d²` at depth `d` (1/m), as for stereo and
    /// structured-light sensors.
    pub depth: f64,
    /// Standard deviation of the natural log of a per-frame gain on the colour values.
    pub exposure: f64,
}

impl CameraNoise {
    pub fn is_zero(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraConfig {
    /// Image size (pixels).
    pub width: u32,
    pub height: u32,
    /// Horizontal field of view (degrees).
    pub fov_deg: f64,
    pub mount: Mount,
    /// Frames per second; the period must be a whole number of policy steps.
    pub rate_hz: u32,
    /// Delay of the images in frames: the image seen at a frame is the one captured `latency`
    /// frames earlier.
    pub latency: u32,
    /// Clip distances along the optical axis (m); nothing beyond `far` is seen.
    pub near: f64,
    pub far: f64,
    #[serde(skip_serializing_if = "CameraNoise::is_zero")]
    pub noise: CameraNoise,
}

impl Default for CameraConfig {
    /// A 64×64 forward camera with a 90° field of view at 25 Hz.
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            fov_deg: 90.0,
            mount: Mount::default(),
            rate_hz: 25,
            latency: 0,
            near: 0.05,
            far: 1000.0,
            noise: CameraNoise::default(),
        }
    }
}

/// Largest image side (pixels).
pub const MAX_IMAGE_SIDE: u32 = 4096;

impl CameraConfig {
    pub fn validate(&self) -> Result<(), SensorError> {
        let n = &self.noise;
        let ok = (1..=MAX_IMAGE_SIDE).contains(&self.width)
            && (1..=MAX_IMAGE_SIDE).contains(&self.height)
            && self.fov_deg > 0.0
            && self.fov_deg < 180.0
            && self.near > 0.0
            && self.far > self.near
            && self.far.is_finite()
            && self.mount.position.is_finite()
            && self.mount.rotation.is_finite()
            && self.latency <= 64
            && [n.pixel, n.depth, n.exposure].iter().all(|x| x.is_finite() && *x >= 0.0);
        if ok { Ok(()) } else { Err(SensorError::InvalidConfig(format!("{self:?}"))) }
    }
}

/// What a camera sees, row-major from the top-left pixel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CameraImage {
    pub width: u32,
    pub height: u32,
    /// sRGB-encoded RGB, 3 bytes per pixel.
    pub rgb: Vec<u8>,
    /// Depth along the optical axis (m); 0 where nothing was hit.
    pub depth: Vec<f32>,
    /// Semantic class ids (the `render` crate's `SemanticClass`).
    pub class: Vec<u8>,
}

impl CameraImage {
    pub fn pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Hash of the image contents.
    pub fn digest(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(&self.width.to_le_bytes());
        h.update(&self.height.to_le_bytes());
        h.update(&self.rgb);
        for d in &self.depth {
            h.update(&d.to_le_bytes());
        }
        h.update(&self.class);
        *h.finalize().as_bytes()
    }
}

#[derive(Clone, Debug)]
pub struct Camera {
    config: CameraConfig,
    /// Physics ticks per frame.
    divider: u32,
    rng: SimRng,
    /// The last `latency + 1` frames, oldest first; the oldest is the visible one.
    frames: VecDeque<Arc<Stamped<CameraImage>>>,
}

impl Camera {
    pub fn new(config: CameraConfig, clock: &Clock, seed: Seed) -> Result<Self, SensorError> {
        config.validate()?;
        let divider = clock.divider("camera", config.rate_hz)?;
        Ok(Self { divider, rng: seed.rng(), frames: VecDeque::new(), config })
    }

    pub fn config(&self) -> &CameraConfig {
        &self.config
    }

    /// Physics ticks per frame.
    pub fn divider(&self) -> u32 {
        self.divider
    }

    pub fn reset(&mut self, seed: Seed) {
        self.rng = seed.rng();
        self.frames.clear();
    }

    /// Whether a frame is captured at `tick`.
    pub fn is_due(&self, tick: u64) -> bool {
        tick.is_multiple_of(u64::from(self.divider))
    }

    /// Take the frame rendered at `tick`: add noise and queue it. The first frame after a
    /// reset fills the whole delay line, so an image is visible from the start.
    pub fn capture(&mut self, tick: u64, time: f64, mut image: CameraImage) {
        assert_eq!((image.width, image.height), (self.config.width, self.config.height), "camera image size");
        self.add_noise(&mut image);
        let frame = Arc::new(Stamped { tick, time, value: image });
        let len = self.config.latency as usize + 1;
        if self.frames.is_empty() {
            self.frames.resize(len, frame);
        } else {
            self.frames.push_back(frame);
            while self.frames.len() > len {
                self.frames.pop_front();
            }
        }
    }

    fn add_noise(&mut self, image: &mut CameraImage) {
        let n = self.config.noise;
        let rng = &mut self.rng;
        if n.exposure > 0.0 || n.pixel > 0.0 {
            let gain = if n.exposure > 0.0 { (n.exposure * rng.normal()).exp() } else { 1.0 };
            let sigma = 255.0 * n.pixel;
            for c in &mut image.rgb {
                let mut x = f64::from(*c) * gain;
                if sigma > 0.0 {
                    x += sigma * rng.normal();
                }
                *c = x.round().clamp(0.0, 255.0) as u8;
            }
        }
        if n.depth > 0.0 {
            let (near, far) = (self.config.near as f32, self.config.far as f32);
            for d in image.depth.iter_mut().filter(|d| **d > 0.0) {
                let z = f64::from(*d);
                *d = ((z + n.depth * z * z * rng.normal()) as f32).clamp(near, far);
            }
        }
    }

    /// The visible frame (captured `latency` frames ago), if any since the reset.
    pub fn latest(&self) -> Option<&Stamped<CameraImage>> {
        self.frames.front().map(|f| &**f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(v: u8) -> CameraImage {
        CameraImage { width: 2, height: 1, rgb: vec![v; 6], depth: vec![10.0, 0.0], class: vec![1, 0] }
    }

    #[test]
    fn frames_are_delayed_by_whole_frames() {
        let clock = Clock::new(100);
        let config = CameraConfig { width: 2, height: 1, rate_hz: 10, latency: 2, ..CameraConfig::default() };
        let mut cam = Camera::new(config, &clock, Seed::from_u64(1)).unwrap();
        assert_eq!(cam.divider(), 10);
        assert!(cam.latest().is_none());
        assert!(cam.is_due(0) && !cam.is_due(5) && cam.is_due(20));
        cam.capture(0, 0.0, image(0));
        assert_eq!(cam.latest().unwrap().tick, 0);
        let mut seen = Vec::new();
        for k in 1..6u8 {
            cam.capture(u64::from(k) * 10, 0.0, image(k));
            seen.push((cam.latest().unwrap().tick, cam.latest().unwrap().value.rgb[0]));
        }
        assert_eq!(seen, [(0, 0), (0, 0), (10, 1), (20, 2), (30, 3)]);
        cam.reset(Seed::from_u64(1));
        assert!(cam.latest().is_none());
    }

    #[test]
    fn noise_is_seeded_and_spares_the_sky() {
        let clock = Clock::new(100);
        let noise = CameraNoise { pixel: 0.02, depth: 0.01, exposure: 0.1 };
        let config = CameraConfig { width: 2, height: 1, rate_hz: 10, noise, ..CameraConfig::default() };
        let shot = |seed| {
            let mut cam = Camera::new(config.clone(), &clock, Seed::from_u64(seed)).unwrap();
            cam.capture(0, 0.0, image(100));
            cam.latest().unwrap().value.clone()
        };
        let (a, b, c) = (shot(1), shot(1), shot(2));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a.rgb, image(100).rgb);
        assert_ne!(a.depth[0], 10.0);
        assert_eq!((a.depth[1], a.class.clone()), (0.0, vec![1, 0]));
        // Without noise the image passes through unchanged.
        let clean = CameraConfig { noise: CameraNoise::default(), ..config };
        let mut cam = Camera::new(clean, &clock, Seed::from_u64(1)).unwrap();
        cam.capture(0, 0.0, image(100));
        assert_eq!(cam.latest().unwrap().value, image(100));
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let clock = Clock::new(100);
        for bad in [
            CameraConfig { width: 0, ..CameraConfig::default() },
            CameraConfig { fov_deg: 180.0, ..CameraConfig::default() },
            CameraConfig { near: 2000.0, ..CameraConfig::default() },
            CameraConfig { rate_hz: 30, ..CameraConfig::default() },
        ] {
            assert!(Camera::new(bad, &clock, Seed::from_u64(0)).is_err());
        }
    }
}
