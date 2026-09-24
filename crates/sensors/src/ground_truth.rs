//! Ground truth: the exact vehicle state plus height above the surface and clearance to the
//! nearest collidable geometry (for observations, rewards and logging).

use crate::latency::Stamped;
use crate::{BodyKinematics, SensorEnv, SensorError, Timing};
use autonomousim_core::time::Clock;
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroundTruthConfig {
    pub rate_hz: u32,
    /// Clearance is searched up to this distance (m; 0: not computed).
    pub clearance_max: f64,
}

impl Default for GroundTruthConfig {
    fn default() -> Self {
        Self { rate_hz: 50, clearance_max: 20.0 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GroundTruth {
    pub position: DVec3,
    pub attitude: DQuat,
    /// World and body-frame velocity (m/s).
    pub velocity: DVec3,
    pub velocity_body: DVec3,
    /// Body angular velocity (rad/s) and acceleration (rad/s²).
    pub rates: DVec3,
    pub ang_acc: DVec3,
    /// Specific force at the centre of mass (body frame, m/s²).
    pub specific_force: DVec3,
    /// Height above the ground or water surface directly below (m).
    pub agl: f64,
    /// Distance to the nearest terrain or solid obstacle (m), capped at `clearance_max`.
    pub clearance: f64,
}

#[derive(Clone, Debug)]
pub struct GroundTruthSensor {
    config: GroundTruthConfig,
    timing: Timing,
    current: Option<Stamped<GroundTruth>>,
}

impl GroundTruthSensor {
    pub fn new(config: GroundTruthConfig, clock: &Clock) -> Result<Self, SensorError> {
        if !(config.clearance_max >= 0.0 && config.clearance_max.is_finite()) {
            return Err(SensorError::InvalidConfig(format!("{config:?}")));
        }
        let timing = Timing::new("ground_truth", clock, config.rate_hz, 0.0)?;
        Ok(Self { config, timing, current: None })
    }

    pub fn config(&self) -> &GroundTruthConfig {
        &self.config
    }

    pub fn reset(&mut self) {
        self.current = None;
    }

    pub fn measure(&self, kin: &BodyKinematics, env: &SensorEnv) -> GroundTruth {
        let p = kin.position;
        let clearance = if self.config.clearance_max > 0.0 {
            env.world.clearance(p, self.config.clearance_max)
        } else {
            f64::INFINITY
        };
        GroundTruth {
            position: p,
            attitude: kin.attitude,
            velocity: kin.velocity,
            velocity_body: kin.attitude.inverse() * kin.velocity,
            rates: kin.rates,
            ang_acc: kin.ang_acc,
            specific_force: kin.specific_force,
            agl: p.z - env.world.surface_height(p.x, p.y),
            clearance,
        }
    }

    pub fn update(&mut self, tick: u64, time: f64, kin: &BodyKinematics, env: &SensorEnv) -> bool {
        if self.timing.is_due(tick) {
            self.current = Some(Stamped { tick, time, value: self.measure(kin, env) });
            true
        } else {
            false
        }
    }

    pub fn latest(&self) -> Option<&Stamped<GroundTruth>> {
        self.current.as_ref()
    }
}
