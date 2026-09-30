//! Hybrid physics (`physics = "hybrid"`, see [`HybridSpec`](crate::scenario::HybridSpec)):
//! the handover from the kinematic model to the multibody model and its controller.
//!
//! A vehicle promoted at speed keeps its pose and velocities exactly
//! ([`Wheeled::resume_dynamics`]); for the ground controller to hold that speed without first
//! sagging, its speed integrator is preset to its steady value (bumpless transfer): its value
//! when holding that speed on flat asphalt (interpolated in a table calibrated once per group
//! when the scenario is compiled), plus the grade `g·sin θ`.

use autonomousim_control::ground::{GroundController, GroundEstimate, GroundSetpoint};
use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::{GroundStepEnv, Wheeled, WheeledDef};
use autonomousim_vehicles::multirotor::AirData;
use glam::DVec3;
use std::sync::Arc;

/// Calibration speeds (m/s) and the time run at each (s), the last `AVERAGE` of it averaged.
pub const SPEEDS: [f64; 7] = [3.0, 6.0, 9.0, 12.0, 16.0, 22.0, 30.0];
const RUN: f64 = 5.0;
const AVERAGE: f64 = 1.0;

/// The ground controller's steady speed integrator (m/s²) at the calibration [`SPEEDS`]
/// (linear in between, constant beyond); all zero when not calibrated.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CruiseTrim {
    pub trim: [f64; SPEEDS.len()],
}

impl CruiseTrim {
    /// Run vehicle `def` with `controller` straight at the calibration speeds on flat asphalt
    /// (physics step `dt`, gravity `g`). Zero for vehicles the controller does not drive through
    /// the speed loop (side drives, two-wheelers).
    pub fn calibrate(def: &Arc<WheeledDef>, controller: &GroundController, dt: f64, g: f64) -> Self {
        if !controller.has_steering() || controller.is_single_track() {
            return Self::default();
        }
        let terrain = FlatTerrain::new(0.0, MaterialId::ASPHALT);
        let materials = MaterialTable::standard();
        let env = GroundStepEnv {
            scene: StaticScene { terrain: &terrain, obstacles: &NoObstacles, materials: &materials },
            gravity: DVec3::new(0.0, 0.0, -g),
            air: AirData::default(),
        };
        let steps = (RUN / dt).round() as usize;
        let tail = (AVERAGE / dt).round() as usize;
        let trim = SPEEDS.map(|v| {
            let mut w = Wheeled::new(def.clone(), dt);
            let init = w.rest(DVec3::ZERO, 0.0, v);
            w.reset(&init);
            // As a promotion leaves it.
            w.resume_dynamics();
            let mut c = controller.clone();
            c.reset();
            let sp = GroundSetpoint::SpeedCurvature { speed: v, curvature: 0.0 };
            let mut sum = 0.0;
            for k in 0..steps {
                let input = c.update(&sp, &GroundEstimate::of(&w));
                if w.step(&input, &env).is_err() {
                    return 0.0;
                }
                if k + tail >= steps {
                    sum += c.speed_integral();
                }
            }
            sum / tail as f64
        });
        Self { trim }
    }

    /// Steady integrator at forward speed `speed` on a grade whose chassis x axis has vertical
    /// component `rise` (the sine of the pitch up), with gravity `g`.
    pub fn at(&self, speed: f64, rise: f64, g: f64) -> f64 {
        let v = speed.abs();
        let k = SPEEDS.iter().position(|&s| s >= v).unwrap_or(SPEEDS.len());
        let resistance = match k {
            0 => self.trim[0] * v / SPEEDS[0],
            k if k == SPEEDS.len() => self.trim[k - 1],
            k => {
                let t = (v - SPEEDS[k - 1]) / (SPEEDS[k] - SPEEDS[k - 1]);
                self.trim[k - 1] + t * (self.trim[k] - self.trim[k - 1])
            }
        };
        resistance.copysign(speed) + g * rise
    }
}
