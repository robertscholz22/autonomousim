//! Scenarios: everything that defines a distribution of episodes. Rates, the map, the
//! environment and its randomisation, event thresholds, and the agent groups (vehicle,
//! controller, action mode, sensors, observations, spawn and goal sampling, parameter
//! randomisation). Loaded from TOML or JSON; every field has a default.
//!
//! ```toml
//! name = "hover"
//! physics_hz = 500
//! policy_hz = 50
//! map = { type = "testworld", kind = "forest_patch", size = 200.0, density = 150.0, seed = 1 }
//! # or a pool of generated maps, one drawn per episode:
//! # map = { type = "wild", seed = 0, count = 16, preset = "training", config = { size = 256.0 } }
//!
//! [randomize_environment]
//! wind_speed = [0.0, 5.0]
//!
//! [[groups]]
//! name = "quad"
//! vehicle = "cf2x"
//! action_mode = "ctbr"
//! spawn = { agl = [1.0, 3.0], tilt_deg = 10.0 }
//! goals = { kind = "random", distance = [0.0, 2.0] }
//! obs = [
//!     { term = "goal_rel_world", scale = 0.5, clip = 5.0 },
//!     { term = "rot6d" },
//!     { term = "lin_vel_world", scale = 0.5 },
//!     { term = "ang_vel_body", scale = 0.1 },
//!     { term = "last_action" },
//! ]
//!
//! [[groups]]
//! name = "cars"
//! vehicle = "sedan_like"   # wheeled: modes raw, vk (default), vw, per_wheel
//! action_mode = "vk"
//! ground_action_limits = { speed = 10.0 }
//! ```
//!
//! Angles in scenario files are in degrees (`*_deg`); everything else is SI.

use crate::SimError;
use crate::bay::BayGoals;
use crate::drive::{self, DrivableSpec, DriveGrid};
use crate::driver::DriverSpec;
use crate::lane::RouteGoals;
use crate::obs::{CompiledObs, ObsTerm, default_obs};
use autonomousim_control::fixedwing::{FixedWingActionLimits, FixedWingConfig};
use autonomousim_control::ground::{GroundActionLimits, GroundConfig};
use autonomousim_control::multirotor::{ActionLimits, ControllerConfig};
use autonomousim_control::rotorcraft::{HelicopterActionLimits, HelicopterConfig};
use autonomousim_control::tiltrotor::{TiltrotorActionLimits, TiltrotorConfig};
use autonomousim_control::{ActionMapping, AgentActionMode, Controller};
use autonomousim_core::geometry::{HitMask, StaticGeometry};
use autonomousim_core::material::MaterialId;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::from_yaw;
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::terrain::Terrain;
use autonomousim_core::time::Clock;
use autonomousim_procgen::MapCache;
use autonomousim_procgen::rural::{self, RuralConfig, RuralPreset};
use autonomousim_procgen::urban::{self, UrbanConfig, UrbanPreset};
use autonomousim_procgen::wild::{self, WildConfig, WildPreset};
use autonomousim_sensors::{Sensor, SensorConfig, SensorSpec};
use autonomousim_vehicles::fixedwing::FixedWingInput;
use autonomousim_vehicles::ground::{TrailerDef, Wheeled};
use autonomousim_vehicles::multirotor::{MotorInit, MultirotorScales};
use autonomousim_vehicles::presets;
use autonomousim_vehicles::rotorcraft::HelicopterInput;
use autonomousim_vehicles::tiltrotor::TiltrotorInput;
use autonomousim_vehicles::{Family, SharedDef, VehicleDef};
use autonomousim_world::environment::{EnvironmentConfig, Gust};
use autonomousim_world::testworlds;
use autonomousim_world::{MapHash, StaticWorld};
use glam::{DQuat, DVec2, DVec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// Default physics rates of worlds with and without ground vehicles (Hz).
pub const GROUND_PHYSICS_HZ: u32 = 1000;
pub const AERIAL_PHYSICS_HZ: u32 = 500;

/// Candidate positions tried before falling back to the best one seen.
const MAX_ATTEMPTS: usize = 200;

/// Height (m) above a landing zone up to which it must be clear (see `GoalSpec::landing_slope`).
const LANDING_COLUMN: f64 = 40.0;
/// Spawn candidates drawn from one tile of a tiled map before moving to another.
const TILE_BATCH: usize = 25;
/// Candidate centres of a spawn cluster.
const CLUSTER_ATTEMPTS: usize = 32;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Scenario {
    pub name: String,
    /// Physics rate; 0 (the default) picks 1000 Hz when any group drives a ground vehicle
    /// (tyre relaxation and stiff suspensions) and 500 Hz otherwise.
    pub physics_hz: u32,
    /// Rate at which actions are applied and observations produced.
    pub policy_hz: u32,
    /// Rate of the wind, turbulence and air-density updates.
    pub environment_hz: u32,
    pub map: MapSource,
    /// Nominal environment of every episode.
    pub environment: EnvironmentConfig,
    /// Per-episode changes to the environment; `None` keeps it fixed.
    pub randomize_environment: Option<EnvironmentRandomization>,
    pub events: EventConfig,
    pub groups: Vec<GroupSpec>,
}

impl Default for Scenario {
    fn default() -> Self {
        Self {
            name: "default".into(),
            physics_hz: 0,
            policy_hz: 50,
            environment_hz: 100,
            map: MapSource::default(),
            environment: EnvironmentConfig::default(),
            randomize_environment: None,
            events: EventConfig::default(),
            groups: vec![GroupSpec::default()],
        }
    }
}

impl Scenario {
    pub fn from_toml(s: &str) -> Result<Self, SimError> {
        toml::from_str(s).map_err(|e| SimError::Scenario(e.to_string()))
    }

    pub fn from_json(s: &str) -> Result<Self, SimError> {
        serde_json::from_str(s).map_err(|e| SimError::Scenario(e.to_string()))
    }

    /// Load a `.toml` or `.json` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SimError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        match path.extension().and_then(|e| e.to_str()) {
            Some("json") => Self::from_json(&text),
            _ => Self::from_toml(&text),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("scenario serialises")
    }

    /// Validate, build the maps and resolve the vehicles.
    pub fn compile(self) -> Result<CompiledScenario, SimError> {
        CompiledScenario::new(self)
    }
}

// ------------------------------------------------------------------------------------ maps

/// Where the maps of the episodes come from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MapSource {
    /// A hand-made test world.
    Testworld(Testworld),
    /// A pool of generated wild maps.
    Wild(WildMaps),
    /// A pool of generated rural maps (farmland with roads).
    Rural(RuralMaps),
    /// A pool of generated urban maps (a town or city with its street network).
    Urban(UrbanMaps),
}

/// `count` wild maps with the seeds `seed, seed + 1, …`; each episode draws one of them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WildMaps {
    pub seed: u64,
    pub count: u32,
    pub preset: WildPreset,
    /// Values that override the preset, in the layout of
    /// [`WildConfig`](autonomousim_procgen::WildConfig), e.g. `{ size = 256.0 }`.
    pub config: Option<serde_json::Value>,
    /// Keep generated maps in the user's map cache and load them from there.
    pub cache: bool,
}

impl Default for WildMaps {
    fn default() -> Self {
        Self { seed: 0, count: 1, preset: WildPreset::Training, config: None, cache: true }
    }
}

impl WildMaps {
    pub fn config(&self) -> Result<WildConfig, SimError> {
        WildConfig::from_preset(self.preset, self.config.as_ref()).map_err(|e| SimError::Scenario(format!("map: {e}")))
    }

    fn build(&self) -> Result<Vec<(StaticWorld, MapHash)>, SimError> {
        if !(1..=256).contains(&self.count) {
            return Err(SimError::Scenario("map: count must be in 1..=256".into()));
        }
        let config = self.config()?;
        let cache = if self.cache { MapCache::user() } else { None };
        (0..self.count as u64)
            .into_par_iter()
            .map(|k| {
                let seed = self.seed.wrapping_add(k);
                match &cache {
                    Some(c) => c.wild(&config, seed).map(|m| (m.world, m.hash)),
                    None => wild::generate(&config, seed).map(|(w, _)| {
                        let h = w.content_hash();
                        (w, h)
                    }),
                }
                .map_err(|e| SimError::Scenario(format!("map (seed {seed}): {e}")))
            })
            .collect()
    }
}

/// `count` rural maps with the seeds `seed, seed + 1, …`; each episode draws one of them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuralMaps {
    pub seed: u64,
    pub count: u32,
    pub preset: RuralPreset,
    /// Values that override the preset, in the layout of
    /// [`RuralConfig`](autonomousim_procgen::RuralConfig), e.g. `{ size = 256.0 }`.
    pub config: Option<serde_json::Value>,
    /// Keep generated maps in the user's map cache and load them from there.
    pub cache: bool,
}

impl Default for RuralMaps {
    fn default() -> Self {
        Self { seed: 0, count: 1, preset: RuralPreset::Training, config: None, cache: true }
    }
}

impl RuralMaps {
    pub fn config(&self) -> Result<RuralConfig, SimError> {
        RuralConfig::from_preset(self.preset, self.config.as_ref()).map_err(|e| SimError::Scenario(format!("map: {e}")))
    }

    fn build(&self) -> Result<Vec<(StaticWorld, MapHash)>, SimError> {
        if !(1..=256).contains(&self.count) {
            return Err(SimError::Scenario("map: count must be in 1..=256".into()));
        }
        let config = self.config()?;
        let cache = if self.cache { MapCache::user() } else { None };
        (0..self.count as u64)
            .into_par_iter()
            .map(|k| {
                let seed = self.seed.wrapping_add(k);
                match &cache {
                    Some(c) => c.rural(&config, seed).map(|m| (m.world, m.hash)),
                    None => rural::generate(&config, seed).map(|(w, _)| {
                        let h = w.content_hash();
                        (w, h)
                    }),
                }
                .map_err(|e| SimError::Scenario(format!("map (seed {seed}): {e}")))
            })
            .collect()
    }
}

/// `count` urban maps with the seeds `seed, seed + 1, …`; each episode draws one of them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UrbanMaps {
    pub seed: u64,
    pub count: u32,
    pub preset: UrbanPreset,
    /// Values that override the preset, in the layout of
    /// [`UrbanConfig`](autonomousim_procgen::UrbanConfig), e.g. `{ size = 256.0 }`.
    pub config: Option<serde_json::Value>,
    /// Keep generated maps in the user's map cache and load them from there.
    pub cache: bool,
}

impl Default for UrbanMaps {
    fn default() -> Self {
        Self { seed: 0, count: 1, preset: UrbanPreset::Training, config: None, cache: true }
    }
}

impl UrbanMaps {
    pub fn config(&self) -> Result<UrbanConfig, SimError> {
        UrbanConfig::from_preset(self.preset, self.config.as_ref()).map_err(|e| SimError::Scenario(format!("map: {e}")))
    }

    fn build(&self) -> Result<Vec<(StaticWorld, MapHash)>, SimError> {
        if !(1..=256).contains(&self.count) {
            return Err(SimError::Scenario("map: count must be in 1..=256".into()));
        }
        let config = self.config()?;
        let cache = if self.cache { MapCache::user() } else { None };
        (0..self.count as u64)
            .into_par_iter()
            .map(|k| {
                let seed = self.seed.wrapping_add(k);
                match &cache {
                    Some(c) => c.urban(&config, seed).map(|m| (m.world, m.hash)),
                    None => urban::generate(&config, seed).map(|(w, _)| {
                        let h = w.content_hash();
                        (w, h)
                    }),
                }
                .map_err(|e| SimError::Scenario(format!("map (seed {seed}): {e}")))
            })
            .collect()
    }
}

impl Default for MapSource {
    fn default() -> Self {
        MapSource::Testworld(Testworld::Flat { size: 200.0 })
    }
}

impl MapSource {
    /// Build the maps with their content hashes.
    pub fn build(&self) -> Result<Vec<(StaticWorld, MapHash)>, SimError> {
        match self {
            MapSource::Testworld(t) => {
                let w = t.build()?;
                let h = w.content_hash();
                Ok(vec![(w, h)])
            }
            MapSource::Wild(w) => w.build(),
            MapSource::Rural(r) => r.build(),
            MapSource::Urban(u) => u.build(),
        }
    }

    /// Seed of the first generated map (`None` for test worlds).
    pub fn seed(&self) -> Option<u64> {
        match self {
            MapSource::Testworld(_) => None,
            MapSource::Wild(w) => Some(w.seed),
            MapSource::Rural(r) => Some(r.seed),
            MapSource::Urban(u) => Some(u.seed),
        }
    }

    /// Generate `count` maps from `seed` (test worlds are left as they are).
    pub fn set_pool(&mut self, seed: u64, count: u32, cache: bool) {
        match self {
            MapSource::Testworld(_) => {}
            MapSource::Wild(w) => (w.seed, w.count, w.cache) = (seed, count, cache),
            MapSource::Rural(r) => (r.seed, r.count, r.cache) = (seed, count, cache),
            MapSource::Urban(u) => (u.seed, u.count, u.cache) = (seed, count, cache),
        }
    }

    /// Generate from `seed`, keeping the pool size.
    pub fn set_seed(&mut self, seed: u64) {
        match self {
            MapSource::Testworld(_) => {}
            MapSource::Wild(w) => w.seed = seed,
            MapSource::Rural(r) => r.seed = seed,
            MapSource::Urban(u) => u.seed = seed,
        }
    }
}

/// The worlds of [`autonomousim_world::testworlds`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Testworld {
    /// Grass plane at `z = 0`, `size` m square.
    Flat {
        size: f64,
    },
    /// Plane rising along +x.
    Incline {
        size: f64,
        angle_deg: f64,
        #[serde(default)]
        material: MaterialId,
    },
    SineHills {
        size: f64,
        amplitude: f64,
        wavelength: f64,
    },
    /// Bowl with a lake in the middle.
    Lake {
        size: f64,
        depth: f64,
        water_level: f64,
    },
    SingleTree,
    /// `count × count` pillars.
    Pillars {
        count: usize,
        spacing: f64,
        radius: f64,
        height: f64,
    },
    WalledArena {
        half_size: f64,
        height: f64,
    },
    /// Hilly forest with `density` trees per hectare.
    ForestPatch {
        size: f64,
        density: f64,
        seed: u64,
    },
}

impl Testworld {
    fn build(&self) -> Result<StaticWorld, SimError> {
        let positive = |x: f64, what: &str| {
            if x > 0.0 && x.is_finite() { Ok(x) } else { Err(SimError::Scenario(format!("map {what} must be > 0"))) }
        };
        Ok(match *self {
            Testworld::Flat { size } => testworlds::flat(positive(size, "size")?),
            Testworld::Incline { size, angle_deg, material } => {
                testworlds::incline(positive(size, "size")?, angle_deg.to_radians(), material)
            }
            Testworld::SineHills { size, amplitude, wavelength } => {
                testworlds::sine_hills(positive(size, "size")?, amplitude, positive(wavelength, "wavelength")?)
            }
            Testworld::Lake { size, depth, water_level } => {
                testworlds::lake(positive(size, "size")?, depth, water_level)
            }
            Testworld::SingleTree => testworlds::single_tree(),
            Testworld::Pillars { count, spacing, radius, height } => {
                testworlds::pillars(count, spacing, radius, height)
            }
            Testworld::WalledArena { half_size, height } => {
                testworlds::walled_arena(positive(half_size, "size")?, height)
            }
            Testworld::ForestPatch { size, density, seed } => {
                testworlds::forest_patch(positive(size, "size")?, density.max(0.0), seed)
            }
        })
    }
}

// ----------------------------------------------------------------------------- environment

/// Per-episode randomisation of the environment. Ranges are sampled uniformly; `None` keeps
/// the nominal value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnvironmentRandomization {
    /// Mean wind speed at the reference height (m/s); the direction is uniform.
    pub wind_speed: Option<[f64; 2]>,
    /// Turbulence intensity `W20` (m/s; 7.7 light, 15.4 moderate).
    pub turbulence_w20: Option<[f64; 2]>,
    /// Expected gusts per minute (Poisson) within the first `gust_horizon` seconds, each with
    /// a uniform horizontal direction, peak speed and duration.
    pub gust_rate: f64,
    pub gust_speed: [f64; 2],
    pub gust_duration: [f64; 2],
    pub gust_horizon: f64,
    /// ISA temperature offset (K).
    pub temperature_offset: Option<[f64; 2]>,
    /// Sea-level pressure (Pa).
    pub sea_level_pressure: Option<[f64; 2]>,
}

impl Default for EnvironmentRandomization {
    fn default() -> Self {
        Self {
            wind_speed: None,
            turbulence_w20: None,
            gust_rate: 0.0,
            gust_speed: [2.0, 6.0],
            gust_duration: [1.0, 4.0],
            gust_horizon: 60.0,
            temperature_offset: None,
            sea_level_pressure: None,
        }
    }
}

impl EnvironmentRandomization {
    pub fn validate(&self) -> Result<(), SimError> {
        let ranges = [
            self.wind_speed,
            self.turbulence_w20,
            Some(self.gust_speed),
            Some(self.gust_duration),
            self.temperature_offset,
            self.sea_level_pressure,
        ];
        let ok = ranges.iter().flatten().all(|r| valid_range(*r))
            && self.gust_rate >= 0.0
            && self.gust_horizon >= 0.0
            && self.gust_duration[0] > 0.0
            && self.wind_speed.is_none_or(|r| r[0] >= 0.0)
            && self.turbulence_w20.is_none_or(|r| r[0] >= 0.0)
            && self.sea_level_pressure.is_none_or(|r| r[0] > 0.0);
        if ok { Ok(()) } else { Err(SimError::Scenario(format!("invalid environment randomisation {self:?}"))) }
    }

    pub fn apply(&self, env: &mut EnvironmentConfig, rng: &mut SimRng) {
        // Draw every value unconditionally so that the streams do not depend on which fields
        // are set.
        let heading = rng.range(-std::f64::consts::PI, std::f64::consts::PI);
        let speed = sample(rng, self.wind_speed.unwrap_or([0.0; 2]));
        if self.wind_speed.is_some() {
            env.wind.mean = DVec2::from_angle(heading) * speed;
        }
        let w20 = sample(rng, self.turbulence_w20.unwrap_or([0.0; 2]));
        if self.turbulence_w20.is_some() {
            env.wind.turbulence_w20 = w20;
        }
        let dt = sample(rng, self.temperature_offset.unwrap_or([0.0; 2]));
        if self.temperature_offset.is_some() {
            env.atmosphere.temperature_offset = dt;
        }
        let p = sample(rng, self.sea_level_pressure.unwrap_or([1.0; 2]));
        if self.sea_level_pressure.is_some() {
            env.atmosphere.sea_level_pressure = p;
        }
        // Poisson process of gust start times over the horizon.
        if self.gust_rate > 0.0 {
            let rate = self.gust_rate / 60.0;
            let mut t = 0.0;
            loop {
                t += -(1.0 - rng.uniform()).ln() / rate;
                if t >= self.gust_horizon {
                    break;
                }
                let dir = DVec2::from_angle(rng.range(-std::f64::consts::PI, std::f64::consts::PI));
                let amplitude = (dir * sample(rng, self.gust_speed)).extend(0.0);
                env.wind.gusts.push(Gust { start: t, duration: sample(rng, self.gust_duration), amplitude });
            }
        }
    }
}

// ---------------------------------------------------------------------------------- events

/// Thresholds of the event bits (see [`Events`](crate::Events)).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EventConfig {
    /// Approach speed (m/s) above which any contact with terrain or a solid obstacle is a
    /// crash, even on the landing gear. Contacts of the airframe or rotors always are.
    pub crash_speed: f64,
    /// Speed (m/s) and body rate (rad/s) below which a vehicle on its gear counts as landed.
    pub landed_speed: f64,
    pub landed_rate: f64,
    /// Out of bounds closer than this to the map edge (m).
    pub bounds_margin: f64,
    /// Out of bounds higher than this above the surface (m).
    pub max_agl: Option<f64>,
    /// Out of bounds above this altitude (map z, m).
    pub ceiling: Option<f64>,
    /// Thresholds that only apply to ground vehicles.
    #[serde(skip_serializing_if = "is_default")]
    pub ground: GroundEventConfig,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            crash_speed: 2.0,
            landed_speed: 0.1,
            landed_rate: 0.3,
            bounds_margin: 0.0,
            max_agl: None,
            ceiling: None,
            ground: GroundEventConfig::default(),
        }
    }
}

/// Event thresholds of ground vehicles (`[events.ground]`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroundEventConfig {
    /// Rollover when the up axis of any unit tilts more than this from the vertical (degrees).
    pub rollover_deg: f64,
    /// Jackknife when a trailer's (or dolly's) yaw relative to the unit ahead exceeds this
    /// (degrees).
    pub jackknife_deg: f64,
    /// Stuck after moving less than `stuck_distance` (m) for `stuck_time` seconds (0: never).
    pub stuck_time: f64,
    pub stuck_distance: f64,
}

impl Default for GroundEventConfig {
    fn default() -> Self {
        Self { rollover_deg: 60.0, jackknife_deg: 70.0, stuck_time: 5.0, stuck_distance: 0.5 }
    }
}

// ---------------------------------------------------------------------------------- groups

/// A vehicle definition: a built-in preset name, a path to a TOML file, or inline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VehicleRef {
    Name(String),
    Inline(Box<VehicleDef>),
}

impl Default for VehicleRef {
    fn default() -> Self {
        VehicleRef::Name("cf2x".into())
    }
}

impl VehicleRef {
    pub fn resolve(&self) -> Result<VehicleDef, SimError> {
        Ok(match self {
            VehicleRef::Name(n) if n.ends_with(".toml") || n.contains('/') => VehicleDef::load(n)?,
            VehicleRef::Name(n) => presets::get(n)?,
            VehicleRef::Inline(d) => (**d).clone(),
        })
    }
}

/// Agents that share a vehicle type, action mode and observation layout. Python sees each
/// group as arrays of shape `[num_envs, count, dim]`.
///
/// `controller`, `action_limits` and `randomize` configure multirotors; `ground_controller`,
/// `ground_action_limits` and `drivable` wheeled vehicles; `fixed_wing_controller` and
/// `fixed_wing_action_limits` fixed-wing aircraft; `helicopter_controller` and
/// `helicopter_action_limits` helicopters; `tiltrotor_controller` and
/// `tiltrotor_action_limits` tiltrotors. Setting those of another family is an error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroupSpec {
    pub name: String,
    pub count: usize,
    pub vehicle: VehicleRef,
    /// Trailers coupled behind a ground vehicle, first to last: built-in names
    /// (`presets::trailer_names`) or paths to trailer TOML files.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub trailers: Vec<String>,
    pub controller: ControllerConfig,
    /// A mode of the vehicle's family (`ctbr`, `velocity`, …; `vk`, `vw`, …). `None`: the
    /// family's default, filled in when the scenario is compiled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_mode: Option<AgentActionMode>,
    pub action_limits: ActionLimits,
    #[serde(skip_serializing_if = "is_default")]
    pub ground_controller: GroundConfig,
    #[serde(skip_serializing_if = "is_default")]
    pub ground_action_limits: GroundActionLimits,
    #[serde(skip_serializing_if = "is_default")]
    pub fixed_wing_controller: FixedWingConfig,
    #[serde(skip_serializing_if = "is_default")]
    pub fixed_wing_action_limits: FixedWingActionLimits,
    #[serde(skip_serializing_if = "is_default")]
    pub helicopter_controller: HelicopterConfig,
    #[serde(skip_serializing_if = "is_default")]
    pub helicopter_action_limits: HelicopterActionLimits,
    #[serde(skip_serializing_if = "is_default")]
    pub tiltrotor_controller: TiltrotorConfig,
    #[serde(skip_serializing_if = "is_default")]
    pub tiltrotor_action_limits: TiltrotorActionLimits,
    /// Terrain that ground vehicles can drive over: spawns and goals only use drivable cells,
    /// and goals must be reachable from the spawn.
    #[serde(skip_serializing_if = "is_default")]
    pub drivable: DrivableSpec,
    pub sensors: Vec<SensorSpec>,
    /// Observation terms in order; empty: the default hover observation.
    pub obs: Vec<ObsTerm>,
    pub spawn: SpawnSpec,
    pub goals: GoalSpec,
    pub randomize: VehicleRandomization,
    /// Freeze an agent after a terminal event until the next reset (it then no longer moves,
    /// collides or appears in other agents' sensors).
    pub disable_on_terminal: bool,
    /// A scripted driver ([`DriverSpec`]): the group's agents are driven by it and take no
    /// actions (a scripted group; its action arrays are ignored and may be left out).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver: Option<DriverSpec>,
}

impl Default for GroupSpec {
    fn default() -> Self {
        Self {
            name: "agents".into(),
            count: 1,
            vehicle: VehicleRef::default(),
            trailers: Vec::new(),
            controller: ControllerConfig::default(),
            action_mode: None,
            action_limits: ActionLimits::default(),
            ground_controller: GroundConfig::default(),
            ground_action_limits: GroundActionLimits::default(),
            fixed_wing_controller: FixedWingConfig::default(),
            fixed_wing_action_limits: FixedWingActionLimits::default(),
            helicopter_controller: HelicopterConfig::default(),
            helicopter_action_limits: HelicopterActionLimits::default(),
            tiltrotor_controller: TiltrotorConfig::default(),
            tiltrotor_action_limits: TiltrotorActionLimits::default(),
            drivable: DrivableSpec::default(),
            sensors: Vec::new(),
            obs: Vec::new(),
            spawn: SpawnSpec::default(),
            goals: GoalSpec::default(),
            randomize: VehicleRandomization::default(),
            disable_on_terminal: true,
            driver: None,
        }
    }
}

impl GroupSpec {
    /// The vehicle with its trailers coupled.
    pub fn resolve_vehicle(&self) -> Result<VehicleDef, SimError> {
        let def = self.vehicle.resolve()?;
        if self.trailers.is_empty() {
            return Ok(def);
        }
        let VehicleDef::Wheeled(towing) = def else {
            return Err(SimError::Scenario("only ground vehicles tow trailers".into()));
        };
        let trailers = self
            .trailers
            .iter()
            .map(|t| if t.ends_with(".toml") || t.contains('/') { TrailerDef::load(t) } else { presets::trailer(t) })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(VehicleDef::Wheeled(towing.with_trailers(&trailers)?))
    }
}

fn is_default<T: Default + PartialEq>(x: &T) -> bool {
    *x == T::default()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpawnMotors {
    /// Rotors at the nominal hover speed.
    #[default]
    Hover,
    Idle,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpawnLayout {
    /// Independent uniform positions in the region, in free space and apart from each other.
    #[default]
    Random,
    /// Square grid centred in the region (no free-space check), at the middle of the AGL range.
    Grid { spacing: f64 },
}

/// Where and how agents start an episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpawnSpec {
    pub layout: SpawnLayout,
    /// Horizontal region `[min, max]` (m); default: the map shrunk by `margin`.
    pub region: Option<[DVec2; 2]>,
    pub margin: f64,
    /// Side of a square (m), placed at random inside the region each episode, that the
    /// group spawns in (a cluster); default: the whole region. The centre is the most open
    /// of up to 32 random candidates (distance to the nearest solid obstacle at the middle
    /// spawn height, enough at half the side).
    pub cluster: Option<f64>,
    /// Height of the centre of mass above the ground or water surface (m).
    pub agl: [f64; 2],
    /// Start resting on the ground (motors idle unless set otherwise; aircraft on their gear
    /// with the brakes set); ignores `agl`. Ground vehicles always do.
    pub on_ground: bool,
    /// Fixed-wing aircraft, helicopters and tiltrotors in the air: airspeed range (m/s); they
    /// start trimmed for straight and level flight at a sampled airspeed and the spawn heading,
    /// in the steady wind (tiltrotors with the rotors forward from 1.2 times the stall speed,
    /// else up). Default: 1.5 times the stall speed at the spawn altitude for aircraft, hover
    /// for helicopters and tiltrotors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub airspeed: Option<[f64; 2]>,
    /// Start in a lane of a random road of the map, facing along it (along the route with
    /// `route` goals); ignores `layout`, `region`, `cluster` and `yaw_deg`. Needs a map with
    /// roads and a vehicle on the ground.
    pub on_road: bool,
    /// Free radius around the vehicle (m).
    pub clearance: f64,
    /// Smallest distance between agents (m).
    pub min_separation: f64,
    /// Do not start above water.
    pub avoid_water: bool,
    /// Largest initial speed (m/s), uniform direction.
    pub speed: f64,
    /// Largest initial tilt (degrees), uniform axis.
    pub tilt_deg: f64,
    /// Heading range (degrees).
    pub yaw_deg: [f64; 2],
    /// Largest initial body rate (rad/s), uniform axis.
    pub rates: f64,
    pub motors: SpawnMotors,
    /// Start around the agents of an earlier group (e.g. a drone above a car); replaces
    /// `region` and `cluster`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub near: Option<SpawnNear>,
}

/// Spawns around the agents of another group: agent `k` of the group starts in a square of
/// half side `offset` (m) around agent `k` (cycling) of `group`, horizontally, inside the
/// map's spawn region; the height, clearance and separation settings apply as usual. The
/// other group must come earlier in the scenario (it is placed first).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpawnNear {
    pub group: String,
    #[serde(default)]
    pub offset: f64,
}

impl Default for SpawnSpec {
    fn default() -> Self {
        Self {
            layout: SpawnLayout::Random,
            region: None,
            margin: 10.0,
            cluster: None,
            agl: [1.0, 3.0],
            on_ground: false,
            airspeed: None,
            on_road: false,
            clearance: 1.0,
            min_separation: 1.0,
            avoid_water: true,
            speed: 0.0,
            tilt_deg: 0.0,
            yaw_deg: [-180.0, 180.0],
            rates: 0.0,
            motors: SpawnMotors::Hover,
            near: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalKind {
    /// One goal at the spawn pose (hold position).
    #[default]
    Spawn,
    /// `count` waypoints, each sampled from the previous one (the first from the spawn).
    Random,
    /// One goal per agent: slot `k` (the agent's index in its group) of a formation of the
    /// whole group, centred on the centroid of the group's spawns and turned by a heading
    /// from `yaw_deg` (also every slot's goal heading); `agl` above the surface at each slot.
    Formation,
    /// A route along the roads to a destination (see [`RouteGoals`]) whose length lies in
    /// `distance`, with goals every `route.step` m along its lane (`count` is ignored). The
    /// agent keeps the route for the `road` and `route` observation terms. Needs a map with
    /// roads.
    Route,
    /// One goal: a bay at the far side of a farm yard for the tail of the vehicle's last unit
    /// (see [`BayGoals`]); the vehicle spawns in the yard ahead of it, facing the yard's road
    /// (the spawn spec's position settings are ignored). Ground vehicles on maps with farm
    /// yards.
    Bay,
    /// One goal: the pad of a farm yard `distance` (horizontally) from the pad of another
    /// where the vehicle spawns, `agl` above the surface (the closest yard to the range when
    /// none lies in it; the spawn spec's position settings are ignored, its height settings
    /// and `on_ground` apply). Aerial vehicles on maps with at least two farm yards.
    Yard,
}

/// Slot layout of `GoalKind::Formation`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormationShape {
    /// Rows of `ceil(sqrt(count))` slots, `spacing` apart.
    #[default]
    Grid,
    /// A ring with neighbouring slots `spacing` apart.
    Circle,
}

/// Goals (waypoints) of each agent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GoalSpec {
    pub kind: GoalKind,
    pub count: usize,
    /// Horizontal distance from the previous point (m).
    pub distance: [f64; 2],
    /// Height above the surface (m).
    pub agl: [f64; 2],
    pub clearance: f64,
    /// Heading range (degrees).
    pub yaw_deg: [f64; 2],
    /// Distance kept from the map edge (m).
    pub margin: f64,
    /// Reach radius (m): an agent whose centre comes this close to its current goal moves on
    /// to the next one and reports `GOAL_REACHED` (`FINISHED` after the last). 0: goals are
    /// only advanced explicitly (`WorldInstance::advance_goal`).
    pub radius: f64,
    /// Layout of `GoalKind::Formation`.
    pub formation: FormationShape,
    /// Distance between neighbouring formation slots (m).
    pub spacing: f64,
    /// Settings of `GoalKind::Route`.
    pub route: RouteGoals,
    /// Settings of `GoalKind::Bay`.
    #[serde(skip_serializing_if = "is_default")]
    pub bay: BayGoals,
    /// `Random` goals of ground vehicles: plan the cheapest drivable path from the spawn
    /// through the goals ([`DriveGrid::legs`](crate::drive::DriveGrid::legs), weighted by
    /// `drivable.resistance_cost`). The agent follows the leg to its current goal as its route,
    /// for the `road` and `route` observation terms. It starts facing along the path, turned
    /// by the sampled `spawn.yaw_deg`.
    #[serde(skip_serializing_if = "is_default")]
    pub path: bool,
    /// `Random` goals of ground vehicles: away from the roads (goals on a road are drawn only
    /// when no draw lands off them).
    #[serde(skip_serializing_if = "is_default")]
    pub off_road: bool,
    /// `Random` goals of aerial vehicles: largest height difference to the previous point per
    /// metre of horizontal distance (a climb or descent the vehicle can fly straight); steeper
    /// draws are used only when no draw meets it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grade: Option<f64>,
    /// `Random` goals of aerial vehicles: landing zones, `agl` above the surface where no
    /// obstacle or foliage comes within `clearance` of the vertical from 1 to 40 m above the
    /// ground (the descent), no
    /// water lies within it and the terrain within it rises or falls at most this much per
    /// metre from the centre. Draws that fail rank below all that pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub landing_slope: Option<f64>,
    /// Radius of a landing pad drawn on the ground under the agent's current goal (m), seen
    /// by cameras as [`SemanticClass::Marker`](autonomousim_render::SemanticClass::Marker);
    /// 0: none. The pad is only visual: it has no collider.
    #[serde(skip_serializing_if = "is_default")]
    pub pad: f64,
}

impl Default for GoalSpec {
    fn default() -> Self {
        Self {
            kind: GoalKind::Spawn,
            count: 1,
            distance: [0.0, 5.0],
            agl: [1.0, 3.0],
            clearance: 1.0,
            yaw_deg: [-180.0, 180.0],
            margin: 10.0,
            radius: 0.0,
            formation: FormationShape::Grid,
            spacing: 2.0,
            route: RouteGoals::default(),
            bay: BayGoals::default(),
            path: false,
            off_road: false,
            grade: None,
            landing_slope: None,
            pad: 0.0,
        }
    }
}

/// A waypoint: position and heading.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Goal {
    pub position: DVec3,
    pub yaw: f64,
}

/// Relative spreads of the vehicle parameters: each scale is uniform in `[1 − s, 1 + s]`
/// (per axis for the inertia, per rotor for the thrust coefficient). The controller keeps the
/// nominal values.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VehicleRandomization {
    pub mass: f64,
    pub inertia: f64,
    pub k_thrust: f64,
    pub k_torque: f64,
    pub motor_tau: f64,
    pub body_drag: f64,
    pub rotor_drag: f64,
}

impl VehicleRandomization {
    fn validate(&self) -> Result<(), SimError> {
        let s =
            [self.mass, self.inertia, self.k_thrust, self.k_torque, self.motor_tau, self.body_drag, self.rotor_drag];
        if s.iter().all(|s| (0.0..1.0).contains(s)) {
            Ok(())
        } else {
            Err(SimError::Scenario(format!("randomisation spreads must be in [0, 1): {self:?}")))
        }
    }

    pub fn sample(&self, num_rotors: usize, rng: &mut SimRng) -> MultirotorScales {
        let mut f = |s: f64| 1.0 + s * rng.range(-1.0, 1.0);
        MultirotorScales {
            mass: f(self.mass),
            inertia: DVec3::new(f(self.inertia), f(self.inertia), f(self.inertia)),
            k_thrust: (0..num_rotors).map(|_| f(self.k_thrust)).collect(),
            k_torque: f(self.k_torque),
            motor_tau: f(self.motor_tau),
            body_drag: f(self.body_drag),
            rotor_drag: f(self.rotor_drag),
        }
    }
}

// ------------------------------------------------------------------------------- compiled

/// A validated scenario with its maps built and vehicles resolved, shared by `Arc` between
/// the world instances of a batch. `spec` has the defaults that depend on the vehicle filled
/// in (the action mode; ground vehicles start on the ground).
#[derive(Debug)]
pub struct CompiledScenario {
    pub spec: Scenario,
    /// Physics clock at tick 0.
    pub clock: Clock,
    /// Physics steps per policy step.
    pub decimation: u32,
    /// Physics steps per environment update.
    pub environment_divider: u32,
    /// The map pool (one map is drawn per episode) and the content hashes of its maps.
    pub maps: Vec<Arc<StaticWorld>>,
    pub map_hashes: Vec<MapHash>,
    /// The maps episodes are drawn from: all of them, or with `bay` goals those with farm
    /// yards.
    pub episode_maps: Vec<usize>,
    pub groups: Vec<CompiledGroup>,
}

#[derive(Debug)]
pub struct CompiledGroup {
    /// The group as specified, with the action mode filled in.
    pub spec: GroupSpec,
    pub def: SharedDef,
    /// Controller at its initial state; agents start from a clone.
    pub controller: Controller,
    pub action_map: ActionMapping,
    pub obs: CompiledObs,
    /// Depth of the lowest collider point below the centre of mass (m).
    pub bottom: f64,
    /// Radius of the sphere about the centre of mass that contains all colliders (m).
    pub radius: f64,
    /// Id of the group's first agent (agents are numbered group by group).
    pub first_agent: usize,
    /// Ground vehicles and aircraft: the rest pose on flat ground at the origin (heading +x);
    /// ground vehicles: half the vehicle's width (m), and where it can drive on each map of
    /// the pool.
    pub rest: Pose,
    pub half_width: f64,
    pub drive: Vec<Arc<DriveGrid>>,
}

impl CompiledGroup {
    pub fn act_dim(&self) -> usize {
        self.action_map.dim()
    }

    /// Driven by a scripted driver (takes no actions).
    pub fn scripted(&self) -> bool {
        self.spec.driver.is_some()
    }

    pub fn obs_dim(&self) -> usize {
        self.obs.dim()
    }

    pub fn family(&self) -> Family {
        self.def.family()
    }

    pub fn action_mode(&self) -> AgentActionMode {
        self.action_map.mode()
    }
}

impl CompiledScenario {
    fn new(mut spec: Scenario) -> Result<Self, SimError> {
        if spec.groups.is_empty() {
            return Err(SimError::Scenario("a scenario needs at least one agent group".into()));
        }
        let defs = spec
            .groups
            .iter()
            .map(|g| {
                g.resolve_vehicle()
                    .map(SharedDef::from)
                    .map_err(|e| SimError::Scenario(format!("group {:?}: {e}", g.name)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if spec.physics_hz == 0 {
            let ground = defs.iter().any(|d| d.family() == Family::Wheeled);
            spec.physics_hz = if ground { GROUND_PHYSICS_HZ } else { AERIAL_PHYSICS_HZ };
        }
        let clock = Clock::new(spec.physics_hz);
        let decimation = clock.divider("policy", spec.policy_hz)?;
        let environment_divider = clock.divider("environment", spec.environment_hz)?;
        if let Some(r) = &spec.randomize_environment {
            r.validate()?;
        }
        let e = &spec.events;
        if !(e.crash_speed > 0.0
            && e.landed_speed >= 0.0
            && e.landed_rate >= 0.0
            && e.bounds_margin >= 0.0
            && e.ground.rollover_deg > 0.0
            && e.ground.rollover_deg <= 180.0
            && e.ground.jackknife_deg > 0.0
            && e.ground.stuck_time >= 0.0
            && e.ground.stuck_distance >= 0.0)
        {
            return Err(SimError::Scenario(format!("invalid event thresholds {e:?}")));
        }
        let (maps, map_hashes): (Vec<_>, Vec<_>) = spec.map.build()?.into_iter().map(|(w, h)| (Arc::new(w), h)).unzip();
        let mut groups = Vec::with_capacity(spec.groups.len());
        let mut first_agent = 0;
        for (gi, (g, def)) in spec.groups.iter().zip(defs).enumerate() {
            if g.count == 0 {
                return Err(SimError::Scenario(format!("group {:?} has no agents", g.name)));
            }
            if spec.groups[..gi].iter().any(|o| o.name == g.name) {
                return Err(SimError::Scenario(format!("duplicate group name {:?}", g.name)));
            }
            if let Some(near) = &g.spawn.near {
                let earlier = spec.groups[..gi].iter().any(|o| o.name == near.group);
                if !earlier || !near.offset.is_finite() || near.offset < 0.0 {
                    return Err(SimError::Scenario(format!(
                        "group {:?}: `spawn.near` needs an earlier group and an offset ≥ 0, got {near:?}",
                        g.name
                    )));
                }
                if g.spawn.on_road || matches!(g.goals.kind, GoalKind::Bay | GoalKind::Yard) {
                    return Err(SimError::Scenario(format!(
                        "group {:?}: `spawn.near` does not go with `on_road` spawns or `bay` and `yard` goals",
                        g.name
                    )));
                }
            }
            groups.push(CompiledGroup::new(g.clone(), def, &clock, decimation, first_agent)?);
            first_agent += g.count;
        }
        for g in &groups {
            let roads = g.spec.spawn.on_road || g.spec.goals.kind == GoalKind::Route;
            if roads && let Some(k) = maps.iter().position(|m| m.roads().is_empty()) {
                return Err(SimError::Scenario(format!(
                    "group {:?}: `on_road` spawns and `route` goals need roads, map {k} has none",
                    g.spec.name
                )));
            }
        }
        let mut episode_maps: Vec<usize> = (0..maps.len()).collect();
        if let Some(g) = groups.iter().find(|g| g.spec.goals.kind == GoalKind::Bay) {
            episode_maps.retain(|&k| !crate::bay::yards(&maps[k]).is_empty());
            if episode_maps.is_empty() {
                return Err(SimError::Scenario(format!(
                    "group {:?}: `bay` goals need farm yards, no map has any",
                    g.spec.name
                )));
            }
        }
        if let Some(g) = groups.iter().find(|g| g.spec.goals.kind == GoalKind::Yard) {
            episode_maps.retain(|&k| crate::bay::yards(&maps[k]).len() >= 2);
            if episode_maps.is_empty() {
                return Err(SimError::Scenario(format!(
                    "group {:?}: `yard` goals need two farm yards, no map has them",
                    g.spec.name
                )));
            }
        }
        if let Some(g) = groups.iter().find(|g| g.family() == Family::Wheeled)
            && maps.iter().any(|m| m.is_tiled())
        {
            return Err(SimError::Scenario(format!(
                "group {:?}: ground vehicles need drive grids, which tiled (large) maps do not have",
                g.spec.name
            )));
        }
        build_drive_grids(&mut groups, &maps);
        // Defaults that depend on the vehicle, filled in.
        for (s, g) in spec.groups.iter_mut().zip(&groups) {
            s.clone_from(&g.spec);
        }
        Ok(Self { spec, clock, decimation, environment_divider, maps, map_hashes, episode_maps, groups })
    }

    pub fn num_agents(&self) -> usize {
        self.groups.iter().map(|g| g.spec.count).sum()
    }

    pub fn dt(&self) -> f64 {
        self.clock.dt()
    }

    /// Duration of a policy step (s).
    pub fn policy_dt(&self) -> f64 {
        self.clock.dt() * f64::from(self.decimation)
    }

    /// Whether any agent carries sensors.
    pub fn has_sensors(&self) -> bool {
        self.groups.iter().any(|g| !g.spec.sensors.is_empty())
    }

    pub fn group_index(&self, name: &str) -> Option<usize> {
        self.groups.iter().position(|g| g.spec.name == name)
    }
}

impl CompiledGroup {
    fn new(
        mut spec: GroupSpec,
        def: SharedDef,
        clock: &Clock,
        decimation: u32,
        first_agent: usize,
    ) -> Result<Self, SimError> {
        let family = def.family();
        let name = spec.name.clone();
        let fail = move |what: String| SimError::Scenario(format!("group {name:?}: {what}"));
        let multirotor = [
            ("controller", !is_default(&spec.controller)),
            ("action_limits", !is_default(&spec.action_limits)),
            ("randomize", !is_default(&spec.randomize)),
        ];
        let ground = [
            ("ground_controller", !is_default(&spec.ground_controller)),
            ("ground_action_limits", !is_default(&spec.ground_action_limits)),
            ("drivable", !is_default(&spec.drivable)),
        ];
        let fixed_wing = [
            ("fixed_wing_controller", !is_default(&spec.fixed_wing_controller)),
            ("fixed_wing_action_limits", !is_default(&spec.fixed_wing_action_limits)),
        ];
        let helicopter = [
            ("helicopter_controller", !is_default(&spec.helicopter_controller)),
            ("helicopter_action_limits", !is_default(&spec.helicopter_action_limits)),
        ];
        let tiltrotor = [
            ("tiltrotor_controller", !is_default(&spec.tiltrotor_controller)),
            ("tiltrotor_action_limits", !is_default(&spec.tiltrotor_action_limits)),
        ];
        let foreign: Vec<_> = match family {
            Family::Multirotor => ground.into_iter().chain(fixed_wing).chain(helicopter).chain(tiltrotor).collect(),
            Family::Wheeled => multirotor.into_iter().chain(fixed_wing).chain(helicopter).chain(tiltrotor).collect(),
            Family::FixedWing => multirotor.into_iter().chain(ground).chain(helicopter).chain(tiltrotor).collect(),
            Family::Rotorcraft => multirotor.into_iter().chain(ground).chain(fixed_wing).chain(tiltrotor).collect(),
            Family::Tiltrotor => multirotor.into_iter().chain(ground).chain(fixed_wing).chain(helicopter).collect(),
        };
        if let Some((field, _)) = foreign.iter().find(|f| f.1) {
            return Err(fail(format!("`{field}` does not apply to {family} vehicles ({:?})", def.name())));
        }
        let mode = spec.action_mode.map_or(AgentActionMode::default_for(family), |m| m.resolve(family));
        spec.action_mode = Some(mode);
        if family == Family::Wheeled {
            spec.spawn.on_ground = true;
        }
        let controller = Controller::new(
            &def,
            clock.dt(),
            &spec.controller,
            &spec.ground_controller,
            &spec.fixed_wing_controller,
            &spec.helicopter_controller,
            &spec.tiltrotor_controller,
        )?;
        let action_map = ActionMapping::new(
            mode,
            &spec.action_limits,
            &spec.ground_action_limits,
            &spec.fixed_wing_action_limits,
            &spec.helicopter_action_limits,
            &spec.tiltrotor_action_limits,
            &def,
            &controller,
        )
        .map_err(|e| fail(e.to_string()))?;
        spec.randomize.validate()?;
        for (i, s) in spec.sensors.iter().enumerate() {
            if spec.sensors[..i].iter().any(|o| o.name == s.name) {
                return Err(fail(format!("duplicate sensor name {:?}", s.name)));
            }
            if let Sensor::Camera(c) = Sensor::new(&s.config, clock, Seed::from_u64(0))?
                && !c.divider().is_multiple_of(decimation)
            {
                return Err(fail(format!("camera {:?}: its period must be a whole number of policy steps", s.name)));
            }
            let units = def.as_wheeled().map_or(1, |d| d.num_units());
            let carried = matches!(s.config, SensorConfig::Rangefinder(_) | SensorConfig::Lidar(_));
            if s.unit >= units || (s.unit > 0 && !carried) {
                return Err(fail(format!(
                    "sensor {:?} on unit {}: the vehicle has {units} units, and only rangefinders and LiDARs ride on units behind the first",
                    s.name, s.unit
                )));
            }
        }
        let terms = if spec.obs.is_empty() { default_obs() } else { spec.obs.clone() };
        let num_rotors = def.as_multirotor().map_or(0, |d| d.rotors.len());
        let num_wheels = def.as_wheeled().map_or(0, |d| d.num_wheels());
        let steering_head = def.as_wheeled().is_some_and(|d| d.steering_head().is_some());
        let obs = CompiledObs::new(&terms, &spec.sensors, action_map.dim(), num_rotors, num_wheels, steering_head)
            .map_err(|e| fail(e.to_string()))?;
        let sp = &spec.spawn;
        let spawn_ok = valid_range(sp.agl)
            && sp.agl[0] >= 0.0
            && sp.clearance >= 0.0
            && sp.min_separation >= 0.0
            && sp.margin >= 0.0
            && sp.speed >= 0.0
            && sp.rates >= 0.0
            && (0.0..=180.0).contains(&sp.tilt_deg)
            && valid_range(sp.yaw_deg)
            && sp.region.is_none_or(|[lo, hi]| lo.cmple(hi).all())
            && sp.cluster.is_none_or(|c| c > 0.0)
            && sp.airspeed.is_none_or(|a| valid_range(a) && a[0] >= 0.0)
            && !matches!(sp.layout, SpawnLayout::Grid { spacing } if spacing.is_nan() || spacing <= 0.0);
        if !spawn_ok {
            return Err(fail(format!("invalid spawn {sp:?}")));
        }
        if sp.on_road && !sp.on_ground {
            return Err(fail("`spawn.on_road` needs `spawn.on_ground`".into()));
        }
        if let Some(d) = &spec.driver {
            d.validate().map_err(&fail)?;
            let single = def.as_wheeled().is_some_and(|d| d.num_units() == 1 && !d.is_single_track());
            if !single || !sp.on_road || spec.goals.kind == GoalKind::Route {
                return Err(fail(format!(
                    "the `{}` driver drives ground vehicles without trailers (not two-wheelers) spawned `on_road`, without `route` goals",
                    d.name()
                )));
            }
        }
        match (sp.airspeed, family) {
            (None, _) | (Some(_), Family::Rotorcraft | Family::Tiltrotor) => {}
            (Some(a), Family::FixedWing) if a[0] > 0.0 => {}
            (Some(_), Family::FixedWing) => {
                return Err(fail("`spawn.airspeed` of an aircraft must be positive".into()));
            }
            _ => return Err(fail("`spawn.airspeed` needs an aircraft (fixed-wing, helicopter or tiltrotor)".into())),
        }
        let gl = &spec.goals;
        if !(valid_range(gl.distance)
            && gl.distance[0] >= 0.0
            && valid_range(gl.agl)
            && valid_range(gl.yaw_deg)
            && gl.clearance >= 0.0
            && gl.radius >= 0.0
            && gl.spacing > 0.0
            && gl.route.step > 0.0
            && gl.count >= 1
            && gl.grade.is_none_or(|g| g > 0.0)
            && gl.landing_slope.is_none_or(|g| g > 0.0)
            && gl.pad >= 0.0
            && gl.pad <= 100.0)
        {
            return Err(fail(format!("invalid goals {gl:?}")));
        }
        gl.bay.validate().map_err(&fail)?;
        if gl.kind == GoalKind::Bay && family != Family::Wheeled {
            return Err(fail("`bay` goals need a ground vehicle".into()));
        }
        if gl.kind == GoalKind::Yard && family == Family::Wheeled {
            return Err(fail("`yard` goals need an aerial vehicle".into()));
        }
        if (gl.path || gl.off_road) && (gl.kind != GoalKind::Random || family != Family::Wheeled) {
            return Err(fail("`goals.path` and `goals.off_road` need `random` goals and a ground vehicle".into()));
        }
        if (gl.grade.is_some() || gl.landing_slope.is_some())
            && (gl.kind != GoalKind::Random || family == Family::Wheeled)
        {
            return Err(fail(
                "`goals.grade` and `goals.landing_slope` need `random` goals and an aerial vehicle".into(),
            ));
        }
        // Trailers in line behind the towing unit.
        let colliders = match def.as_wheeled() {
            Some(d) => d.colliders_in_line(),
            None => def.sphere_colliders(),
        };
        // Aircraft on the ground rest on their gear.
        let fixed_wing_rest = def.as_fixed_wing().map(|d| {
            let (rot, height) =
                d.resting_pose(autonomousim_core::math::frames::STANDARD_GRAVITY).unwrap_or((DQuat::IDENTITY, 0.0));
            Pose::new(DVec3::new(0.0, 0.0, height), rot)
        });
        // Ground vehicles are placed from the ground point, helicopters from their skids.
        let bottom = match family {
            Family::Multirotor => colliders.iter().map(|c| c.radius - c.center.z).fold(0.0, f64::max),
            Family::Rotorcraft => def.as_helicopter().and_then(|d| d.skid_height()).unwrap_or(0.0),
            Family::Tiltrotor => def.as_tiltrotor().and_then(|d| d.gear_height()).unwrap_or(0.0),
            Family::Wheeled => 0.0,
            Family::FixedWing => fixed_wing_rest.map_or(0.0, |p| p.pos.z),
        };
        let radius = colliders.iter().map(|c| c.center.length() + c.radius).fold(0.0, f64::max);
        spec.drivable.validate().map_err(fail)?;
        let (rest, half_width) = match def.as_wheeled() {
            Some(d) => (Wheeled::new(d.clone(), clock.dt()).rest(DVec3::ZERO, 0.0, 0.0).pose, drive::half_width(d)),
            None => (fixed_wing_rest.unwrap_or(Pose::IDENTITY), 0.0),
        };
        Ok(Self {
            spec,
            def,
            controller,
            action_map,
            obs,
            bottom,
            radius,
            first_agent,
            rest,
            half_width,
            drive: Vec::new(),
        })
    }

    /// Where this group's vehicles stand on the map, for spawning and goals (ground vehicles).
    pub(crate) fn ground<'a>(&'a self, map: usize) -> Option<GroundSampling<'a>> {
        self.drive.get(map).map(|grid| GroundSampling {
            grid,
            ride: self.rest.pos.z,
            radius: self.radius,
            max_slope: self
                .def
                .as_wheeled()
                .and_then(|d| d.feet)
                .map_or(self.spec.drivable.spawn_slope_deg, |f| f.max_slope_deg.min(self.spec.drivable.spawn_slope_deg))
                .to_radians(),
        })
    }
}

/// Drive grids of the ground groups on every map; groups with the same drivable spec and
/// width share them.
fn build_drive_grids(groups: &mut [CompiledGroup], maps: &[Arc<StaticWorld>]) {
    for i in 0..groups.len() {
        if groups[i].family() != Family::Wheeled {
            continue;
        }
        let (spec, hw) = (&groups[i].spec.drivable, groups[i].half_width);
        let def = groups[i].def.as_wheeled().expect("wheeled group");
        // With resistance costs the grid also depends on the vehicle's running gear.
        let same = |o: &CompiledGroup| {
            o.spec.drivable == *spec
                && o.half_width == hw
                && (spec.resistance_cost == 0.0 || o.def.as_wheeled().is_some_and(|d| d == def))
        };
        let shared = groups[..i].iter().find(|o| !o.drive.is_empty() && same(o));
        groups[i].drive = match shared {
            Some(o) => o.drive.clone(),
            None => maps
                .par_iter()
                .map(|m| {
                    let g = autonomousim_core::math::frames::STANDARD_GRAVITY;
                    Arc::new(DriveGrid::with_resistance(m, spec, hw, |mat| def.motion_resistance(mat, g)))
                })
                .collect(),
        };
    }
}

/// What spawn and goal sampling of ground vehicles needs.
#[derive(Clone, Copy)]
pub(crate) struct GroundSampling<'a> {
    pub grid: &'a DriveGrid,
    /// Height of the chassis frame above the ground at rest (m).
    pub ride: f64,
    /// Radius kept clear of solid obstacles at the spawn (m).
    pub radius: f64,
    /// Steepest spawn (rad).
    pub max_slope: f64,
}

// ------------------------------------------------------------------------------- sampling

fn valid_range(r: [f64; 2]) -> bool {
    r[0].is_finite() && r[1].is_finite() && r[0] <= r[1]
}

fn sample(rng: &mut SimRng, r: [f64; 2]) -> f64 {
    r[0] + (r[1] - r[0]) * rng.uniform()
}

/// Offset of slot `k` of an `n`-slot formation with unit spacing, centred on the origin.
fn formation_slot(shape: FormationShape, n: usize, k: usize) -> DVec2 {
    match shape {
        FormationShape::Grid => {
            let cols = (n as f64).sqrt().ceil() as usize;
            let rows = n.div_ceil(cols);
            DVec2::new((k % cols) as f64 - 0.5 * (cols - 1) as f64, (k / cols) as f64 - 0.5 * (rows - 1) as f64)
        }
        FormationShape::Circle if n < 2 => DVec2::ZERO,
        FormationShape::Circle => {
            let step = std::f64::consts::TAU / n as f64;
            DVec2::from_angle(step * k as f64) * (0.5 / (0.5 * step).sin())
        }
    }
}

/// Region `[min, max]` inside the map, shrunk by `margin` (never inverted).
pub(crate) fn region(world: &StaticWorld, region: Option<[DVec2; 2]>, margin: f64) -> [DVec2; 2] {
    let (lo, hi) = world.extent();
    let (lo, hi) = match region {
        Some([a, b]) => (a.max(lo), b.min(hi)),
        None => (lo, hi),
    };
    let m = DVec2::splat(margin).min(0.5 * (hi - lo).max(DVec2::ZERO));
    [lo + m, (hi - m).max(lo + m)]
}

/// Initial state of one agent.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Placement {
    pub pose: Pose,
    pub lin_vel: DVec3,
    pub ang_vel: DVec3,
    pub motors: MotorInit,
    /// Fixed-wing aircraft in the air: their trim (none: at rest on the brakes).
    pub fixed_wing: Option<FixedWingStart>,
    /// Helicopters in the air: their trim (none: on the skids, collective down).
    pub helicopter: Option<HelicopterStart>,
    /// Tiltrotors in the air: their trim (none: on the gear, throttles closed).
    pub tiltrotor: Option<TiltrotorStart>,
}

/// Controls and rotor speeds a tiltrotor starts with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TiltrotorStart {
    pub controls: TiltrotorInput,
    pub rotor_speed: [f64; 4],
    /// Air density of the trim (kg/m³).
    pub density: f64,
}

/// Controls and rotor speed a fixed-wing aircraft starts with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FixedWingStart {
    pub controls: FixedWingInput,
    pub rotor_speed: f64,
}

/// Controls and rotor speed a helicopter starts with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HelicopterStart {
    pub controls: HelicopterInput,
    pub rotor_speed: f64,
    /// Air density of the trim (kg/m³).
    pub density: f64,
}

impl SpawnSpec {
    /// Positions of `count` agents; `placed` holds positions of agents placed before (other
    /// groups) and receives these. Ground vehicles (`ground`) get the position of their chassis
    /// frame at rest, in drivable cells of the largest component, clear of obstacles by their
    /// radius.
    pub(crate) fn sample_positions(
        &self,
        world: &StaticWorld,
        count: usize,
        bottom: f64,
        ground: Option<GroundSampling>,
        placed: &mut Vec<DVec3>,
        rng: &mut SimRng,
    ) -> Vec<DVec3> {
        let [lo, hi] = region(world, self.region, self.margin);
        let [lo, hi] = match self.cluster {
            Some(side) => {
                let half = DVec2::splat(0.5 * side).min(0.5 * (hi - lo));
                // The most open of a few random centres: the group takes off from a clearing
                // when there is one nearby, not from wherever the square happens to land. Lakes
                // are free of obstacles too, so openness counts only in proportion to the dry
                // land in the square (with `avoid_water`).
                let open_enough = half.max_element();
                let mid = 0.5 * (self.agl[0] + self.agl[1]);
                let mut best = (f64::NEG_INFINITY, DVec2::ZERO);
                for _ in 0..CLUSTER_ATTEMPTS {
                    let centre = DVec2::new(
                        sample(rng, [lo.x + half.x, hi.x - half.x]),
                        sample(rng, [lo.y + half.y, hi.y - half.y]),
                    );
                    let p = centre.extend(world.surface_height(centre.x, centre.y) + mid);
                    let dry = if self.avoid_water { dry_fraction(world, centre, half) } else { 1.0 };
                    let open = world.obstacle_clearance(p, open_enough) * dry;
                    if open > best.0 {
                        best = (open, centre);
                    }
                    if open >= open_enough {
                        break;
                    }
                }
                [best.1 - half, best.1 + half]
            }
            None => [lo, hi],
        };
        let height = |xy: DVec2, rng: &mut SimRng| {
            let surface = world.surface_height(xy.x, xy.y);
            if let Some(g) = ground {
                world.terrain().height(xy.x, xy.y) + g.ride
            } else if self.on_ground {
                world.terrain().height(xy.x, xy.y) + bottom + 1e-3
            } else {
                surface + sample(rng, self.agl)
            }
        };
        let mut out = Vec::with_capacity(count);
        match self.layout {
            SpawnLayout::Grid { spacing } => {
                let cols = (count as f64).sqrt().ceil() as usize;
                let rows = count.div_ceil(cols);
                let centre = 0.5 * (lo + hi);
                let mid = 0.5 * (self.agl[0] + self.agl[1]);
                for i in 0..count {
                    let offset = DVec2::new(
                        (i % cols) as f64 - 0.5 * (cols - 1) as f64,
                        (i / cols) as f64 - 0.5 * (rows - 1) as f64,
                    );
                    let xy = centre + offset * spacing;
                    let z = if self.on_ground || ground.is_some() {
                        height(xy, rng)
                    } else {
                        world.surface_height(xy.x, xy.y) + mid
                    };
                    out.push(xy.extend(z));
                }
            }
            SpawnLayout::Random => {
                // On tiled maps candidates come in batches from one tile each, so that a
                // crowded map does not generate a tile per attempt.
                let layout = world.terrain().tiled().map(|t| *t.layout());
                for _ in 0..count {
                    let mut best = (f64::NEG_INFINITY, DVec3::ZERO);
                    let mut area = [lo, hi];
                    for attempt in 0..MAX_ATTEMPTS {
                        if let Some(l) = &layout
                            && attempt % TILE_BATCH == 0
                        {
                            let p = DVec2::new(sample(rng, [lo.x, hi.x]), sample(rng, [lo.y, hi.y]));
                            let (tx, ty) = l.tile_at(p.x, p.y);
                            let (a, b) = l.core(tx, ty);
                            area = [a.max(lo), b.min(hi).max(a.max(lo))];
                        }
                        let [lo, hi] = area;
                        let xy = DVec2::new(sample(rng, [lo.x, hi.x]), sample(rng, [lo.y, hi.y]));
                        let p = xy.extend(height(xy, rng));
                        let score = self.score(world, p, ground, placed.iter().chain(&out));
                        if score > best.0 {
                            best = (score, p);
                        }
                        if score >= 1.0 {
                            break;
                        }
                    }
                    out.push(best.1);
                }
            }
        }
        placed.extend_from_slice(&out);
        out
    }

    /// 1 or more if `p` is acceptable; otherwise how close it comes (for the fallback): in
    /// [0, 1) for a point too close to obstacles, below 0 for one too close to another agent
    /// or over water.
    fn score<'a>(
        &self,
        world: &StaticWorld,
        p: DVec3,
        ground: Option<GroundSampling>,
        others: impl Iterator<Item = &'a DVec3>,
    ) -> f64 {
        if self.avoid_water && over_water(world, p.truncate()) {
            return -1.0;
        }
        // On the ground, check the space just above it instead.
        let probe = if self.on_ground { DVec3::new(p.x, p.y, p.z + self.clearance) } else { p };
        let free = if let Some(g) = ground {
            let xy = p.truncate();
            let r = g.radius.max(self.clearance);
            match g.grid.component(xy) {
                None => return -1.0,
                // Off the largest component, few goals can be reached.
                Some(c) if c != g.grid.largest() => 0.3,
                Some(_) if drive::slope(world, xy, 1.5) > g.max_slope => 0.5,
                Some(_) => world.obstacles().nearest_distance(p, r, HitMask::SOLID).map_or(1.0, |d| d / r),
            }
        } else if self.clearance > 0.0 {
            let c = world.clearance(probe, self.clearance) / self.clearance;
            let foliage = world.is_free(probe, self.clearance, true, false);
            if foliage { c } else { c.min(0.5) }
        } else {
            1.0
        };
        let apart = if self.min_separation > 0.0 {
            others.map(|o| o.distance(p) / self.min_separation).fold(f64::INFINITY, f64::min)
        } else {
            f64::INFINITY
        };
        // Agents too close together start in contact: rank such points below every point
        // that keeps the separation, however cramped it is otherwise.
        if apart < 1.0 { apart - 1.0 } else { free }
    }

    /// Orientation, velocities and motor state at `position`.
    pub(crate) fn sample_state(&self, position: DVec3, hover_omega: f64, rng: &mut SimRng) -> Placement {
        let yaw = sample(rng, self.yaw_deg).to_radians();
        let tilt_axis = DVec2::from_angle(rng.range(-std::f64::consts::PI, std::f64::consts::PI));
        let tilt = self.tilt_deg.to_radians() * rng.uniform();
        let speed = self.speed * rng.uniform();
        let v_dir = rng.unit_vector();
        let rate = self.rates * rng.uniform();
        let w_dir = rng.unit_vector();
        let (tilt, speed, rate) = if self.on_ground { (0.0, 0.0, 0.0) } else { (tilt, speed, rate) };
        let rot = from_yaw(yaw) * DQuat::from_scaled_axis((tilt_axis * tilt).extend(0.0));
        let motors = match (self.motors, self.on_ground) {
            (SpawnMotors::Hover, false) => MotorInit::Speed(hover_omega),
            _ => MotorInit::Idle,
        };
        Placement {
            pose: Pose::new(position, rot),
            lin_vel: v_dir * speed,
            ang_vel: w_dir * rate,
            motors,
            fixed_wing: None,
            helicopter: None,
            tiltrotor: None,
        }
    }
}

/// Whether `xy` lies in a lake or river.
fn over_water(world: &StaticWorld, xy: DVec2) -> bool {
    world.terrain().water_level(xy.x, xy.y).is_some_and(|w| w > world.terrain().height(xy.x, xy.y))
}

/// Share of a 5 × 5 grid of points over the square `centre ± half` that is not over water.
fn dry_fraction(world: &StaticWorld, centre: DVec2, half: DVec2) -> f64 {
    const N: usize = 5;
    let mut dry = 0;
    for i in 0..N {
        for j in 0..N {
            let f = DVec2::new(i as f64, j as f64) / (N - 1) as f64 * 2.0 - 1.0;
            dry += usize::from(!over_water(world, centre + f * half));
        }
    }
    dry as f64 / (N * N) as f64
}

impl GoalSpec {
    /// Goals from `spawn`. Ground vehicles' goals (`ground`) are where their chassis frame would
    /// rest on the terrain, in drivable cells reachable from the spawn; the height range is
    /// ignored.
    pub(crate) fn sample(
        &self,
        world: &StaticWorld,
        spawn: &Pose,
        ground: Option<GroundSampling>,
        rng: &mut SimRng,
    ) -> Vec<Goal> {
        use autonomousim_core::math::quat::yaw;
        match self.kind {
            // Route goals come from `lane`; this is the fallback when no route is found.
            GoalKind::Spawn | GoalKind::Formation | GoalKind::Route | GoalKind::Bay | GoalKind::Yard => {
                vec![Goal { position: spawn.pos, yaw: yaw(spawn.rot) }]
            }
            GoalKind::Random => {
                let [lo, hi] = region(world, None, self.margin);
                let mut prev3 = spawn.pos;
                let mut goals = Vec::with_capacity(self.count);
                for _ in 0..self.count {
                    let mut best = (f64::NEG_INFINITY, DVec3::ZERO);
                    for _ in 0..MAX_ATTEMPTS {
                        let dir = DVec2::from_angle(rng.range(-std::f64::consts::PI, std::f64::consts::PI));
                        let prev = prev3.truncate();
                        let raw = prev + dir * sample(rng, self.distance);
                        let xy = raw.clamp(lo, hi);
                        let agl = sample(rng, self.agl);
                        let (p, free) = match ground {
                            Some(g) => {
                                let p = xy.extend(world.terrain().height(xy.x, xy.y) + g.ride);
                                // Unreachable goals rank below any reachable one, goals on
                                // roads (with `off_road`) below any off them.
                                let free = if !g.grid.reachable(spawn.pos.truncate(), xy) {
                                    -3.0
                                } else if self.off_road && world.roads().on_road(xy).is_some() {
                                    0.5
                                } else {
                                    1.0
                                };
                                (p, free)
                            }
                            None => {
                                let p = xy.extend(world.surface_height(xy.x, xy.y) + agl);
                                // Too steep from the previous point: below any draw that is not.
                                let steep = self.grade.is_some_and(|g| (p.z - prev3.z).abs() > g * xy.distance(prev));
                                let free = match self.landing_slope {
                                    Some(slope) => self.landing_zone(world, xy, slope),
                                    None => self.free(world, p),
                                };
                                (p, free - if steep { 1.5 } else { 0.0 })
                            }
                        };
                        // A point moved onto the region's edge is off the distance range: only
                        // used when no draw lands inside.
                        let score = if xy == raw { free } else { free - 2.0 };
                        if score > best.0 {
                            best = (score, p);
                        }
                        if score >= 1.0 {
                            break;
                        }
                    }
                    let yaw = sample(rng, self.yaw_deg).to_radians();
                    goals.push(Goal { position: best.1, yaw });
                    prev3 = best.1;
                }
                goals
            }
        }
    }

    /// Formation slots of a group from its spawn positions, one goal per agent (for
    /// `GoalKind::Formation`). Ground vehicles' slots (`ground`) rest on the terrain.
    pub(crate) fn formation(
        &self,
        world: &StaticWorld,
        spawns: &[DVec3],
        ground: Option<GroundSampling>,
        rng: &mut SimRng,
    ) -> Vec<Goal> {
        let n = spawns.len();
        let centre = spawns.iter().map(|p| p.truncate()).sum::<DVec2>() / n.max(1) as f64;
        let heading = sample(rng, self.yaw_deg).to_radians();
        let agl = sample(rng, self.agl);
        let turn = DVec2::from_angle(heading);
        let [lo, hi] = region(world, None, self.margin);
        (0..n)
            .map(|k| {
                let offset = formation_slot(self.formation, n, k) * self.spacing;
                let xy = (centre + turn.rotate(offset)).clamp(lo, hi);
                let z = match ground {
                    Some(g) => world.terrain().height(xy.x, xy.y) + g.ride,
                    None => world.surface_height(xy.x, xy.y) + agl,
                };
                Goal { position: xy.extend(z), yaw: heading }
            })
            .collect()
    }

    /// Score of a landing zone at `xy` (see `landing_slope`): the share of the clearance free
    /// of obstacles and foliage (1 when clear), less 1.5 when the terrain is steeper than
    /// `slope` and 2 when there is water.
    fn landing_zone(&self, world: &StaticWorld, xy: DVec2, slope: f64) -> f64 {
        let r = self.clearance.max(1.0);
        let t = world.terrain();
        let h = t.height(xy.x, xy.y);
        // Rise or fall from the centre to 16 points on the rim and 8 halfway.
        let rim = (0..16).map(|k| (f64::from(k) * std::f64::consts::FRAC_PI_8, r));
        let half = (0..8).map(|k| (f64::from(k) * std::f64::consts::FRAC_PI_4, 0.5 * r));
        let steepest = rim
            .chain(half)
            .map(|(a, d)| {
                let q = xy + DVec2::from_angle(a) * d;
                (t.height(q.x, q.y) - h).abs() / d
            })
            .fold(0.0, f64::max);
        // Obstacles and foliage around the column the descent comes down (canopies overhang
        // clearings), from 1 m to `LANDING_COLUMN` above the ground.
        let steps = ((LANDING_COLUMN - 1.0) / (0.5 * r)).ceil() as usize;
        let open = (0..=steps)
            .map(|k| {
                let p = xy.extend(h + (1.0 + 0.5 * r * k as f64).min(LANDING_COLUMN));
                world.obstacles().nearest_distance(p, r, HitMask::SOLID | HitMask::FOLIAGE).map_or(1.0, |d| d / r)
            })
            .fold(1.0, f64::min);
        let wet = dry_fraction(world, xy, DVec2::splat(r)) < 1.0;
        open - if steepest > slope { 1.5 } else { 0.0 } - if wet { 2.0 } else { 0.0 }
    }

    /// 1 when a goal at `p` is free; otherwise how close it comes.
    fn free(&self, world: &StaticWorld, p: DVec3) -> f64 {
        if self.clearance > 0.0 {
            let c = world.clearance(p, self.clearance) / self.clearance;
            if world.is_free(p, self.clearance, true, false) { c } else { c.min(0.5) }
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_map_and_groups() {
        let s = Scenario::from_toml(
            r#"
            name = "t"
            policy_hz = 100
            map = { type = "testworld", kind = "forest_patch", size = 100.0, density = 100.0, seed = 3 }
            [randomize_environment]
            wind_speed = [0.0, 5.0]
            [[groups]]
            name = "quad"
            vehicle = "iris_like"
            action_mode = "velocity"
            spawn = { layout = { type = "grid", spacing = 2.0 }, agl = [2.0, 2.0] }
            sensors = [ { name = "imu", type = "imu" } ]
            obs = [ { term = "imu", sensor = "imu", scale = 0.1 } ]
            "#,
        )
        .unwrap();
        assert_eq!(s.map, MapSource::Testworld(Testworld::ForestPatch { size: 100.0, density: 100.0, seed: 3 }));
        let c = s.compile().unwrap();
        assert_eq!((c.decimation, c.environment_divider), (5, 5));
        assert_eq!((c.groups[0].act_dim(), c.groups[0].obs_dim()), (4, 6));
        assert!(c.groups[0].bottom > 0.0 && c.groups[0].radius > 0.2);

        // Round trip through JSON.
        let json = c.spec.to_json();
        assert_eq!(Scenario::from_json(&json).unwrap(), c.spec);

        // Errors.
        for bad in [
            "policy_hz = 7",
            "groups = []",
            "[[groups]]\ncount = 0",
            "[[groups]]\nvehicle = \"nope\"",
            "[[groups]]\nobs = [ { term = \"imu\", sensor = \"missing\" } ]",
            "[[groups]]\nspawn = { agl = [3.0, 1.0] }",
            "[[groups]]\nrandomize = { mass = 1.5 }",
            "map = { type = \"testworld\", kind = \"flat\", size = -1.0 }",
            "unknown = 1",
        ] {
            let r = Scenario::from_toml(bad).and_then(Scenario::compile);
            assert!(r.is_err(), "{bad}");
        }
    }

    #[test]
    fn spawns_are_free_and_apart() {
        let world = testworlds::forest_patch(120.0, 300.0, 2);
        let spec = SpawnSpec { min_separation: 3.0, clearance: 1.5, ..SpawnSpec::default() };
        let mut rng = Seed::from_u64(1).rng();
        let mut placed = Vec::new();
        let ps = spec.sample_positions(&world, 30, 0.05, None, &mut placed, &mut rng);
        for (i, p) in ps.iter().enumerate() {
            assert!(world.is_free(*p, 1.5, true, true), "{p}");
            let agl = p.z - world.surface_height(p.x, p.y);
            assert!((1.0..=3.0).contains(&agl));
            for q in &ps[..i] {
                assert!(p.distance(*q) >= 3.0);
            }
        }
        // Goals chain from the spawn and stay free.
        let goals = GoalSpec { kind: GoalKind::Random, count: 5, distance: [4.0, 8.0], ..GoalSpec::default() };
        let g = goals.sample(&world, &Pose::from_translation(ps[0]), None, &mut rng);
        assert_eq!(g.len(), 5);
        let mut prev = ps[0];
        for goal in &g {
            let d = (goal.position - prev).truncate().length();
            assert!((4.0 - 1e-9..=8.0 + 1e-9).contains(&d), "{d}");
            assert!(world.is_free(goal.position, 1.0, true, false));
            prev = goal.position;
        }
    }

    #[test]
    fn clusters_form_on_dry_land() {
        // No trees anywhere, so every centre is equally open; one over the lake would squeeze
        // the group onto its shore.
        let world = testworlds::lake(120.0, 2.0, -0.5);
        let spec = SpawnSpec { cluster: Some(20.0), min_separation: 4.0, ..SpawnSpec::default() };
        let mut rng = Seed::from_u64(4).rng();
        for _ in 0..20 {
            let ps = spec.sample_positions(&world, 8, 0.05, None, &mut Vec::new(), &mut rng);
            for (i, p) in ps.iter().enumerate() {
                assert!(!over_water(&world, p.truncate()), "{p}");
                for q in &ps[..i] {
                    assert!(p.distance(*q) >= 4.0, "{p} {q}");
                }
            }
        }
    }

    #[test]
    fn cramped_spawns_keep_their_separation() {
        // So dense that hardly any point is free of trunks and canopies: the fallback may
        // start agents near trees, but never on top of each other.
        let world = testworlds::forest_patch(120.0, 1500.0, 7);
        let spec = SpawnSpec { cluster: Some(20.0), min_separation: 4.0, clearance: 2.0, ..SpawnSpec::default() };
        let mut rng = Seed::from_u64(2).rng();
        for _ in 0..20 {
            let ps = spec.sample_positions(&world, 8, 0.05, None, &mut Vec::new(), &mut rng);
            for (i, p) in ps.iter().enumerate() {
                for q in &ps[..i] {
                    assert!(p.distance(*q) >= 4.0, "{p} {q}");
                }
            }
        }
    }

    #[test]
    fn cluster_spawns_and_formation_slots() {
        let world = testworlds::flat(200.0);
        let spawn = SpawnSpec { cluster: Some(8.0), min_separation: 1.5, ..SpawnSpec::default() };
        let mut rng = Seed::from_u64(4).rng();
        for shape in [FormationShape::Grid, FormationShape::Circle] {
            let ps = spawn.sample_positions(&world, 7, 0.05, None, &mut Vec::new(), &mut rng);
            for p in &ps {
                for q in &ps {
                    assert!((p - q).truncate().abs().max_element() <= 8.0 + 1e-9);
                }
            }
            let goals = GoalSpec {
                kind: GoalKind::Formation,
                formation: shape,
                spacing: 3.0,
                agl: [2.0, 2.0],
                ..GoalSpec::default()
            };
            let slots = goals.formation(&world, &ps, None, &mut rng);
            assert_eq!(slots.len(), 7);
            let centre = |v: Vec<DVec2>| v.iter().sum::<DVec2>() / v.len() as f64;
            let c = centre(ps.iter().map(|p| p.truncate()).collect());
            let nearest = |i: usize| {
                (0..7)
                    .filter(|&j| j != i)
                    .map(|j| slots[i].position.distance(slots[j].position))
                    .fold(f64::INFINITY, f64::min)
            };
            for (i, s) in slots.iter().enumerate() {
                assert!((s.position.z - 2.0).abs() < 1e-9);
                assert!((nearest(i) - 3.0).abs() < 1e-9, "{shape:?} {}", nearest(i));
                assert_eq!(s.yaw, slots[0].yaw);
            }
            match shape {
                // A 3×3 grid with 7 slots is not centred; the ring is.
                FormationShape::Grid => {
                    assert!(centre(slots.iter().map(|g| g.position.truncate()).collect()).distance(c) < 3.0)
                }
                FormationShape::Circle => {
                    assert!(centre(slots.iter().map(|g| g.position.truncate()).collect()).distance(c) < 1e-9)
                }
            }
        }
    }

    #[test]
    fn environment_randomisation_draws_gusts() {
        let r = EnvironmentRandomization {
            wind_speed: Some([3.0, 3.0]),
            gust_rate: 6.0,
            gust_horizon: 600.0,
            ..EnvironmentRandomization::default()
        };
        let mut env = EnvironmentConfig::default();
        r.apply(&mut env, &mut Seed::from_u64(4).rng());
        assert!((env.wind.mean.length() - 3.0).abs() < 1e-12);
        // 6 per minute over 10 minutes: 60 ± 3σ.
        let n = env.wind.gusts.len();
        assert!((37..=83).contains(&n), "{n}");
        assert!(env.wind.gusts.windows(2).all(|w| w[0].start < w[1].start));
    }
}
