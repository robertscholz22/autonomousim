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
//! ```
//!
//! Angles in scenario files are in degrees (`*_deg`); everything else is SI.

use crate::SimError;
use crate::obs::{CompiledObs, ObsTerm, default_obs};
use autonomousim_control::multirotor::{ActionLimits, ActionMap, ActionMode, ControllerConfig, MultirotorController};
use autonomousim_core::material::MaterialId;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::from_yaw;
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::terrain::Terrain;
use autonomousim_core::time::Clock;
use autonomousim_procgen::MapCache;
use autonomousim_procgen::wild::{self, WildConfig, WildPreset};
use autonomousim_sensors::{Sensor, SensorSpec};
use autonomousim_vehicles::VehicleDef;
use autonomousim_vehicles::multirotor::{MotorInit, MultirotorDef, MultirotorScales};
use autonomousim_vehicles::presets;
use autonomousim_world::environment::{EnvironmentConfig, Gust};
use autonomousim_world::testworlds;
use autonomousim_world::{MapHash, StaticWorld};
use glam::{DQuat, DVec2, DVec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// Candidate positions tried before falling back to the best one seen.
const MAX_ATTEMPTS: usize = 200;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Scenario {
    pub name: String,
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
            physics_hz: 500,
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
}

impl Default for EventConfig {
    fn default() -> Self {
        Self { crash_speed: 2.0, landed_speed: 0.1, landed_rate: 0.3, bounds_margin: 0.0, max_agl: None, ceiling: None }
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroupSpec {
    pub name: String,
    pub count: usize,
    pub vehicle: VehicleRef,
    pub controller: ControllerConfig,
    pub action_mode: ActionMode,
    pub action_limits: ActionLimits,
    pub sensors: Vec<SensorSpec>,
    /// Observation terms in order; empty: the default hover observation.
    pub obs: Vec<ObsTerm>,
    pub spawn: SpawnSpec,
    pub goals: GoalSpec,
    pub randomize: VehicleRandomization,
    /// Freeze an agent after a terminal event until the next reset (it then no longer moves,
    /// collides or appears in other agents' sensors).
    pub disable_on_terminal: bool,
}

impl Default for GroupSpec {
    fn default() -> Self {
        Self {
            name: "agents".into(),
            count: 1,
            vehicle: VehicleRef::default(),
            controller: ControllerConfig::default(),
            action_mode: ActionMode::default(),
            action_limits: ActionLimits::default(),
            sensors: Vec::new(),
            obs: Vec::new(),
            spawn: SpawnSpec::default(),
            goals: GoalSpec::default(),
            randomize: VehicleRandomization::default(),
            disable_on_terminal: true,
        }
    }
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
    /// Height of the centre of mass above the ground or water surface (m).
    pub agl: [f64; 2],
    /// Start resting on the ground (motors idle unless set otherwise); ignores `agl`.
    pub on_ground: bool,
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
}

impl Default for SpawnSpec {
    fn default() -> Self {
        Self {
            layout: SpawnLayout::Random,
            region: None,
            margin: 10.0,
            agl: [1.0, 3.0],
            on_ground: false,
            clearance: 1.0,
            min_separation: 1.0,
            avoid_water: true,
            speed: 0.0,
            tilt_deg: 0.0,
            yaw_deg: [-180.0, 180.0],
            rates: 0.0,
            motors: SpawnMotors::Hover,
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
/// the world instances of a batch.
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
    pub groups: Vec<CompiledGroup>,
}

#[derive(Debug)]
pub struct CompiledGroup {
    pub spec: GroupSpec,
    pub def: Arc<MultirotorDef>,
    /// Controller at its initial state; agents start from a clone.
    pub controller: MultirotorController,
    pub action_map: ActionMap,
    pub obs: CompiledObs,
    /// Depth of the lowest collider point below the centre of mass (m).
    pub bottom: f64,
    /// Radius of the sphere about the centre of mass that contains all colliders (m).
    pub radius: f64,
    /// Id of the group's first agent (agents are numbered group by group).
    pub first_agent: usize,
}

impl CompiledGroup {
    pub fn act_dim(&self) -> usize {
        self.action_map.dim()
    }

    pub fn obs_dim(&self) -> usize {
        self.obs.dim()
    }
}

impl CompiledScenario {
    fn new(spec: Scenario) -> Result<Self, SimError> {
        let clock = Clock::new(spec.physics_hz.max(1));
        if spec.physics_hz == 0 {
            return Err(SimError::Scenario("physics_hz must be positive".into()));
        }
        let decimation = clock.divider("policy", spec.policy_hz)?;
        let environment_divider = clock.divider("environment", spec.environment_hz)?;
        if let Some(r) = &spec.randomize_environment {
            r.validate()?;
        }
        let e = &spec.events;
        if !(e.crash_speed > 0.0 && e.landed_speed >= 0.0 && e.landed_rate >= 0.0 && e.bounds_margin >= 0.0) {
            return Err(SimError::Scenario(format!("invalid event thresholds {e:?}")));
        }
        if spec.groups.is_empty() {
            return Err(SimError::Scenario("a scenario needs at least one agent group".into()));
        }
        let (maps, map_hashes) = spec.map.build()?.into_iter().map(|(w, h)| (Arc::new(w), h)).unzip();
        let mut groups = Vec::with_capacity(spec.groups.len());
        let mut first_agent = 0;
        for (gi, g) in spec.groups.iter().enumerate() {
            if g.count == 0 {
                return Err(SimError::Scenario(format!("group {:?} has no agents", g.name)));
            }
            if spec.groups[..gi].iter().any(|o| o.name == g.name) {
                return Err(SimError::Scenario(format!("duplicate group name {:?}", g.name)));
            }
            groups.push(CompiledGroup::new(g.clone(), &clock, first_agent)?);
            first_agent += g.count;
        }
        Ok(Self { spec, clock, decimation, environment_divider, maps, map_hashes, groups })
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
    fn new(spec: GroupSpec, clock: &Clock, first_agent: usize) -> Result<Self, SimError> {
        let def = match spec.vehicle.resolve()? {
            VehicleDef::Multirotor(m) => Arc::new(m),
            VehicleDef::Wheeled(w) => {
                return Err(SimError::Scenario(format!(
                    "group {:?}: wheeled vehicle {:?} is not supported in scenarios yet",
                    spec.name, w.name
                )));
            }
        };
        let name = spec.name.clone();
        let fail = move |what: String| SimError::Scenario(format!("group {name:?}: {what}"));
        let controller = MultirotorController::new(&def, clock.dt(), &spec.controller)?;
        let action_map = ActionMap::new(spec.action_mode, spec.action_limits.clone(), &def, controller.max_thrust());
        spec.randomize.validate()?;
        for (i, s) in spec.sensors.iter().enumerate() {
            if spec.sensors[..i].iter().any(|o| o.name == s.name) {
                return Err(fail(format!("duplicate sensor name {:?}", s.name)));
            }
            Sensor::new(&s.config, clock, Seed::from_u64(0))?;
        }
        let terms = if spec.obs.is_empty() { default_obs() } else { spec.obs.clone() };
        let obs = CompiledObs::new(&terms, &spec.sensors, action_map.dim(), def.rotors.len())
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
            && !matches!(sp.layout, SpawnLayout::Grid { spacing } if spacing.is_nan() || spacing <= 0.0);
        if !spawn_ok {
            return Err(fail(format!("invalid spawn {sp:?}")));
        }
        let gl = &spec.goals;
        if !(valid_range(gl.distance)
            && gl.distance[0] >= 0.0
            && valid_range(gl.agl)
            && valid_range(gl.yaw_deg)
            && gl.clearance >= 0.0
            && gl.radius >= 0.0
            && gl.count >= 1)
        {
            return Err(fail(format!("invalid goals {gl:?}")));
        }
        let colliders = def.sphere_colliders();
        let bottom = colliders.iter().map(|c| c.radius - c.center.z).fold(0.0, f64::max);
        let radius = colliders.iter().map(|c| c.center.length() + c.radius).fold(0.0, f64::max);
        Ok(Self { spec, def, controller, action_map, obs, bottom, radius, first_agent })
    }
}

// ------------------------------------------------------------------------------- sampling

fn valid_range(r: [f64; 2]) -> bool {
    r[0].is_finite() && r[1].is_finite() && r[0] <= r[1]
}

fn sample(rng: &mut SimRng, r: [f64; 2]) -> f64 {
    r[0] + (r[1] - r[0]) * rng.uniform()
}

/// Region `[min, max]` inside the map, shrunk by `margin` (never inverted).
fn region(world: &StaticWorld, region: Option<[DVec2; 2]>, margin: f64) -> [DVec2; 2] {
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
}

impl SpawnSpec {
    /// Positions of `count` agents; `placed` holds positions of agents placed before (other
    /// groups) and receives these.
    pub(crate) fn sample_positions(
        &self,
        world: &StaticWorld,
        count: usize,
        bottom: f64,
        placed: &mut Vec<DVec3>,
        rng: &mut SimRng,
    ) -> Vec<DVec3> {
        let [lo, hi] = region(world, self.region, self.margin);
        let height = |xy: DVec2, rng: &mut SimRng| {
            let surface = world.surface_height(xy.x, xy.y);
            if self.on_ground {
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
                    let z = if self.on_ground { height(xy, rng) } else { world.surface_height(xy.x, xy.y) + mid };
                    out.push(xy.extend(z));
                }
            }
            SpawnLayout::Random => {
                for _ in 0..count {
                    let mut best = (f64::NEG_INFINITY, DVec3::ZERO);
                    for _ in 0..MAX_ATTEMPTS {
                        let xy = DVec2::new(sample(rng, [lo.x, hi.x]), sample(rng, [lo.y, hi.y]));
                        let p = xy.extend(height(xy, rng));
                        let score = self.score(world, p, placed.iter().chain(&out));
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

    /// 1 or more if `p` is acceptable; otherwise how close it comes (for the fallback).
    fn score<'a>(&self, world: &StaticWorld, p: DVec3, others: impl Iterator<Item = &'a DVec3>) -> f64 {
        if self.avoid_water
            && world.terrain().water_level(p.x, p.y).is_some_and(|w| w > world.terrain().height(p.x, p.y))
        {
            return -1.0;
        }
        // On the ground, check the space just above it instead.
        let probe = if self.on_ground { DVec3::new(p.x, p.y, p.z + self.clearance) } else { p };
        let free = if self.clearance > 0.0 {
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
        free.min(apart)
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
        Placement { pose: Pose::new(position, rot), lin_vel: v_dir * speed, ang_vel: w_dir * rate, motors }
    }
}

impl GoalSpec {
    pub(crate) fn sample(&self, world: &StaticWorld, spawn: &Pose, rng: &mut SimRng) -> Vec<Goal> {
        use autonomousim_core::math::quat::yaw;
        match self.kind {
            GoalKind::Spawn => vec![Goal { position: spawn.pos, yaw: yaw(spawn.rot) }],
            GoalKind::Random => {
                let [lo, hi] = region(world, None, self.margin);
                let mut prev = spawn.pos.truncate();
                let mut goals = Vec::with_capacity(self.count);
                for _ in 0..self.count {
                    let mut best = (f64::NEG_INFINITY, DVec3::ZERO);
                    for _ in 0..MAX_ATTEMPTS {
                        let dir = DVec2::from_angle(rng.range(-std::f64::consts::PI, std::f64::consts::PI));
                        let raw = prev + dir * sample(rng, self.distance);
                        let xy = raw.clamp(lo, hi);
                        let p = xy.extend(world.surface_height(xy.x, xy.y) + sample(rng, self.agl));
                        let free = if self.clearance > 0.0 {
                            let c = world.clearance(p, self.clearance) / self.clearance;
                            if world.is_free(p, self.clearance, true, false) { c } else { c.min(0.5) }
                        } else {
                            1.0
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
                    prev = best.1.truncate();
                }
                goals
            }
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
        let ps = spec.sample_positions(&world, 30, 0.05, &mut placed, &mut rng);
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
        let g = goals.sample(&world, &Pose::from_translation(ps[0]), &mut rng);
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
