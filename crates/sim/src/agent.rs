//! Agents: a vehicle of any family with its controller, held action, sensors, goals and wind
//! state.
//!
//! The per-tick work is split into phases so that the world can run them for all agents
//! (in parallel when there are many): [`pre_step`](Agent::pre_step) (air, controller,
//! actuators (rotors, or drive and tyres), static and agent contact forces) and [`post_step`](Agent::post_step) (integration, events,
//! shape) run back to back per agent; [`sense`](Agent::sense) needs every agent's new shape
//! and runs after all of them.

use crate::driver::{DriverGeometry, DriverSpec, RoadDriver};
use crate::events::Events;
use crate::interaction::{AgentContacts, AgentGrid, AgentShape, Body, SceneRays, Sphere};
use crate::obs::ObsInput;
use crate::scenario::{CompiledGroup, EventConfig, Goal, GroundEventConfig, Placement};
use crate::traffic::{RoadTrack, Signals};
use autonomousim_control::ground::GroundEstimate;
use autonomousim_control::multirotor::StateEstimate;
use autonomousim_control::{Command, Controller};
use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::HitKind;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::yaw;
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::terrain::Terrain;
use autonomousim_core::time::Clock;
use autonomousim_sensors::{BodyKinematics, Sensor, SensorEnv};
use autonomousim_vehicles::fixedwing::FixedWingInit;
use autonomousim_vehicles::ground::{Wheeled, WheeledInit};
use autonomousim_vehicles::multirotor::{AirData, GroundPlane, InitialState, MAX_ROTORS, MultirotorScales};
use autonomousim_vehicles::rotorcraft::HelicopterInit;
use autonomousim_vehicles::tiltrotor::TiltrotorInit;
use autonomousim_vehicles::{Family, Vehicle};
use autonomousim_world::environment::{Dryden, EnvironmentConfig, MagneticField};
use autonomousim_world::{Polyline, StaticWorld};
use glam::{DQuat, DVec2, DVec3};
use smallvec::SmallVec;
use std::sync::Arc;

/// Heights above the surface (m) below which rotors see ground effect (wings: 1.2 spans).
const GROUND_EFFECT_RANGE: f64 = 5.0;

/// Environment of one episode.
#[derive(Clone, Debug)]
pub struct EnvState {
    pub config: EnvironmentConfig,
    pub magnetic: MagneticField,
    pub gravity: DVec3,
    pub turbulence_axis: DVec2,
    /// Altitude of the map origin above sea level (m).
    pub origin_altitude: f64,
}

impl EnvState {
    pub fn new(config: EnvironmentConfig, world: &StaticWorld) -> Self {
        let origin = &world.meta.geo_origin;
        Self {
            magnetic: config.magnetic_field(origin),
            gravity: DVec3::new(0.0, 0.0, -config.gravity),
            turbulence_axis: config.wind.turbulence_axis(),
            origin_altitude: origin.origin.alt,
            config,
        }
    }
}

/// One simulated vehicle and everything attached to it.
#[derive(Clone, Debug)]
pub struct Agent {
    /// Index in the world (agents are numbered group by group).
    pub id: u32,
    pub group: usize,
    pub vehicle: Vehicle,
    pub controller: Controller,
    command: Command,
    /// Rotor speed commands of a multirotor.
    cmd: [f64; MAX_ROTORS],
    /// Normalised action held since the last policy step (zeros after a reset).
    pub action: SmallVec<[f64; MAX_ROTORS]>,
    pub sensors: Vec<Sensor>,
    /// Unit carrying each sensor (see `SensorSpec::unit`).
    sensor_units: Vec<usize>,
    pub goals: Vec<Goal>,
    /// Index of the current goal; `goals.len()` once the last one has been reached.
    pub goal_index: usize,
    /// The line the `road` and `route` observation terms follow: the lane line of a `route`
    /// goal, or the planned path to the current goal (`path` goals).
    pub route: Option<Arc<Polyline>>,
    /// Planned paths to each goal (`path` goals): `route` is the current goal's (the last one's
    /// once all are reached).
    pub legs: Vec<Arc<Polyline>>,
    /// Reach radius of the goals (0: advanced explicitly only).
    goal_radius: f64,
    /// Events since the start of the current policy step.
    pub events: Events,
    /// The agent it rested on at the end of the last physics step (see
    /// [`AgentContacts::support`]).
    pub support: Option<u32>,
    /// Where a ground vehicle is on the roads (urban maps; see [`traffic`](crate::traffic)).
    pub track: RoadTrack,
    /// The scripted driver of an agent of a `driver` group.
    pub driver: Option<RoadDriver>,
    pub disabled: bool,
    pub spawn: Pose,
    air: AirData,
    ground: Option<GroundPlane>,
    agl: f64,
    /// Where a ground vehicle last moved away from, and for how long it has not (s).
    anchor: DVec3,
    still_time: f64,
    turbulence: Dryden,
    turbulence_rng: SimRng,
}

/// Kinematics of unit `u` (≥ 1) of a wheeled vehicle at the start of the last physics step,
/// for the sensors it carries (no accelerations: those carry no inertial sensors).
fn unit_kinematics(w: &Wheeled, u: usize, wind: DVec3) -> BodyKinematics {
    let pose = w.unit_pose(u);
    let (velocity, omega) = w.unit_velocity(u);
    BodyKinematics {
        position: pose.pos,
        attitude: pose.rot,
        velocity,
        rates: pose.rot.inverse() * omega,
        specific_force: DVec3::ZERO,
        ang_acc: DVec3::ZERO,
        wind,
    }
}

impl Agent {
    pub(crate) fn new(group: &CompiledGroup, group_index: usize, id: u32, clock: &Clock) -> Self {
        let sensors = group
            .spec
            .sensors
            .iter()
            .map(|s| Sensor::new(&s.config, clock, Seed::from_u64(0)).expect("validated when compiled"))
            .collect();
        let sensor_units = group.spec.sensors.iter().map(|s| s.unit).collect();
        let vehicle = Vehicle::new(&group.def, clock.dt());
        Self {
            id,
            group: group_index,
            command: Command::hold(&vehicle),
            vehicle,
            controller: group.controller.clone(),
            cmd: [0.0; MAX_ROTORS],
            action: SmallVec::from_elem(0.0, group.act_dim()),
            sensors,
            sensor_units,
            goals: vec![Goal::default()],
            goal_index: 0,
            route: None,
            legs: Vec::new(),
            goal_radius: group.spec.goals.radius,
            events: Events::NONE,
            support: None,
            track: RoadTrack::default(),
            driver: group.spec.driver.as_ref().map(|DriverSpec::Road(r)| RoadDriver::new(r, DriverGeometry::of(group))),
            disabled: false,
            spawn: Pose::IDENTITY,
            air: AirData::default(),
            ground: None,
            agl: 0.0,
            anchor: DVec3::ZERO,
            still_time: 0.0,
            turbulence: Dryden::default(),
            turbulence_rng: Seed::from_u64(0).rng(),
        }
    }

    /// Start an episode. `seed` is this agent's stream of the episode; `scales` perturb a
    /// multirotor's parameters. A ground vehicle's placement is its chassis pose (at rest on
    /// the terrain, see [`ground_pose`](crate::drive::ground_pose)).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reset(
        &mut self,
        group: &CompiledGroup,
        placement: &Placement,
        scales: Option<&MultirotorScales>,
        goals: Vec<Goal>,
        seed: Seed,
        env: &EnvState,
        world: &StaticWorld,
    ) {
        let pose = match &mut self.vehicle {
            Vehicle::Multirotor(v) => {
                if let Some(s) = scales {
                    v.set_scales(s);
                }
                v.reset(&InitialState {
                    pose: placement.pose,
                    lin_vel_world: placement.lin_vel,
                    ang_vel_body: placement.ang_vel,
                    motors: placement.motors,
                    soc: 1.0,
                });
                placement.pose
            }
            Vehicle::Wheeled(v) => {
                v.reset(&WheeledInit {
                    pose: placement.pose,
                    lin_vel_world: placement.lin_vel,
                    ang_vel_body: placement.ang_vel,
                });
                placement.pose
            }
            Vehicle::FixedWing(v) => {
                let init = match placement.fixed_wing {
                    Some(s) => FixedWingInit {
                        pose: placement.pose,
                        lin_vel_world: placement.lin_vel,
                        ang_vel_body: placement.ang_vel,
                        controls: s.controls,
                        rotor_speed: Some(s.rotor_speed),
                        soc: 1.0,
                    },
                    None => FixedWingInit::at_rest(placement.pose),
                };
                v.reset(&init);
                placement.pose
            }
            Vehicle::Helicopter(v) => {
                let init = match placement.helicopter {
                    Some(s) => HelicopterInit {
                        pose: placement.pose,
                        lin_vel_world: placement.lin_vel,
                        ang_vel_body: placement.ang_vel,
                        controls: s.controls,
                        rotor_speed: Some(s.rotor_speed),
                        density: s.density,
                    },
                    None => HelicopterInit::at_rest(v.def(), placement.pose),
                };
                v.reset(&init);
                placement.pose
            }
            Vehicle::Tiltrotor(v) => {
                let base = TiltrotorInit {
                    lin_vel_world: placement.lin_vel,
                    ang_vel_body: placement.ang_vel,
                    ..TiltrotorInit::at_rest(placement.pose)
                };
                v.reset(&match placement.tiltrotor {
                    Some(s) => TiltrotorInit {
                        controls: s.controls,
                        rotor_speed: Some(s.rotor_speed),
                        density: s.density,
                        ..base
                    },
                    None => base,
                });
                placement.pose
            }
        };
        self.controller.reset(&self.vehicle);
        // Hold the spawn pose (or the trim) until the first action arrives.
        self.command = Command::hold(&self.vehicle);
        self.action.clear();
        self.action.resize(group.act_dim(), 0.0);
        let sensor_seed = seed.child("sensor");
        for (s, spec) in self.sensors.iter_mut().zip(&group.spec.sensors) {
            s.reset(sensor_seed.child(&spec.name));
        }
        self.goals = goals;
        self.goal_index = 0;
        self.events = Events::NONE;
        self.support = None;
        self.disabled = false;
        self.spawn = pose;
        self.anchor = pose.pos;
        self.still_time = 0.0;
        self.turbulence_rng = seed.child("turbulence").rng();
        self.turbulence = Dryden::stationary(&mut self.turbulence_rng);
        if self.vehicle.gust_span().is_some() {
            self.turbulence = self.turbulence.with_rotational(&mut self.turbulence_rng);
        }
        self.update_air(world, env, 0.0, Some(0.0));
        self.reset_track(world);
    }

    /// Start lane tracking from the current pose (ground vehicles on maps with road sections).
    pub(crate) fn reset_track(&mut self, world: &StaticWorld) {
        self.track = RoadTrack::default();
        if self.vehicle.as_wheeled().is_some() {
            let pose = self.vehicle.pose();
            self.track.reset(world.roads(), pose.pos.truncate(), yaw(pose.rot));
        }
    }

    /// Update lane tracking after a policy step (see [`traffic`](crate::traffic)).
    pub(crate) fn update_track(&mut self, world: &StaticWorld, signals: &Signals, t: f64, cfg: &EventConfig) {
        if self.disabled || self.vehicle.as_wheeled().is_none() || !world.roads().has_sections() {
            return;
        }
        let pose = self.vehicle.pose();
        let v = (pose.rot * self.vehicle.lin_vel_body()).truncate();
        let speed = cfg.ground.off_road_speed;
        self.events |= self.track.update(world.roads(), signals, pose.pos.truncate(), yaw(pose.rot), v, t, speed);
    }

    /// Hold `action` (normalised; clipped to [−1, 1], non-finite read as 0) until the next one.
    pub fn set_action(&mut self, group: &CompiledGroup, action: &[f32]) {
        assert_eq!(action.len(), self.action.len(), "action length of agent {}", self.id);
        for (a, &x) in self.action.iter_mut().zip(action) {
            let x = f64::from(x);
            *a = if x.is_finite() { x.clamp(-1.0, 1.0) } else { 0.0 };
        }
        self.command = group.action_map.command(&self.action, &self.vehicle);
    }

    /// Command the controller directly (scripted agents, the viewer); the command must be of
    /// the vehicle's family.
    pub fn set_command(&mut self, command: impl Into<Command>) {
        let command = command.into();
        assert!(command.fits(self.vehicle.family()), "command {command:?} for a {} vehicle", self.vehicle.family());
        self.command = command;
    }

    pub fn command(&self) -> &Command {
        &self.command
    }

    /// The current goal (the last one once all have been reached).
    pub fn goal(&self) -> Goal {
        self.goals[self.goal_index.min(self.goals.len() - 1)]
    }

    /// Whether the last goal has been reached (only with a reach radius).
    pub fn finished(&self) -> bool {
        self.goal_index >= self.goals.len()
    }

    /// Move on to the next goal; false if this was the last one.
    pub fn advance_goal(&mut self) -> bool {
        if self.goal_index + 1 < self.goals.len() {
            self.goal_index += 1;
            self.follow_leg();
            true
        } else {
            false
        }
    }

    /// Air data the rotors see (density, wind including turbulence and gusts).
    pub fn air(&self) -> &AirData {
        &self.air
    }

    /// Height above the ground or water surface at the start of the last tick (m).
    pub fn agl(&self) -> f64 {
        self.agl
    }

    pub fn kinematics(&self) -> BodyKinematics {
        let v = &self.vehicle;
        let attitude = v.orientation();
        BodyKinematics {
            position: v.position(),
            attitude,
            velocity: attitude * v.lin_vel_body(),
            rates: v.ang_vel_body(),
            specific_force: v.specific_force_body(),
            ang_acc: v.ang_acc_body(),
            wind: self.air.wind,
        }
    }

    fn disable(&mut self) {
        self.disabled = true;
        self.events |= Events::DISABLED;
    }

    /// Ground plane below the vehicle every tick; wind, turbulence and density when
    /// `env_step` is given (the time since the last update, or `Some(0)` to recompute without
    /// advancing the turbulence).
    fn update_air(&mut self, world: &StaticWorld, env: &EnvState, time: f64, env_step: Option<f64>) {
        let p = self.vehicle.position();
        let plane = GroundPlane::below(world.terrain(), p, f64::INFINITY);
        self.agl = plane.map_or(f64::INFINITY, |g| p.z - g.point.z);
        self.ground = plane.filter(|_| self.agl < self.vehicle.ground_effect_range(GROUND_EFFECT_RANGE));
        let Some(dt) = env_step else { return };
        let w = &env.config.wind;
        let agl = self.agl.max(0.0);
        let mut wind = w.steady_at(agl, time);
        let mut gust_rates = DVec3::ZERO;
        if w.has_turbulence() {
            let scales = w.turbulence_scales(agl);
            let span = self.vehicle.gust_span();
            if dt > 0.0 {
                let airspeed = (self.vehicle.lin_vel_world() - self.air.wind).length();
                self.turbulence.step(dt, airspeed, &scales, &mut self.turbulence_rng);
                if let Some(b) = span {
                    self.turbulence.step_rotational(dt, airspeed, b, &mut self.turbulence_rng);
                }
            }
            wind += self.turbulence.velocity(&scales, env.turbulence_axis);
            gust_rates = span.map_or(DVec3::ZERO, |b| self.turbulence.rates(&scales, b));
        }
        let state = env.config.atmosphere.at_altitude(env.origin_altitude + p.z);
        self.air = AirData { density: state.density, wind, speed_of_sound: state.speed_of_sound, gust_rates };
    }

    /// Phase 1: air, controller, rotor forces, contacts with the static world and with other
    /// agents (computed before, from the shapes at the start of the tick).
    pub(crate) fn pre_step(
        &mut self,
        world: &StaticWorld,
        env: &EnvState,
        time: f64,
        env_step: Option<f64>,
        agents: &AgentContacts,
    ) {
        if self.disabled {
            return;
        }
        self.update_air(world, env, time, env_step);
        let scene =
            StaticScene { terrain: world.terrain(), obstacles: world.obstacles(), materials: world.materials() };
        match (&mut self.vehicle, &mut self.controller, &self.command) {
            (Vehicle::Multirotor(v), Controller::Multirotor(c), Command::Multirotor(sp)) => {
                if v.battery().is_some() {
                    let (lo, hi) = v.speed_range();
                    c.set_speed_limits(lo, hi);
                }
                let n = v.num_rotors();
                c.update(sp, &StateEstimate::of(v), &mut self.cmd[..n]);
                v.begin_step();
                v.apply_rotors(&self.cmd[..n], &self.air, self.ground.as_ref());
                v.apply_contacts(&scene);
            }
            (Vehicle::Wheeled(v), Controller::Ground(c), Command::Ground(sp)) => {
                let input = c.update(sp, &GroundEstimate::of(v));
                v.begin_step();
                v.apply_drive(&input, &self.air);
                v.apply_tires(&scene);
                v.apply_contacts(&scene);
            }
            (Vehicle::FixedWing(v), Controller::FixedWing(c), Command::FixedWing(sp)) => {
                let input = c.update(sp, v);
                v.begin_step();
                v.apply_controls(&input, &self.air, self.ground.as_ref());
                v.apply_gear(&scene);
                v.apply_contacts(&scene);
            }
            (Vehicle::Helicopter(v), Controller::Helicopter(c), Command::Helicopter(sp)) => {
                let input = c.update(sp, v);
                v.begin_step();
                v.apply_controls(&input, &self.air, self.ground.as_ref());
                v.apply_contacts(&scene);
            }
            (Vehicle::Tiltrotor(v), Controller::Tiltrotor(c), Command::Tiltrotor(sp)) => {
                let input = c.update(sp, v);
                v.begin_step();
                v.apply_controls(&input, &self.air, self.ground.as_ref());
                v.apply_contacts(&scene);
            }
            (v, c, sp) => unreachable!("{} vehicle with a {} controller and {sp:?}", v.family(), c.family()),
        }
        match &mut self.vehicle {
            Vehicle::Wheeled(v) => {
                for &(force, point, link) in &agents.forces {
                    v.apply_force_on(link.into(), force, point);
                }
            }
            v => {
                for &(force, point, _) in &agents.forces {
                    v.apply_force(force, point);
                }
            }
        }
        if agents.crashed {
            self.events |= Events::CRASH_AGENT;
        }
    }

    /// Phase 2: integrate, then derive the events of the step and the new shape.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn post_step(
        &mut self,
        world: &StaticWorld,
        env: &EnvState,
        dt: f64,
        events: &EventConfig,
        disable_on_terminal: bool,
        agents: &AgentContacts,
        shape: &mut AgentShape,
    ) {
        if self.disabled {
            shape.active = false;
            return;
        }
        let ok = self.vehicle.finish_step(env.gravity).is_ok();
        let state = self.vehicle.state();
        let finite = ok && state.q.iter().chain(state.v.iter()).all(|x| x.is_finite());
        if !finite {
            self.events |= Events::NAN;
            self.disable();
            shape.active = false;
            return;
        }
        self.update_shape(shape);
        self.support = agents.support.map(|s| s.agent);
        self.events |= self.detect_events(world, events, shape, agents);
        if self.vehicle.family() == Family::Wheeled {
            let e = self.ground_events(dt, &events.ground);
            self.events |= e;
        }
        if self.events.is_terminal() {
            if disable_on_terminal {
                self.disable();
                shape.active = false;
            }
        } else {
            self.check_goal();
        }
    }

    /// Move on to the next goal when the current one is within the reach radius.
    fn check_goal(&mut self) {
        let r = self.goal_radius;
        let Some(goal) = self.goals.get(self.goal_index) else { return };
        if r > 0.0 && self.vehicle.position().distance_squared(goal.position) <= r * r {
            self.goal_index += 1;
            self.follow_leg();
            self.events |= Events::GOAL_REACHED;
            if self.goal_index == self.goals.len() {
                self.events |= Events::FINISHED;
            }
        }
    }

    /// Route along the planned path to the current goal, if there are planned paths.
    pub(crate) fn follow_leg(&mut self) {
        if let Some(leg) = self.legs.get(self.goal_index.min(self.legs.len().saturating_sub(1))) {
            self.route = Some(leg.clone());
        }
    }

    pub(crate) fn update_shape(&self, shape: &mut AgentShape) {
        let v = &self.vehicle;
        let pose = v.pose();
        shape.active = !self.disabled;
        shape.id = self.id;
        shape.center = pose.pos;
        shape.velocity = pose.rot * v.lin_vel_body();
        shape.ang_vel = pose.rot * v.ang_vel_body();
        shape.spheres.clear();
        shape.bodies.clear();
        // Units behind the towing unit are bodies 1…; their frames as of the step's start.
        let units = v.as_wheeled().map_or(1, |w| w.num_units());
        let unit_pose = |u: usize| match (u, v.as_wheeled()) {
            (0, _) | (_, None) => pose,
            (u, Some(w)) => w.unit_pose(u),
        };
        if let Some(w) = v.as_wheeled() {
            for u in 1..units {
                let origin = w.unit_pose(u).pos;
                let (velocity, ang_vel) = w.unit_velocity(u);
                shape.bodies.push(Body { origin, velocity, ang_vel, link: w.unit_link(u) as u16 });
            }
        }
        let mut radius: f64 = 0.0;
        for c in v.colliders() {
            let body = v.as_wheeled().and_then(|w| w.link_unit(c.link)).unwrap_or(0);
            let center = unit_pose(body).transform_point(c.center);
            radius = radius.max(center.distance(pose.pos) + c.radius);
            let gear = v.is_gear(c.group);
            shape.spheres.push(Sphere { center, radius: c.radius, friction: c.friction, gear, body: body as u8 });
        }
        if let Vehicle::Wheeled(w) = v {
            for i in 0..w.num_wheels() {
                let (center, r) = (w.wheel_pose(i).pos, w.wheel_radius(i));
                radius = radius.max(center.distance(pose.pos) + r);
                let body = w.def().wheel_unit(i) as u8;
                shape.spheres.push(Sphere { center, radius: r, friction: 1.0, gear: true, body });
            }
        }
        shape.radius = radius;
    }

    fn detect_events(
        &self,
        world: &StaticWorld,
        cfg: &EventConfig,
        shape: &AgentShape,
        agents: &AgentContacts,
    ) -> Events {
        let v = &self.vehicle;
        let mut e = if agents.supported { Events::GROUND_CONTACT } else { Events::NONE };
        let colliders = v.colliders();
        let mut crashed = false;
        for c in v.contacts() {
            let gear = v.is_gear(colliders[c.collider as usize].group);
            let crash = !gear || c.normal_velocity < -cfg.crash_speed;
            match c.kind {
                HitKind::Terrain | HitKind::Solid(_) if crash => {
                    crashed = true;
                    e |= if c.kind == HitKind::Terrain { Events::CRASH_TERRAIN } else { Events::CRASH_OBSTACLE };
                }
                HitKind::Terrain | HitKind::Solid(_) => e |= Events::GROUND_CONTACT,
                HitKind::Foliage(_) => e |= Events::FOLIAGE,
                HitKind::Water => e |= Events::WATER,
                HitKind::Agent(_) => {}
            }
        }
        // Landed: at rest, relative to the agent carrying it if any.
        let (lin, ang) = match agents.support {
            Some(s) => {
                let rot = v.orientation();
                ((rot * v.lin_vel_body() - s.velocity).length(), (rot * v.ang_vel_body() - s.ang_vel).length())
            }
            None => (v.lin_vel_body().length(), v.ang_vel_body().length()),
        };
        if e.contains(Events::GROUND_CONTACT) && !crashed && lin < cfg.landed_speed && ang < cfg.landed_rate {
            e |= Events::LANDED;
        }
        let p = shape.center;
        let terrain = world.terrain();
        if let Some(w) = terrain.water_level(p.x, p.y)
            && p.z - shape.radius < w
            && shape
                .spheres
                .iter()
                .any(|s| terrain.water_level(s.center.x, s.center.y).is_some_and(|w| s.center.z - s.radius < w))
        {
            e |= Events::WATER;
        }
        if v.as_fixed_wing().is_some_and(|f| f.stalled()) {
            e |= Events::STALL;
        }
        let (lo, hi) = world.extent();
        let m = cfg.bounds_margin;
        let outside = p.x < lo.x + m || p.x > hi.x - m || p.y < lo.y + m || p.y > hi.y - m;
        let too_high = cfg.ceiling.is_some_and(|c| p.z > c) || cfg.max_agl.is_some_and(|h| self.agl > h);
        if outside || too_high {
            e |= Events::OUT_OF_BOUNDS;
        }
        e
    }

    /// Rollover and stuck (ground vehicles).
    fn ground_events(&mut self, dt: f64, cfg: &GroundEventConfig) -> Events {
        let mut e = Events::NONE;
        let tilted = |rot: DQuat| (rot * DVec3::Z).z < cfg.rollover_deg.to_radians().cos();
        if tilted(self.vehicle.orientation()) {
            e |= Events::ROLLOVER;
        }
        if let Some(w) = self.vehicle.as_wheeled() {
            if (1..w.num_units()).any(|u| tilted(w.unit_pose(u).rot)) {
                e |= Events::ROLLOVER;
            }
            if w.articulations().any(|(a, _)| a.abs() > cfg.jackknife_deg.to_radians()) {
                e |= Events::JACKKNIFE;
            }
        }
        let p = self.vehicle.position();
        if p.distance_squared(self.anchor) > cfg.stuck_distance * cfg.stuck_distance {
            self.anchor = p;
            self.still_time = 0.0;
        } else {
            self.still_time += dt;
            if cfg.stuck_time > 0.0 && self.still_time >= cfg.stuck_time {
                e |= Events::STUCK;
            }
        }
        e
    }

    /// Phase 3: sensors at the new tick.
    pub(crate) fn sense(
        &mut self,
        tick: u64,
        time: f64,
        world: &StaticWorld,
        env: &EnvState,
        shapes: &[AgentShape],
        index: usize,
    ) {
        if self.disabled || self.sensors.is_empty() {
            return;
        }
        let kin = self.kinematics();
        let rays = SceneRays { world, agents: shapes, exclude: index };
        let senv = SensorEnv {
            world,
            rays: &rays,
            geo: &world.meta.geo_origin,
            atmosphere: &env.config.atmosphere,
            magnetic: &env.magnetic,
            gravity: env.config.gravity,
        };
        for (s, &u) in self.sensors.iter_mut().zip(&self.sensor_units) {
            let unit_kin;
            let kin = match (u, self.vehicle.as_wheeled()) {
                (1.., Some(w)) => {
                    unit_kin = unit_kinematics(w, u, self.air.wind);
                    &unit_kin
                }
                _ => &kin,
            };
            s.update(tick, time, kin, &senv);
        }
    }

    /// Height above the ground or water surface now (m).
    pub fn agl_now(&self, world: &StaticWorld) -> f64 {
        let p = self.vehicle.position();
        p.z - world.surface_height(p.x, p.y)
    }

    /// Write the observation; `agents` are the world's agent shapes, `me` this agent's index.
    pub(crate) fn observe(
        &self,
        group: &CompiledGroup,
        world: &StaticWorld,
        agents: &[AgentShape],
        grid: &AgentGrid,
        me: usize,
        signals: &Signals,
        time: f64,
        out: &mut [f32],
    ) {
        let kin = self.kinematics();
        let (motors, motor_range) = match &self.vehicle {
            Vehicle::Multirotor(v) => (v.motor_speeds(), v.speed_range()),
            _ => (&[][..], (0.0, 1.0)),
        };
        group.obs.write(
            &ObsInput {
                kin: &kin,
                goal: self.goal(),
                agl: self.agl_now(world),
                motors,
                motor_range,
                last_action: &self.action,
                wheeled: self.vehicle.as_wheeled(),
                sensors: &self.sensors,
                world,
                agents,
                me,
                grid,
                route: self.route.as_deref(),
                track: &self.track,
                signals,
                time,
            },
            out,
        );
    }
}
