//! One simulated world: a map, the environment of the current episode and its agents.
//!
//! **Physics tick** (`tick → tick + 1`):
//! 1. contact forces between agents, from their shapes at the start of the tick;
//! 2. per agent: ground plane, wind/turbulence/density (every environment step), controller
//!    at the held setpoint, rotor forces, contacts with the static world and the agent
//!    contacts of step 1; integration, events, new shape;
//! 3. the clock advances, then every agent's sensors sample the new state (skipped when no
//!    agent has sensors).
//!
//! Steps 2 and 3 run in parallel over agents when at least [`PARALLEL_AGENTS`] are simulated
//! in full, or at the ticks kinematic vehicles move when there are at least
//! [`PARALLEL_KINEMATIC`] agents (two fork–joins per tick); each agent only writes its own state, so results do not depend
//! on the thread count. Parallel phases are cheapest when the world is stepped on a rayon
//! worker thread, as [`BatchSim`](crate::BatchSim) does.
//!
//! **Seeding**: a world has a base seed; episode `k` uses `base/episode/k` with the streams
//! `map`, `environment`, `spawn`, `goals` and `agent/<id>` (vehicle parameters, sensors
//! `sensor/<name>`, turbulence). Resetting with an explicit seed replaces the base and restarts
//! the count, so `reset(Some(s))` always reproduces the same episode.

use crate::agent::{Agent, EnvState};
use crate::camera::Capture;
use crate::drive::ground_pose;
use crate::driver::{Driver, DriverGeometry, DriverSpec, Traffic};
use crate::events::Events;
use crate::interaction::{AgentContactState, AgentContacts, AgentGrid, AgentShape, agent_contacts, surface_distance};
use crate::lane;
use crate::lane::RoadSpawn;
use crate::obs::{CLEARANCE_RANGE, Seen};
use crate::pedestrians::Crowd;
use crate::scenario::{
    CompiledScenario, FixedWingStart, Goal, GoalKind, HelicopterStart, PhysicsMode, SpawnSpec, TiltrotorStart,
};
use crate::traffic::Signals;
use crate::traffic_driver::{Elem, LaneIndex, Network, Sweeps, lane_spawns, respawn_spot};
use autonomousim_control::Command;
use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::{from_yaw, yaw};
use autonomousim_core::rng::Seed;
use autonomousim_core::terrain::Terrain;
use autonomousim_core::time::Clock;
use autonomousim_sensors::Sensor;
use autonomousim_vehicles::Vehicle;
use autonomousim_vehicles::fixedwing::{FixedWing, FixedWingInput};
use autonomousim_vehicles::rotorcraft::{Helicopter, HelicopterInput};
use autonomousim_vehicles::tiltrotor::{Tiltrotor, TiltrotorInput, TrimLimits};
use autonomousim_world::StaticWorld;
use autonomousim_world::roads::Polyline;
use glam::{DQuat, DVec2, DVec3};
use rayon::prelude::*;
use std::sync::Arc;

/// Count of agents simulated in full (not kinematic) from which the per-agent phases run in
/// parallel.
pub const PARALLEL_AGENTS: usize = 32;
/// Agent count from which the per-agent phases of a tick at which kinematic vehicles move run
/// in parallel.
pub const PARALLEL_KINEMATIC: usize = 128;

/// With `path` goals a ground vehicle starts facing the point of its path this far away (m),
/// turned by its sampled heading.
const PATH_HEADING_REACH: f64 = 8.0;

/// Look-ahead times (s) at whose predicted positions agents on tiled maps have tiles
/// prefetched (a small aircraft covers a 256 m tile in about 10 s).
const PREFETCH_AHEAD: [f64; 3] = [3.0, 6.0, 10.0];

/// Tiles within this distance (m) of those positions are prefetched.
const PREFETCH_RADIUS: f64 = 150.0;

/// Per-agent state row written by [`WorldInstance::write_state`]: `(name, length)` in order.
/// `goal_index` equals the number of goals once the last one has been reached; `clearance` is
/// the distance to the nearest terrain or solid obstacle surface, up to 20 m (for ground
/// vehicles, which sit on the terrain, to the nearest solid obstacle); `agent_clearance` the
/// distance between the agent's colliders and the nearest other active agent's, up to 20 m;
/// `road` the lateral offset from the lane followed, the heading error to it and the distance
/// to the nearest road's surface ([`lane::road_state`]); `articulation` the yaw of the first
/// two trailers (or dollies) relative to the unit ahead (rad, positive pointing left; 0
/// without); `tail` the x, y and heading of the last unit's tail, the reference point for
/// reversing ([`Wheeled::tail_pose`](autonomousim_vehicles::ground::Wheeled::tail_pose); the
/// position and heading for other vehicles); `sinkage` the mean sinkage of a tracked
/// vehicle's loaded patches into soft soil (m; 0 otherwise); `air_data` the true airspeed
/// (m/s), angle of attack and sideslip (rad) relative to the air at the vehicle
/// ([`AirFlow`](autonomousim_vehicles::aero::AirFlow)); `support` the index in the world of
/// the agent it rests on (agents are numbered group by group; −1: none, see
/// [`AgentContacts::support`](crate::interaction::AgentContacts::support)).
pub const STATE_FIELDS: [(&str, usize); 16] = [
    ("position", 3),
    ("orientation", 4),
    ("velocity", 3),
    ("rates", 3),
    ("goal", 3),
    ("goal_yaw", 1),
    ("agl", 1),
    ("goal_index", 1),
    ("clearance", 1),
    ("agent_clearance", 1),
    ("road", 3),
    ("articulation", 2),
    ("tail", 3),
    ("sinkage", 1),
    ("air_data", 3),
    ("support", 1),
];

/// Length of a state row.
pub const STATE_DIM: usize = 34;

#[derive(Clone, Debug)]
pub struct WorldInstance {
    scenario: Arc<CompiledScenario>,
    map: Arc<StaticWorld>,
    map_index: usize,
    env: EnvState,
    agents: Vec<Agent>,
    shapes: Vec<AgentShape>,
    contacts: Vec<AgentContacts>,
    clock: Clock,
    base_seed: Seed,
    /// Resets done since the base seed was set.
    episode: u64,
    episode_seed: Seed,
    /// Policy steps since the reset.
    steps: u64,
    agent_contact_state: AgentContactState,
    /// Neighbour index over `shapes`, rebuilt after every policy step, reset and change of
    /// an agent's placement or status.
    grid: AgentGrid,
    /// The traffic signals of the current map with the episode's offsets.
    signals: Signals,
    /// Scratch index of the agents on the lanes for traffic drivers (rebuilt every step).
    lane_index: Option<LaneIndex>,
    /// Per group of ground vehicles: its footprint (for the `traffic` term).
    footprints: Vec<Option<DriverGeometry>>,
    /// The pedestrians.
    crowd: Crowd,
    /// Per agent: whether its group learns (respawns keep clear of them).
    learning: Vec<bool>,
}

impl WorldInstance {
    /// A world with base seed `seed`, reset to its first episode.
    pub fn new(scenario: Arc<CompiledScenario>, seed: Seed) -> Self {
        let mut agents = Vec::with_capacity(scenario.num_agents());
        for (gi, g) in scenario.groups.iter().enumerate() {
            for k in 0..g.spec.count {
                agents.push(Agent::new(g, gi, (g.first_agent + k) as u32, &scenario.clock));
            }
        }
        let map = scenario.maps[0].clone();
        let mut w = Self {
            env: EnvState::new(scenario.spec.environment.clone(), &map),
            shapes: vec![AgentShape::default(); agents.len()],
            contacts: vec![AgentContacts::default(); agents.len()],
            agents,
            clock: scenario.clock,
            base_seed: seed,
            episode: 0,
            episode_seed: seed,
            steps: 0,
            agent_contact_state: AgentContactState::default(),
            grid: AgentGrid::default(),
            signals: Signals::default(),
            lane_index: None,
            footprints: scenario.groups.iter().map(|g| g.def.as_wheeled().map(|_| DriverGeometry::of(g))).collect(),
            crowd: Crowd::default(),
            learning: scenario.groups.iter().flat_map(|g| std::iter::repeat_n(g.learning(), g.spec.count)).collect(),
            map,
            map_index: 0,
            scenario,
        };
        w.reset(None);
        w
    }

    /// Start a new episode: the next one of the base seed, or the first one of `seed`.
    pub fn reset(&mut self, seed: Option<u64>) {
        if let Some(s) = seed {
            self.base_seed = Seed::from_u64(s);
            self.episode = 0;
        }
        let es = self.base_seed.child("episode").child_index(self.episode);
        self.episode += 1;
        self.episode_seed = es;
        let sc = self.scenario.clone();

        let pick = sc.episode_maps[es.child("map").rng().below(sc.episode_maps.len() as u64) as usize];
        self.map = sc.maps[pick].clone();
        self.map_index = pick;
        let world = &*self.map;
        let mut env = sc.spec.environment.clone();
        if let Some(r) = &sc.spec.randomize_environment {
            r.apply(&mut env, &mut es.child("environment").rng());
        }
        self.env = EnvState::new(env, world);
        self.signals = Signals::new(world, es.child("signals"));
        self.clock = sc.clock;
        self.steps = 0;
        self.agent_contact_state.clear();

        let mut spawn_rng = es.child("spawn").rng();
        let mut goal_rng = es.child("goals").rng();
        let agent_seed = es.child("agent");
        let mut placed = Vec::with_capacity(self.agents.len());
        let mut parked_bays = Vec::new();
        let mut goal_bays = Vec::new();
        // Footprints of the bay-goal spawns: centre, heading and half extents.
        let mut footprints: Vec<(DVec2, f64, DVec2)> = Vec::new();
        let mut long = Vec::new();
        for g in &sc.groups {
            let spawn = &g.spec.spawn;
            let ground = g.ground(pick);
            let route_goals = g.spec.goals.kind == GoalKind::Route;
            // Height of the centre of mass (the chassis frame for ground vehicles) above the
            // terrain on roads and at route goals.
            let lift = ground.map_or(g.bottom, |gs| gs.ride);
            let bay_goals = g.spec.goals.kind == GoalKind::Bay;
            // Long traffic vehicles keep to the network they can drive on this map.
            let network = match g.spec.driver {
                Some(DriverSpec::Traffic(_)) => Network::of_long(world.roads().lanes(), DriverGeometry::of(g)),
                _ => None,
            };
            if let Some(n) = &network {
                long.push((n.clone(), DriverGeometry::of(g)));
            }
            let mut bays = Vec::new();
            let mut trips = Vec::new();
            let mut crossings = Vec::new();
            let (positions, mut road_spawns) = if bay_goals {
                // Spawn and goal in a farm yard or at a parking bay.
                let d = g.def.as_wheeled().expect("bay goals are for ground vehicles");
                let slots = crate::bay::slots(world, &g.spec.goals.bay);
                let mut used = Vec::new();
                for _ in 0..g.spec.count {
                    let b = crate::bay::sample(world, &slots, &g.spec.goals, d, lift, &mut used, &mut goal_rng)
                        .expect("maps of bay goals have yards or bays");
                    placed.push(b.xy.extend(0.0));
                    // Nobody parks in a goal bay or on the spawn.
                    parked_bays.extend(b.bay);
                    goal_bays.extend(b.bay);
                    let geo = DriverGeometry::of(g);
                    let centre = b.xy + DVec2::from_angle(b.yaw) * (0.5 * (geo.front + geo.rear));
                    footprints.push((centre, b.yaw, DVec2::new(0.5 * (geo.front - geo.rear), geo.half_width)));
                    bays.push(b);
                }
                let p: Vec<DVec3> = bays.iter().map(|b| b.xy.extend(0.0)).collect();
                let n = p.len();
                (p, (0..n).map(|_| None).collect::<Vec<_>>())
            } else if g.spec.goals.kind == GoalKind::Yard {
                // Spawn on one yard's pad, the goal on another's.
                let yards = crate::bay::yards(world);
                let mut used = Vec::new();
                let mut p = Vec::with_capacity(g.spec.count);
                for _ in 0..g.spec.count {
                    let t = crate::bay::trip(world, &yards, &g.spec.goals, &mut used, &mut goal_rng)
                        .expect("maps of yard goals have two yards");
                    let z = if spawn.on_ground {
                        world.terrain().height(t.from.x, t.from.y) + g.bottom + 1e-3
                    } else {
                        world.surface_height(t.from.x, t.from.y) + spawn_rng.range(spawn.agl[0], spawn.agl[1])
                    };
                    let agl = goal_rng.range(g.spec.goals.agl[0], g.spec.goals.agl[1]);
                    let goal = t.to.extend(world.surface_height(t.to.x, t.to.y) + agl);
                    placed.push(t.from.extend(z));
                    p.push(t.from.extend(z));
                    trips.push(Goal { position: goal, yaw: (t.to - t.from).to_angle() });
                }
                let n = p.len();
                (p, (0..n).map(|_| None).collect::<Vec<_>>())
            } else if g.spec.goals.kind == GoalKind::Rooftop {
                // Spawn on a sidewalk or a pad, the goal on another pad.
                let mut used = Vec::new();
                let mut p = Vec::with_capacity(g.spec.count);
                for _ in 0..g.spec.count {
                    let t = crate::rooftop::trip(world, &g.spec.goals, &mut used, &mut goal_rng)
                        .expect("maps of rooftop goals have two pads");
                    let z = t.from.z
                        + if spawn.on_ground { g.bottom + 1e-3 } else { spawn_rng.range(spawn.agl[0], spawn.agl[1]) };
                    let agl = goal_rng.range(g.spec.goals.agl[0], g.spec.goals.agl[1]);
                    let from = t.from.truncate();
                    placed.push(from.extend(z));
                    p.push(from.extend(z));
                    trips.push(Goal { position: t.to.centre + DVec3::Z * agl, yaw: t.to.yaw });
                }
                let n = p.len();
                (p, (0..n).map(|_| None).collect::<Vec<_>>())
            } else if g.spec.goals.kind == GoalKind::Junction {
                // On different entries of one junction, routed through it.
                let d = g.def.as_wheeled().expect("junction goals are for ground vehicles");
                let sites = crate::junction::sites(world, &g.spec.goals.junction.kinds);
                crossings = crate::junction::sample(world, &sites, &g.spec.goals, g.spec.count, lift, &mut goal_rng)
                    .expect("maps of junction goals have a junction with enough arms");
                let mut p = Vec::with_capacity(crossings.len());
                let mut rs = Vec::with_capacity(crossings.len());
                for c in &crossings {
                    let pose = ground_pose(world, d, &g.rest, c.position.truncate(), c.yaw);
                    placed.push(pose.pos);
                    p.push(pose.pos);
                    rs.push(Some(RoadSpawn {
                        position: pose.pos,
                        yaw: c.yaw,
                        route: Some(c.route.clone()),
                        walk: None,
                    }));
                }
                (p, rs)
            } else if spawn.in_bays {
                // Parked: centred in distinct random bays, facing along them.
                let d = g.def.as_wheeled().expect("checked when compiled");
                let geo = DriverGeometry::of(g);
                let mid = 0.5 * (geo.front + geo.rear);
                let all = &world.sites().bays;
                let mut p = Vec::with_capacity(g.spec.count);
                let mut rs = Vec::with_capacity(g.spec.count);
                // Bays whose car would overlap an earlier bay-goal spawn (with 0.3 m to spare)
                // are taken.
                let half = DVec2::new(0.5 * (geo.front - geo.rear), geo.half_width + 0.3);
                for (k, b) in all.iter().enumerate() {
                    if footprints.iter().any(|f| rects_overlap(f, &(b.centre.truncate(), b.yaw, half))) {
                        parked_bays.push(k);
                    }
                }
                // With `near_bay_goals`: the free bays nearest to the goal bays, twice as many
                // as park.
                let near = (spawn.near_bay_goals && !goal_bays.is_empty()).then(|| {
                    let gap = |k: usize| {
                        let c = all[k].centre.truncate();
                        goal_bays.iter().map(|&j| all[j].centre.truncate().distance(c)).fold(f64::INFINITY, f64::min)
                    };
                    let mut free: Vec<usize> = (0..all.len()).filter(|k| !parked_bays.contains(k)).collect();
                    free.sort_by(|&a, &b| gap(a).total_cmp(&gap(b)));
                    free.truncate(2 * g.spec.count);
                    free
                });
                for _ in 0..g.spec.count {
                    let pool = near.as_ref().filter(|n| n.iter().any(|k| !parked_bays.contains(k)));
                    let mut free: Vec<usize> = match pool {
                        Some(n) => n.iter().copied().filter(|k| !parked_bays.contains(k)).collect(),
                        None => (0..all.len()).filter(|k| !parked_bays.contains(k)).collect(),
                    };
                    if free.is_empty() {
                        // (Only taken bays left: any not parked in.)
                        let parked = &parked_bays[parked_bays.len() - p.len()..];
                        free = (0..all.len()).filter(|k| !parked.contains(k)).collect();
                    }
                    let k = free[spawn_rng.below(free.len() as u64) as usize];
                    parked_bays.push(k);
                    let b = &all[k];
                    let xy = b.point(DVec2::new(-mid, 0.0));
                    let pose = ground_pose(world, d, &g.rest, xy, b.yaw);
                    placed.push(pose.pos);
                    p.push(pose.pos);
                    rs.push(Some(RoadSpawn { position: pose.pos, yaw: b.yaw, route: None, walk: None }));
                }
                (p, rs)
            } else if spawn.on_road
                && matches!(g.spec.driver, Some(DriverSpec::Traffic(_)))
                && let Some(ls) = lane_spawns(
                    world,
                    g.spec.count,
                    -DriverGeometry::of(g).rear,
                    network.as_ref().map(|n| n.lanes.as_slice()),
                    DriverGeometry::of(g).single_track,
                    spawn.min_separation,
                    lift,
                    &mut placed,
                    &mut spawn_rng,
                )
            {
                let rs = ls.into_iter().map(|(position, yaw)| RoadSpawn { position, yaw, route: None, walk: None });
                let rs: Vec<RoadSpawn> = rs.collect();
                (rs.iter().map(|r| r.position).collect(), rs.into_iter().map(Some).collect())
            } else if spawn.on_road {
                let only = self.agents[g.first_agent].driver.as_ref().and_then(Driver::as_road).map(|d| d.roads(world));
                let rs = lane::road_spawns(
                    world,
                    spawn,
                    route_goals.then_some(&g.spec.goals),
                    g.spec.count,
                    lift,
                    only.as_deref(),
                    &mut placed,
                    &mut spawn_rng,
                    &mut goal_rng,
                );
                (rs.iter().map(|r| r.position).collect(), rs.into_iter().map(Some).collect())
            } else if let Some(near) = &spawn.near {
                // One at a time, in a square around its agent of the other group.
                let other = sc.groups.iter().find(|o| o.spec.name == near.group).expect("checked when compiled");
                let [lo, hi] = crate::scenario::region(world, spawn.region, spawn.margin);
                let mut p = Vec::with_capacity(g.spec.count);
                for k in 0..g.spec.count {
                    let at = self.agents[other.first_agent + k % other.spec.count].vehicle.pose().pos.truncate();
                    let (a, b) = ((at - near.offset).clamp(lo, hi), (at + near.offset).clamp(lo, hi));
                    let square = SpawnSpec { region: Some([a, b]), margin: 0.0, cluster: None, ..spawn.clone() };
                    p.extend(square.sample_positions(world, 1, g.bottom, ground, &mut placed, &mut spawn_rng));
                }
                let n = p.len();
                (p, (0..n).map(|_| None).collect::<Vec<_>>())
            } else {
                let p = spawn.sample_positions(world, g.spec.count, g.bottom, ground, &mut placed, &mut spawn_rng);
                let n = p.len();
                (p, (0..n).map(|_| None).collect::<Vec<_>>())
            };
            let mut formation = match g.spec.goals.kind {
                GoalKind::Formation => g.spec.goals.formation(world, &positions, ground, &mut goal_rng),
                _ => Vec::new(),
            }
            .into_iter();
            for (k, p) in positions.into_iter().enumerate() {
                let id = g.first_agent + k;
                let density = self.env.config.atmosphere.density(self.env.origin_altitude + p.z);
                let hover = g.def.as_multirotor().map_or(0.0, |d| d.hover_omega(self.env.config.gravity, density));
                let mut placement = spawn.sample_state(p, hover, &mut spawn_rng);
                let road_spawn = road_spawns[k].take();
                if let Some(rs) = &road_spawn {
                    placement.pose.rot = from_yaw(rs.yaw);
                }
                if let Some(b) = bays.get(k) {
                    placement.pose.rot = from_yaw(b.yaw);
                }
                if let Some(d) = g.def.as_wheeled() {
                    placement.pose = ground_pose(world, d, &g.rest, p.truncate(), yaw(placement.pose.rot));
                }
                if let Some(d) = g.def.as_fixed_wing() {
                    if spawn.on_ground {
                        placement.pose.rot *= g.rest.rot;
                    } else {
                        let gravity = self.env.config.gravity;
                        let airspeed = match spawn.airspeed {
                            Some([lo, hi]) => lo + (hi - lo) * spawn_rng.uniform(),
                            None => 1.5 * d.stall_speed(density, gravity),
                        };
                        let aircraft = self.agents[id].vehicle.as_fixed_wing().expect("fixed-wing group");
                        let (attitude, v_body, start) = fixed_wing_start(aircraft, airspeed, density, gravity);
                        let agl = (p.z - world.surface_height(p.x, p.y)).max(0.0);
                        placement.pose.rot *= attitude;
                        placement.lin_vel += placement.pose.rot * v_body + self.env.config.wind.steady_at(agl, 0.0);
                        placement.fixed_wing = Some(start);
                    }
                }
                if g.def.as_helicopter().is_some() && !spawn.on_ground {
                    let speed = match spawn.airspeed {
                        Some([lo, hi]) => lo + (hi - lo) * spawn_rng.uniform(),
                        None => 0.0,
                    };
                    let heli = self.agents[id].vehicle.as_helicopter().expect("helicopter group");
                    let (attitude, v_body, start) = helicopter_start(heli, speed, density, self.env.config.gravity);
                    let agl = (p.z - world.surface_height(p.x, p.y)).max(0.0);
                    placement.pose.rot *= attitude;
                    placement.lin_vel += placement.pose.rot * v_body + self.env.config.wind.steady_at(agl, 0.0);
                    placement.helicopter = Some(start);
                }
                if g.def.as_tiltrotor().is_some() && !spawn.on_ground {
                    let speed = match spawn.airspeed {
                        Some([lo, hi]) => lo + (hi - lo) * spawn_rng.uniform(),
                        None => 0.0,
                    };
                    let tilt = self.agents[id].vehicle.as_tiltrotor().expect("tiltrotor group");
                    let (attitude, v_body, start) = tiltrotor_start(tilt, speed, density, self.env.config.gravity);
                    let agl = (p.z - world.surface_height(p.x, p.y)).max(0.0);
                    placement.pose.rot *= attitude;
                    placement.lin_vel += placement.pose.rot * v_body + self.env.config.wind.steady_at(agl, 0.0);
                    placement.tiltrotor = Some(start);
                }
                let walk = road_spawn.as_ref().and_then(|rs| rs.walk);
                let mut route = road_spawn.and_then(|rs| rs.route);
                if route_goals && route.is_none() {
                    route = lane::plan_route(world, p.truncate(), &g.spec.goals, &mut goal_rng);
                }
                let goals = match (formation.next(), &route) {
                    _ if bay_goals => vec![bays[k].goal],
                    _ if !trips.is_empty() => vec![trips[k]],
                    _ if !crossings.is_empty() => vec![crossings[k].goal],
                    (Some(slot), _) => vec![slot],
                    (None, Some(lane)) => lane::route_goals(world, lane, g.spec.goals.route.step, lift),
                    (None, None) => g.spec.goals.sample(world, &placement.pose, ground, &mut goal_rng),
                };
                let seed = agent_seed.child_index(id as u64);
                let scales = g
                    .def
                    .as_multirotor()
                    .map(|d| g.spec.randomize.sample(d.rotors.len(), &mut seed.child("vehicle").rng()));
                let agent = &mut self.agents[id];
                // Planned paths through `path` goals.
                let legs = match ground.filter(|_| g.spec.goals.path) {
                    Some(gs) => {
                        let targets: Vec<DVec2> = goals.iter().map(|q| q.position.truncate()).collect();
                        let lift = |p: DVec2| p.extend(world.terrain().height(p.x, p.y));
                        gs.grid
                            .legs(placement.pose.pos.truncate(), &targets)
                            .map(|legs| {
                                legs.into_iter()
                                    .map(|l| Arc::new(Polyline::new(l.into_iter().map(lift).collect())))
                                    .collect()
                            })
                            .unwrap_or_default()
                    }
                    None => Vec::new(),
                };
                // The vehicle starts facing along its path, turned by the sampled heading.
                if let (Some(d), Some(leg)) = (g.def.as_wheeled(), legs.first()) {
                    let p0 = placement.pose.pos.truncate();
                    let pts = leg.points();
                    let ahead = pts.iter().map(|q| q.truncate()).find(|q| q.distance(p0) >= PATH_HEADING_REACH);
                    if let Some(q) = ahead.or_else(|| pts.last().map(|q| q.truncate())).filter(|q| q.distance(p0) > 0.5)
                    {
                        let dir = q - p0;
                        let heading = dir.y.atan2(dir.x) + yaw(placement.pose.rot);
                        placement.pose = ground_pose(world, d, &g.rest, p0, heading);
                    }
                }
                agent.reset(g, &placement, scales.as_ref(), goals, seed, &self.env, world);
                agent.route = route.map(Arc::new);
                agent.legs = legs;
                agent.follow_leg();
                match &mut agent.driver {
                    Some(Driver::Road(d)) => agent.route = d.reset(seed.child("driver"), world, walk),
                    Some(Driver::Traffic(d)) => {
                        d.set_network(network.clone());
                        d.reset(seed.child("driver"), world, &agent.vehicle.pose());
                        if let Some(c) = crossings.get(k) {
                            d.take_route(&c.connectors, world);
                        }
                    }
                    Some(Driver::Parked) | None => {}
                }
                let contact = g.def.contact_model(agent.vehicle.mass(), sc.dt());
                self.shapes[id].set_contact(&contact.solid);
                agent.update_shape(&mut self.shapes[id]);
            }
        }
        // Stop lines set back clear of the long vehicles' sweeps.
        if sc.groups.iter().any(|g| matches!(g.spec.driver, Some(DriverSpec::Traffic(_)))) {
            let lanes = world.roads().lanes();
            let mut index = LaneIndex::new(lanes);
            index.set_sweeps(Arc::new(Sweeps::new(lanes, &long)));
            self.lane_index = Some(index);
        }
        self.crowd.reset(&sc.spec.pedestrians, world, es.child("pedestrians"), &self.shapes);
        self.grid.build(&self.shapes);
    }

    // ------------------------------------------------------------------------ stepping

    /// Hold normalised actions for all agents of `group` (`count × act_dim` values).
    pub fn set_actions(&mut self, group: usize, actions: &[f32]) {
        let g = &self.scenario.groups[group];
        let dim = g.act_dim();
        assert_eq!(actions.len(), g.spec.count * dim, "action array of group {:?}", g.spec.name);
        for (k, a) in actions.chunks_exact(dim).enumerate() {
            self.agents[g.first_agent + k].set_action(g, a);
        }
    }

    /// Hold a normalised action for one agent.
    pub fn set_action(&mut self, agent: usize, action: &[f32]) {
        let g = &self.scenario.groups[self.agents[agent].group];
        self.agents[agent].set_action(g, action);
    }

    /// Command one agent's controller directly (a multirotor
    /// [`Setpoint`](autonomousim_control::multirotor::Setpoint) or a
    /// [`GroundSetpoint`](autonomousim_control::ground::GroundSetpoint)).
    pub fn set_command(&mut self, agent: usize, command: impl Into<Command>) {
        self.agents[agent].set_command(command);
    }

    /// One policy step (`decimation` physics ticks).
    pub fn step(&mut self) {
        self.step_with(&mut |_| {});
    }

    /// One policy step, calling `after_tick` after every physics tick (e.g. a recorder).
    pub fn step_with(&mut self, after_tick: &mut dyn FnMut(&WorldInstance)) {
        self.prefetch_tiles();
        for a in &mut self.agents {
            a.events = if a.disabled { Events::DISABLED } else { Events::NONE };
        }
        self.switch_physics();
        self.drive();
        let n = self.scenario.decimation;
        for k in 0..n {
            self.tick();
            if k + 1 == n {
                // Lane tracking once per policy step, before the last tick is reported.
                self.track_roads();
            }
            after_tick(self);
        }
        self.steps += 1;
        self.grid.build(&self.shapes);
    }

    /// Promote and demote the agents of `hybrid` groups (see
    /// [`HybridSpec`](crate::scenario::HybridSpec)), from the state at the start of the policy
    /// step. [`step`](Self::step) does this; when ticking by hand, call it before
    /// [`drive`](Self::drive).
    pub fn switch_physics(&mut self) {
        let sc = &*self.scenario;
        if !sc.groups.iter().any(|g| g.spec.physics == PhysicsMode::Hybrid) {
            return;
        }
        let world = &*self.map;
        let active = |i: usize| self.shapes[i].active && !self.agents[i].disabled;
        let learners: Vec<usize> =
            (0..self.agents.len()).filter(|&i| active(i) && sc.groups[self.agents[i].group].learning()).collect();
        let full: Vec<usize> =
            (0..self.agents.len()).filter(|&i| active(i) && !self.agents[i].is_kinematic()).collect();
        // Decide from the state at the step's start, then switch.
        let mut switch = Vec::new();
        for i in 0..self.agents.len() {
            let spec = &sc.groups[self.agents[i].group].spec;
            if spec.physics != PhysicsMode::Hybrid || !active(i) {
                continue;
            }
            let h = &spec.hybrid;
            let me = &self.shapes[i];
            let learner_distance =
                learners.iter().map(|&j| me.center.distance(self.shapes[j].center)).fold(f64::INFINITY, f64::min);
            let near = full.iter().any(|&j| j != i && surface_distance(me, &self.shapes[j], h.margin) < h.margin);
            let a = &self.agents[i];
            if a.is_kinematic() {
                if learner_distance <= h.promote || near {
                    switch.push(i);
                }
            } else if learner_distance > h.demote && !near && a.calm >= h.calm {
                switch.push(i);
            }
        }
        for &i in &switch {
            let a = &mut self.agents[i];
            if a.is_kinematic() {
                a.promote(world, self.env.config.gravity);
            } else {
                a.demote(world);
                a.update_shape(&mut self.shapes[i]);
            }
        }
        if !switch.is_empty() {
            self.grid.build(&self.shapes);
        }
    }

    /// Command the agents of scripted groups for the coming policy step. [`step`](Self::step)
    /// does this; when ticking by hand, call it before the first tick of every policy step.
    pub fn drive(&mut self) {
        if !self.agents.iter().any(|a| a.driver.is_some()) {
            return;
        }
        self.respawn_traffic();
        let dt = f64::from(self.scenario.decimation) * self.scenario.dt();
        let t = self.clock.time();
        let world = &*self.map;
        let traffic: Vec<(usize, Traffic)> = (self.shapes.iter().enumerate())
            .filter(|(_, s)| s.active)
            .map(|(i, s)| (i, Traffic { center: s.center, radius: s.radius, spheres: &s.spheres }))
            .collect();
        // Traffic drivers: their places, then the index of everyone on the lanes.
        let lanes = world.roads().lanes();
        let traffic_drivers = self.agents.iter().any(|a| matches!(a.driver, Some(Driver::Traffic(_))));
        if traffic_drivers {
            if self.lane_index.as_ref().is_none_or(|i| !i.fits(lanes)) {
                self.lane_index = Some(LaneIndex::new(lanes));
            }
            let index = self.lane_index.as_mut().expect("made");
            index.clear();
            for (i, a) in self.agents.iter_mut().enumerate() {
                if a.disabled || !self.shapes[i].active {
                    continue;
                }
                let v = &a.vehicle;
                if let Some(Driver::Traffic(d)) = &mut a.driver {
                    d.locate(world, &v.pose());
                    d.occupy(lanes, index, i as u32, &v.pose(), ground_speed(v));
                } else {
                    let s = &self.shapes[i];
                    let learner = a.driver.is_none();
                    index.insert_other(
                        world,
                        i as u32,
                        s.center,
                        yaw(v.pose().rot),
                        v.lin_vel_world(),
                        s.radius,
                        learner,
                    );
                }
            }
            index.sort();
            index.set_crossing_users(self.crowd.crossing_users());
        }
        let mut others = Vec::with_capacity(traffic.len());
        for (i, a) in self.agents.iter_mut().enumerate().filter(|(_, a)| !a.disabled) {
            let v = &a.vehicle;
            let command = match &mut a.driver {
                Some(Driver::Road(d)) => {
                    others.clear();
                    others.extend(traffic.iter().filter(|(j, _)| *j != i).map(|(_, t)| *t));
                    d.drive(world, &mut a.route, &v.pose(), v.lin_vel_body().x, dt, &others)
                }
                Some(Driver::Traffic(d)) => {
                    let index = self.lane_index.as_mut().expect("built above");
                    d.drive(world, &v.pose(), ground_speed(v), dt, i as u32, index, &self.signals, t)
                }
                Some(Driver::Parked) => GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 },
                None => continue,
            };
            a.set_command(command);
        }
    }

    /// Put back traffic drivers that want it (see [`traffic_driver`](crate::traffic_driver)):
    /// at rest at a lane point away from the learning agents, drawn from the episode's
    /// `respawn` stream by step and agent.
    fn respawn_traffic(&mut self) {
        let world = self.map.clone();
        let (lo, hi) = world.extent();
        let wanted: Vec<usize> = (0..self.agents.len())
            .filter(|&i| {
                let a = &self.agents[i];
                let p = a.vehicle.position();
                let off_map = p.x < lo.x || p.y < lo.y || p.x > hi.x || p.y > hi.y;
                !a.disabled && matches!(&a.driver, Some(Driver::Traffic(d)) if d.wants_respawn() || off_map)
            })
            .collect();
        if wanted.is_empty() {
            return;
        }
        let sc = self.scenario.clone();
        let seed = self.episode_seed.child("respawn").child_index(self.steps);
        for i in wanted {
            let active = |j: usize| !self.agents[j].disabled && self.shapes[j].active && j != i;
            let at = |j: usize| self.agents[j].vehicle.position().truncate();
            let learners: Vec<DVec2> = (0..self.agents.len())
                .filter(|&j| active(j) && sc.groups[self.agents[j].group].learning())
                .map(at)
                .collect();
            let others: Vec<DVec2> = (0..self.agents.len()).filter(|&j| active(j)).map(at).collect();
            let g = &sc.groups[self.agents[i].group];
            let Some(Driver::Traffic(d)) = &self.agents[i].driver else { continue };
            let (clear, back) = (d.spec().respawn_clear, -d.geometry().rear);
            let only = d.network().map(|n| n.lanes.as_slice());
            let s = seed.child_index(i as u64);
            let (Some(def), Some((xy, heading))) = (
                g.def.as_wheeled(),
                respawn_spot(&world, back, only, d.geometry().single_track, &learners, &others, clear, &mut s.rng()),
            ) else {
                continue;
            };
            let pose = ground_pose(&world, def, &g.rest, xy, heading);
            self.place_agent(i, pose, DVec3::ZERO, DVec3::ZERO);
            let a = &mut self.agents[i];
            if let Some(Driver::Traffic(d)) = &mut a.driver {
                let n = d.respawns;
                d.reset(s.child("driver"), &world, &a.vehicle.pose());
                d.respawns = n + 1;
            }
        }
    }

    /// Lane tracking and its events for every ground vehicle (see [`traffic`](crate::traffic)).
    /// [`step`](Self::step) does this after the last tick of every policy step.
    pub fn track_roads(&mut self) {
        let world = &*self.map;
        if !world.roads().has_sections() {
            return;
        }
        let t = self.clock.time();
        let cfg = &self.scenario.spec.events;
        let signals = &self.signals;
        let track = |a: &mut Agent| a.update_track(world, signals, t, cfg);
        if self.agents.len() >= PARALLEL_AGENTS {
            self.agents.par_iter_mut().for_each(track);
        } else {
            self.agents.iter_mut().for_each(track);
        }
    }

    /// On tiled maps: have the tiles ahead of each moving agent generated in the background.
    fn prefetch_tiles(&self) {
        let Some(tiles) = self.map.terrain().tiled() else { return };
        for a in self.agents.iter().filter(|a| !a.disabled) {
            let p = a.vehicle.position().truncate();
            let v = a.vehicle.lin_vel_world().truncate();
            if !v.is_finite() {
                continue;
            }
            for t in PREFETCH_AHEAD {
                let q = p + v * t;
                // Tiles within PREFETCH_RADIUS of the point (rays and turns reach them).
                let (lo, hi) = (q - PREFETCH_RADIUS, q + PREFETCH_RADIUS);
                let ((x0, y0), (x1, y1)) = (tiles.layout().tile_at(lo.x, lo.y), tiles.layout().tile_at(hi.x, hi.y));
                for ty in y0..=y1 {
                    for tx in x0..=x1 {
                        tiles.prefetch(tx, ty);
                    }
                }
            }
        }
    }

    /// One physics tick.
    pub fn tick(&mut self) {
        let Self {
            scenario,
            map,
            env,
            agents,
            shapes,
            contacts,
            clock,
            agent_contact_state,
            crowd,
            signals,
            learning,
            ..
        } = self;
        let sc = &**scenario;
        let world = &**map;
        let env = &*env;
        let time = clock.time();
        let env_step = (clock.is_due(sc.environment_divider))
            .then(|| if clock.tick == 0 { 0.0 } else { sc.dt() * f64::from(sc.environment_divider) });

        agent_contacts(shapes, sc.dt(), sc.spec.events.crash_speed, agent_contact_state, contacts);

        // Kinematic vehicles move at the end of every `kinematic_divider` ticks.
        let kinematic_step = (clock.tick + 1)
            .is_multiple_of(u64::from(sc.kinematic_divider))
            .then(|| sc.dt() * f64::from(sc.kinematic_divider));
        // Kinematic vehicles cost little between their steps and little more at them: spread
        // over threads are the agents simulated in full, and crowds at a kinematic step.
        let full = agents.iter().filter(|a| !a.is_kinematic()).count();
        let parallel = full >= PARALLEL_AGENTS || (kinematic_step.is_some() && agents.len() >= PARALLEL_KINEMATIC);
        let step = |((a, s), c): ((&mut Agent, &mut AgentShape), &AgentContacts)| {
            let g = &sc.groups[a.group];
            a.pre_step(world, env, time, env_step, kinematic_step, c);
            a.post_step(world, env, sc.dt(), &sc.spec.events, g.spec.disable_on_terminal, kinematic_step, c, s);
        };
        if parallel {
            agents.par_iter_mut().zip(shapes.par_iter_mut()).zip(contacts.par_iter()).for_each(step);
        } else {
            agents.iter_mut().zip(shapes.iter_mut()).zip(contacts.iter()).for_each(step);
        }

        clock.advance();
        // The crowd moves at the end of every `pedestrian_divider` ticks.
        if !crowd.peds.is_empty() && clock.tick.is_multiple_of(u64::from(sc.pedestrian_divider)) {
            let mut events: Vec<Events> = agents.iter().map(|a| a.events).collect();
            let dt = sc.dt() * f64::from(sc.pedestrian_divider);
            crowd.update(&sc.spec.pedestrians, world, signals, clock.time(), dt, shapes, learning, &mut events);
            for (a, e) in agents.iter_mut().zip(events) {
                a.events |= e;
            }
        }
        if !sc.has_sensors() {
            return;
        }
        let (tick, time) = (clock.tick, clock.time());
        let shapes = &*shapes;
        let peds = &crowd.peds[..];
        let sense = |(i, a): (usize, &mut Agent)| a.sense(tick, time, world, env, shapes, peds, i);
        if parallel {
            agents.par_iter_mut().enumerate().for_each(sense);
        } else {
            agents.iter_mut().enumerate().for_each(sense);
        }
    }

    // ------------------------------------------------------------------------ outputs

    /// Every active ground vehicle as the `traffic` term sees it (by agent).
    fn seen(&self) -> Vec<Option<Seen>> {
        (self.agents.iter().enumerate())
            .map(|(i, a)| {
                let geo = self.footprints[a.group].filter(|_| !a.disabled && self.shapes[i].active)?;
                let pose = a.vehicle.pose();
                let heading = yaw(pose.rot);
                let center = pose.pos.truncate() + 0.5 * (geo.front + geo.rear) * DVec2::from_angle(heading);
                Some(Seen {
                    center,
                    velocity: a.vehicle.lin_vel_world().truncate(),
                    heading,
                    length: geo.front - geo.rear,
                    width: 2.0 * geo.half_width,
                    lane: a.track.lane,
                })
            })
            .collect()
    }

    /// Observations of `group` (`count × obs_dim` values).
    pub fn observe(&self, group: usize, out: &mut [f32]) {
        let g = &self.scenario.groups[group];
        let dim = g.obs_dim();
        assert_eq!(out.len(), g.spec.count * dim, "observation array of group {:?}", g.spec.name);
        if dim == 0 {
            // Camera terms only.
            return;
        }
        let vehicles = if g.obs.needs_traffic() { self.seen() } else { Vec::new() };
        let write = |(k, o): (usize, &mut [f32])| {
            let me = g.first_agent + k;
            let a = &self.agents[me];
            a.observe(g, &self.map, &self.shapes, &self.grid, me, &vehicles, &self.signals, self.clock.time(), o);
        };
        if g.spec.count >= PARALLEL_AGENTS {
            out.par_chunks_exact_mut(dim).enumerate().for_each(write);
        } else {
            out.chunks_exact_mut(dim).enumerate().for_each(write);
        }
    }

    /// Camera images of `group` (`count × image_len` bytes; see [`obs`](crate::obs)).
    pub fn observe_images(&self, group: usize, out: &mut [u8]) {
        let g = &self.scenario.groups[group];
        let len = g.obs.image_len();
        assert_eq!(out.len(), g.spec.count * len, "image array of group {:?}", g.spec.name);
        if len == 0 {
            return;
        }
        for (k, o) in out.chunks_exact_mut(len).enumerate() {
            g.obs.write_image(&self.agents[g.first_agent + k].sensors, o);
        }
    }

    /// Hand rendered frames to their cameras (see [`camera`](crate::camera)); they are stamped
    /// with the current tick.
    pub fn deliver(&mut self, captures: impl IntoIterator<Item = Capture>) {
        let (tick, time) = (self.clock.tick, self.clock.time());
        for c in captures {
            match &mut self.agents[c.agent].sensors[c.sensor] {
                Sensor::Camera(cam) => cam.capture(tick, time, c.image),
                _ => panic!("sensor {} of agent {} is not a camera", c.sensor, c.agent),
            }
        }
    }

    /// State rows ([`STATE_FIELDS`]) of `group` (`count × STATE_DIM` values).
    pub fn write_state(&self, group: usize, out: &mut [f64]) {
        let g = &self.scenario.groups[group];
        assert_eq!(out.len(), g.spec.count * STATE_DIM, "state array of group {:?}", g.spec.name);
        let write = |(k, row): (usize, &mut [f64; STATE_DIM])| {
            let a = &self.agents[g.first_agent + k];
            let v = &a.vehicle;
            let q = v.orientation();
            let goal = a.goal();
            row[0..3].copy_from_slice(&v.position().to_array());
            row[3..7].copy_from_slice(&q.to_array());
            row[7..10].copy_from_slice(&(q * v.lin_vel_body()).to_array());
            row[10..13].copy_from_slice(&v.ang_vel_body().to_array());
            row[13..16].copy_from_slice(&goal.position.to_array());
            row[16] = goal.yaw;
            row[17] = a.agl_now(&self.map);
            row[18] = a.goal_index as f64;
            row[19] = match v {
                Vehicle::Wheeled(_) => self.map.obstacle_clearance(v.position(), CLEARANCE_RANGE),
                _ => self.map.clearance(v.position(), CLEARANCE_RANGE),
            };
            row[20] = self.grid.clearance(&self.shapes, g.first_agent + k, CLEARANCE_RANGE);
            let road = lane::road_state(a.route.as_deref(), &self.map, v.position().truncate(), yaw(q));
            row[21..24].copy_from_slice(&road);
            let (art, tail) = match v.as_wheeled() {
                Some(w) => {
                    let mut art = [0.0; 2];
                    for (d, (a, _)) in art.iter_mut().zip(w.articulations()) {
                        *d = a;
                    }
                    (art, w.tail_pose())
                }
                None => ([0.0; 2], v.pose()),
            };
            row[24..26].copy_from_slice(&art);
            row[26..29].copy_from_slice(&[tail.pos.x, tail.pos.y, yaw(tail.rot)]);
            row[29] = v.as_wheeled().map_or(0.0, |w| w.sinkage());
            let flow = a.air().flow(q, q * v.lin_vel_body(), v.ang_vel_body());
            row[30..33].copy_from_slice(&[flow.airspeed, flow.alpha, flow.beta]);
            row[33] = a.support.map_or(-1.0, f64::from);
        };
        let rows = out.as_chunks_mut::<STATE_DIM>().0;
        if g.spec.count >= PARALLEL_AGENTS {
            rows.par_iter_mut().enumerate().for_each(write);
        } else {
            rows.iter_mut().enumerate().for_each(write);
        }
    }

    /// Events of `group` since the start of the last policy step.
    pub fn write_events(&self, group: usize, out: &mut [u32]) {
        let g = &self.scenario.groups[group];
        for (o, a) in out.iter_mut().zip(&self.agents[g.first_agent..g.first_agent + g.spec.count]) {
            *o = a.events.0;
        }
    }

    // ------------------------------------------------------------------------ access

    pub fn scenario(&self) -> &Arc<CompiledScenario> {
        &self.scenario
    }

    pub fn map(&self) -> &Arc<StaticWorld> {
        &self.map
    }

    /// Index of the current map in the scenario's map pool.
    pub fn map_index(&self) -> usize {
        self.map_index
    }

    pub fn env(&self) -> &EnvState {
        &self.env
    }

    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }

    pub fn agent(&self, i: usize) -> &Agent {
        &self.agents[i]
    }

    pub fn agent_mut(&mut self, i: usize) -> &mut Agent {
        &mut self.agents[i]
    }

    /// Agent shapes as of the last tick (for rendering and debugging).
    pub fn shapes(&self) -> &[AgentShape] {
        &self.shapes
    }

    /// Contact forces between agents in the last tick, per agent.
    pub fn agent_contacts(&self) -> &[AgentContacts] {
        &self.contacts
    }

    pub fn clock(&self) -> &Clock {
        &self.clock
    }

    /// Simulated time since the reset (s).
    pub fn time(&self) -> f64 {
        self.clock.time()
    }

    /// The pedestrians.
    pub fn crowd(&self) -> &Crowd {
        &self.crowd
    }

    /// The pedestrians, to move by hand (tests, tools).
    pub fn crowd_mut(&mut self) -> &mut Crowd {
        &mut self.crowd
    }

    /// The traffic signals of the current map and episode.
    pub fn signals(&self) -> &Signals {
        &self.signals
    }

    /// Replace the signal offsets (replays set the recorded ones).
    pub fn set_signals(&mut self, signals: Signals) {
        self.signals = signals;
    }

    /// Policy steps since the reset.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Seed of the current episode.
    pub fn episode_seed(&self) -> Seed {
        self.episode_seed
    }

    /// Use map `index` of the pool until the next reset draws one (replays show the recorded
    /// map this way). Its signals run without offsets until they are set ([`set_signals`](Self::set_signals)).
    pub fn set_map(&mut self, index: usize) {
        self.map = self.scenario.maps[index].clone();
        self.map_index = index;
        self.env = EnvState::new(self.env.config.clone(), &self.map);
        self.signals = Signals::default();
    }

    /// Stop agent `i` for the rest of the episode, as a terminal event would: it freezes and
    /// drops out of contacts and sensors at once.
    pub fn disable_agent(&mut self, i: usize) {
        self.agents[i].disabled = true;
        self.shapes[i].active = false;
        self.grid.build(&self.shapes);
    }

    /// Put agent `i` at `pose` with the given velocities (world linear, body angular) and clear
    /// its transient vehicle state; its shape follows at once.
    pub fn place_agent(&mut self, i: usize, pose: Pose, lin_vel_world: DVec3, ang_vel_body: DVec3) {
        self.agents[i].vehicle.place(pose, lin_vel_world, ang_vel_body);
        self.agents[i].replace_kinematic(&self.map);
        self.agents[i].reset_track(&self.map);
        self.agents[i].update_shape(&mut self.shapes[i]);
        self.grid.build(&self.shapes);
    }

    /// Replace the goals of an agent (at least one) and restart at the first.
    pub fn set_goals(&mut self, agent: usize, goals: Vec<Goal>) {
        assert!(!goals.is_empty(), "an agent needs at least one goal");
        let a = &mut self.agents[agent];
        a.goals = goals;
        a.goal_index = 0;
        a.route = None;
        a.legs.clear();
    }

    /// Advance the goal of `agent`; false if it was at its last goal.
    pub fn advance_goal(&mut self, agent: usize) -> bool {
        self.agents[agent].advance_goal()
    }

    /// Everything that evolves, for later [`restore`](Self::restore). The map and scenario are
    /// shared, so this is cheap.
    pub fn snapshot(&self) -> WorldInstance {
        self.clone()
    }

    pub fn restore(&mut self, snapshot: &WorldInstance) {
        self.clone_from(snapshot);
    }

    /// Hash of the dynamic state: time, and per agent the multibody state, rotor speeds (or a
    /// ground vehicle's steering, tyre and powertrain state), events, goal index and latest
    /// sensor readings. Equal hashes mean bit-identical states.
    pub fn state_hash(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(&self.clock.tick.to_le_bytes());
        let mut f = |x: f64| {
            h.update(&x.to_bits().to_le_bytes());
        };
        for a in &self.agents {
            let st = a.vehicle.state();
            st.q.iter().chain(st.v.iter()).for_each(|x| f(*x));
            match &a.vehicle {
                Vehicle::Multirotor(v) => v.motor_speeds().iter().for_each(|x| f(*x)),
                Vehicle::Wheeled(v) => {
                    f(v.steering_angle());
                    for w in v.wheels() {
                        let t = &w.tire;
                        for x in [w.steer, w.drive_torque, w.brake_torque, t.force.x, t.force.y, t.force.z] {
                            f(x);
                        }
                    }
                    let p = v.powertrain();
                    for x in [f64::from(p.gear), p.engine_speed, p.engine_torque] {
                        f(x);
                    }
                }
                Vehicle::FixedWing(v) => {
                    f(v.rotor_speed());
                    v.surfaces().iter().for_each(|x| f(*x));
                }
                Vehicle::Helicopter(v) => {
                    for x in [v.rotor_speed(), v.engine_torque()] {
                        f(x);
                    }
                    v.pitches().iter().for_each(|x| f(*x));
                    for r in [v.main_rotor_state(), v.tail_rotor_state()] {
                        r.flap.iter().chain([&r.inflow]).for_each(|x| f(*x));
                    }
                }
                Vehicle::Tiltrotor(v) => {
                    v.rotor_speeds().iter().chain(v.tilts()).chain(&v.channels()).for_each(|x| f(*x));
                }
            }
            if let Some(k) = &a.kinematic {
                for x in
                    [k.xy.x, k.xy.y, k.yaw, k.speed, k.lateral, k.yaw_rate, k.steer, k.trim].iter().chain(&k.unit_yaw)
                {
                    f(*x);
                }
            }
            f(f64::from(a.events.0));
            f(a.goal_index as f64);
            match &a.driver {
                Some(Driver::Road(d)) => {
                    for x in [d.cruise, d.stop_left, d.station, d.turn.map_or(-1.0, |t| t.target)] {
                        f(x);
                    }
                }
                Some(Driver::Traffic(d)) => {
                    let (elem, station) = d.place.map_or((-1.0, 0.0), |p| match p.elem {
                        Elem::Lane(l) => (f64::from(l), p.station),
                        Elem::Connector(c) => (-2.0 - f64::from(c), p.station),
                    });
                    let change = d.change.map_or([-1.0; 3], |c| [f64::from(c.from), c.start, c.length]);
                    for x in [elem, station, d.since_change, d.stuck, f64::from(d.changes)].iter().chain(&change) {
                        f(*x);
                    }
                    let id = |x: Option<u32>| x.map_or(-1.0, f64::from);
                    for x in [id(d.granted), id(d.stopped_at), d.waiting, d.standing, d.lost, f64::from(d.respawns)] {
                        f(x);
                    }
                    for c in &d.plan {
                        f(f64::from(*c));
                    }
                }
                Some(Driver::Parked) | None => {}
            }
            for s in &a.sensors {
                hash_sensor(s, &mut f);
            }
        }
        self.crowd.hash_into(&mut f);
        *h.finalize().as_bytes()
    }
}

/// Horizontal speed of `v` along its heading (m/s; negative backwards): the speed kinematic
/// vehicles hold.
fn ground_speed(v: &Vehicle) -> f64 {
    v.lin_vel_world().truncate().dot(DVec2::from_angle(yaw(v.pose().rot)))
}

/// Attitude without heading, body-frame air velocity, controls and rotor speed of `aircraft`
/// trimmed for straight and level flight at `airspeed`; where it cannot be trimmed, level with
/// the surfaces centred at full throttle.
fn fixed_wing_start(aircraft: &FixedWing, airspeed: f64, density: f64, gravity: f64) -> (DQuat, DVec3, FixedWingStart) {
    match aircraft.trim(airspeed, density, 0.0, 0.0, gravity) {
        Ok(t) => {
            (t.attitude(0.0), t.velocity_body(), FixedWingStart { controls: t.controls, rotor_speed: t.rotor_speed })
        }
        Err(_) => {
            let supply = aircraft.def().battery.as_ref().map_or(0.0, |b| b.full_voltage());
            let rotor_speed = aircraft.propulsion().steady_omega(1.0, airspeed, density, supply);
            let controls = FixedWingInput { throttle: 1.0, ..FixedWingInput::default() };
            (DQuat::IDENTITY, DVec3::X * airspeed, FixedWingStart { controls, rotor_speed })
        }
    }
}

/// Attitude without heading, body-frame air velocity, controls and rotor speed of `heli`
/// trimmed for straight and level flight at `speed`; where it cannot be trimmed, level at rest
/// with the collective at mid travel.
fn helicopter_start(heli: &Helicopter, speed: f64, density: f64, gravity: f64) -> (DQuat, DVec3, HelicopterStart) {
    let rated = heli.def().engine.rated_speed;
    match heli.trim(speed, density, gravity) {
        Ok(t) => {
            let start = HelicopterStart { controls: t.controls, rotor_speed: rated, density };
            (t.attitude(0.0), t.velocity_body, start)
        }
        Err(_) => {
            let controls = HelicopterInput::default();
            (DQuat::IDENTITY, DVec3::X * speed, HelicopterStart { controls, rotor_speed: rated, density })
        }
    }
}

/// Attitude without heading, body-frame air velocity, controls and rotor speeds of `aircraft`
/// trimmed for straight and level flight at `speed`: in aeroplane mode (rotors forward) from
/// 1.2 times the stall speed, else with the rotors up, else at the first feasible tilt from
/// forward to up; where none is feasible, level with the rotors up at the hover rotor speed.
fn tiltrotor_start(aircraft: &Tiltrotor, speed: f64, density: f64, gravity: f64) -> (DQuat, DVec3, TiltrotorStart) {
    let def = aircraft.def();
    let range = &def.controls.tilt;
    let fast = speed >= 1.2 * def.stall_speed(density, gravity);
    let first = if fast { range.max } else { 0.0f64.clamp(range.min, range.max) };
    let sweep = (0..=18).map(|i| range.max - (range.max - range.min) * f64::from(i) / 18.0);
    let limits = TrimLimits::default();
    let trim = std::iter::once(first)
        .chain(sweep)
        .filter_map(|tilt| aircraft.trim(speed, tilt, density, gravity).ok())
        .find(|t| t.feasible(def, &limits));
    match trim {
        Some(t) => {
            let start = TiltrotorStart { controls: t.controls, rotor_speed: t.rotor_speed, density };
            (t.attitude(0.0), t.velocity_body, start)
        }
        None => {
            let omega = aircraft.hover_rotor_speed(density, gravity);
            let tilt = 0.0f64.clamp(range.min, range.max);
            let controls = TiltrotorInput { throttle: [0.5; 4], tilt: [tilt; 4], ..TiltrotorInput::default() };
            (DQuat::IDENTITY, DVec3::X * speed, TiltrotorStart { controls, rotor_speed: [omega; 4], density })
        }
    }
}

fn hash_sensor(s: &Sensor, f: &mut impl FnMut(f64)) {
    let mut v3 = |v: glam::DVec3| v.to_array().into_iter().for_each(&mut *f);
    match s {
        Sensor::Imu(s) => {
            if let Some(r) = s.latest() {
                v3(r.value.accel);
                v3(r.value.gyro);
            }
        }
        Sensor::Gps(s) => {
            if let Some(r) = s.latest() {
                v3(r.value.position);
                v3(r.value.velocity);
            }
        }
        Sensor::Baro(s) => {
            if let Some(r) = s.latest() {
                f(r.value.pressure)
            }
        }
        Sensor::Mag(s) => {
            if let Some(r) = s.latest() {
                v3(r.value.field)
            }
        }
        Sensor::Pitot(s) => {
            if let Some(r) = s.latest() {
                f(r.value.differential_pressure)
            }
        }
        Sensor::Rangefinder(s) => {
            if let Some(r) = s.latest() {
                f(r.value.range.unwrap_or(-1.0))
            }
        }
        Sensor::Lidar(s) => {
            if let Some(scan) = s.latest() {
                scan.ranges.iter().for_each(|r| f(f64::from(*r)))
            }
        }
        Sensor::Camera(s) => {
            if let Some(r) = s.latest() {
                f(r.tick as f64);
                for b in r.value.digest().as_chunks::<4>().0 {
                    f(f64::from(u32::from_le_bytes(*b)));
                }
            }
        }
        Sensor::GroundTruth(s) => {
            if let Some(r) = s.latest() {
                v3(r.value.position)
            }
        }
    }
}

/// Whether two rectangles (centre, heading, half extents) overlap (separating axes).
fn rects_overlap(a: &(DVec2, f64, DVec2), b: &(DVec2, f64, DVec2)) -> bool {
    let axes = |r: &(DVec2, f64, DVec2)| {
        let x = DVec2::from_angle(r.1);
        [x, x.perp()]
    };
    let extent = |r: &(DVec2, f64, DVec2), n: DVec2| {
        let [x, y] = axes(r);
        r.2.x * x.dot(n).abs() + r.2.y * y.dot(n).abs()
    };
    let d = b.0 - a.0;
    axes(a).into_iter().chain(axes(b)).all(|n| d.dot(n).abs() <= extent(a, n) + extent(b, n))
}
