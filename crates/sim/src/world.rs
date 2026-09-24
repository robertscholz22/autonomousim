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
//! Steps 2 and 3 run in parallel over agents when there are at least [`PARALLEL_AGENTS`]
//! (two fork–joins per tick); each agent only writes its own state, so results do not depend
//! on the thread count. Parallel phases are cheapest when the world is stepped on a rayon
//! worker thread, as [`BatchSim`](crate::BatchSim) does.
//!
//! **Seeding**: a world has a base seed; episode `k` uses `base/episode/k` with the streams
//! `map`, `environment`, `spawn`, `goals` and `agent/<id>` (vehicle parameters, sensors
//! `sensor/<name>`, turbulence). Resetting with an explicit seed replaces the base and restarts
//! the count, so `reset(Some(s))` always reproduces the same episode.

use crate::agent::{Agent, EnvState};
use crate::drive::ground_pose;
use crate::events::Events;
use crate::interaction::{AgentContacts, AgentShape, agent_contacts};
use crate::obs::CLEARANCE_RANGE;
use crate::scenario::{CompiledScenario, Goal};
use autonomousim_control::Command;
use autonomousim_core::math::quat::yaw;
use autonomousim_core::rng::Seed;
use autonomousim_core::time::Clock;
use autonomousim_sensors::Sensor;
use autonomousim_vehicles::Vehicle;
use autonomousim_world::StaticWorld;
use rayon::prelude::*;
use std::sync::Arc;

/// Agent count from which the per-agent phases run in parallel.
pub const PARALLEL_AGENTS: usize = 32;

/// Per-agent state row written by [`WorldInstance::write_state`]: `(name, length)` in order.
/// `goal_index` equals the number of goals once the last one has been reached; `clearance` is
/// the distance to the nearest terrain or solid obstacle surface, up to 20 m.
pub const STATE_FIELDS: [(&str, usize); 9] = [
    ("position", 3),
    ("orientation", 4),
    ("velocity", 3),
    ("rates", 3),
    ("goal", 3),
    ("goal_yaw", 1),
    ("agl", 1),
    ("goal_index", 1),
    ("clearance", 1),
];

/// Length of a state row.
pub const STATE_DIM: usize = 20;

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
    order: Vec<(f64, u32)>,
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
            order: Vec::new(),
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

        let pick = es.child("map").rng().below(sc.maps.len() as u64) as usize;
        self.map = sc.maps[pick].clone();
        self.map_index = pick;
        let world = &*self.map;
        let mut env = sc.spec.environment.clone();
        if let Some(r) = &sc.spec.randomize_environment {
            r.apply(&mut env, &mut es.child("environment").rng());
        }
        self.env = EnvState::new(env, world);
        self.clock = sc.clock;
        self.steps = 0;

        let mut spawn_rng = es.child("spawn").rng();
        let mut goal_rng = es.child("goals").rng();
        let agent_seed = es.child("agent");
        let mut placed = Vec::with_capacity(self.agents.len());
        for g in &sc.groups {
            let spawn = &g.spec.spawn;
            let ground = g.ground(pick);
            let positions = spawn.sample_positions(world, g.spec.count, g.bottom, ground, &mut placed, &mut spawn_rng);
            for (k, p) in positions.into_iter().enumerate() {
                let id = g.first_agent + k;
                let density = self.env.config.atmosphere.density(self.env.origin_altitude + p.z);
                let hover = g.def.as_multirotor().map_or(0.0, |d| d.hover_omega(self.env.config.gravity, density));
                let mut placement = spawn.sample_state(p, hover, &mut spawn_rng);
                if let Some(d) = g.def.as_wheeled() {
                    placement.pose = ground_pose(world, d, &g.rest, p.truncate(), yaw(placement.pose.rot));
                }
                let goals = g.spec.goals.sample(world, &placement.pose, ground, &mut goal_rng);
                let seed = agent_seed.child_index(id as u64);
                let scales = g
                    .def
                    .as_multirotor()
                    .map(|d| g.spec.randomize.sample(d.rotors.len(), &mut seed.child("vehicle").rng()));
                let agent = &mut self.agents[id];
                agent.reset(g, &placement, scales.as_ref(), goals, seed, &self.env, world);
                let contact = g.def.contact_model(agent.vehicle.mass(), sc.dt());
                self.shapes[id].set_contact(&contact.solid);
                agent.update_shape(&mut self.shapes[id]);
            }
        }
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
        for a in &mut self.agents {
            a.events = if a.disabled { Events::DISABLED } else { Events::NONE };
        }
        for _ in 0..self.scenario.decimation {
            self.tick();
            after_tick(self);
        }
        self.steps += 1;
    }

    /// One physics tick.
    pub fn tick(&mut self) {
        let Self { scenario, map, env, agents, shapes, contacts, clock, order, .. } = self;
        let sc = &**scenario;
        let world = &**map;
        let env = &*env;
        let parallel = agents.len() >= PARALLEL_AGENTS;
        let time = clock.time();
        let env_step = (clock.is_due(sc.environment_divider))
            .then(|| if clock.tick == 0 { 0.0 } else { sc.dt() * f64::from(sc.environment_divider) });

        agent_contacts(shapes, order, contacts);

        let step = |((a, s), c): ((&mut Agent, &mut AgentShape), &AgentContacts)| {
            let g = &sc.groups[a.group];
            a.pre_step(world, env, time, env_step, c);
            a.post_step(world, env, sc.dt(), &sc.spec.events, g.spec.disable_on_terminal, s);
        };
        if parallel {
            agents.par_iter_mut().zip(shapes.par_iter_mut()).zip(contacts.par_iter()).for_each(step);
        } else {
            agents.iter_mut().zip(shapes.iter_mut()).zip(contacts.iter()).for_each(step);
        }

        clock.advance();
        if !sc.has_sensors() {
            return;
        }
        let (tick, time) = (clock.tick, clock.time());
        let shapes = &*shapes;
        let sense = |(i, a): (usize, &mut Agent)| a.sense(tick, time, world, env, shapes, i);
        if parallel {
            agents.par_iter_mut().enumerate().for_each(sense);
        } else {
            agents.iter_mut().enumerate().for_each(sense);
        }
    }

    // ------------------------------------------------------------------------ outputs

    /// Observations of `group` (`count × obs_dim` values).
    pub fn observe(&self, group: usize, out: &mut [f32]) {
        let g = &self.scenario.groups[group];
        let dim = g.obs_dim();
        assert_eq!(out.len(), g.spec.count * dim, "observation array of group {:?}", g.spec.name);
        for (k, o) in out.chunks_exact_mut(dim).enumerate() {
            self.agents[g.first_agent + k].observe(g, &self.map, o);
        }
    }

    /// State rows ([`STATE_FIELDS`]) of `group` (`count × STATE_DIM` values).
    pub fn write_state(&self, group: usize, out: &mut [f64]) {
        let g = &self.scenario.groups[group];
        assert_eq!(out.len(), g.spec.count * STATE_DIM, "state array of group {:?}", g.spec.name);
        for (k, row) in out.as_chunks_mut::<STATE_DIM>().0.iter_mut().enumerate() {
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
            row[19] = self.map.clearance(v.position(), CLEARANCE_RANGE);
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

    pub fn clock(&self) -> &Clock {
        &self.clock
    }

    /// Simulated time since the reset (s).
    pub fn time(&self) -> f64 {
        self.clock.time()
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
    /// map this way).
    pub fn set_map(&mut self, index: usize) {
        self.map = self.scenario.maps[index].clone();
        self.map_index = index;
        self.env = EnvState::new(self.env.config.clone(), &self.map);
    }

    /// Replace the goals of an agent (at least one) and restart at the first.
    pub fn set_goals(&mut self, agent: usize, goals: Vec<Goal>) {
        assert!(!goals.is_empty(), "an agent needs at least one goal");
        let a = &mut self.agents[agent];
        a.goals = goals;
        a.goal_index = 0;
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
            }
            f(f64::from(a.events.0));
            f(a.goal_index as f64);
            for s in &a.sensors {
                hash_sensor(s, &mut f);
            }
        }
        *h.finalize().as_bytes()
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
        Sensor::GroundTruth(s) => {
            if let Some(r) = s.latest() {
                v3(r.value.position)
            }
        }
    }
}
