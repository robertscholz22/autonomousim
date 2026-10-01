//! Pedestrians (M8c step 2): a crowd walking the map's pedestrian network
//! ([`Walkways`]) with the social force model.
//!
//! - **Model**: social force (Helbing & Molnár 1995) with the interaction of Moussaïd et al.
//!   (2009): a driving term `(v₀·e₀ − v)/τ` towards a point [`LOOK_AHEAD`] m on along the route
//!   (at the pedestrian's own offset from the walkway's line), and from every other
//!   pedestrian within `range` a repulsion `−A·exp(−d/B)·[exp(−(n′Bθ)²)·t + K·exp(−(nBθ)²)·n]`
//!   along and across the interaction direction `t` (the relative velocity weighted by `λ`
//!   plus the direction to the other), `B = γ·|D|`, `θ` the angle between `t` and the
//!   direction to the other, `K` its sign (head-on: to the right). Overlapping bodies are
//!   pushed apart, and pedestrians keep within their walkway's width. Speeds are capped at
//!   1.3 times the desired speed, which is drawn per pedestrian from `speed` (mean, standard
//!   deviation; clamped to 0.5–2.5 m/s).
//! - **Trips**: from a place ([`Place`]: building entrances, bus stops, parks) or a random
//!   point of the network to a random place by the shortest route (A*). At the destination a
//!   pedestrian dwells (`dwell`, drawn), then enters the building (and is respawned) or walks
//!   on to another place.
//! - **Crossings**: approaching a crossing a pedestrian decides at the curb, and waits there
//!   until it may go. At a signalized one it goes on the walk light when it can cross before
//!   the clearance time runs out (`length / speed ≤ left + length / 1.2 m/s`); with
//!   probability `jaywalk` (drawn per crossing) it ignores the light. An attentive pedestrian
//!   (probability `attention`, drawn per crossing) also needs a gap: every vehicle heading for
//!   the crossing must be able to stop before it at [`YIELD_DECEL`] (pedestrians have
//!   priority) or arrive only after the pedestrian is across; a vehicle standing on the band
//!   blocks it. On the carriageway an attentive pedestrian avoids vehicles by time to
//!   collision: it walks on, hurries or stops, whichever keeps clear of every vehicle over
//!   [`HORIZON`] s (or keeps clear longest, or passes farthest).
//! - **Drivers** see the pedestrians on each crossing and those committed to it
//!   ([`Crowd::crossing_users`]): a `traffic` driver stops before a crossing until every
//!   pedestrian on it has passed its lane, and does not drive into one in its path.
//! - **Hits**: a vehicle (an active agent near the ground) moving faster than [`HIT_SPEED`]
//!   whose colliders touch a pedestrian's cylinder raises [`Events::PEDESTRIAN_HIT`]; the
//!   pedestrian stops and is respawned after [`HIT_HOLD`] s.
//! - **Respawn**: pedestrians entering a building, hit, or waiting longer than [`STUCK`] s
//!   are put back at a random point of the network at least `respawn_clear` m from every
//!   learning agent (out of its sensors' range), so that the count holds.
//! - **Sensing**: pedestrians are vertical cylinders (`radius`, `height`) to LiDAR and
//!   rangefinders ([`HitKind::Pedestrian`](autonomousim_core::geometry::HitKind)) and are
//!   drawn for cameras with the `Pedestrian` class.
//!
//! The crowd steps every [`CompiledScenario::pedestrian_divider`](crate::CompiledScenario)
//! physics ticks (about 25 Hz, [`PEDESTRIAN_PERIOD`]), in index order from a snapshot of the
//! positions, so the result does not depend on threads. Its draws come from the episode's
//! `pedestrians` stream (one generator per pedestrian, and `respawn/<update>/<index>`).

use crate::events::Events;
use crate::interaction::AgentShape;
use crate::traffic::Signals;
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::terrain::Terrain;
use autonomousim_world::lanes::CROSSWALK;
use autonomousim_world::signals::CLEARANCE_SPEED;
use autonomousim_world::{PlaceKind, StaticWorld, WalkKind, Walkways};
use glam::DVec2;
use serde::{Deserialize, Serialize};

/// Longest step of the crowd (s): it moves at about 25 Hz whatever the physics rate.
pub const PEDESTRIAN_PERIOD: f64 = 0.04;

/// Distance (m) ahead along the route that pedestrians head for.
pub const LOOK_AHEAD: f64 = 1.5;
/// Within this distance (m) of a crossing's start a pedestrian decides whether to go.
pub const CURB: f64 = 2.0;
/// Deceleration (m/s²) at which pedestrians expect vehicles to stop for them.
pub const YIELD_DECEL: f64 = 3.0;
/// Time horizon (s) of the avoidance of vehicles on the carriageway.
pub const HORIZON: f64 = 3.0;
/// Clearance (m) kept from vehicles.
pub const VEHICLE_CLEARANCE: f64 = 0.3;
/// Vehicles slower than this (m/s) do not hit pedestrians.
pub const HIT_SPEED: f64 = 0.5;
/// A hit pedestrian stays this long (s) before it is respawned.
pub const HIT_HOLD: f64 = 10.0;
/// Waiting longer than this (s), a pedestrian is respawned.
pub const STUCK: f64 = 300.0;
/// Vehicles farther than this (m) are not considered at crossings.
const VEHICLE_RANGE: f64 = 80.0;
/// Height (m) above the ground up to which agents count as vehicles on the ground.
const GROUND_HEIGHT: f64 = 3.0;
/// Hurrying pedestrians walk this much faster.
const HURRY: f64 = 1.6;
/// Head-on bias of the interaction's side (rad): a little to the right.
const SIDE_BIAS: f64 = 0.005;
/// Draws tried for a respawn spot.
const SPOT_TRIES: usize = 32;
/// Speeds are capped at this factor of the desired speed.
const SPEED_CAP: f64 = 1.3;
/// Within this distance (m) of the end of its walkway a pedestrian goes on to the next.
const NODE_REACH: f64 = 1.0;
/// Within this distance (m) of a walkway's ends pedestrians may leave its width.
const END_SLACK: f64 = 1.0;
/// Within this distance (m) of its route's end a pedestrian has arrived.
const ARRIVE: f64 = 0.5;

/// Settings of the crowd ([`Scenario::pedestrians`](crate::Scenario)); none by default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PedestrianSpec {
    /// Number of pedestrians (on maps with a pedestrian network).
    pub count: usize,
    /// Desired walking speed: mean and standard deviation (m/s).
    pub speed: [f64; 2],
    /// Body radius (m).
    pub radius: f64,
    /// Height range (m), drawn uniformly.
    pub height: [f64; 2],
    /// Time spent at a destination (s), drawn uniformly.
    pub dwell: [f64; 2],
    /// Probability (0–1), drawn per crossing, of crossing against the light.
    pub jaywalk: f64,
    /// Probability (0–1), drawn per crossing, of minding the traffic there.
    pub attention: f64,
    /// Respawns at least this far (m) from every learning agent.
    pub respawn_clear: f64,
    pub social: SocialForce,
}

impl Default for PedestrianSpec {
    fn default() -> Self {
        Self {
            count: 0,
            speed: [1.34, 0.26],
            radius: 0.25,
            height: [1.55, 1.95],
            dwell: [2.0, 30.0],
            jaywalk: 0.0,
            attention: 1.0,
            respawn_clear: 60.0,
            social: SocialForce::default(),
        }
    }
}

/// Parameters of the social force (Moussaïd et al. 2009).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SocialForce {
    /// Relaxation time `τ` of the driving term (s).
    pub relaxation: f64,
    /// Interaction strength `A` (m/s²).
    pub strength: f64,
    /// `γ`, `n`, `n′` and `λ` of the interaction.
    pub gamma: f64,
    pub n: f64,
    pub n_prime: f64,
    pub lambda: f64,
    /// Other pedestrians farther than this (m) are ignored.
    pub range: f64,
}

impl Default for SocialForce {
    fn default() -> Self {
        Self { relaxation: 0.54, strength: 4.5, gamma: 0.35, n: 2.0, n_prime: 3.0, lambda: 2.0, range: 5.0 }
    }
}

impl SocialForce {
    /// Acceleration (m/s²) of a pedestrian at `p` moving at `v` from another at `q` moving at
    /// `w`.
    pub fn interaction(&self, p: DVec2, v: DVec2, q: DVec2, w: DVec2) -> DVec2 {
        let r = q - p;
        let d = r.length();
        if d < 1e-9 || d > self.range {
            return DVec2::ZERO;
        }
        let e = r / d;
        let big_d = self.lambda * (v - w) + e;
        let len = big_d.length();
        if len < 1e-9 {
            return DVec2::ZERO;
        }
        let t = big_d / len;
        let n = t.perp();
        let theta = t.perp_dot(e).atan2(t.dot(e));
        let b = self.gamma * len;
        let k = if theta + SIDE_BIAS >= 0.0 { 1.0 } else { -1.0 };
        let base = -self.strength * (-d / b).exp();
        base * ((-(self.n_prime * b * theta).powi(2)).exp() * t + k * (-(self.n * b * theta).powi(2)).exp() * n)
    }

    /// Acceleration (m/s²) towards the desired velocity.
    pub fn driving(&self, v: DVec2, desired: DVec2) -> DVec2 {
        (desired - v) / self.relaxation
    }
}

/// What a pedestrian is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PedState {
    Walking,
    /// At a curb, waiting to cross.
    Waiting,
    /// Committed to a crossing (from the curb) or on it.
    Crossing,
    /// At its destination.
    Dwelling,
    /// Hit by a vehicle: standing until respawned.
    Hit,
}

/// One pedestrian.
#[derive(Clone, Debug)]
pub struct Pedestrian {
    /// Position on the ground (m) and the ground's height there.
    pub pos: DVec2,
    pub z: f64,
    pub vel: DVec2,
    /// Facing (rad): the direction of motion, kept while standing.
    pub heading: f64,
    /// Desired speed (m/s).
    pub speed: f64,
    pub radius: f64,
    pub height: f64,
    pub state: PedState,
    /// The route: walkway edges and whether each is walked from its `a` to its `b`.
    legs: Vec<(u32, bool)>,
    leg: usize,
    /// The place the route leads to.
    dest: Option<u32>,
    /// Time left dwelling or hit, or spent waiting (s).
    timer: f64,
    /// Drawn at the curb for the next crossing: minding the traffic, ignoring the light.
    attentive: bool,
    jaywalking: bool,
    decided: bool,
    rng: SimRng,
}

impl Pedestrian {
    /// The walkway edge it is on (None while respawning).
    pub fn edge(&self) -> Option<u32> {
        self.legs.get(self.leg).map(|l| l.0)
    }

    /// Stand at `p` for `t` s, then be respawned (tests, tools).
    pub fn stand_at(&mut self, p: DVec2, t: f64) {
        self.pos = p;
        self.vel = DVec2::ZERO;
        self.state = PedState::Dwelling;
        self.timer = t;
        self.dest = None;
    }

    /// Whether it is on (or committed to) crossing `k` of the lane graph.
    fn on_crossing(&self, w: &Walkways) -> Option<(u32, u32, bool)> {
        if self.state != PedState::Crossing {
            return None;
        }
        // Committed at the curb: the next leg; on it: this one.
        for k in [self.leg, self.leg + 1] {
            if let Some(&(e, fwd)) = self.legs.get(k)
                && let Some(c) = w.edges()[e as usize].crossing
            {
                return Some((c, e, fwd));
            }
        }
        None
    }
}

/// Counts over the episode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CrowdStats {
    /// Crossings started, and of those against the light.
    pub crossings: u64,
    pub red_starts: u64,
    /// Pedestrians hit by vehicles.
    pub hits: u64,
    /// Destinations reached, and respawns.
    pub arrivals: u64,
    pub respawns: u64,
}

/// A vehicle as the crowd sees it.
#[derive(Clone, Copy, Debug)]
struct Car {
    agent: usize,
    pos: DVec2,
    vel: DVec2,
    /// Horizontal reach of its colliders around `pos` (m).
    reach: f64,
}

/// All pedestrians of a world.
#[derive(Clone, Debug, Default)]
pub struct Crowd {
    pub peds: Vec<Pedestrian>,
    pub stats: CrowdStats,
    /// Per crossing of the lane graph: (position, unit direction of travel) of each
    /// pedestrian on it or committed to it.
    users: Vec<Vec<(DVec2, DVec2)>>,
    /// Cumulative lengths of the edges pedestrians spawn on (not crossings).
    spawn_edges: Vec<(f64, u32)>,
    grid: Grid,
    /// Updates since the reset.
    updates: u64,
    seed: Option<Seed>,
}

impl Crowd {
    /// A new crowd for the episode on `world` (empty without a pedestrian network), placed
    /// clear of `shapes`.
    pub fn reset(&mut self, spec: &PedestrianSpec, world: &StaticWorld, seed: Seed, shapes: &[AgentShape]) {
        self.peds.clear();
        self.stats = CrowdStats::default();
        self.updates = 0;
        self.seed = Some(seed);
        self.users = vec![Vec::new(); world.roads().lanes().crossings().len()];
        self.spawn_edges.clear();
        if spec.count == 0 || !world.roads().has_sections() {
            return;
        }
        let w = world.walkways();
        if w.is_empty() {
            return;
        }
        let mut total = 0.0;
        for (k, e) in w.edges().iter().enumerate() {
            if e.kind != WalkKind::Crossing {
                total += e.line.length();
                self.spawn_edges.push((total, k as u32));
            }
        }
        let cars = cars(world, shapes);
        for i in 0..spec.count {
            let mut rng = seed.child_index(i as u64).rng();
            let speed = (spec.speed[0] + spec.speed[1] * rng.normal()).clamp(0.5, 2.5);
            let height = rng.range(spec.height[0], spec.height[1]);
            let mut p = Pedestrian {
                pos: DVec2::ZERO,
                z: 0.0,
                vel: DVec2::ZERO,
                heading: 0.0,
                speed,
                radius: spec.radius,
                height,
                state: PedState::Walking,
                legs: Vec::new(),
                leg: 0,
                dest: None,
                timer: 0.0,
                attentive: true,
                jaywalking: false,
                decided: false,
                rng,
            };
            let mut prng = p.rng.clone();
            self.place(&mut p, world, &[], 0.0, &cars, &mut prng);
            p.rng = prng;
            self.peds.push(p);
        }
        self.update_users(w);
    }

    /// Put `p` at a random point of the network at least `clear` m from `learners`, clear of
    /// the vehicles and other pedestrians, on a trip to a random place; false if no spot was
    /// found (it then stays where it is).
    fn place(
        &self,
        p: &mut Pedestrian,
        world: &StaticWorld,
        learners: &[DVec2],
        clear: f64,
        cars: &[Car],
        rng: &mut SimRng,
    ) -> bool {
        let w = world.walkways();
        let Some(&(total, _)) = self.spawn_edges.last() else { return false };
        for _ in 0..SPOT_TRIES {
            let x = rng.uniform() * total;
            let k = self.spawn_edges.partition_point(|&(c, _)| c < x).min(self.spawn_edges.len() - 1);
            let e = self.spawn_edges[k].1;
            let line = &w.edges()[e as usize].line;
            let s = rng.uniform() * line.length();
            let at = line.point_at(s).truncate();
            if learners.iter().any(|l| l.distance(at) < clear)
                || cars.iter().any(|c| c.pos.distance(at) < c.reach + p.radius + 0.5)
                || self.peds.iter().any(|q| q.pos.distance(at) < q.radius + p.radius + 0.1)
            {
                continue;
            }
            let Some(legs) = trip_from_edge(w, e, s, rng) else { continue };
            p.pos = at;
            p.z = world.terrain().height(at.x, at.y);
            p.vel = DVec2::ZERO;
            p.heading = leg_heading(w, legs.0[0], s);
            p.legs = legs.0;
            p.dest = Some(legs.1);
            p.leg = 0;
            p.state = PedState::Walking;
            p.timer = 0.0;
            p.decided = false;
            return true;
        }
        false
    }

    /// The pedestrians on each crossing of the lane graph or committed to it: (position, unit
    /// direction of travel), by crossing.
    pub fn crossing_users(&self) -> &[Vec<(DVec2, DVec2)>] {
        &self.users
    }

    fn update_users(&mut self, w: &Walkways) {
        self.users.iter_mut().for_each(Vec::clear);
        for p in &self.peds {
            if let Some((c, e, fwd)) = p.on_crossing(w) {
                let line = &w.edges()[e as usize].line;
                let (a, b) = (line.points()[0].truncate(), line.points().last().expect("points").truncate());
                let dir = if fwd { b - a } else { a - b };
                if let Some(u) = self.users.get_mut(c as usize) {
                    u.push((p.pos, dir.normalize_or_zero()));
                }
            }
        }
    }

    /// One update of `dt` s at time `t` (after the tick): move every pedestrian, then find the
    /// vehicles among `shapes` (agents) that hit one and raise their event in `events`.
    /// `learning[i]` tells whether agent `i` is a learning agent (for respawns).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        spec: &PedestrianSpec,
        world: &StaticWorld,
        signals: &Signals,
        t: f64,
        dt: f64,
        shapes: &[AgentShape],
        learning: &[bool],
        events: &mut [Events],
    ) {
        if self.peds.is_empty() {
            return;
        }
        let w = world.walkways();
        let lanes = world.roads().lanes();
        let cars = cars(world, shapes);
        let sf = &spec.social;
        let n = self.peds.len();
        let (lo, hi) = world.extent();
        self.grid.build(lo, hi, sf.range.max(1.0), self.peds.iter().map(|p| p.pos));
        let snapshot: Vec<(DVec2, DVec2)> = self.peds.iter().map(|p| (p.pos, p.vel)).collect();
        let mut accel = vec![DVec2::ZERO; n];
        let mut stats = self.stats;
        for i in 0..n {
            let p = &mut self.peds[i];
            let desired = match p.state {
                PedState::Dwelling | PedState::Hit => {
                    p.timer -= dt;
                    DVec2::ZERO
                }
                _ => steer(p, w, lanes, signals, spec, &cars, t, dt, &mut stats),
            };
            let mut a = sf.driving(p.vel, desired);
            if p.state != PedState::Hit {
                let (pi, vi) = snapshot[i];
                self.grid.visit(pi, sf.range, |j| {
                    if j as usize != i {
                        let (q, wj) = snapshot[j as usize];
                        a += sf.interaction(pi, vi, q, wj);
                    }
                });
            }
            accel[i] = a;
        }
        // Integrate, keep to the walkways.
        for (p, a) in self.peds.iter_mut().zip(&accel) {
            if p.state == PedState::Hit {
                p.vel = DVec2::ZERO;
                continue;
            }
            p.vel += *a * dt;
            let cap = SPEED_CAP * p.speed * if p.state == PedState::Crossing { HURRY } else { 1.0 };
            if p.vel.length() > cap {
                p.vel *= cap / p.vel.length();
            }
            p.pos += p.vel * dt;
            confine(p, w);
            if p.vel.length() > 0.1 {
                p.heading = p.vel.to_angle();
            }
        }
        // Push overlapping bodies apart (half each).
        self.grid.build(lo, hi, sf.range.max(1.0), self.peds.iter().map(|p| p.pos));
        for i in 0..n {
            let (pi, ri) = (self.peds[i].pos, self.peds[i].radius);
            let mut push = DVec2::ZERO;
            self.grid.visit(pi, 2.0 * ri + 0.5, |j| {
                let j = j as usize;
                if j == i {
                    return;
                }
                let q = &self.peds[j];
                let r = pi - q.pos;
                let d = r.length();
                let overlap = ri + q.radius - d;
                if overlap > 0.0 {
                    let dir = if d > 1e-9 { r / d } else { DVec2::from_angle(i as f64) };
                    push += 0.5 * overlap * dir;
                }
            });
            let p = &mut self.peds[i];
            if p.state != PedState::Hit {
                p.pos += push;
                confine(p, w);
            }
            p.z = world.terrain().height(p.pos.x, p.pos.y);
        }
        // Hits.
        for c in cars.iter().filter(|c| c.vel.length() > HIT_SPEED) {
            let s = &shapes[c.agent];
            for p in self.peds.iter_mut().filter(|p| p.state != PedState::Hit) {
                if p.pos.distance(c.pos) > c.reach + p.radius {
                    continue;
                }
                let touch = s.spheres.iter().any(|sp| {
                    sp.center.truncate().distance(p.pos) < sp.radius + p.radius
                        && sp.center.z + sp.radius > p.z
                        && sp.center.z - sp.radius < p.z + p.height
                });
                if touch {
                    p.state = PedState::Hit;
                    p.timer = HIT_HOLD;
                    p.vel = DVec2::ZERO;
                    stats.hits += 1;
                    events[c.agent] |= Events::PEDESTRIAN_HIT;
                }
            }
        }
        // Arrivals, departures and respawns.
        let learners: Vec<DVec2> = (shapes.iter().enumerate())
            .filter(|&(k, s)| s.active && learning.get(k).copied().unwrap_or(false))
            .map(|(_, s)| s.center.truncate())
            .collect();
        let rs = self.seed.unwrap_or(Seed::from_u64(0)).child("respawn").child_index(self.updates);
        for i in 0..n {
            let p = &mut self.peds[i];
            let respawn = match p.state {
                PedState::Hit => p.timer <= 0.0,
                PedState::Dwelling if p.timer <= 0.0 => {
                    // Into the building, or on to another place.
                    let place = p.dest.map(|d| w.places()[d as usize]);
                    match place {
                        Some(pl) if !matches!(pl.kind, PlaceKind::Entrance { .. }) => {
                            match trip_from_node(w, pl.node, &mut p.rng) {
                                Some((legs, dest)) => {
                                    p.legs = legs;
                                    p.leg = 0;
                                    p.dest = Some(dest);
                                    p.state = PedState::Walking;
                                    false
                                }
                                None => true,
                            }
                        }
                        _ => true,
                    }
                }
                PedState::Waiting => p.timer > STUCK,
                _ => p.legs.is_empty(),
            };
            if respawn {
                let mut p = self.peds[i].clone();
                let mut rng = rs.child_index(i as u64).rng();
                if self.place(&mut p, world, &learners, spec.respawn_clear, &cars, &mut rng) {
                    stats.respawns += 1;
                    self.peds[i] = p;
                }
            }
        }
        self.stats = stats;
        self.updates += 1;
        self.update_users(w);
    }

    /// Hash input for [`WorldInstance::state_hash`](crate::WorldInstance::state_hash).
    pub fn hash_into(&self, f: &mut dyn FnMut(f64)) {
        for p in &self.peds {
            for x in [p.pos.x, p.pos.y, p.vel.x, p.vel.y, p.leg as f64, p.state as u8 as f64] {
                f(x);
            }
        }
    }
}

/// The vehicles among `shapes`: active agents near the ground.
fn cars(world: &StaticWorld, shapes: &[AgentShape]) -> Vec<Car> {
    (shapes.iter().enumerate())
        .filter(|(_, s)| s.active && !s.spheres.is_empty())
        .filter(|(_, s)| s.center.z - world.terrain().height(s.center.x, s.center.y) < GROUND_HEIGHT)
        .map(|(k, s)| {
            let pos = s.center.truncate();
            let reach = s.spheres.iter().map(|sp| sp.center.truncate().distance(pos) + sp.radius).fold(0.0, f64::max);
            Car { agent: k, pos, vel: s.velocity.truncate(), reach }
        })
        .collect()
}

/// The line of a leg in its direction of travel: (edge, station along it in that
/// direction → point).
fn leg_point(w: &Walkways, (e, fwd): (u32, bool), s: f64) -> DVec2 {
    let line = &w.edges()[e as usize].line;
    let s = if fwd { s } else { line.length() - s };
    line.point_at(s.clamp(0.0, line.length())).truncate()
}

fn leg_heading(w: &Walkways, (e, fwd): (u32, bool), s: f64) -> f64 {
    let line = &w.edges()[e as usize].line;
    let s = if fwd { s } else { line.length() - s };
    let h = line.heading_at(s.clamp(0.0, line.length()));
    if fwd { h } else { h + std::f64::consts::PI }
}

/// Station (m, in the direction of travel) and offset (m, positive to the left of that
/// direction) of `p` on a leg.
fn leg_place(w: &Walkways, (e, fwd): (u32, bool), p: DVec2) -> (f64, f64, f64) {
    let line = &w.edges()[e as usize].line;
    let pr = line.project(p);
    if fwd { (pr.station, pr.offset, line.length()) } else { (line.length() - pr.station, -pr.offset, line.length()) }
}

/// A trip from station `s` of edge `e` to a random place: the legs and the place.
fn trip_from_edge(w: &Walkways, e: u32, s: f64, rng: &mut SimRng) -> Option<(Vec<(u32, bool)>, u32)> {
    let edge = &w.edges()[e as usize];
    for _ in 0..4 {
        let dest = rng.below(w.places().len() as u64) as u32;
        let to = w.places()[dest as usize].node;
        let (ra, rb) = (w.route(edge.a, to)?, w.route(edge.b, to)?);
        let len = edge.line.length();
        let (fwd, r) = if s + rb.length < len - s + ra.length { (true, rb) } else { (false, ra) };
        // Walk to its end in the direction chosen, then on.
        let legs: Vec<(u32, bool)> = std::iter::once((e, fwd)).chain(r.edges).collect();
        if legs.len() > 1 || r.length > 0.0 || (fwd && len - s > 1.0) || (!fwd && s > 1.0) {
            return Some((legs, dest));
        }
    }
    None
}

/// A trip from node `n` to a random other place.
fn trip_from_node(w: &Walkways, n: u32, rng: &mut SimRng) -> Option<(Vec<(u32, bool)>, u32)> {
    for _ in 0..4 {
        let dest = rng.below(w.places().len() as u64) as u32;
        let to = w.places()[dest as usize].node;
        if to == n {
            continue;
        }
        let r = w.route(n, to)?;
        if !r.edges.is_empty() {
            return Some((r.edges, dest));
        }
    }
    None
}

/// Keep `p` within its walkway's width (except within [`END_SLACK`] m of the ends of its
/// line, where walkways meet).
fn confine(p: &mut Pedestrian, w: &Walkways) {
    let Some(&leg) = p.legs.get(p.leg) else { return };
    let edge = &w.edges()[leg.0 as usize];
    let pr = edge.line.project(p.pos);
    if pr.station <= END_SLACK || pr.station >= edge.line.length() - END_SLACK {
        return;
    }
    let half = (0.5 * edge.width - p.radius).max(0.0);
    if pr.offset.abs() > half {
        let left = DVec2::from_angle(pr.heading).perp();
        let side = pr.offset.signum();
        p.pos -= (pr.offset - side * half) * left;
        let out = p.vel.dot(left) * side;
        if out > 0.0 {
            p.vel -= out * side * left;
        }
    }
}

/// The desired velocity of a walking, waiting or crossing pedestrian, advancing its route
/// and making its decisions at crossings.
#[allow(clippy::too_many_arguments)]
fn steer(
    p: &mut Pedestrian,
    w: &Walkways,
    lanes: &autonomousim_world::lanes::LaneGraph,
    signals: &Signals,
    spec: &PedestrianSpec,
    cars: &[Car],
    t: f64,
    dt: f64,
    stats: &mut CrowdStats,
) -> DVec2 {
    let Some(&leg) = p.legs.get(p.leg) else { return DVec2::ZERO };
    let (mut s, mut offset, mut len) = leg_place(w, leg, p.pos);
    // On to the next leg past the end (onto a crossing only once committed).
    while (s >= len - 0.05 || p.pos.distance(leg_point(w, p.legs[p.leg], len)) < NODE_REACH) && p.leg + 1 < p.legs.len()
    {
        let next = p.legs[p.leg + 1];
        let next_crossing = w.edges()[next.0 as usize].crossing.is_some();
        if next_crossing && p.state != PedState::Crossing {
            break;
        }
        let was_crossing = w.edges()[p.legs[p.leg].0 as usize].crossing.is_some();
        p.leg += 1;
        if was_crossing && !next_crossing {
            p.state = PedState::Walking;
            p.decided = false;
        }
        (s, offset, len) = leg_place(w, next, p.pos);
    }
    let leg = p.legs[p.leg];
    let half = (0.5 * w.edges()[leg.0 as usize].width - p.radius).max(0.0);
    let offset = offset.clamp(-half, half);
    let last = p.leg + 1 == p.legs.len();
    let left = len - s;
    if last && left < ARRIVE {
        p.state = PedState::Dwelling;
        p.timer = p.rng.range(spec.dwell[0], spec.dwell[1]);
        p.decided = false;
        stats.arrivals += 1;
        return DVec2::ZERO;
    }
    // A crossing next: decide at the curb.
    let next_crossing =
        (!last).then(|| p.legs[p.leg + 1]).and_then(|l| w.edges()[l.0 as usize].crossing.map(|c| (c, l)));
    let on_crossing = w.edges()[leg.0 as usize].crossing.is_some();
    let mut speed = p.speed;
    if let Some((c, cl)) = next_crossing.filter(|_| p.state != PedState::Crossing && left < CURB) {
        if !p.decided {
            p.attentive = p.rng.chance(spec.attention);
            p.jaywalking = p.rng.chance(spec.jaywalk);
            p.decided = true;
            p.timer = 0.0;
        }
        let length = w.edges()[cl.0 as usize].line.length();
        let light = signals.crossing_walk(lanes, c, t).map(|(walk, left)| {
            walk && length / p.speed <= left + lanes.crossings()[c as usize].length / CLEARANCE_SPEED
        });
        let green = light.unwrap_or(true);
        let go = (green || p.jaywalking) && (!p.attentive || gap(w, cl, length / p.speed + 1.0, cars));
        if go {
            p.state = PedState::Crossing;
            stats.crossings += 1;
            if !green {
                stats.red_starts += 1;
            }
        } else {
            // Wait at the curb, at its own offset.
            p.state = PedState::Waiting;
            p.timer += dt;
            let end = leg_point(w, leg, len);
            let h = leg_heading(w, leg, len);
            let target = end + offset * DVec2::from_angle(h).perp();
            let d = target - p.pos;
            let dist = d.length();
            return if dist < 0.2 { DVec2::ZERO } else { d / dist * p.speed.min(dist / 0.5) };
        }
    }
    if on_crossing && p.state == PedState::Crossing && p.attentive {
        speed = avoid(p, w, leg, s, cars);
    }
    // Head for the point LOOK_AHEAD on along the route, at the same offset.
    let mut ahead = s + LOOK_AHEAD;
    let mut k = p.leg;
    while ahead > w.edges()[p.legs[k].0 as usize].line.length() && k + 1 < p.legs.len() {
        ahead -= w.edges()[p.legs[k].0 as usize].line.length();
        k += 1;
    }
    let target_leg = p.legs[k];
    let ahead = ahead.min(w.edges()[target_leg.0 as usize].line.length());
    let h = leg_heading(w, target_leg, ahead);
    let target_half = (0.5 * w.edges()[target_leg.0 as usize].width - p.radius).max(0.0);
    let target =
        leg_point(w, target_leg, ahead) + offset.clamp(-target_half, target_half) * DVec2::from_angle(h).perp();
    let d = target - p.pos;
    let dist = d.length();
    if dist < 1e-6 {
        return DVec2::ZERO;
    }
    // Slow down into the destination.
    let route_left = left + p.legs[p.leg + 1..].iter().map(|l| w.edges()[l.0 as usize].line.length()).sum::<f64>();
    let v = speed.min(route_left / 0.5 + 0.1);
    d / dist * v
}

/// Whether every vehicle heading for crossing leg `cl` can stop before it or arrives only
/// after `need` s.
fn gap(w: &Walkways, cl: (u32, bool), need: f64, cars: &[Car]) -> bool {
    let line = &w.edges()[cl.0 as usize].line;
    let (a, b) = (line.points()[0].truncate(), line.points().last().expect("points").truncate());
    let length = a.distance(b);
    if length < 1e-6 {
        return true;
    }
    let axis = (b - a) / length;
    let normal = axis.perp();
    let mid = 0.5 * (a + b);
    for c in cars.iter().filter(|c| c.pos.distance(mid) < VEHICLE_RANGE) {
        let h = (c.pos - a).dot(normal);
        let u = (c.pos - a).dot(axis);
        let on_band = h.abs() < c.reach + 0.5 * CROSSWALK && u > -c.reach && u < length + c.reach;
        let speed = c.vel.length();
        if speed < HIT_SPEED {
            // Standing: blocks only where it stands on the band's middle.
            if h.abs() < 0.5 * c.reach && u > 0.0 && u < length {
                return false;
            }
            continue;
        }
        let closing = -c.vel.dot(normal) * h.signum();
        if closing <= 0.0 {
            // Moving away, or along: blocks while on the band.
            if on_band && h.abs() < c.reach {
                return false;
            }
            continue;
        }
        let t_line = h.abs() / closing;
        let q = c.pos + c.vel * t_line;
        let uq = (q - a).dot(axis);
        if uq < -c.reach || uq > length + c.reach {
            continue;
        }
        if on_band {
            return false;
        }
        let path = speed * t_line - c.reach - 0.5 * CROSSWALK;
        if path > speed * speed / (2.0 * YIELD_DECEL) + 1.0 {
            continue;
        }
        if path.max(0.0) / speed < need {
            return false;
        }
    }
    true
}

/// Speed (m/s) at which a pedestrian on a crossing keeps clear of the vehicles: its desired
/// speed, hurrying, or standing. Of options that all come too close, the one doing so latest,
/// and of those (already too close) the one passing farthest from them.
fn avoid(p: &Pedestrian, w: &Walkways, leg: (u32, bool), s: f64, cars: &[Car]) -> f64 {
    let dir = DVec2::from_angle(leg_heading(w, leg, s));
    let options = [p.speed, (HURRY * p.speed).min(2.5), 0.0];
    let mut best = (f64::NEG_INFINITY, f64::NEG_INFINITY, p.speed);
    for &v in &options {
        // Earliest time within the horizon at which it would come too close to a vehicle, and
        // the least margin by which it passes them.
        let (mut first, mut margin) = (f64::INFINITY, f64::INFINITY);
        for c in cars.iter().filter(|c| c.pos.distance(p.pos) < VEHICLE_RANGE * 0.5) {
            let r = c.pos - p.pos;
            let u = c.vel - dir * v;
            let need = c.reach + p.radius + VEHICLE_CLEARANCE;
            let uu = u.length_squared();
            let tc = if uu > 1e-12 { (-r.dot(u) / uu).clamp(0.0, HORIZON) } else { 0.0 };
            let closest = (r + u * tc).length();
            margin = margin.min(closest - need);
            if closest < need {
                // When it gets within `need`.
                let b = r.dot(u);
                let cc = r.length_squared() - need * need;
                let disc = b * b - uu * cc;
                let enter = if cc <= 0.0 || uu < 1e-12 { 0.0 } else { (-b - disc.max(0.0).sqrt()) / uu };
                first = first.min(enter.max(0.0));
            }
        }
        if first.is_infinite() {
            return v;
        }
        if first > best.0 || (first == best.0 && margin > best.1) {
            best = (first, margin, v);
        }
    }
    best.2
}

/// A uniform grid over the map of point indices, rebuilt by counting sort.
#[derive(Clone, Debug, Default)]
struct Grid {
    lo: DVec2,
    cell: f64,
    nx: usize,
    ny: usize,
    start: Vec<u32>,
    items: Vec<u32>,
    points: Vec<DVec2>,
}

impl Grid {
    fn key(&self, p: DVec2) -> usize {
        let x = (((p.x - self.lo.x) / self.cell).floor().max(0.0) as usize).min(self.nx - 1);
        let y = (((p.y - self.lo.y) / self.cell).floor().max(0.0) as usize).min(self.ny - 1);
        y * self.nx + x
    }

    fn build(&mut self, lo: DVec2, hi: DVec2, cell: f64, points: impl Iterator<Item = DVec2>) {
        self.lo = lo;
        self.cell = cell;
        self.nx = (((hi.x - lo.x) / cell).ceil() as usize).max(1);
        self.ny = (((hi.y - lo.y) / cell).ceil() as usize).max(1);
        self.points.clear();
        self.points.extend(points);
        self.start.clear();
        self.start.resize(self.nx * self.ny + 1, 0);
        for k in 0..self.points.len() {
            let c = self.key(self.points[k]);
            self.start[c + 1] += 1;
        }
        for c in 0..self.nx * self.ny {
            self.start[c + 1] += self.start[c];
        }
        self.items.clear();
        self.items.resize(self.points.len(), 0);
        let mut fill = self.start.clone();
        for k in 0..self.points.len() {
            let c = self.key(self.points[k]);
            self.items[fill[c] as usize] = k as u32;
            fill[c] += 1;
        }
    }

    /// Call `f` with every point within `r` of `p` (in cell order).
    fn visit(&self, p: DVec2, r: f64, mut f: impl FnMut(u32)) {
        if self.points.is_empty() {
            return;
        }
        let cx = |x: f64| (((x - self.lo.x) / self.cell).floor().max(0.0) as usize).min(self.nx - 1);
        let cy = |y: f64| (((y - self.lo.y) / self.cell).floor().max(0.0) as usize).min(self.ny - 1);
        for y in cy(p.y - r)..=cy(p.y + r) {
            for x in cx(p.x - r)..=cx(p.x + r) {
                let c = y * self.nx + x;
                for &k in &self.items[self.start[c] as usize..self.start[c + 1] as usize] {
                    if self.points[k as usize].distance_squared(p) <= r * r {
                        f(k);
                    }
                }
            }
        }
    }
}
