//! Scripted drivers: groups whose agents are driven by Rust code instead of taking actions.
//!
//! A group with `driver = { type = "road" | "traffic", ... }` takes no actions from the caller (its action
//! array may be empty and is ignored); at the start of every policy step the driver commands
//! each active agent's controller with a speed and a path curvature
//! ([`GroundSetpoint::SpeedCurvature`]), whatever the group's action mode.
//!
//! The `road` driver drives around the road network at random, in its lane, on the roads of
//! its `roads` classes (paved by default: rural gravel roads and tracks are too narrow to turn
//! a car on). Its agents spawn on those roads, away from the map's edges, and follow a random
//! walk over them that turns at random at every junction, planned `leg` metres at a time as
//! the agent's `route` (which the `road` and `route` observation terms follow) and extended
//! before its end. At dead ends and before the map's edges the vehicle stops and turns round
//! in a K-turn (forward on full left lock, back on full right lock, … within the road's
//! edges), then walks on the other way. Steering is pure pursuit on the lane line; the speed
//! is a random cruise speed per leg, limited ahead of bends by the lateral acceleration and the
//! braking deceleration, with optional random stops. It keeps its distance from agents ahead
//! near the ground (vehicles, landed drones; it does not overtake), and K-turns keep clear of
//! other agents and solid obstacles. Everything is drawn from the agent's seed stream, so the
//! driving is deterministic.
//!
//! The `traffic` driver drives the lane graph with IDM and MOBIL (see
//! [`traffic_driver`](crate::traffic_driver)).

use crate::interaction::Sphere;
use crate::lane::{self, WalkPos};
use crate::scenario::CompiledGroup;
use crate::traffic_driver::{TrafficDriver, TrafficDriverSpec};
use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::wrap_angle;
use autonomousim_core::math::quat::yaw;
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::terrain::Terrain;
use autonomousim_world::{Polyline, RoadClass, StaticWorld};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A scripted driver of a group of ground vehicles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DriverSpec {
    /// Random routes along the roads (see the module documentation).
    Road(RoadDriverSpec),
    /// Traffic on the lane graph (IDM, MOBIL; see [`traffic_driver`](crate::traffic_driver)).
    Traffic(TrafficDriverSpec),
}

impl DriverSpec {
    pub fn name(&self) -> &'static str {
        match self {
            DriverSpec::Road(_) => "road",
            DriverSpec::Traffic(_) => "traffic",
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            DriverSpec::Road(r) => r.validate(),
            DriverSpec::Traffic(t) => t.validate(),
        }
    }

    /// A driver of this kind for an agent of `group`.
    pub fn driver(&self, group: &CompiledGroup) -> Driver {
        match self {
            DriverSpec::Road(r) => Driver::Road(Box::new(RoadDriver::new(r, DriverGeometry::of(group)))),
            DriverSpec::Traffic(t) => Driver::Traffic(Box::new(TrafficDriver::new(t, DriverGeometry::of(group)))),
        }
    }
}

/// The scripted driver of an agent.
#[derive(Clone, Debug)]
pub enum Driver {
    Road(Box<RoadDriver>),
    Traffic(Box<TrafficDriver>),
}

impl Driver {
    pub fn as_road(&self) -> Option<&RoadDriver> {
        match self {
            Driver::Road(d) => Some(d),
            Driver::Traffic(_) => None,
        }
    }

    pub fn as_traffic(&self) -> Option<&TrafficDriver> {
        match self {
            Driver::Traffic(d) => Some(d),
            Driver::Road(_) => None,
        }
    }
}

/// Settings of the `road` driver.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RoadDriverSpec {
    /// Range of the cruise speed, drawn per leg (m/s).
    pub speed: [f64; 2],
    /// Largest lateral acceleration in bends (m/s²).
    pub lateral_accel: f64,
    /// Deceleration planned for when slowing down ahead of bends (m/s²).
    pub decel: f64,
    /// Pure-pursuit look-ahead: `lookahead[0] + lookahead[1]·speed` (m, s).
    pub lookahead: [f64; 2],
    /// Route length range of each leg (m).
    pub leg: [f64; 2],
    /// Random stops per second of driving (0: none), each lasting a time in `stop_time` (s).
    pub stop_rate: f64,
    pub stop_time: [f64; 2],
    /// Road classes driven on.
    pub roads: Vec<RoadClass>,
}

impl Default for RoadDriverSpec {
    fn default() -> Self {
        Self {
            speed: [3.0, 12.0],
            lateral_accel: 2.0,
            decel: 2.0,
            lookahead: [3.0, 0.5],
            leg: [150.0, 500.0],
            stop_rate: 0.0,
            stop_time: [2.0, 6.0],
            roads: vec![RoadClass::Paved],
        }
    }
}

impl RoadDriverSpec {
    fn validate(&self) -> Result<(), String> {
        let range = |r: [f64; 2]| r[0].is_finite() && r[1].is_finite() && 0.0 <= r[0] && r[0] <= r[1];
        let ok = range(self.speed)
            && self.speed[1] > 0.0
            && range(self.leg)
            && self.leg[1] >= 20.0
            && range(self.stop_time)
            && self.lateral_accel > 0.0
            && self.decel > 0.0
            && self.lookahead[0] > 0.0
            && self.lookahead[1] >= 0.0
            && self.stop_rate >= 0.0
            && self.stop_rate.is_finite()
            && !self.roads.is_empty();
        if ok { Ok(()) } else { Err(format!("invalid road driver {self:?}")) }
    }
}

/// Extend the route when less than this much of it lies ahead, beyond the braking distance (m).
const EXTEND_AHEAD: f64 = 40.0;
/// Route behind the vehicle kept when extending (m).
const KEEP_BEHIND: f64 = 10.0;
/// Spacing of the curvature samples ahead (m).
const PREVIEW_STEP: f64 = 2.5;
/// Part of the route searched for the vehicle's station each step: from `BACK` m behind the
/// last station to `SEARCH` m past it (so that routes crossing themselves do not confuse it).
const BACK: f64 = 2.0;
const SEARCH: f64 = 30.0;
/// K-turns: speed (m/s), part of the steering lock used, how far the corners may overhang the
/// road's edges (m, over the verge), clearance kept from obstacles and other agents and
/// look-ahead of the corners leading (m), time to steer at a standstill before each shunt
/// (s), heading error at which the turn ends (rad), and the most shunts before giving up.
const TURN_SPEED: f64 = 0.8;
const TURN_LOCK: f64 = 0.9;
const TURN_OVERHANG: f64 = 0.6;
const TURN_MARGIN: f64 = 0.3;
const TURN_LOOKAHEAD: f64 = 0.3;
/// Height of the ground under the corners above the road's centre line at which they stop (m).
const TURN_BANK: f64 = 0.15;
const TURN_SETTLE: f64 = 1.0;
const TURN_DONE: f64 = 0.35;
const TURN_SHUNTS: u32 = 40;
/// Following: agents count as ahead up to this far (m) and within this height of the vehicle
/// (m), and the vehicle stops this far behind them (m, between its front and their centre
/// less their radius, at most `FOLLOW_RADIUS`).
const FOLLOW_RANGE: f64 = 60.0;
const FOLLOW_HEIGHT: f64 = 2.5;
const FOLLOW_GAP: f64 = 4.0;
/// The gap kept when the vehicle's route ends in a turn within `TURN_QUEUE` (m): room for the
/// vehicle ahead to turn round first.
const TURN_GAP: f64 = 12.0;
const TURN_QUEUE: f64 = 40.0;
/// While another agent is within `TURN_ZONE` (m) of its turning point, a vehicle stops
/// `TURN_HOLD` (m) before it.
const TURN_ZONE: f64 = 10.0;
const TURN_HOLD: f64 = 16.0;
const FOLLOW_RADIUS: f64 = 2.5;
/// Clearance: the vehicle's footprint, moved along its steered arc in steps of `CLEAR_STEP`
/// (m) up to its braking distance (at most `CLEAR_RANGE`), keeps `CLEAR_MARGIN` (m) from the
/// spheres of other agents.
const CLEAR_STEP: f64 = 0.5;
const CLEAR_RANGE: f64 = 20.0;
const CLEAR_MARGIN: f64 = 0.4;

/// Another agent as drivers see it: its centre and bounding radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Traffic<'a> {
    pub center: DVec3,
    pub radius: f64,
    /// Its collision spheres (world frame).
    pub spheres: &'a [Sphere],
}

/// The shape of a vehicle for K-turns: its front and rear ends along x and half its width
/// (m, chassis frame), and the path curvature of its steering lock (1/m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriverGeometry {
    pub front: f64,
    pub rear: f64,
    pub half_width: f64,
    pub max_curvature: f64,
}

impl DriverGeometry {
    /// That of a group of (single-unit) ground vehicles.
    pub fn of(group: &CompiledGroup) -> Self {
        let d = group.def.as_wheeled().expect("drivers drive ground vehicles");
        let wheels = (0..d.num_wheels()).map(|w| (d.wheel_position(w).x, d.wheel_tire(w).radius()));
        let spheres = d.sphere_colliders().into_iter().map(|c| (c.center.x, c.radius));
        let parts: Vec<(f64, f64)> = wheels.chain(spheres).collect();
        Self {
            front: parts.iter().map(|&(x, r)| x + r).fold(0.0, f64::max),
            rear: parts.iter().map(|&(x, r)| x - r).fold(0.0, f64::min),
            half_width: group.half_width,
            max_curvature: group.action_map.as_ground().map_or(0.2, |m| m.curvature()),
        }
    }
}

/// A K-turn in progress.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turn {
    /// Heading to turn to (rad).
    pub target: f64,
    pub forward: bool,
    /// Time left steering at a standstill before moving (s).
    pub settle: f64,
    pub shunts: u32,
}

/// State of a `road` driver.
#[derive(Clone, Debug)]
pub struct RoadDriver {
    spec: RoadDriverSpec,
    geometry: DriverGeometry,
    rng: SimRng,
    /// Cruise speed of the current leg (m/s).
    pub cruise: f64,
    /// Time left of the current stop (s).
    pub stop_left: f64,
    /// Station of the vehicle along its route (m).
    pub station: f64,
    /// Where the route ends on the network, and whether it ends at a turning point.
    pub end: Option<WalkPos>,
    pub turn_at_end: bool,
    pub turn: Option<Turn>,
}

impl RoadDriver {
    pub fn new(spec: &RoadDriverSpec, geometry: DriverGeometry) -> Self {
        Self {
            spec: spec.clone(),
            geometry,
            rng: Seed::from_u64(0).rng(),
            cruise: 0.0,
            stop_left: 0.0,
            station: 0.0,
            end: None,
            turn_at_end: false,
            turn: None,
        }
    }

    /// The roads of `world` driven on.
    pub fn roads(&self, world: &StaticWorld) -> Vec<bool> {
        world.roads().roads().iter().map(|r| self.spec.roads.contains(&r.class)).collect()
    }

    /// Start an episode at `start` (the spawn's place on the network; `None`: the vehicle
    /// stands still) with a new random stream: the first route.
    pub fn reset(&mut self, seed: Seed, world: &StaticWorld, start: Option<WalkPos>) -> Option<Arc<Polyline>> {
        self.rng = seed.rng();
        self.stop_left = 0.0;
        self.station = 0.0;
        self.end = None;
        self.turn_at_end = false;
        self.turn = None;
        self.cruise = self.draw_cruise();
        Some(Arc::new(self.walk(world, start?)))
    }

    /// The next leg from `from`, which becomes the route's end.
    fn walk(&mut self, world: &StaticWorld, from: WalkPos) -> Polyline {
        let roads = self.roads(world);
        let length = self.rng.range(self.spec.leg[0], self.spec.leg[1]).max(20.0);
        let w = lane::drive_walk(world, &roads, from, length, &mut self.rng);
        self.end = Some(w.end);
        self.turn_at_end = w.turn;
        w.line
    }

    fn draw_cruise(&mut self) -> f64 {
        self.rng.range(self.spec.speed[0], self.spec.speed[1])
    }

    /// The command for the next policy step of `dt` of a vehicle at `pose` moving at `speed`
    /// (m/s, forward) among the other agents `others`; extends `route` when its end comes near.
    pub fn drive(
        &mut self,
        world: &StaticWorld,
        route: &mut Option<Arc<Polyline>>,
        pose: &Pose,
        speed: f64,
        dt: f64,
        others: &[Traffic<'_>],
    ) -> GroundSetpoint {
        let stand = GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 };
        let Some(mut line) = route.clone() else { return stand };
        if self.turn.is_some() {
            return self.turning(world, route, pose, speed, dt, others);
        }
        let (xy, heading) = (pose.pos.truncate(), yaw(pose.rot));
        self.station = local_station(&line, xy, self.station);
        let braking = self.cruise * self.cruise / (2.0 * self.spec.decel);
        if !self.turn_at_end
            && line.length() - self.station < EXTEND_AHEAD + braking
            && let Some(next) = self.extend(world, &line)
        {
            line = Arc::new(next);
            *route = Some(line.clone());
            self.station = local_station(&line, xy, 0.0);
        }
        let left = line.length() - self.station;
        // Another agent near the turning point: wait back from it until it has gone.
        let end = line.point_at(line.length());
        let busy = self.turn_at_end
            && others.iter().any(|o| {
                (o.center - end).z.abs() < FOLLOW_HEIGHT
                    && (o.center - end).truncate().length() - o.radius.min(FOLLOW_RADIUS) < TURN_ZONE
            });
        if self.turn_at_end && !busy && left < 4.0 && speed.abs() < 0.3 {
            let target = wrap_angle(line.heading_at(line.length()) + std::f64::consts::PI);
            self.turn = Some(Turn { target, forward: true, settle: TURN_SETTLE, shunts: 0 });
            return self.turning(world, route, pose, speed, dt, others);
        }
        let s = &self.spec;

        // Stops.
        if self.stop_left > 0.0 {
            self.stop_left -= dt;
        } else if s.stop_rate > 0.0 && self.rng.uniform() < 1.0 - (-s.stop_rate * dt).exp() {
            self.stop_left = self.rng.range(s.stop_time[0], s.stop_time[1]);
        }

        // Speed: the cruise speed, slowed ahead of bends and before the end of the route.
        let mut target = if self.stop_left > 0.0 { 0.0 } else { self.cruise };
        let preview = (s.lookahead[0] + braking).min(left);
        let mut d = 0.0;
        while d <= preview {
            let k = line.curvature_at(self.station + d).abs().max(1e-6);
            target = target.min((s.lateral_accel / k + 2.0 * s.decel * d).sqrt());
            d += PREVIEW_STEP;
        }
        let hold = if busy { TURN_HOLD } else { 2.0 };
        target = target.min((2.0 * s.decel * (left - hold).max(0.0)).sqrt());
        // Steering: pure pursuit towards the lane point one look-ahead further on.
        let reach = s.lookahead[0] + s.lookahead[1] * speed.abs();
        let to = line.point_at(self.station + reach).truncate() - xy;
        let local = DVec2::from_angle(-heading).rotate(to);
        let dist2 = local.length_squared();
        let curvature = if dist2 > 1e-6 { 2.0 * local.y / dist2 } else { 0.0 };
        // Following the agents ahead: centres beyond the front (those over the vehicle itself,
        // such as a drone landing on its roof, are not ahead of it).
        let g = self.geometry;
        let corridor = g.half_width + 1.2;
        let gap = if self.turn_at_end && left < TURN_QUEUE { TURN_GAP } else { FOLLOW_GAP };
        for o in others {
            let rel = o.center - pose.pos;
            let local = DVec2::from_angle(-heading).rotate(rel.truncate());
            if local.x > g.front && local.x < FOLLOW_RANGE && local.y.abs() < corridor && rel.z.abs() < FOLLOW_HEIGHT {
                let free = local.x - g.front - o.radius.min(FOLLOW_RADIUS) - gap;
                target = target.min((2.0 * s.decel * free.max(0.0)).sqrt());
            }
        }

        // Clearance: the footprint swept along the steered arc stops short of other agents'
        // spheres (ignoring those it already touches unless it would press closer).
        let reach = (target * target / (2.0 * s.decel) + CLEAR_STEP).min(CLEAR_RANGE);
        for o in others.iter().filter(|o| (o.center - pose.pos).truncate().length() < reach + g.front + o.radius) {
            for sp in o.spheres.iter().filter(|sp| (sp.center.z - pose.pos.z).abs() < FOLLOW_HEIGHT) {
                let at = |d: f64| footprint_distance(&g, xy, heading, curvature, d, sp.center.truncate()) - sp.radius;
                let limit = CLEAR_MARGIN.min(at(0.0) - 1e-3);
                let mut d = CLEAR_STEP;
                while d <= reach {
                    if at(d) < limit {
                        target = target.min((2.0 * s.decel * (d - CLEAR_STEP).max(0.0)).sqrt());
                        break;
                    }
                    d += CLEAR_STEP;
                }
            }
        }

        GroundSetpoint::SpeedCurvature { speed: target, curvature }
    }

    /// A step of the K-turn: shunts forward on left lock and back on right lock, each until
    /// its leading corners near the road's edges; then the walk goes on the other way.
    fn turning(
        &mut self,
        world: &StaticWorld,
        route: &mut Option<Arc<Polyline>>,
        pose: &Pose,
        speed: f64,
        dt: f64,
        others: &[Traffic<'_>],
    ) -> GroundSetpoint {
        let mut t = self.turn.expect("turning");
        let g = self.geometry;
        let error = wrap_angle(t.target - yaw(pose.rot));
        if error.abs() < TURN_DONE || t.shunts > TURN_SHUNTS {
            self.turn = None;
            let from = self.end.expect("a route's end").reversed();
            *route = Some(Arc::new(self.walk(world, from)));
            self.station = 0.0;
            self.cruise = self.draw_cruise();
            return GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 };
        }
        if t.settle > 0.0 {
            if speed.abs() < 0.05 {
                t.settle -= dt;
            }
        } else {
            // The corners leading, a little ahead.
            let x = if t.forward { g.front + TURN_LOOKAHEAD } else { g.rear - TURN_LOOKAHEAD };
            let net = world.roads();
            let outside = [g.half_width, 0.0, -g.half_width].into_iter().any(|y| {
                let p = pose.transform_point(DVec3::new(x, y, 0.0));
                // Beyond the verge, or over ground rising above the road.
                let terrain = world.terrain();
                let off_road = net.nearest(p.truncate(), 10.0).is_none_or(|rp| {
                    let q = rp.projection.point;
                    rp.projection.distance > 0.5 * net.roads()[rp.road as usize].width + TURN_OVERHANG
                        || terrain.height(p.x, p.y) > terrain.height(q.x, q.y) + TURN_BANK
                });
                let lifted = p + 0.5 * DVec3::Z;
                off_road
                    || world.obstacle_clearance(lifted, 1.0) < TURN_MARGIN
                    || others.iter().any(|o| {
                        (o.center - p).z.abs() < FOLLOW_HEIGHT
                            && (o.center - p).truncate().length() < o.radius.min(FOLLOW_RADIUS) + TURN_MARGIN
                    })
            });
            if outside {
                t = Turn { forward: !t.forward, settle: TURN_SETTLE, shunts: t.shunts + 1, ..t };
            }
        }
        self.turn = Some(t);
        let sign_now = if t.forward { 1.0 } else { -1.0 };
        let curvature = sign_now * TURN_LOCK * g.max_curvature;
        let speed = if t.settle > 0.0 { 0.0 } else { sign_now * TURN_SPEED };
        GroundSetpoint::SpeedCurvature { speed, curvature }
    }

    /// The route from `KEEP_BEHIND` behind the vehicle on, continued by the next leg of the
    /// walk.
    fn extend(&mut self, world: &StaticWorld, line: &Polyline) -> Option<Polyline> {
        let from = self.end?;
        let next = self.walk(world, from);
        self.cruise = self.draw_cruise();
        let mut pts: Vec<DVec3> = line.slice((self.station - KEEP_BEHIND).max(0.0), line.length());
        for p in next.points() {
            if pts.last().is_none_or(|q| q.truncate().distance(p.truncate()) > 0.5) {
                pts.push(*p);
            }
        }
        Some(Polyline::new(pts))
    }
}

/// Station of `xy` on `line`, searched near `last`.
/// Distance from `p` to the footprint of a vehicle at `xy` heading `heading` after driving
/// `d` forward along an arc of curvature `k` (0 inside).
fn footprint_distance(g: &DriverGeometry, xy: DVec2, heading: f64, k: f64, d: f64, p: DVec2) -> f64 {
    let (dx, dy) = if (k * d).abs() < 1e-9 { (d, 0.0) } else { ((k * d).sin() / k, (1.0 - (k * d).cos()) / k) };
    let at = xy + DVec2::from_angle(heading).rotate(DVec2::new(dx, dy));
    let local = DVec2::from_angle(-(heading + k * d)).rotate(p - at);
    let outside =
        DVec2::new((g.rear - local.x).max(local.x - g.front).max(0.0), (local.y.abs() - g.half_width).max(0.0));
    outside.length()
}

fn local_station(line: &Polyline, xy: DVec2, last: f64) -> f64 {
    let a = (last - BACK).max(0.0);
    let b = (last + SEARCH).min(line.length());
    if b - a < 1e-6 {
        return last.clamp(0.0, line.length());
    }
    a + Polyline::new(line.slice(a, b)).project(xy).station
}
