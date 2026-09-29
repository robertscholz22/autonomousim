//! Driving on the road network: lanes, routes to destinations, spawns on roads, and the
//! reference line the `road` and `route` observation terms follow.
//!
//! Traffic keeps right: on paved roads the lane centre lies a quarter of the width right of
//! the centre line; gravel roads and tracks are single-lane and driven on their centre line.

use crate::scenario::{Goal, GoalSpec, SpawnSpec};
use autonomousim_core::math::quat::wrap_angle;
use autonomousim_core::rng::SimRng;
use autonomousim_core::terrain::Terrain;
use autonomousim_world::{NodeKind, Polyline, Road, RoadClass, RoadNetwork, StaticWorld};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

/// How far from a road the road terms still find it when there is no route (m).
pub const ROAD_REACH: f64 = 30.0;

/// Where a `route` goal leads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteDestination {
    /// A farm yard.
    #[default]
    Yard,
    /// A random point on a road.
    Road,
}

/// Settings of `GoalKind::Route`: a route along the roads to a destination whose route length
/// lies in the goal spec's `distance` range, with goals every `step` m along its lane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RouteGoals {
    pub destination: RouteDestination,
    pub step: f64,
}

impl Default for RouteGoals {
    fn default() -> Self {
        Self { destination: RouteDestination::Yard, step: 25.0 }
    }
}

/// Offset of the lane centre to the right of `road`'s centre line (m).
pub fn lane_offset(road: &Road) -> f64 {
    match road.class {
        RoadClass::Paved => 0.25 * road.width,
        RoadClass::Gravel | RoadClass::Track => 0.0,
    }
}

/// Unit vector to the right of heading `h`.
fn right(h: f64) -> DVec2 {
    DVec2::new(h.sin(), -h.cos())
}

/// A centre-line route moved onto the lanes of the roads it follows.
pub fn lane_line(net: &RoadNetwork, route: &Polyline) -> Polyline {
    let pts = route.points();
    let n = pts.len();
    let moved = (0..n)
        .map(|i| {
            let (a, b) = (pts[i.saturating_sub(1)], pts[(i + 1).min(n - 1)]);
            let d = (b - a).truncate();
            let h = d.y.atan2(d.x);
            let p = pts[i];
            let off = net.nearest(p.truncate(), 1.0).map_or(0.0, |rp| lane_offset(&net.roads()[rp.road as usize]));
            (p.truncate() + off * right(h)).extend(p.z)
        })
        .collect();
    Polyline::new(moved)
}

/// Least distance of a road-point destination from the map's edges (m).
const EDGE_CLEARANCE: f64 = 20.0;

/// Least distance of a route from the map's edges (m): roads may run along them, and agents
/// meet `OUT_OF_BOUNDS` there.
const ROUTE_EDGE_CLEARANCE: f64 = 10.0;

/// A uniformly random point on the network (by length), away from the road ends: the road
/// and the station.
fn road_point(net: &RoadNetwork, rng: &mut SimRng) -> (usize, f64) {
    road_point_on(net, None, rng)
}

/// [`road_point`] on the roads marked in `only` (all without it or when none is marked).
fn road_point_on(net: &RoadNetwork, only: Option<&[bool]>, rng: &mut SimRng) -> (usize, f64) {
    let only = only.filter(|o| o.iter().any(|&b| b));
    let on = |k: usize| only.is_none_or(|o| o[k]);
    let total: f64 = net.roads().iter().enumerate().filter(|&(k, _)| on(k)).map(|(_, r)| r.line.length()).sum();
    let mut u = rng.range(0.0, total);
    let last = (0..net.roads().len()).rev().find(|&k| on(k)).expect("a network has roads");
    for (k, r) in net.roads().iter().enumerate() {
        if !on(k) {
            continue;
        }
        let len = r.line.length();
        if u <= len || k == last {
            let end = 8.0f64.min(0.5 * len);
            return (k, u.clamp(end, len - end));
        }
        u -= len;
    }
    unreachable!("a network has roads")
}

/// A place on the network and a direction of travel: `station` along road `road`'s centre
/// line, driven towards its end when `forward`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WalkPos {
    pub road: u32,
    pub forward: bool,
    pub station: f64,
}

impl WalkPos {
    /// The same place, facing the other way.
    pub fn reversed(self) -> Self {
        Self { forward: !self.forward, ..self }
    }
}

/// A leg of a random walk over the roads: its lane line, where it ends, and whether it ends
/// at a turning point (a dead end, or the map's edge ahead) rather than on its length.
pub struct Walk {
    pub line: Polyline,
    pub end: WalkPos,
    pub turn: bool,
}

/// Distance from the map's edges within which walks turn round, and outside which drivers
/// spawn (m).
pub const WALK_EDGE: f64 = 30.0;
/// Walks turn round this far before a dead end (m).
const DEAD_END_TURN: f64 = 10.0;
/// Spacing of the checks against the map's edges along a walk (m).
const EDGE_STEP: f64 = 2.0;

/// Whether `p` lies at least [`WALK_EDGE`] inside `world`'s edges.
pub(crate) fn clear_of_edges(world: &StaticWorld, p: DVec2) -> bool {
    let (lo, hi) = world.extent();
    p.cmpge(lo + WALK_EDGE).all() && p.cmple(hi - WALK_EDGE).all()
}

/// A random walk of `length` m over the roads marked in `roads` from `from`, turning at
/// random at every junction but never back. It stops early at a turning point:
/// [`DEAD_END_TURN`] before a dead end (a node without another marked road), or
/// [`WALK_EDGE`] + 5 m from the map's edges.
pub(crate) fn drive_walk(world: &StaticWorld, roads: &[bool], from: WalkPos, length: f64, rng: &mut SimRng) -> Walk {
    let net = world.roads();
    let mut pts: Vec<DVec3> = Vec::new();
    let mut push = |part: Vec<DVec3>| {
        for p in part {
            if pts.last().is_none_or(|q: &DVec3| q.truncate().distance(p.truncate()) > 0.5) {
                pts.push(p);
            }
        }
    };
    let (mut at, mut left, mut turn) = (from, length, false);
    loop {
        let r = &net.roads()[at.road as usize];
        let dir = if at.forward { 1.0 } else { -1.0 };
        let end = if at.forward { r.line.length() } else { 0.0 };
        let ahead = (end - at.station).abs();
        // The first point ahead near the map's edges, if any.
        let mut d = EDGE_STEP;
        let mut edge = None;
        while d <= ahead.min(left) + EDGE_STEP {
            let s = at.station + dir * d.min(ahead);
            if !clear_of_edges(world, r.line.point_at(s).truncate()) {
                edge = Some((d - 5.0).max(0.0));
                break;
            }
            d += EDGE_STEP;
        }
        if let Some(e) = edge.filter(|&e| e < left) {
            let stop = at.station + dir * e;
            push(r.line.slice(at.station, stop));
            at.station = stop;
            turn = true;
            break;
        }
        if ahead >= left {
            let stop = at.station + dir * left;
            push(r.line.slice(at.station, stop));
            at.station = stop;
            break;
        }
        let node = if at.forward { r.end } else { r.start };
        let options: Vec<WalkPos> = net
            .roads()
            .iter()
            .enumerate()
            .filter(|&(k, _)| roads[k])
            .flat_map(|(k, q)| {
                let k = k as u32;
                [
                    (q.start == node).then_some(WalkPos { road: k, forward: true, station: 0.0 }),
                    (q.end == node).then_some(WalkPos { road: k, forward: false, station: q.line.length() }),
                ]
            })
            .flatten()
            .filter(|w| !(w.road == at.road && w.forward != at.forward))
            .collect();
        if options.is_empty() {
            let stop = at.station + dir * (ahead - DEAD_END_TURN).max(0.0);
            push(r.line.slice(at.station, stop));
            at.station = stop;
            turn = true;
            break;
        }
        push(r.line.slice(at.station, end));
        left -= ahead;
        at = options[rng.below(options.len() as u64) as usize];
    }
    if pts.len() < 2 {
        // A walk of (nearly) zero length: a stub along the road.
        let r = &net.roads()[at.road as usize];
        let s = at.station.clamp(0.0, r.line.length() - 1.0);
        pts = if at.forward { r.line.slice(s, s + 1.0) } else { r.line.slice(s + 1.0, s) };
    }
    Walk { line: lane_line(net, &Polyline::new(pts)), end: at, turn }
}

/// A route (as a lane line) from `from` to a destination of `spec.route.destination` whose
/// length lies in `spec.distance`; the one closest to the range when none does.
pub(crate) fn plan_route(world: &StaticWorld, from: DVec2, spec: &GoalSpec, rng: &mut SimRng) -> Option<Polyline> {
    let net = world.roads();
    if net.is_empty() {
        return None;
    }
    let route = match spec.route.destination {
        RouteDestination::Yard => {
            let yards = net.nodes().iter().filter(|n| n.kind == NodeKind::Yard).map(|n| n.position.truncate());
            best_route(world, from, yards.collect(), spec.distance, rng)
        }
        RouteDestination::Road => None,
    };
    // Some maps have no farm (or none reachable): a road point instead, clear of the map's
    // edges (roads leave the map).
    route.or_else(|| {
        let (lo, hi) = world.extent();
        let inside = |p: DVec2| p.cmpge(lo + EDGE_CLEARANCE).all() && p.cmple(hi - EDGE_CLEARANCE).all();
        let points = (0..32)
            .map(|_| {
                let (k, s) = road_point(net, rng);
                net.roads()[k].line.point_at(s).truncate()
            })
            .filter(|&p| inside(p))
            .collect();
        best_route(world, from, points, spec.distance, rng)
    })
}

/// The lane line of the shortest road route from `from` to one of `targets` whose length is in
/// `[lo, hi]`, or else closest to it; routes nearer the map's edges than
/// [`ROUTE_EDGE_CLEARANCE`] are skipped.
fn best_route(
    world: &StaticWorld,
    from: DVec2,
    mut targets: Vec<DVec2>,
    [lo, hi]: [f64; 2],
    rng: &mut SimRng,
) -> Option<Polyline> {
    let net = world.roads();
    let (lo_xy, hi_xy) = world.extent();
    let clear = |p: DVec2| p.cmpge(lo_xy + ROUTE_EDGE_CLEARANCE).all() && p.cmple(hi_xy - ROUTE_EDGE_CLEARANCE).all();
    // Fisher–Yates, so that ties in the range go to a random destination.
    for i in (1..targets.len()).rev() {
        targets.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let mut best: Option<(f64, Polyline)> = None;
    for to in targets {
        let Some(route) = net.route(from, to, 50.0) else { continue };
        let len = route.line.length();
        let miss = (lo - len).max(len - hi).max(0.0);
        if len < 1.0
            || best.as_ref().is_some_and(|b| b.0 <= miss)
            || !route.line.points().iter().all(|p| clear(p.truncate()))
        {
            continue;
        }
        best = Some((miss, route.line));
        if miss == 0.0 {
            break;
        }
    }
    best.map(|(_, line)| lane_line(net, &line))
}

/// Goals every `step` m along `lane`, the last at its end; each `lift` above the terrain.
pub(crate) fn route_goals(world: &StaticWorld, lane: &Polyline, step: f64, lift: f64) -> Vec<Goal> {
    let len = lane.length();
    let count = ((len / step).ceil() as usize).max(1);
    (1..=count)
        .map(|k| {
            let s = (k as f64 * step).min(len);
            let p = lane.point_at(s);
            Goal { position: p.truncate().extend(world.terrain().height(p.x, p.y) + lift), yaw: lane.heading_at(s) }
        })
        .collect()
}

/// A spawn on a road: position (`lift` above the terrain), heading, and the route when the
/// goals follow one.
pub(crate) struct RoadSpawn {
    pub position: DVec3,
    pub yaw: f64,
    pub route: Option<Polyline>,
    /// Where on the network it starts and which way it faces (spawns on `through` roads).
    pub walk: Option<WalkPos>,
}

/// `count` spawns in random lanes, facing along the road (along the route with route goals),
/// kept `min_separation` from `placed` where possible (appended to it). With `only` (scripted
/// drivers), on the roads it marks and [`WALK_EDGE`] + 10 m inside the map's edges.
#[allow(clippy::too_many_arguments)]
pub(crate) fn road_spawns(
    world: &StaticWorld,
    spawn: &SpawnSpec,
    goals: Option<&GoalSpec>,
    count: usize,
    lift: f64,
    only: Option<&[bool]>,
    placed: &mut Vec<DVec3>,
    spawn_rng: &mut SimRng,
    goal_rng: &mut SimRng,
) -> Vec<RoadSpawn> {
    let net = world.roads();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best: Option<(f64, RoadSpawn)> = None;
        for _ in 0..64 {
            let (k, s) = road_point_on(net, only, spawn_rng);
            let road = &net.roads()[k];
            let centre = road.line.point_at(s).truncate();
            let (lo, hi) = world.extent();
            let inner = centre.cmpge(lo + WALK_EDGE + 10.0).all() && centre.cmple(hi - WALK_EDGE - 10.0).all();
            if only.is_some() && !inner {
                continue;
            }
            let (xy, yaw, route, walk) = match goals {
                Some(g) => {
                    let Some(lane) = plan_route(world, centre, g, goal_rng) else { continue };
                    (lane.point_at(0.0).truncate(), lane.heading_at(0.0), Some(lane), None)
                }
                None => {
                    let (xy, h) = lane_spawn(road, s, spawn_rng);
                    let forward = wrap_angle(h - road.line.heading_at(s)).abs() < 0.5 * std::f64::consts::PI;
                    let walk = only.map(|_| WalkPos { road: k as u32, forward, station: s });
                    (xy, h, None, walk)
                }
            };
            let position = xy.extend(world.terrain().height(xy.x, xy.y) + lift);
            let apart = placed.iter().map(|q| q.truncate().distance(xy)).fold(f64::INFINITY, f64::min);
            if best.as_ref().is_none_or(|b| apart > b.0) {
                best = Some((apart, RoadSpawn { position, yaw, route, walk }));
            }
            if apart >= spawn.min_separation {
                break;
            }
        }
        // No route from any draw (only with route goals): a spawn in a lane, and the spawn goal.
        let s = best.map(|b| b.1).unwrap_or_else(|| {
            let (k, s) = road_point(net, spawn_rng);
            let (xy, yaw) = lane_spawn(&net.roads()[k], s, spawn_rng);
            RoadSpawn { position: xy.extend(world.terrain().height(xy.x, xy.y) + lift), yaw, route: None, walk: None }
        });
        placed.push(s.position);
        out.push(s);
    }
    out
}

/// A point in the lane at station `s` of `road`, facing along it in a random direction.
fn lane_spawn(road: &Road, s: f64, rng: &mut SimRng) -> (DVec2, f64) {
    let mut h = road.line.heading_at(s);
    if rng.uniform() < 0.5 {
        h = wrap_angle(h + std::f64::consts::PI);
    }
    (road.line.point_at(s).truncate() + lane_offset(road) * right(h), h)
}

/// The line an agent follows in its travel direction: its route's lane, or else the lane of
/// the nearest road (within [`ROAD_REACH`]) in the direction closer to its heading.
pub struct Follow<'a> {
    line: &'a Polyline,
    reversed: bool,
    /// Lane offset to the right of `line` (only without a route).
    lane: f64,
    /// Station along the travel direction and the lateral offset from the lane (+ left).
    pub station: f64,
    pub offset: f64,
}

impl<'a> Follow<'a> {
    pub fn new(route: Option<&'a Polyline>, world: &'a StaticWorld, xy: DVec2, yaw: f64) -> Option<Self> {
        if let Some(line) = route {
            let pr = line.project(xy);
            return Some(Self { line, reversed: false, lane: 0.0, station: pr.station, offset: pr.offset });
        }
        let net = world.roads();
        let rp = net.nearest(xy, ROAD_REACH)?;
        let road = &net.roads()[rp.road as usize];
        let pr = rp.projection;
        let forward = wrap_angle(pr.heading - yaw).cos() >= 0.0;
        let lane = lane_offset(road);
        let (station, off) =
            if forward { (pr.station, pr.offset) } else { (road.line.length() - pr.station, -pr.offset) };
        Some(Self { line: &road.line, reversed: !forward, lane, station, offset: off + lane })
    }

    fn line_station(&self, ahead: f64) -> f64 {
        let len = self.line.length();
        let s = (self.station + ahead).clamp(0.0, len);
        if self.reversed { len - s } else { s }
    }

    /// Heading of the lane `ahead` m further on.
    pub fn heading(&self, ahead: f64) -> f64 {
        let h = self.line.heading_at(self.line_station(ahead));
        if self.reversed { wrap_angle(h + std::f64::consts::PI) } else { h }
    }

    /// Curvature of the lane `ahead` m further on (1/m, positive to the left).
    pub fn curvature(&self, ahead: f64) -> f64 {
        let k = self.line.curvature_at(self.line_station(ahead));
        if self.reversed { -k } else { k }
    }

    /// Lane centre `ahead` m further on.
    pub fn point(&self, ahead: f64) -> DVec2 {
        let p = self.line.point_at(self.line_station(ahead)).truncate();
        p + self.lane * right(self.heading(ahead))
    }
}

/// The `road` state column: lateral offset from the lane followed (m, + left), lane heading −
/// heading (rad), and distance from `xy` to the nearest road's surface (m; 0 on a road).
/// Without a lane to follow the first two are 0, without a road within [`ROAD_REACH`] the
/// distance is `ROAD_REACH`.
pub fn road_state(route: Option<&Polyline>, world: &StaticWorld, xy: DVec2, yaw: f64) -> [f64; 3] {
    let net = world.roads();
    let off_road = match net.nearest(xy, ROAD_REACH) {
        Some(rp) => (rp.projection.distance - 0.5 * net.roads()[rp.road as usize].width).clamp(0.0, ROAD_REACH),
        None => ROAD_REACH,
    };
    match Follow::new(route, world, xy, yaw) {
        Some(f) => [f.offset, wrap_angle(f.heading(0.0) - yaw), off_road],
        None => [0.0, 0.0, off_road],
    }
}
