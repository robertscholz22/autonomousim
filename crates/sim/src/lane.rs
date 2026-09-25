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

/// A uniformly random point on the network (by length), away from the road ends: the road
/// and the station.
fn road_point(net: &RoadNetwork, rng: &mut SimRng) -> (usize, f64) {
    let total: f64 = net.roads().iter().map(|r| r.line.length()).sum();
    let mut u = rng.range(0.0, total);
    for (k, r) in net.roads().iter().enumerate() {
        let len = r.line.length();
        if u <= len || k + 1 == net.roads().len() {
            let end = 8.0f64.min(0.5 * len);
            return (k, u.clamp(end, len - end));
        }
        u -= len;
    }
    unreachable!("a network has roads")
}

/// A route (as a lane line) from `from` to a destination of `spec.route.destination` whose
/// length lies in `spec.distance`; the one closest to the range when none does.
pub(crate) fn plan_route(world: &StaticWorld, from: DVec2, spec: &GoalSpec, rng: &mut SimRng) -> Option<Polyline> {
    let net = world.roads();
    if net.is_empty() {
        return None;
    }
    let mut targets: Vec<DVec2> = match spec.route.destination {
        RouteDestination::Yard => {
            net.nodes().iter().filter(|n| n.kind == NodeKind::Yard).map(|n| n.position.truncate()).collect()
        }
        RouteDestination::Road => (0..32)
            .map(|_| {
                let (k, s) = road_point(net, rng);
                net.roads()[k].line.point_at(s).truncate()
            })
            .collect(),
    };
    // Fisher–Yates, so that ties in the range go to a random destination.
    for i in (1..targets.len()).rev() {
        targets.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let [lo, hi] = spec.distance;
    let mut best: Option<(f64, Polyline)> = None;
    for to in targets {
        let Some(route) = net.route(from, to, 50.0) else { continue };
        let len = route.line.length();
        let miss = (lo - len).max(len - hi).max(0.0);
        if len < 1.0 || best.as_ref().is_some_and(|b| b.0 <= miss) {
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
}

/// `count` spawns in random lanes, facing along the road (along the route with route goals),
/// kept `min_separation` from `placed` where possible (appended to it).
#[allow(clippy::too_many_arguments)]
pub(crate) fn road_spawns(
    world: &StaticWorld,
    spawn: &SpawnSpec,
    goals: Option<&GoalSpec>,
    count: usize,
    lift: f64,
    placed: &mut Vec<DVec3>,
    spawn_rng: &mut SimRng,
    goal_rng: &mut SimRng,
) -> Vec<RoadSpawn> {
    let net = world.roads();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best: Option<(f64, RoadSpawn)> = None;
        for _ in 0..64 {
            let (k, s) = road_point(net, spawn_rng);
            let road = &net.roads()[k];
            let centre = road.line.point_at(s).truncate();
            let (xy, yaw, route) = match goals {
                Some(g) => {
                    let Some(lane) = plan_route(world, centre, g, goal_rng) else { continue };
                    (lane.point_at(0.0).truncate(), lane.heading_at(0.0), Some(lane))
                }
                None => {
                    let mut h = road.line.heading_at(s);
                    if spawn_rng.uniform() < 0.5 {
                        h = wrap_angle(h + std::f64::consts::PI);
                    }
                    (centre + lane_offset(road) * right(h), h, None)
                }
            };
            let position = xy.extend(world.terrain().height(xy.x, xy.y) + lift);
            let apart = placed.iter().map(|q| q.truncate().distance(xy)).fold(f64::INFINITY, f64::min);
            if best.as_ref().is_none_or(|b| apart > b.0) {
                best = Some((apart, RoadSpawn { position, yaw, route }));
            }
            if apart >= spawn.min_separation {
                break;
            }
        }
        let (_, s) = best.expect("a network with roads yields spawns");
        placed.push(s.position);
        out.push(s);
    }
    out
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
