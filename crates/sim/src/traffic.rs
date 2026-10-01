//! Traffic signals and lane tracking on urban maps (M8a step 4).
//!
//! **Signals**: the map's fixed-time controllers ([`autonomousim_world::signals`]) run with a
//! per-episode offset each, drawn uniformly over its cycle from the episode's `signals`
//! stream. Controller `k` shows at world time `t` what it shows at `t + offset[k]` into its
//! cycle, so the state is a pure function of time and the draw (snapshots, replays and
//! batches stay deterministic).
//!
//! **Lane tracking** ([`RoadTrack`]): after every policy step (and at the reset) each ground
//! vehicle on a map with road sections is matched to the lane it drives in, which raises the
//! non-terminal events
//! - `OFF_ROAD`: on a sidewalk, a median or off the roads (lots included) faster than
//!   `events.ground.off_road_speed`;
//! - `WRONG_WAY`: inside a lane whose direction differs by more than 120° from the motion,
//!   faster than that speed, with no lane the other way holding the point;
//! - `RED_LIGHT`: crossing the stop line (the end of the lane followed) of a signalized
//!   junction while every movement from the lane is red. When only some are, the crossing
//!   is kept until the next lane shows which movement was taken, and counts if that one was
//!   red when the line was crossed (given up after 20 s). Amber is not red; all-red is.
//!
//! The `signal` and `lanes` observation terms read the lane followed.

use crate::events::Events;
use autonomousim_core::rng::Seed;
use autonomousim_world::lanes::{Area, LaneGraph, Turn};
use autonomousim_world::{Lane, Light, Polyline, RoadNetwork, StaticWorld};
use glam::DVec2;

/// Radius (m) within which the lane followed is searched.
pub const TRACK_RADIUS: f64 = 6.0;

/// A crossing on a partly red signal is given up after this long without a next lane (s).
pub const PENDING_TIMEOUT: f64 = 20.0;

/// Wrong way when the cosine between the lane's direction and the motion is below this.
const WRONG_WAY_COS: f64 = -0.5;

/// Hard limit on the connectors and lanes walked ahead.
const MAX_WALK: usize = 16;

/// The signals of the current map with the episode's offsets.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Signals {
    offsets: Vec<f64>,
}

impl Signals {
    /// Offsets for the controllers of `world` from `seed` (none on maps without road
    /// sections).
    pub fn new(world: &StaticWorld, seed: Seed) -> Self {
        let net = world.roads();
        if !net.has_sections() {
            return Self::default();
        }
        let mut rng = seed.rng();
        Self { offsets: net.lanes().controllers().iter().map(|c| rng.uniform() * c.cycle()).collect() }
    }

    /// Given offsets (replays; controllers without one run with offset 0).
    pub fn with_offsets(offsets: Vec<f64>) -> Self {
        Self { offsets }
    }

    /// Offset (s) of each controller.
    pub fn offsets(&self) -> &[f64] {
        &self.offsets
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// The phase controller `k` runs at time `t` and its light (red during the all-red).
    pub fn state(&self, lanes: &LaneGraph, k: usize, t: f64) -> (usize, Light) {
        let c = &lanes.controllers()[k];
        let local = t + self.offsets.get(k).copied().unwrap_or(0.0);
        let (p, _) = c.active(local);
        (p, c.light(p, local))
    }

    /// Whether pedestrians may start over crossing `k` at time `t`, and how long that stays
    /// so (s); None for unsignalized crossings (see [`LaneGraph::crossing_walk`]).
    pub fn crossing_walk(&self, lanes: &LaneGraph, k: u32, t: f64) -> Option<(bool, f64)> {
        let (ctl, _) = lanes.crossings()[k as usize].signal?;
        lanes.crossing_walk(k, t + self.offsets.get(ctl as usize).copied().unwrap_or(0.0))
    }

    /// The light for connector `c` at time `t` (green when it is not signalled).
    pub fn light(&self, lanes: &LaneGraph, c: u32, t: f64) -> Light {
        self.light_left(lanes, c, t).0
    }

    /// The light for connector `c` at time `t` and how long it stays so (s; infinite when
    /// it is not signalled).
    pub fn light_left(&self, lanes: &LaneGraph, c: u32, t: f64) -> (Light, f64) {
        match lanes.connector_signal(c) {
            Some((k, p)) => {
                let o = self.offsets.get(k as usize).copied().unwrap_or(0.0);
                lanes.controllers()[k as usize].light_left(p as usize, t + o)
            }
            None => (Light::Green, f64::INFINITY),
        }
    }
}

/// Where a ground vehicle is on the roads, updated after every policy step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoadTrack {
    /// The lane followed: the last one the vehicle was inside, heading along it.
    pub lane: Option<u32>,
    /// Station along it (m; beyond its length past the end).
    pub station: f64,
    /// Signed distance past its end (m; negative before the stop line).
    pub past_end: f64,
    /// Offset from its centre line (m, positive to the left).
    pub offset: f64,
    /// Area class under the vehicle.
    pub area: Option<Area>,
    /// A stop line crossed with some movements red: (lane, time).
    pending: Option<(u32, f64)>,
}

/// Distance past the end of `lane` along its final direction, and to the left of it (m).
fn beyond_end(lane: &Lane, p: DVec2) -> (f64, f64) {
    let end = lane.line.points().last().expect("points").truncate();
    let d = DVec2::from_angle(lane.line.heading_at(lane.line.length()));
    let r = p - end;
    (r.dot(d), d.perp().dot(r))
}

impl RoadTrack {
    /// Start following at `p` with heading `heading` (no events).
    pub(crate) fn reset(&mut self, net: &RoadNetwork, p: DVec2, heading: f64) {
        *self = Self::default();
        if net.has_sections() {
            self.update(net, &Signals::default(), p, heading, DVec2::ZERO, 0.0, f64::INFINITY);
        }
    }

    /// Follow the vehicle at `p` with heading `heading` and horizontal velocity `v` at time
    /// `t`; returns the events raised (see the module docs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update(
        &mut self,
        net: &RoadNetwork,
        signals: &Signals,
        p: DVec2,
        heading: f64,
        v: DVec2,
        t: f64,
        min_speed: f64,
    ) -> Events {
        let mut e = Events::NONE;
        let lanes = net.lanes();
        let speed = v.length();
        let area = net.area(p);
        self.area = Some(area);
        let moving = speed > min_speed;
        if moving && matches!(area, Area::Sidewalk | Area::Median | Area::Off) {
            e |= Events::OFF_ROAD;
        }
        let inside = |l: u32, s: f64, off: f64| {
            let lane = &lanes.lanes()[l as usize];
            off.abs() <= 0.5 * lane.width && s > 1e-6 && s < lane.line.length() - 1e-6
        };
        if moving {
            let motion = v.y.atan2(v.x);
            let along = |l: u32, s: f64| {
                DVec2::from_angle(lanes.lanes()[l as usize].line.heading_at(s)).dot(DVec2::from_angle(motion))
            };
            if let Some((l, s, off)) = lanes.nearest_lane(p, TRACK_RADIUS, None)
                && inside(l, s, off)
                && along(l, s) < WRONG_WAY_COS
            {
                // Unless a lane the other way holds the point too (shared lanes).
                let other = lanes
                    .nearest_lane(p, TRACK_RADIUS, Some(motion))
                    .is_some_and(|(m, s, off)| inside(m, s, off) && along(m, s) > -WRONG_WAY_COS);
                if !other {
                    e |= Events::WRONG_WAY;
                }
            }
        }
        // Crossing the stop line at the end of the lane followed.
        if let Some(l) = self.lane {
            let lane = &lanes.lanes()[l as usize];
            let (d, lateral) = beyond_end(lane, p);
            if self.past_end < 0.0
                && d >= 0.0
                && lateral.abs() <= 0.5 * lane.width + 0.5
                && lanes.junction_controller(lane.to_node).is_some()
                && !lane.successors.is_empty()
            {
                let red = lane.successors.iter().filter(|&&c| signals.light(lanes, c, t) == Light::Red).count();
                if red == lane.successors.len() {
                    e |= Events::RED_LIGHT;
                } else if red > 0 {
                    self.pending = Some((l, t));
                }
            }
            self.past_end = d;
            self.station = lane.line.length() + d;
        }
        // The lane now followed.
        if let Some((m, s, off)) = lanes.nearest_lane(p, TRACK_RADIUS, Some(heading))
            && inside(m, s, off)
        {
            if self.lane != Some(m) {
                if let Some((pl, t0)) = self.pending.take() {
                    let from = &lanes.lanes()[pl as usize];
                    if let Some(&c) = from.successors.iter().find(|&&c| lanes.connectors()[c as usize].to == m)
                        && signals.light(lanes, c, t0) == Light::Red
                    {
                        e |= Events::RED_LIGHT;
                    }
                }
                self.lane = Some(m);
                self.past_end = beyond_end(&lanes.lanes()[m as usize], p).0;
            }
            self.station = s;
            self.offset = off;
        } else if let Some(l) = self.lane {
            self.offset = lanes.lanes()[l as usize].line.project(p).offset;
        }
        if self.pending.is_some_and(|(_, t0)| t - t0 > PENDING_TIMEOUT) {
            self.pending = None;
        }
        e
    }
}

/// The connector a vehicle leaving `lane` takes: the one ending nearest to `route` if given,
/// else the straight one (or the first).
pub fn next_connector(lanes: &LaneGraph, lane: u32, route: Option<&Polyline>) -> Option<u32> {
    let succ = &lanes.lanes()[lane as usize].successors;
    if let Some(r) = route.filter(|_| succ.len() > 1) {
        let miss = |c: u32| {
            let line = &lanes.connectors()[c as usize].line;
            r.project(line.point_at(0.75 * line.length()).truncate()).distance
        };
        return succ.iter().copied().min_by(|&a, &b| miss(a).total_cmp(&miss(b)).then(a.cmp(&b)));
    }
    succ.iter().copied().find(|&c| lanes.connectors()[c as usize].turn == Turn::Straight).or(succ.first().copied())
}

/// The point `ahead` metres along the lanes from station `s` of `lane`, through the
/// connectors [`next_connector`] picks (straight on past the last one).
pub fn walk(lanes: &LaneGraph, lane: u32, s: f64, ahead: f64, route: Option<&Polyline>) -> DVec2 {
    let mut l = lane;
    let mut rem = s + ahead;
    for _ in 0..MAX_WALK {
        let line = &lanes.lanes()[l as usize].line;
        if rem <= line.length() {
            return line.point_at(rem.max(0.0)).truncate();
        }
        rem -= line.length();
        let Some(c) = next_connector(lanes, l, route) else { return extend(line, rem) };
        let cl = &lanes.connectors()[c as usize].line;
        if rem <= cl.length() {
            return cl.point_at(rem).truncate();
        }
        rem -= cl.length();
        l = lanes.connectors()[c as usize].to;
    }
    extend(&lanes.lanes()[l as usize].line, 0.0)
}

/// The next movement from `lane`'s road in its direction: the connector (among those leaving
/// its lane and the lanes beside it) ending nearest to `route` if given (fewest lane changes
/// among equals), else the one [`next_connector`] picks from `lane`; with the lane changes to
/// the lane it leaves from (positive to the left).
pub fn next_movement(lanes: &LaneGraph, lane: u32, route: Option<&Polyline>) -> Option<(u32, i32)> {
    let Some(r) = route else { return next_connector(lanes, lane, None).map(|c| (c, 0)) };
    let side = |step: fn(&Lane) -> Option<u32>, sign: i32| {
        std::iter::successors(step(&lanes.lanes()[lane as usize]).map(|l| (l, sign)), move |&(l, k)| {
            step(&lanes.lanes()[l as usize]).map(|m| (m, k + sign))
        })
        .take(MAX_SIDE)
    };
    let miss = |c: u32| {
        let line = &lanes.connectors()[c as usize].line;
        r.project(line.point_at(0.75 * line.length()).truncate()).distance
    };
    std::iter::once((lane, 0))
        .chain(side(|l| l.left, 1))
        .chain(side(|l| l.right, -1))
        .flat_map(|(l, k)| lanes.lanes()[l as usize].successors.iter().map(move |&c| (c, k)))
        .min_by(|a, b| miss(a.0).total_cmp(&miss(b.0)).then(a.1.abs().cmp(&b.1.abs())).then(a.0.cmp(&b.0)))
}

/// Most lanes beside a lane looked at.
const MAX_SIDE: usize = 8;

/// The point `d` metres straight on past the end of `line`.
fn extend(line: &Polyline, d: f64) -> DVec2 {
    let end = line.points().last().expect("points").truncate();
    end + DVec2::from_angle(line.heading_at(line.length())) * d
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_procgen::urban::{self, UrbanPreset};
    use autonomousim_world::lanes::JunctionKind;

    /// Positions every 0.5 m along `lines` (lane, connector, lane …).
    fn path(lines: &[&Polyline], from: f64, to: f64) -> Vec<(DVec2, f64)> {
        let mut out = Vec::new();
        let mut base = 0.0;
        for line in lines {
            let len = line.length();
            let mut s = (from - base).max(0.0);
            while s < len && base + s <= to {
                out.push((line.point_at(s).truncate(), line.heading_at(s)));
                s += 0.5;
            }
            base += len;
        }
        out
    }

    /// Drive along `pts` at 10 m/s from time `t0`; the events raised.
    fn drive(net: &RoadNetwork, signals: &Signals, pts: &[(DVec2, f64)], t0: f64, speed: f64, back: bool) -> Events {
        let mut track = RoadTrack::default();
        let (p, h) = pts[0];
        let h0 = if back { h + std::f64::consts::PI } else { h };
        track.reset(net, p, h0);
        let mut e = Events::NONE;
        for (k, &(p, h)) in pts.iter().enumerate() {
            let h = if back { h + std::f64::consts::PI } else { h };
            let t = t0 + k as f64 * 0.5 / speed;
            e |= track.update(net, signals, p, h, DVec2::from_angle(h) * speed, t, 2.0);
        }
        e
    }

    #[test]
    fn events_of_drives_through_signals_against_lanes_and_on_sidewalks() {
        let (w, _) = urban::generate(&UrbanPreset::Training.config(), 2).unwrap();
        let net = w.roads();
        let g = net.lanes();
        let signals = Signals::with_offsets(vec![0.0; g.controllers().len()]);
        let (mut all_red, mut mixed, mut legal) = (0, 0, 0);
        for ctl in g.controllers() {
            let j = &g.junctions()[ctl.junction as usize];
            assert_eq!(j.kind, JunctionKind::Signal);
            for a in &j.approaches {
                for &l in &a.lanes {
                    let lane = &g.lanes()[l as usize];
                    if lane.line.length() < 25.0 {
                        continue;
                    }
                    for &c in &lane.successors {
                        let conn = &g.connectors()[c as usize];
                        let next = &g.lanes()[conn.to as usize];
                        if next.line.length() < 15.0 {
                            continue;
                        }
                        let len = lane.line.length();
                        let pts =
                            path(&[&lane.line, &conn.line, &next.line], len - 20.0, len + conn.line.length() + 10.0);
                        // The stop line is 20 m (2 s) in; cross it every 1.5 s of the cycle.
                        for q in 0..(ctl.cycle() / 1.5) as usize {
                            let tc = q as f64 * 1.5;
                            let lights: Vec<Light> = lane.successors.iter().map(|&s| signals.light(g, s, tc)).collect();
                            let mine = signals.light(g, c, tc);
                            // Clear of changes around the crossing.
                            if [tc - 0.1, tc + 0.1].iter().any(|&t| signals.light(g, c, t) != mine) {
                                continue;
                            }
                            let e = drive(net, &signals, &pts, tc - 2.0 + 0.025, 10.0, false);
                            assert!(!e.intersects(Events::WRONG_WAY | Events::OFF_ROAD), "lane {l} via {c}: {e:?}");
                            let red = mine == Light::Red;
                            assert_eq!(e.contains(Events::RED_LIGHT), red, "lane {l} via {c} at {tc}: {lights:?}");
                            if !red {
                                legal += 1;
                            } else if lights.iter().all(|&x| x == Light::Red) {
                                all_red += 1;
                            } else {
                                mixed += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(legal > 0 && all_red > 0 && mixed > 0, "{legal} {all_red} {mixed}");

        // Against a lane: wrong way (not the other way round, nor when slow).
        let l = (0..g.lanes().len()).find(|&l| g.lanes()[l].line.length() > 40.0).unwrap();
        let line = &g.lanes()[l].line;
        let mut pts = path(&[line], 5.0, 35.0);
        assert!(drive(net, &signals, &pts, 0.0, 10.0, false).is_empty());
        pts.reverse();
        assert!(drive(net, &signals, &pts, 0.0, 10.0, true).contains(Events::WRONG_WAY));
        assert!(drive(net, &signals, &pts, 0.0, 1.0, true).is_empty());

        // On a sidewalk.
        let (i, road) = net
            .roads()
            .iter()
            .enumerate()
            .find(|(i, r)| net.section(*i).sidewalk[0] > 1.0 && r.line.length() > 60.0)
            .unwrap();
        let s = net.section(i);
        let side = 0.5 * road.width + 0.5 * s.sidewalk[0];
        let mut walk = Vec::new();
        for k in 0..40 {
            let st = 10.0 + 0.5 * f64::from(k);
            let h = road.line.heading_at(st);
            // The sidewalk on the right of the road's direction.
            walk.push((road.line.point_at(st).truncate() - DVec2::from_angle(h).perp() * side, h));
        }
        if net.area(walk[0].0) != Area::Sidewalk {
            for q in &mut walk {
                q.0 += DVec2::from_angle(q.1).perp() * 2.0 * side;
            }
        }
        assert_eq!(net.area(walk[10].0), Area::Sidewalk);
        assert!(drive(net, &signals, &walk, 0.0, 5.0, false).contains(Events::OFF_ROAD));
        assert!(!drive(net, &signals, &walk, 0.0, 1.0, false).contains(Events::OFF_ROAD));
    }

    /// The next movement along a route that turns from the lane beside: one lane change over.
    #[test]
    fn next_movement_counts_the_lane_changes() {
        let (w, _) = urban::generate(&UrbanPreset::Training.config(), 2).unwrap();
        let g = w.roads().lanes();
        let turns =
            |l: u32, t: Turn| g.lanes()[l as usize].successors.iter().any(|&c| g.connectors()[c as usize].turn == t);
        let mut checked = 0;
        for (l, lane) in g.lanes().iter().enumerate() {
            let l = l as u32;
            let Some(m) = lane.left else { continue };
            if turns(l, Turn::Left) || !turns(m, Turn::Left) {
                continue;
            }
            let &c = g.lanes()[m as usize]
                .successors
                .iter()
                .find(|&&c| g.connectors()[c as usize].turn == Turn::Left)
                .expect("a left turn");
            let route = &g.connectors()[c as usize].line;
            let (got, changes) = next_movement(g, l, Some(route)).expect("a movement");
            assert_eq!((got, changes), (c, 1), "lane {l}");
            assert_eq!(next_movement(g, m, Some(route)), Some((c, 0)));
            // Without a route: straight on from its own lane.
            assert_eq!(next_movement(g, l, None).map(|x| x.1), Some(0));
            checked += 1;
        }
        assert!(checked >= 3, "{checked} lanes");
    }
}
