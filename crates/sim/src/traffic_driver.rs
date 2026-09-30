//! The `traffic` driver (M8b): NPC vehicles driving the lane graph
//! ([`autonomousim_world::lanes`]) like human drivers.
//!
//! - **Following**: the Intelligent Driver Model (Treiber, Hennecke & Helbing 2000),
//!   `a = a_max·[1 − (v/v₀)⁴ − (s*/s)²]` with `s* = s₀ + max(0, vT + v·Δv/(2√(a_max·b)))`,
//!   behind the nearest vehicle ahead along the lanes and connectors to be driven. The
//!   desired speed `v₀` is a factor (drawn per driver) of the speed limit, lowered ahead of
//!   slower lanes, connectors and bends (so that it can brake to them at `b`).
//! - **Lane changes**: MOBIL (Kesting, Treiber & Helbing 2007): into a neighbouring lane when
//!   `ã_c − a_c + p·(ã_n − a_n + ã_o − a_o)` exceeds the threshold (plus the keep-right bias
//!   when changing left, less it when changing right), and the new follower need not brake
//!   harder than `safe_decel`. Lane changes the route needs (into the lane its next connector
//!   leaves from) are mandatory: only safety counts, changes away are not made, and the
//!   vehicle waits at the lane's end until it can change. A change follows a smooth (quintic)
//!   lateral path over `change_time` at the current speed.
//! - **Routes**: at the end of each lane the driver takes a connector drawn at random among
//!   those leaving its road's lanes in its direction, weighted by the class rank of the road
//!   they lead into (U-turns only at dead ends).
//! - **Steering**: pure pursuit on the lane line (blended across during a change).
//!
//! Every policy step the world first has each driver find its place on the graph, then builds
//! a [`LaneIndex`] of every agent on the lanes (drivers by their place, other agents near the
//! ground by the lane nearest to them), sorted by station, and then has each driver decide
//! from that snapshot (in agent order, so the result does not depend on threads). Leaders and
//! followers are found in O(log N) per lane.
//!
//! Junction rules (signals, give way, conflicts) come in M8b step 3: for now drivers enter
//! junctions without looking at crossing traffic.

use crate::driver::DriverGeometry;
use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::{wrap_angle, yaw};
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::terrain::Terrain;
use autonomousim_world::lanes::{LaneGraph, Turn, class_rank};
use autonomousim_world::{Polyline, StaticWorld};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

/// Settings of the `traffic` driver; ranges are drawn uniformly per driver and episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrafficDriverSpec {
    /// Desired speed as a factor of the speed limit.
    pub speed_factor: [f64; 2],
    /// Time headway `T` (s).
    pub headway: [f64; 2],
    /// Gap kept at a standstill `s₀` (m).
    pub min_gap: f64,
    /// Largest acceleration `a` (m/s²).
    pub accel: [f64; 2],
    /// Comfortable deceleration `b` (m/s²).
    pub decel: [f64; 2],
    /// MOBIL politeness `p`.
    pub politeness: [f64; 2],
    /// MOBIL threshold (m/s²).
    pub threshold: f64,
    /// Keep-right bias (m/s²): added to the threshold for changes to the left, taken off it
    /// for changes to the right.
    pub keep_right: f64,
    /// Hardest braking a lane change may impose on the new follower (m/s²).
    pub safe_decel: f64,
    /// Duration of a lane change (s).
    pub change_time: [f64; 2],
    /// Largest lateral acceleration in bends (m/s²).
    pub lateral_accel: f64,
    /// Pure-pursuit look-ahead: `lookahead[0] + lookahead[1]·speed` (m, s).
    pub lookahead: [f64; 2],
}

impl Default for TrafficDriverSpec {
    fn default() -> Self {
        Self {
            speed_factor: [0.9, 1.1],
            headway: [1.0, 2.0],
            min_gap: 2.0,
            accel: [1.0, 2.0],
            decel: [2.0, 3.0],
            politeness: [0.0, 0.5],
            threshold: 0.1,
            keep_right: 0.3,
            safe_decel: 4.0,
            change_time: [3.0, 5.0],
            lateral_accel: 2.0,
            lookahead: [3.0, 0.5],
        }
    }
}

impl TrafficDriverSpec {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let range = |r: [f64; 2], lo: f64| r[0].is_finite() && r[1].is_finite() && lo <= r[0] && r[0] <= r[1];
        let ok = range(self.speed_factor, 0.0)
            && self.speed_factor[1] > 0.0
            && range(self.headway, 0.0)
            && range(self.accel, 0.0)
            && self.accel[0] > 0.0
            && range(self.decel, 0.0)
            && self.decel[0] > 0.0
            && range(self.politeness, 0.0)
            && range(self.change_time, 0.5)
            && self.min_gap >= 0.0
            && self.threshold >= 0.0
            && self.keep_right >= 0.0
            && self.safe_decel > 0.0
            && self.lateral_accel > 0.0
            && self.lookahead[0] > 0.0
            && self.lookahead[1] >= 0.0;
        if ok { Ok(()) } else { Err(format!("invalid traffic driver {self:?}")) }
    }
}

/// The car-following parameters of one driver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Idm {
    /// Desired speed (m/s).
    pub v0: f64,
    /// Time headway (s).
    pub headway: f64,
    /// Standstill gap (m).
    pub min_gap: f64,
    pub accel: f64,
    pub decel: f64,
}

impl Idm {
    /// Parameters assumed for agents that are not traffic drivers (learning agents, other
    /// drivers) when judging lane changes in front of them.
    pub const OTHER: Idm = Idm { v0: 14.0, headway: 1.5, min_gap: 2.0, accel: 1.5, decel: 2.5 };

    /// Acceleration at speed `v` behind a leader `gap` m ahead (bumper to bumper) closing at
    /// `dv` (m/s, positive when faster than it), or on a free road.
    pub fn accel(&self, v: f64, leader: Option<(f64, f64)>) -> f64 {
        let v = v.max(0.0);
        let free = 1.0 - (v / self.v0.max(0.1)).powi(4);
        let interaction = leader.map_or(0.0, |(gap, dv)| {
            let s_star = self.min_gap + (v * self.headway + v * dv / (2.0 * (self.accel * self.decel).sqrt())).max(0.0);
            (s_star / gap.max(0.1)).powi(2)
        });
        self.accel * (free - interaction)
    }

    /// Bumper-to-bumper gap (m) in steady traffic at speed `v` (the IDM equilibrium).
    pub fn equilibrium_gap(&self, v: f64) -> f64 {
        (self.min_gap + v * self.headway) / (1.0 - (v / self.v0).powi(4)).max(1e-9).sqrt()
    }
}

/// A lane or a connector of the lane graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Elem {
    Lane(u32),
    Connector(u32),
}

impl Elem {
    pub fn line(self, g: &LaneGraph) -> &Polyline {
        match self {
            Elem::Lane(l) => &g.lanes()[l as usize].line,
            Elem::Connector(c) => &g.connectors()[c as usize].line,
        }
    }

    fn speed_limit(self, g: &LaneGraph) -> f64 {
        match self {
            Elem::Lane(l) => g.lanes()[l as usize].speed,
            Elem::Connector(c) => g.connectors()[c as usize].speed,
        }
    }
}

/// A place on the lane graph: an element and a station along it (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub elem: Elem,
    pub station: f64,
}

/// A lane change in progress, into the lane of the driver's place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaneChange {
    /// The lane left.
    pub from: u32,
    /// Stations (along the new lane) where it started and its length (m).
    pub start: f64,
    pub length: f64,
}

/// An agent on the lane graph as the drivers see it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Occupant {
    pub agent: u32,
    /// Station of its reference point (m).
    pub station: f64,
    /// How far it reaches ahead of and behind that point (m, both ≥ 0).
    pub front: f64,
    pub rear: f64,
    /// Speed along the lane (m/s).
    pub speed: f64,
    pub idm: Idm,
}

/// The agents on each lane and connector, sorted by station (then agent), rebuilt each
/// policy step.
#[derive(Clone, Debug, Default)]
pub struct LaneIndex {
    lanes: Vec<Vec<Occupant>>,
    connectors: Vec<Vec<Occupant>>,
}

/// Search range for leaders and followers (m).
const SEARCH: f64 = 200.0;
/// Other agents count as on a lane within this much beyond its half width (m) and this
/// height above the ground (m).
const LANE_MARGIN: f64 = 0.5;
const GROUND_HEIGHT: f64 = 3.0;
/// Time constant with which the speed command realises the IDM acceleration (s): that of
/// the kinematic model's speed response.
const SPEED_LAG: f64 = 0.5;
/// Drivers lost further than this from their element (m) find their place anew.
const LOST: f64 = 3.0;
/// Radius within which a lost driver looks for a lane (m).
const FIND: f64 = 6.0;
/// Spacing of the bend preview (m).
const PREVIEW_STEP: f64 = 2.5;
/// Shortest lane change (m), and no discretionary change within this of a lane's end (m).
const MIN_CHANGE: f64 = 10.0;
const CHANGE_CLEAR: f64 = 60.0;
/// Length of lane needed per lane change for a movement (m).
const CHANGE_ROOM: f64 = 40.0;
/// No discretionary change away from the lane the route needs within this of its end (m).
const KEEP_ROUTE: f64 = 150.0;
/// Time after a change before the next one (s).
const COOLDOWN: f64 = 3.0;
/// Where a mandatory change is still pending, the vehicle stops this far before the lane's
/// end (m), and takes a connector of its own lane after waiting there this long (s).
const END_HOLD: f64 = 1.0;
const STUCK: f64 = 10.0;

impl LaneIndex {
    pub fn new(g: &LaneGraph) -> Self {
        Self { lanes: vec![Vec::new(); g.lanes().len()], connectors: vec![Vec::new(); g.connectors().len()] }
    }

    /// Whether it is sized for `g`.
    pub fn fits(&self, g: &LaneGraph) -> bool {
        self.lanes.len() == g.lanes().len() && self.connectors.len() == g.connectors().len()
    }

    /// Empty it for a new step.
    pub fn clear(&mut self) {
        self.lanes.iter_mut().chain(&mut self.connectors).for_each(Vec::clear);
    }

    fn list(&self, e: Elem) -> &[Occupant] {
        match e {
            Elem::Lane(l) => &self.lanes[l as usize],
            Elem::Connector(c) => &self.connectors[c as usize],
        }
    }

    pub fn insert(&mut self, e: Elem, o: Occupant) {
        match e {
            Elem::Lane(l) => self.lanes[l as usize].push(o),
            Elem::Connector(c) => self.connectors[c as usize].push(o),
        }
    }

    /// Add an agent that is not a traffic driver at `pos`, heading `heading`, moving at
    /// `vel`, reaching `radius` around its centre: to the lane under it, if any.
    pub fn insert_other(&mut self, world: &StaticWorld, agent: u32, pos: DVec3, heading: f64, vel: DVec3, radius: f64) {
        let g = world.roads().lanes();
        if g.is_empty() || pos.z - world.terrain().height(pos.x, pos.y) > GROUND_HEIGHT {
            return;
        }
        let Some((lane, station, offset)) = g.nearest_lane(pos.truncate(), FIND, Some(heading)) else { return };
        let l = &g.lanes()[lane as usize];
        if offset.abs() > 0.5 * l.width + LANE_MARGIN {
            return;
        }
        let speed = vel.truncate().dot(DVec2::from_angle(l.line.heading_at(station)));
        let o = Occupant { agent, station, front: radius, rear: radius, speed, idm: Idm::OTHER };
        self.lanes[lane as usize].push(o);
    }

    /// Sort every list by station (then agent).
    pub fn sort(&mut self) {
        for l in self.lanes.iter_mut().chain(&mut self.connectors) {
            l.sort_by(|a, b| a.station.total_cmp(&b.station).then(a.agent.cmp(&b.agent)));
        }
    }

    /// The first occupant other than `me` ahead of station `from` along `chain` (elements
    /// with the distance from the searcher to their start): (distance between reference
    /// points, occupant).
    fn ahead(&self, chain: &[(Elem, f64)], from: f64, me: u32) -> Option<(f64, Occupant)> {
        for (k, &(e, base)) in chain.iter().enumerate() {
            if base > SEARCH {
                break;
            }
            let list = self.list(e);
            let start = if k == 0 { from } else { f64::NEG_INFINITY };
            let i = list.partition_point(|o| o.station <= start);
            if let Some(o) = list[i..].iter().find(|o| o.agent != me) {
                return Some((base + o.station, *o));
            }
        }
        None
    }

    /// The nearest occupant other than `me` behind station `from` of lane `lane`, also on the
    /// connectors into it and the lanes before those: (distance, occupant).
    fn behind(&self, g: &LaneGraph, lane: u32, from: f64, me: u32) -> Option<(f64, Occupant)> {
        let list = &self.lanes[lane as usize];
        let i = list.partition_point(|o| o.station < from);
        if let Some(o) = list[..i].iter().rev().find(|o| o.agent != me) {
            return Some((from - o.station, *o));
        }
        let mut best: Option<(f64, Occupant)> = None;
        let mut consider = |d: f64, o: &Occupant| {
            if d <= SEARCH && best.is_none_or(|b| d < b.0) {
                best = Some((d, *o));
            }
        };
        for &c in &g.lanes()[lane as usize].predecessors {
            let cl = &g.connectors()[c as usize].line;
            if let Some(o) = self.connectors[c as usize].iter().rev().find(|o| o.agent != me) {
                consider(from + cl.length() - o.station, o);
                continue;
            }
            let prev = g.connectors()[c as usize].from;
            let pl = &g.lanes()[prev as usize].line;
            if let Some(o) = self.lanes[prev as usize].iter().rev().find(|o| o.agent != me) {
                consider(from + cl.length() + pl.length() - o.station, o);
            }
        }
        best
    }
}

/// State of a `traffic` driver.
#[derive(Clone, Debug)]
pub struct TrafficDriver {
    spec: TrafficDriverSpec,
    geometry: DriverGeometry,
    rng: SimRng,
    /// Drawn per episode.
    pub factor: f64,
    /// Car-following parameters; `v0` is the desired speed of the last step (after the
    /// preview of limits and bends).
    pub idm: Idm,
    pub politeness: f64,
    pub change_time: f64,
    /// Where it is on the lane graph (`None`: off the lanes; it stands still).
    pub place: Option<Place>,
    /// The next connectors to take (up to two).
    pub plan: Vec<u32>,
    pub change: Option<LaneChange>,
    /// Time since the last lane change ended (s), and waiting at a lane's end (s).
    pub since_change: f64,
    pub stuck: f64,
    /// Lane changes made this episode.
    pub changes: u32,
}

impl TrafficDriver {
    pub fn new(spec: &TrafficDriverSpec, geometry: DriverGeometry) -> Self {
        let mut d = Self {
            spec: spec.clone(),
            geometry,
            rng: Seed::from_u64(0).rng(),
            factor: 1.0,
            idm: Idm::OTHER,
            politeness: 0.0,
            change_time: 4.0,
            place: None,
            plan: Vec::new(),
            change: None,
            since_change: COOLDOWN,
            stuck: 0.0,
            changes: 0,
        };
        d.draw();
        d
    }

    pub fn spec(&self) -> &TrafficDriverSpec {
        &self.spec
    }

    pub fn geometry(&self) -> DriverGeometry {
        self.geometry
    }

    fn draw(&mut self) {
        let s = &self.spec;
        let mut pick = |r: [f64; 2]| if r[1] > r[0] { self.rng.range(r[0], r[1]) } else { r[0] };
        self.factor = pick(s.speed_factor);
        let headway = pick(s.headway);
        let accel = pick(s.accel);
        let decel = pick(s.decel);
        self.politeness = pick(s.politeness);
        self.change_time = pick(s.change_time);
        self.idm = Idm { v0: self.idm.v0, headway, min_gap: s.min_gap, accel, decel };
    }

    /// Start an episode with a new random stream at `pose`.
    pub fn reset(&mut self, seed: Seed, world: &StaticWorld, pose: &Pose) {
        self.rng = seed.rng();
        self.draw();
        self.place = None;
        self.plan.clear();
        self.change = None;
        self.since_change = COOLDOWN;
        self.stuck = 0.0;
        self.changes = 0;
        self.locate(world, pose);
    }

    /// Where the lane graph goes on from the place: the elements with the distance from the
    /// vehicle's reference point to their start, up to [`SEARCH`] m (following the plan), and
    /// whether it ends at a lane without a planned connector (a dead end).
    fn chain(&self, g: &LaneGraph) -> (Vec<(Elem, f64)>, bool) {
        let Some(p) = self.place else { return (Vec::new(), false) };
        let mut out = vec![(p.elem, -p.station)];
        let mut base = p.elem.line(g).length() - p.station;
        let mut plan = self.plan.iter();
        let mut at = p.elem;
        while base <= SEARCH {
            let next = match at {
                Elem::Lane(l) => match plan.next() {
                    Some(&c) if g.connectors()[c as usize].from == l => Elem::Connector(c),
                    _ => return (out, true),
                },
                Elem::Connector(c) => Elem::Lane(g.connectors()[c as usize].to),
            };
            out.push((next, base));
            base += next.line(g).length();
            at = next;
        }
        (out, false)
    }

    /// Lanes of the same road and direction as `lane` (the lanes it may change into and
    /// itself), left to right.
    fn group(g: &LaneGraph, lane: u32) -> Vec<u32> {
        let mut first = lane;
        while let Some(l) = g.lanes()[first as usize].left {
            first = l;
        }
        let mut out = vec![first];
        while let Some(r) = g.lanes()[*out.last().expect("one") as usize].right {
            out.push(r);
        }
        out
    }

    /// Whether lane `l` can be reached by lane changes along `lane` ([`CHANGE_ROOM`] m per
    /// change; `lane` itself always).
    fn reachable(g: &LaneGraph, lane: u32, l: u32) -> bool {
        let here = &g.lanes()[lane as usize];
        let k = g.lanes()[l as usize].index.abs_diff(here.index);
        f64::from(k) * CHANGE_ROOM <= here.line.length()
    }

    /// The connector making the same movement as `c` (the same turn into the same road and
    /// direction) from the lane of `lane`'s group nearest to `lane` (the right one of two)
    /// that can be reached along it.
    fn equivalent(g: &LaneGraph, c: u32, lane: u32) -> Option<u32> {
        let cc = &g.connectors()[c as usize];
        let target = &g.lanes()[cc.to as usize];
        let here = i32::from(g.lanes()[lane as usize].index);
        let mut best: Option<(i32, i32, u32)> = None;
        for l in Self::group(g, lane).into_iter().filter(|&l| Self::reachable(g, lane, l)) {
            let from = &g.lanes()[l as usize];
            for &k in &from.successors {
                let kc = &g.connectors()[k as usize];
                let to = &g.lanes()[kc.to as usize];
                if kc.turn == cc.turn && to.road == target.road && to.dir == target.dir {
                    let key = ((i32::from(from.index) - here).abs(), -i32::from(from.index), k);
                    if best.is_none_or(|b| key < b) {
                        best = Some(key);
                    }
                }
            }
        }
        best.map(|b| b.2)
    }

    /// Re-take the planned movements from the lanes nearest to `lane` onwards; from the first
    /// that cannot be made any more, plan anew.
    fn remap(&mut self, g: &LaneGraph, world: &StaticWorld, lane: u32) {
        let mut base = lane;
        for k in 0..self.plan.len() {
            match Self::equivalent(g, self.plan[k], base) {
                Some(c) => {
                    self.plan[k] = c;
                    base = g.connectors()[c as usize].to;
                }
                None => {
                    self.plan.truncate(k);
                    break;
                }
            }
        }
        self.fill_plan(g, world);
    }

    /// A connector leaving the lanes of `lane`'s group (only `lane` itself when `own`), drawn
    /// with weight 1 + the class rank of the road it leads into; U-turns only when nothing
    /// else leaves.
    fn choose(&mut self, g: &LaneGraph, world: &StaticWorld, lane: u32, own: bool) -> Option<u32> {
        let leaving = |lanes: &[u32]| -> Vec<u32> {
            lanes.iter().flat_map(|&l| g.lanes()[l as usize].successors.iter().copied()).collect()
        };
        let all = if own {
            leaving(&[lane])
        } else {
            let group = Self::group(g, lane);
            let near: Vec<u32> = group.iter().copied().filter(|&l| Self::reachable(g, lane, l)).collect();
            let all = leaving(&near);
            // Nothing leaves within reach: any lane of the road.
            if all.is_empty() { leaving(&group) } else { all }
        };
        // Into lanes with a way on (connectors leaving lanes within reach), and U-turns only
        // when nothing else is left.
        let way_on = |c: &u32| {
            let to = g.connectors()[*c as usize].to;
            Self::group(g, to)
                .into_iter()
                .any(|l| Self::reachable(g, to, l) && !g.lanes()[l as usize].successors.is_empty())
        };
        let on: Vec<u32> = all.iter().copied().filter(way_on).collect();
        let all = if on.is_empty() { all } else { on };
        let turns: Vec<u32> = all.iter().copied().filter(|&c| g.connectors()[c as usize].turn != Turn::UTurn).collect();
        let options = if turns.is_empty() { all } else { turns };
        let roads = world.roads().roads();
        let weight = |c: u32| {
            let to = &g.lanes()[g.connectors()[c as usize].to as usize];
            1.0 + f64::from(class_rank(roads[to.road as usize].class))
        };
        let total: f64 = options.iter().map(|&c| weight(c)).sum();
        if options.is_empty() {
            return None;
        }
        let mut x = self.rng.uniform() * total;
        for &c in &options {
            x -= weight(c);
            if x < 0.0 {
                return Some(c);
            }
        }
        options.last().copied()
    }

    /// Keep two connectors planned ahead.
    fn fill_plan(&mut self, g: &LaneGraph, world: &StaticWorld) {
        let Some(p) = self.place else { return };
        let mut lane = match p.elem {
            Elem::Lane(l) => l,
            Elem::Connector(c) => g.connectors()[c as usize].to,
        };
        if let Some(&c) = self.plan.last() {
            lane = g.connectors()[c as usize].to;
        }
        while self.plan.len() < 2 {
            let Some(c) = self.choose(g, world, lane, false) else { break };
            let c = Self::equivalent(g, c, lane).unwrap_or(c);
            self.plan.push(c);
            lane = g.connectors()[c as usize].to;
        }
    }

    /// Find the place for `pose`: carried on along the graph from the last one, or searched
    /// anew when lost.
    pub fn locate(&mut self, world: &StaticWorld, pose: &Pose) {
        let g = world.roads().lanes();
        if g.is_empty() {
            self.place = None;
            return;
        }
        let xy = pose.pos.truncate();
        if let Some(mut p) = self.place {
            for _ in 0..4 {
                let line = p.elem.line(g);
                let lo = (p.station - 2.0).max(0.0);
                let hi = (p.station + 40.0).min(line.length());
                let s = if hi - lo > 1e-6 { lo + Polyline::new(line.slice(lo, hi)).project(xy).station } else { lo };
                p.station = s;
                // Beyond the end: on to the next element.
                let end = line.length();
                let past = beyond(line, xy);
                if s >= end - 1e-6 && past > 0.0 {
                    let next = match p.elem {
                        Elem::Lane(l) => {
                            let own = &g.lanes()[l as usize].successors;
                            let c = match self.plan.first() {
                                Some(&c) if own.contains(&c) => c,
                                _ => {
                                    self.plan.clear();
                                    match self.choose(g, world, l, true) {
                                        Some(c) => c,
                                        None => break,
                                    }
                                }
                            };
                            if self.plan.first() == Some(&c) {
                                self.plan.remove(0);
                            }
                            self.change = None;
                            Elem::Connector(c)
                        }
                        Elem::Connector(c) => Elem::Lane(g.connectors()[c as usize].to),
                    };
                    p = Place { elem: next, station: 0.0 };
                    continue;
                }
                break;
            }
            let off = p.elem.line(g).point_at(p.station).truncate().distance(xy);
            let allowed = LOST + self.change.map_or(0.0, |_| 4.0);
            if off <= allowed {
                self.place = Some(p);
                self.fill_plan(g, world);
                return;
            }
        }
        // Lost (or new): the nearest lane or connector along the heading.
        self.plan.clear();
        self.change = None;
        self.place = find(g, xy, yaw(pose.rot));
        self.fill_plan(g, world);
    }

    /// Its entries in the index: at its place, and in the lane it is leaving while changing.
    pub fn occupy(&self, g: &LaneGraph, index: &mut LaneIndex, agent: u32, speed: f64) {
        let Some(p) = self.place else { return };
        let geo = self.geometry;
        let idm = Idm { v0: self.desired(g), ..self.idm };
        let o = Occupant { agent, station: p.station, front: geo.front, rear: -geo.rear, speed, idm };
        index.insert(p.elem, o);
        if let (Some(c), Elem::Lane(l)) = (self.change, p.elem) {
            let s = p.station * g.lanes()[c.from as usize].line.length() / g.lanes()[l as usize].line.length();
            index.insert(Elem::Lane(c.from), Occupant { station: s, ..o });
        }
    }

    /// Desired speed at the place, before the preview (m/s).
    pub fn desired(&self, g: &LaneGraph) -> f64 {
        self.place.map_or(0.0, |p| self.factor * p.elem.speed_limit(g))
    }

    /// The lane to be in at the end of the current lane: where the next connector leaves.
    fn required_lane(&self, g: &LaneGraph) -> Option<u32> {
        let p = self.place?;
        let Elem::Lane(l) = p.elem else { return None };
        let c = *self.plan.first()?;
        let from = g.connectors()[c as usize].from;
        (Self::group(g, l).contains(&from)).then_some(from)
    }

    /// IDM acceleration at speed `v` behind the first occupant ahead along `chain` from
    /// station `from` (the driver's own reference point there).
    fn follow(&self, index: &LaneIndex, chain: &[(Elem, f64)], from: f64, me: u32, v: f64, idm: &Idm) -> f64 {
        let leader = index.ahead(chain, from, me).map(|(d, o)| (d - self.geometry.front - o.rear, v - o.speed));
        idm.accel(v, leader)
    }

    /// The command for the next policy step of `dt` of agent `me` at `pose` moving at `speed`
    /// (m/s, forward), from the snapshot `index`.
    pub fn drive(
        &mut self,
        world: &StaticWorld,
        pose: &Pose,
        speed: f64,
        dt: f64,
        me: u32,
        index: &LaneIndex,
    ) -> GroundSetpoint {
        let stand = GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 };
        let g = world.roads().lanes();
        let Some(p) = self.place else { return stand };
        let v = speed.max(0.0);
        self.since_change += dt;
        if let Some(c) = self.change
            && p.station >= c.start + c.length
        {
            self.change = None;
            self.since_change = 0.0;
        }

        // Desired speed: the limit here, and those ahead and the bends within braking reach.
        let (chain, dead) = self.chain(g);
        let mut v0 = self.desired(g);
        let b = self.idm.decel;
        let reach = v0 * v0 / (2.0 * b) + 20.0;
        for &(e, base) in chain.iter().skip(1).take_while(|(_, base)| *base < reach) {
            let lim = self.factor * e.speed_limit(g);
            v0 = v0.min((lim * lim + 2.0 * b * base).sqrt());
        }
        let mut d = 0.0;
        while d < reach {
            if let Some((e, s)) = at_distance(g, &chain, d) {
                let k = e.line(g).curvature_at(s).abs().max(1e-6);
                v0 = v0.min((self.spec.lateral_accel / k + 2.0 * b * d).sqrt());
            }
            d += PREVIEW_STEP;
        }
        let idm = Idm { v0, ..self.idm };
        self.idm.v0 = v0;

        // Following: the vehicle ahead along the chain, and while changing, also in the lane
        // being left.
        let mut acc = self.follow(index, &chain, p.station, me, v, &idm);
        if let (Some(c), Elem::Lane(l)) = (self.change, p.elem) {
            let s = p.station * g.lanes()[c.from as usize].line.length() / g.lanes()[l as usize].line.length();
            acc = acc.min(self.follow(index, &[(Elem::Lane(c.from), -s)], s, me, v, &idm));
        }
        // The end of a dead end, or of a lane that must be left for the next connector: a
        // standing obstacle there.
        let required = self.required_lane(g);
        if dead && let Some(&(e, base)) = chain.last() {
            let end = base + e.line(g).length();
            acc = acc.min(idm.accel(v, Some(((end - self.geometry.front - END_HOLD).max(0.0), v))));
        }
        if let Elem::Lane(l) = p.elem
            && (required.is_some_and(|r| r != l) || dead && chain.len() == 1)
        {
            let left = p.elem.line(g).length() - p.station;
            acc = acc.min(idm.accel(v, Some(((left - self.geometry.front - END_HOLD).max(0.0), v))));
            if left < self.geometry.front + 5.0 && v < 0.5 {
                self.stuck += dt;
            } else {
                self.stuck = 0.0;
            }
            if self.stuck > STUCK {
                // Give up the turn: take a connector from this lane.
                self.plan.clear();
                if let Some(c) = self.choose(g, world, l, true) {
                    self.plan.push(c);
                }
                self.fill_plan(g, world);
                self.stuck = 0.0;
            }
        }

        // Lane changes.
        if self.change.is_none()
            && self.since_change >= COOLDOWN
            && let Elem::Lane(l) = p.elem
            && let Some(to) = self.mobil(g, index, me, l, p.station, v, &idm, required)
        {
            let lt = g.lanes()[to as usize].line.length();
            let s = p.station * lt / g.lanes()[l as usize].line.length();
            let left = lt - s;
            let length = (v * self.change_time).max(MIN_CHANGE).min(left - self.geometry.front - 2.0).max(MIN_CHANGE);
            self.change = Some(LaneChange { from: l, start: s, length });
            self.place = Some(Place { elem: Elem::Lane(to), station: s });
            self.changes += 1;
            // The planned movements from the lanes nearest to the new one.
            self.remap(g, world, to);
        }

        // Steering: pure pursuit on the lane line, blended across during a change.
        let place = self.place.expect("placed");
        let (chain, _) = self.chain(g);
        let look = self.spec.lookahead[0] + self.spec.lookahead[1] * v;
        let target = self.reference(g, &chain, place, look);
        let heading = yaw(pose.rot);
        let local = DVec2::from_angle(-heading).rotate(target - pose.pos.truncate());
        let d2 = local.length_squared();
        let curvature = if d2 > 1e-6 { 2.0 * local.y / d2 } else { 0.0 };
        let limit = self.geometry.max_curvature;
        let curvature = curvature.clamp(-limit, limit);
        let command = (v + acc.max(-3.0 * b) * SPEED_LAG).max(0.0);
        GroundSetpoint::SpeedCurvature { speed: command, curvature }
    }

    /// The point `ahead` m on along the chain, shifted towards the lane being left.
    fn reference(&self, g: &LaneGraph, chain: &[(Elem, f64)], place: Place, ahead: f64) -> DVec2 {
        let Some((e, s)) = at_distance(g, chain, ahead) else {
            return place.elem.line(g).point_at(place.station).truncate();
        };
        let point = e.line(g).point_at(s).truncate();
        match (self.change, e) {
            (Some(c), Elem::Lane(l)) if e == place.elem => {
                let t = ((s - c.start) / c.length).clamp(0.0, 1.0);
                let w = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t);
                let from = &g.lanes()[c.from as usize].line;
                let sf = s * from.length() / g.lanes()[l as usize].line.length();
                point + (1.0 - w) * (from.point_at(sf).truncate() - point)
            }
            _ => point,
        }
    }

    /// MOBIL: the lane to change into, if any.
    #[allow(clippy::too_many_arguments)]
    fn mobil(
        &self,
        g: &LaneGraph,
        index: &LaneIndex,
        me: u32,
        lane: u32,
        station: f64,
        v: f64,
        idm: &Idm,
        required: Option<u32>,
    ) -> Option<u32> {
        let l = &g.lanes()[lane as usize];
        let left_here = l.line.length() - station;
        let geo = self.geometry;
        let group = Self::group(g, lane);
        let pos = |x: u32| group.iter().position(|&y| y == x).expect("in the group");
        // Mandatory: towards the required lane.
        let must = required.filter(|&r| r != lane).map(|r| if pos(r) < pos(lane) { l.left } else { l.right });
        let (front, rear) = (geo.front, -geo.rear);
        let chain_here = self.lane_chain(g, lane, station);
        let a_c = self.follow(index, &chain_here, station, me, v, idm);
        // The old follower (distances between reference points), now and with the driver gone.
        let leader_here = index.ahead(&chain_here, station, me);
        let mut da_o = 0.0;
        if let Some((d, o)) = index.behind(g, lane, station, me) {
            let now = o.idm.accel(o.speed, Some((d - o.front - rear, o.speed - v)));
            let then =
                o.idm.accel(o.speed, leader_here.map(|(dl, ol)| (d + dl - o.front - ol.rear, o.speed - ol.speed)));
            da_o = then - now;
        }
        let mut best: Option<(f64, u32)> = None;
        for (side, target) in [(-1.0, l.left), (1.0, l.right)] {
            let Some(t) = target else { continue };
            let mandatory = must == Some(Some(t));
            if must.is_some() && !mandatory {
                continue;
            }
            // Discretionary changes only well before the lane's end, and not away from the
            // lane the route needs when nearing it.
            let away = required.is_some_and(|r| pos(t).abs_diff(pos(r)) > pos(lane).abs_diff(pos(r)));
            if !mandatory && (left_here < CHANGE_CLEAR || away && left_here < KEEP_ROUTE) {
                continue;
            }
            let lt = &g.lanes()[t as usize];
            let s = station * lt.line.length() / l.line.length();
            if lt.line.length() - s < MIN_CHANGE + front {
                continue;
            }
            // Neighbours in the target lane: the first whose reference point lies ahead of
            // the driver's rear, and the last behind its front (distances from `s`).
            let chain_t = self.lane_chain(g, t, s);
            let leader = index.ahead(&chain_t, s - rear, me);
            let follower = index.behind(g, t, s + front, me).map(|(d, o)| (d - front, o));
            let leader_gap = leader.map(|(d, o)| (d - o.rear - front, o));
            let follower_gap = follower.map(|(d, o)| (d - o.front - rear, d, o));
            // Room beside it.
            if leader_gap.is_some_and(|(gap, _)| gap < 0.5) || follower_gap.is_some_and(|(gap, _, _)| gap < 0.5) {
                continue;
            }
            // Safety: the new follower's braking behind the driver.
            let mut da_n = 0.0;
            if let Some((gap, d, o)) = follower_gap {
                let then = o.idm.accel(o.speed, Some((gap, o.speed - v)));
                if then < -self.spec.safe_decel {
                    continue;
                }
                let now = o.idm.accel(o.speed, leader.map(|(dl, ol)| (d + dl - o.front - ol.rear, o.speed - ol.speed)));
                da_n = then - now;
            }
            if mandatory {
                return Some(t);
            }
            let a_t = idm.accel(v, leader_gap.map(|(gap, o)| (gap, v - o.speed)));
            let incentive = a_t - a_c + self.politeness * (da_n + da_o);
            let needed = self.spec.threshold - side * self.spec.keep_right;
            if incentive > needed && best.is_none_or(|b| incentive - needed > b.0) {
                best = Some((incentive - needed, t));
            }
        }
        best.map(|b| b.1)
    }

    /// The chain of lane `lane` from station `station`, continued through the planned
    /// connector when it leaves that lane.
    fn lane_chain(&self, g: &LaneGraph, lane: u32, station: f64) -> Vec<(Elem, f64)> {
        let mut out = vec![(Elem::Lane(lane), -station)];
        let base = g.lanes()[lane as usize].line.length() - station;
        if let Some(&c) = self.plan.first()
            && g.connectors()[c as usize].from == lane
        {
            out.push((Elem::Connector(c), base));
            let cl = g.connectors()[c as usize].line.length();
            out.push((Elem::Lane(g.connectors()[c as usize].to), base + cl));
        }
        out
    }
}

/// The lane or connector nearest to `xy` within [`FIND`] whose direction there lies within
/// 90° of `heading`: lanes from the graph's index, connectors (not indexed) by a scan of
/// those starting or ending within reach.
fn find(g: &LaneGraph, xy: DVec2, heading: f64) -> Option<Place> {
    let fits = |line: &Polyline, s: f64| wrap_angle(line.heading_at(s) - heading).abs() <= 0.5 * std::f64::consts::PI;
    let mut best: Option<(f64, Place)> = None;
    if let Some((l, s, _)) = g.nearest_lane(xy, FIND, Some(heading)) {
        let d = g.lanes()[l as usize].line.point_at(s).truncate().distance(xy);
        best = Some((d, Place { elem: Elem::Lane(l), station: s }));
    }
    for (k, c) in g.connectors().iter().enumerate() {
        let (a, b) = (c.line.point_at(0.0).truncate(), c.line.point_at(c.line.length()).truncate());
        if xy.distance(a).min(xy.distance(b)) > c.line.length() + FIND {
            continue;
        }
        let p = c.line.project(xy);
        if p.distance <= FIND && fits(&c.line, p.station) && best.is_none_or(|b| p.distance < b.0) {
            best = Some((p.distance, Place { elem: Elem::Connector(k as u32), station: p.station }));
        }
    }
    best.map(|b| b.1)
}

/// The element and station `d` m ahead of the vehicle along `chain`.
fn at_distance(g: &LaneGraph, chain: &[(Elem, f64)], d: f64) -> Option<(Elem, f64)> {
    let mut found = None;
    for &(e, base) in chain {
        if base <= d {
            found = Some((e, (d - base).min(e.line(g).length())));
        } else {
            break;
        }
    }
    found
}

/// Distance of `p` past the end of `line` along its final direction (m).
fn beyond(line: &Polyline, p: DVec2) -> f64 {
    let end = line.point_at(line.length()).truncate();
    (p - end).dot(DVec2::from_angle(line.heading_at(line.length())))
}

/// `count` spawns in random lanes (drawn by length, 5 m clear of their ends) along their
/// direction, at least `min_separation` from `placed` where possible (appended to it); `lift`
/// above the ground. `None` on maps without lanes.
pub(crate) fn lane_spawns(
    world: &StaticWorld,
    count: usize,
    min_separation: f64,
    lift: f64,
    placed: &mut Vec<DVec3>,
    rng: &mut SimRng,
) -> Option<Vec<(DVec3, f64)>> {
    let g = world.roads().lanes();
    let lanes: Vec<(u32, f64)> = (g.lanes().iter().enumerate())
        .filter(|(_, l)| l.line.length() > 12.0)
        .map(|(k, l)| (k as u32, l.line.length()))
        .collect();
    let total: f64 = lanes.iter().map(|l| l.1 - 10.0).sum();
    if lanes.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best: Option<(f64, DVec3, f64)> = None;
        for _ in 0..64 {
            let mut x = rng.uniform() * total;
            let &(k, len) = lanes
                .iter()
                .find(|l| {
                    x -= l.1 - 10.0;
                    x < 0.0
                })
                .unwrap_or(lanes.last().expect("lanes"));
            let s = 5.0 + rng.uniform() * (len - 10.0);
            let line = &g.lanes()[k as usize].line;
            let xy = line.point_at(s).truncate();
            let pos = xy.extend(world.terrain().height(xy.x, xy.y) + lift);
            let apart = placed.iter().map(|q| q.truncate().distance(xy)).fold(f64::INFINITY, f64::min);
            if best.is_none_or(|b| apart > b.0) {
                best = Some((apart, pos, line.heading_at(s)));
            }
            if apart >= min_separation {
                break;
            }
        }
        let (_, pos, heading) = best.expect("drawn");
        placed.push(pos);
        out.push((pos, wrap_angle(heading)));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idm_holds_the_equilibrium_gap() {
        let p = Idm { v0: 15.0, headway: 1.5, min_gap: 2.0, accel: 1.4, decel: 2.0 };
        for v in [0.0, 3.0, 8.0, 12.0] {
            let gap = p.equilibrium_gap(v);
            assert!(p.accel(v, Some((gap, 0.0))).abs() < 1e-9, "{v}");
        }
        // Free road: accelerates below v₀, brakes above.
        assert!((p.accel(0.0, None) - 1.4).abs() < 1e-12);
        assert!(p.accel(16.0, None) < 0.0);
        // Closing in brakes harder than holding the same gap.
        assert!(p.accel(10.0, Some((20.0, 5.0))) < p.accel(10.0, Some((20.0, 0.0))));
    }
}
