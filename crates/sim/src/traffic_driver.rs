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
//! - **Steering**: pure pursuit on the lane line (blended across during a change); long
//!   vehicles from their rear axle.
//!
//! Every policy step the world first has each driver find its place on the graph, then builds
//! a [`LaneIndex`] of every agent on the lanes (drivers by their place, other agents near the
//! ground by the lane nearest to them), sorted by station, and then has each driver decide
//! from that snapshot (in agent order, so the result does not depend on threads). Leaders and
//! followers are found in O(log N) per lane.
//!
//! **Junctions**: from [`DECIDE`] m (or its braking distance) before the end of its lane, a
//! driver asks whether it may enter its next connector; until it may, the stop line is a
//! standing obstacle (or, when it can no longer stop there, the junction's edge). It may not
//! on red, on amber when it can stop comfortably or would not clear the line before red (at
//! the speed it plans, bends beyond included), at a stop sign before having stopped at the line, without
//! room for itself beyond the junction (unless the vehicle there moves on), or when a
//! conflicting connector is taken: a vehicle on it, or one granted it, passes through the
//! conflict zone (the conflict's stretch of both connectors, with a [`ZONE`] m margin) less
//! than [`TIME_MARGIN`] s apart from the driver's own passage. Where it gives way
//! (the conflict's `yields`), it also needs a gap of at least its critical gap (drawn per
//! driver, default 4–6 s) and of its own passage + [`TIME_MARGIN`] before any vehicle
//! approaching the other connector (on its lane or up to two elements before it, planning to
//! take it; agents that are not traffic drivers are assumed to), unless that one faces red.
//! Within [`GRANT`] m (or 3 s) of the line an allowed entry becomes a *grant*: the driver is
//! committed, and every later decision treats it as on the connector. Decisions are made in
//! agent order and grants are entered in the index at once, so two drivers never take
//! conflicting grants in the same step. A granted driver still stops when the light turns
//! and it can, and inside a junction (and when about to enter one, the next junction too)
//! it stops before a conflict zone that another vehicle occupies. Connectors leaving the same
//! lane (diverging) are no conflict for entering: the driver follows a vehicle ahead on the
//! other one until they have parted; inside a merge zone it follows the vehicle nearer the
//! common end. A driver also does not enter a junction when it could not go on through the
//! next one that a lane too short to wait on (its length and standstill gap before its line,
//! held back for sweeps and crossings) leads to: the
//! next one is asked for beforehand (and so on; its light and stop sign aside, as they only
//! delay), so that it does not wait inside a junction for other traffic; once committed, it
//! brakes in time for a light there.
//!
//! **Deadlocks**: a driver that has waited at a line for `deadlock` s no longer gives way to
//! vehicles that stand waiting themselves (the first in agent order goes, the others then
//! see its grant).
//!
//! **Learning agents** near the ground are on the lanes as leaders, followers and crossing
//! traffic like any vehicle, but a driver notices each only with probability `attention`,
//! drawn when it first comes within [`ENCOUNTER`] m (and again once it has been farther than
//! [`ENCOUNTER_END`] m); an agent not noticed is not there for that driver.
//!
//! **Long vehicles** (offtracking length over [`SWING_LENGTH`] m: trucks, buses) keep to the
//! connectors their lock allows ([`Network`]) and steer from their rear axle, so that axle runs
//! on the line and the front swings wide; those towing trailers also swing out in left bends
//! so that the trailers cut in less. Each allowed connector sweeps a band (the body's reach
//! both ways, until it has straightened out on the lane after); stop lines of lanes whose
//! waiting vehicles the band would reach are set back ([`Sweeps`]), for everyone. A long
//! vehicle does not enter a connector while someone stands in its band or could not stop
//! short of it, and while one is granted it or sweeps through, the others wait behind that
//! connector's setback even when allowed through. Other drivers never pass a standing long
//! vehicle within a junction.
//!
//! **Pedestrians** ([`crate::pedestrians`]): a driver stops [`CROSSWALK_GAP`] m before a
//! crossing ahead while a pedestrian on it (or committed to it) has not passed its lane by
//! [`CROSSWALK_CLEAR`] m; a crossing its front has reached is cleared once rolling (over
//! [`CROSSWALK_ROLLING`]), but pedestrians in its path stay obstacles. Stop lines of lanes
//! ending at a crossing are set back behind the band.
//!
//! **Cyclists** (single-track vehicles, M8c) keep right: in a bike lane where there is one,
//! else [`CYCLIST_EDGE`] m from their lane's right edge ([`keep_right`]; on connectors blended
//! from the lane before to the lane after), and take connectors from their own lane only,
//! without discretionary lane changes. A cyclist and a car (or two cyclists) beside each other
//! in a lane are no leader for each other when they lie clear by [`PASS_CLEAR`] m
//! ([`CYCLIST_CLEAR`] m) and the lane does not end within [`PASS_END`] m. Cyclists, who lean
//! late and cannot lean hard, also brake to each bend's speed (from the curvature of the
//! offset path, with a margin, and by the speed loop's lag early), look farther ahead
//! ([`CYCLIST_LOOKAHEAD`] + [`CYCLIST_PREVIEW`] s), ride off briskly from a standstill and take
//! no bend slower than [`CYCLIST_START`] (at walking pace they ride on their feet, wobble and
//! on grades roll back).
//!
//! **Overtaking**: on a road with one lane each way and no median, a car closing in on a
//! cyclist shifts left by what it takes to pass it [`PASS_CLEAR`] m clear (at most a lane
//! width) when the pass, simulated at its speed, ends [`PASS_END`] m before the lane's end
//! without a crossing on the way, the oncoming lane (and the elements leading into it) stays
//! clear of traffic at up to [`ONCOMING`] times the limit for the pass and [`PASS_MARGIN`] s
//! more, and there is room ahead to return; it shifts back (at [`SHIFT_RATE`]) once
//! [`PASS_GAP`] m past.
//!
//! **Buses** (`bus`) drive a loop with stops instead of at random (see [`BusRoute`]).
//!
//! **Respawn**: a driver standing still for `respawn` s, lost off the lanes for
//! [`LOST_RESPAWN`] s or off the map is put back at rest at a random lane position at least
//! `respawn_clear` m from every learning agent and 20 m from every other agent (see
//! [`respawn_spot`]; the world does this at the start of a policy step).

use crate::driver::DriverGeometry;
use crate::traffic::Signals;
use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::{wrap_angle, yaw};
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::terrain::Terrain;
use autonomousim_world::lanes::{CROSSWALK, ConflictKind, Control, LaneGraph, Turn, class_rank};
use autonomousim_world::{Light, Polyline, StaticWorld};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    /// Smallest gap (s) accepted in priority traffic when giving way.
    pub critical_gap: [f64; 2],
    /// Waiting at a line this long (s), no longer give way to vehicles waiting themselves.
    pub deadlock: f64,
    /// Standing still this long (s) puts the vehicle back elsewhere (0: never).
    pub respawn: f64,
    /// Respawns at least this far (m) from every learning agent.
    pub respawn_clear: f64,
    /// A bus line: drives a loop with stops instead of turning at random.
    pub bus: Option<BusSpec>,
    /// Probability (0–1) that the driver notices a learning agent on the lanes, drawn per
    /// encounter; one not noticed is not there for it (0: oblivious).
    pub attention: f64,
    /// Desired speed (m/s), where lower than `speed_factor` of the limit (cyclists).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<[f64; 2]>,
}

/// A bus line of a `traffic` driver (see [`BusRoute`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BusSpec {
    /// Length of the loop (m), drawn per route.
    pub length: [f64; 2],
    /// Distance between stops (m).
    pub stop_spacing: f64,
    /// Time standing at a stop (s), drawn per stop.
    pub dwell: [f64; 2],
}

impl Default for BusSpec {
    fn default() -> Self {
        Self { length: [1500.0, 3000.0], stop_spacing: 300.0, dwell: [10.0, 30.0] }
    }
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
            critical_gap: [4.0, 6.0],
            deadlock: 10.0,
            respawn: 120.0,
            respawn_clear: 100.0,
            bus: None,
            attention: 1.0,
            speed: None,
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
            && self.lookahead[1] >= 0.0
            && range(self.critical_gap, 0.0)
            && self.deadlock > 0.0
            && self.respawn >= 0.0
            && self.respawn_clear >= 0.0
            && (0.0..=1.0).contains(&self.attention)
            && self.speed.is_none_or(|s| range(s, 0.0) && s[1] > 0.0)
            && self.bus.as_ref().is_none_or(|b| {
                range(b.length, 0.0) && b.length[1] > 0.0 && b.stop_spacing > 0.0 && range(b.dwell, 0.0)
            });
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
    /// The lane itself, or the lane a connector leads into.
    pub fn lane_after(self, g: &LaneGraph) -> u32 {
        match self {
            Elem::Lane(l) => l,
            Elem::Connector(c) => g.connectors()[c as usize].to,
        }
    }

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
    /// The next two connectors it will take ([`ANY`]: unknown, it may take any).
    pub next: [u32; 2],
    /// Whether it stands waiting at a line.
    pub waiting: bool,
    /// Where it is (m, world), heading (rad) and half its width (m).
    pub pos: DVec2,
    pub heading: f64,
    pub half_width: f64,
    /// Whether it tows trailers (`pos`, `heading` are the towing unit's).
    pub articulated: bool,
    /// Whether it sweeps wide in turns ([`TrafficDriver::is_long`]), and the connector it
    /// came onto its lane by ([`ANY`]: none or unknown).
    pub long: bool,
    pub via: u32,
    /// Whether it is a learning agent (subject to drivers' `attention`).
    pub learner: bool,
    /// Whether this entry is its tail on an element behind its place (`station` counted on
    /// from there, past the element's end).
    pub tail: bool,
    /// Its lateral offset from the element's line (m, positive to the left; drivers: where
    /// they steer to), and whether it is a two-wheeler.
    pub offset: f64,
    pub cyclist: bool,
}

/// A driver's lateral extent, for passing occupants of its lane beside it (see
/// [`LaneIndex::ahead`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Side {
    offset: f64,
    half_width: f64,
    cyclist: bool,
    /// Occupants from this station on are passed by no one (m), but the one being overtaken.
    until: f64,
    overtaken: Option<u32>,
}

impl Side {
    /// Whether occupant `o` lies clear beside: by [`PASS_CLEAR`] m (two cyclists:
    /// [`CYCLIST_CLEAR`] m) where one of the two is a cyclist; cars do not pass each other in
    /// a lane.
    fn passes(&self, o: &Occupant) -> bool {
        let clear = match (self.cyclist, o.cyclist) {
            (false, false) => return false,
            (true, true) => CYCLIST_CLEAR,
            _ => PASS_CLEAR,
        };
        !o.tail
            && (o.station < self.until || self.overtaken == Some(o.agent))
            && (o.offset - self.offset).abs() >= o.half_width + self.half_width + clear
    }
}

/// Passing a slow cyclist on a road with one lane each way: shifted partly into the oncoming
/// lane (see [`TrafficDriver::overtaking`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Overtake {
    pub agent: u32,
    /// Lateral offset (m, to the left of the lane line) passing it.
    pub shift: f64,
}

/// A connector in [`Occupant::next`] that is not known.
pub const ANY: u32 = u32::MAX;

/// A driver committed to a connector it has not reached yet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grant {
    pub agent: u32,
    /// Distance from its front to the connector's start (m).
    pub distance: f64,
    pub speed: f64,
    /// Its length (m).
    pub length: f64,
    /// Whether it is a long vehicle (it sweeps a band; see [`Sweeps`]).
    pub long: bool,
    /// Its acceleration (m/s²) and the speed it plans (m/s), for when it is through.
    pub accel: f64,
    pub v0: f64,
}

/// The agents on each lane and connector, sorted by station (then agent), rebuilt each
/// policy step.
#[derive(Clone, Debug, Default)]
pub struct LaneIndex {
    lanes: Vec<Vec<Occupant>>,
    connectors: Vec<Vec<Occupant>>,
    grants: Vec<Vec<Grant>>,
    /// Where long vehicles sweep past the lines (see [`Sweeps`]).
    sweeps: Arc<Sweeps>,
    /// Learning agents on the lanes and where they are.
    learners: Vec<(u32, DVec2)>,
    /// Learning agents the driver deciding now has not noticed (sorted).
    ignored: Vec<u32>,
    /// Per lane: the crossings over it, with the station of their band's centre.
    lane_crossings: Vec<Vec<(u32, f64)>>,
    /// Per crossing: the pedestrians on it or committed to it (position, unit direction of
    /// travel; see [`Crowd::crossing_users`](crate::pedestrians::Crowd::crossing_users)).
    crossing_users: Vec<Vec<(DVec2, DVec2)>>,
    /// Per lane: the setback of its stop line behind a crossing at its end (m).
    crosswalk_back: Vec<f64>,
    /// Per lane that is the only one of its direction: the oncoming lane beside it (no median).
    opposite: Vec<Option<u32>>,
}

/// Search range for leaders and followers (m).
const SEARCH: f64 = 200.0;
/// Clearance (m) of cars passing cyclists (and cyclists cars) beside them in a lane, and of
/// cyclists passing each other.
pub const PASS_CLEAR: f64 = 1.0;
const CYCLIST_CLEAR: f64 = 0.3;
/// No one passes beside others in a lane within this distance of its end (m), nor overtakes
/// into it.
pub const PASS_END: f64 = 30.0;
/// A cyclist rides this far from its lane's right edge without a bike lane (m), and in a bike
/// lane of at least `BIKE_LANE` m.
const CYCLIST_EDGE: f64 = 0.75;
/// Shortest distance (m) over which cyclists count on braking to a bend's speed.
const BEND_BRAKE: f64 = 1.0;
/// Fraction of a bend's squared speed that cyclists brake to.
const BEND_MARGIN_CYCLIST: f64 = 0.8;
/// Speed (m/s) up to which cyclists free to go ask for at least this speed.
const CYCLIST_START: f64 = 2.5;
/// Pure-pursuit look-ahead of cyclists, at least (m, plus s times the speed).
const CYCLIST_LOOKAHEAD: f64 = 4.0;
const CYCLIST_PREVIEW: f64 = 0.7;
const BIKE_LANE: f64 = 1.0;
/// Overtaking: a cyclist within this gap (m) and this much slower than the desired speed
/// (m/s); the overtaken one is passed by this gap before shifting back (m); oncoming traffic
/// is counted on at up to this factor of its lane's limit, with this margin (s); the shift
/// changes at up to this rate (m/s).
const OVERTAKE_REACH: f64 = 30.0;
const OVERTAKE_DV: f64 = 2.0;
const PASS_GAP: f64 = 5.0;
const ONCOMING: f64 = 1.2;
const PASS_MARGIN: f64 = 3.0;
const SHIFT_RATE: f64 = 1.0;
/// Time step (s) of the passing estimate, and the longest pass (s).
const PASS_STEP: f64 = 0.1;
const PASS_LONGEST: f64 = 20.0;
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
/// Junction rules apply from this far (m) before the line, or the braking distance + 2 s.
pub const DECIDE: f64 = 30.0;
/// An allowed entry is granted within this far (m) of the line, or 3 s.
pub const GRANT: f64 = 15.0;
/// Drivers stop this far (m) before a crossing's band for pedestrians.
const CROSSWALK_GAP: f64 = 1.0;
/// A pedestrian has passed a lane once this far (m) beyond its side.
const CROSSWALK_CLEAR: f64 = 1.0;
/// Pedestrians on a crossing less than this beside a vehicle's path are in its way (m).
const CROSSWALK_SIDE: f64 = 1.0;
/// Above this speed (m/s) a vehicle whose front is on a crossing's band goes on over it.
const CROSSWALK_ROLLING: f64 = 1.0;
/// Waiting vehicles stop this far (m) before the line.
const STOP_LINE: f64 = 1.0;
/// Margin (m) added to both ends of a conflict zone.
pub const ZONE: f64 = 1.0;
/// Clearance (m) a standing vehicle must leave beside a driver's path for the driver to pass
/// it in a conflict zone.
pub const PASSING: f64 = 0.5;
/// Least time (s) between two vehicles' passages of a conflict zone.
pub const TIME_MARGIN: f64 = 1.0;
/// Priority traffic is looked for this far (m) back from its line.
const LOOK_BACK: f64 = 150.0;
/// Slower than this (m/s), a vehicle stands.
const STANDING: f64 = 0.5;
/// Lost off the lanes this long (s), a driver respawns.
pub const LOST_RESPAWN: f64 = 5.0;

impl LaneIndex {
    pub fn new(g: &LaneGraph) -> Self {
        let n = g.connectors().len();
        let m = g.lanes().len();
        Self {
            lanes: vec![Vec::new(); m],
            connectors: vec![Vec::new(); n],
            grants: vec![Vec::new(); n],
            sweeps: Arc::new(Sweeps::none(g)),
            learners: Vec::new(),
            ignored: Vec::new(),
            lane_crossings: {
                let mut lc = vec![Vec::new(); m];
                for (k, c) in g.crossings().iter().enumerate() {
                    for &(l, s) in &c.lanes {
                        lc[l as usize].push((k as u32, s));
                    }
                }
                lc
            },
            crossing_users: vec![Vec::new(); g.crossings().len()],
            crosswalk_back: (0..m)
                .map(|l| {
                    let len = g.lanes()[l].line.length();
                    (g.crossings().iter().flat_map(|c| &c.lanes))
                        .filter(|&&(cl, s)| cl as usize == l && s > len - 2.0 * CROSSWALK)
                        .map(|&(_, s)| (len - (s - 0.5 * CROSSWALK) - STOP_LINE + CROSSWALK_GAP).max(0.0))
                        .fold(0.0, f64::max)
                })
                .collect(),
            opposite: {
                let lanes = g.lanes();
                let mut by_road: Vec<Vec<u32>> = Vec::new();
                for (k, l) in lanes.iter().enumerate() {
                    let r = l.road as usize;
                    if by_road.len() <= r {
                        by_road.resize(r + 1, Vec::new());
                    }
                    by_road[r].push(k as u32);
                }
                (lanes.iter())
                    .map(|l| {
                        if l.left.is_some() || l.right.is_some() {
                            return None;
                        }
                        by_road[l.road as usize].iter().copied().find(|&o| {
                            let o = &lanes[o as usize];
                            o.dir != l.dir
                                && o.left.is_none()
                                && o.right.is_none()
                                && (l.offset + o.offset - 0.5 * (l.width + o.width)).abs() < 0.05
                        })
                    })
                    .collect()
            },
        }
    }

    /// Set the pedestrians on (or committed to) each crossing for this step.
    pub fn set_crossing_users(&mut self, users: &[Vec<(DVec2, DVec2)>]) {
        for (mine, u) in self.crossing_users.iter_mut().zip(users) {
            mine.clone_from(u);
        }
    }

    /// Set where long vehicles sweep.
    pub fn set_sweeps(&mut self, sweeps: Arc<Sweeps>) {
        assert_eq!(sweeps.hold_back.len(), self.lanes.len(), "one per lane");
        self.sweeps = sweeps;
    }

    /// How much farther back than the stop line vehicles wait on `lane` (m): clear of long
    /// vehicles' sweeps and [`CROSSWALK_GAP`] m before a crossing at the lane's end.
    pub fn hold_back(&self, lane: u32) -> f64 {
        self.sweeps.hold_back[lane as usize].max(self.crosswalk_back[lane as usize])
    }

    /// Whether a long vehicle sweeps past the line of `sweep.lane` through connector `c` now or
    /// is about to: granted it, on it, or on the lane after it came by, its last axle not yet
    /// `sweep.after` m on.
    fn sweeping(&self, g: &LaneGraph, c: u32, sweep: &Sweep) -> bool {
        self.connectors[c as usize].iter().any(|o| o.long)
            || self.grants[c as usize].iter().any(|gr| gr.long)
            || self.lanes[g.connectors()[c as usize].to as usize]
                .iter()
                .any(|o| o.long && !o.tail && o.via == c && o.station - o.rear < sweep.after)
    }

    /// Whether it is sized for `g`.
    pub fn fits(&self, g: &LaneGraph) -> bool {
        self.lanes.len() == g.lanes().len() && self.connectors.len() == g.connectors().len()
    }

    /// Empty it for a new step.
    pub fn clear(&mut self) {
        self.lanes.iter_mut().chain(&mut self.connectors).for_each(Vec::clear);
        self.grants.iter_mut().for_each(Vec::clear);
        self.learners.clear();
        self.ignored.clear();
    }

    /// Whether the driver `me` deciding now does not see occupant `o`: itself, or a learning
    /// agent it has not noticed.
    fn skips(&self, o: &Occupant, me: u32) -> bool {
        o.agent == me || (o.learner && self.ignored.binary_search(&o.agent).is_ok())
    }

    /// Enter a grant of connector `c`.
    pub fn grant(&mut self, c: u32, grant: Grant) {
        let list = &mut self.grants[c as usize];
        list.retain(|x| x.agent != grant.agent);
        list.push(grant);
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
    /// `vel`, reaching `radius` around its centre: to the lane under it, if any. A `learner`
    /// is also listed for the drivers' encounters (see [`TrafficDriverSpec::attention`]).
    #[allow(clippy::too_many_arguments)]
    pub fn insert_other(
        &mut self,
        world: &StaticWorld,
        agent: u32,
        pos: DVec3,
        heading: f64,
        vel: DVec3,
        radius: f64,
        learner: bool,
    ) {
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
        let o = Occupant {
            agent,
            station,
            front: radius,
            rear: radius,
            speed,
            idm: Idm::OTHER,
            next: [ANY; 2],
            waiting: false,
            pos: pos.truncate(),
            heading,
            half_width: radius,
            articulated: false,
            long: false,
            via: ANY,
            learner,
            tail: false,
            offset,
            cyclist: false,
        };
        self.lanes[lane as usize].push(o);
        if learner {
            self.learners.push((agent, o.pos));
        }
    }

    /// Sort every list by station (then agent).
    pub fn sort(&mut self) {
        for l in self.lanes.iter_mut().chain(&mut self.connectors) {
            l.sort_by(|a, b| a.station.total_cmp(&b.station).then(a.agent.cmp(&b.agent)));
        }
    }

    /// The first occupant other than `me` ahead of station `from` along `chain` (elements
    /// with the distance from the searcher to their start): (distance between reference
    /// points, occupant). With `side`, those on the first element (a lane) that it passes
    /// beside ([`Side::passes`]) are left out.
    fn ahead(&self, chain: &[(Elem, f64)], from: f64, me: u32, side: Option<Side>) -> Option<(f64, Occupant)> {
        for (k, &(e, base)) in chain.iter().enumerate() {
            if base > SEARCH {
                break;
            }
            let list = self.list(e);
            let start = if k == 0 { from } else { f64::NEG_INFINITY };
            let i = list.partition_point(|o| o.station <= start);
            let beside = |o: &Occupant| k == 0 && matches!(e, Elem::Lane(_)) && side.is_some_and(|s| s.passes(o));
            if let Some(o) = list[i..].iter().find(|o| !self.skips(o, me) && !beside(o)) {
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
        if let Some(o) = list[..i].iter().rev().find(|o| !self.skips(o, me) && !o.tail) {
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
            if let Some(o) = self.connectors[c as usize].iter().rev().find(|o| !self.skips(o, me) && !o.tail) {
                consider(from + cl.length() - o.station, o);
                continue;
            }
            let prev = g.connectors()[c as usize].from;
            let pl = &g.lanes()[prev as usize].line;
            if let Some(o) = self.lanes[prev as usize].iter().rev().find(|o| !self.skips(o, me) && !o.tail) {
                consider(from + cl.length() + pl.length() - o.station, o);
            }
        }
        best
    }

    /// Calls `f` with every occupant (other than `me`) that will take connector `via`, within
    /// `max` m of its start: on the lane it leaves and up to two elements before, with the
    /// distance from its front to the connector.
    fn approaching(&self, g: &LaneGraph, via: u32, max: f64, me: u32, f: &mut dyn FnMut(f64, &Occupant)) {
        let takes = |x: u32, want: u32| x == want || x == ANY;
        let me_or_tail = |o: &Occupant| self.skips(o, me) || o.tail;
        let lane = g.connectors()[via as usize].from;
        let len = g.lanes()[lane as usize].line.length();
        for o in self.lanes[lane as usize].iter().filter(|o| !me_or_tail(o) && takes(o.next[0], via)) {
            f((len - o.station - o.front).max(0.0), o);
        }
        for &pc in &g.lanes()[lane as usize].predecessors {
            let cl = g.connectors()[pc as usize].line.length();
            for o in self.connectors[pc as usize].iter().filter(|o| !me_or_tail(o) && takes(o.next[0], via)) {
                f(cl - o.station - o.front + len, o);
            }
            if cl + len > max {
                continue;
            }
            let pl = g.connectors()[pc as usize].from;
            let pll = g.lanes()[pl as usize].line.length();
            for o in self.lanes[pl as usize].iter() {
                if !me_or_tail(o) && takes(o.next[0], pc) && takes(o.next[1], via) {
                    let d = pll - o.station - o.front + cl + len;
                    if d <= max {
                        f(d, o);
                    }
                }
            }
        }
    }
}

/// The part of the lane graph a long vehicle keeps to: the largest strongly connected part
/// (by lane length, lanes of a road and direction together) of the connectors it can follow,
/// U-turns and bends more than [`BEND_MARGIN`] times tighter than its steering lock left out.
#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    /// Per lane and connector: whether it belongs.
    pub lanes: Vec<bool>,
    pub connectors: Vec<bool>,
}

impl Network {
    /// That of a vehicle of `geometry` if it is long (see [`TrafficDriver::is_long`]).
    pub fn of_long(g: &LaneGraph, geometry: DriverGeometry) -> Option<Arc<Self>> {
        (geometry.tracking > SWING_LENGTH).then(|| Arc::new(Self::new(g, geometry.max_curvature)))
    }

    pub fn new(g: &LaneGraph, max_curvature: f64) -> Self {
        let n = g.lanes().len();
        let group = |mut l: u32| {
            while let Some(x) = g.lanes()[l as usize].left {
                l = x;
            }
            l as usize
        };
        let fits: Vec<bool> = (g.connectors().iter())
            .map(|c| c.turn != Turn::UTurn && bend(&c.line) <= BEND_MARGIN * max_curvature)
            .collect();
        let mut adj = vec![Vec::new(); n];
        let mut radj = vec![Vec::new(); n];
        for (conn, _) in g.connectors().iter().zip(&fits).filter(|(_, f)| **f) {
            let (a, b) = (group(conn.from), group(conn.to));
            adj[a].push(b);
            radj[b].push(a);
        }
        // Kosaraju: finishing order on the graph, then components on the reversed one.
        let mut order = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        for s in 0..n {
            if seen[s] {
                continue;
            }
            seen[s] = true;
            let mut stack = vec![(s, 0usize)];
            while let Some(&mut (u, ref mut k)) = stack.last_mut() {
                if let Some(&v) = adj[u].get(*k) {
                    *k += 1;
                    if !seen[v] {
                        seen[v] = true;
                        stack.push((v, 0));
                    }
                } else {
                    order.push(u);
                    stack.pop();
                }
            }
        }
        let mut comp = vec![usize::MAX; n];
        let mut count = 0;
        for &s in order.iter().rev() {
            if comp[s] != usize::MAX {
                continue;
            }
            comp[s] = count;
            let mut stack = vec![s];
            while let Some(u) = stack.pop() {
                for &v in &radj[u] {
                    if comp[v] == usize::MAX {
                        comp[v] = count;
                        stack.push(v);
                    }
                }
            }
            count += 1;
        }
        // The largest by lane length (the first of equals).
        let mut length = vec![0.0; count];
        for l in 0..n {
            length[comp[group(l as u32)]] += g.lanes()[l].line.length();
        }
        let best = (0..count)
            .fold(None, |b: Option<usize>, k| if b.is_none_or(|b| length[k] > length[b]) { Some(k) } else { b });
        let lanes: Vec<bool> = (0..n).map(|l| Some(comp[group(l as u32)]) == best).collect();
        let connectors = (g.connectors().iter().enumerate())
            .map(|(c, conn)| fits[c] && lanes[conn.from as usize] && lanes[conn.to as usize])
            .collect();
        Self { lanes, connectors }
    }
}

/// Where long vehicles sweep past the lines of other lanes at junctions: the band of each
/// connector of their networks (and 3 [`DriverGeometry::tracking`] m on). The band runs from
/// the front's outer edge (swung wide in left bends, see [`TrafficDriver::reference`]) to the
/// inner edge of the last unit, which runs the offtracking of the connector's tightest bend
/// inside the front's path, then less and less on the lane after (decaying over `tracking`, as
/// a trailing axle's does). Lanes the connector leaves or joins are left out.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sweeps {
    /// Per connector: the lines its band reaches.
    pub connectors: Vec<Vec<Sweep>>,
    /// Per lane: the connectors whose bands reach its line.
    pub lanes: Vec<Vec<u32>>,
    /// Per lane: how much farther back than the stop line vehicles wait (m), clear of all
    /// bands.
    pub hold_back: Vec<f64>,
}

/// A connector's band reaching the line of a lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sweep {
    pub lane: u32,
    /// How far back from the stop line vehicles on it stay clear (m).
    pub back: f64,
    /// How far on the lane after the connector the band still reaches it (m).
    pub after: f64,
}

impl Sweeps {
    /// None: no long vehicles.
    pub fn none(g: &LaneGraph) -> Self {
        Self {
            connectors: vec![Vec::new(); g.connectors().len()],
            lanes: vec![Vec::new(); g.lanes().len()],
            hold_back: vec![0.0; g.lanes().len()],
        }
    }

    /// Those of long vehicles of `networks`.
    pub fn new(g: &LaneGraph, networks: &[(Arc<Network>, DriverGeometry)]) -> Self {
        /// Half width of the waiting vehicles cleared (m), their margin, and sampling step.
        const WAITING: f64 = 1.5;
        const MARGIN: f64 = 0.5;
        const STEP: f64 = 0.5;
        let lanes = g.lanes();
        let mut sweeps = Self::none(g);
        let mut ending = std::collections::BTreeMap::<u32, Vec<u32>>::new();
        for (l, lane) in lanes.iter().enumerate() {
            ending.entry(lane.to_node).or_default().push(l as u32);
        }
        // Cross-sections of a band: point, left normal, tangent, extent to the left `[lo, hi]`.
        // (With the station on the lane after, 0 on the connector.)
        let mut band: Vec<(DVec2, DVec2, DVec2, f64, f64, f64)> = Vec::new();
        for (net, geo) in networks {
            // The trailers' offtracking inside the towing unit's rear axle, which runs on the
            // line (shifted by the swing), and its front's reach outside.
            let l = geo.trailing();
            let off = |k: f64| {
                let r = 1.0 / k.abs().max(1e-6);
                if r > l { r - (r * r - l * l).sqrt() } else { r }
            };
            let f = geo.reach();
            let wide = |k: f64| {
                let r = 1.0 / k.abs().max(1e-6);
                (r * r + f * f).sqrt() - r
            };
            let swings = l > SWING_LENGTH;
            for (c, conn) in g.connectors().iter().enumerate().filter(|(c, _)| net.connectors[*c]) {
                let Some(waiting) = ending.get(&conn.node) else { continue };
                let line = &conn.line;
                let len = line.length();
                let smooth = |s: f64| (-5..=5).map(|d| line.curvature_at(s + f64::from(d))).sum::<f64>() / 11.0;
                let n = (len / STEP).ceil() as usize;
                let turn = (0..=n).map(|k| smooth(k as f64 * STEP)).sum::<f64>().signum();
                let cut = (0..=n).map(|k| off(smooth(k as f64 * STEP))).fold(0.0, f64::max);
                let reach = (0..=n).map(|k| wide(smooth(k as f64 * STEP))).fold(0.0, f64::max);
                band.clear();
                // Extents inside (towards the turn) and outside of the line.
                let mut section = |p: DVec2, heading: f64, swing: f64, inside: f64, outside: f64, after: f64| {
                    let t = DVec2::from_angle(heading);
                    let (lo, hi) = if turn > 0.0 { (-swing - outside, inside - swing) } else { (-inside, outside) };
                    band.push((p, t.perp(), t, lo - geo.half_width, hi + geo.half_width, after));
                };
                for k in 0..=n {
                    let s = (k as f64 * STEP).min(len);
                    let swing = if swings { (0.5 * off(smooth(s).max(0.0))).min(SWING_MAX) } else { 0.0 };
                    section(line.point_at(s).truncate(), line.heading_at(s), swing, cut, reach, 0.0);
                }
                // After it, the trailers straighten out over about their length, the front
                // over its reach.
                let next = &lanes[conn.to as usize].line;
                let m = ((3.0 * l.max(f)).min(next.length()) / STEP).ceil() as usize;
                for k in 1..=m {
                    let s = k as f64 * STEP;
                    let inside = if l > 0.0 { cut * (-s / l).exp() } else { 0.0 };
                    section(next.point_at(s).truncate(), next.heading_at(s), 0.0, inside, reach * (-s / f).exp(), s);
                }
                // How far on the lane after the band reaches `q`, if it does.
                let hits = |q: DVec2| {
                    band.iter()
                        .filter(|&&(p, n, t, lo, hi, _)| {
                            let rel = q - p;
                            let y = rel.dot(n);
                            let dy = if y < lo {
                                lo - y
                            } else if y > hi {
                                y - hi
                            } else {
                                0.0
                            };
                            rel.dot(t).hypot(dy) < WAITING + 0.5 * STEP
                        })
                        .map(|b| b.5)
                        .reduce(f64::max)
                };
                for &w in waiting.iter().filter(|&&w| w != conn.from && w != conn.to) {
                    let wl = &lanes[w as usize].line;
                    let reach = wl.length().min(HOLD_BACK_MAX + STOP_LINE);
                    let (mut far, mut after) = (None, 0.0f64);
                    for k in 0..=(reach / STEP) as usize {
                        let d = k as f64 * STEP;
                        if let Some(a) = hits(wl.point_at(wl.length() - d).truncate()) {
                            far = Some(d);
                            after = after.max(a);
                        }
                    }
                    let Some(d) = far else { continue };
                    let back = (d + MARGIN - STOP_LINE).clamp(0.0, HOLD_BACK_MAX);
                    let list = &mut sweeps.connectors[c];
                    match list.iter_mut().find(|x| x.lane == w) {
                        Some(x) => {
                            x.back = x.back.max(back);
                            x.after = x.after.max(after + STEP);
                        }
                        None => {
                            list.push(Sweep { lane: w, back, after: after + STEP });
                            sweeps.lanes[w as usize].push(c as u32);
                        }
                    }
                    let b = &mut sweeps.hold_back[w as usize];
                    *b = b.max(back);
                }
            }
        }
        sweeps
    }
}

/// State of a `traffic` driver.
#[derive(Clone, Debug)]
pub struct TrafficDriver {
    spec: TrafficDriverSpec,
    geometry: DriverGeometry,
    rng: SimRng,
    /// For the encounters with learning agents (its own stream, so that they do not change
    /// the other draws).
    attention_rng: SimRng,
    /// Learning agents met (within [`ENCOUNTER`] m, until beyond [`ENCOUNTER_END`] m) and
    /// whether it noticed them.
    pub encounters: Vec<(u32, bool)>,
    /// Drawn per episode.
    pub factor: f64,
    /// Car-following parameters; `v0` is the desired speed of the last step (after the
    /// preview of limits and bends).
    pub idm: Idm,
    pub politeness: f64,
    pub change_time: f64,
    /// Where it is on the lane graph (`None`: off the lanes; it stands still).
    pub place: Option<Place>,
    /// The elements it came along before its place's (the last one last; up to [`TRAIL`]).
    pub trail: Vec<Elem>,
    /// Long vehicles: the part of the lane graph they keep to.
    network: Option<Arc<Network>>,
    /// The next connectors to take (up to two).
    pub plan: Vec<u32>,
    /// Connectors to take before choosing at random, in order (see [`TrafficDriver::take_route`]).
    pub fixed: Vec<u32>,
    pub change: Option<LaneChange>,
    /// Time since the last lane change ended (s), and waiting at a lane's end (s).
    pub since_change: f64,
    pub stuck: f64,
    /// Lane changes made this episode.
    pub changes: u32,
    /// Smallest gap (s) accepted when giving way.
    pub critical_gap: f64,
    /// The connector it is committed to while still before it.
    pub granted: Option<u32>,
    /// The lane at whose stop sign it has stopped.
    pub stopped_at: Option<u32>,
    /// Time waiting at a line (s), standing still (s) and lost off the lanes (s).
    pub waiting: f64,
    pub standing: f64,
    pub lost: f64,
    /// Times put back elsewhere this episode.
    pub respawns: u32,
    /// A bus's line (see [`TrafficDriverSpec::bus`]).
    pub bus: Option<BusRoute>,
    /// Times a bus left its loop and took a new one since its (re)spawn.
    pub reroutes: u32,
    /// Desired speed drawn from [`TrafficDriverSpec::speed`] (m/s).
    pub cruise: Option<f64>,
    /// Lateral offset it steers to from its line (m, positive to the left), as of its last
    /// [`locate`](Self::locate); a car's shift out of its lane line while overtaking.
    pub offset: f64,
    pub shift: f64,
    /// The cyclist it is passing, and the overtakes begun this episode.
    pub overtake: Option<Overtake>,
    pub overtakes: u32,
}

/// The loop of a bus: connectors (movements, made from the lane of each road nearest to the
/// bus; [`TrafficDriver::equivalent`]) and stops on the lanes between them.
///
/// It is drawn inside the bus's [`Network`] (or among all but U-turns): the shortest ways
/// over lanes, connectors and the lane changes the bus can make before each connector (by
/// length, a lane change counting [`CHANGE_ROOM`] m) from its lane through two random others
/// and back, the one of eight draws nearest to a length drawn from [`BusSpec::length`]. Stops lie halfway along
/// lanes at least [`STOP_LANE`] m long, the first after [`BusSpec::stop_spacing`] m, each
/// next at least that far on. The bus stops there in its lane (with its front at the stop)
/// and stands for a time drawn from [`BusSpec::dwell`]. When it takes another movement
/// (it could not reach the lane of the planned one) it draws a new loop from where it is.
#[derive(Clone, Debug, PartialEq)]
pub struct BusRoute {
    pub connectors: Vec<u32>,
    /// Per connector: the stop on the lane after it (station as a fraction of its length).
    pub stops: Vec<Option<f64>>,
    /// Its length (m, along the lanes after the connectors and the connectors).
    pub length: f64,
    /// Index of the next connector to take.
    pub next: usize,
    /// Standing at a stop: time stood and time to stand (s).
    pub dwelling: Option<(f64, f64)>,
    /// The stop served last (the index of the connector before it).
    pub served: Option<usize>,
    /// Connectors taken along it, and stops served and passed by, since it was drawn.
    pub taken: u32,
    pub stops_served: u32,
    pub stops_missed: u32,
}

impl TrafficDriver {
    pub fn new(spec: &TrafficDriverSpec, geometry: DriverGeometry) -> Self {
        let mut d = Self {
            spec: spec.clone(),
            geometry,
            rng: Seed::from_u64(0).rng(),
            attention_rng: Seed::from_u64(0).rng(),
            encounters: Vec::new(),
            factor: 1.0,
            idm: Idm::OTHER,
            politeness: 0.0,
            change_time: 4.0,
            place: None,
            trail: Vec::new(),
            network: None,
            plan: Vec::new(),
            fixed: Vec::new(),
            change: None,
            since_change: COOLDOWN,
            stuck: 0.0,
            changes: 0,
            critical_gap: 5.0,
            granted: None,
            stopped_at: None,
            waiting: 0.0,
            standing: 0.0,
            lost: 0.0,
            respawns: 0,
            bus: None,
            reroutes: 0,
            cruise: None,
            offset: 0.0,
            shift: 0.0,
            overtake: None,
            overtakes: 0,
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

    /// Whether it keeps to a [`Network`] (vehicles longer than [`SWING_LENGTH`]).
    pub fn is_long(&self) -> bool {
        self.geometry.tracking > SWING_LENGTH
    }

    /// Set the network it keeps to on the current map (see [`Network`]).
    pub fn set_network(&mut self, network: Option<Arc<Network>>) {
        self.network = network;
    }

    pub fn network(&self) -> Option<&Network> {
        self.network.as_deref()
    }

    /// Whether it may take connector `c`.
    fn allowed(&self, c: u32) -> bool {
        self.network.as_ref().is_none_or(|n| n.connectors[c as usize])
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
        self.critical_gap = pick(s.critical_gap);
        let cruise = s.speed.map(&mut pick);
        self.idm = Idm { v0: self.idm.v0, headway, min_gap: s.min_gap, accel, decel };
        self.cruise = cruise;
    }

    /// Start an episode with a new random stream at `pose`.
    pub fn reset(&mut self, seed: Seed, world: &StaticWorld, pose: &Pose) {
        self.rng = seed.rng();
        self.attention_rng = seed.child("attention").rng();
        self.encounters.clear();
        self.draw();
        self.place = None;
        self.trail.clear();
        self.plan.clear();
        self.fixed.clear();
        self.change = None;
        self.since_change = COOLDOWN;
        self.stuck = 0.0;
        self.changes = 0;
        self.granted = None;
        self.stopped_at = None;
        self.waiting = 0.0;
        self.standing = 0.0;
        self.lost = 0.0;
        self.respawns = 0;
        self.bus = None;
        self.reroutes = 0;
        self.shift = 0.0;
        self.overtake = None;
        self.overtakes = 0;
        self.locate(world, pose);
        if self.spec.bus.is_some()
            && let Some(p) = self.place
        {
            let g = world.roads().lanes();
            self.bus = self.bus_route(g, p.elem.lane_after(g));
            self.plan.clear();
            self.fill_plan(g, world);
        }
    }

    /// Take `connectors` next, in order (each, or the same movement from the lane it is
    /// in; [`TrafficDriver::equivalent`]), then choose at random again (e.g. a junction
    /// crossing's route; see [`GoalKind::Junction`](crate::scenario::GoalKind::Junction)).
    pub fn take_route(&mut self, connectors: &[u32], world: &StaticWorld) {
        self.fixed = connectors.to_vec();
        self.plan.clear();
        self.fill_plan(world.roads().lanes(), world);
    }

    /// A new loop for a bus from `lane` (see [`BusRoute`]).
    fn bus_route(&mut self, g: &LaneGraph, lane: u32) -> Option<BusRoute> {
        let spec = self.spec.bus.clone()?;
        let lanes = g.lanes();
        let n = lanes.len();
        let step = |c: u32| {
            let conn = &g.connectors()[c as usize];
            conn.line.length() + lanes[conn.to as usize].line.length()
        };
        // Lanes on: by a connector, or by lane changes along the lane ([`u32::MAX`]).
        let mut out: Vec<Vec<(u32, u32, f64)>> = vec![Vec::new(); n];
        for (c, conn) in g.connectors().iter().enumerate() {
            if conn.turn != Turn::UTurn && self.allowed(c as u32) {
                out[conn.from as usize].push((conn.to, c as u32, step(c as u32)));
            }
        }
        for l in 0..n as u32 {
            for k in Self::group(g, l).into_iter().filter(|&k| k != l && Self::reachable(g, l, k)) {
                out[l as usize].push((k, u32::MAX, CHANGE_ROOM));
            }
        }
        // Shortest way (its connectors) from lane `a` to lane `b` (a ≠ b), by Dijkstra.
        let way = |a: u32, b: u32| -> Option<Vec<u32>> {
            let mut dist = vec![f64::INFINITY; n];
            let mut via = vec![(u32::MAX, u32::MAX); n];
            let mut heap = std::collections::BinaryHeap::new();
            dist[a as usize] = 0.0;
            heap.push((std::cmp::Reverse(OrdF64(0.0)), a));
            while let Some((std::cmp::Reverse(OrdF64(d)), u)) = heap.pop() {
                if d > dist[u as usize] {
                    continue;
                }
                if u == b {
                    break;
                }
                for &(v, c, cost) in &out[u as usize] {
                    let nd = d + cost;
                    if nd < dist[v as usize] {
                        dist[v as usize] = nd;
                        via[v as usize] = (u, c);
                        heap.push((std::cmp::Reverse(OrdF64(nd)), v));
                    }
                }
            }
            if !dist[b as usize].is_finite() {
                return None;
            }
            let mut path = Vec::new();
            let mut at = b;
            while at != a {
                let (u, c) = via[at as usize];
                if c != u32::MAX {
                    path.push(c);
                }
                at = u;
            }
            path.reverse();
            Some(path)
        };
        let start = lane;
        let candidates: Vec<u32> =
            (0..n as u32).filter(|&l| out[l as usize].iter().any(|&(_, c, _)| c != u32::MAX)).collect();
        if candidates.is_empty() {
            return None;
        }
        let target = if spec.length[1] > spec.length[0] {
            self.rng.range(spec.length[0], spec.length[1])
        } else {
            spec.length[0]
        };
        let mut best: Option<(f64, Vec<u32>)> = None;
        for _ in 0..8 {
            let pick = |r: &mut SimRng| {
                candidates[((r.uniform() * candidates.len() as f64) as usize).min(candidates.len() - 1)]
            };
            let (w1, w2) = (pick(&mut self.rng), pick(&mut self.rng));
            if w1 == start || w2 == w1 || w2 == start {
                continue;
            }
            let (Some(a), Some(b), Some(c)) = (way(start, w1), way(w1, w2), way(w2, start)) else { continue };
            let route: Vec<u32> = a.into_iter().chain(b).chain(c).collect();
            let length: f64 = route.iter().map(|&c| step(c)).sum();
            if best.as_ref().is_none_or(|(l, _)| (length - target).abs() < (l - target).abs()) {
                best = Some((length, route));
            }
        }
        let (length, connectors) = best?;
        let mut stops = vec![None; connectors.len()];
        let mut since = 0.0;
        for (k, &c) in connectors.iter().enumerate() {
            let to = lanes[g.connectors()[c as usize].to as usize].line.length();
            since += g.connectors()[c as usize].line.length();
            if to >= STOP_LANE && since + 0.5 * to >= spec.stop_spacing {
                stops[k] = Some(0.5);
                since = 0.5 * to;
            } else {
                since += to;
            }
        }
        Some(BusRoute {
            connectors,
            stops,
            length,
            next: 0,
            dwelling: None,
            served: None,
            taken: 0,
            stops_served: 0,
            stops_missed: 0,
        })
    }

    /// A bus taking connector `c`: on along its loop, or a new loop from after `c`.
    fn bus_took(&mut self, g: &LaneGraph, c: u32) {
        let Some(route) = &self.bus else {
            // (One without a loop tries again.)
            if self.spec.bus.is_some() {
                self.plan.clear();
                self.bus = self.bus_route(g, g.connectors()[c as usize].to);
            }
            return;
        };
        let planned = route.connectors[route.next];
        if same_movement(g, c, planned) {
            let route = self.bus.as_mut().expect("a bus");
            route.next = (route.next + 1) % route.connectors.len();
            route.taken += 1;
        } else {
            self.reroutes += 1;
            self.plan.clear();
            self.bus = self.bus_route(g, g.connectors()[c as usize].to);
        }
    }

    /// A bus at its stops: the stop ahead on its lane as a standing obstacle, standing
    /// there for the drawn time.
    fn bus_stop(&mut self, g: &LaneGraph, p: Place, v: f64, idm: &Idm, dt: f64) -> f64 {
        let front = self.geometry.front;
        let Some(dwell) = self.spec.bus.as_ref().map(|b| b.dwell) else { return f64::INFINITY };
        let Some(route) = self.bus.as_mut() else { return f64::INFINITY };
        let n = route.connectors.len();
        let leg = (route.next + n - 1) % n;
        let Elem::Lane(l) = p.elem else { return f64::INFINITY };
        let (Some(f), true) = (route.stops[leg], route.served != Some(leg)) else { return f64::INFINITY };
        let to = g.connectors()[route.connectors[leg] as usize].to;
        if TrafficDriver::group(g, to)[0] != TrafficDriver::group(g, l)[0] {
            return f64::INFINITY;
        }
        let gap = f * g.lanes()[l as usize].line.length() - p.station - front;
        if let Some((stood, time)) = &mut route.dwelling {
            *stood += dt;
            if *stood >= *time {
                route.dwelling = None;
                route.served = Some(leg);
                route.stops_served += 1;
                return f64::INFINITY;
            }
            return hold(idm, v, 0.0).min(0.0);
        }
        if gap < -STOP_MISSED {
            route.served = Some(leg);
            route.stops_missed += 1;
            return f64::INFINITY;
        }
        if gap < STOP_REACHED && v < STANDING {
            let time = if dwell[1] > dwell[0] { self.rng.range(dwell[0], dwell[1]) } else { dwell[0] };
            self.bus.as_mut().expect("a bus").dwelling = Some((0.0, time));
            return hold(idm, v, 0.0).min(0.0);
        }
        hold(idm, v, gap)
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
    fn equivalent(g: &LaneGraph, c: u32, lane: u32, ok: &dyn Fn(u32) -> bool) -> Option<u32> {
        let cc = &g.connectors()[c as usize];
        let target = &g.lanes()[cc.to as usize];
        let here = i32::from(g.lanes()[lane as usize].index);
        let mut best: Option<(i32, i32, u32)> = None;
        for l in Self::group(g, lane).into_iter().filter(|&l| Self::reachable(g, lane, l)) {
            let from = &g.lanes()[l as usize];
            for &k in &from.successors {
                let kc = &g.connectors()[k as usize];
                let to = &g.lanes()[kc.to as usize];
                if kc.turn == cc.turn && to.road == target.road && to.dir == target.dir && ok(k) {
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
            match Self::equivalent(g, self.plan[k], base, &|c| self.allowed(c)) {
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
    /// else leaves, and bends tighter than the steering lock only when nothing else fits (then
    /// the least bent).
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
        // Long vehicles: within their network, if it leaves from here.
        let kept: Vec<u32> = all.iter().copied().filter(|&c| self.allowed(c)).collect();
        let all = if kept.is_empty() { all } else { kept };
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
        // Within the steering lock, or else the least bent.
        let bent = |c: u32| bend(&g.connectors()[c as usize].line);
        let fit: Vec<u32> = options.iter().copied().filter(|&c| bent(c) <= self.geometry.max_curvature).collect();
        let options = if fit.is_empty() {
            options.iter().copied().min_by(|&a, &b| bent(a).total_cmp(&bent(b))).into_iter().collect()
        } else {
            fit
        };
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
        // Buses: along their loop; a new one when its next movement cannot be made from here
        // (and when that one cannot either, its first movement all the same).
        let mut fresh = false;
        while self.bus.is_some() && self.plan.len() < 2 {
            let route = self.bus.as_ref().expect("a bus");
            let c = route.connectors[(route.next + self.plan.len()) % route.connectors.len()];
            match Self::equivalent(g, c, lane, &|c| self.allowed(c)) {
                Some(c) => {
                    self.plan.push(c);
                    lane = g.connectors()[c as usize].to;
                }
                None if fresh => {
                    self.plan.push(c);
                    lane = g.connectors()[c as usize].to;
                }
                None => {
                    self.reroutes += 1;
                    self.plan.clear();
                    lane = p.elem.lane_after(g);
                    self.bus = self.bus_route(g, lane);
                    fresh = true;
                }
            }
        }
        while self.plan.len() < 2 && !self.fixed.is_empty() {
            let c = self.fixed.remove(0);
            let c = Self::equivalent(g, c, lane, &|c| self.allowed(c)).unwrap_or(c);
            self.plan.push(c);
            lane = g.connectors()[c as usize].to;
        }
        while self.plan.len() < 2 {
            // (Cyclists keep to their lane.)
            let Some(c) = self.choose(g, world, lane, self.geometry.single_track) else { break };
            let c = Self::equivalent(g, c, lane, &|c| self.allowed(c)).unwrap_or(c);
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
            let mut passed = Vec::new();
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
                            self.bus_took(g, c);
                            self.change = None;
                            Elem::Connector(c)
                        }
                        Elem::Connector(c) => Elem::Lane(g.connectors()[c as usize].to),
                    };
                    passed.push(p.elem);
                    p = Place { elem: next, station: 0.0 };
                    continue;
                }
                break;
            }
            // (From the path it steers to: cyclists ride off the line.)
            let line = p.elem.line(g);
            let side = self.lateral(world, p.elem, p.station) * DVec2::from_angle(line.heading_at(p.station)).perp();
            let off = (line.point_at(p.station).truncate() + side).distance(xy);
            // (Long vehicles swing wide, and run wider still on bends beyond their lock.)
            let long = if self.geometry.tracking > SWING_LENGTH { SWING_MAX + LOST_WIDE } else { 0.0 };
            let allowed = LOST + self.change.map_or(0.0, |_| 4.0) + long;
            if off <= allowed {
                self.place = Some(p);
                self.offset = self.lateral(world, p.elem, p.station);
                self.trail.extend(passed);
                let extra = self.trail.len().saturating_sub(TRAIL);
                self.trail.drain(..extra);
                self.fill_plan(g, world);
                return;
            }
        }
        // Lost (or new): the nearest lane or connector along the heading.
        self.plan.clear();
        self.trail.clear();
        self.change = None;
        self.place = find(g, xy, yaw(pose.rot));
        self.offset = self.place.map_or(0.0, |p| self.lateral(world, p.elem, p.station));
        self.fill_plan(g, world);
    }

    /// Lateral offset (m, positive to the left) it steers to from the line of `elem` at
    /// `station`: cyclists keep right ([`keep_right`]; on connectors, blended from the lane
    /// before to the lane after), cars by their overtaking shift.
    fn lateral(&self, world: &StaticWorld, elem: Elem, station: f64) -> f64 {
        if !self.geometry.single_track {
            return self.shift;
        }
        match elem {
            Elem::Lane(l) => keep_right(world, l),
            Elem::Connector(c) => {
                let g = world.roads().lanes();
                let conn = &g.connectors()[c as usize];
                let t = (station / conn.line.length().max(1e-9)).clamp(0.0, 1.0);
                let w = t * t * (3.0 - 2.0 * t);
                (1.0 - w) * keep_right(world, conn.from) + w * keep_right(world, conn.to)
            }
        }
    }

    /// Its extent across its lane, for passing beside others there (see [`Side`]).
    fn side(&self, g: &LaneGraph, lane: Elem) -> Side {
        Side {
            offset: self.offset,
            half_width: self.geometry.half_width,
            cyclist: self.geometry.single_track,
            until: lane.line(g).length() - PASS_END,
            overtaken: self.overtake.map(|o| o.agent),
        }
    }

    /// Its entries in the index: at its place, in the lane it is leaving while changing, and
    /// its grant.
    pub fn occupy(&self, g: &LaneGraph, index: &mut LaneIndex, agent: u32, pose: &Pose, speed: f64) {
        let Some(p) = self.place else { return };
        let geo = self.geometry;
        let idm = Idm { v0: self.desired(g), ..self.idm };
        let next = [0, 1].map(|k| self.plan.get(k).copied().unwrap_or(ANY));
        let waiting = self.waiting > 1.0;
        let o = Occupant {
            agent,
            station: p.station,
            front: geo.front,
            rear: -geo.rear,
            speed,
            idm,
            next,
            waiting,
            pos: pose.pos.truncate(),
            heading: yaw(pose.rot),
            half_width: geo.half_width,
            articulated: geo.articulated,
            long: self.is_long(),
            via: match (p.elem, self.trail.last()) {
                (Elem::Lane(_), Some(&Elem::Connector(c))) => c,
                _ => ANY,
            },
            learner: false,
            tail: false,
            offset: self.offset,
            cyclist: geo.single_track,
        };
        index.insert(p.elem, o);
        // Its tail on the elements behind, as far back as it reaches.
        let mut station = p.station;
        for &e in self.trail.iter().rev() {
            if station - o.rear >= 0.0 {
                break;
            }
            station += e.line(g).length();
            index.insert(e, Occupant { station, tail: true, ..o });
        }
        if let (Some(c), Elem::Lane(l)) = (self.change, p.elem) {
            let s = p.station * g.lanes()[c.from as usize].line.length() / g.lanes()[l as usize].line.length();
            index.insert(Elem::Lane(c.from), Occupant { station: s, ..o });
        }
        if let Some(c) = self.granted {
            // Its front's distance to the connector: from its lane, or the connector before.
            let from = g.connectors()[c as usize].from;
            let left = p.elem.line(g).length() - p.station - geo.front;
            let distance = match p.elem {
                Elem::Lane(l) if l == from => Some(left),
                Elem::Connector(k) if g.connectors()[k as usize].to == from => {
                    Some(left + g.lanes()[from as usize].line.length())
                }
                _ => None,
            };
            if let Some(d) = distance {
                index.grant(
                    c,
                    Grant {
                        agent,
                        distance: d.max(0.0),
                        speed,
                        length: geo.front - geo.rear,
                        long: self.is_long(),
                        accel: self.idm.accel,
                        v0: self.idm.v0,
                    },
                );
            }
        }
    }

    /// Whether it should be put back elsewhere (see the module notes).
    pub fn wants_respawn(&self) -> bool {
        self.lost > LOST_RESPAWN || self.spec.respawn > 0.0 && self.standing > self.spec.respawn
    }

    /// Desired speed at the place, before the preview (m/s).
    pub fn desired(&self, g: &LaneGraph) -> f64 {
        self.place.map_or(0.0, |p| self.limited(p.elem.speed_limit(g)))
    }

    /// Desired speed under speed limit `limit` (m/s).
    fn limited(&self, limit: f64) -> f64 {
        let v = self.factor * limit;
        self.cruise.map_or(v, |c| c.min(v))
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
    /// station `from` (the driver's own reference point there), but for those it passes
    /// beside on its lane (see [`Side`]).
    #[allow(clippy::too_many_arguments)]
    fn follow(
        &self,
        g: &LaneGraph,
        index: &LaneIndex,
        chain: &[(Elem, f64)],
        from: f64,
        me: u32,
        v: f64,
        idm: &Idm,
    ) -> f64 {
        let side = chain.first().map(|&(e, _)| self.side(g, e));
        let leader = index.ahead(chain, from, me, side).map(|(d, o)| (d - self.geometry.front - o.rear, v - o.speed));
        idm.accel(v, leader)
    }

    /// The junction rules (see the module notes) at place `p` along `chain`: the acceleration
    /// they allow (infinite when they do not hold the driver back). Takes and gives up the
    /// grant.
    #[allow(clippy::too_many_arguments)]
    fn junction(
        &mut self,
        g: &LaneGraph,
        index: &mut LaneIndex,
        chain: &[(Elem, f64)],
        p: Place,
        me: u32,
        v: f64,
        idm: &Idm,
        signals: &Signals,
        t: f64,
        dt: f64,
    ) -> f64 {
        let geo = self.geometry;
        let reach = v * v / (2.0 * self.idm.decel) + 10.0;
        // Inside a connector: stop before conflict zones that others occupy.
        let mut acc = f64::INFINITY;
        if let Elem::Connector(c) = p.elem {
            if let Some(d) = self.occupied_zone(g, index, c, p.station + geo.front, me, reach) {
                acc = hold(idm, v, d - STOP_LINE);
            }
            acc = acc.min(self.shared_leader(g, index, c, -p.station, me, v, idm));
        }
        // The next connector ahead (the chain follows the plan), and the lane it leaves.
        let Some(k) = chain.iter().skip(1).position(|(e, _)| matches!(e, Elem::Connector(_))).map(|k| k + 1) else {
            self.granted = None;
            self.waiting = 0.0;
            return acc;
        };
        let (Elem::Connector(c), base) = chain[k] else { unreachable!("a connector") };
        let Elem::Lane(lane) = chain[k - 1].0 else { unreachable!("connectors leave lanes") };
        if self.stopped_at.is_some_and(|x| x != lane) {
            self.stopped_at = None;
        }
        if self.granted.is_some_and(|x| x != c) {
            self.granted = None;
        }
        let dl = base - geo.front;
        let back = index.hold_back(lane);
        acc = acc.min(self.shared_leader(g, index, c, base, me, v, idm));
        let granted = self.granted == Some(c);
        if dl > DECIDE.max(reach + 2.0 * v) && !granted {
            self.waiting = 0.0;
            return acc;
        }
        if self.may_enter(g, index, chain, k, dl, me, v, idm, signals, t, lane, false)
            && (granted || self.through(g, index, chain, k, me, v, idm, signals, t))
        {
            if granted || dl < GRANT.max(3.0 * v) {
                self.granted = Some(c);
                index.grant(
                    c,
                    Grant {
                        agent: me,
                        distance: dl.max(0.0),
                        speed: v,
                        length: geo.front - geo.rear,
                        long: self.is_long(),
                        accel: self.idm.accel,
                        v0: self.idm.v0,
                    },
                );
            }
            self.waiting = 0.0;
        } else {
            self.granted = None;
            acc = acc.min(hold(idm, v, self.stop_distance(dl, back, v)));
            self.waiting = if v < STANDING && dl < 5.0 + back { self.waiting + dt } else { 0.0 };
        }
        if g.control(lane) == Control::Stop && v < STANDING && dl < 3.0 + back {
            self.stopped_at = Some(lane);
        }
        // Even when allowed through: clear of long vehicles sweeping past the line now
        // (unless already past where it would wait).
        for &c2 in &index.sweeps.lanes[lane as usize] {
            let sweep = index.sweeps.connectors[c2 as usize].iter().find(|x| x.lane == lane).expect("listed both ways");
            let gap = dl - STOP_LINE - sweep.back;
            if gap > -0.5 && index.sweeping(g, c2, sweep) {
                acc = acc.min(hold(idm, v, gap));
            }
        }
        // Committed: stop before conflict zones that others occupy.
        if self.granted == Some(c)
            && let Some(d) = self.occupied_zone(g, index, c, -dl, me, reach)
        {
            acc = acc.min(hold(idm, v, dl + d - STOP_LINE));
        }
        // Committed: braking in time for lights beyond lanes too short to wait on.
        if self.granted == Some(c) {
            let mut j = k;
            while let (Some(&(Elem::Lane(l), _)), Some(&(Elem::Connector(c2), base))) =
                (chain.get(j + 1), chain.get(j + 2))
                && !self.can_wait_on(g, index, l)
            {
                let (d2, back) = (base - geo.front, index.hold_back(l));
                let (light, left) = signals.light_left(g, c2, t);
                if self.stops_at(g, c2, d2, back, v, light, left) {
                    acc = acc.min(hold(idm, v, self.stop_distance(d2, back, v)));
                    break;
                }
                j += 2;
            }
        }
        acc
    }

    /// The lane length a vehicle waits on at a line, clear of the junction before.
    fn room_needed(&self) -> f64 {
        self.geometry.front - self.geometry.rear + self.idm.min_gap
    }

    /// Whether `lane` is long enough to wait on before its (held back) line.
    fn can_wait_on(&self, g: &LaneGraph, index: &LaneIndex, lane: u32) -> bool {
        g.lanes()[lane as usize].line.length() - index.hold_back(lane) >= self.room_needed()
    }

    /// Distance (m) to where a driver, its front `dl` m before a connector at `v`, stops for
    /// it: at its line held `back` m, or, when it cannot stop there at `safe_decel` (and is
    /// not standing at it), at the junction's edge.
    fn stop_distance(&self, dl: f64, back: f64, v: f64) -> f64 {
        let held = dl - back - STOP_LINE;
        let hard = self.spec.safe_decel.max(self.idm.decel);
        if v * v <= 2.0 * hard * held.max(0.0) + STANDING * STANDING { held } else { dl - STOP_LINE }
    }

    /// Whether a driver not yet allowed into connector `c`, its front `dl` m before it, stops
    /// for the `light` there (`left` s) at its line held `back` m (see
    /// [`stop_distance`](Self::stop_distance)): always at red; at amber if comfortable, or
    /// when it would not get through before red (the reference point crossing the line, no
    /// faster than the connector and the speed it plans allow) and can stop.
    #[allow(clippy::too_many_arguments)]
    fn stops_at(&self, g: &LaneGraph, c: u32, dl: f64, back: f64, v: f64, light: Light, left: f64) -> bool {
        let need = v * v / (2.0 * self.stop_distance(dl, back, v).max(0.1));
        match light {
            Light::Red => true,
            Light::Amber => {
                let pass = v.min(self.idm.v0).min(self.factor * g.connectors()[c as usize].speed);
                let clears = dl + self.geometry.front < pass * left - 0.2;
                need <= self.idm.decel || !clears && need <= self.spec.safe_decel.max(self.idm.decel)
            }
            Light::Green => false,
        }
    }

    /// Whether the driver, allowed into connector `chain[k]`, may also go on through the
    /// connectors after it that lanes too short to wait on lead to (see the module notes).
    #[allow(clippy::too_many_arguments)]
    fn through(
        &self,
        g: &LaneGraph,
        index: &LaneIndex,
        chain: &[(Elem, f64)],
        k: usize,
        me: u32,
        v: f64,
        idm: &Idm,
        signals: &Signals,
        t: f64,
    ) -> bool {
        let mut j = k;
        while let (Some(&(Elem::Lane(l), _)), Some(&(Elem::Connector(_), base))) = (chain.get(j + 1), chain.get(j + 2))
        {
            if self.can_wait_on(g, index, l) {
                return true;
            }
            let dl = base - self.geometry.front;
            if !self.may_enter(g, index, chain, j + 2, dl, me, v, idm, signals, t, l, true) {
                return false;
            }
            j += 2;
        }
        true
    }

    /// Whether the driver may enter connector `chain[k]` (its front `dl` m before it) from
    /// `lane` (see the module notes); a granted driver only gives up when it can stop.
    /// `ahead`: asked beforehand for a connector after the next one (its light and stop sign
    /// left out).
    #[allow(clippy::too_many_arguments)]
    fn may_enter(
        &self,
        g: &LaneGraph,
        index: &LaneIndex,
        chain: &[(Elem, f64)],
        k: usize,
        dl: f64,
        me: u32,
        v: f64,
        idm: &Idm,
        signals: &Signals,
        t: f64,
        lane: u32,
        ahead: bool,
    ) -> bool {
        let (Elem::Connector(c), _) = chain[k] else { unreachable!("a connector") };
        let geo = self.geometry;
        let b = self.idm.decel;
        let hard = self.spec.safe_decel.max(b);
        let granted = self.granted == Some(c);
        // Deceleration to stop at the line.
        let back = index.hold_back(lane);
        let need = v * v / (2.0 * self.stop_distance(dl, back, v).max(0.1));
        // (Asked ahead, the light is left out: it only delays.)
        let (light, left) = if ahead { (Light::Green, f64::INFINITY) } else { signals.light_left(g, c, t) };
        match light {
            Light::Red => return granted && need > hard,
            Light::Amber => {
                if self.stops_at(g, c, dl, back, v, light, left) {
                    return false;
                }
                if granted {
                    return true;
                }
            }
            Light::Green => {}
        }
        // From here on measured from the line.
        let dl = dl - back;
        if granted {
            return need > b || self.room(index, chain, k, me);
        }
        // Long vehicles: not while others stand in the band they would sweep, or could not
        // stop short of it.
        if self.is_long()
            && index.sweeps.connectors[c as usize].iter().any(|sw| {
                let end = g.lanes()[sw.lane as usize].line.length() - STOP_LINE - sw.back;
                index.lanes[sw.lane as usize].iter().any(|o| {
                    !index.skips(o, me)
                        && !o.tail
                        && o.station + o.front + o.speed * o.speed / (2.0 * SWEEP_BRAKE) > end + 0.3
                })
            })
        {
            return false;
        }
        if !ahead && g.control(lane) == Control::Stop && self.stopped_at != Some(lane) {
            return false;
        }
        if !self.room(index, chain, k, me) {
            return false;
        }
        // Conflicts: its own passage of each zone from now, at full acceleration.
        let a = idm.accel.max(0.5);
        let vmax = (self.factor * g.connectors()[c as usize].speed).max(1.0);
        let length = geo.front - geo.rear;
        let deadlock = self.waiting > self.spec.deadlock;
        for conf in g.connectors()[c as usize].conflicts.iter().filter(|x| x.kind != ConflictKind::Diverge) {
            // The zone along both (with a margin), entered with the front, left with the rear.
            let (sc, ec) = (conf.station - ZONE, conf.station + conf.length + ZONE);
            let (so, eo) = (conf.other_station - ZONE, conf.other_station + conf.other_length + ZONE);
            let t_in = travel_time(dl + sc, v, a, vmax);
            let t_out = travel_time(dl + ec + length, v, a, vmax);
            let clash = |ti: f64, to: f64| ti < t_out + TIME_MARGIN && to > t_in - TIME_MARGIN;
            // On the other connector, not yet through the zone (unless standing clear of the
            // path).
            for o in index.list(Elem::Connector(conf.other)) {
                let (front, rear) = (o.station + o.front, o.station - o.rear);
                let (line, other) = (&g.connectors()[c as usize].line, &g.connectors()[conf.other as usize].line);
                if index.skips(o, me) || rear > eo || self.passes(line, other, eo, o) {
                    continue;
                }
                // (It may speed up again: held for a zone, or setting off.)
                let ti =
                    if front >= so { 0.0 } else { travel_time(so - front, o.speed, Idm::OTHER.accel, Idm::OTHER.v0) };
                let to = if o.speed < STANDING { f64::INFINITY } else { (eo - rear) / o.speed };
                if clash(ti, to) {
                    return false;
                }
            }
            // Committed to it.
            for gr in index.grants[conf.other as usize].iter().filter(|gr| gr.agent != me) {
                // (In at the earliest, out at its own pace: a bus is through later than a car.)
                let ti =
                    travel_time(gr.distance + so, gr.speed, Idm::OTHER.accel.max(gr.accel), Idm::OTHER.v0.max(gr.v0));
                let to = travel_time(gr.distance + eo + gr.length, gr.speed, gr.accel, gr.v0);
                if clash(ti, to) {
                    return false;
                }
            }
            // Giving way: a gap in the traffic approaching it (unless that faces red).
            if conf.yields && signals.light(g, conf.other, t) != Light::Red {
                let gap = self.critical_gap.max(t_out + TIME_MARGIN);
                let mut blocked = false;
                index.approaching(g, conf.other, LOOK_BACK, me, &mut |d, o| {
                    if blocked || deadlock && o.waiting {
                        return;
                    }
                    let ti = if o.speed < STANDING {
                        travel_time(d + so, 0.0, o.idm.accel, o.idm.v0)
                    } else {
                        (d + so).max(0.0) / o.speed
                    };
                    blocked = ti < gap;
                });
                if blocked {
                    return false;
                }
            }
        }
        true
    }

    /// Whether there is room beyond connector `chain[k]`: the first vehicle past its start
    /// leaves the driver's length and standstill gap on the lane after it, or moves on.
    fn room(&self, index: &LaneIndex, chain: &[(Elem, f64)], k: usize, me: u32) -> bool {
        let Some(&(_, after)) = chain.get(k + 1) else { return true };
        let need = self.room_needed();
        index.ahead(&chain[k..], -1.0, me, None).is_none_or(|(d, o)| o.speed > 3.0 || d - o.rear - after >= need)
    }

    /// IDM acceleration behind the vehicles ahead on connectors sharing a stretch with
    /// connector `c` (its start `base` m ahead of the driver's reference point): diverging ones
    /// (measured from the common start) before they have parted, merging ones (measured to the
    /// common end) once inside the zone.
    #[allow(clippy::too_many_arguments)]
    fn shared_leader(&self, g: &LaneGraph, index: &LaneIndex, c: u32, base: f64, me: u32, v: f64, idm: &Idm) -> f64 {
        let geo = self.geometry;
        let len = g.connectors()[c as usize].line.length();
        let mut acc = f64::INFINITY;
        for conf in &g.connectors()[c as usize].conflicts {
            let other = g.connectors()[conf.other as usize].line.length();
            for o in index.list(Elem::Connector(conf.other)).iter().filter(|o| !index.skips(o, me)) {
                let gap = match conf.kind {
                    ConflictKind::Diverge => {
                        let end = conf.other_station + conf.other_length + ZONE;
                        if o.station <= -base || o.station - o.rear > end {
                            continue;
                        }
                        base + o.station - o.rear - geo.front
                    }
                    ConflictKind::Merge => {
                        // Distances to the common end.
                        let mine = base + len - geo.front;
                        if o.station + o.front < conf.other_station - ZONE || other - o.station - o.front >= mine {
                            continue;
                        }
                        mine - (other - o.station + o.rear)
                    }
                    _ => continue,
                };
                acc = acc.min(idm.accel(v, Some((gap.max(0.0), v - o.speed))));
            }
        }
        acc
    }

    /// Whether occupant `o` of connector `other` stands clear of the driver's path along
    /// `line`: every corner of it on the same side, at least half the driver's width and
    /// [`PASSING`] m away, and its own path on up to station `end` keeping that far too (it
    /// may move on). Never a long vehicle (towing trailers, they do not line up behind it;
    /// rigid, its front swings wide in bends).
    fn passes(&self, line: &Polyline, other: &Polyline, end: f64, o: &Occupant) -> bool {
        if o.speed >= STANDING || o.articulated || o.long {
            return false;
        }
        let need = self.geometry.half_width + PASSING;
        let (along, across) = (DVec2::from_angle(o.heading), DVec2::from_angle(o.heading).perp());
        let mut side = 0.0;
        for (x, y) in [(o.front, 1.0), (o.front, -1.0), (-o.rear, 1.0), (-o.rear, -1.0)] {
            let pr = line.project(o.pos + along * x + across * (y * o.half_width));
            if pr.distance < need || pr.offset * side < 0.0 {
                return false;
            }
            side = pr.offset;
        }
        let mut s = o.station + o.front;
        while s < end.min(other.length()) {
            if line.project(other.point_at(s).truncate()).distance < need + o.half_width {
                return false;
            }
            s += ZONE_STEP;
        }
        true
    }

    /// Distance (m) from the driver's front, at station `front` along connector `c`, to the
    /// nearest conflict zone ahead within `reach` that another vehicle occupies.
    fn occupied_zone(&self, g: &LaneGraph, index: &LaneIndex, c: u32, front: f64, me: u32, reach: f64) -> Option<f64> {
        let mut best: Option<f64> = None;
        for conf in g.connectors()[c as usize].conflicts.iter().filter(|x| x.kind != ConflictKind::Diverge) {
            let d = conf.station - ZONE - front;
            if d < 0.0 || d > reach || best.is_some_and(|b| b <= d) {
                continue;
            }
            let (so, eo) = (conf.other_station - ZONE, conf.other_station + conf.other_length + ZONE);
            let (line, other) = (&g.connectors()[c as usize].line, &g.connectors()[conf.other as usize].line);
            let busy = index.list(Elem::Connector(conf.other)).iter().any(|o| {
                !index.skips(o, me)
                    && o.station + o.front >= so
                    && o.station - o.rear <= eo
                    && !self.passes(line, other, eo, o)
            });
            if busy {
                best = Some(d);
            }
        }
        best
    }

    /// The command for the next policy step of `dt` of agent `me` at `pose` moving at `speed`
    /// (m/s, forward), from the snapshot `index` (to which it adds its grant), with the
    /// `signals` at time `t`.
    #[allow(clippy::too_many_arguments)]
    pub fn drive(
        &mut self,
        world: &StaticWorld,
        pose: &Pose,
        speed: f64,
        dt: f64,
        me: u32,
        index: &mut LaneIndex,
        signals: &Signals,
        t: f64,
    ) -> GroundSetpoint {
        self.notice(index, pose.pos.truncate());
        index.ignored.extend(self.encounters.iter().filter(|e| !e.1).map(|e| e.0));
        index.ignored.sort_unstable();
        let command = self.decide(world, pose, speed, dt, me, index, signals, t);
        index.ignored.clear();
        command
    }

    /// Meet the learning agents on the lanes near `pos`: each new one is noticed with
    /// probability `attention`; those gone or beyond [`ENCOUNTER_END`] m are forgotten.
    fn notice(&mut self, index: &LaneIndex, pos: DVec2) {
        if self.spec.attention >= 1.0 {
            return;
        }
        let near = |a: u32, r: f64| index.learners.iter().any(|&(b, p)| b == a && p.distance(pos) < r);
        self.encounters.retain(|e| near(e.0, ENCOUNTER_END));
        for &(a, p) in &index.learners {
            if p.distance(pos) < ENCOUNTER && !self.encounters.iter().any(|e| e.0 == a) {
                let noticed = self.attention_rng.uniform() < self.spec.attention;
                self.encounters.push((a, noticed));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decide(
        &mut self,
        world: &StaticWorld,
        pose: &Pose,
        speed: f64,
        dt: f64,
        me: u32,
        index: &mut LaneIndex,
        signals: &Signals,
        t: f64,
    ) -> GroundSetpoint {
        let stand = GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 };
        let g = world.roads().lanes();
        let v = speed.max(0.0);
        self.standing = if v < STANDING { self.standing + dt } else { 0.0 };
        let Some(p) = self.place else {
            self.lost += dt;
            return stand;
        };
        self.lost = 0.0;
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
            let lim = self.limited(e.speed_limit(g));
            v0 = v0.min((lim * lim + 2.0 * b * base).sqrt());
        }
        // Cyclists, who cannot lean hard, also brake to each bend's speed by its start.
        let mut d = 0.0;
        let mut bend = f64::INFINITY;
        // (Cyclists, whose lean lags their steering, look farther ahead.)
        let look = self.spec.lookahead[0] + self.spec.lookahead[1] * v;
        let look = if self.geometry.single_track { look.max(CYCLIST_LOOKAHEAD + CYCLIST_PREVIEW * v) } else { look };
        while d < reach {
            if let Some((e, s)) = at_distance(g, &chain, d) {
                // (On the path at its lateral offset.)
                let k = e.line(g).curvature_at(s);
                let k = (k / (1.0 - self.lateral(world, e, s) * k).max(0.1)).abs().max(1e-6);
                let v_bend2 = self.spec.lateral_accel / k;
                // (Cyclists no slower than CYCLIST_START: crawling they cannot lean into it.)
                let v_bend2 =
                    if self.geometry.single_track { v_bend2.max(CYCLIST_START * CYCLIST_START) } else { v_bend2 };
                v0 = v0.min((v_bend2 + 2.0 * b * d).sqrt());
                // (By the speed loop's lag earlier, and with a margin.)
                let v_bend2 = (BEND_MARGIN_CYCLIST * v_bend2).max(CYCLIST_START * CYCLIST_START);
                if self.geometry.single_track && v * v > v_bend2 {
                    bend = bend.min((v_bend2 - v * v) / (2.0 * (d - v * SPEED_LAG).max(BEND_BRAKE)));
                }
            }
            d += PREVIEW_STEP;
        }
        let idm = Idm { v0, ..self.idm };
        self.idm.v0 = v0;

        // Overtaking a cyclist: shifting out at up to SHIFT_RATE.
        let want = self.overtaking(g, index, p, me, v, v0);
        if !self.geometry.single_track {
            self.shift += (want - self.shift).clamp(-SHIFT_RATE * dt, SHIFT_RATE * dt);
            self.offset = self.shift;
        }

        // Following: the vehicle ahead along the chain, and while changing, also in the lane
        // being left.
        let mut acc = self.follow(g, index, &chain, p.station, me, v, &idm).min(bend);
        if let (Some(c), Elem::Lane(l)) = (self.change, p.elem) {
            let s = p.station * g.lanes()[c.from as usize].line.length() / g.lanes()[l as usize].line.length();
            acc = acc.min(self.follow(g, index, &[(Elem::Lane(c.from), -s)], s, me, v, &idm));
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

        // Junctions, crossings, and a bus's stops.
        acc = acc.min(self.junction(g, index, &chain, p, me, v, &idm, signals, t, dt));
        acc = acc.min(self.crosswalks(world, index, &chain, v, &idm));
        acc = acc.min(self.bus_stop(g, p, v, &idm, dt));

        // Lane changes.
        if self.change.is_none()
            && self.overtake.is_none()
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
        let target = self.reference(world, &chain, place, look);
        let heading = yaw(pose.rot);
        // Long vehicles from their rear axle (the look-ahead still counted from the front):
        // from the front, a look-ahead as short as their wheelbase and slow steering make
        // them weave.
        let from = if self.is_long() {
            pose.pos.truncate() - self.geometry.wheelbase * DVec2::from_angle(heading)
        } else {
            pose.pos.truncate()
        };
        let local = DVec2::from_angle(-heading).rotate(target - from);
        let d2 = local.length_squared();
        let curvature = if d2 > 1e-6 { 2.0 * local.y / d2 } else { 0.0 };
        let limit = self.geometry.max_curvature;
        let curvature = curvature.clamp(-limit, limit);
        let mut command = (v + acc.max(-3.0 * b) * SPEED_LAG).max(0.0);
        // Cyclists free to go ride off briskly: lingering at walking pace (on their feet, and
        // on grades barely moving) they wobble and roll back.
        if self.geometry.single_track && acc > 0.0 && v < CYCLIST_START {
            command = command.max(CYCLIST_START.min(idm.v0));
        }
        GroundSetpoint::SpeedCurvature { speed: command, curvature }
    }

    /// Crossings ahead along `chain` with pedestrians on them (or committed to them) that have
    /// not passed the lane yet: a standing obstacle [`CROSSWALK_GAP`] m before the band. The
    /// acceleration they allow (infinite when none); a crossing the front has reached is
    /// cleared while rolling (not when standing on it), but its users in the vehicle's path (within [`CROSSWALK_SIDE`] m of its sides)
    /// remain obstacles [`CROSSWALK_GAP`] m ahead of them.
    fn crosswalks(&self, world: &StaticWorld, index: &LaneIndex, chain: &[(Elem, f64)], v: f64, idm: &Idm) -> f64 {
        let g = world.roads().lanes();
        let reach = v * v / (2.0 * self.idm.decel) + 30.0;
        let mut acc = f64::INFINITY;
        for &(e, base) in chain {
            if base > reach {
                break;
            }
            let Elem::Lane(l) = e else { continue };
            let lane = &g.lanes()[l as usize];
            for &(c, station) in &index.lane_crossings[l as usize] {
                let users = &index.crossing_users[c as usize];
                if users.is_empty() {
                    continue;
                }
                // Users in the way: standing obstacles wherever the front is.
                let side = self.geometry.half_width + CROSSWALK_SIDE;
                let off = self.lateral(world, e, station);
                for &(p, _) in users {
                    let pr = lane.line.project(p);
                    let ahead = base + pr.station - self.geometry.front;
                    if (pr.offset - off).abs() < side && ahead > -CROSSWALK_GAP && ahead < reach {
                        acc = acc.min(idm.accel(v, Some(((ahead - CROSSWALK_GAP).max(0.0), v))));
                    }
                }
                let gap = base + station - 0.5 * CROSSWALK - self.geometry.front - CROSSWALK_GAP;
                if (gap < -CROSSWALK_GAP && v > CROSSWALK_ROLLING) || gap > reach {
                    continue;
                }
                let at = lane.line.point_at(station).truncate()
                    + off * DVec2::from_angle(lane.line.heading_at(station)).perp();
                let clear = 0.5 * lane.width + CROSSWALK_CLEAR;
                if users.iter().any(|&(p, d)| (at - p).dot(d) > -clear) {
                    acc = acc.min(idm.accel(v, Some((gap.max(0.0), v))));
                }
            }
        }
        acc
    }

    /// The point `ahead` m on along the chain, shifted towards the lane being left and by
    /// its lateral offset ([`TrafficDriver::lateral`]).
    fn reference(&self, world: &StaticWorld, chain: &[(Elem, f64)], place: Place, ahead: f64) -> DVec2 {
        let g = world.roads().lanes();
        let Some((e, s)) = at_distance(g, chain, ahead) else {
            return place.elem.line(g).point_at(place.station).truncate();
        };
        let lateral = self.lateral(world, e, s) * DVec2::from_angle(e.line(g).heading_at(s)).perp();
        let line = e.line(g);
        let mut point = line.point_at(s).truncate();
        // Vehicles towing trailers swing wide in left bends: their towing unit's rear axle
        // out by half the offtracking of the trailers' (up to SWING_MAX), which then run as far
        // inside (curvature averaged over ±5 m). In right bends they would swing into the
        // oncoming lane, where traffic waits at the line; there the trailers cut over the
        // corner instead.
        let l = self.geometry.trailing();
        if l > SWING_LENGTH {
            let k = ((-5..=5).map(|d| line.curvature_at(s + f64::from(d))).sum::<f64>() / 11.0).max(0.0);
            let r = 1.0 / k.abs().max(1e-6);
            let off = if r > l { r - (r * r - l * l).sqrt() } else { r };
            point -= k.signum() * (0.5 * off).min(SWING_MAX) * DVec2::from_angle(line.heading_at(s)).perp();
        } else if self.is_long() {
            // Rigid long vehicles (buses), steered from the rear axle, keep it inside by half
            // their front axle's swing (`wb²/2R`), which then runs only as far out: on the line,
            // the front swings into the oncoming lane at a bend's end.
            let k = (-5..=5).map(|d| line.curvature_at(s + f64::from(d))).sum::<f64>() / 11.0;
            let wb = self.geometry.wheelbase;
            point +=
                k.signum() * (0.25 * wb * wb * k.abs()).min(SWING_MAX) * DVec2::from_angle(line.heading_at(s)).perp();
        }
        match (self.change, e) {
            (Some(c), Elem::Lane(l)) if e == place.elem => {
                let t = ((s - c.start) / c.length).clamp(0.0, 1.0);
                let w = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t);
                let from = &g.lanes()[c.from as usize].line;
                let sf = s * from.length() / g.lanes()[l as usize].line.length();
                point + (1.0 - w) * (from.point_at(sf).truncate() - point) + lateral
            }
            _ => point + lateral,
        }
    }

    /// Overtaking (cars, not long vehicles): a cyclist ahead on a lane that is the only one
    /// of its direction, with an oncoming lane beside it, is passed shifted out by
    /// [`PASS_CLEAR`] m beyond it when the pass (at the desired speed `v0`, until the rear is
    /// [`PASS_GAP`] m ahead of the cyclist) ends [`PASS_END`] m before the lane's end with
    /// room there, no crossing lies on the way, and the oncoming lane stays clear meanwhile
    /// ([`TrafficDriver::pass_clear`]). Given up before drawing level when the oncoming lane
    /// no longer stays clear. The lateral offset to steer to.
    fn overtaking(&mut self, g: &LaneGraph, index: &LaneIndex, p: Place, me: u32, v: f64, v0: f64) -> f64 {
        let geo = self.geometry;
        let lane = match p.elem {
            Elem::Lane(l) if !geo.single_track && !self.is_long() && self.change.is_none() => l,
            _ => {
                self.overtake = None;
                return 0.0;
            }
        };
        let Some(opp) = index.opposite[lane as usize] else {
            self.overtake = None;
            return 0.0;
        };
        let list = &index.lanes[lane as usize];
        if let Some(o) = self.overtake {
            let Some(c) = list.iter().find(|x| x.agent == o.agent && !x.tail) else {
                self.overtake = None;
                return 0.0;
            };
            let passed = p.station + geo.rear > c.station + c.front + PASS_GAP;
            let level = p.station + geo.front >= c.station - c.rear;
            if passed || !level && !self.pass_clear(g, index, lane, opp, p, c, me, v, v0) {
                self.overtake = None;
                return 0.0;
            }
            return o.shift;
        }
        let i = list.partition_point(|x| x.station <= p.station);
        let mut ahead = list[i..].iter().filter(|x| !index.skips(x, me) && !x.tail);
        let Some(c) = ahead.next().copied() else { return 0.0 };
        let gap = c.station - c.rear - p.station - geo.front;
        let shift = c.offset + c.half_width + PASS_CLEAR + geo.half_width;
        let width = g.lanes()[lane as usize].width;
        if !c.cyclist || gap > OVERTAKE_REACH || v0 < c.speed + OVERTAKE_DV || shift <= 0.0 || shift > width {
            return 0.0;
        }
        // Room after it.
        let back_in = c.station + c.front + PASS_GAP + geo.front - geo.rear;
        if ahead.next().is_some_and(|n| n.station - n.rear < back_in + self.idm.min_gap) {
            return 0.0;
        }
        if !self.pass_clear(g, index, lane, opp, p, &c, me, v, v0) {
            return 0.0;
        }
        self.overtake = Some(Overtake { agent: c.agent, shift });
        self.overtakes += 1;
        shift
    }

    /// Whether passing cyclist `c` on `lane` from place `p` at speed `v` up to `v0` ends
    /// [`PASS_END`] m before the lane's end, crosses no crossing, and the oncoming lane `opp`
    /// stays clear meanwhile (plus [`PASS_MARGIN`] s) of vehicles at up to [`ONCOMING`] times
    /// its limit, coming down it or the connectors into it and the lanes before those.
    #[allow(clippy::too_many_arguments)]
    fn pass_clear(
        &self,
        g: &LaneGraph,
        index: &LaneIndex,
        lane: u32,
        opp: u32,
        p: Place,
        c: &Occupant,
        me: u32,
        v: f64,
        v0: f64,
    ) -> bool {
        let geo = self.geometry;
        // Accelerating to `v0`, until it has gained the distance on the cyclist.
        let u = v0.max(v);
        let distance = (c.station + c.front + PASS_GAP - (p.station + geo.rear)).max(0.0);
        let (mut x, mut vel, mut t) = (0.0, v, 0.0);
        while x - c.speed * t < distance {
            if t > PASS_LONGEST {
                return false;
            }
            vel = (vel + self.idm.accel * PASS_STEP).min(u);
            x += vel * PASS_STEP;
            t += PASS_STEP;
        }
        let end = p.station + x + geo.front;
        let len = g.lanes()[lane as usize].line.length();
        if end + PASS_END > len {
            return false;
        }
        if index.lane_crossings[lane as usize].iter().any(|&(_, s)| s > p.station - geo.front && s < end + CROSSWALK) {
            return false;
        }
        // On the oncoming lane (stations running the other way).
        let ol = &g.lanes()[opp as usize];
        let olen = ol.line.length();
        let at = olen * (1.0 - p.station / len);
        let reach = (u + ONCOMING * ol.speed) * (t + PASS_MARGIN) + geo.front;
        let (lo, hi) = (at - reach, at - geo.rear + PASS_GAP);
        let blocks = |o: &Occupant, s: f64| !index.skips(o, me) && s + o.front > lo && s - o.rear < hi;
        if index.lanes[opp as usize].iter().any(|o| blocks(o, o.station)) {
            return false;
        }
        if lo < 0.0 {
            for &k in &ol.predecessors {
                let cl = g.connectors()[k as usize].line.length();
                if index.connectors[k as usize].iter().any(|o| blocks(o, o.station - cl)) {
                    return false;
                }
                let from = g.connectors()[k as usize].from;
                let fl = g.lanes()[from as usize].line.length();
                if index.lanes[from as usize].iter().any(|o| blocks(o, o.station - cl - fl)) {
                    return false;
                }
            }
        }
        true
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
        let a_c = self.follow(g, index, &chain_here, station, me, v, idm);
        // The old follower (distances between reference points), now and with the driver gone.
        let leader_here = index.ahead(&chain_here, station, me, None);
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
            if must.is_some() && !mandatory || !mandatory && geo.single_track {
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
            let leader = index.ahead(&chain_t, s - rear, me, None);
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

/// Whether connectors `a` and `b` make the same movement: from the same road and direction,
/// the same turn into the same road and direction.
fn same_movement(g: &LaneGraph, a: u32, b: u32) -> bool {
    let (ca, cb) = (&g.connectors()[a as usize], &g.connectors()[b as usize]);
    let lane = |l: u32| &g.lanes()[l as usize];
    let (fa, fb, ta, tb) = (lane(ca.from), lane(cb.from), lane(ca.to), lane(cb.to));
    ca.turn == cb.turn && fa.road == fb.road && fa.dir == fb.dir && ta.road == tb.road && ta.dir == tb.dir
}

/// An f64 ordered by [`f64::total_cmp`] (for heaps).
#[derive(Clone, Copy, Debug, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// IDM acceleration of a driver at speed `v` that is to stop `d` m ahead.
fn hold(idm: &Idm, v: f64, d: f64) -> f64 {
    idm.accel(v, Some((d.max(0.0) + idm.min_gap, v)))
}

/// Largest |curvature| along `line` (1/m), sampled every metre.
fn bend(line: &Polyline) -> f64 {
    let n = line.length().ceil() as usize;
    (0..=n).map(|k| line.curvature_at(k as f64).abs()).fold(0.0, f64::max)
}

/// Time (s) to cover `d` m from speed `v`, accelerating at `a` up to `vmax`.
fn travel_time(d: f64, v: f64, a: f64, vmax: f64) -> f64 {
    if d <= 0.0 {
        return 0.0;
    }
    let vmax = vmax.max(0.5);
    let v = v.clamp(0.0, vmax);
    let a = a.max(0.1);
    let t1 = (vmax - v) / a;
    let d1 = 0.5 * (v + vmax) * t1;
    if d <= d1 { ((v * v + 2.0 * a * d).sqrt() - v) / a } else { t1 + (d - d1) / vmax }
}

/// Where a cyclist rides on lane `lane` (m from its line, positive to the left): in the
/// middle of the bike lane beside the rightmost lane of a road side with one, else
/// [`CYCLIST_EDGE`] m from the lane's right edge.
pub fn keep_right(world: &StaticWorld, lane: u32) -> f64 {
    let l = &world.roads().lanes().lanes()[lane as usize];
    let section = world.roads().section(l.road as usize);
    let bike =
        if l.right.is_none() { section.bike[if section.one_way() { 0 } else { usize::from(l.dir) }] } else { 0.0 };
    if bike >= BIKE_LANE { -0.5 * (l.width + bike) } else { -(0.5 * l.width - CYCLIST_EDGE).max(0.0) }
}

/// Distance of `p` past the end of `line` along its final direction (m).
fn beyond(line: &Polyline, p: DVec2) -> f64 {
    let end = line.point_at(line.length()).truncate();
    (p - end).dot(DVec2::from_angle(line.heading_at(line.length())))
}

/// Lanes long enough to spawn on (with their lengths), for [`draw_lane_point`]: a vehicle
/// reaching `back` m behind its reference point, keeping to `only` lanes if given (cyclists:
/// to the rightmost lanes).
fn spawn_lanes(g: &LaneGraph, back: f64, only: Option<&[bool]>, cyclist: bool) -> Vec<(u32, f64)> {
    (g.lanes().iter().enumerate())
        .filter(|(k, l)| {
            l.line.length() > spawn_start(back) + 7.0 && only.is_none_or(|o| o[*k]) && (!cyclist || l.right.is_none())
        })
        .map(|(k, l)| (k as u32, l.line.length()))
        .collect()
}

/// Smallest station of a spawn (m): 5 m, or 1 m more than the vehicle reaches back.
fn spawn_start(back: f64) -> f64 {
    5.0f64.max(back + 1.0)
}

/// A point of a random lane of `lanes` (drawn by length, 5 m clear of its end and
/// [`spawn_start`] of its start; for a cyclist, where it rides: [`keep_right`]) and the
/// lane's heading there.
fn draw_lane_point(
    world: &StaticWorld,
    lanes: &[(u32, f64)],
    back: f64,
    cyclist: bool,
    rng: &mut SimRng,
) -> (DVec2, f64) {
    let g = world.roads().lanes();
    let lo = spawn_start(back);
    let total: f64 = lanes.iter().map(|l| l.1 - lo - 5.0).sum();
    let mut x = rng.uniform() * total;
    let &(k, len) = lanes
        .iter()
        .find(|l| {
            x -= l.1 - lo - 5.0;
            x < 0.0
        })
        .unwrap_or(lanes.last().expect("lanes"));
    let s = lo + rng.uniform() * (len - lo - 5.0);
    let line = &g.lanes()[k as usize].line;
    let off = if cyclist { keep_right(world, k) } else { 0.0 };
    let heading = line.heading_at(s);
    (line.point_at(s).truncate() + off * DVec2::from_angle(heading).perp(), wrap_angle(heading))
}

/// `count` spawns in random lanes ([`draw_lane_point`], for vehicles reaching `back` m behind
/// their reference point, on `only` lanes if given, cyclists where they ride) along their direction, at least
/// `min_separation` from `placed` where possible (appended to it); `lift` above the ground.
/// `None` on maps without (such) lanes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lane_spawns(
    world: &StaticWorld,
    count: usize,
    back: f64,
    only: Option<&[bool]>,
    cyclist: bool,
    min_separation: f64,
    lift: f64,
    placed: &mut Vec<DVec3>,
    rng: &mut SimRng,
) -> Option<Vec<(DVec3, f64)>> {
    let lanes = spawn_lanes(world.roads().lanes(), back, only, cyclist);
    if lanes.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best: Option<(f64, DVec3, f64)> = None;
        for _ in 0..64 {
            let (xy, heading) = draw_lane_point(world, &lanes, back, cyclist, rng);
            let pos = xy.extend(world.terrain().height(xy.x, xy.y) + lift);
            let apart = placed.iter().map(|q| q.truncate().distance(xy)).fold(f64::INFINITY, f64::min);
            if best.is_none_or(|b| apart > b.0) {
                best = Some((apart, pos, heading));
            }
            if apart >= min_separation {
                break;
            }
        }
        let (_, pos, heading) = best.expect("drawn");
        placed.push(pos);
        out.push((pos, heading));
    }
    Some(out)
}

/// Vehicles whose [`DriverGeometry::tracking`] length exceeds this (m) swing wide in bends.
const SWING_LENGTH: f64 = 3.0;

/// A long vehicle's [`Network`] takes bends up to this factor tighter than its lock (it runs
/// a little wide there).
const BEND_MARGIN: f64 = 1.25;

/// Largest outward shift of a long vehicle's steering reference in bends (m).
const SWING_MAX: f64 = 1.5;

/// How much farther (m) than [`LOST`] and its swing a long vehicle may be off its line before
/// it is lost (on bends tighter than its lock it runs wide).
const LOST_WIDE: f64 = 3.0;
/// A learning agent is met within this distance (m) and forgotten beyond the second (see
/// [`TrafficDriverSpec::attention`]).
const ENCOUNTER: f64 = 60.0;
const ENCOUNTER_END: f64 = 80.0;
/// Shortest lane with a bus stop (m).
const STOP_LANE: f64 = 40.0;
/// A bus stands at its stop within this distance of it (m), and passes it by when this far
/// beyond (m).
const STOP_REACHED: f64 = 1.0;
const STOP_MISSED: f64 = 5.0;
/// Deceleration (m/s²) others are counted on to stop with short of a long vehicle's sweep.
const SWEEP_BRAKE: f64 = 2.5;
/// Largest setback of a stop line for long vehicles' sweeps (m; [`hold_backs`]).
const HOLD_BACK_MAX: f64 = 25.0;

/// Elements a driver remembers behind its place, for its tail ([`TrafficDriver::trail`]).
const TRAIL: usize = 8;

/// Step (m) of the check that a standing vehicle's path stays clear ([`TrafficDriver::passes`]).
const ZONE_STEP: f64 = 0.5;

/// Clearance (m) of a respawn from every other agent.
const RESPAWN_APART: f64 = 20.0;

/// Where a driver respawns: a random lane point (and heading; [`draw_lane_point`], reaching
/// `back` m behind it, on `only` lanes if given, cyclists where they ride) at least `clear` m from every learning agent in `learners` and
/// [`RESPAWN_APART`] m from `others`, or the draw of 64 that comes closest. `None` on maps
/// without lanes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn respawn_spot(
    world: &StaticWorld,
    back: f64,
    only: Option<&[bool]>,
    cyclist: bool,
    learners: &[DVec2],
    others: &[DVec2],
    clear: f64,
    rng: &mut SimRng,
) -> Option<(DVec2, f64)> {
    let lanes = spawn_lanes(world.roads().lanes(), back, only, cyclist);
    if lanes.is_empty() {
        return None;
    }
    let nearest = |set: &[DVec2], p: DVec2| set.iter().map(|q| q.distance(p)).fold(f64::INFINITY, f64::min);
    let mut best: Option<(f64, DVec2, f64)> = None;
    for _ in 0..64 {
        let (xy, heading) = draw_lane_point(world, &lanes, back, cyclist, rng);
        let score = (nearest(learners, xy) / clear.max(1e-9)).min(1.0) + (nearest(others, xy) / RESPAWN_APART).min(1.0);
        if best.is_none_or(|b| score > b.0) {
            best = Some((score, xy, heading));
        }
        if score >= 2.0 {
            break;
        }
    }
    best.map(|b| (b.1, b.2))
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
