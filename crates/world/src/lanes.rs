//! The lane graph of a road network, derived from its roads and their [`Section`]s: directed
//! lanes along the roads, connectors through the junctions with their conflicts and
//! priorities, U-turns at dead ends, junction control, lane-level routes and the area class of
//! a point (lane, median, bike lane, parking lane, sidewalk, crosswalk, junction).
//!
//! The graph is a pure function of the network, built whenever a network is built, so it is
//! never stored in map files and adds nothing to map hashes.
//!
//! Conventions: lane offsets are measured to the right of the lane's direction of travel;
//! lane index 0 is the leftmost (innermost) lane of its direction. Traffic keeps right.

use crate::roads::{NodeKind, Polyline, Road, RoadClass, RoadNode, Section, SegmentGrid, wrap_angle};
use crate::signals::{self, Controller};
use glam::{DVec2, DVec3};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_6, PI, TAU};

/// Smallest turning radius (m) the connectors are built for (a car's, at the kerb).
pub const MIN_TURN_RADIUS: f64 = 5.5;

/// Distance (m) between two connector lines within which vehicles on them are in each other's
/// way: their conflict zone (a car's width and a margin; less than the narrowest lane).
pub const CONFLICT_GAP: f64 = 2.5;

/// Lanes shorter than this (m) are cramped: vehicles through the connectors at their ends
/// follow their lines less closely (the path bends sharply there), so those connectors'
/// conflict zones reach [`CRAMPED_MARGIN`] m further.
pub const CRAMPED: f64 = 5.0;

/// See [`CRAMPED`].
pub const CRAMPED_MARGIN: f64 = 1.0;

/// Sampling step (m) of the conflict zones.
const ZONE_STEP: f64 = 0.5;

/// Turning radius (m) the junction setbacks leave room for, with a margin over
/// [`MIN_TURN_RADIUS`] for the Bézier's deviation from an arc.
const DESIGN_TURN_RADIUS: f64 = 6.5;

/// Largest lateral acceleration (m/s²) that sets a connector's speed limit.
const LATERAL_ACCEL: f64 = 2.5;

/// Extra route cost (m) of a lane change.
const LANE_CHANGE_COST: f64 = 30.0;

/// Length (m) of a crossing's band along its road ([`Area::Crosswalk`]).
pub const CROSSWALK: f64 = 3.0;

/// Spacing (m) of mid-block crossings: a street gets one per this length between the
/// crosswalks at its ends.
pub const MID_BLOCK_SPACING: f64 = 120.0;

/// Speed limit (m/s) of a road class.
pub fn speed_limit(class: RoadClass) -> f64 {
    match class {
        RoadClass::Arterial | RoadClass::Collector | RoadClass::Gravel => 50.0 / 3.6,
        RoadClass::Local | RoadClass::Track => 30.0 / 3.6,
        RoadClass::Paved => 80.0 / 3.6,
    }
}

/// Priority rank of a road class (higher has the right of way).
pub fn class_rank(class: RoadClass) -> u8 {
    match class {
        RoadClass::Arterial => 5,
        RoadClass::Paved => 4,
        RoadClass::Collector => 3,
        RoadClass::Local => 2,
        RoadClass::Gravel => 1,
        RoadClass::Track => 0,
    }
}

/// A directed lane along a road, between the junction areas at its ends.
#[derive(Clone, Debug, PartialEq)]
pub struct Lane {
    pub road: u32,
    /// 0: from the road's `start` to its `end`; 1: against it.
    pub dir: u8,
    /// 0 is the leftmost lane of its direction.
    pub index: u8,
    pub width: f64,
    /// Speed limit (m/s).
    pub speed: f64,
    /// Offset of the centre line to the right of travel from the road's centre line (m).
    pub offset: f64,
    pub line: Polyline,
    /// The nodes it comes from and leads to.
    pub from_node: u32,
    pub to_node: u32,
    /// Neighbouring lanes of the same direction (traffic may change into them).
    pub left: Option<u32>,
    pub right: Option<u32>,
    /// Connectors leaving its end and entering its start.
    pub successors: Vec<u32>,
    pub predecessors: Vec<u32>,
}

/// The movement a connector makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Turn {
    Straight,
    Left,
    Right,
    /// Turning round at a dead end.
    UTurn,
}

/// How two connectors meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConflictKind {
    /// The lines cross.
    Cross,
    /// Both lead into the same lane.
    Merge,
    /// The lines pass closer than [`CONFLICT_GAP`] without crossing (such as opposing left
    /// turns): drivers keep apart, but signals may let both go together.
    Near,
    /// Both leave the same lane: the zone is where they have not yet parted, and neither
    /// gives way (the vehicle behind follows the one ahead).
    Diverge,
}

impl ConflictKind {
    /// Whether signals must keep the two connectors in different phases.
    pub fn exclusive(self) -> bool {
        matches!(self, Self::Cross | Self::Merge)
    }
}

/// Where a connector meets another one: the stretches of both closer than [`CONFLICT_GAP`] to
/// the other's line (for a merge, up to the common end).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conflict {
    pub other: u32,
    pub kind: ConflictKind,
    /// Station (m) where the zone starts along this connector, and its length there.
    pub station: f64,
    pub length: f64,
    /// The same along the other connector.
    pub other_station: f64,
    pub other_length: f64,
    /// Whether this connector must give way to the other one (exactly one of the two does,
    /// except for a [`Diverge`](ConflictKind::Diverge), where neither does).
    pub yields: bool,
}

/// A path from the end of one lane to the start of another through a junction.
#[derive(Clone, Debug, PartialEq)]
pub struct Connector {
    pub from: u32,
    pub to: u32,
    /// The node it crosses.
    pub node: u32,
    pub turn: Turn,
    /// Speed limit (m/s): the lanes', lowered for the curvature.
    pub speed: f64,
    pub line: Polyline,
    pub conflicts: Vec<Conflict>,
}

/// How traffic through a junction is controlled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JunctionKind {
    /// Traffic lights (M8a step 4).
    Signal,
    /// The minor approaches stop.
    Stop,
    /// The minor approaches give way.
    Yield,
    /// The entries give way to the ring.
    Roundabout,
    /// Equal roads: give way to the right.
    Uncontrolled,
    /// Two roads continuing into each other.
    Through,
    /// A dead end (or the map's edge), where traffic turns round.
    DeadEnd,
}

/// What an approach must do before entering the junction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Control {
    None,
    Signal,
    Stop,
    Yield,
}

/// The lanes of one road entering a junction, with their stop line.
#[derive(Clone, Debug, PartialEq)]
pub struct Approach {
    pub road: u32,
    pub dir: u8,
    pub lanes: Vec<u32>,
    pub control: Control,
    /// Across the approach's lanes at their ends, from the left edge to the right edge.
    pub stop_line: [DVec3; 2],
}

/// A node of the network with its control, approaches and connectors.
#[derive(Clone, Debug, PartialEq)]
pub struct Junction {
    pub node: u32,
    pub kind: JunctionKind,
    /// Radius (m) of the area around the node that the lanes leave free (their largest
    /// setback).
    pub radius: f64,
    pub approaches: Vec<Approach>,
    pub connectors: Vec<u32>,
}

/// A marked pedestrian crossing over a road: a band [`CROSSWALK`] m long across its
/// carriageway. Urban streets with sidewalks get one at each end that meets two or more other
/// roads (ring roads of roundabouts excepted; at signals, only where some phase lets
/// pedestrians walk), just beyond the lanes' ends, and streets with
/// sidewalks on both sides and no parking lanes get mid-block crossings, one per
/// [`MID_BLOCK_SPACING`] m between them (a third of the way into each stretch, away from the
/// lanes' middles where buses stop).
#[derive(Clone, Debug, PartialEq)]
pub struct Crossing {
    pub road: u32,
    /// Centre of the band (m along the road from its start).
    pub station: f64,
    /// The node of the junction it lies at; None mid-block.
    pub node: Option<u32>,
    /// Length (m) of the walk across: the carriageway's width.
    pub length: f64,
    /// The lanes it crosses, each with the station along it of the band's centre.
    pub lanes: Vec<(u32, f64)>,
    /// At a signalized junction: the controller and the phases (bit k for phase k) during
    /// which pedestrians may start to cross ([`Controller::walk`]).
    pub signal: Option<(u32, u32)>,
}

/// What lies at a point of the road surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Area {
    Lane,
    Median,
    BikeLane,
    Parking,
    Sidewalk,
    Crosswalk,
    /// Inside a junction's area, on the carriageway.
    Junction,
    /// Off the roads.
    Off,
}

/// A step of a lane-level route.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RouteStep {
    Lane(u32),
    Connector(u32),
    /// A lane change from the first lane into its neighbour.
    Change(u32, u32),
}

/// A lane-level route: the steps from a lane to a lane, and its length along the lanes (m).
#[derive(Clone, Debug, PartialEq)]
pub struct LaneRoute {
    pub steps: Vec<RouteStep>,
    pub length: f64,
}

/// The lanes and connectors of a road network.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LaneGraph {
    lanes: Vec<Lane>,
    connectors: Vec<Connector>,
    junctions: Vec<Junction>,
    /// Setback (m) of the lanes from each end of each road: `[start, end]`.
    setbacks: Vec<[f64; 2]>,
    grid: SegmentGrid,
    /// Signal controllers, and each connector's (controller, phase) and junction's controller.
    controllers: Vec<Controller>,
    connector_signals: Vec<Option<(u32, u8)>>,
    junction_controllers: Vec<Option<u32>>,
    /// Pedestrian crossings, and per road the crossings over it (by station).
    crossings: Vec<Crossing>,
    road_crossings: Vec<Vec<u32>>,
    /// Whether each road is a piece of a roundabout's ring.
    rings: Vec<bool>,
}

/// One end of a road at a node.
#[derive(Clone, Copy, Debug)]
struct End {
    road: u32,
    /// Whether the road starts here.
    start: bool,
    /// Heading of the road leaving the node.
    heading: f64,
}

impl End {
    /// Travel direction of the road leaving the node, and arriving at it.
    fn out_dir(&self) -> u8 {
        if self.start { 0 } else { 1 }
    }

    fn in_dir(&self) -> u8 {
        1 - self.out_dir()
    }
}

/// Lanes of a road's section as (direction, index, offset right of travel, width).
fn lane_layout(road: &Road, section: &Section) -> Vec<(u8, u8, f64, f64)> {
    let mut out = Vec::new();
    // Gravel roads and tracks: one shared lane each way, on the centre line.
    if matches!(road.class, RoadClass::Gravel | RoadClass::Track) {
        return vec![(0, 0, 0.0, road.width), (1, 0, 0.0, road.width)];
    }
    let lw = section.lane_width;
    if section.one_way() {
        let left = -0.5 * section.width();
        for i in 0..section.lanes[0] {
            out.push((0, i, left + (f64::from(i) + 0.5) * lw, lw));
        }
        return out;
    }
    for dir in 0..2u8 {
        for i in 0..section.lanes[dir as usize] {
            out.push((dir, i, 0.5 * section.median + (f64::from(i) + 0.5) * lw, lw));
        }
    }
    out
}

/// Unit vector along heading `h`.
fn unit(h: f64) -> DVec2 {
    DVec2::new(h.cos(), h.sin())
}

/// Unit vector to the right of heading `h`.
fn right(h: f64) -> DVec2 {
    DVec2::new(h.sin(), -h.cos())
}

/// `points` moved `offset` m to the right of their direction, without the loops an offset
/// makes on the inside of a sharp bend: points that would run backwards, and points before
/// them that a later point lies behind, are left out (the ends always stay).
fn offset_line(points: &[DVec3], offset: f64) -> Vec<DVec3> {
    let n = points.len();
    let moved: Vec<(DVec3, DVec2)> = (0..n)
        .map(|i| {
            let (a, b) = (points[i.saturating_sub(1)].truncate(), points[(i + 1).min(n - 1)].truncate());
            let d = (b - a).normalize_or_zero();
            let r = DVec2::new(d.y, -d.x);
            ((points[i].truncate() + r * offset).extend(points[i].z), d)
        })
        .collect();
    let mut out: Vec<(DVec3, DVec2)> = vec![moved[0]];
    for (i, &(p, d)) in moved.iter().enumerate().skip(1) {
        let behind = |q: &(DVec3, DVec2)| (p - q.0).truncate().dot(q.1) <= 1e-3;
        while out.len() > 1 && behind(out.last().expect("points")) {
            out.pop();
        }
        let step = (p - out.last().expect("points").0).truncate();
        if (step.dot(d) > 1e-3 && step.length() > 0.05) || i == n - 1 {
            out.push((p, d));
        }
    }
    out.into_iter().map(|(p, _)| p).collect()
}

/// A cubic Bézier from `p0` (heading `h0`) to `p3` (heading `h3`), sampled about every metre;
/// the handles fit a circular arc for the turn angle.
fn bezier(p0: DVec3, h0: f64, p3: DVec3, h3: f64) -> Vec<DVec3> {
    let d = p0.truncate().distance(p3.truncate());
    let theta = wrap_angle(h3 - h0).abs();
    let k = if theta < 1e-3 { d / 3.0 } else { d * (4.0 / 3.0) * (theta / 4.0).tan() / (2.0 * (theta / 2.0).sin()) };
    let (a, b) = (p0.truncate(), p3.truncate());
    let (c1, c2) = (a + unit(h0) * k, b - unit(h3) * k);
    let n = (d.ceil() as usize).max(4);
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let u = 1.0 - t;
            let q = a * (u * u * u) + c1 * (3.0 * u * u * t) + c2 * (3.0 * u * t * t) + b * (t * t * t);
            q.extend(p0.z + (p3.z - p0.z) * t)
        })
        .collect()
}

/// A path from `p0` (heading `h0`) to `p3` (heading `h3`): where the headings' lines meet ahead,
/// straight on along the longer leg, then an arc-like Bézier as wide as the shorter leg allows.
fn connect(p0: DVec3, h0: f64, p3: DVec3, h3: f64) -> Vec<DVec3> {
    let (a, b) = (p0.truncate(), p3.truncate());
    let (u0, u3) = (unit(h0), unit(h3));
    let den = u0.perp_dot(u3);
    if wrap_angle(h3 - h0).abs() < 0.05 || den.abs() < 1e-9 {
        return bezier(p0, h0, p3, h3);
    }
    // a + u0·t0 = b − u3·t3.
    let t0 = (b - a).perp_dot(u3) / den;
    let t3 = u0.perp_dot(b - a) / den;
    if t0 <= 0.0 || t3 <= 0.0 {
        return bezier(p0, h0, p3, h3);
    }
    let m = t0.min(t3);
    let i = a + u0 * t0;
    let (q0, q3) = (i - u0 * m, i + u3 * m);
    let len = t0 + t3;
    let z = |q: DVec2| {
        let s = if q.distance(a) <= t0 { q.distance(a) } else { t0 + q.distance(i) };
        p0.z + (p3.z - p0.z) * (s / len).clamp(0.0, 1.0)
    };
    let mut out = Vec::new();
    let lead = t0 - m;
    if lead > 1e-6 {
        let n = (lead.ceil() as usize).max(1);
        out.extend((0..n).map(|k| (a + u0 * (lead * k as f64 / n as f64)).extend(p0.z)));
    }
    let z0 = z(q0);
    let z3 = if t3 > m { p0.z + (p3.z - p0.z) * ((t0 + m) / len) } else { p3.z };
    out.extend(bezier(q0.extend(z0), h0, q3.extend(z3), h3));
    let tail = t3 - m;
    if tail > 1e-6 {
        let n = (tail.ceil() as usize).max(1);
        out.extend((1..=n).map(|k| {
            let f = k as f64 / n as f64;
            (q3 + u3 * (tail * f)).extend(z3 + (p3.z - z3) * f)
        }));
    }
    out
}

/// Points along the circle about `c` from `a` to `b` (the angle swept counter-clockwise when
/// `ccw`, else clockwise), about every metre.
fn arc(c: DVec2, a: DVec2, b: DVec2, ccw: bool) -> Vec<DVec2> {
    let r = a.distance(c);
    let (a0, a1) = ((a - c).y.atan2((a - c).x), (b - c).y.atan2((b - c).x));
    let sweep = if ccw { (a1 - a0).rem_euclid(TAU) } else { -(a0 - a1).rem_euclid(TAU) };
    let n = ((sweep.abs() * r).ceil() as usize).max(2);
    (0..=n).map(|i| c + r * unit(a0 + sweep * i as f64 / n as f64)).collect()
}

/// The turning bulb of a dead end: from `p0` (heading `h`) round to `p3`, `d` m to the left
/// and heading back; returns the points and how far ahead of `p0` it reaches.
fn u_turn(p0: DVec3, h: f64, p3: DVec3, d: f64) -> (Vec<DVec3>, f64) {
    let r = MIN_TURN_RADIUS;
    let (x, y) = (unit(h), -right(h));
    let o = p0.truncate();
    let pts: Vec<DVec2>;
    let reach;
    let end = p3.truncate();
    if d >= 2.0 * r {
        // Room for a plain half circle.
        let c = o + y * (0.5 * d);
        pts = arc(c, o, end, true);
        reach = 0.5 * d;
    } else {
        // Out to the right, round the bulb, back in from the left.
        let c1 = o - y * r;
        let c3 = end + y * r;
        let dy = 0.5 * d + r;
        let xc = ((2.0 * r).powi(2) - dy * dy).max(0.0).sqrt();
        let c2 = o + x * xc + y * (0.5 * d);
        let t1 = c1 + (c2 - c1).normalize() * r;
        let t3 = c3 + (c2 - c3).normalize() * r;
        let mut p = arc(c1, o, t1, false);
        p.extend(arc(c2, t1, t3, true).into_iter().skip(1));
        p.extend(arc(c3, t3, end, false).into_iter().skip(1));
        pts = p;
        reach = xc + r;
    }
    let n = pts.len() - 1;
    let out = pts.iter().enumerate().map(|(i, q)| q.extend(p0.z + (p3.z - p0.z) * i as f64 / n as f64)).collect();
    (out, reach)
}

/// The conflict zone of connectors `a` and `b`: the stations along `a` and along `b` (first,
/// last) closer than [`CONFLICT_GAP`] (more for [`CRAMPED`] lanes) to the other line, if any.
fn conflict_zone(lanes: &[Lane], a: &Connector, b: &Connector) -> Option<([f64; 2], [f64; 2])> {
    let cramped = |c: &Connector| [c.from, c.to].iter().any(|&l| lanes[l as usize].line.length() < CRAMPED);
    let g = CONFLICT_GAP + if cramped(a) || cramped(b) { CRAMPED_MARGIN } else { 0.0 };
    let (a, b) = (&a.line, &b.line);
    let bbox = |p: &[DVec3]| {
        p.iter().fold((DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)), |(lo, hi), q| {
            (lo.min(q.truncate()), hi.max(q.truncate()))
        })
    };
    let (la, ha) = bbox(a.points());
    let (lb, hb) = bbox(b.points());
    if la.x > hb.x + g || lb.x > ha.x + g || la.y > hb.y + g || lb.y > ha.y + g {
        return None;
    }
    // Samples of each line near the other, and where they project onto it: z[0] along `a`,
    // z[1] along `b`.
    let mut z = [[f64::INFINITY, f64::NEG_INFINITY]; 2];
    for (x, y, k) in [(a, b, 0), (b, a, 1)] {
        let n = (x.length() / ZONE_STEP).ceil().max(1.0) as usize;
        for i in 0..=n {
            let s = x.length() * i as f64 / n as f64;
            let pr = y.project(x.point_at(s).truncate());
            if pr.distance < g {
                z[k] = [z[k][0].min(s), z[k][1].max(s)];
                z[1 - k] = [z[1 - k][0].min(pr.station), z[1 - k][1].max(pr.station)];
            }
        }
    }
    let [za, zb] = z;
    (za[0] <= za[1]).then_some((za, zb))
}

/// Whether two polylines cross.
fn crosses(a: &Polyline, b: &Polyline) -> bool {
    a.points().windows(2).any(|w| {
        b.points().windows(2).any(|v| intersect(w[0].truncate(), w[1].truncate(), v[0].truncate(), v[1].truncate()))
    })
}

/// Whether segments `p–q` and `a–b` cross.
fn intersect(p: DVec2, q: DVec2, a: DVec2, b: DVec2) -> bool {
    let (r, s) = (q - p, b - a);
    let den = r.perp_dot(s);
    if den.abs() < 1e-12 {
        return false;
    }
    let t = (a - p).perp_dot(s) / den;
    let u = (a - p).perp_dot(r) / den;
    (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)
}

/// Maximum |curvature| of a polyline (three consecutive points).
pub fn max_curvature(line: &Polyline) -> f64 {
    line.points()
        .windows(3)
        .map(|w| {
            let (a, b, c) = (w[0].truncate(), w[1].truncate(), w[2].truncate());
            let den = a.distance(b) * b.distance(c) * a.distance(c);
            if den < 1e-12 { 0.0 } else { (2.0 * (b - a).perp_dot(c - b) / den).abs() }
        })
        .fold(0.0, f64::max)
}

impl LaneGraph {
    /// Build the lane graph of `roads` between `nodes`, with each road's section.
    pub(crate) fn build(nodes: &[RoadNode], roads: &[Road], section: &dyn Fn(usize) -> Section) -> Self {
        if roads.is_empty() {
            return Self::default();
        }
        let sections: Vec<Section> = (0..roads.len()).map(section).collect();
        let layouts: Vec<Vec<(u8, u8, f64, f64)>> =
            roads.iter().zip(&sections).map(|(r, s)| lane_layout(r, s)).collect();
        // Road ends at each node.
        let mut ends: Vec<Vec<End>> = vec![Vec::new(); nodes.len()];
        for (i, r) in roads.iter().enumerate() {
            let len = r.line.length();
            ends[r.start as usize].push(End { road: i as u32, start: true, heading: r.line.heading_at(0.0) });
            ends[r.end as usize].push(End {
                road: i as u32,
                start: false,
                heading: wrap_angle(r.line.heading_at(len) + PI),
            });
        }
        // The half width each road claims beyond its centre line (carriageway and sidewalk).
        let claim = |i: usize| {
            let s = &sections[i];
            0.5 * roads[i].width + if roads[i].class.is_urban() { s.sidewalk[0].max(s.sidewalk[1]) } else { 0.0 }
        };
        let ring = |i: usize| {
            let r = &roads[i];
            sections[i].one_way()
                && nodes[r.start as usize].kind == NodeKind::Roundabout
                && nodes[r.end as usize].kind == NodeKind::Roundabout
        };

        // Setbacks: clear of the other roads at a junction, room to turn round at a dead end
        // and to turn between the lanes' ends (so a few passes, with the roads' headings where
        // the lanes end).
        let mut setbacks = vec![[0.0f64; 2]; roads.len()];
        let heading_out = |e: &End, s: f64| {
            let r = &roads[e.road as usize];
            if e.start { r.line.heading_at(s) } else { wrap_angle(r.line.heading_at(r.line.length() - s) + PI) }
        };
        for _ in 0..3 {
            let mut next = setbacks.clone();
            for list in &ends {
                for e in list {
                    let i = e.road as usize;
                    let side = usize::from(!e.start);
                    let he = heading_out(e, setbacks[i][side]);
                    let need = match list.len() {
                        1 => {
                            // The bulb from the rightmost lane in to the rightmost lane out.
                            let (rin, rout) = rightmost(&layouts[i], e.in_dir(), e.out_dir());
                            let d = rin + rout;
                            let r = MIN_TURN_RADIUS;
                            if d >= 2.0 * r {
                                1.0 + 0.5 * d
                            } else {
                                let dy = 0.5 * d + r;
                                1.0 + ((2.0 * r).powi(2) - dy * dy).max(0.0).sqrt() + r
                            }
                        }
                        n => list
                            .iter()
                            .filter(|o| o.road != e.road || o.start != e.start)
                            .map(|o| {
                                // Room for the lanes to turn into each other at the turning radius.
                                let ho = heading_out(o, setbacks[o.road as usize][usize::from(!o.start)]);
                                let turn = PI - wrap_angle(ho - he).abs();
                                // A lane of this road meets a lane of the other where their lines
                                // cross, the other lane's offset out from the node; the arc's
                                // tangent runs R·tan(turn/2) on from there.
                                let lo = &layouts[o.road as usize];
                                let radius = if turn < 5.0 * PI / 6.0 {
                                    reach(lo) + DESIGN_TURN_RADIUS * (0.5 * turn).tan()
                                } else {
                                    0.0
                                } + 0.5;
                                // Straight on, room to shift sideways between lanes at different
                                // offsets (an S bend of shift d over length L curves at most 6·d/L²,
                                // half of L on each side).
                                let radius = if n == 2 || turn < FRAC_PI_6 {
                                    let shift = straight_shift(&layouts[i], e.in_dir(), lo, o.out_dir())
                                        .max(straight_shift(lo, o.in_dir(), &layouts[i], e.out_dir()));
                                    radius + 0.5 * (6.0 * shift * DESIGN_TURN_RADIUS).sqrt()
                                } else {
                                    radius
                                };
                                // Clear of the other roads' carriageways and sidewalks.
                                let clear = if n > 2 {
                                    let phi = wrap_angle(o.heading - e.heading);
                                    clear_of(claim(o.road as usize), 0.5 * roads[i].width, phi) + 1.5
                                } else {
                                    1.0
                                };
                                radius.max(clear)
                            })
                            .fold(0.0, f64::max),
                    };
                    next[i][side] = need.max(setbacks[i][side]);
                }
            }
            for (i, r) in roads.iter().enumerate() {
                let len = r.line.length();
                let [a, b] = next[i];
                // Leave at least a metre of lane.
                let scale = ((len - 1.0) / (a + b)).clamp(0.0, 1.0);
                next[i] = [a * scale, b * scale];
            }
            setbacks = next;
        }

        // Then more room wherever a connector still turns tighter than a car can, as long as
        // that helps (a connector's shape can be set by the other road's lanes alone).
        let mut pass = 0;
        // Per connector: its curvature when last grown for.
        let mut last: Vec<Option<f64>> = Vec::new();
        loop {
            let g = Self::assemble(nodes, roads, &layouts, &ends, setbacks.clone(), &ring);
            pass += 1;
            let mut grown = false;
            let curvature: Vec<f64> = g.connectors.iter().map(|c| max_curvature(&c.line)).collect();
            let mut grew = vec![None; curvature.len()];
            if pass < 6 {
                for (n, c) in g.connectors.iter().enumerate() {
                    let k = curvature[n];
                    // A dead end's bulb keeps its shape wherever the lanes end.
                    if c.turn == Turn::UTurn
                        || k <= 1.0 / MIN_TURN_RADIUS
                        || last.get(n).copied().flatten().is_some_and(|l| k > l - 1e-3)
                    {
                        continue;
                    }
                    grew[n] = Some(k);
                    let (a, b) = (&g.lanes[c.from as usize], &g.lanes[c.to as usize]);
                    for (road, side) in [(a.road, usize::from(a.dir == 0)), (b.road, usize::from(b.dir == 1))] {
                        let i = road as usize;
                        let len = roads[i].line.length();
                        if setbacks[i][0] + setbacks[i][1] < len - 1.0 - 1e-6 {
                            setbacks[i][side] = setbacks[i][side] * 1.25 + 1.0;
                            let [x, y] = setbacks[i];
                            let scale = ((len - 1.0) / (x + y)).clamp(0.0, 1.0);
                            setbacks[i] = [x * scale, y * scale];
                            grown = true;
                        }
                    }
                }
            }
            last = grew;
            if !grown {
                let degree: Vec<usize> = ends.iter().map(Vec::len).collect();
                let rings: Vec<bool> = (0..roads.len()).map(ring).collect();
                return g.with_crossings(roads, &sections, &degree, rings).with_signals();
            }
        }
    }

    /// The lanes, connectors and junctions for the given setbacks.
    fn assemble(
        nodes: &[RoadNode],
        roads: &[Road],
        layouts: &[Vec<(u8, u8, f64, f64)>],
        ends: &[Vec<End>],
        setbacks: Vec<[f64; 2]>,
        ring: &dyn Fn(usize) -> bool,
    ) -> Self {
        // Lanes.
        let mut lanes: Vec<Lane> = Vec::new();
        let mut lane_ids: Vec<Vec<Vec<u32>>> = Vec::with_capacity(roads.len()); // [road][dir][index]
        for (i, r) in roads.iter().enumerate() {
            let len = r.line.length();
            let [a, b] = setbacks[i];
            // Evenly spaced, so the offset lines have no stubs that could turn back.
            let m = ((len - a - b).ceil() as usize).max(1);
            let centre: Vec<DVec3> =
                (0..=m).map(|k| r.line.point_at(a + (len - a - b) * k as f64 / m as f64)).collect();
            let speed = if ring(i) { 30.0 / 3.6 } else { speed_limit(r.class) };
            let mut ids = vec![Vec::new(), Vec::new()];
            for &(dir, index, offset, width) in &layouts[i] {
                let pts = if dir == 0 {
                    offset_line(&centre, offset)
                } else {
                    let mut rev = centre.clone();
                    rev.reverse();
                    offset_line(&rev, offset)
                };
                let (from_node, to_node) = if dir == 0 { (r.start, r.end) } else { (r.end, r.start) };
                ids[dir as usize].push(lanes.len() as u32);
                lanes.push(Lane {
                    road: i as u32,
                    dir,
                    index,
                    width,
                    speed,
                    offset,
                    line: Polyline::new(pts),
                    from_node,
                    to_node,
                    left: None,
                    right: None,
                    successors: Vec::new(),
                    predecessors: Vec::new(),
                });
            }
            for dir_ids in &ids {
                for k in 0..dir_ids.len() {
                    let id = dir_ids[k] as usize;
                    // Shared lanes (gravel, tracks) have no neighbours.
                    if k > 0 {
                        lanes[id].left = Some(dir_ids[k - 1]);
                    }
                    if k + 1 < dir_ids.len() {
                        lanes[id].right = Some(dir_ids[k + 1]);
                    }
                }
            }
            lane_ids.push(ids);
        }

        // Junctions and connectors.
        let mut connectors: Vec<Connector> = Vec::new();
        let mut junctions = Vec::with_capacity(nodes.len());
        for (n, list) in ends.iter().enumerate() {
            let kind = junction_kind(nodes, roads, list, &ring);
            let first = connectors.len();
            // Approaches: the lanes arriving here.
            let mut approaches = Vec::new();
            let top = list.iter().map(|e| class_rank(roads[e.road as usize].class)).max().unwrap_or(0);
            for e in list {
                let ids = &lane_ids[e.road as usize][e.in_dir() as usize];
                if ids.is_empty() {
                    continue;
                }
                let (l0, l1) = (&lanes[ids[0] as usize], &lanes[ids[ids.len() - 1] as usize]);
                let edge = |l: &Lane, side: f64| {
                    let p = *l.line.points().last().expect("points");
                    let h = l.line.heading_at(l.line.length());
                    (p.truncate() + right(h) * side * 0.5 * l.width).extend(p.z)
                };
                let rank = class_rank(roads[e.road as usize].class);
                let control = match kind {
                    JunctionKind::Signal => Control::Signal,
                    JunctionKind::Stop if rank < top => Control::Stop,
                    JunctionKind::Yield if rank < top => Control::Yield,
                    JunctionKind::Roundabout if !ring(e.road as usize) => Control::Yield,
                    _ => Control::None,
                };
                approaches.push(Approach {
                    road: e.road,
                    dir: e.in_dir(),
                    lanes: ids.clone(),
                    control,
                    stop_line: [edge(l0, -1.0), edge(l1, 1.0)],
                });
            }
            for e in list {
                let ins = &lane_ids[e.road as usize][e.in_dir() as usize];
                let n_in = ins.len();
                // Without a way straight on, the left half of the lanes turns left and the
                // right half right (all of them one way when the other has no exit), so that
                // every lane leads somewhere.
                let h_in = ins.first().map_or(0.0, |&l| {
                    let l = &lanes[l as usize];
                    l.line.heading_at(l.line.length())
                });
                let exits: Vec<Turn> = list
                    .iter()
                    .filter(|o| !(o.road == e.road && o.start == e.start))
                    .filter_map(|o| {
                        let outs = &lane_ids[o.road as usize][o.out_dir() as usize];
                        let theta = wrap_angle(lanes[*outs.first()? as usize].line.heading_at(0.0) - h_in);
                        Some(if list.len() == 2 || theta.abs() < FRAC_PI_6 {
                            Turn::Straight
                        } else if theta > 0.0 {
                            Turn::Left
                        } else {
                            Turn::Right
                        })
                    })
                    .collect();
                let straight_on = exits.contains(&Turn::Straight);
                let (lefts, rights) = (exits.contains(&Turn::Left), exits.contains(&Turn::Right));
                for (i, &lin) in ins.iter().enumerate() {
                    let l = &lanes[lin as usize];
                    let p0 = *l.line.points().last().expect("points");
                    let h0 = l.line.heading_at(l.line.length());
                    for o in list {
                        let outs = &lane_ids[o.road as usize][o.out_dir() as usize];
                        if outs.is_empty() {
                            continue;
                        }
                        let same = o.road == e.road && o.start == e.start;
                        let m = outs.len();
                        let targets: Vec<(u32, Turn)> = if same {
                            // Turning round only at dead ends, into the kerb lane.
                            if list.len() != 1 {
                                continue;
                            }
                            vec![(outs[m - 1], Turn::UTurn)]
                        } else {
                            let h3 = lanes[outs[0] as usize].line.heading_at(0.0);
                            let theta = wrap_angle(h3 - h0);
                            if list.len() > 2 && theta.abs() > 5.0 * PI / 6.0 {
                                continue;
                            }
                            let turn = if list.len() == 2 || theta.abs() < FRAC_PI_6 {
                                Turn::Straight
                            } else if theta > 0.0 {
                                Turn::Left
                            } else {
                                Turn::Right
                            };
                            match turn {
                                // Left turns from the leftmost lane, right turns from the rightmost
                                // (single lanes do everything).
                                Turn::Left if n_in == 1 || i == 0 || !straight_on && (2 * i < n_in || !rights) => {
                                    vec![(outs[0], turn)]
                                }
                                Turn::Right
                                    if n_in == 1 || i + 1 == n_in || !straight_on && (2 * i >= n_in || !lefts) =>
                                {
                                    vec![(outs[m - 1], turn)]
                                }
                                Turn::Straight => {
                                    let mut t = vec![(outs[i.min(m - 1)], turn)];
                                    // Added lanes are fed from the rightmost lane.
                                    if i + 1 == n_in {
                                        t.extend((n_in..m).map(|j| (outs[j], turn)));
                                    }
                                    t
                                }
                                _ => Vec::new(),
                            }
                        };
                        for (lout, turn) in targets {
                            let q = &lanes[lout as usize];
                            let p3 = q.line.points()[0];
                            let h3 = q.line.heading_at(0.0);
                            let pts = if turn == Turn::UTurn {
                                let d = (p3.truncate() - p0.truncate()).dot(-right(h0)).max(0.0);
                                u_turn(p0, h0, p3, d).0
                            } else {
                                // The smoother of the two shapes.
                                let (a, b) = (connect(p0, h0, p3, h3), bezier(p0, h0, p3, h3));
                                let k = |p: &[DVec3]| max_curvature(&Polyline::new(p.to_vec()));
                                if k(&b) < k(&a) { b } else { a }
                            };
                            let line = Polyline::new(pts);
                            let k = max_curvature(&line);
                            let limit = if k > 1e-9 { (LATERAL_ACCEL / k).sqrt() } else { f64::INFINITY };
                            let speed = l.speed.min(q.speed).min(limit);
                            connectors.push(Connector {
                                from: lin,
                                to: lout,
                                node: n as u32,
                                turn,
                                speed,
                                line,
                                conflicts: Vec::new(),
                            });
                        }
                    }
                }
            }
            // Conflicts between the junction's connectors.
            let ids: Vec<u32> = (first as u32..connectors.len() as u32).collect();
            for (x, &a) in ids.iter().enumerate() {
                for &b in &ids[x + 1..] {
                    let (ca, cb) = (&connectors[a as usize], &connectors[b as usize]);
                    let Some((za, zb)) = conflict_zone(&lanes, ca, cb) else { continue };
                    let ck = if ca.from == cb.from {
                        ConflictKind::Diverge
                    } else if ca.to == cb.to {
                        ConflictKind::Merge
                    } else if crosses(&ca.line, &cb.line) {
                        ConflictKind::Cross
                    } else {
                        ConflictKind::Near
                    };
                    let a_yields = yields(kind, roads, &lanes, ca, cb, a, b, &ring);
                    let diverge = ck == ConflictKind::Diverge;
                    connectors[a as usize].conflicts.push(Conflict {
                        other: b,
                        kind: ck,
                        station: za[0],
                        length: za[1] - za[0],
                        other_station: zb[0],
                        other_length: zb[1] - zb[0],
                        yields: a_yields && !diverge,
                    });
                    connectors[b as usize].conflicts.push(Conflict {
                        other: a,
                        kind: ck,
                        station: zb[0],
                        length: zb[1] - zb[0],
                        other_station: za[0],
                        other_length: za[1] - za[0],
                        yields: !a_yields && !diverge,
                    });
                }
            }
            let radius = list.iter().map(|e| setbacks[e.road as usize][usize::from(!e.start)]).fold(0.0, f64::max);
            junctions.push(Junction { node: n as u32, kind, radius, approaches, connectors: ids });
        }
        // Conflicts between connectors of different junctions (close ones joined by short
        // lanes), given way as at an uncontrolled junction.
        let bbox = |c: &Connector| {
            c.line
                .points()
                .iter()
                .fold((DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)), |(lo, hi), q| {
                    (lo.min(q.truncate()), hi.max(q.truncate()))
                })
        };
        let boxes: Vec<(DVec2, DVec2)> = connectors.iter().map(bbox).collect();
        for a in 0..connectors.len() {
            for b in a + 1..connectors.len() {
                let (ca, cb) = (&connectors[a], &connectors[b]);
                let ((la, ha), (lb, hb)) = (boxes[a], boxes[b]);
                let g = CONFLICT_GAP + CRAMPED_MARGIN;
                if ca.node == cb.node
                    || ca.to == cb.from
                    || cb.to == ca.from
                    || la.x > hb.x + g
                    || lb.x > ha.x + g
                    || la.y > hb.y + g
                    || lb.y > ha.y + g
                {
                    continue;
                }
                let Some((za, zb)) = conflict_zone(&lanes, ca, cb) else { continue };
                let ck = if crosses(&ca.line, &cb.line) { ConflictKind::Cross } else { ConflictKind::Near };
                let (a, b) = (a as u32, b as u32);
                let a_yields = yields(JunctionKind::Uncontrolled, roads, &lanes, ca, cb, a, b, &ring);
                connectors[a as usize].conflicts.push(Conflict {
                    other: b,
                    kind: ck,
                    station: za[0],
                    length: za[1] - za[0],
                    other_station: zb[0],
                    other_length: zb[1] - zb[0],
                    yields: a_yields,
                });
                connectors[b as usize].conflicts.push(Conflict {
                    other: a,
                    kind: ck,
                    station: zb[0],
                    length: zb[1] - zb[0],
                    other_station: za[0],
                    other_length: za[1] - za[0],
                    yields: !a_yields,
                });
            }
        }
        for (c, conn) in connectors.iter().enumerate() {
            lanes[conn.from as usize].successors.push(c as u32);
            lanes[conn.to as usize].predecessors.push(c as u32);
        }
        let grid = SegmentGrid::of_lines(&lanes.iter().map(|l| &l.line).collect::<Vec<_>>());
        Self { lanes, connectors, junctions, setbacks, grid, ..Self::default() }
    }

    /// With the pedestrian crossings over its roads (`degree`: the number of road ends at
    /// each node; `rings`: whether each road is a piece of a roundabout's ring).
    fn with_crossings(mut self, roads: &[Road], sections: &[Section], degree: &[usize], rings: Vec<bool>) -> Self {
        let mut crossings = Vec::new();
        for (i, r) in roads.iter().enumerate() {
            let s = &sections[i];
            if !r.class.is_urban() || s.sidewalk[0].max(s.sidewalk[1]) <= 0.0 || rings[i] {
                continue;
            }
            let len = r.line.length();
            let [sa, sb] = self.setbacks[i];
            let half = 0.5 * CROSSWALK;
            let mut at = Vec::new();
            let start = degree[r.start as usize] >= 3;
            let end = degree[r.end as usize] >= 3;
            if start {
                at.push((sa + half, Some(r.start)));
            }
            if end && len - sb - half > sa + half + CROSSWALK {
                at.push((len - sb - half, Some(r.end)));
            }
            if s.sidewalk[0].min(s.sidewalk[1]) > 0.0 && s.parking[0].max(s.parking[1]) <= 0.0 {
                let a = sa + if start { CROSSWALK } else { 0.0 };
                let b = len - sb - if end { CROSSWALK } else { 0.0 };
                let n = ((b - a) / MID_BLOCK_SPACING).floor() as usize;
                for k in 0..n {
                    at.push((a + (b - a) * (k as f64 + 1.0 / 3.0) / n as f64, None));
                }
            }
            for (station, node) in at {
                if station < half || station > len - half {
                    continue;
                }
                let centre = r.line.point_at(station).truncate();
                let lanes = self
                    .lanes
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.road == i as u32)
                    .map(|(k, l)| (k as u32, l.line.project(centre).station))
                    .collect();
                crossings.push(Crossing { road: i as u32, station, node, length: r.width, lanes, signal: None });
            }
        }
        crossings.sort_by(|a, b| (a.road, a.station).partial_cmp(&(b.road, b.station)).expect("finite"));
        self.road_crossings = vec![Vec::new(); roads.len()];
        for (k, c) in crossings.iter().enumerate() {
            self.road_crossings[c.road as usize].push(k as u32);
        }
        self.crossings = crossings;
        self.rings = rings;
        self
    }

    /// With the signal controllers of its signalized junctions, and the phases in which
    /// pedestrians may walk over the crossings there.
    fn with_signals(mut self) -> Self {
        let (controllers, of) = signals::build(&self.lanes, &self.connectors, &self.junctions);
        self.junction_controllers = vec![None; self.junctions.len()];
        for (k, c) in controllers.iter().enumerate() {
            self.junction_controllers[c.junction as usize] = Some(k as u32);
        }
        for c in &mut self.crossings {
            let Some(k) = c.node.and_then(|n| self.junction_controllers[n as usize]) else { continue };
            let lanes: Vec<u32> = c.lanes.iter().map(|l| l.0).collect();
            c.signal = Some((k, signals::walk_phases(&controllers[k as usize], &self.connectors, &lanes)));
        }
        // No crossing where pedestrians would never walk.
        self.crossings.retain(|c| c.signal.is_none_or(|s| s.1 != 0));
        for list in &mut self.road_crossings {
            list.clear();
        }
        for (k, c) in self.crossings.iter().enumerate() {
            self.road_crossings[c.road as usize].push(k as u32);
        }
        self.controllers = controllers;
        self.connector_signals = of;
        self
    }

    /// The pedestrian crossings, by road and station.
    pub fn crossings(&self) -> &[Crossing] {
        &self.crossings
    }

    /// The crossings over road `i`, by station.
    pub fn road_crossings(&self, i: usize) -> &[u32] {
        self.road_crossings.get(i).map_or(&[], Vec::as_slice)
    }

    /// Whether pedestrians may start over crossing `k` at time `t` (s), and how long that
    /// stays so (s); None for unsignalized crossings.
    pub fn crossing_walk(&self, k: u32, t: f64) -> Option<(bool, f64)> {
        let c = &self.crossings[k as usize];
        c.signal.map(|(ctl, phases)| self.controllers[ctl as usize].walk(phases, c.length, t))
    }

    /// Whether road `i` is a piece of a roundabout's ring.
    pub fn is_ring(&self, i: usize) -> bool {
        self.rings.get(i).copied().unwrap_or(false)
    }

    pub fn lanes(&self) -> &[Lane] {
        &self.lanes
    }

    pub fn connectors(&self) -> &[Connector] {
        &self.connectors
    }

    /// One per node of the network.
    pub fn junctions(&self) -> &[Junction] {
        &self.junctions
    }

    /// What traffic in `lane` must do at its end before entering the junction there.
    pub fn control(&self, lane: u32) -> Control {
        let j = &self.junctions[self.lanes[lane as usize].to_node as usize];
        j.approaches.iter().find(|a| a.lanes.contains(&lane)).map_or(Control::None, |a| a.control)
    }

    /// Setbacks (m) of the lanes from the start and the end of road `i`.
    pub fn setbacks(&self, i: usize) -> [f64; 2] {
        self.setbacks[i]
    }

    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    /// The signal controllers, one per signalized junction.
    pub fn controllers(&self) -> &[Controller] {
        &self.controllers
    }

    /// The (controller, phase) that lets connector `c` go, if it is signalled.
    pub fn connector_signal(&self, c: u32) -> Option<(u32, u8)> {
        self.connector_signals.get(c as usize).copied().flatten()
    }

    /// The controller of junction `j`, if it is signalized.
    pub fn junction_controller(&self, j: u32) -> Option<u32> {
        self.junction_controllers.get(j as usize).copied().flatten()
    }

    /// The lanes outside the largest strongly connected component (over connectors and lane
    /// changes): those from which some other lane cannot be reached or back. Empty when every
    /// lane reaches every other.
    pub fn stranded(&self) -> Vec<u32> {
        let n = self.lanes.len();
        let next = |u: usize| -> Vec<usize> {
            let l = &self.lanes[u];
            let mut v: Vec<usize> = l.successors.iter().map(|&c| self.connectors[c as usize].to as usize).collect();
            v.extend([l.left, l.right].into_iter().flatten().map(|x| x as usize));
            v
        };
        let prev = |u: usize| -> Vec<usize> {
            let l = &self.lanes[u];
            let mut v: Vec<usize> = l.predecessors.iter().map(|&c| self.connectors[c as usize].from as usize).collect();
            v.extend([l.left, l.right].into_iter().flatten().map(|x| x as usize));
            v
        };
        // Kosaraju: finishing order forwards, then components backwards.
        let mut order = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        for root in 0..n {
            if seen[root] {
                continue;
            }
            seen[root] = true;
            let mut stack = vec![(root, next(root), 0usize)];
            while let Some((u, out, k)) = stack.last_mut() {
                if let Some(&v) = out.get(*k) {
                    *k += 1;
                    if !seen[v] {
                        seen[v] = true;
                        let o = next(v);
                        stack.push((v, o, 0));
                    }
                } else {
                    order.push(*u);
                    stack.pop();
                }
            }
        }
        let mut comp = vec![u32::MAX; n];
        let mut sizes = Vec::new();
        for &root in order.iter().rev() {
            if comp[root] != u32::MAX {
                continue;
            }
            let id = sizes.len() as u32;
            comp[root] = id;
            let mut size = 1;
            let mut stack = vec![root];
            while let Some(u) = stack.pop() {
                for v in prev(u) {
                    if comp[v] == u32::MAX {
                        comp[v] = id;
                        size += 1;
                        stack.push(v);
                    }
                }
            }
            sizes.push(size);
        }
        // The largest component (the first of equal ones).
        let Some(main) = (0..sizes.len()).max_by(|&a, &b| sizes[a].cmp(&sizes[b]).then(b.cmp(&a))) else {
            return Vec::new();
        };
        (0..n as u32).filter(|&i| comp[i as usize] != main as u32).collect()
    }

    /// The lane nearest to `p` within `max_dist` whose direction lies within 90° of `heading`
    /// (any direction when `None`): (lane, station, signed offset, positive to the left).
    pub fn nearest_lane(&self, p: DVec2, max_dist: f64, heading: Option<f64>) -> Option<(u32, f64, f64)> {
        let mut best: Option<(f64, u32, u32, f64)> = None;
        self.grid.visit(p, max_dist, &mut |lane, seg| {
            let line = &self.lanes[lane as usize].line;
            let (s, d2) = line.project_segment(seg as usize, p);
            let fits = heading.is_none_or(|h| wrap_angle(line.heading_at(s) - h).abs() <= FRAC_PI_2);
            let better = match best {
                None => d2 <= max_dist * max_dist,
                Some((b, l, g, _)) => d2 < b || (d2 == b && (lane, seg) < (l, g)),
            };
            if fits && better {
                best = Some((d2, lane, seg, s));
            }
            best.map_or(max_dist, |b| b.0.sqrt())
        });
        best.map(|(_, lane, seg, s)| {
            let pr = self.lanes[lane as usize].line.projection(seg as usize, s, p);
            (lane, s, pr.offset)
        })
    }

    /// Shortest route over lanes, connectors and lane changes from station `s0` of lane `a` to
    /// station `s1` of lane `b` (Dijkstra; ties go to lower ids). A lane change costs
    /// [`LANE_CHANGE_COST`] m on top.
    pub fn route(&self, a: u32, s0: f64, b: u32, s1: f64) -> Option<LaneRoute> {
        if a == b && s1 >= s0 {
            return Some(LaneRoute { steps: vec![RouteStep::Lane(a)], length: s1 - s0 });
        }
        // Graph nodes: lanes (at their end), then connectors (at their end).
        let nl = self.lanes.len();
        let n = nl + self.connectors.len();
        let mut dist = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        // Lanes reached from the first one by lane changes only (still alongside `s0`).
        let mut fresh = vec![false; n];
        let mut heap = BinaryHeap::new();
        let (a, b) = (a as usize, b as usize);
        fresh[a] = true;
        dist[a] = self.lanes[a].line.length() - s0;
        heap.push(Entry(dist[a], a as u32));
        // (length, the node before arriving on `b`).
        let mut best: Option<(f64, usize)> = None;
        let len_b = self.lanes[b].line.length();
        while let Some(Entry(d, u)) = heap.pop() {
            let u = u as usize;
            if d > dist[u] || best.is_some_and(|x| d >= x.0) {
                continue;
            }
            let mut next: Vec<(usize, f64)> = Vec::new();
            if u < nl {
                let lane = &self.lanes[u];
                next.extend(
                    lane.successors.iter().map(|&c| (nl + c as usize, self.connectors[c as usize].line.length())),
                );
                next.extend([lane.left, lane.right].into_iter().flatten().map(|l| (l as usize, LANE_CHANGE_COST)));
            } else {
                let to = self.connectors[u - nl].to as usize;
                next.push((to, self.lanes[to].line.length()));
            }
            for (v, w) in next {
                // A lane change keeps the station (the lanes run side by side).
                let change = u < nl && v < nl;
                if v == b && !(change && fresh[u] && s1 < s0) {
                    // Arriving on `b` (at its start from a connector, alongside after a change).
                    let total = d + w - (len_b - s1);
                    if best.is_none_or(|x| total < x.0) {
                        best = Some((total, u));
                    }
                }
                if d + w < dist[v] {
                    dist[v] = d + w;
                    prev[v] = Some(u);
                    fresh[v] = change && fresh[u];
                    heap.push(Entry(d + w, v as u32));
                }
            }
        }
        let (length, last) = best?;
        let mut chain = vec![b, last];
        let mut v = last;
        while let Some(p) = prev[v] {
            chain.push(p);
            v = p;
        }
        chain.reverse();
        let mut steps = Vec::new();
        for (k, &v) in chain.iter().enumerate() {
            if v >= nl {
                steps.push(RouteStep::Connector((v - nl) as u32));
                continue;
            }
            if k > 0 && chain[k - 1] < nl {
                steps.push(RouteStep::Change(chain[k - 1] as u32, v as u32));
            }
            steps.push(RouteStep::Lane(v as u32));
        }
        Some(LaneRoute { steps, length })
    }

    /// The area class of `p`, given the network's roads and sections.
    /// Points outside the carriageways that a connector's lane width covers are part of its
    /// junction.
    pub(crate) fn area(
        &self,
        roads: &[Road],
        section: &Section,
        road: u32,
        station: f64,
        offset: f64,
        p: DVec2,
    ) -> Area {
        let r = &roads[road as usize];
        let len = r.line.length();
        let half = 0.5 * r.width;
        let urban = r.class.is_urban();
        // Right of travel from start to end.
        let x = -offset;
        let [sa, sb] = self.setbacks[road as usize];
        let near_start = station < sa && self.junctions[r.start as usize].approaches.len() >= 3;
        let near_end = station > len - sb && self.junctions[r.end as usize].approaches.len() >= 3;
        if x.abs() <= half {
            if near_start || near_end {
                return Area::Junction;
            }
            let crosswalk = self
                .road_crossings(road as usize)
                .iter()
                .any(|&k| (station - self.crossings[k as usize].station).abs() <= 0.5 * CROSSWALK);
            if crosswalk {
                return Area::Crosswalk;
            }
            if !urban {
                return Area::Lane;
            }
            // Distance from the carriageway's left edge of the side's travel direction.
            let lw = section.lane_width;
            let (lanes, bike, from_inner) = if section.one_way() {
                (f64::from(section.lanes[0]) * lw, section.bike[0], x + half)
            } else {
                let side = usize::from(x < 0.0);
                let inner = 0.5 * section.median;
                if x.abs() < inner {
                    return Area::Median;
                }
                (f64::from(section.lanes[side]) * lw, section.bike[side], x.abs() - inner)
            };
            return if from_inner <= lanes {
                Area::Lane
            } else if from_inner <= lanes + bike {
                Area::BikeLane
            } else {
                Area::Parking
            };
        }
        // Corners that turning traffic sweeps (outside both carriageways).
        let near = |node: u32| {
            let j = &self.junctions[node as usize];
            j.approaches.len() >= 3
                && j.connectors.iter().any(|&c| {
                    let c = &self.connectors[c as usize];
                    let w = self.lanes[c.from as usize].width;
                    c.line.project(p).distance <= 0.5 * w
                })
        };
        if (station < sa + 2.0 && near(r.start)) || (station > len - sb - 2.0 && near(r.end)) {
            return Area::Junction;
        }
        let side = usize::from(x < 0.0);
        if urban && x.abs() <= half + section.sidewalk[side] {
            return Area::Sidewalk;
        }
        Area::Off
    }
}

/// Largest distance of a lane centre from the centre line.
fn reach(layout: &[(u8, u8, f64, f64)]) -> f64 {
    layout.iter().map(|l| l.2.abs()).fold(0.0, f64::max)
}

/// How far out along a road the edges of its carriageway (`half` m either side of its centre
/// line) stay within `claim` m of another road leaving the same node at `phi` rad from it
/// (the other road's strip starts at the node, so a road carrying straight on needs no room).
fn clear_of(claim: f64, half: f64, phi: f64) -> f64 {
    let (sin, cos) = phi.sin_cos();
    // Nearly parallel roads leaving together: as if they parted at 30°.
    let sin = if cos > 0.0 { sin.signum() * sin.abs().max(0.5) } else { sin };
    let mut out = 0.0f64;
    for y in [-half, half] {
        // Within the strip sideways ...
        let mut x = if sin.abs() < 1e-9 { f64::NEG_INFINITY } else { y * cos / sin + claim / sin.abs() };
        // ... and ahead of the node along the other road.
        if cos < 0.0 {
            x = x.min(-y * sin / cos);
        }
        out = out.max(x);
    }
    out
}

/// Largest sideways shift of a straight-on connector from the lanes of `a` in direction
/// `din` to those of `b` in direction `dout` (each lane seen from its own direction of
/// travel): lane i into lane i, the rightmost also into any lanes added.
fn straight_shift(a: &[(u8, u8, f64, f64)], din: u8, b: &[(u8, u8, f64, f64)], dout: u8) -> f64 {
    let ins: Vec<f64> = a.iter().filter(|l| l.0 == din).map(|l| l.2).collect();
    let outs: Vec<f64> = b.iter().filter(|l| l.0 == dout).map(|l| l.2).collect();
    let (n, m) = (ins.len(), outs.len());
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut shift = 0.0f64;
    for (i, x) in ins.iter().enumerate() {
        shift = shift.max((x - outs[i.min(m - 1)]).abs());
    }
    for y in &outs[n.min(m)..] {
        shift = shift.max((ins[n - 1] - y).abs());
    }
    shift
}

/// Offsets of the rightmost lanes arriving (`din`) and leaving (`dout`).
fn rightmost(layout: &[(u8, u8, f64, f64)], din: u8, dout: u8) -> (f64, f64) {
    let last = |d: u8| layout.iter().filter(|l| l.0 == d).map(|l| l.2).fold(f64::NEG_INFINITY, f64::max);
    let (a, b) = (last(din), last(dout));
    (if a.is_finite() { a } else { 0.0 }, if b.is_finite() { b } else { 0.0 })
}

/// How a node's traffic is controlled, from the roads meeting there.
fn junction_kind(nodes: &[RoadNode], roads: &[Road], list: &[End], ring: &dyn Fn(usize) -> bool) -> JunctionKind {
    match list.len() {
        0 | 1 => return JunctionKind::DeadEnd,
        2 if !list.iter().any(|e| ring(e.road as usize)) => return JunctionKind::Through,
        _ => {}
    }
    let _ = nodes;
    if list.iter().any(|e| ring(e.road as usize)) {
        return JunctionKind::Roundabout;
    }
    let mut ranks: Vec<u8> = list.iter().map(|e| class_rank(roads[e.road as usize].class)).collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a));
    let urban = list.iter().any(|e| roads[e.road as usize].class.is_urban());
    let (top, low) = (ranks[0], ranks[ranks.len() - 1]);
    let collector = class_rank(RoadClass::Collector);
    // Lights where an arterial meets another major road (its own two arms rank first).
    if urban && top == class_rank(RoadClass::Arterial) && ranks[2] >= collector {
        JunctionKind::Signal
    } else if top > low {
        if urban { JunctionKind::Stop } else { JunctionKind::Yield }
    } else {
        JunctionKind::Uncontrolled
    }
}

/// Whether connector `ca` (id `a`) gives way to `cb` (id `b`) where they meet.
#[allow(clippy::too_many_arguments)]
fn yields(
    kind: JunctionKind,
    roads: &[Road],
    lanes: &[Lane],
    ca: &Connector,
    cb: &Connector,
    a: u32,
    b: u32,
    ring: &dyn Fn(usize) -> bool,
) -> bool {
    let (la, lb) = (&lanes[ca.from as usize], &lanes[cb.from as usize]);
    if kind == JunctionKind::Roundabout {
        let (ra, rb) = (ring(la.road as usize), ring(lb.road as usize));
        if ra != rb {
            return !ra;
        }
    }
    let (ka, kb) = (class_rank(roads[la.road as usize].class), class_rank(roads[lb.road as usize].class));
    if ka != kb {
        return ka < kb;
    }
    // Turning across traffic gives way (left turns, then right turns, to going straight).
    let order = |t: Turn| match t {
        Turn::Straight => 0,
        Turn::Right => 1,
        Turn::Left => 2,
        Turn::UTurn => 3,
    };
    match order(ca.turn).cmp(&order(cb.turn)) {
        Ordering::Less => return false,
        Ordering::Greater => return true,
        Ordering::Equal => {}
    }
    // Give way to the right: `b` comes from the right when its heading is turned left of ours.
    let ha = la.line.heading_at(la.line.length());
    let hb = lb.line.heading_at(lb.line.length());
    let rel = wrap_angle(hb - ha);
    if rel > 0.1 && rel < PI - 0.1 {
        return true;
    }
    if rel < -0.1 && rel > -(PI - 0.1) {
        return false;
    }
    a > b
}

#[derive(PartialEq)]
struct Entry(f64, u32);

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then_with(|| other.1.cmp(&self.1))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roads::RoadNetwork;
    use crate::signals::Light;

    fn arterial() -> Section {
        Section { lanes: [2, 2], lane_width: 3.5, median: 2.0, bike: [0.0; 2], parking: [0.0; 2], sidewalk: [3.0; 2] }
    }

    fn street() -> Section {
        Section { lanes: [1, 1], lane_width: 3.0, median: 0.0, bike: [0.0; 2], parking: [2.0; 2], sidewalk: [2.0; 2] }
    }

    /// A cross of an east–west arterial and a north–south street of `class`, arms 80 m long
    /// ending in dead ends: roads 0 east, 1 west, 2 north, 3 south, all from the centre.
    fn cross(class: RoadClass) -> RoadNetwork {
        let at = |x: f64, y: f64, kind| RoadNode { position: DVec3::new(x, y, 0.0), kind };
        let nodes = vec![
            at(0.0, 0.0, NodeKind::Junction),
            at(80.0, 0.0, NodeKind::End),
            at(-80.0, 0.0, NodeKind::End),
            at(0.0, 80.0, NodeKind::End),
            at(0.0, -80.0, NodeKind::End),
        ];
        let road = |end: u32, class: RoadClass, s: &Section| {
            let q = nodes[end as usize].position;
            let pts = (0..=80).map(|i| q * (f64::from(i) / 80.0)).collect();
            Road { class, width: s.width(), start: 0, end, line: Polyline::new(pts) }
        };
        let roads = vec![
            road(1, RoadClass::Arterial, &arterial()),
            road(2, RoadClass::Arterial, &arterial()),
            road(3, class, &street()),
            road(4, class, &street()),
        ];
        RoadNetwork::new(nodes, roads).unwrap().with_sections(vec![arterial(), arterial(), street(), street()]).unwrap()
    }

    /// The lane of `road` in direction `dir` with `index`.
    fn lane(g: &LaneGraph, road: u32, dir: u8, index: u8) -> u32 {
        g.lanes().iter().position(|l| l.road == road && l.dir == dir && l.index == index).unwrap() as u32
    }

    fn connector(g: &LaneGraph, from: u32, to: u32) -> u32 {
        g.connectors().iter().position(|c| c.from == from && c.to == to).unwrap() as u32
    }

    #[test]
    fn lanes_connectors_and_control_of_a_cross() {
        let net = cross(RoadClass::Local);
        let g = net.lanes();
        // 2 + 2 lanes on each arterial arm, 1 + 1 on each street arm.
        assert_eq!(g.lanes().len(), 12);
        assert!(g.stranded().is_empty());
        let east = &g.lanes()[lane(g, 0, 0, 0) as usize];
        assert!((east.offset - 2.75).abs() < 1e-12 && east.left.is_none() && east.right == Some(lane(g, 0, 0, 1)));
        assert!((g.lanes()[lane(g, 0, 0, 1) as usize].offset - 6.25).abs() < 1e-12);
        assert!((east.line.points()[5].y + 2.75).abs() < 1e-9);
        assert!((east.speed - 50.0 / 3.6).abs() < 1e-12);
        // Arterial through a local street: the street stops.
        let j = &g.junctions()[0];
        assert_eq!(j.kind, JunctionKind::Stop);
        for a in &j.approaches {
            let expect = if a.road >= 2 { Control::Stop } else { Control::None };
            assert_eq!(a.control, expect);
            let w = a.stop_line[0].distance(a.stop_line[1]);
            assert!(
                (w - f64::from(net.section(a.road as usize).lanes[1]) * net.section(a.road as usize).lane_width).abs()
                    < 1e-6
            );
        }
        assert_eq!(g.junctions()[1].kind, JunctionKind::DeadEnd);
        // From the west arm: straight from both lanes, left from the inner lane only, right
        // from the outer one only.
        let (w0, w1) = (lane(g, 1, 1, 0), lane(g, 1, 1, 1));
        let turns = |l: u32| {
            let mut t: Vec<Turn> =
                g.lanes()[l as usize].successors.iter().map(|&c| g.connectors()[c as usize].turn).collect();
            t.sort_by_key(|t| *t as u8);
            t
        };
        assert_eq!(turns(w0), vec![Turn::Straight, Turn::Left]);
        assert_eq!(turns(w1), vec![Turn::Straight, Turn::Right]);
        assert_eq!(g.connectors()[connector(g, w0, lane(g, 0, 0, 0)) as usize].turn, Turn::Straight);
        // The street arms turn round at their ends.
        let u = connector(g, lane(g, 2, 0, 0), lane(g, 2, 1, 0));
        assert_eq!(g.connectors()[u as usize].turn, Turn::UTurn);
        for c in g.connectors() {
            assert!(max_curvature(&c.line) <= 1.0 / MIN_TURN_RADIUS + 1e-6, "{:?}", c.turn);
            assert!(c.line.points()[0].distance(*g.lanes()[c.from as usize].line.points().last().unwrap()) < 1e-9);
            assert!(c.line.points().last().unwrap().distance(g.lanes()[c.to as usize].line.points()[0]) < 1e-9);
        }
        // A left turn from the west yields to straight traffic from the east; a street's
        // straight crossing yields to the arterial's.
        let left = connector(g, w0, lane(g, 2, 0, 0));
        let oncoming = connector(g, lane(g, 0, 1, 0), lane(g, 1, 0, 0));
        let e = g.connectors()[left as usize].conflicts.iter().find(|e| e.other == oncoming).unwrap();
        assert!(e.yields && e.kind == ConflictKind::Cross);
        let north = connector(g, lane(g, 3, 1, 0), lane(g, 2, 0, 0));
        let across = connector(g, w1, lane(g, 0, 0, 1));
        let e = g.connectors()[north as usize].conflicts.iter().find(|e| e.other == across).unwrap();
        assert!(e.yields);
        // Right turns from the south and straight on from the west merge into the same lane.
        let right = connector(g, lane(g, 3, 1, 0), lane(g, 0, 0, 1));
        let e = g.connectors()[right as usize].conflicts.iter().find(|e| e.other == across).unwrap();
        assert_eq!(e.kind, ConflictKind::Merge);
        assert!(e.yields);
    }

    #[test]
    fn major_roads_get_lights() {
        let g = cross(RoadClass::Collector);
        let j = &g.lanes().junctions()[0];
        assert_eq!(j.kind, JunctionKind::Signal);
        assert!(j.approaches.iter().all(|a| a.control == Control::Signal));
    }

    /// Every connector of every signalized junction is in one phase, and no two connectors that
    /// conflict are green together at any time of the cycle.
    pub(crate) fn check_signals(g: &LaneGraph) {
        for (k, ctl) in g.controllers().iter().enumerate() {
            let j = &g.junctions()[ctl.junction as usize];
            assert_eq!(g.junction_controller(ctl.junction), Some(k as u32));
            for &c in &j.connectors {
                let (id, p) = g.connector_signal(c).expect("signalled");
                assert_eq!(id, k as u32);
                assert!(ctl.phases[p as usize].connectors.contains(&c));
            }
            let cycle = ctl.cycle();
            assert!((60.0 - 1e-9..=120.0).contains(&cycle), "cycle {cycle}");
            let steps = (cycle / 0.1) as usize;
            for s in 0..steps {
                let t = s as f64 * 0.1;
                let go: Vec<u32> = j
                    .connectors
                    .iter()
                    .copied()
                    .filter(|&c| {
                        let (_, p) = g.connector_signal(c).unwrap();
                        ctl.light(p as usize, t) != Light::Red
                    })
                    .collect();
                for &a in &go {
                    for e in g.connectors()[a as usize].conflicts.iter().filter(|e| e.kind.exclusive()) {
                        assert!(!go.contains(&e.other), "junction {}: {a} and {} at {t}", j.node, e.other);
                    }
                }
            }
        }
    }

    #[test]
    fn signal_phases_of_a_cross() {
        let net = cross(RoadClass::Collector);
        let g = net.lanes();
        assert_eq!(g.controllers().len(), 1);
        check_signals(g);
        let ctl = &g.controllers()[0];
        // The arterial's protected lefts, its main phase, then the street's (single lanes: the
        // lefts cross the oncoming straight movements and get their own phase too).
        let turns = |k: usize| {
            let mut t: Vec<(u32, Turn)> = ctl.phases[k]
                .connectors
                .iter()
                .map(|&c| (g.lanes()[g.connectors()[c as usize].from as usize].road, g.connectors()[c as usize].turn))
                .collect();
            t.sort_by_key(|x| (x.0, x.1 as u8));
            t.dedup();
            t
        };
        assert_eq!(ctl.phases.len(), 4, "{:?}", ctl.phases);
        assert!(ctl.phases[0].protected && !ctl.phases[1].protected);
        assert_eq!(turns(0), vec![(0, Turn::Left), (1, Turn::Left)]);
        assert_eq!(turns(1), vec![(0, Turn::Straight), (0, Turn::Right), (1, Turn::Straight), (1, Turn::Right)]);
        assert!(turns(2).iter().all(|x| x.1 == Turn::Left && x.0 >= 2));
        assert!((ctl.cycle() - 90.0).abs() < 1e-9);
        // Green, amber, then red, with the cycle wrapping round.
        let s1 = ctl.start(1);
        let green = ctl.phases[1].green;
        assert_eq!(ctl.light(1, s1 + 0.1), Light::Green);
        assert_eq!(ctl.light(1, s1 + green + 1.0), Light::Amber);
        assert_eq!(ctl.light(1, s1 + green + ctl.amber + 1.0), Light::Red);
        assert_eq!(ctl.light(1, s1 + 0.1 + 3.0 * ctl.cycle()), Light::Green);
        assert_eq!(ctl.light(0, s1 + 0.1 - ctl.cycle()), Light::Red);
        // No lights at stop junctions.
        assert!(cross(RoadClass::Local).lanes().controllers().is_empty());
    }

    #[test]
    fn lane_routes_change_lanes_to_turn() {
        let net = cross(RoadClass::Local);
        let g = net.lanes();
        // From the outer lane arriving from the west to the northern arm: over to the inner
        // lane, then left.
        let (a, b) = (lane(g, 1, 1, 1), lane(g, 2, 0, 0));
        let r = g.route(a, 10.0, b, 20.0).unwrap();
        let w0 = lane(g, 1, 1, 0);
        let c = connector(g, w0, b);
        assert_eq!(
            r.steps,
            vec![
                RouteStep::Lane(a),
                RouteStep::Change(a, w0),
                RouteStep::Lane(w0),
                RouteStep::Connector(c),
                RouteStep::Lane(b)
            ]
        );
        let expect = g.lanes()[w0 as usize].line.length() - 10.0
            + LANE_CHANGE_COST
            + g.connectors()[c as usize].line.length()
            + 20.0;
        assert!((r.length - expect).abs() < 1e-9);
        // Back along the same lane: round the dead end and back.
        let r = g.route(a, 20.0, a, 10.0).unwrap();
        assert!(
            r.steps
                .iter()
                .any(|s| matches!(s, RouteStep::Connector(c) if g.connectors()[*c as usize].turn == Turn::UTurn))
        );
        assert_eq!(g.route(a, 10.0, a, 20.0).unwrap().steps, vec![RouteStep::Lane(a)]);
    }

    #[test]
    fn areas_and_nearest_lanes() {
        let net = cross(RoadClass::Local);
        let g = net.lanes();
        let [sa, _] = g.setbacks(0);
        assert_eq!(net.area(DVec2::new(40.0, -2.75)), Area::Lane);
        assert_eq!(net.area(DVec2::new(40.0, 0.5)), Area::Median);
        assert_eq!(net.area(DVec2::new(40.0, -9.0)), Area::Sidewalk);
        assert_eq!(net.area(DVec2::new(40.0, 12.0)), Area::Off);
        assert_eq!(net.area(DVec2::new(sa + 1.5, -2.75)), Area::Crosswalk);
        assert_eq!(net.area(DVec2::new(1.0, -2.75)), Area::Junction);
        // The street's parking lane lies beyond its lane (to the right of northbound traffic).
        assert_eq!(net.area(DVec2::new(1.5, 40.0)), Area::Lane);
        assert_eq!(net.area(DVec2::new(4.0, 40.0)), Area::Parking);
        assert_eq!(net.area(DVec2::new(-6.0, 40.0)), Area::Sidewalk);
        // Nearest lane, with and without a heading.
        let (l, s, off) = g.nearest_lane(DVec2::new(40.0, -3.0), 5.0, None).unwrap();
        assert_eq!(l, lane(g, 0, 0, 0));
        assert!((off + 0.25).abs() < 1e-9 && (s - (40.0 - sa)).abs() < 1e-9);
        let (l, _, _) = g.nearest_lane(DVec2::new(40.0, -3.0), 8.0, Some(PI)).unwrap();
        assert_eq!(l, lane(g, 0, 1, 0));
        assert!(g.nearest_lane(DVec2::new(40.0, -30.0), 5.0, None).is_none());
    }

    #[test]
    fn one_way_and_shared_lanes() {
        let at = |x: f64, y: f64| RoadNode { position: DVec3::new(x, y, 0.0), kind: NodeKind::Junction };
        let nodes = vec![at(0.0, 0.0), at(100.0, 0.0)];
        let line = || Polyline::new((0..=100).map(|i| DVec3::new(f64::from(i), 0.0, 0.0)).collect());
        let one = Section {
            lanes: [2, 0],
            lane_width: 3.0,
            median: 0.0,
            bike: [1.5, 0.0],
            parking: [0.0; 2],
            sidewalk: [2.0; 2],
        };
        let roads = vec![Road { class: RoadClass::Local, width: one.width(), start: 0, end: 1, line: line() }];
        let net = RoadNetwork::new(nodes.clone(), roads).unwrap().with_sections(vec![one]).unwrap();
        let g = net.lanes();
        assert_eq!(g.lanes().len(), 2);
        let offsets: Vec<f64> = g.lanes().iter().map(|l| l.offset).collect();
        assert_eq!(offsets, vec![-2.25, 0.75]);
        assert_eq!(net.area(DVec2::new(50.0, -3.0)), Area::BikeLane);
        // Nothing can turn round at a one-way dead end.
        assert!(g.connectors().is_empty());
        // A gravel road: one shared lane each way on the centre line, turning round at the ends.
        let roads = vec![Road { class: RoadClass::Gravel, width: 4.0, start: 0, end: 1, line: line() }];
        let net = RoadNetwork::new(nodes, roads).unwrap();
        let g = net.lanes();
        assert_eq!(g.lanes().iter().map(|l| (l.dir, l.offset)).collect::<Vec<_>>(), vec![(0, 0.0), (1, 0.0)]);
        assert!(g.stranded().is_empty());
        assert_eq!(g.connectors().len(), 2);
        assert!(
            g.connectors()
                .iter()
                .all(|c| c.turn == Turn::UTurn && max_curvature(&c.line) <= 1.0 / MIN_TURN_RADIUS + 1e-6)
        );
    }
}
