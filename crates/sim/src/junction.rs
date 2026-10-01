//! Junction goals (`GoalKind::Junction`): the agents of a group start on different entries of
//! one junction of an urban map (a stop, yield or uncontrolled junction, or a roundabout) and
//! are to cross it to an exit. Each gets a lane-level route from its spawn, `distance` before
//! the entry's end, through the junction (around the ring of a roundabout) to a goal `exit` m
//! along a lane leaving it on another road; the route is the agent's for the `route` and
//! `road` terms, and a traffic driver flying the group follows its movements.
//!
//! A site is one junction, or all the junctions of a roundabout's ring (one-way roads between
//! roundabout nodes). Its entries are the approach lanes from other roads, grouped into arms
//! by road and direction; its exits are the lanes leaving it onto other roads.

use crate::scenario::{Goal, GoalSpec};
use autonomousim_core::rng::SimRng;
use autonomousim_core::terrain::Terrain;
use autonomousim_world::lanes::{JunctionKind, LaneGraph, RouteStep};
use autonomousim_world::{NodeKind, Polyline, StaticWorld};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

/// Kinds of junction for junction goals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteKind {
    Stop,
    Yield,
    Uncontrolled,
    Roundabout,
}

impl SiteKind {
    fn of(kind: JunctionKind) -> Option<Self> {
        match kind {
            JunctionKind::Stop => Some(Self::Stop),
            JunctionKind::Yield => Some(Self::Yield),
            JunctionKind::Uncontrolled => Some(Self::Uncontrolled),
            JunctionKind::Roundabout => Some(Self::Roundabout),
            _ => None,
        }
    }
}

/// Settings of `GoalKind::Junction`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JunctionGoals {
    /// Kinds of junction to use.
    pub kinds: Vec<SiteKind>,
    /// Distance of the goal along the exit lane from its start (m; at most the lane's length).
    pub exit: f64,
}

impl Default for JunctionGoals {
    fn default() -> Self {
        Self { kinds: vec![SiteKind::Stop, SiteKind::Yield, SiteKind::Uncontrolled, SiteKind::Roundabout], exit: 20.0 }
    }
}

impl JunctionGoals {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.kinds.is_empty() || !(self.exit.is_finite() && self.exit > 0.0) {
            return Err(format!("junction goals need kinds and an exit distance > 0: {self:?}"));
        }
        Ok(())
    }
}

/// Clearance of a site's junctions from the map's edges (m).
const EDGE_CLEARANCE: f64 = 80.0;

/// A junction or a roundabout: its entry lanes grouped into arms (by road and direction) and
/// its exit lanes.
#[derive(Clone, Debug, PartialEq)]
pub struct Site {
    pub kind: SiteKind,
    pub junctions: Vec<u32>,
    pub arms: Vec<Vec<u32>>,
    pub exits: Vec<u32>,
}

/// The sites of `world` of the kinds in `kinds`, clear of the map's edges, in junction order.
pub fn sites(world: &StaticWorld, kinds: &[SiteKind]) -> Vec<Site> {
    let net = world.roads();
    let g = net.lanes();
    let nodes = net.nodes();
    let ring = |road: u32| {
        let r = &net.roads()[road as usize];
        net.has_sections()
            && net.section(road as usize).one_way()
            && nodes[r.start as usize].kind == NodeKind::Roundabout
            && nodes[r.end as usize].kind == NodeKind::Roundabout
    };
    let (lo, hi) = world.extent();
    let inside = |p: DVec2| p.cmpge(lo + EDGE_CLEARANCE).all() && p.cmple(hi - EDGE_CLEARANCE).all();
    let junctions = g.junctions();
    let mut taken = vec![false; junctions.len()];
    let mut out = Vec::new();
    for (j, junction) in junctions.iter().enumerate() {
        let Some(kind) = SiteKind::of(junction.kind).filter(|k| kinds.contains(k)) else { continue };
        if taken[j] {
            continue;
        }
        // The site's junctions: this one, or its roundabout's (linked by ring lanes).
        let mut members = vec![j];
        if kind == SiteKind::Roundabout {
            let mut k = 0;
            while k < members.len() {
                let node = junctions[members[k]].node;
                for lane in g.lanes().iter().filter(|l| ring(l.road) && (l.from_node == node || l.to_node == node)) {
                    for other in [lane.from_node, lane.to_node] {
                        if let Some(m) = junctions.iter().position(|x| x.node == other)
                            && !members.contains(&m)
                        {
                            members.push(m);
                        }
                    }
                }
                k += 1;
            }
            members.sort_unstable();
        }
        for &m in &members {
            taken[m] = true;
        }
        if !members.iter().all(|&m| inside(nodes[junctions[m].node as usize].position.truncate())) {
            continue;
        }
        let mut arms: Vec<Vec<u32>> = Vec::new();
        let mut exits = Vec::new();
        for &m in &members {
            for a in junctions[m].approaches.iter().filter(|a| !ring(a.road) && !a.lanes.is_empty()) {
                arms.push(a.lanes.clone());
            }
            for &c in &junctions[m].connectors {
                let to = g.connectors()[c as usize].to;
                if !ring(g.lanes()[to as usize].road) && !exits.contains(&to) {
                    exits.push(to);
                }
            }
        }
        if arms.len() >= 2 && !exits.is_empty() {
            out.push(Site { kind, junctions: members.iter().map(|&m| m as u32).collect(), arms, exits });
        }
    }
    out
}

/// A crossing: the spawn (on the surface, heading), the goal, the route's lane line and the
/// connectors it takes.
#[derive(Clone, Debug, PartialEq)]
pub struct Crossing {
    pub position: DVec3,
    pub yaw: f64,
    pub goal: Goal,
    pub route: Polyline,
    pub connectors: Vec<u32>,
}

/// Crossings of one site with at least `count` arms for `count` agents, each from a different
/// arm (a random lane of it), `spec.distance` before its end (or from its start, if shorter),
/// to a random exit on another road reached without lane changes; goals `lift` above the
/// terrain. None when no site has enough arms.
pub fn sample(
    world: &StaticWorld,
    sites: &[Site],
    spec: &GoalSpec,
    count: usize,
    lift: f64,
    rng: &mut SimRng,
) -> Option<Vec<Crossing>> {
    let g = world.roads().lanes();
    let usable: Vec<&Site> = sites.iter().filter(|s| s.arms.len() >= count).collect();
    if usable.is_empty() {
        return None;
    }
    let site = usable[rng.below(usable.len() as u64) as usize];
    let mut arms: Vec<usize> = (0..site.arms.len()).collect();
    shuffle(&mut arms, rng);
    let mut out = Vec::with_capacity(count);
    for &arm in arms.iter() {
        if out.len() == count {
            break;
        }
        let lanes = &site.arms[arm];
        let entry = lanes[rng.below(lanes.len() as u64) as usize];
        let line = &g.lanes()[entry as usize].line;
        let d = rng.range(spec.distance[0], spec.distance[1]);
        let s0 = (line.length() - d).max(0.0);
        let mut exits: Vec<u32> = site
            .exits
            .iter()
            .copied()
            .filter(|&x| g.lanes()[x as usize].road != g.lanes()[entry as usize].road)
            .collect();
        shuffle(&mut exits, rng);
        let crossing = exits.into_iter().find_map(|exit| {
            let s1 = spec.junction.exit.min(g.lanes()[exit as usize].line.length());
            let route = g.route(entry, s0, exit, s1)?;
            let (line, connectors) = route_line(g, &route.steps, s0, s1)?;
            // Through the site only.
            connectors
                .iter()
                .all(|&c| {
                    site.junctions.iter().any(|&j| g.junctions()[j as usize].node == g.connectors()[c as usize].node)
                })
                .then_some((line, connectors))
        });
        let Some((route, connectors)) = crossing else { continue };
        let end = route.length();
        let p = route.point_at(end);
        let goal =
            Goal { position: p.truncate().extend(world.terrain().height(p.x, p.y) + lift), yaw: route.heading_at(end) };
        let start = route.point_at(0.0);
        out.push(Crossing {
            position: start.truncate().extend(world.terrain().height(start.x, start.y) + lift),
            yaw: route.heading_at(0.0),
            goal,
            route,
            connectors,
        });
    }
    (out.len() == count).then_some(out)
}

/// The lane line of a route's steps from station `s0` of the first lane to `s1` of the last,
/// and its connectors; None with lane changes.
fn route_line(g: &LaneGraph, steps: &[RouteStep], s0: f64, s1: f64) -> Option<(Polyline, Vec<u32>)> {
    let mut points: Vec<DVec3> = Vec::new();
    let mut connectors = Vec::new();
    let n = steps.len();
    for (k, step) in steps.iter().enumerate() {
        let part = match *step {
            RouteStep::Lane(l) => {
                let line = &g.lanes()[l as usize].line;
                let a = if k == 0 { s0 } else { 0.0 };
                let b = if k + 1 == n { s1 } else { line.length() };
                line.slice(a, b.max(a))
            }
            RouteStep::Connector(c) => {
                connectors.push(c);
                g.connectors()[c as usize].line.points().to_vec()
            }
            RouteStep::Change(..) => return None,
        };
        for p in part {
            if points.last().is_none_or(|q| q.truncate().distance(p.truncate()) > 1e-6) {
                points.push(p);
            }
        }
    }
    (points.len() >= 2 && !connectors.is_empty()).then(|| (Polyline::new(points), connectors))
}

fn shuffle<T>(v: &mut [T], rng: &mut SimRng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i as u64 + 1) as usize);
    }
}
