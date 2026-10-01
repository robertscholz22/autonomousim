//! The pedestrian network of an urban map (M8c): walking lines along the sidewalks, corners
//! around the junctions, the crosswalks and mid-block crossings of the lane graph
//! ([`Crossing`]), and paths to building entrances and into parks. Places (entrances, bus
//! stops, parks) are the origins and destinations of pedestrians' trips.
//!
//! Like the lane graph, the network is a pure function of the map (roads, lots and buildings,
//! terrain and obstacles), built when first asked for, never stored in map files and no part
//! of map hashes.
//!
//! Walking lines keep to the back half of each sidewalk, clear of the street trees and lamp
//! posts along the kerb. On ring roads only the outer sidewalk is walked (the island has no
//! crossings).

use crate::lanes::{Area, CROSSWALK, LaneGraph};
use crate::roads::{Polyline, Projection, RoadNetwork, SegmentGrid, wrap_angle};
use crate::sites::Zone;
use crate::static_world::StaticWorld;
use autonomousim_core::terrain::Terrain;
use glam::{DVec2, DVec3};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::{PI, TAU};

/// Spacing (m) of the points of walking lines along roads and around corners.
const STEP: f64 = 2.0;

/// Least length (m) of a lane whose middle is a bus stop (as for the traffic driver's buses).
pub const BUS_STOP_LANE: f64 = 40.0;

/// Clearance (m) from solid obstacles that paths to entrances and into parks keep, from a
/// metre beyond their start.
const PATH_CLEARANCE: f64 = 0.3;

/// Shortest walking line (m) kept along a road's side.
const MIN_LINE: f64 = 2.0;

/// Width (m) of paths to entrances and into parks.
const PATH_WIDTH: f64 = 1.5;

/// What a walkway is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WalkKind {
    /// Along a road's sidewalk.
    Sidewalk,
    /// Around a junction's corner, from one road's sidewalk to the next.
    Corner,
    /// Over a road: a crosswalk or mid-block crossing ([`WalkEdge::crossing`]).
    Crossing,
    /// From a sidewalk to a building's entrance or into a park.
    Path,
}

/// A walkway between two nodes, its line running from `a` to `b`.
#[derive(Clone, Debug, PartialEq)]
pub struct WalkEdge {
    pub a: u32,
    pub b: u32,
    pub kind: WalkKind,
    pub line: Polyline,
    /// Usable width (m).
    pub width: f64,
    /// The road along or over which it runs (sidewalks and crossings).
    pub road: Option<u32>,
    /// The lane graph's crossing (crossings).
    pub crossing: Option<u32>,
}

/// A node: where walkways meet.
#[derive(Clone, Debug, PartialEq)]
pub struct WalkNode {
    pub position: DVec3,
    pub edges: Vec<u32>,
}

/// What a place is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaceKind {
    /// In front of a building's door.
    Entrance { building: u32 },
    /// On the sidewalk beside the middle of a lane (where a bus may stop).
    BusStop { lane: u32 },
    /// The middle of a park.
    Park { lot: u32 },
}

/// An origin or destination of pedestrians' trips, at a node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub node: u32,
    pub kind: PlaceKind,
}

/// A route over the network: its nodes, the edges between them (with whether each runs from
/// its `a` to its `b`) and its length (m).
#[derive(Clone, Debug, PartialEq)]
pub struct WalkRoute {
    pub nodes: Vec<u32>,
    pub edges: Vec<(u32, bool)>,
    pub length: f64,
}

/// The pedestrian network of a map.
#[derive(Clone, Debug, Default)]
pub struct Walkways {
    nodes: Vec<WalkNode>,
    edges: Vec<WalkEdge>,
    places: Vec<Place>,
    grid: SegmentGrid,
}

/// A node to make on a road's walking line, at a station.
#[derive(Clone, Copy, Debug)]
struct Stop {
    station: f64,
    what: Attach,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Attach {
    /// The line's end at the road's start (0) or end (1).
    End(usize),
    Crossing(u32),
    BusStop(u32),
    /// A path (index into the paths).
    Path(usize),
}

/// A path from a place to a road's walking line.
struct PathPlan {
    kind: PlaceKind,
    /// From the place to the walking line (its last point is replaced by the line's node).
    points: Vec<DVec2>,
}

impl Walkways {
    /// The pedestrian network of `world` (empty without sidewalks).
    pub fn build(world: &StaticWorld) -> Self {
        let net = world.roads();
        if net.is_empty() {
            return Self::default();
        }
        let g = net.lanes();
        let terrain = world.terrain();
        let sites = world.sites();
        let roads = net.roads();
        let n_roads = roads.len();
        let lift = |p: DVec2| p.extend(terrain.height(p.x, p.y));

        // Walking line offsets (left of start → end positive) per road and side; None where
        // the side is not walked.
        let lines: Vec<[Option<f64>; 2]> = (0..n_roads).map(|i| walking_offsets(net, g, i)).collect();
        // Extent of each walking line: from the crossing at each end, else the setback.
        let mut extent = vec![[0.0f64; 2]; n_roads];
        for (i, r) in roads.iter().enumerate() {
            let [sa, sb] = g.setbacks(i);
            extent[i] = [sa, r.line.length() - sb];
        }
        for c in g.crossings() {
            if let Some(node) = c.node {
                let r = &roads[c.road as usize];
                let end = usize::from(node == r.end && (node != r.start || c.station > 0.5 * r.line.length()));
                extent[c.road as usize][end] = c.station;
            }
        }

        // Per side, the ends drawn back (in steps of half a metre) out of other roads'
        // carriageways and junction areas where roads meet at a narrow angle or close by.
        let mut ends = vec![[[0.0f64; 2]; 2]; n_roads];
        let mut moved = vec![[[false; 2]; 2]; n_roads];
        let mut stops: Vec<[Vec<Stop>; 2]> = vec![[Vec::new(), Vec::new()]; n_roads];
        for i in 0..n_roads {
            for side in 0..2 {
                let Some(off) = lines[i][side] else { continue };
                let [a0, b0] = extent[i];
                let (mut a, mut b) = (a0, b0);
                let mid = 0.5 * (a + b);
                let n = ((b0 - a0) / 0.5).ceil().max(1.0) as usize;
                for k in 0..=n {
                    let st = a0 + (b0 - a0) * k as f64 / n as f64;
                    if blocked(net, offset_point(&roads[i].line, st, off)) {
                        if st < mid {
                            a = a.max(st + 0.5);
                        } else {
                            b = b.min(st - 0.5);
                        }
                    }
                }
                moved[i][side] = [a > a0, b < b0];
                ends[i][side] = [a, b];
                if b - a >= MIN_LINE {
                    stops[i][side].push(Stop { station: a, what: Attach::End(0) });
                    stops[i][side].push(Stop { station: b, what: Attach::End(1) });
                }
            }
        }
        let sides: Vec<[bool; 2]> = stops.iter().map(|s| [!s[0].is_empty(), !s[1].is_empty()]).collect();
        let walked = |i: usize, side: usize| sides[i][side];
        for (k, c) in g.crossings().iter().enumerate() {
            let i = c.road as usize;
            if c.node.is_none() && walked(i, 0) && walked(i, 1) {
                for side in 0..2 {
                    stops[i][side].push(Stop { station: c.station, what: Attach::Crossing(k as u32) });
                }
            }
        }
        // Bus stops: beside the middle of lanes long enough, on their right.
        for (l, lane) in g.lanes().iter().enumerate() {
            let i = lane.road as usize;
            let side = usize::from(lane.dir == 1);
            let len = lane.line.length();
            if len < BUS_STOP_LANE || !walked(i, side) {
                continue;
            }
            let s = roads[i].line.project(lane.line.point_at(0.5 * len).truncate()).station;
            let [a, b] = ends[i][side];
            if s > a + 1.0 && s < b - 1.0 {
                stops[i][side].push(Stop { station: s, what: Attach::BusStop(l as u32) });
            }
        }
        // Paths to entrances and into parks.
        let mut paths: Vec<PathPlan> = Vec::new();
        let mut plan_path = |kind: PlaceKind, from: DVec2, lot: usize, stops: &mut Vec<[Vec<Stop>; 2]>| {
            let l = &sites.lots[lot];
            let i = l.road as usize;
            let Some(r) = roads.get(i) else { return };
            let pr = r.line.project(l.front);
            let side = usize::from(pr.offset > 0.0);
            let Some(off) = lines[i][side] else { return };
            if !walked(i, side) {
                return;
            }
            let [a, b] = ends[i][side];
            let s = pr.station.clamp(a + 0.5, b - 0.5);
            let to = offset_point(&r.line, s, off);
            let mut points = vec![from];
            if l.front.distance(from) > 1.0 && l.front.distance(to) > 1.0 {
                points.push(l.front);
            }
            points.push(to);
            // Clear of solid obstacles from a metre beyond the place.
            let mut walked = 0.0;
            for w in points.windows(2) {
                let d = w[1] - w[0];
                let n = (d.length() / 0.5).ceil().max(1.0) as usize;
                for k in 0..=n {
                    let p = w[0] + d * (k as f64 / n as f64);
                    let at = walked + d.length() * k as f64 / n as f64;
                    if at > 1.0 && world.obstacle_clearance(lift(p) + DVec3::Z, 1.0) < PATH_CLEARANCE {
                        return;
                    }
                    if blocked(net, p) && p.distance(to) > 0.5 {
                        return;
                    }
                }
                walked += d.length();
            }
            stops[i][side].push(Stop { station: s, what: Attach::Path(paths.len()) });
            paths.push(PathPlan { kind, points });
        };
        for (b, building) in sites.buildings.iter().enumerate() {
            let Some(lot) = sites.lots.get(building.lot as usize) else { continue };
            let door = entrance(building.centre, building.yaw, building.size, lot.front);
            plan_path(PlaceKind::Entrance { building: b as u32 }, door, building.lot as usize, &mut stops);
        }
        for (k, lot) in sites.lots.iter().enumerate() {
            if lot.zone == Zone::Park {
                plan_path(PlaceKind::Park { lot: k as u32 }, lot.centre, k, &mut stops);
            }
        }

        // Nodes along the walking lines (stops within 0.25 m share one) and the sidewalks
        // between them.
        let mut out = Self::default();
        let mut end_node = vec![[[u32::MAX; 2]; 2]; n_roads];
        let mut crossing_node: Vec<[u32; 2]> = vec![[u32::MAX; 2]; g.crossings().len()];
        let mut bus_node: Vec<(u32, u32)> = Vec::new();
        let mut path_node = vec![u32::MAX; paths.len()];
        for i in 0..n_roads {
            let r = &roads[i];
            let w = net.section(i).sidewalk;
            for side in 0..2 {
                let list = &mut stops[i][side];
                if list.is_empty() {
                    continue;
                }
                let off = lines[i][side].expect("walked");
                list.sort_by(|p, q| p.station.total_cmp(&q.station));
                let mut last: Option<(u32, f64)> = None;
                for stop in list.iter() {
                    let node = match last {
                        Some((n, s)) if stop.station - s < 0.25 => n,
                        _ => {
                            let n = out.add_node(lift(offset_point(&r.line, stop.station, off)));
                            if let Some((m, s)) = last {
                                let line = walking_line(&r.line, s, stop.station, off, &lift);
                                out.add_edge(m, n, WalkKind::Sidewalk, line, w[side], Some(i as u32), None);
                            }
                            last = Some((n, stop.station));
                            n
                        }
                    };
                    match stop.what {
                        Attach::End(e) => end_node[i][side][e] = node,
                        Attach::Crossing(k) => crossing_node[k as usize][side] = node,
                        Attach::BusStop(l) => bus_node.push((node, l)),
                        Attach::Path(k) => path_node[k] = node,
                    }
                }
            }
        }

        // Crossings: between the walking lines' ends at junctions, between their middle
        // nodes elsewhere.
        for (k, c) in g.crossings().iter().enumerate() {
            let i = c.road as usize;
            let r = &roads[i];
            let ends = match c.node {
                Some(node) => {
                    let e = usize::from(node == r.end && (node != r.start || c.station > 0.5 * r.line.length()));
                    if moved[i][0][e] || moved[i][1][e] {
                        continue;
                    }
                    [end_node[i][0][e], end_node[i][1][e]]
                }
                None => crossing_node[k],
            };
            if ends.contains(&u32::MAX) {
                continue;
            }
            let (pa, pb) = (out.nodes[ends[0] as usize].position, out.nodes[ends[1] as usize].position);
            let line = straight(pa.truncate(), pb.truncate(), &lift);
            out.add_edge(ends[0], ends[1], WalkKind::Crossing, line, CROSSWALK, Some(c.road), Some(k as u32));
        }

        // Corners: around each node, from the left walking line of each road end (seen
        // leaving the node) to the right one of the next road end counter-clockwise.
        let mut at_node: Vec<Vec<(f64, u32, bool)>> = vec![Vec::new(); net.nodes().len()];
        for (i, r) in roads.iter().enumerate() {
            let len = r.line.length();
            at_node[r.start as usize].push((r.line.heading_at(0.0), i as u32, true));
            at_node[r.end as usize].push((wrap_angle(r.line.heading_at(len) + PI), i as u32, false));
        }
        for (n, list) in at_node.iter_mut().enumerate() {
            if list.len() < 2 {
                continue;
            }
            list.sort_by(|p, q| p.0.total_cmp(&q.0).then((p.1, p.2).cmp(&(q.1, q.2))));
            let centre = net.nodes()[n].position.truncate();
            for k in 0..list.len() {
                let (ha, ra, sa) = list[k];
                let (hb, rb, sb) = list[(k + 1) % list.len()];
                // Left of leaving along a road from its start is its side 1.
                let (side_a, end_a) = (usize::from(sa), usize::from(!sa));
                let (side_b, end_b) = (usize::from(!sb), usize::from(!sb));
                let (na, nb) = (end_node[ra as usize][side_a][end_a], end_node[rb as usize][side_b][end_b]);
                if na == u32::MAX || nb == u32::MAX || na == nb {
                    continue;
                }
                let (pa, pb) = (out.nodes[na as usize].position.truncate(), out.nodes[nb as usize].position.truncate());
                let turn = (hb - ha).rem_euclid(TAU);
                let mut points = corner(centre, pa, ha, pb, hb, turn);
                // Out of carriageways and the junction's area, away from the node.
                let last = points.len() - 1;
                for p in &mut points[1..last] {
                    let out = (*p - centre).try_normalize().unwrap_or(DVec2::X);
                    for _ in 0..60 {
                        if !blocked(net, *p) {
                            break;
                        }
                        *p += out * 0.25;
                    }
                }
                let points = dense(&points);
                if points[1..points.len() - 1].iter().any(|&p| blocked(net, p)) {
                    continue;
                }
                let line = Polyline::new(points.into_iter().map(lift).collect());
                let w = net.section(ra as usize).sidewalk[side_a].min(net.section(rb as usize).sidewalk[side_b]);
                out.add_edge(na, nb, WalkKind::Corner, line, w, None, None);
            }
        }

        // Places and their paths.
        for (k, p) in paths.iter().enumerate() {
            if path_node[k] == u32::MAX {
                continue;
            }
            let place = out.add_node(lift(p.points[0]));
            let mut pts: Vec<DVec3> = p.points[..p.points.len() - 1].iter().map(|&q| lift(q)).collect();
            pts.push(out.nodes[path_node[k] as usize].position);
            out.add_edge(place, path_node[k], WalkKind::Path, Polyline::new(pts), PATH_WIDTH, None, None);
            out.places.push(Place { node: place, kind: p.kind });
        }
        for (node, lane) in bus_node {
            out.places.push(Place { node, kind: PlaceKind::BusStop { lane } });
        }
        out.keep_largest();
        out.grid = SegmentGrid::of_lines(&out.edges.iter().map(|e| &e.line).collect::<Vec<_>>());
        out
    }

    /// Keep only the largest connected part (bits of sidewalk cut off where roads crowd
    /// each other go), renumbering nodes and edges in order.
    fn keep_largest(&mut self) {
        let comp = self.components();
        let mut size = vec![0usize; self.nodes.len()];
        for &c in &comp {
            size[c as usize] += 1;
        }
        let Some(best) = (0..size.len()).max_by_key(|&c| (size[c], std::cmp::Reverse(c))) else { return };
        let keep: Vec<bool> = comp.iter().map(|&c| c as usize == best).collect();
        let mut node_id = vec![u32::MAX; self.nodes.len()];
        let mut nodes = Vec::new();
        for (n, node) in self.nodes.iter().enumerate() {
            if keep[n] {
                node_id[n] = nodes.len() as u32;
                nodes.push(WalkNode { position: node.position, edges: Vec::new() });
            }
        }
        let mut edges = Vec::new();
        for e in self.edges.drain(..) {
            if keep[e.a as usize] {
                let (a, b) = (node_id[e.a as usize], node_id[e.b as usize]);
                nodes[a as usize].edges.push(edges.len() as u32);
                nodes[b as usize].edges.push(edges.len() as u32);
                edges.push(WalkEdge { a, b, ..e });
            }
        }
        self.places.retain(|p| keep[p.node as usize]);
        for p in &mut self.places {
            p.node = node_id[p.node as usize];
        }
        self.nodes = nodes;
        self.edges = edges;
    }

    fn add_node(&mut self, position: DVec3) -> u32 {
        self.nodes.push(WalkNode { position, edges: Vec::new() });
        self.nodes.len() as u32 - 1
    }

    #[allow(clippy::too_many_arguments)]
    fn add_edge(
        &mut self,
        a: u32,
        b: u32,
        kind: WalkKind,
        line: Polyline,
        width: f64,
        road: Option<u32>,
        crossing: Option<u32>,
    ) {
        let e = self.edges.len() as u32;
        self.nodes[a as usize].edges.push(e);
        self.nodes[b as usize].edges.push(e);
        self.edges.push(WalkEdge { a, b, kind, line, width, road, crossing });
    }

    pub fn nodes(&self) -> &[WalkNode] {
        &self.nodes
    }

    pub fn edges(&self) -> &[WalkEdge] {
        &self.edges
    }

    pub fn places(&self) -> &[Place] {
        &self.places
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// The node at the other end of edge `e` from node `n`.
    pub fn other(&self, e: u32, n: u32) -> u32 {
        let edge = &self.edges[e as usize];
        if edge.a == n { edge.b } else { edge.a }
    }

    /// The nearest walkway within `max_dist` of `p` and the projection onto its line.
    pub fn nearest(&self, p: DVec2, max_dist: f64) -> Option<(u32, Projection)> {
        let mut best: Option<(u32, Projection)> = None;
        self.grid.visit(p, max_dist, &mut |e, _| {
            let pr = self.edges[e as usize].line.project(p);
            if pr.distance <= max_dist && best.is_none_or(|(b, q)| (pr.distance, e) < (q.distance, b)) {
                best = Some((e, pr));
            }
            best.map_or(max_dist, |b| b.1.distance)
        });
        best
    }

    /// The shortest route from node `from` to node `to` (A*), None if they are not connected.
    pub fn route(&self, from: u32, to: u32) -> Option<WalkRoute> {
        let goal = self.nodes[to as usize].position.truncate();
        let h = |n: u32| self.nodes[n as usize].position.truncate().distance(goal);
        let mut cost = vec![f64::INFINITY; self.nodes.len()];
        let mut came: Vec<Option<u32>> = vec![None; self.nodes.len()];
        let mut open = BinaryHeap::new();
        cost[from as usize] = 0.0;
        open.push(Entry { f: h(from), node: from });
        while let Some(Entry { f, node }) = open.pop() {
            if node == to {
                break;
            }
            if f > cost[node as usize] + h(node) + 1e-9 {
                continue;
            }
            for &e in &self.nodes[node as usize].edges {
                let next = self.other(e, node);
                let c = cost[node as usize] + self.edges[e as usize].line.length();
                if c < cost[next as usize] {
                    cost[next as usize] = c;
                    came[next as usize] = Some(e);
                    open.push(Entry { f: c + h(next), node: next });
                }
            }
        }
        if !cost[to as usize].is_finite() {
            return None;
        }
        let (mut nodes, mut edges) = (vec![to], Vec::new());
        let mut n = to;
        while let Some(e) = came[n as usize] {
            let prev = self.other(e, n);
            edges.push((e, self.edges[e as usize].a == prev));
            nodes.push(prev);
            n = prev;
        }
        nodes.reverse();
        edges.reverse();
        Some(WalkRoute { nodes, edges, length: cost[to as usize] })
    }

    /// The connected component of each node (numbered by their lowest node).
    pub fn components(&self) -> Vec<u32> {
        let mut comp = vec![u32::MAX; self.nodes.len()];
        for start in 0..self.nodes.len() {
            if comp[start] != u32::MAX {
                continue;
            }
            let mut stack = vec![start as u32];
            comp[start] = start as u32;
            while let Some(n) = stack.pop() {
                for &e in &self.nodes[n as usize].edges {
                    let m = self.other(e, n);
                    if comp[m as usize] == u32::MAX {
                        comp[m as usize] = start as u32;
                        stack.push(m);
                    }
                }
            }
        }
        comp
    }
}

/// Offsets (left of start → end positive) of road `i`'s walking lines on its right (0) and
/// left (1) side: in the back half of the sidewalk, clear of the trees (1.2 m) and lamp posts
/// (0.4 m) along the kerb. None where there is no sidewalk, and on a ring's island side.
fn walking_offsets(net: &RoadNetwork, g: &LaneGraph, i: usize) -> [Option<f64>; 2] {
    let r = &net.roads()[i];
    let s = net.section(i);
    if !r.class.is_urban() {
        return [None, None];
    }
    let half = 0.5 * r.width;
    let ring = g.is_ring(i);
    let at = |side: usize| {
        let w = s.sidewalk[side];
        (w > 0.0 && !(ring && side == 1)).then(|| {
            let d = half + (0.6 * w).max(w - 0.6).min(w - 0.3).max(0.5 * w);
            if side == 1 { d } else { -d }
        })
    };
    [at(0), at(1)]
}

/// Whether `p` lies on a carriageway or in a junction's area.
fn blocked(net: &RoadNetwork, p: DVec2) -> bool {
    !matches!(net.area(p), Area::Sidewalk | Area::Off)
}

/// The point `off` m left of `line` at station `s`.
fn offset_point(line: &Polyline, s: f64, off: f64) -> DVec2 {
    let h = line.heading_at(s);
    line.point_at(s).truncate() + DVec2::new(-h.sin(), h.cos()) * off
}

/// The walking line `off` m left of `line` from station `a` to `b`, on the terrain.
fn walking_line(line: &Polyline, a: f64, b: f64, off: f64, lift: &dyn Fn(DVec2) -> DVec3) -> Polyline {
    let n = ((b - a) / STEP).ceil().max(1.0) as usize;
    Polyline::new((0..=n).map(|k| lift(offset_point(line, a + (b - a) * k as f64 / n as f64, off))).collect())
}

/// A straight line from `a` to `b` on the terrain, with points every [`STEP`].
fn straight(a: DVec2, b: DVec2, lift: &dyn Fn(DVec2) -> DVec3) -> Polyline {
    let n = (a.distance(b) / STEP).ceil().max(1.0) as usize;
    Polyline::new((0..=n).map(|k| lift(a.lerp(b, k as f64 / n as f64))).collect())
}

/// The corner from `pa` (on the walking line left of a road leaving the node at `centre` with
/// heading `ha`) to `pb` (right of the next road counter-clockwise, heading `hb`), `turn` rad
/// counter-clockwise from the first. A concave corner runs along both walking lines to where
/// they meet (straight across when they meet beyond the ends); a straight one runs straight;
/// a convex one runs along the first walking line to abreast of the node, around it at that
/// distance and along the second.
fn corner(centre: DVec2, pa: DVec2, ha: f64, pb: DVec2, hb: f64, turn: f64) -> Vec<DVec2> {
    let (ua, ub) = (DVec2::from_angle(ha), DVec2::from_angle(hb));
    if turn < PI - 0.1 {
        // Lines pa − ta·ua and pb − tb·ub (towards the node) meet where
        // ta·ua − tb·ub = pa − pb.
        let den = ua.perp_dot(ub);
        let d = pa - pb;
        let (ta, tb) = (d.perp_dot(ub) / den, d.perp_dot(ua) / den);
        if den.abs() > 1e-9 && ta > 0.0 && tb > 0.0 {
            return dense(&[pa, pa - ua * ta, pb]);
        }
        return dense(&[pa, pb]);
    }
    if turn <= PI + 0.1 {
        return dense(&[pa, pb]);
    }
    // Abreast of the node on each walking line, then around the node.
    let (oa, ob) = ((pa - centre).dot(ua.perp()), (pb - centre).dot(-ub.perp()));
    let (qa, qb) = (centre + ua.perp() * oa, centre - ub.perp() * ob);
    let a0 = ha + 0.5 * PI;
    let sweep = (hb - 0.5 * PI - a0).rem_euclid(TAU);
    let n = ((0.5 * (oa + ob) * sweep) / STEP).ceil().max(1.0) as usize;
    let mut out = dense(&[pa, qa]);
    out.extend((1..n).map(|k| {
        let u = k as f64 / n as f64;
        centre + DVec2::from_angle(a0 + sweep * u) * (oa + (ob - oa) * u)
    }));
    out.extend(dense(&[qb, pb]));
    out
}

/// The polyline through `points` with points every [`STEP`] at most.
fn dense(points: &[DVec2]) -> Vec<DVec2> {
    let mut out = vec![points[0]];
    for w in points.windows(2) {
        let n = (w[0].distance(w[1]) / STEP).ceil().max(1.0) as usize;
        out.extend((1..=n).map(|k| w[0].lerp(w[1], k as f64 / n as f64)));
    }
    out
}

/// A building's entrance: the point of its footprint (`centre`, `yaw`, `size`) nearest to
/// `front`, half a metre out from it.
fn entrance(centre: DVec2, yaw: f64, size: DVec2, front: DVec2) -> DVec2 {
    let rot = DVec2::from_angle(yaw);
    let local = DVec2::from_angle(-yaw).rotate(front - centre);
    let half = 0.5 * size;
    let inside = local.clamp(-half, half);
    // Onto the nearest side when the front lies within the footprint's extent.
    let edge = if inside == local {
        let gaps = [half.x - local.x, local.x + half.x, half.y - local.y, local.y + half.y];
        let k = (0..4).min_by(|&a, &b| gaps[a].total_cmp(&gaps[b])).expect("four");
        match k {
            0 => DVec2::new(half.x, local.y),
            1 => DVec2::new(-half.x, local.y),
            2 => DVec2::new(local.x, half.y),
            _ => DVec2::new(local.x, -half.y),
        }
    } else {
        inside
    };
    let out = (local - edge).try_normalize().unwrap_or(DVec2::Y);
    centre + rot.rotate(edge + out * 0.5)
}

#[derive(PartialEq)]
struct Entry {
    f: f64,
    node: u32,
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f.total_cmp(&self.f).then(other.node.cmp(&self.node))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
