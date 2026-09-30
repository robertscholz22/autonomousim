//! Road networks: roads as polylines between nodes, with nearest-point queries and routes.
//!
//! A road runs from one node to another along a polyline resampled at about 1 m, with `z` on
//! the (blended) terrain surface. Stations `s` are arc lengths along a polyline; lateral offsets
//! are positive to the left of the direction of increasing `s`; headings are yaw angles in the
//! ENU frame (0 along +x, counter-clockwise).

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::OnceLock;

use crate::lanes::{Area, LaneGraph};

/// Surface and size class of a road.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoadClass {
    /// Asphalt, two lanes.
    Paved,
    Gravel,
    /// Dirt track to fields.
    Track,
    /// Urban main road: several lanes per direction, sometimes a median.
    Arterial,
    /// Urban street linking neighbourhoods to the arterials.
    Collector,
    /// Urban residential or access street.
    Local,
}

impl RoadClass {
    /// Urban classes (with a [`Section`] of their own).
    pub fn is_urban(self) -> bool {
        matches!(self, Self::Arterial | Self::Collector | Self::Local)
    }
}

/// What lies at a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Junction,
    /// A road leaving the map or ending without a destination.
    End,
    /// The yard of a farm.
    Yard,
    /// A gate into a field.
    Gate,
    /// Where a road meets the ring of a roundabout (the ring's roads are one-way).
    Roundabout,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoadNode {
    pub position: DVec3,
    pub kind: NodeKind,
}

/// A polyline with cumulative stations.
#[derive(Clone, Debug, PartialEq)]
pub struct Polyline {
    points: Vec<DVec3>,
    /// `stations[i]`: horizontal arc length from the first point to point `i`.
    stations: Vec<f64>,
}

impl Polyline {
    /// At least two points; consecutive duplicates (horizontally) are dropped.
    pub fn new(points: Vec<DVec3>) -> Self {
        let mut pts: Vec<DVec3> = Vec::with_capacity(points.len());
        for p in points {
            if pts.last().is_none_or(|q| q.truncate().distance_squared(p.truncate()) > 1e-18) {
                pts.push(p);
            }
        }
        assert!(pts.len() >= 2, "a polyline needs two distinct points");
        let mut stations = Vec::with_capacity(pts.len());
        let mut s = 0.0;
        stations.push(0.0);
        for w in pts.windows(2) {
            s += w[0].truncate().distance(w[1].truncate());
            stations.push(s);
        }
        Self { points: pts, stations }
    }

    pub fn points(&self) -> &[DVec3] {
        &self.points
    }

    pub fn length(&self) -> f64 {
        *self.stations.last().expect("two points")
    }

    /// Segment containing station `s` (clamped).
    fn segment_at(&self, s: f64) -> usize {
        let i = self.stations.partition_point(|&t| t <= s);
        i.clamp(1, self.points.len() - 1) - 1
    }

    /// Point at station `s` (clamped to the ends).
    pub fn point_at(&self, s: f64) -> DVec3 {
        let s = s.clamp(0.0, self.length());
        let i = self.segment_at(s);
        let (s0, s1) = (self.stations[i], self.stations[i + 1]);
        self.points[i].lerp(self.points[i + 1], (s - s0) / (s1 - s0))
    }

    /// Direction of travel at station `s` (heading of its segment).
    pub fn heading_at(&self, s: f64) -> f64 {
        let i = self.segment_at(s.clamp(0.0, self.length()));
        let d = self.points[i + 1].truncate() - self.points[i].truncate();
        d.y.atan2(d.x)
    }

    /// Signed curvature at station `s` (1/m, positive turning left): that of the circle through
    /// the points at `s − h`, `s` and `s + h`, with `h` = 2 m kept within the polyline.
    pub fn curvature_at(&self, s: f64) -> f64 {
        const H: f64 = 2.0;
        let len = self.length();
        let h = H.min(0.5 * len);
        let s = s.clamp(h, len - h);
        let (a, b, c) = (self.point_at(s - h).truncate(), self.point_at(s).truncate(), self.point_at(s + h).truncate());
        let denom = a.distance(b) * b.distance(c) * a.distance(c);
        if denom < 1e-12 { 0.0 } else { 2.0 * (b - a).perp_dot(c - b) / denom }
    }

    /// Closest point to `p` (horizontally) on segment `i`: (station, squared distance).
    pub(crate) fn project_segment(&self, i: usize, p: DVec2) -> (f64, f64) {
        let (a, b) = (self.points[i].truncate(), self.points[i + 1].truncate());
        let d = b - a;
        let t = ((p - a).dot(d) / d.length_squared()).clamp(0.0, 1.0);
        let q = a + d * t;
        (self.stations[i] + t * (self.stations[i + 1] - self.stations[i]), q.distance_squared(p))
    }

    /// Projection of `p` onto the whole polyline (first segment wins ties).
    pub fn project(&self, p: DVec2) -> Projection {
        let mut best = (0usize, 0.0, f64::INFINITY);
        for i in 0..self.points.len() - 1 {
            let (s, d2) = self.project_segment(i, p);
            if d2 < best.2 {
                best = (i, s, d2);
            }
        }
        self.projection(best.0, best.1, p)
    }

    pub(crate) fn projection(&self, segment: usize, station: f64, p: DVec2) -> Projection {
        let point = self.point_at(station);
        let d = self.points[segment + 1].truncate() - self.points[segment].truncate();
        let heading = d.y.atan2(d.x);
        let rel = p - point.truncate();
        let offset = d.perp_dot(rel) / d.length();
        Projection { station, point, heading, offset, distance: rel.length() }
    }

    /// The part between stations `a` and `b`, reversed when `b < a`.
    pub fn slice(&self, a: f64, b: f64) -> Vec<DVec3> {
        let (lo, hi) = (a.min(b).clamp(0.0, self.length()), a.max(b).clamp(0.0, self.length()));
        let mut out = vec![self.point_at(lo)];
        out.extend(self.points.iter().zip(&self.stations).filter(|&(_, &s)| s > lo && s < hi).map(|(p, _)| *p));
        out.push(self.point_at(hi));
        if b < a {
            out.reverse();
        }
        out
    }
}

/// Where a point lies relative to a polyline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    pub station: f64,
    /// The closest point on the polyline.
    pub point: DVec3,
    /// Direction of the polyline there.
    pub heading: f64,
    /// Signed horizontal distance, positive to the left.
    pub offset: f64,
    /// Unsigned horizontal distance (equals `|offset|` except beyond the ends).
    pub distance: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Road {
    pub class: RoadClass,
    /// Full width of the road surface (m).
    pub width: f64,
    pub start: u32,
    pub end: u32,
    pub line: Polyline,
}

/// Cross-section of a road, from its centre line outwards: the lanes of each direction (the
/// first entry for travel from `start` to `end`, the second against it), then per side of the
/// road (the right of each direction) a bike lane, a parking lane and a sidewalk. The median,
/// if any, is centred on the centre line. The carriageway ([`Road::width`]) spans the median,
/// the lanes, the bike lanes and the parking lanes; sidewalks lie beyond it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Section {
    pub lanes: [u8; 2],
    pub lane_width: f64,
    pub median: f64,
    pub bike: [f64; 2],
    pub parking: [f64; 2],
    pub sidewalk: [f64; 2],
}

impl Section {
    /// Two-way road of `width` with one lane per direction and nothing else (the rural roads).
    pub fn plain(width: f64) -> Self {
        Self {
            lanes: [1, 1],
            lane_width: 0.5 * width,
            median: 0.0,
            bike: [0.0; 2],
            parking: [0.0; 2],
            sidewalk: [0.0; 2],
        }
    }

    /// Width of the carriageway (m).
    pub fn width(&self) -> f64 {
        let lanes = f64::from(self.lanes[0]) + f64::from(self.lanes[1]);
        lanes * self.lane_width + self.median + self.bike[0] + self.bike[1] + self.parking[0] + self.parking[1]
    }

    /// Whether traffic runs only from `start` to `end`.
    pub fn one_way(&self) -> bool {
        self.lanes[1] == 0
    }

    /// Carriageway offsets (positive to the left of `start → end`) of the two edges of the
    /// part of direction `dir` (0 along, 1 against): `(inner, outer)` distances from the
    /// centre line, the outer edge including the bike and parking lanes. One-way roads have
    /// their lanes centred.
    pub fn half_widths(&self, dir: usize) -> (f64, f64) {
        if self.one_way() {
            let w = 0.5 * self.width();
            return if dir == 0 { (-w, w) } else { (0.0, 0.0) };
        }
        let inner = 0.5 * self.median;
        (inner, inner + f64::from(self.lanes[dir]) * self.lane_width + self.bike[dir] + self.parking[dir])
    }
}

/// The nearest road to a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoadPoint {
    pub road: u32,
    pub projection: Projection,
}

/// A path over the network: a polyline plus the roads it uses.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub line: Polyline,
    /// Roads in order of travel, with `true` when driven from `start` to `end`.
    pub roads: Vec<(u32, bool)>,
}

/// Roads and nodes of a map, with a uniform grid over the segments for queries.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoadNetwork {
    nodes: Vec<RoadNode>,
    roads: Vec<Road>,
    /// One per road, or none (then every road is [`Section::plain`]).
    sections: Vec<Section>,
    grid: SegmentGrid,
    /// Built on first use; a function of the rest.
    lanes: LazyLanes,
}

/// The lane graph of a network, built when first asked for (it never takes part in equality).
#[derive(Clone, Debug, Default)]
struct LazyLanes(OnceLock<LaneGraph>);

impl PartialEq for LazyLanes {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RoadError {
    #[error("road {0} refers to node {1}, which does not exist")]
    MissingNode(usize, u32),
    #[error("road {0} has width {1}")]
    Width(usize, f64),
    #[error("{0} sections for {1} roads")]
    Sections(usize, usize),
    #[error("road {0}: its section is {1} m wide, the road {2} m")]
    SectionWidth(usize, f64, f64),
}

impl RoadNetwork {
    pub fn new(nodes: Vec<RoadNode>, roads: Vec<Road>) -> Result<Self, RoadError> {
        for (i, r) in roads.iter().enumerate() {
            for n in [r.start, r.end] {
                if n as usize >= nodes.len() {
                    return Err(RoadError::MissingNode(i, n));
                }
            }
            if !(r.width > 0.0 && r.width.is_finite()) {
                return Err(RoadError::Width(i, r.width));
            }
        }
        let grid = SegmentGrid::build(&roads);
        Ok(Self { nodes, roads, sections: Vec::new(), grid, lanes: LazyLanes::default() })
    }

    /// The same network with a cross-section per road, each as wide as its road.
    pub fn with_sections(mut self, sections: Vec<Section>) -> Result<Self, RoadError> {
        if !sections.is_empty() && sections.len() != self.roads.len() {
            return Err(RoadError::Sections(sections.len(), self.roads.len()));
        }
        for (i, (s, r)) in sections.iter().zip(&self.roads).enumerate() {
            if (s.width() - r.width).abs() > 1e-9 || s.lanes[0] == 0 {
                return Err(RoadError::SectionWidth(i, s.width(), r.width));
            }
        }
        self.sections = sections;
        self.lanes = LazyLanes::default();
        Ok(self)
    }

    /// The lane graph (built on the first call).
    pub fn lanes(&self) -> &LaneGraph {
        self.lanes.0.get_or_init(|| LaneGraph::build(&self.nodes, &self.roads, &|i| self.section(i)))
    }

    /// What lies at `p`: the area class on the nearest road within 30 m, else [`Area::Off`].
    pub fn area(&self, p: DVec2) -> Area {
        let Some(rp) = self.nearest(p, 30.0) else { return Area::Off };
        let section = self.section(rp.road as usize);
        self.lanes().area(&self.roads, &section, rp.road, rp.projection.station, rp.projection.offset)
    }

    /// Cross-section of road `i`.
    pub fn section(&self, i: usize) -> Section {
        self.sections.get(i).copied().unwrap_or_else(|| Section::plain(self.roads[i].width))
    }

    /// Whether the roads have sections of their own (urban maps).
    pub fn has_sections(&self) -> bool {
        !self.sections.is_empty()
    }

    pub fn is_empty(&self) -> bool {
        self.roads.is_empty()
    }

    pub fn nodes(&self) -> &[RoadNode] {
        &self.nodes
    }

    pub fn roads(&self) -> &[Road] {
        &self.roads
    }

    /// Distance from `p` to the nearest road edge among the roads whose centre line lies
    /// within `reach` (negative on a road), with that road; `None` when no road is that close.
    /// The edge is the carriageway's, plus the sidewalk on `p`'s side when `sidewalks`;
    /// road `except` does not count. Ties go to the lower road.
    pub fn edge_distance(&self, p: DVec2, reach: f64, sidewalks: bool, except: Option<u32>) -> Option<(f64, u32)> {
        let mut best: Option<(f64, u32)> = None;
        self.grid.visit(p, reach, &mut |road, seg| {
            if except == Some(road) {
                return reach;
            }
            let r = &self.roads[road as usize];
            let (s, d2) = r.line.project_segment(seg as usize, p);
            if d2 <= reach * reach {
                let pr = r.line.projection(seg as usize, s, p);
                let walk =
                    if sidewalks { self.section(road as usize).sidewalk[usize::from(pr.offset > 0.0)] } else { 0.0 };
                let d = pr.distance - 0.5 * r.width - walk;
                if best.is_none_or(|(b, k)| d < b || (d == b && road < k)) {
                    best = Some((d, road));
                }
            }
            reach
        });
        best
    }

    /// Nearest road point to `p` within `max_dist` (horizontal); ties go to the lower road
    /// and segment index.
    pub fn nearest(&self, p: DVec2, max_dist: f64) -> Option<RoadPoint> {
        let mut best: Option<(f64, u32, u32, f64)> = None;
        self.grid.visit(p, max_dist, &mut |road, seg| {
            let (s, d2) = self.roads[road as usize].line.project_segment(seg as usize, p);
            let better = match best {
                None => d2 <= max_dist * max_dist,
                Some((b, r, g, _)) => d2 < b || (d2 == b && (road, seg) < (r, g)),
            };
            if better {
                best = Some((d2, road, seg, s));
            }
            best.map_or(max_dist, |b| b.0.sqrt())
        });
        best.map(|(_, road, seg, s)| RoadPoint {
            road,
            projection: self.roads[road as usize].line.projection(seg as usize, s, p),
        })
    }

    /// The road whose surface contains `p`, if any (the nearest one when several do).
    pub fn on_road(&self, p: DVec2) -> Option<RoadPoint> {
        let max_half = self.roads.iter().map(|r| 0.5 * r.width).fold(0.0, f64::max);
        let rp = self.nearest(p, max_half)?;
        (rp.projection.distance <= 0.5 * self.roads[rp.road as usize].width).then_some(rp)
    }

    /// Shortest route (by length) from the road point nearest `from` to the one nearest `to`,
    /// both within `max_dist` of the network; `None` when they are not connected.
    pub fn route(&self, from: DVec2, to: DVec2, max_dist: f64) -> Option<Route> {
        let a = self.nearest(from, max_dist)?;
        let b = self.nearest(to, max_dist)?;
        let (ra, rb) = (&self.roads[a.road as usize], &self.roads[b.road as usize]);
        let (sa, sb) = (a.projection.station, b.projection.station);
        // Driving along a single road.
        let mut best: Option<(f64, Route)> = None;
        if a.road == b.road {
            let line = Polyline::new(ra.line.slice(sa, sb));
            best = Some(((sb - sa).abs(), Route { line, roads: vec![(a.road, sb >= sa)] }));
        }
        // Leaving a's road by either end and joining b's road by either end.
        let sources = [(ra.start, sa), (ra.end, ra.line.length() - sa)];
        let tree = self.dijkstra(&sources);
        for (entry, tail) in [(rb.start, sb), (rb.end, rb.line.length() - sb)] {
            let d = tree.dist[entry as usize] + tail;
            if !d.is_finite() || best.as_ref().is_some_and(|b| b.0 <= d) {
                continue;
            }
            // Walk back from `entry` to the source node.
            let mut hops = Vec::new();
            let mut n = entry;
            while let Some((road, from_node)) = tree.prev[n as usize] {
                hops.push((road, from_node));
                n = from_node;
            }
            hops.reverse();
            let exit_forward = tree.origin[entry as usize] == 1;
            let mut pts = ra.line.slice(sa, if exit_forward { ra.line.length() } else { 0.0 });
            let mut roads = vec![(a.road, exit_forward)];
            for &(road, from_node) in &hops {
                let r = &self.roads[road as usize];
                let forward = r.start == from_node;
                let len = r.line.length();
                pts.extend(r.line.slice(if forward { 0.0 } else { len }, if forward { len } else { 0.0 }));
                roads.push((road, forward));
            }
            let enter_forward = entry == rb.start;
            pts.extend(rb.line.slice(if enter_forward { 0.0 } else { rb.line.length() }, sb));
            roads.push((b.road, enter_forward));
            best = Some((d, Route { line: Polyline::new(pts), roads }));
        }
        best.map(|b| b.1)
    }

    /// Shortest distances from several source nodes (with start costs) over the roads. Ties
    /// go to the lower node index.
    fn dijkstra(&self, sources: &[(u32, f64)]) -> ShortestPaths {
        let n = self.nodes.len();
        let mut adj: Vec<Vec<(u32, u32, f64)>> = vec![Vec::new(); n];
        for (i, r) in self.roads.iter().enumerate() {
            let len = r.line.length();
            adj[r.start as usize].push((r.end, i as u32, len));
            adj[r.end as usize].push((r.start, i as u32, len));
        }
        let mut t = ShortestPaths { dist: vec![f64::INFINITY; n], prev: vec![None; n], origin: vec![0; n] };
        let mut heap = BinaryHeap::new();
        for (k, &(node, cost)) in sources.iter().enumerate() {
            if cost < t.dist[node as usize] {
                t.dist[node as usize] = cost;
                t.origin[node as usize] = k;
                heap.push(Entry(cost, node));
            }
        }
        while let Some(Entry(d, u)) = heap.pop() {
            if d > t.dist[u as usize] {
                continue;
            }
            for &(v, road, len) in &adj[u as usize] {
                let nd = d + len;
                if nd < t.dist[v as usize] {
                    t.dist[v as usize] = nd;
                    t.prev[v as usize] = Some((road, u));
                    t.origin[v as usize] = t.origin[u as usize];
                    heap.push(Entry(nd, v));
                }
            }
        }
        t
    }
}

/// Result of [`RoadNetwork::dijkstra`], per node.
struct ShortestPaths {
    dist: Vec<f64>,
    /// The road and node it was reached from (`None` for sources and unreached nodes).
    prev: Vec<Option<(u32, u32)>>,
    /// Index of the source it was reached from.
    origin: Vec<usize>,
}

/// Min-heap entry: smaller distance first, then smaller node index.
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

/// Wrap an angle to (−π, π].
pub fn wrap_angle(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let w = a.rem_euclid(t);
    if w > std::f64::consts::PI { w - t } else { w }
}

/// Uniform xy grid listing, per cell, the road segments whose bounding box meets it.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SegmentGrid {
    cell: f64,
    lo: DVec2,
    nx: usize,
    ny: usize,
    /// `start[c]..start[c + 1]` indexes `items` for cell `c = y·nx + x`.
    start: Vec<u32>,
    /// (road, segment).
    items: Vec<(u32, u32)>,
}

impl SegmentGrid {
    const CELL: f64 = 16.0;

    fn build(roads: &[Road]) -> Self {
        Self::of_lines(&roads.iter().map(|r| &r.line).collect::<Vec<_>>())
    }

    /// A grid over the segments of `lines` (ids are indices into `lines`).
    pub(crate) fn of_lines(lines: &[&Polyline]) -> Self {
        let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
        for p in lines.iter().flat_map(|l| l.points()) {
            lo = lo.min(p.truncate());
            hi = hi.max(p.truncate());
        }
        if lines.is_empty() {
            return Self::default();
        }
        let cell = Self::CELL;
        let nx = ((hi.x - lo.x) / cell).floor() as usize + 1;
        let ny = ((hi.y - lo.y) / cell).floor() as usize + 1;
        let mut per_cell: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nx * ny];
        let index = |v: f64, lo: f64, n: usize| (((v - lo) / cell).floor().max(0.0) as usize).min(n - 1);
        for (r, line) in lines.iter().enumerate() {
            for (i, w) in line.points().windows(2).enumerate() {
                let (a, b) = (w[0].truncate(), w[1].truncate());
                let (x0, x1) = (index(a.x.min(b.x), lo.x, nx), index(a.x.max(b.x), lo.x, nx));
                let (y0, y1) = (index(a.y.min(b.y), lo.y, ny), index(a.y.max(b.y), lo.y, ny));
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        per_cell[y * nx + x].push((r as u32, i as u32));
                    }
                }
            }
        }
        let mut start = Vec::with_capacity(nx * ny + 1);
        let mut items = Vec::new();
        for c in per_cell {
            start.push(items.len() as u32);
            items.extend(c);
        }
        start.push(items.len() as u32);
        Self { cell, lo, nx, ny, start, items }
    }

    /// Call `f(road, segment)` for segments in cells ring by ring around `p`, while a ring can
    /// still hold a segment within the radius `f` returns (the best distance so far, at most
    /// `max_dist`). A segment may be visited more than once.
    pub(crate) fn visit(&self, p: DVec2, max_dist: f64, f: &mut impl FnMut(u32, u32) -> f64) {
        if self.items.is_empty() {
            return;
        }
        let c = ((p - self.lo) / self.cell).floor();
        let (cx, cy) = (c.x as i64, c.y as i64);
        let (nx, ny) = (self.nx as i64, self.ny as i64);
        // Distance from p to the nearest cell, so rings far outside the grid are skipped.
        let outside = DVec2::new(
            (self.lo.x - p.x).max(p.x - (self.lo.x + self.cell * self.nx as f64)).max(0.0),
            (self.lo.y - p.y).max(p.y - (self.lo.y + self.cell * self.ny as f64)).max(0.0),
        )
        .length();
        let mut radius = max_dist;
        if outside > radius {
            return;
        }
        let max_ring = nx.max(ny) + cx.abs().max(cy.abs());
        for ring in 0..=max_ring {
            // Every cell in this ring is at least (ring − 1)·cell away.
            if (ring - 1).max(0) as f64 * self.cell > radius {
                break;
            }
            for y in cy - ring..=cy + ring {
                if y < 0 || y >= ny {
                    continue;
                }
                let on_edge_row = y == cy - ring || y == cy + ring;
                let step = if on_edge_row { 1 } else { (2 * ring).max(1) };
                let mut x = cx - ring;
                while x <= cx + ring {
                    if x >= 0 && x < nx {
                        let k = (y * nx + x) as usize;
                        for &(road, seg) in &self.items[self.start[k] as usize..self.start[k + 1] as usize] {
                            radius = radius.min(f(road, seg));
                        }
                    }
                    x += step;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------ serialisation

/// Stored form of a network (the grid is rebuilt on load).
#[derive(Serialize)]
pub(crate) struct RoadsRef<'a> {
    nodes: &'a [RoadNode],
    roads: Vec<RoadRef<'a>>,
}

#[derive(Serialize)]
struct RoadRef<'a> {
    class: RoadClass,
    width: f64,
    start: u32,
    end: u32,
    points: &'a [DVec3],
}

#[derive(Deserialize)]
pub(crate) struct RoadsData {
    nodes: Vec<RoadNode>,
    roads: Vec<RoadData>,
}

#[derive(Deserialize)]
struct RoadData {
    class: RoadClass,
    width: f64,
    start: u32,
    end: u32,
    points: Vec<DVec3>,
}

impl RoadNetwork {
    /// Stored form of the sections (empty for maps without).
    pub(crate) fn stored_sections(&self) -> &[Section] {
        &self.sections
    }

    pub(crate) fn stored(&self) -> RoadsRef<'_> {
        RoadsRef {
            nodes: &self.nodes,
            roads: self
                .roads
                .iter()
                .map(|r| RoadRef {
                    class: r.class,
                    width: r.width,
                    start: r.start,
                    end: r.end,
                    points: r.line.points(),
                })
                .collect(),
        }
    }
}

impl TryFrom<RoadsData> for RoadNetwork {
    type Error = String;
    fn try_from(d: RoadsData) -> Result<Self, String> {
        let roads = d
            .roads
            .into_iter()
            .map(|r| {
                if r.points.len() < 2 {
                    return Err("a road with fewer than two points".to_owned());
                }
                Ok(Road { class: r.class, width: r.width, start: r.start, end: r.end, line: Polyline::new(r.points) })
            })
            .collect::<Result<Vec<_>, _>>()?;
        RoadNetwork::new(d.nodes, roads).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::rng::Seed;

    fn node(x: f64, y: f64) -> RoadNode {
        RoadNode { position: DVec3::new(x, y, 0.0), kind: NodeKind::Junction }
    }

    /// A straight road from node `a` to node `b`, resampled every metre.
    fn straight(nodes: &[RoadNode], a: u32, b: u32) -> Road {
        let (p, q) = (nodes[a as usize].position, nodes[b as usize].position);
        let n = (p.distance(q).ceil() as usize).max(1);
        let pts = (0..=n).map(|i| p.lerp(q, i as f64 / n as f64)).collect();
        Road { class: RoadClass::Gravel, width: 4.0, start: a, end: b, line: Polyline::new(pts) }
    }

    /// A 4 × 4 lattice of nodes 50 m apart with some links missing, plus a curved road.
    fn lattice() -> RoadNetwork {
        let mut nodes = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                nodes.push(node(50.0 * x as f64, 50.0 * y as f64));
            }
        }
        let mut roads = Vec::new();
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = y * 4 + x;
                if x < 3 && !(y == 1 && x == 1) {
                    roads.push(straight(&nodes, i, i + 1));
                }
                if y < 3 && !(x == 2 && y == 0) {
                    roads.push(straight(&nodes, i, i + 4));
                }
            }
        }
        // A quarter circle from node 3 (150, 0) bulging out to the east, ending at node 7 (150, 50).
        let pts = (0..=80)
            .map(|k| {
                let a = -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * k as f64 / 80.0;
                DVec3::new(150.0 + 25.0 * a.cos(), 25.0 + 25.0 * a.sin(), 0.0)
            })
            .collect();
        roads.push(Road { class: RoadClass::Paved, width: 6.0, start: 3, end: 7, line: Polyline::new(pts) });
        RoadNetwork::new(nodes, roads).unwrap()
    }

    fn brute_nearest(net: &RoadNetwork, p: DVec2) -> (f64, u32) {
        let mut best = (f64::INFINITY, 0);
        for (r, road) in net.roads().iter().enumerate() {
            let d = road.line.project(p).distance;
            if d < best.0 {
                best = (d, r as u32);
            }
        }
        best
    }

    #[test]
    fn polyline_stations_headings_and_curvature() {
        let net = lattice();
        let arc = &net.roads().last().unwrap().line;
        // A semicircle of radius 25 m, turning left.
        assert!((arc.length() - std::f64::consts::PI * 25.0).abs() < 0.05);
        assert!((arc.curvature_at(0.5 * arc.length()) - 1.0 / 25.0).abs() < 1e-3);
        assert!((arc.heading_at(0.0)).abs() < 0.03);
        let p = arc.point_at(0.5 * arc.length());
        assert!((p - DVec3::new(175.0, 25.0, 0.0)).length() < 0.05);
        // Offsets are positive to the left of travel.
        let line = &net.roads()[0].line; // (0, 0) → (50, 0)
        let pr = line.project(DVec2::new(20.0, 3.0));
        assert!((pr.offset - 3.0).abs() < 1e-12 && (pr.station - 20.0).abs() < 1e-12 && pr.heading.abs() < 1e-12);
        assert!((line.project(DVec2::new(20.0, -2.0)).offset + 2.0).abs() < 1e-12);
        // Slices, forward and reversed.
        let s = line.slice(10.5, 12.5);
        assert_eq!((s.len(), s[0].x, s[s.len() - 1].x), (4, 10.5, 12.5));
        let r = line.slice(12.5, 10.5);
        assert_eq!((r[0].x, r[r.len() - 1].x), (12.5, 10.5));
        assert!((wrap_angle(3.5 * std::f64::consts::PI) + 0.5 * std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn nearest_and_on_road_match_brute_force() {
        let net = lattice();
        let mut rng = Seed::from_u64(1).rng();
        for _ in 0..2000 {
            let p = DVec2::new(rng.range(-60.0, 240.0), rng.range(-60.0, 210.0));
            let (d, _) = brute_nearest(&net, p);
            let max = rng.range(1.0, 80.0);
            match net.nearest(p, max) {
                Some(rp) => {
                    assert!((rp.projection.distance - d).abs() < 1e-9, "{p}: {} vs {d}", rp.projection.distance);
                    assert!(d <= max);
                }
                None => assert!(d > max, "{p}: missed a road {d} m away (max {max})"),
            }
            let on = net.on_road(p);
            let expect = net.roads().iter().any(|r| r.line.project(p).distance <= 0.5 * r.width);
            assert_eq!(on.is_some(), expect, "{p}");
        }
        assert!(RoadNetwork::default().nearest(DVec2::ZERO, 1e9).is_none());
    }

    #[test]
    fn routes_are_shortest_and_continuous() {
        let net = lattice();
        // Node-to-node distances by Floyd–Warshall.
        let n = net.nodes().len();
        let mut d = vec![vec![f64::INFINITY; n]; n];
        for (i, row) in d.iter_mut().enumerate() {
            row[i] = 0.0;
        }
        for r in net.roads() {
            let (a, b, l) = (r.start as usize, r.end as usize, r.line.length());
            d[a][b] = d[a][b].min(l);
            d[b][a] = d[b][a].min(l);
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    d[i][j] = d[i][j].min(d[i][k] + d[k][j]);
                }
            }
        }
        let mut rng = Seed::from_u64(2).rng();
        for _ in 0..300 {
            let pick = |rng: &mut autonomousim_core::rng::SimRng| {
                let r = &net.roads()[rng.below(net.roads().len() as u64) as usize];
                let s = rng.range(0.0, r.line.length());
                let p = r.line.point_at(s).truncate();
                (p, r, s)
            };
            let ((pa, ra, sa), (pb, rb, sb)) = (pick(&mut rng), pick(&mut rng));
            let route = net.route(pa, pb, 1.0).expect("the lattice is connected");
            // Expected: the best combination of end nodes, or along the same road.
            let ends = |r: &Road, s: f64| [(r.start as usize, s), (r.end as usize, r.line.length() - s)];
            let mut expect = f64::INFINITY;
            for (na, ca) in ends(ra, sa) {
                for (nb, cb) in ends(rb, sb) {
                    expect = expect.min(ca + d[na][nb] + cb);
                }
            }
            if std::ptr::eq(ra, rb) {
                expect = expect.min((sa - sb).abs());
            }
            assert!((route.line.length() - expect).abs() < 1e-6, "{} vs {expect}", route.line.length());
            let pts = route.line.points();
            assert!(pts[0].truncate().distance(pa) < 1e-6 && pts[pts.len() - 1].truncate().distance(pb) < 1e-6);
            for w in pts.windows(2) {
                assert!(w[0].distance(w[1]) < 1.01, "gap {} at {}", w[0].distance(w[1]), w[0]);
            }
        }
    }

    #[test]
    fn disconnected_points_have_no_route() {
        let nodes = vec![node(0.0, 0.0), node(40.0, 0.0), node(0.0, 100.0), node(40.0, 100.0)];
        let roads = vec![straight(&nodes, 0, 1), straight(&nodes, 2, 3)];
        let net = RoadNetwork::new(nodes, roads).unwrap();
        assert!(net.route(DVec2::new(5.0, 0.0), DVec2::new(5.0, 100.0), 2.0).is_none());
        assert!(net.route(DVec2::new(5.0, 0.0), DVec2::new(30.0, 0.0), 2.0).is_some());
        assert!(net.route(DVec2::new(5.0, 50.0), DVec2::new(30.0, 0.0), 2.0).is_none(), "too far from any road");
        let bad = RoadNetwork::new(vec![node(0.0, 0.0)], vec![straight(&[node(0.0, 0.0), node(1.0, 0.0)], 0, 1)]);
        assert!(matches!(bad, Err(RoadError::MissingNode(0, 1))));
    }
}
