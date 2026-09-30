//! Urban street surfaces (M8a step 5): carriageways, medians, bike lanes, sidewalks with
//! curbs, and the markings of the lane graph (lane lines, stop lines, crosswalks, turn arrows,
//! parking bays); and the signal heads with the movement each one shows.
//!
//! Everything is drawn as flat strips a few centimetres above the terrain, like the rural
//! ribbons ([`crate::roads`]): the terrain carries the materials, the strips the crisp edges,
//! the markings and the semantic classes.

use crate::mesh::srgb;
use crate::props::chunk_index;
use crate::roads::{Paint, Ribbons};
use autonomousim_core::terrain::Terrain;
use autonomousim_world::lanes::{Area, Control, JunctionKind, LaneGraph, Turn};
use autonomousim_world::obstacles::tags;
use autonomousim_world::{ObstacleShape, Polyline, RoadClass, RoadNetwork, StaticWorld};
use glam::{DQuat, DVec2, DVec3, Vec3};

/// Heights above the terrain (m) of the layers, lowest first.
const CARRIAGEWAY: f64 = 0.06;
const BIKE: f64 = 0.063;
const MEDIAN: f64 = 0.068;
const MARKING: f64 = 0.075;
const SIDEWALK: f64 = 0.08;
const CURB: f64 = 0.085;

/// Colours (sRGB).
const ASPHALT: [u8; 3] = [64, 66, 70];
const BIKE_LANE: [u8; 3] = [122, 72, 66];
const ISLAND: [u8; 3] = [150, 152, 144];
const PAVEMENT: [u8; 3] = [176, 174, 168];
const KERB: [u8; 3] = [206, 206, 200];
const WHITE: [u8; 3] = [232, 232, 226];

/// Width of lane lines, and of edge and centre lines (m).
const LINE: f64 = 0.12;
const EDGE_LINE: f64 = 0.15;
/// Dashed lines: dash length and period (m).
const DASH: f64 = 3.0;
const DASH_PERIOD: f64 = 8.0;
/// Width of a stop line (m); yield lines are dashed.
const STOP_LINE: f64 = 0.4;
/// Length of the crosswalk band before the lanes' ends (m; [`Area::Crosswalk`]).
const CROSSWALK: f64 = 3.0;
/// Width of a curb as drawn (m).
const CURB_WIDTH: f64 = 0.2;
/// Longest piece of a strip (m): pieces go to the chunk of their start.
const PIECE: f64 = 16.0;

/// Unit vector to the left of heading `h`.
fn left(h: f64) -> DVec2 {
    DVec2::new(-h.sin(), h.cos())
}

/// A row of a strip: a point on its guide line and the unit vector to the left.
type Row = (DVec2, DVec2);

/// Rows along `line` from station `s0` to `s1`, at most `step` apart.
fn rows(line: &Polyline, s0: f64, s1: f64, step: f64) -> Vec<Row> {
    let n = ((s1 - s0) / step).ceil().max(1.0) as usize;
    (0..=n)
        .map(|k| {
            let s = s0 + (s1 - s0) * k as f64 / n as f64;
            // The heading of the segment ahead, except at the very end.
            let h = line.heading_at(if k == n { s - 1e-6 } else { s + 1e-6 });
            (line.point_at(s).truncate(), left(h))
        })
        .collect()
}

/// A strip's look: lateral offsets (m, + left, ascending) of its columns, height and colour.
#[derive(Clone, Copy)]
struct Look<'a> {
    offsets: &'a [f64],
    lift: f64,
    color: [u8; 3],
    paint: Paint,
}

/// Where the strips of a map go: one [`Ribbons`] per chunk.
struct Out<'a> {
    world: &'a StaticWorld,
    chunks: &'a mut [Ribbons],
    size: usize,
}

impl Out<'_> {
    /// Append a strip over `rows` (cut into pieces of at most [`PIECE`] m).
    fn strip(&mut self, rows: &[Row], look: Look) {
        let mut start = 0;
        while start + 1 < rows.len() {
            let mut end = start + 1;
            let mut run = rows[start].0.distance(rows[end].0);
            while end + 1 < rows.len() && run < PIECE {
                run += rows[end].0.distance(rows[end + 1].0);
                end += 1;
            }
            self.piece(&rows[start..=end], look);
            start = end;
        }
    }

    fn piece(&mut self, rows: &[Row], look: Look) {
        let k = chunk_index(self.world.grid(), self.size, rows[0].0.extend(0.0));
        let out = &mut self.chunks[k];
        let cols = look.offsets.len() as u32;
        let color = srgb(look.color);
        let base = out.mesh.vertex_count() as u32;
        for &(p, l) in rows {
            for &off in look.offsets {
                let q = p + off * l;
                let (h, n) = self.world.grid().height_normal(q.x, q.y);
                out.push_vertex(q.extend(h + look.lift).as_vec3(), n.as_vec3(), color, look.paint);
            }
        }
        for r in 0..rows.len() as u32 - 1 {
            for c in 0..cols - 1 {
                // Rows along the guide, columns from right to left: counter-clockwise from above.
                let (i00, i01) = (base + r * cols + c, base + r * cols + c + 1);
                let (i10, i11) = (i00 + cols, i01 + cols);
                out.mesh.push_triangle(i00, i10, i11);
                out.mesh.push_triangle(i00, i11, i01);
            }
        }
    }

    /// Append the rows of `line` between `s0` and `s1` as dashes of `dash` m every `period` m,
    /// the first starting `phase` m after `s0`.
    fn dashed(&mut self, line: &Polyline, s0: f64, s1: f64, phase: f64, (dash, period): (f64, f64), look: Look) {
        let mut s = s0 + phase;
        while s < s1 {
            let e = (s + dash).min(s1);
            if e - s > 0.3 * dash {
                self.strip(&rows(line, s, e, 1.0), look);
            }
            s += period;
        }
    }

    /// Append flat triangles (horizontal points), each wound counter-clockwise from above.
    fn triangles(&mut self, tris: &[[DVec2; 3]], lift: f64, color: [u8; 3], paint: Paint) {
        let Some(first) = tris.first() else { return };
        let k = chunk_index(self.world.grid(), self.size, first[0].extend(0.0));
        let out = &mut self.chunks[k];
        let color = srgb(color);
        for t in tris {
            let t = if (t[1] - t[0]).perp_dot(t[2] - t[0]) < 0.0 { [t[0], t[2], t[1]] } else { *t };
            let base = out.mesh.vertex_count() as u32;
            for q in t {
                let (h, n) = self.world.grid().height_normal(q.x, q.y);
                out.push_vertex(q.extend(h + lift).as_vec3(), n.as_vec3(), color, paint);
            }
            out.mesh.push_triangle(base, base + 1, base + 2);
        }
    }
}

/// A white marking with columns at `offsets`.
fn white(offsets: &[f64]) -> Look<'_> {
    Look { offsets, lift: MARKING, color: WHITE, paint: Paint::Marking }
}

/// Offsets of a line of width `w` centred at `y`.
fn line_at(y: f64, w: f64) -> [f64; 2] {
    [y - 0.5 * w, y + 0.5 * w]
}

/// Append the surfaces and markings of the urban streets of `world` to `chunks` (one per
/// terrain chunk of `size` cells).
pub(crate) fn streets(world: &StaticWorld, chunks: &mut [Ribbons], size: usize) {
    let net = world.roads();
    if !net.has_sections() {
        return;
    }
    let g = net.lanes();
    let mut out = Out { world, chunks, size };
    let mut lanes_of = vec![Vec::new(); net.roads().len()];
    for (i, l) in g.lanes().iter().enumerate() {
        lanes_of[l.road as usize].push(i);
    }
    for (i, road) in net.roads().iter().enumerate() {
        if !road.class.is_urban() || road.line.points().len() < 2 {
            continue;
        }
        surfaces(&mut out, net, i);
        lane_lines(&mut out, net, &lanes_of[i]);
        crosswalks(&mut out, net, i);
    }
    stop_lines(&mut out, net);
    arrows(&mut out, g);
    {
        for bay in &world.sites().bays {
            let c = bay.corners();
            for k in 0..4 {
                let (a, b) = (c[k], c[(k + 1) % 4]);
                let d = b - a;
                let h = d.y.atan2(d.x);
                // The outline lies inside the bay: the corners run counter-clockwise.
                out.strip(
                    &[(a, left(h)), (b, left(h))],
                    Look { offsets: &[0.0, 0.1], lift: MARKING, color: WHITE, paint: Paint::Marking },
                );
            }
        }
    }
}

/// Carriageway, median, bike lanes, sidewalks and curbs of road `i`.
fn surfaces(out: &mut Out, net: &RoadNetwork, i: usize) {
    let road = &net.roads()[i];
    let section = net.section(i);
    let g = net.lanes();
    let [sa, sb] = g.setbacks(i);
    let len = road.line.length();
    let half = 0.5 * road.width;
    let across = [-half, -0.5 * half, 0.0, 0.5 * half, half];
    out.strip(
        &rows(&road.line, 0.0, len, 2.0),
        Look { offsets: &across, lift: CARRIAGEWAY, color: ASPHALT, paint: Paint::Road },
    );
    let (s0, s1) = (sa, (len - sb).max(sa));
    let between = rows(&road.line, s0, s1, 2.0);
    if between.len() >= 2 {
        if section.median > 0.0 {
            let m = 0.5 * section.median;
            out.strip(&between, Look { offsets: &[-m, m], lift: MEDIAN, color: ISLAND, paint: Paint::Sidewalk });
        }
        for side in 0..2 {
            if section.bike[side] <= 0.0 {
                continue;
            }
            // Side 0 lies right of the road's direction (y < 0), side 1 left of it; a one-way
            // street's lanes run from its left edge, the bike lane after them.
            let (a, b) = if section.one_way() {
                let e = half - f64::from(section.lanes[0]) * section.lane_width;
                (e - section.bike[0], e)
            } else {
                let sign = if side == 0 { -1.0 } else { 1.0 };
                let inner = 0.5 * section.median + f64::from(section.lanes[side]) * section.lane_width;
                (sign * inner, sign * (inner + section.bike[side]))
            };
            let look = Look { offsets: &[a.min(b), a.max(b)], lift: BIKE, color: BIKE_LANE, paint: Paint::Road };
            out.strip(&between, look);
        }
    }
    // Sidewalks and curbs, where the area classes say sidewalk (not across other streets, nor
    // in the corners that turning traffic sweeps).
    let edge = rows(&road.line, 0.0, len, 1.0);
    for side in 0..2 {
        let w = section.sidewalk[side];
        if w <= 0.0 {
            continue;
        }
        let sign = if side == 0 { -1.0 } else { 1.0 };
        // Cells of 1 m along and at most 0.6 m across, kept where all four corners are
        // sidewalk (corners inset by 5 cm from the kerb and the back of the sidewalk).
        let n = (w / 0.6).ceil() as usize;
        let across: Vec<f64> = (0..=n).map(|k| half + w * k as f64 / n as f64).collect();
        let probe = |k: usize| across[k].clamp(half + 0.05, half + w - 0.05);
        let ok: Vec<Vec<bool>> = edge
            .iter()
            .map(|&(p, l)| (0..=n).map(|k| net.area(p + sign * probe(k) * l) == Area::Sidewalk).collect())
            .collect();
        let cell = |r: usize, k: usize| ok[r][k] && ok[r][k + 1] && ok[r + 1][k] && ok[r + 1][k + 1];
        let full = |r: usize| (0..n).all(|k| cell(r, k));
        let sorted = |a: f64, b: f64| [a.min(b), a.max(b)];
        let curb = sorted(sign * half, sign * (half + CURB_WIDTH));
        // Runs of rows (`pick` of each interval) as strips of `offsets`.
        let mut runs = |pick: &dyn Fn(usize) -> bool, offsets: &[f64], with_curb: bool| {
            let mut r = 0;
            while r + 1 < edge.len() {
                if !pick(r) {
                    r += 1;
                    continue;
                }
                let mut e = r + 1;
                while e + 1 < edge.len() && pick(e) {
                    e += 1;
                }
                let run = &edge[r..=e];
                out.strip(run, Look { offsets, lift: SIDEWALK, color: PAVEMENT, paint: Paint::Sidewalk });
                if with_curb {
                    out.strip(run, Look { offsets: &curb, lift: CURB, color: KERB, paint: Paint::Sidewalk });
                }
                r = e;
            }
        };
        // Whole width where every cell is kept, else cell by cell.
        runs(&full, &sorted(sign * half, sign * (half + w)), true);
        for k in 0..n {
            let walk = sorted(sign * across[k], sign * across[k + 1]);
            runs(&|r| !full(r) && cell(r, k), &walk, k == 0);
        }
    }
}

/// Lines between and beside the lanes of a road (`lanes`: its lanes).
fn lane_lines(out: &mut Out, net: &RoadNetwork, lanes: &[usize]) {
    let g = net.lanes();
    for &i in lanes {
        let l = &g.lanes()[i];
        let len = l.line.length();
        if len < 1.0 {
            continue;
        }
        let r = &net.roads()[l.road as usize];
        let section = net.section(l.road as usize);
        let hw = 0.5 * l.width;
        // Right boundary: dashed to the next lane, else the solid edge of the travel lanes.
        match l.right {
            Some(_) => out.dashed(&l.line, 0.0, len, 1.0, (DASH, DASH_PERIOD), white(&line_at(-hw, LINE))),
            None => out.strip(&rows(&l.line, 0.0, len, 1.0), white(&line_at(-hw + 0.5 * EDGE_LINE, EDGE_LINE))),
        }
        if l.index != 0 {
            continue;
        }
        // Left boundary of the leftmost lane: the centre line (drawn once, by direction 0),
        // or the solid edge beside a median or of a one-way street.
        let centre = !section.one_way() && section.median == 0.0;
        if !centre {
            out.strip(&rows(&l.line, 0.0, len, 1.0), white(&line_at(hw - 0.5 * EDGE_LINE, EDGE_LINE)));
        } else if l.dir == 0 {
            if matches!(r.class, RoadClass::Arterial | RoadClass::Collector) {
                out.strip(&rows(&l.line, 0.0, len, 1.0), white(&line_at(hw, EDGE_LINE)));
            } else {
                out.dashed(&l.line, 0.0, len, 1.0, (DASH, DASH_PERIOD), white(&line_at(hw, LINE)));
            }
        }
    }
}

/// Whether node `n` is a junction with crossings (three or more approaches).
fn crossing(g: &LaneGraph, n: u32) -> bool {
    g.junctions()[n as usize].approaches.len() >= 3
}

/// Zebra crossings at both ends of road `i` where it meets a junction and has sidewalks: the
/// [`Area::Crosswalk`] band, 3 m beyond the setback.
fn crosswalks(out: &mut Out, net: &RoadNetwork, i: usize) {
    let road = &net.roads()[i];
    let section = net.section(i);
    if section.sidewalk[0].max(section.sidewalk[1]) <= 0.0 {
        return;
    }
    let g = net.lanes();
    let [sa, sb] = g.setbacks(i);
    let len = road.line.length();
    let half = 0.5 * road.width;
    let mut bands = Vec::new();
    if crossing(g, road.start) {
        bands.push((sa + 0.3, sa + CROSSWALK - 0.3));
    }
    if crossing(g, road.end) {
        bands.push((len - sb - CROSSWALK + 0.3, len - sb - 0.3));
    }
    for (s0, s1) in bands {
        if s0 < 0.0 || s1 > len || s1 <= s0 {
            continue;
        }
        let r = rows(&road.line, s0, s1, 1.0);
        let mut y = -half + 0.4;
        while y + 0.5 <= half - 0.3 {
            out.strip(&r, Look { offsets: &[y, y + 0.5], lift: MARKING, color: WHITE, paint: Paint::Marking });
            y += 1.0;
        }
    }
}

/// Stop lines of signal and stop approaches; dashed give-way lines of yield approaches. They
/// lie at the lanes' ends, or behind the crosswalk where the street has one (the crosswalk
/// band lies before the lanes' ends; the events keep the lanes' ends).
fn stop_lines(out: &mut Out, net: &RoadNetwork) {
    let g = net.lanes();
    for j in g.junctions() {
        if j.kind == JunctionKind::Through || j.kind == JunctionKind::DeadEnd {
            continue;
        }
        for a in &j.approaches {
            let section = net.section(a.road as usize);
            let urban = net.roads()[a.road as usize].class.is_urban();
            let crosswalk = urban && section.sidewalk[0].max(section.sidewalk[1]) > 0.0 && j.approaches.len() >= 3;
            let (l, r) = (a.stop_line[0].truncate(), a.stop_line[1].truncate());
            let d = r - l;
            if d.length() < 0.5 {
                continue;
            }
            // Rows from the left edge to the right one: "left" of that is the travel direction,
            // so the line lies at negative offsets.
            let n = left(d.y.atan2(d.x));
            let back = if crosswalk { CROSSWALK } else { 0.0 };
            match a.control {
                Control::Signal | Control::Stop => {
                    let steps = d.length().ceil() as usize;
                    let r: Vec<Row> = (0..=steps).map(|k| (l + d * (k as f64 / steps as f64), n)).collect();
                    out.strip(&r, white(&[-back - STOP_LINE, -back]));
                }
                Control::Yield => {
                    let (len, u) = (d.length(), d.normalize());
                    let mut s = 0.1;
                    while s + 0.5 <= len {
                        out.strip(&[(l + u * s, n), (l + u * (s + 0.5), n)], white(&[-back - 0.3, -back]));
                        s += 0.9;
                    }
                }
                Control::None => {}
            }
        }
    }
}

/// Turn arrows on the lanes that lead into junctions with crossings, 4–9 m before their ends
/// (just behind a stop line drawn behind a crosswalk).
fn arrows(out: &mut Out, g: &LaneGraph) {
    for l in g.lanes() {
        let len = l.line.length();
        if len < 16.0 || !crossing(g, l.to_node) {
            continue;
        }
        let turns = |t: Turn| l.successors.iter().any(|&c| g.connectors()[c as usize].turn == t);
        let (straight, lft, rgt) = (turns(Turn::Straight), turns(Turn::Left), turns(Turn::Right));
        if !(straight || lft || rgt) {
            continue;
        }
        let s = len - 10.0;
        let o = l.line.point_at(s).truncate();
        let h = l.line.heading_at(s);
        let (x, y) = (DVec2::new(h.cos(), h.sin()), left(h));
        let at = |u: f64, v: f64| o + x * u + y * v;
        let quad = |u0: f64, v0: f64, u1: f64, v1: f64| {
            [[at(u0, v0), at(u1, v0), at(u1, v1)], [at(u0, v0), at(u1, v1), at(u0, v1)]]
        };
        let mut tris: Vec<[DVec2; 3]> = Vec::new();
        let shaft = if straight { 3.6 } else { 2.9 };
        tris.extend(quad(0.0, -0.08, shaft, 0.08));
        if straight {
            tris.push([at(3.6, -0.35), at(5.0, 0.0), at(3.6, 0.35)]);
        }
        for (on, side) in [(lft, 1.0), (rgt, -1.0)] {
            if on {
                tris.extend(quad(2.6, 0.0, 2.9, side * 0.8));
                tris.push([at(2.3, side * 0.8), at(3.2, side * 0.8), at(2.75, side * 1.4)]);
            }
        }
        out.triangles(&tris, MARKING, WHITE, Paint::Marking);
    }
}

/// A signal head: where it is, which way its lamps face, and the connector whose light it
/// shows (straight through where the approach has one).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignalHead {
    /// Centre of the head's box.
    pub position: DVec3,
    /// Rotation of the box; the lamps are on its +x face, towards the traffic.
    pub rotation: DQuat,
    /// Half extents of the box (m).
    pub half_extents: DVec3,
    pub connector: u32,
}

/// The signal heads of `world`: its `SIGNAL` boxes, each matched to the approach whose stop
/// line is nearest and faces it.
pub fn signal_heads(world: &StaticWorld) -> Vec<SignalHead> {
    let net = world.roads();
    if !net.has_sections() {
        return Vec::new();
    }
    let g = net.lanes();
    let mut approaches = Vec::new();
    for j in g.junctions() {
        if j.kind != JunctionKind::Signal {
            continue;
        }
        for a in &j.approaches {
            let pick = |t: Option<Turn>| {
                a.lanes.iter().flat_map(|&l| &g.lanes()[l as usize].successors).copied().find(|&c| {
                    t.is_none_or(|t| g.connectors()[c as usize].turn == t) && g.connector_signal(c).is_some()
                })
            };
            let Some(c) = pick(Some(Turn::Straight)).or_else(|| pick(None)) else { continue };
            let mid = 0.5 * (a.stop_line[0] + a.stop_line[1]).truncate();
            let d = a.stop_line[1].truncate() - a.stop_line[0].truncate();
            // Towards the arriving traffic: against the travel direction.
            let back = -left(d.y.atan2(d.x));
            approaches.push((mid, back, c));
        }
    }
    let mut out = Vec::new();
    for o in world.obstacle_set().obstacles() {
        let ObstacleShape::Cuboid { half_extents } = o.shape else { continue };
        if o.tag != tags::SIGNAL {
            continue;
        }
        let p = o.pose.pos.truncate();
        let face = (o.pose.rot * DVec3::X).truncate();
        let best = approaches
            .iter()
            .filter(|(_, back, _)| back.dot(face) > 0.9)
            .map(|&(mid, _, c)| (mid.distance(p), c))
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((d, c)) = best
            && d < 20.0
        {
            out.push(SignalHead { position: o.pose.pos, rotation: o.pose.rot, half_extents, connector: c });
        }
    }
    out
}

/// Which lamp of a head is lit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lamp {
    Dark,
    Red,
    Amber,
    Green,
}

/// The lamps of a head (red on top, amber, green) in the head's frame, just in front of the
/// +x face of a box of half extents `h`, with `lit` lit (the others dim).
pub fn signal_lamps(h: DVec3, lit: Lamp) -> crate::MeshData {
    let mut m = crate::MeshData::new();
    let size = (0.6 * h.y).min(0.3 * h.z) as f32;
    for (k, (lamp, on, off)) in [
        (Lamp::Red, [255, 40, 30], [70, 24, 22]),
        (Lamp::Amber, [255, 176, 20], [70, 52, 18]),
        (Lamp::Green, [40, 255, 110], [20, 64, 34]),
    ]
    .into_iter()
    .enumerate()
    {
        let z = (h.z * (0.62 - 0.62 * k as f64)) as f32;
        let x = h.x as f32 + 0.01;
        let c = srgb(if lamp == lit { on } else { off });
        let (a, b, cc, d) = (
            Vec3::new(x, -size, z - size),
            Vec3::new(x, size, z - size),
            Vec3::new(x, size, z + size),
            Vec3::new(x, -size, z + size),
        );
        m.push_flat_triangle(a, b, cc, c);
        m.push_flat_triangle(a, cc, d, c);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roads::roads_by_chunk;
    use crate::terrain::chunks;
    use autonomousim_core::material::MaterialId;
    use autonomousim_core::math::Pose;
    use autonomousim_world::{NodeKind, Obstacle, ObstacleSet, Road, RoadNode, Section, testworlds};

    fn arterial() -> Section {
        Section { lanes: [2, 2], lane_width: 3.5, median: 2.0, bike: [1.5; 2], parking: [0.0; 2], sidewalk: [3.0; 2] }
    }

    fn undivided() -> Section {
        // Lanes wide enough for a U-turn in a half circle at the dead end.
        Section { median: 0.0, lane_width: 3.75, ..arterial() }
    }

    fn street() -> Section {
        Section { lanes: [1, 1], lane_width: 3.0, median: 0.0, bike: [0.0; 2], parking: [2.0; 2], sidewalk: [2.0; 2] }
    }

    /// A signalled cross of an east–west arterial and a north–south collector on flat ground,
    /// arms 150 m long (roads 0 east, 1 west, 2 north, 3 south, all from the centre; the west arm
    /// without a median), with a signal head beside the stop line of every approach, facing the traffic.
    fn cross() -> StaticWorld {
        let at = |x: f64, y: f64, kind| RoadNode { position: DVec3::new(x, y, 0.0), kind };
        let nodes = vec![
            at(0.0, 0.0, NodeKind::Junction),
            at(150.0, 0.0, NodeKind::End),
            at(-150.0, 0.0, NodeKind::End),
            at(0.0, 150.0, NodeKind::End),
            at(0.0, -150.0, NodeKind::End),
        ];
        let road = |end: u32, class: RoadClass, s: &Section| {
            let q = nodes[end as usize].position;
            let pts = (0..=150).map(|i| q * (f64::from(i) / 150.0)).collect();
            Road { class, width: s.width(), start: 0, end, line: Polyline::new(pts) }
        };
        let roads = vec![
            road(1, RoadClass::Arterial, &arterial()),
            road(2, RoadClass::Arterial, &undivided()),
            road(3, RoadClass::Collector, &street()),
            road(4, RoadClass::Collector, &street()),
        ];
        let net = RoadNetwork::new(nodes, roads)
            .unwrap()
            .with_sections(vec![arterial(), undivided(), street(), street()])
            .unwrap();
        let mut heads = Vec::new();
        for a in &net.lanes().junctions()[0].approaches {
            let (l, r) = (a.stop_line[0].truncate(), a.stop_line[1].truncate());
            let d = (r - l).normalize();
            // Travel is d turned left; the head stands beyond the right end, facing back.
            let travel = left(d.y.atan2(d.x));
            let p = r + d * 1.0;
            let yaw = (-travel).y.atan2((-travel).x);
            let shape = ObstacleShape::Cuboid { half_extents: DVec3::new(0.15, 0.15, 0.45) };
            let pose = Pose::new(p.extend(4.0), DQuat::from_rotation_z(yaw));
            heads.push(Obstacle::solid(shape, pose, MaterialId::METAL).with_tag(tags::SIGNAL));
        }
        let flat = testworlds::flat(400.0);
        StaticWorld::new(flat.meta.clone(), flat.grid().clone(), ObstacleSet::new(heads), flat.materials().clone())
            .with_roads(net)
    }

    /// Paint of the topmost ribbon triangle over `p`, if any.
    fn paint_at(chunks: &[Ribbons], p: DVec2) -> Option<Paint> {
        let mut best: Option<(f32, Paint)> = None;
        for r in chunks {
            for t in r.mesh.indices.as_chunks::<3>().0 {
                let v = t.map(|i| Vec3::from_array(r.mesh.positions[i as usize]));
                let q = p.as_vec2();
                let e = |a: Vec3, b: Vec3| (b.truncate() - a.truncate()).perp_dot(q - a.truncate());
                let inside = e(v[0], v[1]) >= 0.0 && e(v[1], v[2]) >= 0.0 && e(v[2], v[0]) >= 0.0;
                if inside && best.is_none_or(|(z, _)| v[0].z > z) {
                    best = Some((v[0].z, r.paint[t[0] as usize]));
                }
            }
        }
        best.map(|b| b.1)
    }

    #[test]
    fn streets_are_painted_as_their_areas() {
        let w = cross();
        let net = w.roads();
        let cs = chunks(w.grid(), 32);
        let out = roads_by_chunk(&w, &cs, 32);
        for r in &out {
            assert_eq!(r.paint.len(), r.mesh.vertex_count());
            for t in r.mesh.indices.as_chunks::<3>().0 {
                let [a, b, c] = t.map(|i| Vec3::from_array(r.mesh.positions[i as usize]));
                assert!((b - a).cross(c - a).z > 0.0, "{a} {b} {c}");
                assert!(a.z > 0.055 && a.z < 0.09, "{a}");
            }
        }
        let at = |x: f64, y: f64| paint_at(&out, DVec2::new(x, y));
        // Arterial (y < 0 eastbound): median ±1, lanes to −8, bike lane to −9.5, sidewalk to −12.5.
        assert_eq!(net.area(DVec2::new(50.0, -2.75)), Area::Lane);
        assert_eq!(at(50.0, -2.75), Some(Paint::Road));
        assert_eq!(at(50.0, 0.0), Some(Paint::Sidewalk), "median island");
        assert_eq!(at(50.0, -8.75), Some(Paint::Road), "bike lane");
        assert_eq!(at(50.0, -11.0), Some(Paint::Sidewalk));
        assert_eq!(at(50.0, -13.5), None);
        // Solid edge beside the median and at the bike lane; the lane divider is dashed.
        for x in [30.0, 33.0, 36.0, 39.0] {
            assert_eq!(at(x, -1.07), Some(Paint::Marking), "median edge at {x}");
            assert_eq!(at(x, -7.93), Some(Paint::Marking), "edge line at {x}");
        }
        let dashes = (0..40).filter(|k| at(30.0 + 0.5 * f64::from(*k), -4.5) == Some(Paint::Marking)).count();
        assert!((12..=20).contains(&dashes), "{dashes} of 40 on the divider");
        // The west arm has no median: a solid centre line (arterial) between its directions.
        let g = net.lanes();
        assert_eq!(at(-50.0, 0.0), Some(Paint::Marking));
        assert_eq!(at(-50.0, 1.875), Some(Paint::Road));
        assert_eq!(at(-50.0, -1.875), Some(Paint::Road));
        let [sa, _] = g.setbacks(0);
        // Crosswalk stripes across the arterial beyond the setback, alternating with asphalt.
        let stripes: Vec<bool> =
            (0..32).map(|k| at(sa + 1.5, -7.9 + 0.25 * f64::from(k)) == Some(Paint::Marking)).collect();
        assert!(stripes.iter().filter(|&&m| m).count() >= 12 && stripes.contains(&false), "{stripes:?}");
        // Stop lines at every approach, behind the crosswalk (all arms have sidewalks), then
        // asphalt.
        for a in &g.junctions()[0].approaches {
            let mid = 0.5 * (a.stop_line[0] + a.stop_line[1]).truncate();
            let d = a.stop_line[1].truncate() - a.stop_line[0].truncate();
            let back = -left(d.y.atan2(d.x));
            // Between the lanes of the approach (not on a lane line).
            let lane_mid = g.lanes()[a.lanes[0] as usize].line.points().last().unwrap().truncate();
            let q = mid + (lane_mid - mid).dot(d.normalize()) * d.normalize();
            assert_eq!(paint_at(&out, q + back * (CROSSWALK + 0.2)), Some(Paint::Marking));
            assert_eq!(paint_at(&out, q + back * (CROSSWALK + 0.6)), Some(Paint::Road));
            // A turn arrow's shaft on each lane, 6.5–9 m before the lane's end.
            for &l in &a.lanes {
                let lane = &g.lanes()[l as usize];
                let s = lane.line.length() - 8.5;
                assert_eq!(paint_at(&out, lane.line.point_at(s).truncate()), Some(Paint::Marking), "lane {l}");
            }
        }
        // Signal heads find their approaches; each shows a signalled movement straight on.
        let heads = signal_heads(&w);
        assert_eq!(heads.len(), 4);
        for h in &heads {
            let c = &g.connectors()[h.connector as usize];
            assert_eq!(c.turn, Turn::Straight);
            assert!(g.connector_signal(h.connector).is_some());
            let lane = &g.lanes()[c.from as usize];
            // The lamps face the arriving traffic.
            let travel = DVec2::from_angle(lane.line.heading_at(lane.line.length()));
            assert!((h.rotation * DVec3::X).truncate().dot(travel) < -0.99);
        }
    }

    #[test]
    fn lamps_light_one_colour() {
        let h = DVec3::new(0.15, 0.15, 0.45);
        let brightest = |m: &crate::MeshData| m.colors.iter().flat_map(|c| &c[..3]).copied().fold(0.0, f32::max);
        for lit in [Lamp::Red, Lamp::Amber, Lamp::Green] {
            let m = signal_lamps(h, lit);
            assert_eq!(m.triangle_count(), 6);
            // Lamps lie just in front of the +x face and face +x.
            assert!(m.positions.iter().all(|p| (p[0] - 0.16).abs() < 1e-6 && p[2].abs() < 0.45));
            assert!(m.normals.iter().all(|n| n[0] > 0.99));
            let lit_quads = m.colors.chunks(6).filter(|c| c[0].iter().take(3).any(|&v| v > 0.9)).count();
            assert_eq!(lit_quads, 1, "{lit:?}");
            assert!(brightest(&m) > 0.9);
        }
        let dark = signal_lamps(h, Lamp::Dark);
        assert!(brightest(&dark) < 0.1);
    }
}
