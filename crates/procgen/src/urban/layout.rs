//! The street layout of an urban map, on a [`Graph`]:
//! 1. **Arterials**: radial roads from the downtown out to the map edge (becoming rural paved
//!    roads beyond the city) and a ring road, meandering gently.
//! 2. **Districts**: the Voronoi cells of seeds spread over the city. Downtown districts share
//!    one perturbed grid; the others get a grid of their own (own orientation and blocks) or
//!    grow organic streets.
//! 3. **Grids**: lines of each district's grid, clipped to the district, every n-th one a
//!    collector.
//! 4. **Organic streets**: grown from the arterials and collectors into organic districts in
//!    the manner of Parish & Müller (2001): segments of random length with a drifting
//!    direction, branching at right angles, ending where they meet the graph (snapped to
//!    nearby nodes and streets), leave their district or come too close to another street;
//!    some ends stay as cul-de-sacs.
//! 5. **Clean-up**: dead ends joined to the graph ahead, short segments collapsed, sharp
//!    junction angles and crowded junctions resolved, short spurs pruned, the main component
//!    kept.
//! 6. **Roundabouts** at some junctions: the incident streets cut at the ring radius and
//!    joined by a one-way ring (counter-clockwise, for right-hand traffic).

use super::graph::{End, Graph, NodeId, angle_between, rank};
use super::{DistrictKind, UrbanConfig};
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_world::RoadClass;
use glam::DVec2;
use std::f64::consts::{PI, TAU};

/// A roundabout: its ring's street id, centre and radius.
#[derive(Clone, Debug)]
pub(crate) struct Ring {
    pub street: u32,
    pub centre: DVec2,
    pub radius: f64,
}

/// What the layout needs to know about the site.
pub(crate) struct Site<'a> {
    pub c: &'a UrbanConfig,
    pub centre: DVec2,
    pub radius: f64,
    /// City share at a point: 1 in the city, 0 in the countryside.
    pub city: &'a (dyn Fn(DVec2) -> f64 + Sync),
    /// Whether a point is too close to water for a street.
    pub wet: &'a (dyn Fn(DVec2) -> bool + Sync),
    pub districts: &'a Districts,
}

impl Site<'_> {
    fn in_city(&self, p: DVec2) -> bool {
        (self.city)(p) >= 0.5
    }

    fn in_map(&self, p: DVec2) -> bool {
        let h = 0.5 * self.c.size - 1.0;
        p.x.abs() <= h && p.y.abs() <= h
    }

    /// A street may run from `p` to `q`: on the map and dry all along.
    fn ok(&self, p: DVec2, q: DVec2) -> bool {
        let n = (p.distance(q) / 2.0).ceil().max(1.0) as usize;
        (0..=n).all(|k| {
            let x = p.lerp(q, k as f64 / n as f64);
            self.in_map(x) && !(self.wet)(x)
        })
    }
}

/// District seeds with their kinds and grid parameters.
pub(crate) struct Districts {
    pub seeds: Vec<DVec2>,
    pub kinds: Vec<DistrictKind>,
    /// Grid orientation (rad) and block sizes (m) along and across it.
    pub angle: Vec<f64>,
    pub block: Vec<[f64; 2]>,
    /// Everything within `downtown_radius` of the centre is downtown (district `downtown_id`).
    centre: DVec2,
    downtown_radius: f64,
    downtown_id: usize,
}

impl Districts {
    /// Seeds on a jittered grid over the city disk; those within the downtown radius share
    /// the downtown grid.
    pub fn new(c: &UrbanConfig, centre: DVec2, radius: f64, rng: &mut SimRng) -> Self {
        let d = &c.districts;
        let k = (radius / d.spacing).ceil() as i64;
        let downtown_angle = rng.range(0.0, 0.5 * PI);
        let downtown_block =
            [rng.range(d.downtown_block[0], d.downtown_block[1]), rng.range(d.downtown_block[0], d.downtown_block[1])];
        let mut out = Self {
            seeds: Vec::new(),
            kinds: Vec::new(),
            angle: Vec::new(),
            block: Vec::new(),
            centre,
            downtown_radius: d.downtown * radius,
            downtown_id: 0,
        };
        for gy in -k..=k {
            for gx in -k..=k {
                let (jx, jy, u_kind, angle) = (rng.uniform(), rng.uniform(), rng.uniform(), rng.range(0.0, 0.5 * PI));
                let (bx, by) = (rng.range(d.block[0], d.block[1]), rng.range(d.block[0], d.block[1]));
                let p = centre + d.spacing * DVec2::new(gx as f64 + 0.8 * (jx - 0.5), gy as f64 + 0.8 * (jy - 0.5));
                let r = p.distance(centre);
                if r > radius + 0.5 * d.spacing {
                    continue;
                }
                let kind = if r < d.downtown * radius {
                    DistrictKind::Downtown
                } else if u_kind < d.grid_share {
                    DistrictKind::Grid
                } else {
                    DistrictKind::Organic
                };
                out.seeds.push(p);
                out.kinds.push(kind);
                if kind == DistrictKind::Downtown {
                    out.angle.push(downtown_angle);
                    out.block.push(downtown_block);
                } else {
                    out.angle.push(angle);
                    out.block.push([bx, by]);
                }
            }
        }
        // At least one downtown district: the seed nearest the centre.
        if !out.kinds.contains(&DistrictKind::Downtown)
            && let Some(i) = (0..out.seeds.len()).min_by(|&i, &j| {
                out.seeds[i].distance_squared(centre).total_cmp(&out.seeds[j].distance_squared(centre)).then(i.cmp(&j))
            })
        {
            out.kinds[i] = DistrictKind::Downtown;
            out.angle[i] = downtown_angle;
            out.block[i] = downtown_block;
        }
        out.downtown_id = out.kinds.iter().position(|&k| k == DistrictKind::Downtown).unwrap_or(0);
        out
    }

    /// The district containing `p` (nearest seed, lowest index on ties).
    pub fn at(&self, p: DVec2) -> usize {
        if p.distance(self.centre) < self.downtown_radius {
            return self.downtown_id;
        }
        let mut best = (usize::MAX, f64::INFINITY);
        for (i, s) in self.seeds.iter().enumerate() {
            let d = s.distance_squared(p);
            if d < best.1 {
                best = (i, d);
            }
        }
        best.0
    }

    /// Downtown districts count as one (they share their grid).
    fn same(&self, a: usize, b: usize) -> bool {
        a == b || (self.kinds[a] == DistrictKind::Downtown && self.kinds[b] == DistrictKind::Downtown)
    }
}

fn rotate(v: DVec2, a: f64) -> DVec2 {
    let (s, c) = (libm::sin(a), libm::cos(a));
    DVec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
}

fn unit(a: f64) -> DVec2 {
    DVec2::new(libm::cos(a), libm::sin(a))
}

/// Street ids: one per street, in the order streets are laid out.
struct Streets(u32);

impl Streets {
    fn next(&mut self) -> u32 {
        self.0 += 1;
        self.0 - 1
    }
}

/// Lay out the streets; returns the graph and its roundabouts.
pub(crate) fn layout(site: &Site, seed: &Seed) -> (Graph, Vec<Ring>) {
    let s = &site.c.streets;
    let mut g = Graph::new(site.c.size, s.snap);
    let mut ids = Streets(0);
    arterials(site, &mut g, &mut ids, &mut seed.child("arterials").rng());
    grids(site, &mut g, &mut ids, &seed.child("grids"));
    organic(site, &mut g, &mut ids, &mut seed.child("organic").rng());
    clean_up(site, &mut g);
    let rings = roundabouts(site, &mut g, &mut ids, &mut seed.child("roundabouts").rng());
    (g, rings)
}

// ------------------------------------------------------------------------------ arterials

fn arterials(site: &Site, g: &mut Graph, ids: &mut Streets, rng: &mut SimRng) {
    let c = site.c;
    let s = &c.streets;
    let half = 0.5 * c.size - 1.0;
    let class_at = |p: DVec2| if site.in_city(p) { RoadClass::Arterial } else { RoadClass::Paved };
    // Radials: from inside the downtown out to the map edge.
    let n = s.radials[0] + rng.below(u64::from(s.radials[1] - s.radials[0] + 1)) as u32;
    let base = rng.range(0.0, TAU);
    for k in 0..n {
        let street = ids.next();
        let spread = TAU / f64::from(n.max(1));
        let aim = base + spread * f64::from(k) + 0.25 * spread * rng.range(-1.0, 1.0);
        let r0 = 0.5 * c.districts.downtown * site.radius;
        // From the first dry point outwards.
        let Some(mut p) = (0..20)
            .map(|i| site.centre + unit(aim) * (r0 + 10.0 * i as f64))
            .find(|&p| site.in_map(p) && !(site.wet)(p))
        else {
            continue;
        };
        let mut prev = End::At(p);
        let mut drift = 0.0;
        loop {
            drift = 0.8 * drift + s.arterial_wiggle * rng.normal();
            // Steer round water: the least turn that keeps the segment dry.
            let step = [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2].into_iter().find_map(|turn| {
                let heading = drift + turn;
                if heading.abs() > 1.4 {
                    return None;
                }
                let mut q = p + unit(aim + heading) * s.arterial_step;
                let mut last = false;
                // Clip to the map edge.
                if q.x.abs() > half || q.y.abs() > half {
                    let d = q - p;
                    let tx = if d.x.abs() > 1e-12 { ((half * d.x.signum()) - p.x) / d.x } else { f64::INFINITY };
                    let ty = if d.y.abs() > 1e-12 { ((half * d.y.signum()) - p.y) / d.y } else { f64::INFINITY };
                    q = p + d * tx.min(ty).clamp(0.0, 1.0);
                    last = true;
                }
                (q.distance(p) >= 1.0 && site.ok(p, q)).then_some((heading, q, last))
            });
            let Some((heading, q, last)) = step else { break };
            drift = heading;
            let n = g.insert(prev, End::At(q), class_at(0.5 * (p + q)), street);
            prev = End::Node(n);
            p = q;
            if last {
                g.nodes[n as usize].fixed = true;
                break;
            }
        }
    }
    // The ring road.
    if s.ring_road > 0.0 {
        let street = ids.next();
        let r = s.ring_road * site.radius;
        let count = ((TAU * r / s.arterial_step).ceil() as usize).max(8);
        let phase = rng.range(0.0, TAU);
        let wobble: Vec<f64> = (0..count).map(|_| rng.range(-1.0, 1.0)).collect();
        let points: Vec<DVec2> = (0..count)
            .map(|i| {
                let a = phase + TAU * i as f64 / count as f64;
                // A smooth radius: the wobble averaged with the neighbours'.
                let w = (wobble[(i + count - 1) % count] + 2.0 * wobble[i] + wobble[(i + 1) % count]) / 4.0;
                site.centre + unit(a) * r * (1.0 + 0.06 * w)
            })
            .collect();
        let mut prev: Option<NodeId> = None;
        for i in 0..=count {
            let q = points[i % count];
            match prev {
                Some(pn) if site.ok(g.p(pn), q) => {
                    prev = Some(g.insert(End::Node(pn), End::At(q), RoadClass::Arterial, street));
                }
                _ => {
                    // Start (or restart after water): a free node.
                    prev = Some(g.insert(End::At(q), End::At(q), RoadClass::Arterial, street));
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------ grids

fn grids(site: &Site, g: &mut Graph, ids: &mut Streets, seed: &Seed) {
    let c = site.c;
    let ds = site.districts;
    let d = &c.districts;
    // Downtown districts share one grid, laid out once from the first of them.
    let mut done_downtown = false;
    for (k, kind) in ds.kinds.iter().enumerate() {
        match kind {
            DistrictKind::Organic => continue,
            DistrictKind::Downtown if done_downtown => continue,
            DistrictKind::Downtown => done_downtown = true,
            DistrictKind::Grid => {}
        }
        let every = if *kind == DistrictKind::Downtown { d.downtown_collector_every } else { d.collector_every };
        let origin = ds.seeds[k];
        let angle = ds.angle[k];
        // How far the district (all downtown ones, for the downtown grid) may reach.
        let reach =
            if *kind == DistrictKind::Downtown { d.downtown * site.radius + 1.5 * d.spacing } else { 1.5 * d.spacing };
        let jitter = d.block_jitter;
        for (family, &block) in ds.block[k].iter().enumerate() {
            // Lines along `u`, spaced by `block` along `w`.
            let u = unit(angle + if family == 0 { 0.0 } else { 0.5 * PI });
            let w = u.perp();
            let mut rng = seed.child_index(k as u64).child_index(family as u64).rng();
            // Offsets, both ways from the seed, each gap jittered.
            let mut offsets = vec![0.0];
            let (mut up, mut down) = (0.0, 0.0);
            while up < reach {
                up += block * (1.0 + jitter * rng.range(-1.0, 1.0));
                offsets.push(up);
                down -= block * (1.0 + jitter * rng.range(-1.0, 1.0));
                offsets.push(down);
            }
            offsets.sort_by(f64::total_cmp);
            let zero = offsets.iter().position(|&o| o == 0.0).expect("the seed's line");
            for (i, &o) in offsets.iter().enumerate() {
                let class = if (i as i64 - zero as i64).rem_euclid(every.max(1) as i64) == 0 {
                    RoadClass::Collector
                } else {
                    RoadClass::Local
                };
                let inside = |t: f64| {
                    let p = origin + w * o + u * t;
                    site.in_city(p) && site.in_map(p) && !(site.wet)(p) && ds.same(ds.at(p), k)
                };
                let step = 4.0;
                let span = reach;
                let mut t = -span;
                let mut run: Option<f64> = None;
                let street = ids.next();
                while t <= span + step {
                    let ins = t <= span && inside(t);
                    match (run, ins) {
                        (None, true) => run = Some(t),
                        (Some(t0), false) => {
                            let t1 = t - step;
                            if t1 - t0 >= c.streets.min_length {
                                let (a, b) = (origin + w * o + u * t0, origin + w * o + u * t1);
                                g.insert(End::At(a), End::At(b), class, street);
                            }
                            run = None;
                        }
                        _ => {}
                    }
                    t += step;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------ organic

struct Grow {
    node: NodeId,
    dir: DVec2,
    class: RoadClass,
    street: u32,
    /// Segments left before the street ends.
    left: u32,
    /// The street it leaves from (not counted as too close).
    parent: u32,
}

fn organic(site: &Site, g: &mut Graph, ids: &mut Streets, rng: &mut SimRng) {
    let c = site.c;
    let d = &c.districts;
    let ds = site.districts;
    let min_angle = c.streets.min_angle_deg.to_radians();
    if !ds.kinds.contains(&DistrictKind::Organic) {
        return;
    }
    let organic_at = |p: DVec2| site.in_city(p) && ds.kinds[ds.at(p)] == DistrictKind::Organic;
    // Seeds: points every `organic_seed` m or so along the streets laid out so far.
    let mut queue = std::collections::VecDeque::new();
    // (Edges are short pieces between crossings: the distance carries over from one to the next.)
    let mut next = rng.range(0.0, d.organic_seed);
    for e in g.alive_edges().collect::<Vec<_>>() {
        let (a, b) = (g.p(g.edges[e as usize].a), g.p(g.edges[e as usize].b));
        let len = a.distance(b);
        while next < len {
            let p = a.lerp(b, next / len);
            let side = if rng.chance(0.5) { 1.0 } else { -1.0 };
            let dir = (b - a).normalize().perp() * side;
            if next > 5.0 && next < len - 5.0 && organic_at(p + dir * 20.0) && organic_at(p + dir) {
                queue.push_back((p, dir));
            }
            next += d.organic_seed * rng.range(0.7, 1.3);
        }
        next -= len;
    }
    let mut starts = Vec::new();
    for (p, dir) in queue.drain(..) {
        // The seed edge may have been split since: find the edge there now.
        let Some((edge, q, _)) = g.edge_near(p, 0.5, &[]) else { continue };
        // Keep seeds apart from existing streets leaving the same edge.
        let parent = g.edges[edge as usize].street;
        if g.edge_near_where(q + dir * 15.0, d.organic_spacing * 0.6, &|e| e.street != parent).is_some() {
            continue;
        }
        let node = g.split(edge, q);
        let street = ids.next();
        let left = rng.range(f64::from(d.organic_length[0]), f64::from(d.organic_length[1]) + 1.0) as u32;
        starts.push(Grow { node, dir, class: RoadClass::Collector, street, left, parent });
    }
    let mut queue: std::collections::VecDeque<Grow> = starts.into();
    let mut budget = 20_000usize;
    while let Some(item) = queue.pop_front() {
        budget = budget.saturating_sub(1);
        if budget == 0 {
            break;
        }
        let p = g.p(item.node);
        let len = rng.range(d.organic_step[0], d.organic_step[1]);
        let dir = rotate(item.dir, d.organic_turn * rng.normal()).normalize();
        let q = p + dir * len;
        // Leaves its district or the city, or crosses water: a dead end.
        if !organic_at(q) || !site.ok(p, q) {
            continue;
        }
        // Meets the graph: join it at the crossing, if at a fair angle.
        if let Some((edge, x)) = g.first_crossing(p, q, &[item.node]) {
            let ed = &g.edges[edge as usize];
            let along = (g.p(ed.b) - g.p(ed.a)).normalize();
            let a = angle_between(dir, along);
            if a >= min_angle && a <= PI - min_angle && x.distance(p) >= c.streets.min_length {
                g.insert(End::Node(item.node), End::At(x), item.class, item.street);
            }
            continue;
        }
        // A node close ahead: join it.
        if let Some(t) = g.node_near(q, d.organic_snap, false).filter(|&t| t != item.node) {
            let pt = g.p(t);
            let into = (pt - p).normalize_or_zero();
            let fair = g.nodes[t as usize].edges.iter().all(|&f| angle_between(-into, g.direction(f, t)) >= min_angle)
                && g.nodes[item.node as usize]
                    .edges
                    .iter()
                    .all(|&f| angle_between(into, g.direction(f, item.node)) >= min_angle);
            if fair && pt.distance(p) >= c.streets.min_length && !g.crosses(p, pt, &[item.node, t]) {
                g.insert(End::Node(item.node), End::Node(t), item.class, item.street);
            }
            continue;
        }
        // A street close by: join it if it lies ahead, else stop short of it.
        let (own, parent, node) = (item.street, item.parent, item.node);
        let others = |e: &super::graph::Edge| e.street != own && e.street != parent && e.a != node && e.b != node;
        if let Some((edge, x, _)) = g.edge_near_where(q, d.organic_spacing, &others) {
            let into = (x - p).normalize_or_zero();
            let ed = &g.edges[edge as usize];
            let along = (g.p(ed.b) - g.p(ed.a)).normalize();
            let a = angle_between(into, along);
            let ahead = angle_between(dir, into) < 0.25 * PI;
            if ahead
                && a >= min_angle
                && a <= PI - min_angle
                && x.distance(p) >= c.streets.min_length
                && site.ok(p, x)
                && !g.crosses(p, x, &[item.node, ed.a, ed.b])
            {
                g.insert(End::Node(item.node), End::At(x), item.class, item.street);
            }
            continue;
        }
        let n = g.insert(End::Node(item.node), End::At(q), item.class, item.street);
        if n == item.node {
            continue;
        }
        // Branches at right angles.
        for side in [1.0, -1.0] {
            if rng.chance(d.organic_branch) {
                let street = ids.next();
                let left = rng.range(f64::from(d.organic_length[0]), f64::from(d.organic_length[1]) + 1.0) as u32;
                let bdir = rotate(dir, side * 0.5 * PI + 0.1 * rng.normal());
                queue.push_back(Grow {
                    node: n,
                    dir: bdir,
                    class: RoadClass::Local,
                    street,
                    left,
                    parent: item.street,
                });
            }
        }
        if item.left > 1 {
            queue.push_back(Grow { node: n, dir, left: item.left - 1, ..item });
        } else if g.degree(n) == 1 && rng.chance(d.cul_de_sac) {
            g.nodes[n as usize].cul_de_sac = true;
        }
    }
}

// ------------------------------------------------------------------------------ clean-up

fn clean_up(site: &Site, g: &mut Graph) {
    let s = &site.c.streets;
    // With a margin: smoothing the roads turns their ends a little.
    let min_angle = (s.min_angle_deg + 6.0).to_radians();
    let ok = |p: DVec2, q: DVec2| site.ok(p, q);
    for _ in 0..3 {
        g.collapse_short(0.5 * s.min_length);
        g.connect_dead_ends(s.join_reach, s.join_cone_deg.to_radians(), min_angle, &ok);
        g.collapse_short(s.min_length);
        g.collapse_close_junctions(s.junction_gap);
        g.fix_angles(min_angle);
        g.limit_degree(5);
        g.prune_spurs(s.spur);
        g.trim_crowded_ends(s.dead_end_clearance);
    }
    g.keep_main_component();
}

// ------------------------------------------------------------------------------ roundabouts

fn roundabouts(site: &Site, g: &mut Graph, ids: &mut Streets, rng: &mut SimRng) -> Vec<Ring> {
    let r = &site.c.streets.roundabouts;
    let mut rings: Vec<Ring> = Vec::new();
    if r.share <= 0.0 || r.max == 0 {
        return rings;
    }
    for n in 0..g.nodes.len() as NodeId {
        if rings.len() >= r.max as usize {
            break;
        }
        let node = &g.nodes[n as usize];
        let degree = node.edges.len();
        if !(3..=5).contains(&degree) || node.fixed || !site.in_city(node.p) {
            continue;
        }
        let centre = node.p;
        let major =
            node.edges.iter().filter(|&&e| rank(g.edges[e as usize].class) >= rank(RoadClass::Collector)).count();
        if major == 0 || rings.iter().any(|ring| ring.centre.distance(centre) < r.spacing) {
            continue;
        }
        let radius = rng.range(r.radius[0], r.radius[1]);
        let pick = rng.chance(r.share);
        // Nothing but the arms within the ring and a margin around it.
        if g.edge_near(centre, radius + 6.0, &[n]).is_some() {
            continue;
        }
        // Incident edges by angle, each long enough, with room between them on the ring.
        let mut arms: Vec<(f64, u32)> = node
            .edges
            .iter()
            .map(|&e| {
                let d = g.direction(e, n);
                (libm::atan2(d.y, d.x), e)
            })
            .collect();
        arms.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let long = arms.iter().all(|&(_, e)| g.length(e) >= radius + site.c.streets.min_length + 5.0);
        let spread = (0..arms.len()).all(|i| {
            let (a, b) = (arms[i].0, arms[(i + 1) % arms.len()].0);
            (b - a).rem_euclid(TAU) >= r.min_gap_deg.to_radians()
        });
        if !pick || !long || !spread {
            continue;
        }
        let street = ids.next();
        let entries: Vec<NodeId> = arms
            .iter()
            .map(|&(_, e)| {
                let q = centre + g.direction(e, n) * radius;
                g.split(e, q)
            })
            .collect();
        // Drop the arms' stubs inside the ring, then join the entries around it.
        for e in g.nodes[n as usize].edges.clone() {
            g.remove_edge(e);
        }
        for i in 0..entries.len() {
            let (a, b) = (entries[i], entries[(i + 1) % entries.len()]);
            let (a0, a1) = (arms[i].0, arms[(i + 1) % arms.len()].0);
            let sweep = (a1 - a0).rem_euclid(TAU);
            let steps = ((sweep * radius / r.arc_step).ceil() as usize).max(2);
            let mut prev = a;
            for k in 1..steps {
                let q = centre + unit(a0 + sweep * k as f64 / steps as f64) * radius;
                let m = g.add_node(q);
                g.connect(prev, m, RoadClass::Collector, street);
                prev = m;
            }
            g.connect(prev, b, RoadClass::Collector, street);
        }
        rings.push(Ring { street, centre, radius });
    }
    rings
}
