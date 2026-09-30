//! A planar street graph of straight segments, built incrementally: every inserted segment is
//! split where it crosses the graph, and its ends snap to nearby nodes and segments. Clean-up
//! passes then remove what a street network should not have (tiny segments, sharp junction
//! angles, crowded junctions, short dead ends, disconnected pieces).
//!
//! Everything runs sequentially in index order, so the result is deterministic.

use autonomousim_world::RoadClass;
use glam::DVec2;

pub(crate) type NodeId = u32;
pub(crate) type EdgeId = u32;

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub p: DVec2,
    /// Incident edges (alive ones only).
    pub edges: Vec<EdgeId>,
    /// On the map edge: never moved or pruned.
    pub fixed: bool,
    /// A planned dead end (cul-de-sac), kept by the pruning.
    pub cul_de_sac: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Edge {
    pub a: NodeId,
    pub b: NodeId,
    pub class: RoadClass,
    /// The street it belongs to (sections are drawn per street).
    pub street: u32,
    pub alive: bool,
}

/// Rank of a class: higher wins when edges compete.
pub(crate) fn rank(class: RoadClass) -> u8 {
    match class {
        RoadClass::Arterial => 5,
        RoadClass::Paved => 4,
        RoadClass::Collector => 3,
        RoadClass::Local => 2,
        RoadClass::Gravel => 1,
        RoadClass::Track => 0,
    }
}

const CELL: f64 = 32.0;

/// Uniform grid of ids by cell (entries of dead items are skipped by the users).
#[derive(Clone, Debug)]
struct Grid {
    lo: DVec2,
    n: usize,
    cells: Vec<Vec<u32>>,
}

impl Grid {
    fn new(size: f64) -> Self {
        let n = (size / CELL).ceil() as usize + 2;
        Self { lo: DVec2::splat(-0.5 * size - CELL), n, cells: vec![Vec::new(); n * n] }
    }

    fn index(&self, v: f64, lo: f64) -> usize {
        (((v - lo) / CELL).floor().max(0.0) as usize).min(self.n - 1)
    }

    fn range(&self, lo: DVec2, hi: DVec2) -> (usize, usize, usize, usize) {
        (
            self.index(lo.x, self.lo.x),
            self.index(hi.x, self.lo.x),
            self.index(lo.y, self.lo.y),
            self.index(hi.y, self.lo.y),
        )
    }

    fn insert(&mut self, id: u32, lo: DVec2, hi: DVec2) {
        let (x0, x1, y0, y1) = self.range(lo, hi);
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.cells[y * self.n + x].push(id);
            }
        }
    }

    /// Ids in the cells meeting the box, sorted and deduplicated.
    fn query(&self, lo: DVec2, hi: DVec2) -> Vec<u32> {
        let (x0, x1, y0, y1) = self.range(lo, hi);
        let mut out = Vec::new();
        for y in y0..=y1 {
            for x in x0..=x1 {
                out.extend_from_slice(&self.cells[y * self.n + x]);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Where a segment ends: an existing node or a free position.
#[derive(Clone, Copy, Debug)]
pub(crate) enum End {
    Node(NodeId),
    At(DVec2),
}

/// A crossing of a new segment with an existing edge.
#[derive(Clone, Copy, Debug)]
struct Hit {
    t: f64,
    edge: EdgeId,
    point: DVec2,
}

#[derive(Clone, Debug)]
pub(crate) struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    node_grid: Grid,
    edge_grid: Grid,
    /// Snap radius (m) for segment ends and crossings.
    pub snap: f64,
}

impl Graph {
    pub fn new(size: f64, snap: f64) -> Self {
        Self { nodes: Vec::new(), edges: Vec::new(), node_grid: Grid::new(size), edge_grid: Grid::new(size), snap }
    }

    pub fn p(&self, n: NodeId) -> DVec2 {
        self.nodes[n as usize].p
    }

    pub fn degree(&self, n: NodeId) -> usize {
        self.nodes[n as usize].edges.len()
    }

    pub fn length(&self, e: EdgeId) -> f64 {
        let e = &self.edges[e as usize];
        self.p(e.a).distance(self.p(e.b))
    }

    /// The other end of `e` from `n`.
    pub fn other(&self, e: EdgeId, n: NodeId) -> NodeId {
        let e = &self.edges[e as usize];
        if e.a == n { e.b } else { e.a }
    }

    /// Unit direction of `e` leaving `n`.
    pub fn direction(&self, e: EdgeId, n: NodeId) -> DVec2 {
        (self.p(self.other(e, n)) - self.p(n)).normalize_or_zero()
    }

    pub fn add_node(&mut self, p: DVec2) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(Node { p, edges: Vec::new(), fixed: false, cul_de_sac: false });
        self.node_grid.insert(id, p, p);
        id
    }

    /// Nearest node with edges (or any node when `any`) within `r` of `p`, lowest id on ties.
    pub fn node_near(&self, p: DVec2, r: f64, any: bool) -> Option<NodeId> {
        let mut best: Option<(f64, NodeId)> = None;
        for id in self.node_grid.query(p - r, p + r) {
            let node = &self.nodes[id as usize];
            if !any && node.edges.is_empty() {
                continue;
            }
            let d = node.p.distance_squared(p);
            if d <= r * r && best.is_none_or(|b| d < b.0) {
                best = Some((d, id));
            }
        }
        best.map(|b| b.1)
    }

    /// Nearest point on an alive edge within `r` of `p`, skipping edges incident to `skip`:
    /// (edge, point, distance).
    pub fn edge_near(&self, p: DVec2, r: f64, skip: &[NodeId]) -> Option<(EdgeId, DVec2, f64)> {
        self.edge_near_where(p, r, &|e| !skip.contains(&e.a) && !skip.contains(&e.b))
    }

    /// As [`edge_near`](Self::edge_near), over the edges for which `keep` holds.
    pub fn edge_near_where(&self, p: DVec2, r: f64, keep: &dyn Fn(&Edge) -> bool) -> Option<(EdgeId, DVec2, f64)> {
        let mut best: Option<(EdgeId, DVec2, f64)> = None;
        for id in self.edge_grid.query(p - r, p + r) {
            let e = &self.edges[id as usize];
            if !e.alive || !keep(e) {
                continue;
            }
            let q = closest_on_segment(self.p(e.a), self.p(e.b), p);
            let d = q.distance(p);
            if d <= r && best.is_none_or(|b| d < b.2) {
                best = Some((id, q, d));
            }
        }
        best
    }

    /// Alive edges crossing the open segment `p–q` (not those incident to `skip`), by `t`.
    fn crossings(&self, p: DVec2, q: DVec2, skip: &[NodeId]) -> Vec<Hit> {
        let mut hits = Vec::new();
        for id in self.edge_grid.query(p.min(q), p.max(q)) {
            let e = &self.edges[id as usize];
            if !e.alive || skip.contains(&e.a) || skip.contains(&e.b) {
                continue;
            }
            if let Some((t, point)) = intersect(p, q, self.p(e.a), self.p(e.b)) {
                hits.push(Hit { t, edge: id, point });
            }
        }
        hits.sort_by(|a, b| a.t.total_cmp(&b.t).then(a.edge.cmp(&b.edge)));
        hits
    }

    /// The first edge crossed by the open segment `p–q` (ignoring edges at `skip`) and where.
    pub fn first_crossing(&self, p: DVec2, q: DVec2, skip: &[NodeId]) -> Option<(EdgeId, DVec2)> {
        self.crossings(p, q, skip).first().map(|h| (h.edge, h.point))
    }

    /// Whether the open segment `p–q` crosses the graph (ignoring edges at `skip`).
    pub fn crosses(&self, p: DVec2, q: DVec2, skip: &[NodeId]) -> bool {
        !self.crossings(p, q, skip).is_empty()
    }

    /// Add an edge `a–b` unless it exists or `a == b`; an existing edge takes the higher class.
    pub fn connect(&mut self, a: NodeId, b: NodeId, class: RoadClass, street: u32) -> Option<EdgeId> {
        if a == b {
            return None;
        }
        if let Some(&e) = self.nodes[a as usize].edges.iter().find(|&&e| self.other(e, a) == b) {
            if rank(class) > rank(self.edges[e as usize].class) {
                self.edges[e as usize].class = class;
                self.edges[e as usize].street = street;
            }
            return None;
        }
        let id = self.edges.len() as EdgeId;
        self.edges.push(Edge { a, b, class, street, alive: true });
        self.nodes[a as usize].edges.push(id);
        self.nodes[b as usize].edges.push(id);
        let (pa, pb) = (self.p(a), self.p(b));
        self.edge_grid.insert(id, pa.min(pb), pa.max(pb));
        Some(id)
    }

    pub fn remove_edge(&mut self, e: EdgeId) {
        let edge = &mut self.edges[e as usize];
        if !edge.alive {
            return;
        }
        edge.alive = false;
        let (a, b) = (edge.a, edge.b);
        self.nodes[a as usize].edges.retain(|&x| x != e);
        self.nodes[b as usize].edges.retain(|&x| x != e);
    }

    /// Split edge `e` at `point` (on it); returns the node there (an end if very close).
    pub fn split(&mut self, e: EdgeId, point: DVec2) -> NodeId {
        let Edge { a, b, class, street, .. } = self.edges[e as usize].clone();
        for n in [a, b] {
            if self.p(n).distance(point) < 0.5 {
                return n;
            }
        }
        let n = self.add_node(point);
        self.remove_edge(e);
        self.connect(a, n, class, street);
        self.connect(n, b, class, street);
        n
    }

    /// A node for a segment end: the given node, a node within the snap radius, a point on an
    /// edge within it (split there), or a new node.
    fn resolve(&mut self, end: End) -> NodeId {
        match end {
            End::Node(n) => n,
            End::At(p) => {
                if let Some(n) = self.node_near(p, self.snap, false) {
                    n
                } else if let Some((e, q, _)) = self.edge_near(p, self.snap, &[]) {
                    self.split(e, q)
                } else {
                    self.add_node(p)
                }
            }
        }
    }

    /// Insert the segment `from–to`, split at every crossing (or bent through a node within
    /// the snap radius of a crossing, where that crosses nothing else); returns the end node.
    pub fn insert(&mut self, from: End, to: End, class: RoadClass, street: u32) -> NodeId {
        let a = self.resolve(from);
        let b = self.resolve(to);
        let mut cur = a;
        // Each step moves `cur` closer to `b`, so this ends.
        while cur != b {
            let (pc, pb) = (self.p(cur), self.p(b));
            let Some(h) = self.crossings(pc, pb, &[cur, b]).first().copied() else {
                self.connect(cur, b, class, street);
                break;
            };
            let near = self.node_near(h.point, self.snap, false).filter(|&n| {
                n != cur
                    && n != b
                    && self.p(n).distance(pb) < pc.distance(pb) - 0.1
                    && !self.crosses(pc, self.p(n), &[cur, n])
            });
            let n = near.unwrap_or_else(|| self.split(h.edge, h.point));
            if n == cur || self.p(n).distance(pb) >= pc.distance(pb) {
                // A crossing right at `cur` (split reused an end): nothing to gain.
                self.connect(cur, b, class, street);
                break;
            }
            self.connect(cur, n, class, street);
            cur = n;
        }
        b
    }

    pub fn alive_edges(&self) -> impl Iterator<Item = EdgeId> + '_ {
        (0..self.edges.len() as EdgeId).filter(|&e| self.edges[e as usize].alive)
    }

    // ------------------------------------------------------------------------ clean-up

    /// Merge the ends of edges shorter than `min` (the lower id into... the node that is
    /// fixed or has the higher degree keeps its place). Returns the number of merges.
    pub fn collapse_short(&mut self, min: f64) -> usize {
        self.collapse(min, false)
    }

    /// Merge junctions (three or more edges) joined by an edge shorter than `min`, as
    /// [`Self::collapse_short`]: closer junctions leave no room for the lanes between them.
    pub fn collapse_close_junctions(&mut self, min: f64) -> usize {
        self.collapse(min, true)
    }

    fn collapse(&mut self, min: f64, junctions: bool) -> usize {
        let mut merges = 0;
        loop {
            let mut changed = false;
            for e in 0..self.edges.len() as EdgeId {
                if !self.edges[e as usize].alive || self.length(e) >= min {
                    continue;
                }
                let (a, b) = (self.edges[e as usize].a, self.edges[e as usize].b);
                if junctions && (self.degree(a) < 3 || self.degree(b) < 3) {
                    continue;
                }
                let (na, nb) = (&self.nodes[a as usize], &self.nodes[b as usize]);
                if na.fixed && nb.fixed {
                    continue;
                }
                let keep_a = na.fixed || (!nb.fixed && na.edges.len() >= nb.edges.len());
                // Not when a moved edge would cross the graph: then the other way round, or
                // (between two junctions) drop the edge instead.
                let crossing = |keep: NodeId, gone: NodeId| {
                    let pk = self.p(keep);
                    self.nodes[gone as usize].edges.iter().any(|&f| {
                        let o = self.other(f, gone);
                        o != keep && self.crosses(pk, self.p(o), &[keep, gone, o])
                    })
                };
                let first = if keep_a { (a, b) } else { (b, a) };
                let second = (first.1, first.0);
                let (keep, gone) = if !crossing(first.0, first.1) {
                    first
                } else if !self.nodes[second.1 as usize].fixed && !crossing(second.0, second.1) {
                    second
                } else {
                    if self.degree(a) >= 3 && self.degree(b) >= 3 {
                        self.remove_edge(e);
                        merges += 1;
                        changed = true;
                    }
                    continue;
                };
                self.merge(keep, gone);
                merges += 1;
                changed = true;
            }
            if !changed {
                return merges;
            }
        }
    }

    /// Move every edge of `gone` to `keep`, dropping loops and duplicates.
    fn merge(&mut self, keep: NodeId, gone: NodeId) {
        let edges = std::mem::take(&mut self.nodes[gone as usize].edges);
        for e in edges {
            let Edge { a, b, class, street, .. } = self.edges[e as usize].clone();
            self.edges[e as usize].alive = false;
            let other = if a == gone { b } else { a };
            self.nodes[other as usize].edges.retain(|&x| x != e);
            self.connect(keep, other, class, street);
        }
        if self.nodes[gone as usize].cul_de_sac {
            self.nodes[keep as usize].cul_de_sac = self.nodes[keep as usize].edges.len() == 1;
        }
    }

    /// Remove, at every node, the weaker of two edges less than `min_angle` apart (lower
    /// class, then longer... the one whose far end has more other edges keeps connectivity).
    pub fn fix_angles(&mut self, min_angle: f64) -> usize {
        let mut removed = 0;
        loop {
            let mut changed = false;
            for n in 0..self.nodes.len() as NodeId {
                if let Some(e) = self.sharp_pair(n, min_angle) {
                    self.remove_edge(e);
                    removed += 1;
                    changed = true;
                }
            }
            if !changed {
                return removed;
            }
        }
    }

    /// The edge to drop at `n` for the sharpest pair under `min_angle`, if any.
    fn sharp_pair(&self, n: NodeId, min_angle: f64) -> Option<EdgeId> {
        let edges = &self.nodes[n as usize].edges;
        let mut worst: Option<(f64, EdgeId, EdgeId)> = None;
        for (i, &e) in edges.iter().enumerate() {
            for &f in &edges[i + 1..] {
                let (u, v) = (self.direction(e, n), self.direction(f, n));
                let angle = libm::atan2(u.perp_dot(v).abs(), u.dot(v));
                if angle < min_angle && worst.is_none_or(|w| angle < w.0) {
                    worst = Some((angle, e, f));
                }
            }
        }
        let (_, e, f) = worst?;
        let key = |e: EdgeId| {
            let edge = &self.edges[e as usize];
            // Keep the higher class, then the edge whose far end would be left hanging.
            (rank(edge.class), self.degree(self.other(e, n)) <= 1, std::cmp::Reverse(e))
        };
        Some(if key(e) < key(f) { e } else { f })
    }

    /// Remove the lowest-class edges of nodes with more than `max` edges.
    pub fn limit_degree(&mut self, max: usize) -> usize {
        let mut removed = 0;
        for n in 0..self.nodes.len() as NodeId {
            while self.degree(n) > max {
                let e = *self.nodes[n as usize]
                    .edges
                    .iter()
                    .min_by_key(|&&e| (rank(self.edges[e as usize].class), std::cmp::Reverse(e)))
                    .expect("edges");
                self.remove_edge(e);
                removed += 1;
            }
        }
        removed
    }

    /// The chain from dead end `n` through degree-2 nodes to the next junction or end:
    /// (edges, length, the node it stops at).
    fn spur(&self, n: NodeId) -> (Vec<EdgeId>, f64, NodeId) {
        let (mut cur, mut prev_edge) = (n, None);
        let (mut edges, mut len) = (Vec::new(), 0.0);
        loop {
            let next = self.nodes[cur as usize].edges.iter().copied().find(|&e| Some(e) != prev_edge);
            let Some(e) = next else { return (edges, len, cur) };
            edges.push(e);
            len += self.length(e);
            cur = self.other(e, cur);
            prev_edge = Some(e);
            if self.degree(cur) != 2 || cur == n {
                return (edges, len, cur);
            }
        }
    }

    /// Remove dead-end chains shorter than `min` (except from cul-de-sacs; also those ending
    /// at the map's edge, whose turning space would lie on the junction).
    pub fn prune_spurs(&mut self, min: f64) -> usize {
        let mut removed = 0;
        loop {
            let mut changed = false;
            for n in 0..self.nodes.len() as NodeId {
                let node = &self.nodes[n as usize];
                if node.edges.len() != 1 || node.cul_de_sac {
                    continue;
                }
                let (edges, len, _) = self.spur(n);
                if len < min {
                    for e in edges {
                        self.remove_edge(e);
                    }
                    removed += 1;
                    changed = true;
                }
            }
            if !changed {
                return removed;
            }
        }
    }

    /// Cut back dead ends (cul-de-sacs and the map's edge included) lying within `clear` m of
    /// another edge, one edge at a time, until they are clear (the turning space of a U-turn
    /// would reach the other street). Returns the edges removed.
    pub fn trim_crowded_ends(&mut self, clear: f64) -> usize {
        let mut removed = 0;
        loop {
            let mut changed = false;
            for n in 0..self.nodes.len() as NodeId {
                let node = &self.nodes[n as usize];
                if node.edges.len() != 1 {
                    continue;
                }
                let e = node.edges[0];
                let other = self.other(e, n);
                if self.edge_near(node.p, clear, &[n, other]).is_some() {
                    self.remove_edge(e);
                    removed += 1;
                    changed = true;
                }
            }
            if !changed {
                return removed;
            }
        }
    }

    /// Join dead ends (not fixed, not cul-de-sacs) to the graph ahead of them: the nearest
    /// point within `reach` m and `cone` rad of the dead end's direction, reached without
    /// crossing anything and meeting the graph at `min_angle` or more. Returns the joins.
    pub fn connect_dead_ends(
        &mut self,
        reach: f64,
        cone: f64,
        min_angle: f64,
        ok: &dyn Fn(DVec2, DVec2) -> bool,
    ) -> usize {
        let mut joined = 0;
        for n in 0..self.nodes.len() as NodeId {
            let node = &self.nodes[n as usize];
            if node.edges.len() != 1 || node.fixed || node.cul_de_sac {
                continue;
            }
            let e = node.edges[0];
            let p = node.p;
            let dir = -self.direction(e, n);
            let Edge { class, street, .. } = self.edges[e as usize];
            let (chain, _, _) = self.spur(n);
            let mut chain_nodes: Vec<NodeId> = vec![n];
            for &c in &chain {
                let ed = &self.edges[c as usize];
                chain_nodes.extend([ed.a, ed.b]);
            }
            // Candidates: nodes and edge points within reach, ahead in the cone.
            let mut best: Option<(f64, End, Option<EdgeId>)> = None;
            for id in self.node_grid.query(p - reach, p + reach) {
                let q = self.nodes[id as usize].p;
                if chain_nodes.contains(&id) || self.nodes[id as usize].edges.is_empty() {
                    continue;
                }
                let d = q.distance(p);
                if d > reach || d < 1e-6 || angle_between(dir, (q - p) / d) > cone {
                    continue;
                }
                if best.as_ref().is_none_or(|b| d < b.0) {
                    best = Some((d, End::Node(id), None));
                }
            }
            for id in self.edge_grid.query(p - reach, p + reach) {
                let ed = &self.edges[id as usize];
                if !ed.alive || chain_nodes.contains(&ed.a) || chain_nodes.contains(&ed.b) {
                    continue;
                }
                let q = closest_on_segment(self.p(ed.a), self.p(ed.b), p);
                let d = q.distance(p);
                if d > reach || d < 1e-6 || angle_between(dir, (q - p) / d) > cone {
                    continue;
                }
                if best.as_ref().is_none_or(|b| d < b.0 - 1e-9) {
                    best = Some((d, End::At(q), Some(id)));
                }
            }
            let Some((_, target, edge)) = best else { continue };
            let q = match target {
                End::Node(id) => self.p(id),
                End::At(q) => q,
            };
            if !ok(p, q) || self.crosses(p, q, &chain_nodes) {
                continue;
            }
            // The join must meet the target at a reasonable angle.
            let into = (q - p).normalize();
            let fine = match (target, edge) {
                (End::Node(id), _) => self.nodes[id as usize]
                    .edges
                    .iter()
                    .all(|&f| angle_between(-into, self.direction(f, id)) >= min_angle),
                (_, Some(f)) => {
                    let ed = &self.edges[f as usize];
                    let along = (self.p(ed.b) - self.p(ed.a)).normalize();
                    let a = angle_between(into, along);
                    a >= min_angle && a <= std::f64::consts::PI - min_angle
                }
                _ => false,
            };
            if !fine {
                continue;
            }
            let m = match (target, edge) {
                (End::Node(id), _) => id,
                (_, Some(f)) => self.split(f, q),
                _ => unreachable!(),
            };
            self.connect(n, m, class, street);
            joined += 1;
        }
        joined
    }

    /// Keep only the connected component with the most length of the highest class present
    /// (the arterials'); returns the number of edges removed.
    pub fn keep_main_component(&mut self) -> usize {
        let n = self.nodes.len();
        let mut comp = vec![u32::MAX; n];
        let mut scores: Vec<(u8, f64)> = Vec::new();
        for start in 0..n {
            if comp[start] != u32::MAX || self.nodes[start].edges.is_empty() {
                continue;
            }
            let c = scores.len() as u32;
            let (mut best_rank, mut len) = (0u8, 0.0);
            let mut stack = vec![start as NodeId];
            comp[start] = c;
            while let Some(u) = stack.pop() {
                for &e in &self.nodes[u as usize].edges {
                    let r = rank(self.edges[e as usize].class);
                    let l = self.length(e);
                    match r.cmp(&best_rank) {
                        std::cmp::Ordering::Greater => (best_rank, len) = (r, l),
                        std::cmp::Ordering::Equal => len += l,
                        std::cmp::Ordering::Less => {}
                    }
                    let v = self.other(e, u);
                    if comp[v as usize] == u32::MAX {
                        comp[v as usize] = c;
                        stack.push(v);
                    }
                }
            }
            scores.push((best_rank, len));
        }
        let Some(main) = (0..scores.len()).max_by(|&i, &j| {
            let (a, b) = (scores[i], scores[j]);
            a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(j.cmp(&i))
        }) else {
            return 0;
        };
        let mut removed = 0;
        for e in 0..self.edges.len() as EdgeId {
            if self.edges[e as usize].alive && comp[self.edges[e as usize].a as usize] != main as u32 {
                self.remove_edge(e);
                removed += 1;
            }
        }
        removed
    }
}

/// Angle between two unit vectors, in [0, π].
pub(crate) fn angle_between(u: DVec2, v: DVec2) -> f64 {
    libm::atan2(u.perp_dot(v).abs(), u.dot(v))
}

pub(crate) fn closest_on_segment(a: DVec2, b: DVec2, p: DVec2) -> DVec2 {
    let d = b - a;
    let l2 = d.length_squared();
    if l2 < 1e-18 {
        return a;
    }
    a + d * ((p - a).dot(d) / l2).clamp(0.0, 1.0)
}

/// Proper crossing of segments `p–q` and `a–b` (not at or near their ends): `(t along p–q,
/// point)`.
fn intersect(p: DVec2, q: DVec2, a: DVec2, b: DVec2) -> Option<(f64, DVec2)> {
    let r = q - p;
    let s = b - a;
    let den = r.perp_dot(s);
    if den.abs() < 1e-12 {
        return None;
    }
    let t = (a - p).perp_dot(s) / den;
    let u = (a - p).perp_dot(r) / den;
    let (et, eu) = (1e-6, 1e-6);
    ((et..=1.0 - et).contains(&t) && (eu..=1.0 - eu).contains(&u)).then(|| (t, p + r * t))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    #[test]
    fn crossings_split_both_segments() {
        let mut g = Graph::new(400.0, 3.0);
        g.insert(End::At(v(-50.0, 0.0)), End::At(v(50.0, 0.0)), RoadClass::Local, 0);
        g.insert(End::At(v(0.0, -50.0)), End::At(v(0.0, 50.0)), RoadClass::Collector, 1);
        assert_eq!(g.alive_edges().count(), 4);
        let centre = g.node_near(v(0.0, 0.0), 0.1, false).unwrap();
        assert_eq!(g.degree(centre), 4);
        // Ends snap to nearby nodes and edges.
        g.insert(End::At(v(2.0, 1.0)), End::At(v(30.0, 40.0)), RoadClass::Local, 2);
        assert_eq!(g.degree(centre), 5);
        let n = g.insert(End::At(v(-30.0, 40.0)), End::At(v(-1.0, 22.0)), RoadClass::Local, 3);
        assert!(g.p(n).x.abs() < 1e-9 && g.degree(n) == 3, "{:?}", g.p(n));
    }

    #[test]
    fn clean_up_passes() {
        let mut g = Graph::new(400.0, 1.0);
        // A cross with a tiny extra segment and a sharp spur.
        g.insert(End::At(v(-60.0, 0.0)), End::At(v(60.0, 0.0)), RoadClass::Arterial, 0);
        g.insert(End::At(v(0.0, -60.0)), End::At(v(0.0, 60.0)), RoadClass::Local, 1);
        let c = g.node_near(v(0.0, 0.0), 0.1, false).unwrap();
        g.insert(End::Node(c), End::At(v(40.0, 8.0)), RoadClass::Local, 2);
        assert_eq!(g.fix_angles(30f64.to_radians()), 1);
        assert_eq!(g.degree(c), 4);
        // A short dead end goes, a long one stays.
        let before = g.alive_edges().count();
        g.insert(End::At(v(-30.0, -60.0)), End::At(v(-30.0, -40.0)), RoadClass::Local, 3);
        g.insert(End::At(v(30.0, 60.0)), End::At(v(30.0, 10.0)), RoadClass::Local, 4);
        assert_eq!(g.prune_spurs(25.0), 1);
        assert_eq!(g.alive_edges().count(), before + 1);
        // The long one reaches the arterial ahead, so nothing is detached.
        assert_eq!(g.connect_dead_ends(30.0, 0.5, 0.5, &|_, _| true), 1);
        let n = g.node_near(v(30.0, 0.0), 0.1, false).unwrap();
        assert_eq!(g.degree(n), 3);
        assert_eq!(g.keep_main_component(), 0);
        // Short edges collapse.
        g.insert(End::At(v(-60.0, 30.0)), End::At(v(2.0, 30.0)), RoadClass::Local, 5);
        assert!(g.collapse_short(3.0) >= 1);
        assert!(g.alive_edges().all(|e| g.length(e) >= 3.0));
    }
}
