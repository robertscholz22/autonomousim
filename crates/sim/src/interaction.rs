//! Interactions between agents: penalty contacts between their sphere colliders, ray casts
//! that see other agents, and neighbour queries (nearest agents, distance between colliders).
//!
//! Agent contacts use the static contacts' model (`autonomousim_core::contact`): a
//! spring–damper normal force and a bristle spring for friction per touching sphere pair,
//! capped at `μ·F_n` with `μ` = [`AGENT_FRICTION`] times both spheres' friction factors. The
//! two agents' springs act in series. A ground vehicle's wheels join its shape as spheres at
//! the wheel centres with the tyre radius (they push and carry like rigid wheels; the force
//! goes to the chassis).

use autonomousim_core::contact::PenaltyParams;
use autonomousim_core::geometry::{HitKind, HitMask, Ray, RayHit};
use autonomousim_core::material::MaterialId;
use autonomousim_sensors::RayScene;
use autonomousim_world::StaticWorld;
use glam::DVec3;
use smallvec::SmallVec;

/// Friction coefficient between agents (before the colliders' friction factors).
pub const AGENT_FRICTION: f64 = 0.5;

/// A collider sphere in world coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub center: DVec3,
    pub radius: f64,
    /// Multiplies [`AGENT_FRICTION`].
    pub friction: f64,
    /// Landing gear or wheel: slow contacts on it carry the agent instead of crashing it.
    pub gear: bool,
}

impl Sphere {
    /// A sphere with full friction that is not gear.
    pub fn new(center: DVec3, radius: f64) -> Self {
        Self { center, radius, friction: 1.0, gear: false }
    }
}

/// World-space collision shape of an agent at the start of a physics step.
#[derive(Clone, Debug, Default)]
pub struct AgentShape {
    /// False for disabled agents: they neither collide nor appear in sensors.
    pub active: bool,
    pub id: u32,
    /// Centre of mass and a radius about it containing all spheres.
    pub center: DVec3,
    pub radius: f64,
    pub velocity: DVec3,
    /// Angular velocity (world frame).
    pub ang_vel: DVec3,
    pub spheres: SmallVec<[Sphere; 12]>,
    /// Normal and bristle stiffness and damping of the agent's colliders.
    pub stiffness: f64,
    pub damping: f64,
    pub tangential_stiffness: f64,
    pub tangential_damping: f64,
}

impl AgentShape {
    pub fn set_contact(&mut self, p: &PenaltyParams) {
        self.stiffness = p.stiffness;
        self.damping = p.damping;
        self.tangential_stiffness = p.tangential_stiffness;
        self.tangential_damping = p.tangential_damping;
    }

    /// Velocity of a world point moving with the agent.
    #[inline]
    fn point_velocity(&self, p: DVec3) -> DVec3 {
        self.velocity + self.ang_vel.cross(p - self.center)
    }
}

/// Springs and dampers of two touching colliders act in series.
#[inline]
fn series(a: f64, b: f64) -> f64 {
    if a + b > 0.0 { a * b / (a + b) } else { 0.0 }
}

/// Contact forces on one agent from the others during a tick.
#[derive(Clone, Debug, Default)]
pub struct AgentContacts {
    /// Hit another agent ([`Events::CRASH_AGENT`](crate::Events::CRASH_AGENT)): a contact
    /// without gear on either side, or one approaching faster than the crash speed.
    pub crashed: bool,
    /// Resting on or pushing against another agent with gear or wheels
    /// ([`Events::GROUND_CONTACT`](crate::Events::GROUND_CONTACT)).
    pub supported: bool,
    /// `(force, point)` in world coordinates, in application order.
    pub forces: SmallVec<[(DVec3, DVec3); 2]>,
}

/// Friction state of one touching sphere pair; `a < b` are agent indices.
#[derive(Clone, Copy, Debug)]
struct PairBristle {
    a: u32,
    sa: u16,
    b: u32,
    sb: u16,
    /// Tangential spring deflection of `a`'s sphere relative to `b`'s (world frame).
    deflection: DVec3,
    seen: bool,
}

/// Persistent state of the agent contacts of one world: the sweep order and the friction
/// springs of touching sphere pairs.
#[derive(Clone, Debug, Default)]
pub(crate) struct AgentContactState {
    order: Vec<(f64, u32)>,
    bristles: Vec<PairBristle>,
}

impl AgentContactState {
    pub(crate) fn clear(&mut self) {
        self.bristles.clear();
    }

    fn bristle(&mut self, a: u32, sa: u16, b: u32, sb: u16) -> &mut PairBristle {
        let i = match self.bristles.iter().position(|p| p.a == a && p.sa == sa && p.b == b && p.sb == sb) {
            Some(i) => i,
            None => {
                self.bristles.push(PairBristle { a, sa, b, sb, deflection: DVec3::ZERO, seen: false });
                self.bristles.len() - 1
            }
        };
        let p = &mut self.bristles[i];
        p.seen = true;
        p
    }
}

/// Penalty contacts with friction between the colliders of different agents over one tick of
/// `dt`. Pairs come from a sweep along x over the bounding spheres, in a fixed order, so the
/// force sums are deterministic.
pub(crate) fn agent_contacts(
    shapes: &[AgentShape],
    dt: f64,
    crash_speed: f64,
    state: &mut AgentContactState,
    out: &mut [AgentContacts],
) {
    for c in out.iter_mut() {
        c.crashed = false;
        c.supported = false;
        c.forces.clear();
    }
    for p in &mut state.bristles {
        p.seen = false;
    }
    let order = &mut state.order;
    order.clear();
    order.extend(shapes.iter().enumerate().filter(|(_, s)| s.active).map(|(i, s)| (s.center.x - s.radius, i as u32)));
    order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let order = std::mem::take(order);
    for (n, &(_, i)) in order.iter().enumerate() {
        let x_max = shapes[i as usize].center.x + shapes[i as usize].radius;
        for &(x_min, j) in &order[n + 1..] {
            if x_min > x_max {
                break;
            }
            let (a, b) = (i.min(j), i.max(j));
            let (sa, sb) = (&shapes[a as usize], &shapes[b as usize]);
            let reach = sa.radius + sb.radius;
            if sa.center.distance_squared(sb.center) > reach * reach {
                continue;
            }
            let k = series(sa.stiffness, sb.stiffness);
            let c = series(sa.damping, sb.damping);
            let kt = series(sa.tangential_stiffness, sb.tangential_stiffness);
            let ct = series(sa.tangential_damping, sb.tangential_damping);
            let (mut crashed, mut supported) = (false, false);
            for (ia, ca) in sa.spheres.iter().enumerate() {
                for (ib, cb) in sb.spheres.iter().enumerate() {
                    let d = ca.center - cb.center;
                    let dist = d.length();
                    let depth = ca.radius + cb.radius - dist;
                    if depth <= 0.0 {
                        continue;
                    }
                    // Normal towards a's sphere; the force acts midway into the overlap.
                    let n = if dist > 1e-12 { d / dist } else { DVec3::Z };
                    let point = cb.center + n * (cb.radius - 0.5 * depth);
                    let v = sa.point_velocity(point) - sb.point_velocity(point);
                    let vn = v.dot(n);
                    if (ca.gear || cb.gear) && vn >= -crash_speed {
                        supported = true;
                    } else {
                        crashed = true;
                    }
                    let fn_ = (k * depth - c * vn).max(0.0);

                    let br = state.bristle(a, ia as u16, b, ib as u16);
                    let vt = v - n * vn;
                    let s = br.deflection - n * n.dot(br.deflection) + vt * dt;
                    let trial = -kt * s - ct * vt;
                    let limit = AGENT_FRICTION * ca.friction * cb.friction * fn_;
                    let ft = if trial.length_squared() > limit * limit {
                        let ft = trial.normalize_or_zero() * limit;
                        br.deflection = if kt > 0.0 { -ft / kt } else { DVec3::ZERO };
                        ft
                    } else {
                        br.deflection = s;
                        trial
                    };

                    let f = n * fn_ + ft;
                    if f != DVec3::ZERO {
                        out[a as usize].forces.push((f, point));
                        out[b as usize].forces.push((-f, point));
                    }
                }
            }
            for x in [a, b] {
                out[x as usize].crashed |= crashed;
                out[x as usize].supported |= supported;
            }
        }
    }
    state.order = order;
    state.bristles.retain(|p| p.seen);
}

/// Distance between the colliders of two agents (surface to surface; 0 when they touch),
/// or `max` if it is at least `max`.
pub fn surface_distance(a: &AgentShape, b: &AgentShape, max: f64) -> f64 {
    let mut best = max;
    if a.center.distance(b.center) - a.radius - b.radius >= best {
        return best;
    }
    for ca in &a.spheres {
        for cb in &b.spheres {
            best = best.min(ca.center.distance(cb.center) - ca.radius - cb.radius);
        }
    }
    best.clamp(0.0, max)
}

/// Distance from agent `me`'s colliders to the nearest other active agent's, up to `max`.
pub fn agent_clearance(shapes: &[AgentShape], me: usize, max: f64) -> f64 {
    let mine = &shapes[me];
    let mut best = max;
    for (k, s) in shapes.iter().enumerate() {
        if k != me && s.active {
            best = best.min(surface_distance(mine, s, best));
        }
    }
    best
}

/// The `count` nearest other active agents whose centres are within `range` of agent `me`'s,
/// as `(distance, index)` sorted by distance, ties broken by index.
pub fn nearest_agents(shapes: &[AgentShape], me: usize, range: f64, count: usize, out: &mut Vec<(f64, usize)>) {
    out.clear();
    if count == 0 {
        return;
    }
    let c = shapes[me].center;
    for (k, s) in shapes.iter().enumerate() {
        if k == me || !s.active {
            continue;
        }
        let d = c.distance(s.center);
        if d > range || (out.len() == count && d >= out[count - 1].0) {
            continue;
        }
        // Insert keeping (distance, index) order; indices rise, so equal distances stay in order.
        let at = out.partition_point(|&(e, _)| e <= d);
        if out.len() == count {
            out.pop();
        }
        out.insert(at, (d, k));
    }
}

/// Agent count from which neighbour queries use an [`AgentGrid`] instead of scanning all
/// agents.
pub const GRID_MIN_AGENTS: usize = 32;

/// Uniform grid over the xy plane of the active agents' centres, for neighbour queries
/// ([`clearance`](Self::clearance), [`nearest`](Self::nearest)) in O(n) instead of O(n²)
/// per world. Cells are stored compressed (per cell a range of agent indices, in index
/// order). Results equal the brute-force [`agent_clearance`] and [`nearest_agents`] bit for
/// bit: the grid only skips agents that cannot change them. Below [`GRID_MIN_AGENTS`]
/// active agents it stays empty and the queries scan all agents.
#[derive(Clone, Debug, Default)]
pub struct AgentGrid {
    on: bool,
    cell: f64,
    lo: glam::DVec2,
    nx: usize,
    ny: usize,
    /// `start[c]..start[c + 1]` indexes `items` for cell `c = y·nx + x`.
    start: Vec<u32>,
    items: Vec<u32>,
    max_radius: f64,
}

impl AgentGrid {
    /// Rebuild from the shapes (active agents only).
    pub fn build(&mut self, shapes: &[AgentShape]) {
        let active = shapes.iter().filter(|s| s.active).count();
        self.on = active >= GRID_MIN_AGENTS;
        if !self.on {
            return;
        }
        let (mut lo, mut hi) = (glam::DVec2::splat(f64::INFINITY), glam::DVec2::splat(f64::NEG_INFINITY));
        let mut max_radius: f64 = 0.0;
        for s in shapes.iter().filter(|s| s.active) {
            lo = lo.min(s.center.truncate());
            hi = hi.max(s.center.truncate());
            max_radius = max_radius.max(s.radius);
        }
        // About two agents per cell on average, at least 1 m, at most 64 × 64 cells.
        let extent = (hi - lo).max(glam::DVec2::splat(1e-3));
        let mut cell = (extent.x * extent.y * 2.0 / active as f64).sqrt().max(1.0);
        cell = cell.max(extent.x.max(extent.y) / 64.0);
        self.cell = cell;
        self.lo = lo;
        self.nx = (extent.x / cell) as usize + 1;
        self.ny = (extent.y / cell) as usize + 1;
        self.max_radius = max_radius;
        let cells = self.nx * self.ny;
        self.start.clear();
        self.start.resize(cells + 1, 0);
        for s in shapes.iter().filter(|s| s.active) {
            let c = self.cell_index(s.center);
            self.start[c + 1] += 1;
        }
        for c in 0..cells {
            self.start[c + 1] += self.start[c];
        }
        self.items.clear();
        self.items.resize(active, 0);
        let mut fill: Vec<u32> = self.start[..cells].to_vec();
        for (i, s) in shapes.iter().enumerate().filter(|(_, s)| s.active) {
            let c = self.cell_index(s.center);
            self.items[fill[c] as usize] = i as u32;
            fill[c] += 1;
        }
    }

    #[inline]
    fn cell_xy(&self, p: DVec3) -> (i64, i64) {
        let x = ((p.x - self.lo.x) / self.cell).floor() as i64;
        let y = ((p.y - self.lo.y) / self.cell).floor() as i64;
        (x, y)
    }

    #[inline]
    fn cell_index(&self, p: DVec3) -> usize {
        let (x, y) = self.cell_xy(p);
        let x = x.clamp(0, self.nx as i64 - 1) as usize;
        let y = y.clamp(0, self.ny as i64 - 1) as usize;
        y * self.nx + x
    }

    /// Agents in the cells at Chebyshev distance `r` from cell `(cx, cy)`; false once the
    /// ring lies entirely outside the grid.
    fn ring(&self, cx: i64, cy: i64, r: i64, mut visit: impl FnMut(usize)) -> bool {
        let (nx, ny) = (self.nx as i64, self.ny as i64);
        if cx - r < 0 && cx + r >= nx && cy - r < 0 && cy + r >= ny {
            return false;
        }
        let mut cell = |x: i64, y: i64| {
            if (0..nx).contains(&x) && (0..ny).contains(&y) {
                let c = (y * nx + x) as usize;
                for &i in &self.items[self.start[c] as usize..self.start[c + 1] as usize] {
                    visit(i as usize);
                }
            }
        };
        if r == 0 {
            cell(cx, cy);
            return true;
        }
        for x in cx - r..=cx + r {
            cell(x, cy - r);
            cell(x, cy + r);
        }
        for y in cy - r + 1..cy + r {
            cell(cx - r, y);
            cell(cx + r, y);
        }
        true
    }

    /// Lower bound of the horizontal distance from a point in cell `(cx, cy)` to any point in
    /// a cell at Chebyshev distance `r`.
    #[inline]
    fn ring_gap(&self, r: i64) -> f64 {
        (r - 1).max(0) as f64 * self.cell
    }

    /// [`agent_clearance`] of agent `me`.
    pub fn clearance(&self, shapes: &[AgentShape], me: usize, max: f64) -> f64 {
        if !self.on {
            return agent_clearance(shapes, me, max);
        }
        let mine = &shapes[me];
        let (cx, cy) = self.cell_xy(mine.center);
        let mut best = max;
        for r in 0.. {
            if self.ring_gap(r) - mine.radius - self.max_radius >= best {
                break;
            }
            let inside = self.ring(cx, cy, r, |k| {
                if k != me {
                    best = best.min(surface_distance(mine, &shapes[k], best));
                }
            });
            if !inside {
                break;
            }
        }
        best
    }

    /// [`nearest_agents`] of agent `me`.
    pub fn nearest(&self, shapes: &[AgentShape], me: usize, range: f64, count: usize, out: &mut Vec<(f64, usize)>) {
        if !self.on {
            return nearest_agents(shapes, me, range, count, out);
        }
        out.clear();
        if count == 0 {
            return;
        }
        let c = shapes[me].center;
        let (cx, cy) = self.cell_xy(c);
        for r in 0.. {
            // Every agent not yet visited is at least this far away.
            let gap = self.ring_gap(r);
            if gap > range || (out.len() >= count && out[count - 1].0 < gap) {
                break;
            }
            let inside = self.ring(cx, cy, r, |k| {
                let d = c.distance(shapes[k].center);
                if k != me && d <= range {
                    out.push((d, k));
                }
            });
            out.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            out.truncate(count);
            if !inside {
                break;
            }
        }
    }
}

/// Ray targets for one agent's sensors: the static world plus all other active agents.
pub struct SceneRays<'a> {
    pub world: &'a StaticWorld,
    pub agents: &'a [AgentShape],
    /// Index of the agent that is sensing (never hits itself).
    pub exclude: usize,
}

impl RayScene for SceneRays<'_> {
    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        let hit = self.world.raycast(ray, max_toi, mask);
        if !mask.intersects(HitMask::AGENTS) || self.agents.len() < 2 {
            return hit;
        }
        let mut best = hit.map_or(max_toi, |h| h.toi);
        let mut found: Option<(f64, usize, Sphere)> = None;
        for (k, s) in self.agents.iter().enumerate() {
            if k == self.exclude || !s.active || ray_sphere(ray, s.center, s.radius).is_none_or(|t| t > best) {
                continue;
            }
            for sp in &s.spheres {
                if let Some(t) = ray_sphere(ray, sp.center, sp.radius).filter(|t| *t <= best) {
                    best = t;
                    found = Some((t, k, *sp));
                }
            }
        }
        match found {
            Some((toi, k, sp)) => {
                let point = ray.at(toi);
                let normal = (point - sp.center).try_normalize().unwrap_or(-ray.dir);
                Some(RayHit {
                    toi,
                    point,
                    normal,
                    material: MaterialId::default(),
                    kind: HitKind::Agent(self.agents[k].id),
                })
            }
            None => hit,
        }
    }
}

/// Entry distance of a ray (unit direction) into a sphere; 0 if it starts inside.
#[inline]
fn ray_sphere(ray: &Ray, center: DVec3, radius: f64) -> Option<f64> {
    let oc = ray.origin - center;
    let b = oc.dot(ray.dir);
    let c = oc.length_squared() - radius * radius;
    if c <= 0.0 {
        return Some(0.0);
    }
    if b > 0.0 {
        return None;
    }
    let disc = b * b - c;
    (disc >= 0.0).then(|| -b - disc.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_world::testworlds;

    fn shape(id: u32, center: DVec3) -> AgentShape {
        AgentShape {
            active: true,
            id,
            center,
            radius: 0.5,
            spheres: [DVec3::X, DVec3::NEG_X].iter().map(|d| Sphere::new(center + d * 0.3, 0.2)).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn rays_see_other_agents_but_not_themselves() {
        let world = testworlds::flat(100.0);
        let shapes = vec![shape(7, DVec3::new(0.0, 0.0, 5.0)), shape(9, DVec3::new(10.0, 0.0, 5.0))];
        let scene = SceneRays { world: &world, agents: &shapes, exclude: 0 };
        let ray = Ray::new(DVec3::new(0.0, 0.0, 5.0), DVec3::X);
        let hit = scene.raycast(&ray, 50.0, HitMask::ALL).unwrap();
        assert_eq!(hit.kind, HitKind::Agent(9));
        assert!((hit.toi - (10.0 - 0.3 - 0.2)).abs() < 1e-12);
        assert!((hit.normal - DVec3::NEG_X).length() < 1e-12);
        // Masked out, out of range, or disabled: no agent hit.
        assert!(scene.raycast(&ray, 50.0, HitMask::TERRAIN).is_none());
        assert!(scene.raycast(&ray, 9.0, HitMask::ALL).is_none());
        let mut off = shapes.clone();
        off[1].active = false;
        assert!(SceneRays { world: &world, agents: &off, exclude: 0 }.raycast(&ray, 50.0, HitMask::ALL).is_none());
        // Terrain in front of the agent wins.
        let down = Ray::new(DVec3::new(10.0, 0.0, 20.0), -DVec3::Z);
        let ground = SceneRays { world: &world, agents: &shapes, exclude: 0 }.raycast(&down, 50.0, HitMask::TERRAIN);
        assert_eq!(ground.unwrap().kind, HitKind::Terrain);
    }

    /// Brute-force references on a random swarm: sorted neighbours and surface distances.
    #[test]
    fn grid_queries_equal_the_scans() {
        let mut rng = autonomousim_core::rng::Seed::from_u64(9).rng();
        // A dense block, a sparse spread, a line along y and duplicates at one point.
        let layouts: [(usize, f64, f64); 4] = [(256, 12.0, 12.0), (100, 150.0, 150.0), (64, 0.0, 80.0), (40, 0.0, 0.0)];
        let mut grid = AgentGrid::default();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for (n, wx, wy) in layouts {
            let mut shapes: Vec<AgentShape> = (0..n as u32)
                .map(|i| {
                    let p = DVec3::new(rng.range(-wx, wx + 1e-9), rng.range(-wy, wy + 1e-9), rng.range(0.0, 3.0));
                    shape(i, p)
                })
                .collect();
            for k in (0..n).step_by(5) {
                shapes[k].active = false;
            }
            // An inactive agent far outside the grid still gets answers.
            shapes[0] = AgentShape { active: false, ..shape(0, DVec3::new(500.0, -300.0, 2.0)) };
            grid.build(&shapes);
            assert!(grid.on, "{n} agents");
            for me in 0..n {
                assert_eq!(grid.clearance(&shapes, me, 20.0), agent_clearance(&shapes, me, 20.0), "{n}: agent {me}");
                assert_eq!(grid.clearance(&shapes, me, 0.5), agent_clearance(&shapes, me, 0.5), "{n}: agent {me}");
                for (range, count) in [(5.0, 3), (20.0, 8), (1e3, 16), (0.5, 2)] {
                    grid.nearest(&shapes, me, range, count, &mut a);
                    nearest_agents(&shapes, me, range, count, &mut b);
                    assert_eq!(a, b, "{n}: agent {me}, range {range}, count {count}");
                }
            }
        }
        // Few agents: no grid, the scans themselves.
        let few: Vec<AgentShape> = (0..5).map(|i| shape(i, DVec3::new(i as f64, 0.0, 0.0))).collect();
        grid.build(&few);
        assert!(!grid.on);
        assert_eq!(grid.clearance(&few, 0, 20.0), agent_clearance(&few, 0, 20.0));
    }

    #[test]
    fn neighbour_queries_match_brute_force() {
        let mut rng = autonomousim_core::rng::Seed::from_u64(3).rng();
        let mut shapes: Vec<AgentShape> = (0..60)
            .map(|i| {
                let p = DVec3::new(rng.range(-20.0, 20.0), rng.range(-20.0, 20.0), rng.range(0.0, 5.0));
                shape(i, p)
            })
            .collect();
        for k in (0..60).step_by(7) {
            shapes[k].active = false;
        }
        // Two agents at exactly the same distance from agent 0: the lower index comes first.
        shapes[10] = shape(10, shapes[0].center + DVec3::new(3.0, 0.0, 0.0));
        shapes[20] = shape(20, shapes[0].center + DVec3::new(0.0, -3.0, 0.0));
        let mut out = Vec::new();
        for me in 0..60 {
            for (range, count) in [(10.0, 3), (50.0, 5), (2.0, 4), (50.0, 0)] {
                nearest_agents(&shapes, me, range, count, &mut out);
                let mut all: Vec<(f64, usize)> = (0..60)
                    .filter(|&k| k != me && shapes[k].active)
                    .map(|k| (shapes[me].center.distance(shapes[k].center), k))
                    .filter(|&(d, _)| d <= range)
                    .collect();
                all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                all.truncate(count);
                assert_eq!(out, all, "agent {me}, range {range}, count {count}");
            }
            let brute = (0..60)
                .filter(|&k| k != me && shapes[k].active)
                .flat_map(|k| {
                    let (a, b) = (&shapes[me], &shapes[k]);
                    a.spheres.iter().flat_map(move |x| {
                        b.spheres.iter().map(move |y| x.center.distance(y.center) - x.radius - y.radius)
                    })
                })
                .fold(20.0f64, f64::min)
                .max(0.0);
            assert!((agent_clearance(&shapes, me, 20.0) - brute).abs() < 1e-12, "agent {me}");
        }
        nearest_agents(&shapes, 0, 3.0, 60, &mut out);
        let at = |i: usize| out.iter().position(|e| e.1 == i).unwrap();
        assert_eq!(out[at(10)].0, out[at(20)].0);
        assert_eq!(at(20), at(10) + 1);
        // Overlapping colliders: 0. Alone: the limit.
        let pair = [shape(1, DVec3::ZERO), shape(2, DVec3::new(0.2, 0.0, 0.0))];
        assert_eq!(agent_clearance(&pair, 0, 20.0), 0.0);
        assert_eq!(agent_clearance(&pair[..1], 0, 20.0), 20.0);
    }

    #[test]
    fn ray_sphere_cases() {
        let r = Ray::new(DVec3::ZERO, DVec3::X);
        assert_eq!(ray_sphere(&r, DVec3::new(5.0, 0.0, 0.0), 1.0), Some(4.0));
        assert_eq!(ray_sphere(&r, DVec3::new(-5.0, 0.0, 0.0), 1.0), None);
        assert_eq!(ray_sphere(&r, DVec3::new(5.0, 2.0, 0.0), 1.0), None);
        assert_eq!(ray_sphere(&r, DVec3::new(0.5, 0.0, 0.0), 1.0), Some(0.0));
    }
}
