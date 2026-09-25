//! Interactions between agents: penalty contacts between their sphere colliders, ray casts
//! that see other agents, and neighbour queries (nearest agents, distance between colliders).

use autonomousim_core::contact::PenaltyParams;
use autonomousim_core::geometry::{HitKind, HitMask, Ray, RayHit};
use autonomousim_core::material::MaterialId;
use autonomousim_sensors::RayScene;
use autonomousim_world::StaticWorld;
use glam::DVec3;
use smallvec::SmallVec;

/// A collider sphere in world coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sphere {
    pub center: DVec3,
    pub radius: f64,
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
    /// Normal stiffness and damping of the agent's colliders.
    pub stiffness: f64,
    pub damping: f64,
}

impl AgentShape {
    pub fn set_contact(&mut self, p: &PenaltyParams) {
        self.stiffness = p.stiffness;
        self.damping = p.damping;
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
    /// Touching another agent ([`Events::CRASH_AGENT`](crate::Events::CRASH_AGENT)).
    pub touched: bool,
    /// `(force, point)` in world coordinates, in application order.
    pub forces: SmallVec<[(DVec3, DVec3); 2]>,
}

/// Frictionless penalty contacts between the colliders of different agents. Pairs come from
/// a sweep along x over the bounding spheres, in a fixed order, so the force sums are
/// deterministic.
pub(crate) fn agent_contacts(shapes: &[AgentShape], order: &mut Vec<(f64, u32)>, out: &mut [AgentContacts]) {
    for c in out.iter_mut() {
        c.touched = false;
        c.forces.clear();
    }
    order.clear();
    order.extend(shapes.iter().enumerate().filter(|(_, s)| s.active).map(|(i, s)| (s.center.x - s.radius, i as u32)));
    if order.len() < 2 {
        return;
    }
    order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    for a in 0..order.len() {
        let i = order[a].1 as usize;
        let si = &shapes[i];
        let x_max = si.center.x + si.radius;
        for &(x_min, j) in &order[a + 1..] {
            if x_min > x_max {
                break;
            }
            let j = j as usize;
            let sj = &shapes[j];
            let reach = si.radius + sj.radius;
            if si.center.distance_squared(sj.center) > reach * reach {
                continue;
            }
            let (k, c) = (series(si.stiffness, sj.stiffness), series(si.damping, sj.damping));
            let mut touched = false;
            for ci in &si.spheres {
                for cj in &sj.spheres {
                    let d = ci.center - cj.center;
                    let dist = d.length();
                    let depth = ci.radius + cj.radius - dist;
                    if depth <= 0.0 {
                        continue;
                    }
                    touched = true;
                    let n = if dist > 1e-12 { d / dist } else { DVec3::Z };
                    let point = cj.center + n * (cj.radius - 0.5 * depth);
                    let vn = (si.point_velocity(point) - sj.point_velocity(point)).dot(n);
                    let f = (k * depth - c * vn).max(0.0);
                    if f > 0.0 {
                        out[i].forces.push((n * f, point));
                        out[j].forces.push((-n * f, point));
                    }
                }
            }
            if touched {
                out[i].touched = true;
                out[j].touched = true;
            }
        }
    }
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
            spheres: [DVec3::X, DVec3::NEG_X]
                .iter()
                .map(|d| Sphere { center: center + d * 0.3, radius: 0.2 })
                .collect(),
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
