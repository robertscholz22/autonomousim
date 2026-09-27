//! Where ground vehicles can drive: a coarse grid over the map whose cells are drivable when
//! the terrain is not too steep, dry, and free of solid obstacles (trunks, large rocks) within
//! the vehicle's half-width. Drivable cells are labelled by 4-connected component, so whether
//! a goal can be reached from a spawn is one lookup; [`DriveGrid::path`] finds a path (A*),
//! optionally preferring firm ground: each metre costs more by the vehicle's motion resistance
//! on the cell's material (rolling resistance, and for tracks the soil's compaction).
//! [`DriveGrid::legs`] plans smoothed paths through a chain of goals (`path` goals).
//!
//! Also here: [`ground_pose`], the pose of a ground vehicle resting on uneven terrain.

use autonomousim_core::material::{Material, MaterialId};
use autonomousim_core::math::Pose;
use autonomousim_core::terrain::Terrain;
use autonomousim_vehicles::ground::WheeledDef;
use autonomousim_world::StaticWorld;
use autonomousim_world::obstacles::ObstacleClass;
use glam::{DMat3, DQuat, DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Height band above the ground (m) in which obstacles block a cell.
const BLOCKING_HEIGHT: f64 = 2.0;

/// What ground vehicles of a group can drive over (`drivable` in a group).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DrivableSpec {
    /// Grid cell (m).
    pub cell: f64,
    /// Steepest drivable slope (degrees).
    pub max_slope_deg: f64,
    /// Steepest slope to start on (degrees): standing on rough, steep ground, unevenly loaded
    /// tyres can slide.
    pub spawn_slope_deg: f64,
    /// Room kept between obstacles and the vehicle's sides (m), for turning and steering errors.
    pub margin: f64,
    /// Solid obstacles lower than this above the ground (m) are driven over.
    pub obstacle_height: f64,
    /// Water shallower than this (m) can be forded.
    pub max_water_depth: f64,
    /// Paths: each metre costs `1 + resistance_cost·f`, with `f` the vehicle's motion
    /// resistance (share of its weight) on the ground there; 0 finds the shortest path.
    #[serde(skip_serializing_if = "is_zero")]
    pub resistance_cost: f64,
}

fn is_zero(x: &f64) -> bool {
    *x == 0.0
}

impl Default for DrivableSpec {
    fn default() -> Self {
        Self {
            cell: 2.0,
            max_slope_deg: 25.0,
            spawn_slope_deg: 15.0,
            margin: 1.0,
            obstacle_height: 0.25,
            max_water_depth: 0.0,
            resistance_cost: 0.0,
        }
    }
}

impl DrivableSpec {
    pub fn validate(&self) -> Result<(), String> {
        let ok = self.cell >= 0.25
            && self.cell.is_finite()
            && self.max_slope_deg > 0.0
            && self.max_slope_deg < 90.0
            && self.spawn_slope_deg > 0.0
            && self.spawn_slope_deg < 90.0
            && self.margin >= 0.0
            && self.obstacle_height >= 0.0
            && self.max_water_depth >= 0.0
            && self.resistance_cost >= 0.0
            && self.resistance_cost.is_finite();
        if ok { Ok(()) } else { Err(format!("invalid drivable spec {self:?}")) }
    }
}

/// Drivable cells of one map for one vehicle size, labelled by connected component.
#[derive(Clone, Debug)]
pub struct DriveGrid {
    origin: DVec2,
    cell: f64,
    nx: usize,
    ny: usize,
    /// 0: blocked; otherwise the component (1, 2, … in scan order).
    label: Vec<u32>,
    /// Cells per component (index = label).
    sizes: Vec<usize>,
    largest: u32,
    /// Path cost per metre of each cell (≥ 1); empty when all are 1.
    cost: Vec<f32>,
}

impl DriveGrid {
    /// Grid over `world` for vehicles `half_width` wide (m) on each side; paths ignore the
    /// ground's resistance.
    pub fn new(world: &StaticWorld, spec: &DrivableSpec, half_width: f64) -> Self {
        Self::with_resistance(world, spec, half_width, |_| 0.0)
    }

    /// Grid whose paths weigh each metre by the vehicle's motion `resistance` on the material
    /// under the cell centre (see [`DrivableSpec::resistance_cost`]).
    pub fn with_resistance(
        world: &StaticWorld,
        spec: &DrivableSpec,
        half_width: f64,
        resistance: impl Fn(&Material) -> f64,
    ) -> Self {
        let (lo, hi) = world.extent();
        let cell = spec.cell;
        let nx = (((hi.x - lo.x) / cell).floor() as usize).max(1);
        let ny = (((hi.y - lo.y) / cell).floor() as usize).max(1);
        let terrain = world.terrain();
        let obstacles = world.obstacles();
        let max_grad = spec.max_slope_deg.to_radians().tan();
        let min_nz = spec.max_slope_deg.to_radians().cos();
        let mut hits = Vec::new();
        let mut open = vec![false; nx * ny];
        for iy in 0..ny {
            for ix in 0..nx {
                let c = lo + DVec2::new(ix as f64 + 0.5, iy as f64 + 0.5) * cell;
                let h = |dx: f64, dy: f64| terrain.height(c.x + dx, c.y + dy);
                let r = 0.5 * cell;
                let (hc, n) = terrain.height_normal(c.x, c.y);
                let gx = (h(r, 0.0) - h(-r, 0.0)) / cell;
                let gy = (h(0.0, r) - h(0.0, -r)) / cell;
                if n.z < min_nz || gx.hypot(gy) > max_grad {
                    continue;
                }
                if terrain.water_level(c.x, c.y).is_some_and(|w| w - hc > spec.max_water_depth) {
                    continue;
                }
                let m = DVec2::splat(r + half_width + spec.margin);
                let (hmin, hmax) = terrain.height_bounds(c - m, c + m);
                hits.clear();
                obstacles.query_aabb(
                    (c - m).extend(hmin + spec.obstacle_height),
                    (c + m).extend(hmax + BLOCKING_HEIGHT),
                    &mut hits,
                );
                if hits.iter().any(|&i| obstacles.obstacles()[i].class == ObstacleClass::Solid) {
                    continue;
                }
                open[iy * nx + ix] = true;
            }
        }
        // Components by flood fill in scan order.
        let mut label = vec![0u32; nx * ny];
        let mut sizes = vec![0];
        let mut stack = Vec::new();
        for start in 0..nx * ny {
            if !open[start] || label[start] != 0 {
                continue;
            }
            let id = sizes.len() as u32;
            let mut size = 0;
            label[start] = id;
            stack.push(start);
            while let Some(i) = stack.pop() {
                size += 1;
                let (x, y) = (i % nx, i / nx);
                let mut visit = |j: usize| {
                    if open[j] && label[j] == 0 {
                        label[j] = id;
                        stack.push(j);
                    }
                };
                if x > 0 {
                    visit(i - 1);
                }
                if x + 1 < nx {
                    visit(i + 1);
                }
                if y > 0 {
                    visit(i - nx);
                }
                if y + 1 < ny {
                    visit(i + nx);
                }
            }
            sizes.push(size);
        }
        let largest = (1..sizes.len()).max_by_key(|&k| (sizes[k], Reverse(k))).unwrap_or(0) as u32;
        let mut cost = Vec::new();
        if spec.resistance_cost > 0.0 {
            let table = world.materials();
            let per_material: Vec<f32> = (0..table.len())
                .map(|m| (1.0 + spec.resistance_cost * resistance(table.get(MaterialId(m as u8))).max(0.0)) as f32)
                .collect();
            cost = (0..nx * ny)
                .map(|i| {
                    let c = lo + DVec2::new((i % nx) as f64 + 0.5, (i / nx) as f64 + 0.5) * cell;
                    per_material[terrain.material(c.x, c.y).0 as usize]
                })
                .collect();
        }
        Self { origin: lo, cell, nx, ny, label, sizes, largest, cost }
    }

    pub fn cell_size(&self) -> f64 {
        self.cell
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    fn index(&self, p: DVec2) -> Option<usize> {
        let u = (p - self.origin) / self.cell;
        if u.x < 0.0 || u.y < 0.0 {
            return None;
        }
        let (x, y) = (u.x as usize, u.y as usize);
        (x < self.nx && y < self.ny).then_some(y * self.nx + x)
    }

    fn center(&self, i: usize) -> DVec2 {
        self.origin + DVec2::new((i % self.nx) as f64 + 0.5, (i / self.nx) as f64 + 0.5) * self.cell
    }

    /// Component of the cell containing `p`; `None` when blocked or off the map.
    pub fn component(&self, p: DVec2) -> Option<u32> {
        self.index(p).map(|i| self.label[i]).filter(|&l| l != 0)
    }

    pub fn is_drivable(&self, p: DVec2) -> bool {
        self.component(p).is_some()
    }

    /// The component with the most cells (0 if nothing is drivable).
    pub fn largest(&self) -> u32 {
        self.largest
    }

    /// Cells of component `id`.
    pub fn component_size(&self, id: u32) -> usize {
        self.sizes.get(id as usize).copied().unwrap_or(0)
    }

    /// Share of drivable cells.
    pub fn drivable_share(&self) -> f64 {
        self.sizes.iter().sum::<usize>() as f64 / self.label.len() as f64
    }

    /// Whether `b` can be reached from `a` (both in drivable cells of the same component).
    pub fn reachable(&self, a: DVec2, b: DVec2) -> bool {
        self.component(a).is_some_and(|c| self.component(b) == Some(c))
    }

    /// Cheapest path of cell centres from `a` to `b` (A*, 8 neighbours without cutting
    /// blocked corners; the shortest one unless the grid has resistance costs), starting at
    /// `a` and ending at `b`; `None` if unreachable.
    pub fn path(&self, a: DVec2, b: DVec2) -> Option<Vec<DVec2>> {
        if !self.reachable(a, b) {
            return None;
        }
        let (start, goal) = (self.index(a)?, self.index(b)?);
        let n = self.label.len();
        let mut cost = vec![f64::INFINITY; n];
        let mut from = vec![usize::MAX; n];
        let gc = self.center(goal);
        let h = |i: usize| self.center(i).distance(gc);
        // Keys: f-cost in integer micro-cells (ties broken by index) keep the order total.
        let key = |f: f64| (f / self.cell * 1e6) as u64;
        let mut heap = BinaryHeap::new();
        cost[start] = 0.0;
        heap.push(Reverse((key(h(start)), start)));
        let open = |x: i64, y: i64| {
            x >= 0 && y >= 0 && (x as usize) < self.nx && (y as usize) < self.ny && {
                self.label[y as usize * self.nx + x as usize] != 0
            }
        };
        while let Some(Reverse((_, i))) = heap.pop() {
            if i == goal {
                break;
            }
            let (x, y) = ((i % self.nx) as i64, (i / self.nx) as i64);
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
                let (nx, ny) = (x + dx, y + dy);
                if !open(nx, ny) || (dx != 0 && dy != 0 && !(open(x + dx, y) && open(x, y + dy))) {
                    continue;
                }
                let j = ny as usize * self.nx + nx as usize;
                let step = if dx != 0 && dy != 0 { std::f64::consts::SQRT_2 } else { 1.0 } * self.cell;
                let weight = if self.cost.is_empty() { 1.0 } else { 0.5 * f64::from(self.cost[i] + self.cost[j]) };
                let c = cost[i] + weight * step;
                if c < cost[j] {
                    cost[j] = c;
                    from[j] = i;
                    heap.push(Reverse((key(c + h(j)), j)));
                }
            }
        }
        let mut cells = vec![goal];
        while *cells.last()? != start {
            cells.push(from[*cells.last()?]);
        }
        let mut path: Vec<DVec2> = cells.iter().rev().map(|&i| self.center(i)).collect();
        if path.len() < 2 {
            path.insert(0, a);
        }
        path[0] = a;
        *path.last_mut()? = b;
        Some(path)
    }

    /// The cheapest paths from `from` through `goals` in turn, one leg per goal, smoothed by a
    /// moving average over `2·PATH_SMOOTHING + 1` cells (each leg keeps its ends); `None` if a
    /// goal cannot be reached.
    pub fn legs(&self, from: DVec2, goals: &[DVec2]) -> Option<Vec<Vec<DVec2>>> {
        let mut a = from;
        goals
            .iter()
            .map(|&b| {
                let p = self.path(a, b)?;
                a = b;
                Some(smooth(&p, PATH_SMOOTHING))
            })
            .collect()
    }
}

/// Half-width (cells) of the moving average over planned paths: the grid's 45° steps become
/// curves; a right-angle corner is cut by 1.2 cells, gentler bends by less.
const PATH_SMOOTHING: usize = 3;

/// Moving average over `2·half + 1` points; the window shrinks towards the ends, which stay.
fn smooth(p: &[DVec2], half: usize) -> Vec<DVec2> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let h = half.min(i).min(n - 1 - i);
            p[i - h..=i + h].iter().sum::<DVec2>() / (2 * h + 1) as f64
        })
        .collect()
}

/// Half the width of a ground vehicle (m): its widest wheel or collider.
pub fn half_width(def: &WheeledDef) -> f64 {
    let wheels = (0..def.num_wheels()).map(|w| def.wheel_position_in_line(w).y.abs() + 0.5 * def.tire(w / 2).width());
    let colliders = def.colliders_in_line().into_iter().map(|c| c.center.y.abs() + c.radius);
    wheels.chain(colliders).fold(0.0, f64::max)
}

/// Pose of a ground vehicle resting at `xy` with heading `yaw` on the terrain: `rest` (its
/// rest pose on flat ground at the origin, heading +x) carried into the plane fitted through
/// the ground under its wheels and lifted so that no wheel starts below the ground.
pub fn ground_pose(world: &StaticWorld, def: &WheeledDef, rest: &Pose, xy: DVec2, yaw: f64) -> Pose {
    let terrain = world.terrain();
    let heading = DQuat::from_rotation_z(yaw);
    let contacts: Vec<DVec2> = (0..def.num_wheels())
        .map(|w| (heading * def.wheel_position_in_line(w)).truncate())
        .chain([DVec2::ZERO])
        .collect();
    // Least squares z = a + b·x + c·y over the contacts (relative to xy).
    let mut m = DMat3::ZERO;
    let mut rhs = DVec3::ZERO;
    let heights: Vec<f64> = contacts.iter().map(|d| terrain.height(xy.x + d.x, xy.y + d.y)).collect();
    for (d, &z) in contacts.iter().zip(&heights) {
        let row = DVec3::new(1.0, d.x, d.y);
        m += DMat3::from_cols(row * row.x, row * row.y, row * row.z);
        rhs += row * z;
    }
    let (a, b, c) = if m.determinant().abs() > 1e-9 {
        let s = m.inverse() * rhs;
        (s.x, s.y, s.z)
    } else {
        let (h, n) = terrain.height_normal(xy.x, xy.y);
        (h, -n.x / n.z, -n.y / n.z)
    };
    let lift = contacts.iter().zip(&heights).map(|(d, &z)| z - (a + b * d.x + c * d.y)).fold(0.0, f64::max);
    let normal = DVec3::new(-b, -c, 1.0).normalize();
    let forward = heading * DVec3::X;
    let x = (forward - normal * forward.dot(normal)).normalize();
    let frame = DQuat::from_mat3(&DMat3::from_cols(x, normal.cross(x), normal));
    let ground = xy.extend(a + lift);
    Pose::new(ground + frame * rest.pos, (frame * rest.rot).normalize())
}

/// Slope (radians) of the ground at `xy`: the steeper of the surface normal there and the
/// gradient over `±reach` (m) along x and y.
pub fn slope(world: &StaticWorld, xy: DVec2, reach: f64) -> f64 {
    let t = world.terrain();
    let h = |dx: f64, dy: f64| t.height(xy.x + dx, xy.y + dy);
    let (_, n) = t.height_normal(xy.x, xy.y);
    let gx = (h(reach, 0.0) - h(-reach, 0.0)) / (2.0 * reach);
    let gy = (h(0.0, reach) - h(0.0, -reach)) / (2.0 * reach);
    n.z.clamp(-1.0, 1.0).acos().max(gx.hypot(gy).atan())
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_world::testworlds;

    #[test]
    fn grid_blocks_trunks_water_and_slopes() {
        // A single tree at the origin blocks the cells around it.
        let w = testworlds::single_tree();
        let g = DriveGrid::new(&w, &DrivableSpec::default(), 1.0);
        assert!(!g.is_drivable(DVec2::ZERO) && g.is_drivable(DVec2::new(10.0, 0.0)));
        assert!(g.reachable(DVec2::new(-10.0, 0.0), DVec2::new(10.0, 0.0)));
        let p = g.path(DVec2::new(-10.0, 0.0), DVec2::new(10.0, 0.0)).unwrap();
        assert!(p.iter().all(|q| g.is_drivable(*q)));
        let len: f64 = p.windows(2).map(|s| s[0].distance(s[1])).sum();
        assert!((20.0..26.0).contains(&len), "{len}");

        // The lake's water is blocked; the shore around it is one component.
        let w = testworlds::lake(200.0, 4.0, -1.0);
        let g = DriveGrid::new(&w, &DrivableSpec::default(), 1.0);
        assert!(!g.is_drivable(DVec2::ZERO));
        assert!(g.reachable(DVec2::new(-90.0, -90.0), DVec2::new(90.0, 90.0)));

        // A 30° incline is too steep by default, not at 35°.
        let w = testworlds::incline(100.0, 30f64.to_radians(), Default::default());
        assert_eq!(DriveGrid::new(&w, &DrivableSpec::default(), 1.0).drivable_share(), 0.0);
        let steep = DrivableSpec { max_slope_deg: 35.0, ..DrivableSpec::default() };
        assert_eq!(DriveGrid::new(&w, &steep, 1.0).drivable_share(), 1.0);

        // Walls enclose the arena: inside and outside are separate.
        let w = testworlds::walled_arena(20.0, 3.0);
        let g = DriveGrid::new(&w, &DrivableSpec::default(), 1.0);
        let (lo, hi) = w.extent();
        let outside = DVec2::new(lo.x + 1.0, hi.y - 1.0);
        if g.is_drivable(outside) {
            assert!(!g.reachable(DVec2::ZERO, outside));
        }
        assert!(g.is_drivable(DVec2::ZERO));
    }

    #[test]
    fn legs_run_through_the_goals_in_turn() {
        let w = testworlds::single_tree();
        let g = DriveGrid::new(&w, &DrivableSpec::default(), 1.0);
        let start = DVec2::new(-20.0, 0.0);
        let goals = [DVec2::new(20.0, 0.0), DVec2::new(20.0, 20.0), DVec2::new(-20.0, 20.0)];
        let legs = g.legs(start, &goals).unwrap();
        assert_eq!(legs.len(), 3);
        let mut from = start;
        for (leg, &goal) in legs.iter().zip(&goals) {
            // Each leg runs from the previous goal to the next, off the tree.
            assert!(leg[0].distance(from) < 1e-9 && leg.last().unwrap().distance(goal) < 1e-9);
            assert!(leg.iter().all(|q| g.is_drivable(*q)), "{leg:?}");
            let len: f64 = leg.windows(2).map(|s| s[0].distance(s[1])).sum();
            assert!(len < 1.2 * from.distance(goal) + 4.0, "{len}");
            from = goal;
        }
        // The first leg bends around the tree.
        assert!(legs[0].iter().all(|q| q.length() > 2.0));
        // An unreachable goal: no legs.
        assert!(g.legs(start, &[DVec2::ZERO]).is_none());
        // A goal in the start cell still gives a leg of two points.
        assert_eq!(g.legs(start, &[start + DVec2::new(0.3, 0.0)]).unwrap()[0].len(), 2);
    }

    #[test]
    fn smoothing_keeps_the_ends_and_straight_lines() {
        let line: Vec<DVec2> = (0..10).map(|i| DVec2::new(i as f64, 2.0 * i as f64)).collect();
        assert!(smooth(&line, 3).iter().zip(&line).all(|(a, b)| a.distance(*b) < 1e-12));
        // A right angle (2 m cells): the ends stay, the corner is cut by 1.2 cells.
        let corner: Vec<DVec2> = (0..=6)
            .map(|i| DVec2::new(2.0 * i as f64, 0.0))
            .chain((1..=6).map(|i| DVec2::new(12.0, 2.0 * i as f64)))
            .collect();
        let s = smooth(&corner, 3);
        assert_eq!((s[0], s[12]), (corner[0], corner[12]));
        let cut = s[6].distance(corner[6]);
        assert!((cut - 1.2 * 2.0).abs() < 0.1, "{cut}");
        assert_eq!(smooth(&corner[..1], 3), corner[..1]);
    }

    #[test]
    fn resistance_costs_steer_paths_around_soft_ground() {
        use autonomousim_core::material::MaterialTable;
        use autonomousim_world::{HeightGrid, MapMeta, ObstacleSet};
        // A plowed field across the way on a meadow; around its ends is further.
        let field = |x: f64, y: f64| x.abs() < 10.0 && y.abs() < 15.0;
        let terrain = HeightGrid::from_fn(
            DVec2::splat(-60.0),
            1.0,
            121,
            121,
            |_, _| 0.0,
            |x, y| {
                if field(x, y) { MaterialId::PLOWED } else { MaterialId::MEADOW }
            },
        );
        let w =
            StaticWorld::new(MapMeta::new("field", "test", 0), terrain, ObstacleSet::default(), MaterialTable::rural());
        let (a, b) = (DVec2::new(-30.0, 0.0), DVec2::new(30.0, 0.0));
        let length = |p: &[DVec2]| p.windows(2).map(|s| s[0].distance(s[1])).sum::<f64>();
        let crosses = |p: &[DVec2]| p.iter().any(|q| field(q.x, q.y));
        let preset = |name| autonomousim_vehicles::SharedDef::from(autonomousim_vehicles::presets::get(name).unwrap());
        let (car, apc) = (preset("sedan_like"), preset("tracked_apc"));
        let spec = DrivableSpec { resistance_cost: 30.0, ..DrivableSpec::default() };
        let route = |def: &autonomousim_vehicles::SharedDef, spec: &DrivableSpec| {
            let d = def.as_wheeled().unwrap();
            DriveGrid::with_resistance(&w, spec, 1.0, |m| d.motion_resistance(m, 9.80665)).path(a, b).unwrap()
        };

        // Shortest: straight through. A car goes around the field (rolling resistance 0.14
        // against 0.06); a tracked vehicle hardly sinks in, so it crosses.
        let straight = route(&car, &DrivableSpec::default());
        assert!(crosses(&straight) && length(&straight) < 61.0);
        let car_route = route(&car, &spec);
        assert!(!crosses(&car_route) && length(&car_route) > 64.0, "{}", length(&car_route));
        let apc_route = route(&apc, &spec);
        assert!(crosses(&apc_route) && length(&apc_route) < 61.0);
    }

    #[test]
    fn ground_pose_follows_the_slope() {
        let def = autonomousim_vehicles::SharedDef::from(autonomousim_vehicles::presets::get("sedan_like").unwrap());
        let def = def.as_wheeled().unwrap().clone();
        let rest = Pose::new(DVec3::Z * 0.5, DQuat::IDENTITY);
        let angle = 15f64.to_radians();
        let w = testworlds::incline(100.0, angle, Default::default());
        // (The grid stores f32 heights.) Facing uphill (+x): pitched nose up by the slope angle, standing on the plane.
        let p = ground_pose(&w, &def, &rest, DVec2::new(3.0, 1.0), 0.0);
        let up = p.rot * DVec3::Z;
        assert!((up.angle_between(DVec3::Z) - angle).abs() < 1e-5);
        assert!((p.rot * DVec3::X).z > 0.0);
        let foot = p.pos - up * 0.5;
        assert!((foot.z - w.terrain().height(foot.x, foot.y)).abs() < 1e-4);
        // Facing across the slope: rolled, heading kept.
        let p = ground_pose(&w, &def, &rest, DVec2::new(3.0, 1.0), std::f64::consts::FRAC_PI_2);
        let fwd = p.rot * DVec3::X;
        assert!(fwd.z.abs() < 1e-9 && (fwd.y - 1.0).abs() < 1e-6);
    }
}
