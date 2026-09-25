//! Where ground vehicles can drive: a coarse grid over the map whose cells are drivable when
//! the terrain is not too steep, dry, and free of solid obstacles (trunks, large rocks) within
//! the vehicle's half-width. Drivable cells are labelled by 4-connected component, so whether
//! a goal can be reached from a spawn is one lookup; [`DriveGrid::path`] finds a path (A*).
//!
//! Also here: [`ground_pose`], the pose of a ground vehicle resting on uneven terrain.

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
            && self.max_water_depth >= 0.0;
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
}

impl DriveGrid {
    /// Grid over `world` for vehicles `half_width` wide (m) on each side.
    pub fn new(world: &StaticWorld, spec: &DrivableSpec, half_width: f64) -> Self {
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
        Self { origin: lo, cell, nx, ny, label, sizes, largest }
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

    /// Shortest path of cell centres from `a` to `b` (A*, 8 neighbours without cutting
    /// blocked corners), starting at `a` and ending at `b`; `None` if unreachable.
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
                let c = cost[i] + if dx != 0 && dy != 0 { std::f64::consts::SQRT_2 } else { 1.0 } * self.cell;
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
        path[0] = a;
        *path.last_mut()? = b;
        Some(path)
    }
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
