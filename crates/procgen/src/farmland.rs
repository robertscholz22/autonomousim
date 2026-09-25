//! Farmland for rural maps: field parcels, hedges and fences along their edges, farm
//! buildings, tree lines along roads, and the clearance rules that keep roads and yards free.
//!
//! Parcels are the Voronoi cells of a jittered grid of centres, so every query (which parcel,
//! how far to its edge) is local and runs per cell in parallel. Hedges and fences follow the
//! Voronoi edges, found by sampling the bisector of each pair of neighbouring centres.

use crate::rural::Yard;
use crate::scatter::broadleaf;
use autonomousim_core::material::MaterialId;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::from_yaw;
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::terrain::Terrain;
use autonomousim_world::obstacles::tags;
use autonomousim_world::{HeightGrid, Obstacle, ObstacleShape, RoadNetwork};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FieldsConfig {
    /// Mean distance between parcel centres (m).
    pub spacing: f64,
    /// Jitter of the centres, as a fraction of `spacing` (0 gives a square grid).
    pub jitter: f64,
    /// Relative shares of meadow, crop and plowed parcels.
    pub meadow: f64,
    pub crop: f64,
    pub plowed: f64,
    /// Share of parcels left as woods; parcels steeper than `woods_slope_deg` at their centre
    /// are always woods.
    pub woods: f64,
    pub woods_slope_deg: f64,
    /// Grass strip (m) along parcel edges and beyond road edges.
    pub headland: f64,
    /// Fields whose centre lies farther than this (m) from any road get a track to their gate.
    pub track_distance: f64,
}

impl Default for FieldsConfig {
    fn default() -> Self {
        Self {
            spacing: 120.0,
            jitter: 0.8,
            meadow: 0.35,
            crop: 0.4,
            plowed: 0.25,
            woods: 0.08,
            woods_slope_deg: 15.0,
            headland: 2.0,
            track_distance: 50.0,
        }
    }
}

impl FieldsConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if !(self.spacing >= 20.0 && (0.0..=1.0).contains(&self.jitter)) {
            return Err("fields: spacing ≥ 20 m and jitter in [0, 1]".into());
        }
        let shares = [self.meadow, self.crop, self.plowed];
        if shares.iter().any(|s| *s < 0.0) || shares.iter().sum::<f64>() <= 0.0 {
            return Err("fields: meadow, crop and plowed shares must be ≥ 0 and not all 0".into());
        }
        if !((0.0..=1.0).contains(&self.woods) && self.woods_slope_deg > 0.0 && self.headland >= 0.0) {
            return Err("fields: woods in [0, 1], woods_slope_deg > 0, headland ≥ 0".into());
        }
        if self.track_distance.is_nan() {
            return Err("fields.track_distance must be a number".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScatterConfig {
    /// Shares of parcel edges lined by a hedge or by a fence; the rest are open.
    pub hedge: f64,
    pub fence: f64,
    /// Hedge height range (m) and width (m).
    pub hedge_height: [f64; 2],
    pub hedge_width: f64,
    pub fence_height: f64,
    /// Share of roads lined by trees, and the tree spacing along them (m).
    pub tree_line: f64,
    pub tree_line_spacing: f64,
    /// Farm buildings around the yards.
    pub buildings: bool,
    /// Trees in the woods (`forest_density`) and on grass (`meadow_density`).
    pub trees: crate::TreesConfig,
    /// Obstacles keep this far (m) from road edges, unless they are at least `road_headroom`
    /// above the road (tree crowns).
    pub road_clearance: f64,
    pub road_headroom: f64,
}

impl Default for ScatterConfig {
    fn default() -> Self {
        Self {
            hedge: 0.4,
            fence: 0.3,
            hedge_height: [1.6, 2.8],
            hedge_width: 1.4,
            fence_height: 1.2,
            tree_line: 0.35,
            tree_line_spacing: 12.0,
            buildings: true,
            trees: crate::TreesConfig {
                forest_density: 250.0,
                meadow_density: 0.4,
                min_spacing: 3.5,
                max_slope_deg: 35.0,
                min_height: 8.0,
                max_height: 22.0,
                conifer_low: 0.3,
                conifer_high: 0.3,
            },
            road_clearance: 1.5,
            road_headroom: 4.5,
        }
    }
}

impl ScatterConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if !((0.0..=1.0).contains(&self.hedge) && (0.0..=1.0).contains(&self.fence) && self.hedge + self.fence <= 1.0) {
            return Err("scatter: hedge and fence shares in [0, 1], at most 1 together".into());
        }
        let [lo, hi] = self.hedge_height;
        if !(lo > 0.0 && hi >= lo && self.hedge_width > 0.0 && self.fence_height > 0.0) {
            return Err("scatter: hedge and fence sizes must be positive".into());
        }
        if !((0.0..=1.0).contains(&self.tree_line) && self.tree_line_spacing >= 2.0) {
            return Err("scatter: tree_line in [0, 1], tree_line_spacing ≥ 2 m".into());
        }
        if !(self.road_clearance >= 0.0 && self.road_headroom >= 0.0) {
            return Err("scatter: road_clearance and road_headroom must be ≥ 0".into());
        }
        self.trees.validate()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParcelKind {
    Meadow,
    Crop,
    Plowed,
    Woods,
}

impl ParcelKind {
    pub(crate) fn material(self) -> MaterialId {
        match self {
            Self::Meadow => MaterialId::MEADOW,
            Self::Crop => MaterialId::CROP,
            Self::Plowed => MaterialId::PLOWED,
            Self::Woods => MaterialId::FOREST_FLOOR,
        }
    }
}

/// Voronoi parcels of a jittered grid of centres that extends one cell beyond the map.
pub(crate) struct Parcels {
    origin: DVec2,
    spacing: f64,
    k: usize,
    pub centres: Vec<DVec2>,
    pub kinds: Vec<ParcelKind>,
}

impl Parcels {
    /// `slope(p)`: terrain slope (rise over run) around `p`.
    pub(crate) fn new(c: &FieldsConfig, size: f64, slope: &dyn Fn(DVec2) -> f64, seed: &Seed) -> Self {
        let mut rng = seed.rng();
        let k = (size / c.spacing).ceil() as usize + 2;
        let origin = DVec2::splat(-0.5 * size - c.spacing);
        let mut centres = Vec::with_capacity(k * k);
        let mut kinds = Vec::with_capacity(k * k);
        let woods_slope = libm::tan(c.woods_slope_deg.to_radians());
        let total = c.meadow + c.crop + c.plowed;
        for gy in 0..k {
            for gx in 0..k {
                let (jx, jy, u_woods, u_kind) = (rng.uniform(), rng.uniform(), rng.uniform(), rng.uniform());
                let g = DVec2::new(gx as f64 + 0.5 + c.jitter * (jx - 0.5), gy as f64 + 0.5 + c.jitter * (jy - 0.5));
                let p = origin + g * c.spacing;
                centres.push(p);
                let u = u_kind * total;
                kinds.push(if u_woods < c.woods || slope(p) > woods_slope {
                    ParcelKind::Woods
                } else if u < c.meadow {
                    ParcelKind::Meadow
                } else if u < c.meadow + c.crop {
                    ParcelKind::Crop
                } else {
                    ParcelKind::Plowed
                });
            }
        }
        Self { origin, spacing: c.spacing, k, centres, kinds }
    }

    fn neighbours(&self, p: DVec2) -> impl Iterator<Item = usize> + '_ {
        let g = ((p - self.origin) / self.spacing).floor();
        let (gx, gy) = (g.x as isize, g.y as isize);
        let k = self.k as isize;
        (-2..=2).flat_map(move |dy| {
            (-2..=2).filter_map(move |dx| {
                let (x, y) = (gx + dx, gy + dy);
                (x >= 0 && y >= 0 && x < k && y < k).then_some((y * k + x) as usize)
            })
        })
    }

    /// The parcel containing `p` and the distance from `p` to the parcel's edge.
    pub(crate) fn locate(&self, p: DVec2) -> (usize, f64) {
        let mut best = (usize::MAX, f64::INFINITY);
        for i in self.neighbours(p) {
            let d = self.centres[i].distance_squared(p);
            if d < best.1 || (d == best.1 && i < best.0) {
                best = (i, d);
            }
        }
        let (i, di) = best;
        let si = self.centres[i];
        let mut edge = f64::INFINITY;
        for j in self.neighbours(p) {
            if j != i {
                let sj = self.centres[j];
                edge = edge.min((sj.distance_squared(p) - di) / (2.0 * sj.distance(si)));
            }
        }
        (i, edge)
    }

    /// The Voronoi edges between neighbouring parcels as `(i, j, a, b)` segments, `i < j`.
    pub(crate) fn edges(&self) -> Vec<(usize, usize, DVec2, DVec2)> {
        let mut out = Vec::new();
        let step = 0.5;
        for i in 0..self.centres.len() {
            let si = self.centres[i];
            for j in self.neighbours(si) {
                if j <= i {
                    continue;
                }
                let sj = self.centres[j];
                let mid = 0.5 * (si + sj);
                let dir = (sj - si).normalize().perp();
                let reach = 2.0 * self.spacing;
                let on_edge = |t: f64| {
                    let p = mid + t * dir;
                    let (k, _) = self.locate(p);
                    let d = si.distance_squared(p);
                    k == i || k == j || self.centres[k].distance_squared(p) >= d - 1e-9
                };
                // The edge is one interval of the bisector; find it from the midpoint when the
                // midpoint lies on it (neighbours that share an edge almost always do).
                if !on_edge(0.0) {
                    continue;
                }
                let mut hi = 0.0;
                while hi < reach && on_edge(hi + step) {
                    hi += step;
                }
                let mut lo = 0.0;
                while lo > -reach && on_edge(lo - step) {
                    lo -= step;
                }
                if hi - lo >= 2.0 {
                    out.push((i, j, mid + lo * dir, mid + hi * dir));
                }
            }
        }
        out
    }
}

/// Where obstacles may not go: roads (up to the headroom), yards, water, off the map.
pub(crate) struct Clearance<'a> {
    pub net: &'a RoadNetwork,
    pub yards: &'a [Yard],
    pub yard_half: DVec2,
    pub grid: &'a HeightGrid,
    pub clearance: f64,
    pub headroom: f64,
    pub max_half_width: f64,
}

impl Clearance<'_> {
    /// Whether an obstacle with a bounding circle of `radius` around `centre` and a lowest
    /// point at `bottom` keeps clear of roads and yards and stands on dry land in the map.
    pub(crate) fn allows(&self, centre: DVec2, radius: f64, bottom: f64) -> bool {
        let (lo, hi) = self.grid.extent();
        if centre.x - radius < lo.x || centre.y - radius < lo.y || centre.x + radius > hi.x || centre.y + radius > hi.y
        {
            return false;
        }
        if self.grid.water_level(centre.x, centre.y).is_some() {
            return false;
        }
        if let Some(rp) = self.net.nearest(centre, self.max_half_width + self.clearance + radius) {
            let road = &self.net.roads()[rp.road as usize];
            if rp.projection.distance < 0.5 * road.width + self.clearance + radius
                && bottom < rp.projection.point.z + self.headroom
            {
                return false;
            }
        }
        !self.yards.iter().any(|y| y.distance(centre, self.yard_half) < radius + 0.5)
    }
}

/// Footprint of an upright obstacle as circles (centre, radius) covering it, and its lowest
/// point. Cuboids are covered by circles about 1 m apart, round shapes by one circle.
fn footprint(o: &Obstacle) -> (Vec<(DVec2, f64)>, f64) {
    let p = o.pose.pos;
    let (r, hz) = match &o.shape {
        ObstacleShape::Sphere { radius } => (*radius, *radius),
        ObstacleShape::Capsule { half_height, radius } => (*radius, half_height + radius),
        ObstacleShape::Cylinder { half_height, radius } | ObstacleShape::Cone { half_height, radius } => {
            (*radius, *half_height)
        }
        ObstacleShape::Cuboid { half_extents: he } => {
            let (nx, ny) = ((2.0 * he.x).ceil().max(1.0), (2.0 * he.y).ceil().max(1.0));
            let (dx, dy) = (2.0 * he.x / nx, 2.0 * he.y / ny);
            let r = 0.5 * (dx * dx + dy * dy).sqrt();
            let mut circles = Vec::with_capacity((nx * ny) as usize);
            for i in 0..nx as usize {
                for j in 0..ny as usize {
                    let local = DVec3::new(-he.x + (i as f64 + 0.5) * dx, -he.y + (j as f64 + 0.5) * dy, 0.0);
                    circles.push((o.pose.transform_point(local).truncate(), r));
                }
            }
            return (circles, p.z - he.z);
        }
        ObstacleShape::ConvexHull { points } => (
            points.iter().map(|q| q.truncate().length()).fold(0.0, f64::max),
            points.iter().map(|q| -q.z).fold(0.0, f64::max),
        ),
    };
    (vec![(p.truncate(), r)], p.z - hz)
}

/// Keep the groups (e.g. trunk and crown) whose members are all clear.
pub(crate) fn keep_clear(groups: impl IntoIterator<Item = Vec<Obstacle>>, clear: &Clearance, out: &mut Vec<Obstacle>) {
    for g in groups {
        if g.iter().all(|o| {
            let (circles, bottom) = footprint(o);
            circles.iter().all(|&(c, r)| clear.allows(c, r, bottom))
        }) {
            out.extend(g);
        }
    }
}

/// An upright cuboid from `a` to `b` on the ground, `width` wide and `height` tall above its
/// highest ground point, sunk to below its lowest.
fn wall(grid: &HeightGrid, a: DVec2, b: DVec2, width: f64, height: f64) -> (ObstacleShape, Pose) {
    let (lo, hi) = [a, 0.5 * (a + b), b]
        .iter()
        .map(|p| grid.height(p.x, p.y))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), h| (lo.min(h), hi.max(h)));
    let (bottom, top) = (lo - 0.3, hi + height);
    let d = b - a;
    let centre = (0.5 * (a + b)).extend(0.5 * (bottom + top));
    let shape = ObstacleShape::Cuboid { half_extents: DVec3::new(0.5 * d.length(), 0.5 * width, 0.5 * (top - bottom)) };
    (shape, Pose::new(centre, from_yaw(d.y.atan2(d.x))))
}

/// Hedges and fences along the parcel edges (not between two woods), as obstacle groups.
pub(crate) fn edge_obstacles(
    c: &ScatterConfig,
    parcels: &Parcels,
    grid: &HeightGrid,
    seed: &Seed,
) -> Vec<Vec<Obstacle>> {
    let mut out = Vec::new();
    for (i, j, a, b) in parcels.edges() {
        if parcels.kinds[i] == ParcelKind::Woods && parcels.kinds[j] == ParcelKind::Woods {
            continue;
        }
        let mut rng = seed.child_index((i * parcels.centres.len() + j) as u64).rng();
        let (u, u_height) = (rng.uniform(), rng.uniform());
        let hedge = u < c.hedge;
        if !hedge && u >= c.hedge + c.fence {
            continue;
        }
        let piece = if hedge { 3.0 } else { 4.0 };
        let count = ((b - a).length() / piece).ceil() as usize;
        let height = c.hedge_height[0] + u_height * (c.hedge_height[1] - c.hedge_height[0]);
        for k in 0..count {
            let (p, q) = (a.lerp(b, k as f64 / count as f64), a.lerp(b, (k + 1) as f64 / count as f64));
            if hedge {
                // Pieces overlap a little so that the hedge has no seams.
                let d = (q - p).normalize() * 0.2;
                let (shape, pose) = wall(grid, p - d, q + d, c.hedge_width, height);
                let (core, core_pose) = wall(grid, p, q, 0.3, 0.8 * height);
                out.push(vec![
                    Obstacle::foliage(shape, pose).with_tag(tags::HEDGE),
                    Obstacle::solid(core, core_pose, MaterialId::WOOD).with_tag(tags::HEDGE),
                ]);
            } else {
                let (shape, pose) = wall(grid, p, q, 0.1, c.fence_height);
                out.push(vec![Obstacle::solid(shape, pose, MaterialId::WOOD).with_tag(tags::FENCE)]);
            }
        }
    }
    out
}

/// Farm buildings around a yard, outside it and facing away from its road (local +x).
pub(crate) fn buildings(yard: &Yard, half: DVec2, grid: &HeightGrid, rng: &mut SimRng) -> Vec<Vec<Obstacle>> {
    let (hx, hy) = (half.x, half.y);
    let side = if rng.uniform() < 0.5 { 1.0 } else { -1.0 };
    let scale = |rng: &mut SimRng| rng.range(0.85, 1.15);
    // (local centre, half extents, material, chance)
    let plans = [
        (DVec2::new(-(hx + 7.0), 0.0), DVec3::new(6.0, 11.0, 4.5), MaterialId::WOOD, 0.9),
        (DVec2::new(0.2 * hx, side * (hy + 6.0)), DVec3::new(7.0, 5.0, 3.5), MaterialId::CONCRETE, 1.0),
        (DVec2::new(-0.4 * hx, -side * (hy + 5.0)), DVec3::new(5.0, 4.0, 2.5), MaterialId::METAL, 0.7),
    ];
    let (s, co) = yard.heading.sin_cos();
    let world = |l: DVec2| yard.centre + DVec2::new(co * l.x - s * l.y, s * l.x + co * l.y);
    let mut out = Vec::new();
    for (local, he, material, chance) in plans {
        let (u, k) = (rng.uniform(), scale(rng));
        if u >= chance {
            continue;
        }
        let he = DVec3::new(he.x, he.y, he.z * k);
        let centre = world(local);
        let corners = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)].map(|(a, b)| {
            grid.height(
                world(local + DVec2::new(a * he.x, b * he.y)).x,
                world(local + DVec2::new(a * he.x, b * he.y)).y,
            )
        });
        let lo = corners.iter().copied().fold(f64::INFINITY, f64::min) - 0.3;
        let hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 2.0 * he.z;
        let shape = ObstacleShape::Cuboid { half_extents: DVec3::new(he.x, he.y, 0.5 * (hi - lo)) };
        let pose = Pose::new(centre.extend(0.5 * (lo + hi)), from_yaw(yard.heading));
        out.push(vec![Obstacle::solid(shape, pose, material).with_tag(tags::BUILDING)]);
    }
    if rng.uniform() < 0.5 {
        let (r, hh) = (2.5, 6.0 * scale(rng));
        let centre = world(DVec2::new(-(hx + 3.5), -side * (hy + 5.0)));
        let z = grid.height(centre.x, centre.y) - 0.3;
        let shape = ObstacleShape::Cylinder { half_height: hh, radius: r };
        out.push(vec![
            Obstacle::solid(shape, Pose::from_translation(centre.extend(z + hh)), MaterialId::METAL)
                .with_tag(tags::SILO),
        ]);
    }
    out
}

/// Rows of broadleaf trees beside some of the roads, away from the nodes.
pub(crate) fn tree_lines(c: &ScatterConfig, net: &RoadNetwork, grid: &HeightGrid, seed: &Seed) -> Vec<Vec<Obstacle>> {
    let mut out = Vec::new();
    for (k, road) in net.roads().iter().enumerate() {
        let mut rng = seed.child_index(k as u64).rng();
        let (u, u_sides) = (rng.uniform(), rng.uniform());
        if u >= c.tree_line {
            continue;
        }
        let sides: &[f64] = if u_sides < 0.4 {
            &[1.0, -1.0]
        } else if u_sides < 0.7 {
            &[1.0]
        } else {
            &[-1.0]
        };
        let len = road.line.length();
        let mut s = 15.0;
        while s < len - 15.0 {
            for &side in sides {
                let (jitter, off, h) = (rng.range(-2.0, 2.0), rng.range(0.0, 1.0), rng.range(12.0, 17.0));
                let t = (s + jitter).clamp(0.0, len);
                let p = road.line.point_at(t).truncate();
                let heading = road.line.heading_at(t);
                let q = p + side * (0.5 * road.width + 3.5 + off) * DVec2::new(-heading.sin(), heading.cos());
                // Keep clear of other roads by at least as much as of this one.
                if net
                    .nearest(q, 0.5 * road.width + 3.0)
                    .is_some_and(|rp| rp.projection.distance < 0.5 * road.width + 3.0)
                {
                    continue;
                }
                let base = q.extend(grid.height(q.x, q.y) - 0.2);
                out.push(broadleaf(base, h).to_vec());
            }
            s += c.tree_line_spacing;
        }
    }
    out
}
