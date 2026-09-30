//! Blocks, lots, zoning, buildings, parks, parking and street furniture of an urban map.
//!
//! Everything works on a raster of `lots.cell` (2 m): cells within `margin` of a road's
//! sidewalk are street, cells outside the city, in water or too steep are blocked, the rest
//! is free. Blocks are the 4-connected components of free cells; each is zoned and split
//! recursively across its longer principal axis into lots, keeping only splits that leave
//! both halves connected and on a street. A lot's frame has y pointing away from its street;
//! its building, parking rows or park trees fit into the largest rectangle of the lot in that
//! frame (on a 1 m raster whose cells lie wholly inside the lot).

use super::UrbanConfig;
use super::layout::Districts;
use crate::rural::Heights;
use crate::scatter::broadleaf;
use autonomousim_core::material::MaterialId;
use autonomousim_core::math::Pose;
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_world::lanes::Area;
use autonomousim_world::obstacles::tags;
use autonomousim_world::{
    BayKind, Building, JunctionKind, Lot, Obstacle, ObstacleShape, Pad, ParkingBay, RoadNetwork, Roof, Sites, Zone,
};
use glam::{DQuat, DVec2, DVec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::{FRAC_PI_2, PI};

/// Lots and buildings of one zone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZoneConfig {
    /// Target lot area (m²), drawn per block.
    pub lot_area: [f64; 2],
    /// Setbacks of the building from the street and from the other sides of the lot (m).
    pub front: f64,
    pub side: f64,
    /// Storeys, drawn per building.
    pub storeys: [u8; 2],
    /// Largest building footprint along and across the street (m).
    pub max_size: [f64; 2],
    /// Share of the flat roofs (at least 14 m square) with a landing pad.
    pub pad_share: f64,
    /// Share of the rectangular buildings with a gable roof.
    pub pitched_share: f64,
}

impl Default for ZoneConfig {
    fn default() -> Self {
        Self {
            lot_area: [500.0, 1100.0],
            front: 1.0,
            side: 1.0,
            storeys: [3, 8],
            max_size: [45.0, 35.0],
            pad_share: 0.25,
            pitched_share: 0.0,
        }
    }
}

impl ZoneConfig {
    fn validate(&self, name: &str) -> Result<(), String> {
        let ok = self.lot_area[0] > 0.0
            && self.lot_area[0] <= self.lot_area[1]
            && self.front >= 0.0
            && self.side >= 0.0
            && self.storeys[0] >= 1
            && self.storeys[0] <= self.storeys[1]
            && self.max_size.iter().all(|&m| m >= 6.0)
            && (0.0..=1.0).contains(&self.pad_share)
            && (0.0..=1.0).contains(&self.pitched_share);
        if ok {
            Ok(())
        } else {
            Err(format!(
                "lots.{name}: lot_area 0 < min ≤ max, setbacks ≥ 0, storeys 1 ≤ min ≤ max, max_size ≥ 6, shares in [0, 1]"
            ))
        }
    }
}

/// Blocks, lots and buildings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LotsConfig {
    /// Raster cell (m).
    pub cell: f64,
    /// Clearance of the lots from the sidewalks (m); at least the raster cell's half diagonal.
    pub margin: f64,
    /// Blocks smaller than this stay green (m²).
    pub min_block: f64,
    /// Narrowest lot (m) across a cut.
    pub min_width: f64,
    /// Steepest ground (rise over run) a lot may take.
    pub max_slope: f64,
    /// Shares of the blocks that become parks, of the outer grid blocks (beyond half the city
    /// radius) that become industrial, and of the lots in the denser zones that become parking
    /// lots.
    pub park_share: f64,
    pub industrial_share: f64,
    /// Share of the blocks of the inner districts (within 0.55 of the city radius) that
    /// become commercial.
    pub commercial_share: f64,
    pub parking_share: f64,
    pub storey_height: f64,
    pub downtown: ZoneConfig,
    pub commercial: ZoneConfig,
    pub residential: ZoneConfig,
    pub industrial: ZoneConfig,
    /// Share of the houses with a wall or fence along the street.
    pub fence_share: f64,
}

impl Default for LotsConfig {
    fn default() -> Self {
        Self {
            cell: 2.0,
            margin: 1.5,
            min_block: 300.0,
            min_width: 12.0,
            max_slope: 0.25,
            park_share: 0.08,
            industrial_share: 0.2,
            commercial_share: 0.4,
            parking_share: 0.1,
            storey_height: 3.2,
            downtown: ZoneConfig {
                lot_area: [900.0, 2000.0],
                front: 0.0,
                side: 0.5,
                storeys: [8, 40],
                max_size: [60.0, 50.0],
                pad_share: 0.35,
                pitched_share: 0.0,
            },
            commercial: ZoneConfig::default(),
            residential: ZoneConfig {
                lot_area: [250.0, 550.0],
                front: 4.0,
                side: 2.0,
                storeys: [1, 3],
                max_size: [14.0, 12.0],
                pad_share: 0.0,
                pitched_share: 0.85,
            },
            industrial: ZoneConfig {
                lot_area: [1500.0, 4000.0],
                front: 6.0,
                side: 3.0,
                storeys: [2, 3],
                max_size: [70.0, 50.0],
                pad_share: 0.15,
                pitched_share: 0.0,
            },
            fence_share: 0.5,
        }
    }
}

impl LotsConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let shares =
            [self.park_share, self.industrial_share, self.commercial_share, self.parking_share, self.fence_share];
        if !(self.cell > 0.0
            && self.margin >= 0.5 * std::f64::consts::SQRT_2 * self.cell
            && self.min_block >= 0.0
            && self.min_width > 0.0
            && self.max_slope > 0.0
            && self.storey_height > 0.0
            && shares.iter().all(|s| (0.0..=1.0).contains(s)))
        {
            return Err("lots: cell > 0, margin ≥ cell·√2/2, min_block ≥ 0, min_width > 0, max_slope and storey_height > 0, shares in [0, 1]".into());
        }
        for (name, z) in [
            ("downtown", &self.downtown),
            ("commercial", &self.commercial),
            ("residential", &self.residential),
            ("industrial", &self.industrial),
        ] {
            z.validate(name)?;
        }
        Ok(())
    }

    fn zone(&self, z: Zone) -> &ZoneConfig {
        match z {
            Zone::Downtown => &self.downtown,
            Zone::Residential => &self.residential,
            Zone::Industrial => &self.industrial,
            _ => &self.commercial,
        }
    }
}

/// Trees, lamps and parking along the streets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FurnitureConfig {
    /// Share of the streets with sidewalks ≥ 2.5 m that get trees, and their spacing (m).
    pub street_trees: f64,
    pub tree_spacing: f64,
    /// Spacing of the lamp posts (m), alternating sides.
    pub lamp_spacing: f64,
    /// On-street parking bays: length and gap (m).
    pub bay_length: f64,
    pub bay_gap: f64,
}

impl Default for FurnitureConfig {
    fn default() -> Self {
        Self { street_trees: 0.6, tree_spacing: 15.0, lamp_spacing: 30.0, bay_length: 6.0, bay_gap: 0.5 }
    }
}

impl FurnitureConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if (0.0..=1.0).contains(&self.street_trees)
            && self.tree_spacing > 1.0
            && self.lamp_spacing > 1.0
            && self.bay_length > 1.0
            && self.bay_gap >= 0.0
        {
            Ok(())
        } else {
            Err("furniture: street_trees in [0, 1], spacings and bay_length > 1, bay_gap ≥ 0".into())
        }
    }
}

/// The zone of each raster cell (for the materials).
pub(crate) struct ZoneRaster {
    lo: DVec2,
    cell: f64,
    nx: usize,
    ny: usize,
    zones: Vec<Option<Zone>>,
}

impl ZoneRaster {
    pub(crate) fn at(&self, p: DVec2) -> Option<Zone> {
        let q = ((p - self.lo) / self.cell).floor();
        if q.x < 0.0 || q.y < 0.0 || q.x >= self.nx as f64 || q.y >= self.ny as f64 {
            return None;
        }
        self.zones[q.y as usize * self.nx + q.x as usize]
    }
}

/// What the generator hands over.
pub(crate) struct Output {
    pub sites: Sites,
    pub obstacles: Vec<Obstacle>,
    pub zones: ZoneRaster,
}

/// The city's shape and districts as the sites need them.
pub(crate) struct Town<'a> {
    pub centre: DVec2,
    pub radius: f64,
    pub share: &'a (dyn Fn(DVec2) -> f64 + Sync),
    pub districts: &'a Districts,
    /// Whether a point is in water.
    pub wet: &'a (dyn Fn(DVec2) -> bool + Sync),
    /// Centres and radii of the roundabouts (their islands stay empty).
    pub rings: Vec<(DVec2, f64)>,
}

const STREET: u8 = 1;
const BLOCKED: u8 = 2;

/// The raster of free, street and blocked cells.
struct Raster {
    lo: DVec2,
    cell: f64,
    nx: usize,
    ny: usize,
    state: Vec<u8>,
    /// The nearest road of street cells, and its heading there.
    road: Vec<u32>,
    heading: Vec<f32>,
}

impl Raster {
    fn centre(&self, i: usize) -> DVec2 {
        self.lo + DVec2::new((i % self.nx) as f64 + 0.5, (i / self.nx) as f64 + 0.5) * self.cell
    }

    fn index(&self, p: DVec2) -> Option<usize> {
        let q = ((p - self.lo) / self.cell).floor();
        (q.x >= 0.0 && q.y >= 0.0 && q.x < self.nx as f64 && q.y < self.ny as f64)
            .then(|| q.y as usize * self.nx + q.x as usize)
    }

    /// The 4-neighbours of cell `i`.
    fn neighbours(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let (x, y) = (i % self.nx, i / self.nx);
        [
            (x > 0).then(|| i - 1),
            (x + 1 < self.nx).then(|| i + 1),
            (y > 0).then(|| i - self.nx),
            (y + 1 < self.ny).then(|| i + self.nx),
        ]
        .into_iter()
        .flatten()
    }
}

/// A lot being planned: its raster cells, zone and street.
struct Plan {
    cells: Vec<usize>,
    zone: Zone,
}

/// Generate the sites of an urban map on the final terrain `hs`.
pub(crate) fn generate(c: &UrbanConfig, net: &RoadNetwork, hs: &Heights, town: &Town, seed: &Seed) -> Output {
    let l = &c.lots;
    let cell = l.cell;
    let n = (c.size / cell).floor() as usize;
    let lo = DVec2::splat(-0.5 * c.size);
    let reach = max_reach(net) + l.margin + cell;
    // 1. The raster.
    let mut state = vec![0u8; n * n];
    let mut road = vec![u32::MAX; n * n];
    let mut heading = vec![0f32; n * n];
    let rows = state.par_chunks_mut(n).zip(road.par_chunks_mut(n)).zip(heading.par_chunks_mut(n));
    rows.enumerate().for_each(|(y, ((srow, rrow), hrow))| {
        for x in 0..n {
            let p = lo + DVec2::new(x as f64 + 0.5, y as f64 + 0.5) * cell;
            if let Some((d, r)) = net.edge_distance(p, reach, true, None)
                && d < l.margin
            {
                srow[x] = STREET;
                rrow[x] = r;
                hrow[x] = net.roads()[r as usize].line.project(p).heading as f32;
                continue;
            }
            let edge = x < 2 || y < 2 || x + 2 >= n || y + 2 >= n;
            let d = 0.5 * cell;
            let slope = ((hs.at(p + DVec2::X * d) - hs.at(p - DVec2::X * d)).powi(2)
                + (hs.at(p + DVec2::Y * d) - hs.at(p - DVec2::Y * d)).powi(2))
            .sqrt()
                / cell;
            if edge || (town.share)(p) < 0.5 || (town.wet)(p) || slope > l.max_slope {
                srow[x] = BLOCKED;
            }
        }
    });
    let raster = Raster { lo, cell, nx: n, ny: n, state, road, heading };

    // 2. Blocks: 4-connected free cells, in scan order.
    let mut block = vec![u32::MAX; n * n];
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for start in 0..n * n {
        if raster.state[start] != 0 || block[start] != u32::MAX {
            continue;
        }
        let id = blocks.len() as u32;
        let mut cells = vec![start];
        block[start] = id;
        let mut k = 0;
        while k < cells.len() {
            let i = cells[k];
            k += 1;
            for j in raster.neighbours(i) {
                if raster.state[j] == 0 && block[j] == u32::MAX {
                    block[j] = id;
                    cells.push(j);
                }
            }
        }
        blocks.push(cells);
    }

    // 3. Zoning and lots, per block.
    let area = cell * cell;
    let plans: Vec<Vec<Plan>> = blocks
        .par_iter()
        .enumerate()
        .map(|(b, cells)| {
            let centroid = cells.iter().map(|&i| raster.centre(i)).sum::<DVec2>() / cells.len() as f64;
            let island = town.rings.iter().any(|&(c, r)| c.distance(centroid) < r);
            if (cells.len() as f64) * area < l.min_block || island || !fronts(&raster, cells, l.min_width) {
                return Vec::new();
            }
            let mut rng = seed.child("blocks").child_index(b as u64).rng();
            let zone = block_zone(l, town, centroid, &mut rng);
            let target = if zone == Zone::Park {
                f64::INFINITY
            } else {
                rng.range(l.zone(zone).lot_area[0], l.zone(zone).lot_area[1])
            };
            let mut out = Vec::new();
            split(&raster, cells.clone(), target / area, l.min_width, &mut rng, &mut out);
            out.into_iter()
                .map(|cells| {
                    let parking = matches!(zone, Zone::Downtown | Zone::Commercial | Zone::Industrial)
                        && rng.chance(l.parking_share);
                    Plan { cells, zone: if parking { Zone::Parking } else { zone } }
                })
                .collect()
        })
        .collect();
    let plans: Vec<Plan> = plans.into_iter().flatten().collect();

    // 4. Each lot's frame, building, bays or trees.
    let mut zones = vec![None; n * n];
    let mut owner = vec![u32::MAX; n * n];
    for (k, p) in plans.iter().enumerate() {
        for &i in &p.cells {
            zones[i] = Some(p.zone);
            owner[i] = k as u32;
        }
    }
    let built: Vec<LotOut> = plans
        .par_iter()
        .enumerate()
        .map(|(k, p)| {
            lot(c, net, hs, &raster, &owner, k as u32, p, &mut seed.child("lots").child_index(k as u64).rng())
        })
        .collect();
    let mut sites = Sites::default();
    let mut obstacles = Vec::new();
    for (k, b) in built.into_iter().enumerate() {
        sites.lots.push(b.lot);
        let _ = k;
        if let Some((mut building, pieces, pad)) = b.building {
            let first = obstacles.len() as u32;
            obstacles.extend(pieces);
            building.obstacles = [first, obstacles.len() as u32];
            if let Some(mut pad) = pad {
                pad.building = sites.buildings.len() as u32;
                sites.pads.push(pad);
            }
            sites.buildings.push(building);
        }
        obstacles.extend(b.extras);
        sites.bays.extend(b.bays);
    }

    // 5. Along the streets.
    furniture(c, net, hs, seed, &mut obstacles, &mut sites.bays);

    Output { sites, obstacles, zones: ZoneRaster { lo, cell, nx: n, ny: n, zones } }
}

/// Widest half road plus sidewalk of the network.
fn max_reach(net: &RoadNetwork) -> f64 {
    (0..net.roads().len())
        .map(|i| 0.5 * net.roads()[i].width + net.section(i).sidewalk[0].max(net.section(i).sidewalk[1]))
        .fold(0.0, f64::max)
}

/// The street next to free cell `i`, if any (the lowest road among its street neighbours).
fn frontage(r: &Raster, i: usize) -> Option<u32> {
    r.neighbours(i).filter(|&j| r.state[j] == STREET).map(|j| r.road[j]).min()
}

/// The zone of a block with its centroid at `p`: downtown in the core of the downtown grid,
/// commercial around it and in some blocks of the inner districts, industrial in some outer
/// grid blocks, residential elsewhere; any may be a park.
fn block_zone(l: &LotsConfig, town: &Town, p: DVec2, rng: &mut SimRng) -> Zone {
    let (u_park, u_zone) = (rng.uniform(), rng.uniform());
    if u_park < l.park_share {
        return Zone::Park;
    }
    let r = (p - town.centre).length() / town.radius;
    match town.districts.kinds[town.districts.at(p)] {
        super::DistrictKind::Downtown if r < 0.3 => Zone::Downtown,
        super::DistrictKind::Downtown => Zone::Commercial,
        super::DistrictKind::Grid if r > 0.5 && u_zone < l.industrial_share => Zone::Industrial,
        _ if r < 0.55 && u_zone < l.commercial_share => Zone::Commercial,
        _ => Zone::Residential,
    }
}

/// Split `cells` into lots of about `target` cells (see the module docs), none narrower than
/// `width` (m) across its cut.
fn split(r: &Raster, cells: Vec<usize>, target: f64, width: f64, rng: &mut SimRng, out: &mut Vec<Vec<usize>>) {
    let count = cells.len() as f64;
    if count <= 1.5 * target {
        out.push(cells);
        return;
    }
    // The bounding box of least area among those aligned with the streets along the piece.
    let pts: Vec<DVec2> = cells.iter().map(|&i| r.centre(i)).collect();
    let mut angles: Vec<f64> = cells
        .iter()
        .flat_map(|&i| r.neighbours(i).filter(|&j| r.state[j] == STREET).map(|j| f64::from(r.heading[j])))
        .map(|a| (a.rem_euclid(PI) / 0.05).round() * 0.05)
        .collect();
    angles.sort_by(f64::total_cmp);
    angles.dedup();
    let extent = |a: f64| {
        let u = DVec2::new(libm::cos(a), libm::sin(a));
        let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
        for p in &pts {
            let q = DVec2::new(p.dot(u), p.perp_dot(u));
            lo = lo.min(q);
            hi = hi.max(q);
        }
        hi - lo
    };
    let Some(a) = angles
        .into_iter()
        .map(|a| (a, extent(a)))
        .min_by(|x, y| (x.1.x * x.1.y).total_cmp(&(y.1.x * y.1.y)).then(x.0.total_cmp(&y.0)))
    else {
        out.push(cells);
        return;
    };
    // Cut across the longer side first.
    let axes = if a.1.x >= a.1.y { [a.0, a.0 + FRAC_PI_2] } else { [a.0 + FRAC_PI_2, a.0] };
    let f = rng.range(0.4, 0.6);
    for axis in axes {
        let u = DVec2::new(libm::cos(axis), libm::sin(axis));
        let t: Vec<f64> = pts.iter().map(|p| p.dot(u)).collect();
        let mut sorted = t.clone();
        let k = ((t.len() as f64 * f) as usize).min(t.len() - 1);
        let (_, cut, _) = sorted.select_nth_unstable_by(k, f64::total_cmp);
        let cut = *cut;
        let (t0, t1) = t.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |m, &v| (m.0.min(v), m.1.max(v)));
        if cut - t0 < width || t1 + r.cell - cut < width {
            continue;
        }
        let (lo, hi): (Vec<usize>, Vec<usize>) = cells.iter().zip(&t).partition(|(_, v)| **v < cut).into_split();
        let min = 0.35 * target;
        if (lo.len() as f64) < min || (hi.len() as f64) < min {
            continue;
        }
        if !(fronts(r, &lo, width) && fronts(r, &hi, width) && connected(r, &lo) && connected(r, &hi)) {
            continue;
        }
        split(r, lo, target, width, rng, out);
        split(r, hi, target, width, rng, out);
        return;
    }
    out.push(cells);
}

trait IntoSplit {
    fn into_split(self) -> (Vec<usize>, Vec<usize>);
}

impl IntoSplit for (Vec<(&usize, &f64)>, Vec<(&usize, &f64)>) {
    fn into_split(self) -> (Vec<usize>, Vec<usize>) {
        (self.0.into_iter().map(|(i, _)| *i).collect(), self.1.into_iter().map(|(i, _)| *i).collect())
    }
}

/// Whether `cells` front a street along at least half of `width`.
fn fronts(r: &Raster, cells: &[usize], width: f64) -> bool {
    let n = cells.iter().filter(|&&i| frontage(r, i).is_some()).count();
    n as f64 * r.cell >= 0.5 * width
}

/// Whether `cells` (sorted or not) form one 4-connected piece.
fn connected(r: &Raster, cells: &[usize]) -> bool {
    let mut sorted = cells.to_vec();
    sorted.sort_unstable();
    let inside = |i: usize| sorted.binary_search(&i).is_ok();
    let mut seen = vec![false; sorted.len()];
    let mut stack = vec![0usize];
    seen[0] = true;
    let mut count = 1;
    while let Some(k) = stack.pop() {
        for j in r.neighbours(sorted[k]) {
            if inside(j) {
                let m = sorted.binary_search(&j).expect("inside");
                if !seen[m] {
                    seen[m] = true;
                    count += 1;
                    stack.push(m);
                }
            }
        }
    }
    count == sorted.len()
}

/// A lot with what stands on it.
struct LotOut {
    lot: Lot,
    building: Option<(Building, Vec<Obstacle>, Option<Pad>)>,
    extras: Vec<Obstacle>,
    bays: Vec<ParkingBay>,
}

/// A rectangle in a lot's frame: `[x0, x1] × [y0, y1]` (m).
#[derive(Clone, Copy, Debug)]
struct Rect {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
}

impl Rect {
    fn w(&self) -> f64 {
        self.x1 - self.x0
    }

    fn d(&self) -> f64 {
        self.y1 - self.y0
    }
}

/// A lot's frame: origin, x along the street, y away from it.
#[derive(Clone, Copy)]
struct Frame {
    o: DVec2,
    x: DVec2,
    y: DVec2,
}

impl Frame {
    fn at(&self, x: f64, y: f64) -> DVec2 {
        self.o + self.x * x + self.y * y
    }

    fn yaw(&self) -> f64 {
        libm::atan2(self.x.y, self.x.x)
    }
}

#[allow(clippy::too_many_arguments)]
fn lot(
    c: &UrbanConfig,
    net: &RoadNetwork,
    hs: &Heights,
    r: &Raster,
    owner: &[u32],
    id: u32,
    plan: &Plan,
    rng: &mut SimRng,
) -> LotOut {
    let l = &c.lots;
    let count = plan.cells.len() as f64;
    let centre = plan.cells.iter().map(|&i| r.centre(i)).sum::<DVec2>() / count;
    // The street it fronts most, and the middle of that frontage.
    let mut fr: Vec<(u32, DVec2)> =
        plan.cells.iter().filter_map(|&i| frontage(r, i).map(|k| (k, r.centre(i)))).collect();
    fr.sort_by_key(|f| f.0);
    let mut best = (0usize, u32::MAX);
    let mut k = 0;
    while k < fr.len() {
        let j = fr[k..].iter().take_while(|f| f.0 == fr[k].0).count();
        if j > best.0 {
            best = (j, fr[k].0);
        }
        k += j;
    }
    let street = best.1;
    let mean = fr.iter().filter(|f| f.0 == street).map(|f| f.1).sum::<DVec2>() / best.0 as f64;
    let front = fr
        .iter()
        .filter(|f| f.0 == street)
        .map(|f| f.1)
        .min_by(|a, b| a.distance_squared(mean).total_cmp(&b.distance_squared(mean)))
        .expect("frontage");
    let lot = Lot { zone: plan.zone, centre, area: count * r.cell * r.cell, road: street, front };
    let mut out = LotOut { lot, building: None, extras: Vec::new(), bays: Vec::new() };
    // Frame: y from the street towards the lot.
    let proj = net.roads()[street as usize].line.project(front);
    let along = DVec2::from_angle(proj.heading);
    let side = (centre - proj.point.truncate()).perp_dot(along);
    let y = if side <= 0.0 { along.perp() } else { -along.perp() };
    let frame = Frame { o: front, x: DVec2::new(y.y, -y.x), y };
    let Some((rect, inside)) = largest_rect(r, owner, id, plan, &frame) else { return out };
    let ground = |p: DVec2| hs.at(p);
    match plan.zone {
        Zone::Park => {
            // Trees on a jittered 12 m grid, 3 m inside the lot.
            let mut yv = rect.y0 + 4.0;
            while yv < rect.y1 - 3.0 {
                let mut xv = rect.x0 + 4.0;
                while xv < rect.x1 - 3.0 {
                    let (jx, jy, h) = (rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(9.0, 15.0));
                    let (px, py) = (xv + jx, yv + jy);
                    if inside(px, py, 3.0) {
                        let p = frame.at(px, py);
                        out.extras.extend(broadleaf(p.extend(ground(p) - 0.2), h));
                    }
                    xv += 12.0;
                }
                yv += 12.0;
            }
        }
        Zone::Parking => {
            // Rows of 2.5 × 5 m bays either side of 6 m aisles, nose in.
            let (bw, bl, aisle) = (2.5, 5.0, 6.0);
            let mut y0 = rect.y0 + 1.0;
            let mut rows = Vec::new();
            while y0 + bl + aisle <= rect.y1 - 1.0 {
                rows.push((y0 + 0.5 * bl, -1.0));
                if y0 + 2.0 * bl + aisle <= rect.y1 - 1.0 {
                    rows.push((y0 + 1.5 * bl + aisle, 1.0));
                }
                y0 += 2.0 * bl + aisle;
            }
            for (yc, nose) in rows {
                let mut x = rect.x0 + 1.0;
                while x + bw <= rect.x1 - 1.0 {
                    let p = frame.at(x + 0.5 * bw, yc);
                    out.bays.push(ParkingBay {
                        centre: p.extend(ground(p)),
                        yaw: frame.yaw() + nose * FRAC_PI_2,
                        size: DVec2::new(bl, bw),
                        kind: BayKind::Lot,
                    });
                    x += bw;
                }
            }
        }
        zone => {
            let z = l.zone(zone);
            let mut b =
                Rect { x0: rect.x0 + z.side, x1: rect.x1 - z.side, y0: rect.y0 + z.front, y1: rect.y1 - z.side };
            if b.w() > z.max_size[0] {
                let mid = 0.5 * (b.x0 + b.x1);
                b.x0 = mid - 0.5 * z.max_size[0];
                b.x1 = mid + 0.5 * z.max_size[0];
            }
            b.y1 = b.y1.min(b.y0 + z.max_size[1]);
            if zone == Zone::Residential {
                // A house of its own size near the front.
                let (w, d) = (rng.range(9.0, z.max_size[0]).min(b.w()), rng.range(8.0, z.max_size[1]).min(b.d()));
                let mid = 0.5 * (b.x0 + b.x1);
                b = Rect { x0: mid - 0.5 * w, x1: mid + 0.5 * w, y0: b.y0, y1: b.y0 + d };
                // A wall or fence along the street, with a gap for the drive.
                if rng.chance(l.fence_share) {
                    let wall = rng.chance(0.5);
                    let yf = rect.y0 + 0.3;
                    for (a, e) in [(rect.x0 + 0.3, mid - 1.75), (mid + 1.75, rect.x1 - 0.3)] {
                        if e - a < 1.0 {
                            continue;
                        }
                        let p = frame.at(0.5 * (a + e), yf);
                        let (thick, h, tag, mat) = if wall {
                            (0.25, 0.9, tags::WALL, MaterialId::CONCRETE)
                        } else {
                            (0.1, 1.1, tags::FENCE, MaterialId::WOOD)
                        };
                        let shape = ObstacleShape::Cuboid {
                            half_extents: DVec3::new(0.5 * (e - a), 0.5 * thick, 0.5 * h + 0.2),
                        };
                        let pose = Pose::new(p.extend(ground(p) + 0.5 * h - 0.2), DQuat::from_rotation_z(frame.yaw()));
                        out.extras.push(Obstacle::solid(shape, pose, mat).with_tag(tag));
                    }
                }
                // A tree in the back garden.
                let back = rect.y1 - b.y1;
                if back >= 9.0 {
                    let (px, py) = (0.5 * (b.x0 + b.x1) + rng.range(-2.0, 2.0), b.y1 + 0.5 * back);
                    if inside(px, py, 2.5) {
                        let p = frame.at(px, py);
                        out.extras.extend(broadleaf(p.extend(ground(p) - 0.2), rng.range(8.0, 12.0)));
                    }
                }
            }
            if b.w() >= 6.0 && b.d() >= 6.0 {
                out.building = Some(building(l, zone, id, &frame, b, hs, rng));
            }
        }
    }
    out
}

/// The largest rectangle of the lot in `frame` on a 1 m raster whose cells lie wholly in the
/// lot, and a test of whether a disc (frame coordinates, radius) lies in the lot.
#[allow(clippy::type_complexity)]
fn largest_rect<'a>(
    r: &'a Raster,
    owner: &'a [u32],
    id: u32,
    plan: &Plan,
    frame: &'a Frame,
) -> Option<(Rect, Box<dyn Fn(f64, f64, f64) -> bool + 'a>)> {
    let h = 0.5 * r.cell;
    let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
    for &i in &plan.cells {
        let c = r.centre(i);
        for (dx, dy) in [(-h, -h), (h, -h), (-h, h), (h, h)] {
            let q = c + DVec2::new(dx, dy) - frame.o;
            let l = DVec2::new(q.dot(frame.x), q.dot(frame.y));
            lo = lo.min(l);
            hi = hi.max(l);
        }
    }
    let ours = move |p: DVec2| r.index(p).is_some_and(|i| owner[i] == id);
    let (x0, y0) = (lo.x.floor(), lo.y.floor());
    let (w, d) = ((hi.x.ceil() - x0) as usize, (hi.y.ceil() - y0) as usize);
    if w == 0 || d == 0 {
        return None;
    }
    // Corners of the 1 m cells, then cells with all four corners in the lot.
    let corner: Vec<bool> = (0..=d)
        .flat_map(|j| (0..=w).map(move |i| (i, j)))
        .map(|(i, j)| ours(frame.at(x0 + i as f64, y0 + j as f64)))
        .collect();
    let inside: Vec<bool> = (0..d)
        .flat_map(|j| (0..w).map(move |i| (i, j)))
        .map(|(i, j)| {
            let c = |a: usize, b: usize| corner[b * (w + 1) + a];
            c(i, j) && c(i + 1, j) && c(i, j + 1) && c(i + 1, j + 1)
        })
        .collect();
    // Largest rectangle of true cells (histograms per row, a stack per row).
    let mut heights = vec![0usize; w];
    let mut best = (0usize, 0usize, 0usize, 0usize, 0usize); // area, i0, i1, j0, j1 (exclusive)
    for j in 0..d {
        for i in 0..w {
            heights[i] = if inside[j * w + i] { heights[i] + 1 } else { 0 };
        }
        let mut stack: Vec<usize> = Vec::new();
        for i in 0..=w {
            let hcur = if i < w { heights[i] } else { 0 };
            while let Some(&top) = stack.last() {
                if heights[top] < hcur {
                    break;
                }
                stack.pop();
                let height = heights[top];
                let left = stack.last().map_or(0, |&s| s + 1);
                let area = height * (i - left);
                if area > best.0 {
                    best = (area, left, i, j + 1 - height, j + 1);
                }
            }
            stack.push(i);
        }
    }
    if best.0 == 0 {
        return None;
    }
    let rect = Rect { x0: x0 + best.1 as f64, x1: x0 + best.2 as f64, y0: y0 + best.3 as f64, y1: y0 + best.4 as f64 };
    let test = move |x: f64, y: f64, rad: f64| {
        (0..8).all(|k| {
            let a = PI * k as f64 / 4.0;
            ours(frame.at(x + rad * libm::cos(a), y + rad * libm::sin(a)))
        }) && ours(frame.at(x, y))
    };
    Some((rect, Box::new(test)))
}

/// A building on footprint `b` (lot frame): its record, collision pieces and pad.
fn building(
    l: &LotsConfig,
    zone: Zone,
    lot: u32,
    frame: &Frame,
    b: Rect,
    hs: &Heights,
    rng: &mut SimRng,
) -> (Building, Vec<Obstacle>, Option<Pad>) {
    let z = l.zone(zone);
    let (w, d) = (b.w(), b.d());
    let centre = frame.at(0.5 * (b.x0 + b.x1), 0.5 * (b.y0 + b.y1));
    let yaw = frame.yaw();
    let rot = DQuat::from_rotation_z(yaw);
    let local = |x: f64, y: f64| centre + frame.x * x + frame.y * y;
    let samples =
        [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0)].map(|(a, c)| hs.at(local(a * w, c * d)));
    let low = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let high = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = f64::from(z.storeys[1] - z.storeys[0]);
    // Towers skew towards the low end.
    let u = rng.uniform();
    let storeys = z.storeys[0] + (span * if zone == Zone::Downtown { u * u } else { u } + 0.5) as u8;
    let storeys = storeys.min(z.storeys[1]);
    let base = low - 0.5;
    let top = high + f64::from(storeys) * l.storey_height;
    let material = match zone {
        Zone::Industrial => MaterialId::METAL,
        _ => MaterialId::CONCRETE,
    };
    // Pieces (centre x, y and size along x, y in the footprint frame).
    let shape = rng.uniform();
    let mut pieces: Vec<(f64, f64, f64, f64)> = Vec::new();
    let big = w >= 20.0 && d >= 20.0 && zone != Zone::Residential;
    let t = (0.3 * w.min(d)).min(12.0);
    if big && w >= 30.0 && d >= 30.0 && shape < 0.3 {
        // Courtyard.
        pieces.push((0.0, -0.5 * d + 0.5 * t, w, t));
        pieces.push((0.0, 0.5 * d - 0.5 * t, w, t));
        pieces.push((-0.5 * w + 0.5 * t, 0.0, t, d - 2.0 * t));
        pieces.push((0.5 * w - 0.5 * t, 0.0, t, d - 2.0 * t));
    } else if big && shape < 0.5 {
        // U, open at the back.
        pieces.push((0.0, -0.5 * d + 0.5 * t, w, t));
        pieces.push((-0.5 * w + 0.5 * t, 0.5 * t, t, d - t));
        pieces.push((0.5 * w - 0.5 * t, 0.5 * t, t, d - t));
    } else if big && shape < 0.7 {
        // L: the front and one side.
        let s = if rng.chance(0.5) { -1.0 } else { 1.0 };
        pieces.push((0.0, -0.5 * d + 0.5 * t, w, t));
        pieces.push((s * (0.5 * w - 0.5 * t), 0.5 * t, t, d - t));
    } else {
        pieces.push((0.0, 0.0, w, d));
    }
    let rectangular = pieces.len() == 1;
    let hz = 0.5 * (top - base);
    let mut obstacles: Vec<Obstacle> = pieces
        .iter()
        .map(|&(x, y, sx, sy)| {
            let p = local(x, y).extend(base + hz);
            let shape = ObstacleShape::Cuboid { half_extents: DVec3::new(0.5 * sx, 0.5 * sy, hz) };
            Obstacle::solid(shape, Pose::new(p, rot), material).with_tag(tags::BLOCK)
        })
        .collect();
    let gable = rectangular && rng.chance(z.pitched_share);
    let mut pad = None;
    if gable {
        // Ridge along the longer side, rising 0.35 of the shorter.
        let (hw, hd) = (0.5 * w, 0.5 * d);
        let rise = 0.35 * w.min(d);
        let mut points: Vec<DVec3> =
            [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)].map(|(x, y)| DVec3::new(x, y, 0.0)).to_vec();
        if w >= d {
            points.extend([DVec3::new(-hw, 0.0, rise), DVec3::new(hw, 0.0, rise)]);
        } else {
            points.extend([DVec3::new(0.0, -hd, rise), DVec3::new(0.0, hd, rise)]);
        }
        let pose = Pose::new(centre.extend(top), rot);
        obstacles.push(
            Obstacle::solid(ObstacleShape::ConvexHull { points }, pose, MaterialId::CONCRETE).with_tag(tags::ROOF),
        );
    } else if rectangular {
        // Parapet on taller flat roofs, a pad on some, roof units along the edges.
        if storeys >= 3 {
            let (th, ph) = (0.3, 1.0);
            for (x, y, sx, sy) in [
                (0.0, -0.5 * d + 0.5 * th, w, th),
                (0.0, 0.5 * d - 0.5 * th, w, th),
                (-0.5 * w + 0.5 * th, 0.0, th, d - 2.0 * th),
                (0.5 * w - 0.5 * th, 0.0, th, d - 2.0 * th),
            ] {
                let p = local(x, y).extend(top + 0.5 * ph);
                let shape = ObstacleShape::Cuboid { half_extents: DVec3::new(0.5 * sx, 0.5 * sy, 0.5 * ph) };
                obstacles.push(Obstacle::solid(shape, Pose::new(p, rot), material).with_tag(tags::PARAPET));
            }
        }
        let half = 4.0;
        if w >= 14.0 && d >= 14.0 && rng.chance(z.pad_share) {
            let slab = 0.1;
            let shape = ObstacleShape::Cuboid { half_extents: DVec3::new(half, half, 0.5 * slab) };
            let p = centre.extend(top + 0.5 * slab);
            obstacles.push(Obstacle::solid(shape, Pose::new(p, rot), MaterialId::CONCRETE).with_tag(tags::PAD));
            pad = Some(Pad { centre: centre.extend(top + slab), yaw, half, building: 0 });
        }
        if w >= 10.0 && d >= 10.0 {
            for k in 0..rng.below(3) {
                let (ux, uy, uh) = (rng.range(0.8, 1.5), rng.range(0.8, 1.5), rng.range(0.6, 1.0));
                let sx = if k % 2 == 0 { -1.0 } else { 1.0 };
                let x = sx * (0.5 * w - 1.5 - ux);
                let y = rng.range(-0.5 * d + 1.5 + uy, 0.5 * d - 1.5 - uy);
                if pad.is_some() && x.abs() < half + ux + 0.5 && y.abs() < half + uy + 0.5 {
                    continue;
                }
                let p = local(x, y).extend(top + uh);
                let shape = ObstacleShape::Cuboid { half_extents: DVec3::new(ux, uy, uh) };
                obstacles.push(Obstacle::solid(shape, Pose::new(p, rot), MaterialId::METAL).with_tag(tags::ROOF_UNIT));
            }
        }
    }
    let record = Building {
        lot,
        centre,
        yaw,
        size: DVec2::new(w, d),
        base,
        height: top - base,
        storeys,
        roof: if gable { Roof::Gable } else { Roof::Flat },
        obstacles: [0, 0],
    };
    (record, obstacles, pad)
}

/// Street trees, lamp posts, signal poles and on-street parking bays.
fn furniture(
    c: &UrbanConfig,
    net: &RoadNetwork,
    hs: &Heights,
    seed: &Seed,
    obstacles: &mut Vec<Obstacle>,
    bays: &mut Vec<ParkingBay>,
) {
    let f = &c.furniture;
    let g = net.lanes();
    let reach = max_reach(net) + 2.0;
    // Clear of every other road's carriageway by `d`.
    let clear =
        |p: DVec2, own: usize, d: f64| net.edge_distance(p, reach, false, Some(own as u32)).is_none_or(|e| e.0 > d);
    for (i, road) in net.roads().iter().enumerate() {
        if !road.class.is_urban() {
            continue;
        }
        let s = net.section(i);
        let len = road.line.length();
        let [sa, sb] = g.setbacks(i);
        let (from, to) = (sa + 6.0, len - sb - 6.0);
        let half = 0.5 * road.width;
        let mut rng = seed.child("furniture").child_index(i as u64).rng();
        let at = |st: f64, left: f64| {
            let h = road.line.heading_at(st);
            road.line.point_at(st).truncate() + DVec2::new(-libm::sin(h), libm::cos(h)) * left
        };
        // Sides: 0 right of start → end (offset negative), 1 left.
        let sign = [-1.0, 1.0];
        if s.sidewalk[0].min(s.sidewalk[1]) >= 2.5 && rng.chance(f.street_trees) {
            let mut st = from + 0.5 * f.tree_spacing;
            while st < to {
                for side in 0..2 {
                    let p = at(st, sign[side] * (half + 1.2));
                    if clear(p, i, 1.0) {
                        let h = rng.range(8.0, 12.0);
                        obstacles.extend(broadleaf(p.extend(hs.at(p) - 0.2), h));
                    }
                }
                st += f.tree_spacing;
            }
        }
        if s.sidewalk[0].min(s.sidewalk[1]) >= 1.5 {
            let mut st = from;
            let mut side = 0;
            while st < to {
                let p = at(st, sign[side] * (half + 0.4));
                if clear(p, i, 0.5) {
                    let hh = 3.5;
                    let shape = ObstacleShape::Cylinder { half_height: hh, radius: 0.08 };
                    obstacles.push(
                        Obstacle::solid(
                            shape,
                            Pose::from_translation(p.extend(hs.at(p) - 0.2 + hh)),
                            MaterialId::METAL,
                        )
                        .with_tag(tags::LAMP),
                    );
                }
                side = 1 - side;
                st += f.lamp_spacing;
            }
        }
        // Parking lanes: side 0 on the right of travel from start to end, side 1 on the left
        // (one-way streets have all their parking on the right).
        let parking = if s.one_way() { [s.parking[0] + s.parking[1], 0.0] } else { s.parking };
        for side in 0..2 {
            let w = parking[side];
            if w <= 0.0 {
                continue;
            }
            let left = sign[side] * (half - 0.5 * w);
            let mut st = from + 0.5 * f.bay_length;
            while st + 0.5 * f.bay_length <= to {
                // Along the chord of the parking lane's centre over the bay, kept only where
                // the road network places every point of the bay (5 cm in from its edges, on a
                // half-metre grid) in a parking lane.
                let (a, b) = (at(st - 0.5 * f.bay_length, left), at(st + 0.5 * f.bay_length, left));
                let yaw = (b - a).to_angle() + if side == 0 { 0.0 } else { PI };
                let centre = 0.5 * (a + b);
                let size = DVec2::new(f.bay_length, w);
                let bay =
                    ParkingBay { centre: centre.extend(road.line.point_at(st).z), yaw, size, kind: BayKind::Street };
                let (nx, ny) = ((2.0 * size.x).ceil() as usize, (2.0 * size.y).ceil() as usize);
                let inner = size - DVec2::splat(0.1);
                let mut grid = (0..=nx).flat_map(|i| (0..=ny).map(move |j| (i, j))).map(|(i, j)| {
                    bay.point(DVec2::new(i as f64 / nx as f64 - 0.5, j as f64 / ny as f64 - 0.5) * inner)
                });
                let curvy = road.line.curvature_at(st).abs() > 1.0 / 40.0;
                if !curvy
                    && bay.corners().iter().all(|&q| clear(q, i, 0.3))
                    && grid.all(|q| net.area(q) == Area::Parking)
                {
                    bays.push(bay);
                }
                st += f.bay_length + f.bay_gap;
            }
        }
    }
    // Signal poles at the kerb beside each stop line of a signalled junction.
    for j in g.junctions() {
        if j.kind != JunctionKind::Signal {
            continue;
        }
        for a in &j.approaches {
            let road = &net.roads()[a.road as usize];
            let mid = 0.5 * (a.stop_line[0] + a.stop_line[1]).truncate();
            let pr = road.line.project(mid);
            let right = if a.dir == 0 { -1.0 } else { 1.0 };
            let n = DVec2::new(-libm::sin(pr.heading), libm::cos(pr.heading));
            let p = pr.point.truncate() + n * right * (0.5 * road.width + 0.6);
            let z = hs.at(p) - 0.2;
            let hh = 1.8;
            let pole = ObstacleShape::Cylinder { half_height: hh, radius: 0.1 };
            obstacles.push(
                Obstacle::solid(pole, Pose::from_translation(p.extend(z + hh)), MaterialId::METAL)
                    .with_tag(tags::SIGNAL),
            );
            let head = ObstacleShape::Cuboid { half_extents: DVec3::new(0.15, 0.15, 0.45) };
            let yaw = if a.dir == 0 { pr.heading + PI } else { pr.heading };
            obstacles.push(
                Obstacle::solid(
                    head,
                    Pose::new(p.extend(z + 2.0 * hh + 0.45), DQuat::from_rotation_z(yaw)),
                    MaterialId::METAL,
                )
                .with_tag(tags::SIGNAL),
            );
        }
    }
}
