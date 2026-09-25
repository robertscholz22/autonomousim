//! Rural maps: gentle farmland with a road network, farms and fields.
//!
//! Pipeline:
//! 1. The wild landform (terrain noise, erosion, upsampling, lakes) with gentle relief.
//! 2. Farm sites: flat, dry, low places, at least `farms.spacing` apart; field parcels (the
//!    Voronoi cells of a jittered grid), each meadow, crop, plowed soil or woods.
//! 3. Roads, routed by A* on a coarse grid (length, grade, side slope and turning cost; water
//!    and its surroundings forbidden, other farms avoided): a paved main road across the map,
//!    then a gravel road from each farm and a dirt track to each field far from the roads,
//!    each to the nearest road built so far.
//! 4. The routed paths, split at junctions, become centripetal Catmull–Rom splines resampled
//!    every metre, smoothed until they respect the class's minimum radius.
//! 5. Road profiles: terrain heights along each road, smoothed, pinned to the node heights
//!    (level with a yard across its pad) and limited to the class's maximum grade.
//! 6. Terrain blending: the road surfaces (with a crown) and the farm yards are cut or filled
//!    into the terrain, with shoulders falling off smoothly.
//! 7. Materials per cell: roads (asphalt, gravel, dirt), yards (concrete), lake beds and
//!    shores, rock on steep slopes, marsh, grass verges and headlands, and the parcels' crops.
//! 8. Obstacles: farm buildings, hedges and fences along parcel edges, tree lines along some
//!    roads, woods and single trees; none on a road (below the headroom) or in a yard.
//!
//! As for wild maps, the result depends only on the configuration and the seed.

use crate::ProcgenError;
use crate::farmland::{self, Clearance, FieldsConfig, ParcelKind, Parcels, ScatterConfig};
use crate::noise::smoothstep;
use crate::scatter::{Ground, trees};
use crate::terrain::{ErosionConfig, TerrainConfig};
use crate::wild::{Land, WaterConfig, landform, merge, moisture, nearby_water};
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::rng::Seed;
use autonomousim_world::obstacles::{ObstacleClass, tags};
use autonomousim_world::{
    GeoOrigin, HeightGrid, MapMeta, NodeKind, ObstacleSet, Polyline, Road, RoadClass, RoadNetwork, RoadNode,
    StaticWorld,
};
use glam::DVec2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::time::Instant;

/// Bumped whenever the output for a given configuration and seed changes.
pub const RURAL_VERSION: u32 = 4;

/// Geometry limits of one road class.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassConfig {
    /// Full width of the road surface (m).
    pub width: f64,
    /// Maximum longitudinal grade (rise over run).
    pub max_grade: f64,
    /// Minimum horizontal curve radius (m), away from junctions.
    pub min_radius: f64,
    /// Cross slope from the centre line to the edges (rise over run).
    pub crown: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RoadsConfig {
    pub paved: ClassConfig,
    pub gravel: ClassConfig,
    pub track: ClassConfig,
    /// Cell of the routing grid (m); a multiple of the map cell.
    pub route_cell: f64,
    /// Routes keep this far (m) from water, beyond half the road width.
    pub water_clearance: f64,
    /// Cost per metre of `grade²`, of `side slope²`, and per squared radian of turning.
    pub grade_cost: f64,
    pub side_slope_cost: f64,
    pub turn_cost: f64,
    /// Minimum width (m) of the shoulders that blend the road into the terrain; wider where
    /// the cut or fill is deep (1.5 m per metre of height difference).
    pub shoulder: f64,
    /// Window (m) of the moving average over the terrain heights along a road.
    pub profile_window: f64,
}

impl Default for RoadsConfig {
    fn default() -> Self {
        Self {
            paved: ClassConfig { width: 6.0, max_grade: 0.08, min_radius: 30.0, crown: 0.02 },
            gravel: ClassConfig { width: 4.0, max_grade: 0.12, min_radius: 15.0, crown: 0.02 },
            track: ClassConfig { width: 3.0, max_grade: 0.2, min_radius: 8.0, crown: 0.0 },
            route_cell: 4.0,
            water_clearance: 8.0,
            grade_cost: 60.0,
            side_slope_cost: 10.0,
            turn_cost: 3.0,
            shoulder: 3.0,
            profile_window: 20.0,
        }
    }
}

impl RoadsConfig {
    pub fn class(&self, class: RoadClass) -> &ClassConfig {
        match class {
            RoadClass::Paved => &self.paved,
            RoadClass::Gravel => &self.gravel,
            RoadClass::Track => &self.track,
        }
    }

    fn validate(&self, cell: f64) -> Result<(), String> {
        for (name, c) in [("paved", &self.paved), ("gravel", &self.gravel), ("track", &self.track)] {
            if !(c.width > 0.0 && c.max_grade > 0.0 && c.min_radius > 0.0 && c.crown >= 0.0) {
                return Err(format!("roads.{name}: width, max_grade and min_radius must be positive"));
            }
        }
        let k = self.route_cell / cell;
        if !(k >= 1.0 && k.fract() == 0.0) {
            return Err("roads.route_cell must be a whole multiple of the map cell".into());
        }
        if !(self.water_clearance >= 0.0 && self.shoulder > 0.0 && self.profile_window >= 0.0) {
            return Err("roads: water_clearance and profile_window must be ≥ 0, shoulder > 0".into());
        }
        if !(self.grade_cost >= 0.0 && self.side_slope_cost >= 0.0 && self.turn_cost >= 0.0) {
            return Err("roads: costs must be ≥ 0".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FarmsConfig {
    /// Minimum distance between farms (m).
    pub spacing: f64,
    /// Half extents of a farm yard (m).
    pub yard: [f64; 2],
    /// Largest height difference across a yard site (m) before it is flattened.
    pub max_relief: f64,
    /// Farms keep this far from water (m) and from the map edge.
    pub water_clearance: f64,
    pub margin: f64,
    /// Ground levelled around the yard for the buildings (m beyond the yard).
    pub pad: f64,
}

impl Default for FarmsConfig {
    fn default() -> Self {
        Self { spacing: 160.0, yard: [20.0, 15.0], max_relief: 5.0, water_clearance: 30.0, margin: 60.0, pad: 14.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuralMaterialsConfig {
    /// Bare rock above this slope.
    pub rock_slope_deg: f64,
    /// Sand up to this distance (m) from and height (m) above a lake.
    pub shore_width: f64,
    pub shore_height: f64,
    /// Mud on flat ground wetter than this (moisture in [0, 1]).
    pub marsh_moisture: f64,
}

impl Default for RuralMaterialsConfig {
    fn default() -> Self {
        Self { rock_slope_deg: 35.0, shore_width: 3.0, shore_height: 0.6, marsh_moisture: 0.85 }
    }
}

/// Everything that shapes a rural map (the seed is separate).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuralConfig {
    /// Edge length of the square map (m), centred on the origin.
    pub size: f64,
    /// Final grid cell (m).
    pub cell: f64,
    pub geo_origin: GeoOrigin,
    pub terrain: TerrainConfig,
    pub erosion: ErosionConfig,
    pub water: WaterConfig,
    pub farms: FarmsConfig,
    pub roads: RoadsConfig,
    pub fields: FieldsConfig,
    pub materials: RuralMaterialsConfig,
    pub scatter: ScatterConfig,
}

impl Default for RuralConfig {
    fn default() -> Self {
        Self::training()
    }
}

/// Named starting points for [`RuralConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuralPreset {
    /// 512 m of farmland: a main road and a few farms, for training pools.
    Training,
    /// 2 km of farmland with a few dozen farms.
    Showcase,
}

impl RuralPreset {
    pub fn config(self) -> RuralConfig {
        match self {
            Self::Training => RuralConfig::training(),
            Self::Showcase => RuralConfig::showcase(),
        }
    }
}

impl std::str::FromStr for RuralPreset {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "training" => Ok(Self::Training),
            "showcase" => Ok(Self::Showcase),
            _ => Err(format!("unknown preset {s:?} (training, showcase)")),
        }
    }
}

impl RuralConfig {
    /// Gentle relief: broad hills, hardly any mountains, little fine detail.
    fn terrain() -> TerrainConfig {
        TerrainConfig {
            relief: 100.0,
            hills_wavelength: 600.0,
            hills_amplitude: 0.35,
            mountains_wavelength: 1200.0,
            mountains_amplitude: 0.4,
            mask_wavelength: 1200.0,
            mask_threshold: 0.5,
            warp_wavelength: 500.0,
            warp_strength: 40.0,
            detail_amplitude: 0.05,
            ..TerrainConfig::default()
        }
    }

    pub fn training() -> Self {
        Self {
            size: 512.0,
            cell: 1.0,
            geo_origin: GeoOrigin::default(),
            terrain: Self::terrain(),
            erosion: ErosionConfig::default(),
            water: WaterConfig::default(),
            farms: FarmsConfig::default(),
            roads: RoadsConfig::default(),
            fields: FieldsConfig::default(),
            materials: RuralMaterialsConfig::default(),
            scatter: ScatterConfig::default(),
        }
    }

    pub fn showcase() -> Self {
        Self {
            size: 2048.0,
            terrain: TerrainConfig { relief: 120.0, ..Self::terrain() },
            farms: FarmsConfig { spacing: 260.0, ..FarmsConfig::default() },
            fields: FieldsConfig { spacing: 150.0, ..FieldsConfig::default() },
            ..Self::training()
        }
    }

    /// `preset` with `overrides` merged on top (as [`WildConfig::from_preset`](crate::WildConfig::from_preset)).
    pub fn from_preset(preset: RuralPreset, overrides: Option<&serde_json::Value>) -> Result<Self, ProcgenError> {
        let config = preset.config();
        let Some(overrides) = overrides else { return Ok(config) };
        let mut value = serde_json::to_value(&config).expect("configs serialise to JSON");
        merge(&mut value, overrides);
        let config: Self = serde_json::from_value(value).map_err(|e| ProcgenError::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Vertices along each edge of the final grid.
    pub fn vertices(&self) -> usize {
        (self.size / self.cell).round() as usize + 1
    }

    pub fn validate(&self) -> Result<(), ProcgenError> {
        let check = || -> Result<(), String> {
            if !(0.25..=8.0).contains(&self.cell) {
                return Err("cell must be in [0.25, 8] m".into());
            }
            let cells = self.size / self.cell;
            if !(cells.fract() == 0.0 && (cells as u64).is_multiple_of(2) && (64.0..=16384.0).contains(&cells)) {
                return Err("size / cell must be an even integer in [64, 16384]".into());
            }
            if !(self.size / self.roads.route_cell).fract().eq(&0.0) {
                return Err("size must be a multiple of roads.route_cell".into());
            }
            self.terrain.validate()?;
            self.erosion.validate()?;
            self.roads.validate(self.cell)?;
            let f = &self.farms;
            let sizes = f.spacing > 0.0 && f.yard[0] > 0.0 && f.yard[1] > 0.0;
            if !(sizes && f.max_relief >= 0.0 && f.margin >= 0.0 && f.pad >= 0.0) {
                return Err("farms: spacing and yard must be positive, max_relief, margin and pad ≥ 0".into());
            }
            self.fields.validate()?;
            self.scatter.validate()?;
            let m = &self.materials;
            if !(m.rock_slope_deg > 0.0 && m.rock_slope_deg < 90.0 && m.shore_width >= 0.0) {
                return Err("materials: rock_slope_deg in (0, 90), shore_width ≥ 0".into());
            }
            let w = &self.water;
            if !(w.min_depth >= 0.0 && w.min_area >= 0.0 && (0.0..=1.0).contains(&w.max_catchment_share)) {
                return Err("water: need min_depth, min_area ≥ 0 and max_catchment_share in [0, 1]".into());
            }
            Ok(())
        };
        check().map_err(ProcgenError::Config)
    }
}

/// What happened during generation (for logs and the CLI).
#[derive(Clone, Debug, Default, Serialize)]
pub struct RuralStats {
    /// Wall time of each stage (s).
    pub stages: Vec<(&'static str, f64)>,
    pub vertices: usize,
    pub lakes: usize,
    pub lake_cells: usize,
    /// Farm sites found and farms connected to the road network.
    pub farm_sites: usize,
    pub farms: usize,
    /// Roads (between nodes) and their total length (m) per class: paved, gravel, track.
    pub roads: [usize; 3],
    pub road_length: [f64; 3],
    pub junctions: usize,
    /// Field parcels on the map, and fields reached by a track.
    pub parcels: usize,
    pub tracks: usize,
    /// Obstacles: buildings (with silos), hedge and fence pieces, trees.
    pub buildings: usize,
    pub hedges: usize,
    pub fences: usize,
    pub trees: usize,
    pub height_range: (f64, f64),
    /// Cells per material id.
    pub materials: Vec<(String, usize)>,
}

impl RuralStats {
    fn stage(&mut self, name: &'static str, t: &mut Instant) {
        self.stages.push((name, t.elapsed().as_secs_f64()));
        *t = Instant::now();
    }

    pub fn total_seconds(&self) -> f64 {
        self.stages.iter().map(|s| s.1).sum()
    }
}

/// Vertex heights of the final grid with bilinear sampling.
struct Heights<'a> {
    h: &'a [f32],
    n: usize,
    origin: DVec2,
    cell: f64,
}

impl Heights<'_> {
    fn at(&self, p: DVec2) -> f64 {
        let q = ((p - self.origin) / self.cell).clamp(DVec2::ZERO, DVec2::splat((self.n - 1) as f64));
        let (ix, iy) = ((q.x as usize).min(self.n - 2), (q.y as usize).min(self.n - 2));
        let (fx, fy) = (q.x - ix as f64, q.y - iy as f64);
        let v = |x: usize, y: usize| self.h[y * self.n + x] as f64;
        let a = v(ix, iy) + (v(ix + 1, iy) - v(ix, iy)) * fx;
        let b = v(ix, iy + 1) + (v(ix + 1, iy + 1) - v(ix, iy + 1)) * fx;
        a + (b - a) * fy
    }
}

/// A farm yard: a flat rectangle at `z`, turned by `heading` (its long side along it).
#[derive(Clone, Debug)]
pub(crate) struct Yard {
    pub centre: DVec2,
    pub z: f64,
    pub heading: f64,
}

impl Yard {
    /// `p` in the yard's frame.
    fn local(&self, p: DVec2) -> DVec2 {
        let (s, co) = self.heading.sin_cos();
        let d = p - self.centre;
        DVec2::new(co * d.x + s * d.y, -s * d.x + co * d.y)
    }

    /// Distance from `p` to the rectangle with half extents `half` (0 inside).
    pub(crate) fn distance(&self, p: DVec2, half: DVec2) -> f64 {
        (self.local(p).abs() - half).max(DVec2::ZERO).length()
    }
}

/// Generate a rural map. Uses the current rayon pool; the output does not depend on its size.
pub fn generate(config: &RuralConfig, seed: u64) -> Result<(StaticWorld, RuralStats), ProcgenError> {
    config.validate()?;
    let c = config;
    let mut stats = RuralStats::default();
    let mut t = Instant::now();
    let root = Seed::from_u64(seed).child("map/rural");
    let n = c.vertices();
    let origin = DVec2::splat(-0.5 * c.size);
    stats.vertices = n * n;

    // 1. Landform.
    let land = landform(
        &Land { size: c.size, cell: c.cell, terrain: &c.terrain, erosion: &c.erosion, water: &c.water },
        &root,
        &mut |name| stats.stage(name, &mut t),
    );
    let (mut heights, lakes, acc) = (land.heights, land.lakes, land.accumulation);
    stats.lakes = lakes.count;
    stats.lake_cells = lakes.cells;
    let water = lakes.water;

    // Distance to water: the highest level within the clearances (−∞ where there is none).
    let road_reach = c.roads.water_clearance + 0.5 * c.roads.paved.width.max(c.roads.gravel.width);
    let near_road_water = nearby_water(&water, n - 1, (road_reach / c.cell).ceil() as usize);
    let near_farm_water = nearby_water(&water, n - 1, (c.farms.water_clearance / c.cell).ceil() as usize);
    let wet = |near: &[f32], p: DVec2| {
        let q = ((p - origin) / c.cell).floor();
        let (cx, cy) = ((q.x.max(0.0) as usize).min(n - 2), (q.y.max(0.0) as usize).min(n - 2));
        near[cy * (n - 1) + cx] > f32::NEG_INFINITY
    };

    // 2. Farm sites and field parcels.
    let hs = Heights { h: &heights, n, origin, cell: c.cell };
    let sites = farm_sites(c, &hs, &|p| wet(&near_farm_water, p), &root.child("farms"));
    stats.farm_sites = sites.len();
    let slope = |p: DVec2| {
        let d = 10.0;
        let gx = hs.at(p + DVec2::X * d) - hs.at(p - DVec2::X * d);
        let gy = hs.at(p + DVec2::Y * d) - hs.at(p - DVec2::Y * d);
        (gx * gx + gy * gy).sqrt() / (2.0 * d)
    };
    let parcels = Parcels::new(&c.fields, c.size, &slope, &root.child("parcels"));
    let yard_half = DVec2::new(c.farms.yard[0], c.farms.yard[1]);
    let pad_half = yard_half + c.farms.pad;
    let pad_radius = pad_half.length() + 4.0;
    // Field gates: parcel centres on the map, away from the farms.
    let inner = 0.5 * c.size - 20.0;
    let gates: Vec<DVec2> = parcels
        .centres
        .iter()
        .zip(&parcels.kinds)
        .filter(|&(p, &kind)| {
            kind != ParcelKind::Woods
                && p.x.abs() < inner
                && p.y.abs() < inner
                && sites.iter().all(|s| s.distance(*p) > pad_radius + 10.0)
        })
        .map(|(p, _)| *p)
        .collect();
    stats.stage("farm sites", &mut t);

    // 3. Routing.
    let mut grid = RouteGrid::new(c, &hs, &|p| wet(&near_road_water, p));
    for (k, &site) in sites.iter().enumerate() {
        grid.mark_pad(k, site, pad_radius);
    }
    let plan = route_roads(c, &grid, &sites, &gates, &mut root.child("roads").rng());
    stats.farms = plan.farms.len();
    stats.stage("routing", &mut t);

    // 4. Splines.
    let mut pieces = plan.split(&grid);
    for piece in &mut pieces {
        piece.points = smooth_path(&piece.points, c.roads.class(piece.class).min_radius);
    }
    stats.stage("splines", &mut t);

    // 5. Profiles: yard levels, node heights, then each road pinned to its nodes.
    let yards: Vec<Yard> = plan
        .farms
        .iter()
        .map(|f| {
            let centre = f.centre;
            // Long side along the road leaving the yard.
            let road = pieces.iter().find(|p| p.start == f.node).expect("every farm has its road");
            let d = road.points[road.points.len().min(8) - 1] - road.points[0];
            let heading = d.y.atan2(d.x);
            let (s, co) = heading.sin_cos();
            let mut sum = 0.0;
            let mut count = 0.0;
            for i in -4..=4 {
                for j in -4..=4 {
                    let local = DVec2::new(pad_half.x * i as f64 / 4.0, pad_half.y * j as f64 / 4.0);
                    sum += hs.at(centre + DVec2::new(co * local.x - s * local.y, s * local.x + co * local.y));
                    count += 1.0;
                }
            }
            Yard { centre, z: sum / count, heading }
        })
        .collect();
    let mut node_z: Vec<f64> = plan
        .nodes
        .iter()
        .map(|nd| {
            let ring = (0..8).map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 8.0;
                hs.at(nd.position + 5.0 * DVec2::new(a.cos(), a.sin()))
            });
            (hs.at(nd.position) + ring.sum::<f64>()) / 9.0
        })
        .collect();
    let mut fixed = vec![false; node_z.len()];
    for (f, y) in plan.farms.iter().zip(&yards) {
        node_z[f.node as usize] = y.z;
        fixed[f.node as usize] = true;
    }
    // A road stays level with a yard while it crosses the yard's pad, so that its surface
    // (blended in after the pads) does not cut a ramp into the yard: it climbs only over
    // `points[a..b]`, from the last point on the start's pad to the first on the end's.
    let on_pad = |node: u32, q: DVec2| {
        plan.farms.iter().zip(&yards).any(|(f, y)| f.node == node && y.distance(q, pad_half) == 0.0)
    };
    let ramps: Vec<(usize, usize)> = pieces
        .iter()
        .map(|p| {
            let n = p.points.len();
            let a = p.points.iter().position(|&q| !on_pad(p.start, q)).unwrap_or(n);
            let b = p.points.iter().rposition(|&q| !on_pad(p.end, q)).map_or(0, |i| i + 1);
            if a < b { (a.saturating_sub(1), (b + 1).min(n)) } else { (0, n) }
        })
        .collect();
    let lengths: Vec<f64> = pieces
        .iter()
        .zip(&ramps)
        .map(|(p, &(a, b))| p.points[a..b].windows(2).map(|w| w[0].distance(w[1])).sum())
        .collect();
    reach_nodes(&c.roads, &pieces, &lengths, &mut node_z, &fixed);
    let roads: Vec<Road> = pieces
        .iter()
        .zip(&ramps)
        .map(|(p, &(a, b))| {
            let cc = c.roads.class(p.class);
            let (z0, z1) = (node_z[p.start as usize], node_z[p.end as usize]);
            let mut z = vec![z0; a];
            z.extend(profile(&p.points[a..b], &hs, z0, z1, cc.max_grade, c.roads.profile_window));
            z.resize(p.points.len(), z1);
            let points = p.points.iter().zip(z).map(|(q, z)| q.extend(z)).collect();
            Road { class: p.class, width: cc.width, start: p.start, end: p.end, line: Polyline::new(points) }
        })
        .collect();
    let nodes: Vec<RoadNode> = plan
        .nodes
        .iter()
        .zip(&node_z)
        .map(|(nd, &z)| RoadNode { position: nd.position.extend(z), kind: nd.kind })
        .collect();
    stats.junctions = nodes.iter().filter(|n| n.kind == NodeKind::Junction).count();
    stats.tracks = nodes.iter().filter(|n| n.kind == NodeKind::Gate).count();
    for r in &roads {
        let k = r.class as usize;
        stats.roads[k] += 1;
        stats.road_length[k] += r.line.length();
    }
    let network = RoadNetwork::new(nodes, roads).map_err(|e| ProcgenError::Config(e.to_string()))?;
    stats.stage("profiles", &mut t);

    // 6. Terrain blending.
    blend(c, &mut heights, n, origin, &network, &yards, pad_half);
    stats.stage("blending", &mut t);

    // 7. Materials.
    let moisture = moisture(&acc, n);
    drop(acc);
    let land = Landuse { net: &network, yards: &yards, yard_half, pad_half, parcels: &parcels };
    let (materials, slope) = materials(c, origin, &heights, &water, &moisture, &land);
    drop(moisture);
    let table = MaterialTable::rural();
    let mut counts = [0usize; 256];
    let mut seen = vec![false; parcels.centres.len()];
    for (k, id) in materials.iter().enumerate() {
        counts[id.0 as usize] += 1;
        if matches!(*id, MaterialId::MEADOW | MaterialId::CROP | MaterialId::PLOWED | MaterialId::FOREST_FLOOR) {
            let p = origin + (DVec2::new((k % (n - 1)) as f64, (k / (n - 1)) as f64) + 0.5) * c.cell;
            seen[parcels.locate(p).0] = true;
        }
    }
    stats.parcels = seen.iter().filter(|&&s| s).count();
    stats.materials = (0..256)
        .filter(|&i| counts[i] > 0)
        .map(|i| {
            let name = if i < table.len() { table.get(MaterialId(i as u8)).name.clone() } else { format!("#{i}") };
            (name, counts[i])
        })
        .collect();
    let grid = HeightGrid::new(origin, c.cell, n, n, heights, materials).with_water(water);
    stats.height_range = grid.height_range();
    stats.stage("materials", &mut t);

    // 8. Obstacles.
    let sc = &c.scatter;
    let r = &c.roads;
    let clear = Clearance {
        net: &network,
        yards: &yards,
        yard_half,
        grid: &grid,
        clearance: sc.road_clearance,
        headroom: sc.road_headroom,
        max_half_width: 0.5 * r.paved.width.max(r.gravel.width).max(r.track.width),
    };
    let mut obstacles = Vec::new();
    if sc.buildings {
        let mut rng = root.child("buildings").rng();
        let groups = yards.iter().flat_map(|y| farmland::buildings(y, yard_half, &grid, &mut rng)).collect::<Vec<_>>();
        farmland::keep_clear(groups, &clear, &mut obstacles);
    }
    stats.buildings = obstacles.len();
    let edges = farmland::edge_obstacles(sc, &parcels, &grid, &root.child("edges"));
    // Hedges and fences keep off farm pads and leave gaps where tracks and roads pass.
    let edges = edges.into_iter().filter(|g| {
        let p = g[0].pose.pos.truncate();
        yards.iter().all(|y| y.distance(p, pad_half) > 2.0)
    });
    let before = obstacles.len();
    farmland::keep_clear(edges, &clear, &mut obstacles);
    stats.hedges =
        obstacles[before..].iter().filter(|o| o.tag == tags::HEDGE && o.class == ObstacleClass::Foliage).count();
    stats.fences = obstacles[before..].iter().filter(|o| o.tag == tags::FENCE).count();
    let before = obstacles.len();
    farmland::keep_clear(farmland::tree_lines(sc, &network, &grid, &root.child("tree_lines")), &clear, &mut obstacles);
    let ground = Ground { grid: &grid, slope: &slope, treeline: &|_, _| f64::INFINITY };
    let scattered = trees(&ground, &sc.trees, &root.child("trees"));
    let scattered = scattered.obstacles.chunks(2).map(|g| g.to_vec()).filter(|g| {
        let p = g[0].pose.pos.truncate();
        yards.iter().all(|y| y.distance(p, pad_half) > 1.0)
    });
    farmland::keep_clear(scattered, &clear, &mut obstacles);
    stats.trees = obstacles[before..].iter().filter(|o| o.tag == tags::TRUNK).count();
    stats.stage("obstacles", &mut t);

    let mut meta = MapMeta::new("rural", "rural", seed);
    meta.generator_version = RURAL_VERSION;
    meta.geo_origin = c.geo_origin;
    let world = StaticWorld::new(meta, grid, ObstacleSet::new(obstacles), table).with_roads(network);
    stats.stage("obstacle index", &mut t);
    Ok((world, stats))
}

// ------------------------------------------------------------------------------ farm sites

/// Flat, dry places at least `spacing` apart: candidates on a jittered grid, taken in a
/// random order.
fn farm_sites(c: &RuralConfig, hs: &Heights, wet: &dyn Fn(DVec2) -> bool, seed: &Seed) -> Vec<DVec2> {
    let f = &c.farms;
    let mut rng = seed.rng();
    let step = f.spacing / std::f64::consts::SQRT_2;
    let lo = -0.5 * c.size + f.margin;
    let hi = 0.5 * c.size - f.margin;
    if hi <= lo {
        return Vec::new();
    }
    let k = ((hi - lo) / step).ceil() as usize;
    let mut candidates = Vec::with_capacity(k * k);
    for iy in 0..k {
        for ix in 0..k {
            let p = DVec2::new(
                lo + (ix as f64 + rng.range(0.0, 1.0)) * step,
                lo + (iy as f64 + rng.range(0.0, 1.0)) * step,
            );
            if p.x < hi && p.y < hi {
                candidates.push(p);
            }
        }
    }
    // Fisher–Yates with our own draws.
    for i in (1..candidates.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        candidates.swap(i, j);
    }
    let r = f.yard[0].max(f.yard[1]) * 1.2;
    let mut out: Vec<DVec2> = Vec::new();
    for p in candidates {
        if out.iter().any(|q| q.distance(p) < f.spacing) || wet(p) {
            continue;
        }
        let (mut lo_h, mut hi_h) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut dry = true;
        for i in -3..=3 {
            for j in -3..=3 {
                let q = p + DVec2::new(i as f64, j as f64) * (r / 3.0);
                let h = hs.at(q);
                lo_h = lo_h.min(h);
                hi_h = hi_h.max(h);
                dry &= !wet(q);
            }
        }
        if dry && hi_h - lo_h <= f.max_relief {
            out.push(p);
        }
    }
    out
}

// ------------------------------------------------------------------------------ routing

/// The 16 move directions (about 22.5° apart), with knight moves.
const DIRS: [(i32, i32); 16] = [
    (1, 0),
    (2, 1),
    (1, 1),
    (1, 2),
    (0, 1),
    (-1, 2),
    (-1, 1),
    (-2, 1),
    (-1, 0),
    (-2, -1),
    (-1, -1),
    (-1, -2),
    (0, -1),
    (1, -2),
    (1, -1),
    (2, -1),
];

/// The coarse grid roads are routed on: vertex heights, gradients and blocked vertices.
struct RouteGrid {
    m: usize,
    cell: f64,
    origin: DVec2,
    z: Vec<f64>,
    grad: Vec<DVec2>,
    blocked: Vec<bool>,
    /// Farm pad around each vertex (index + 1; 0 for none): other roads avoid it.
    pad: Vec<u16>,
}

impl RouteGrid {
    fn new(c: &RuralConfig, hs: &Heights, wet: &(dyn Fn(DVec2) -> bool + Sync)) -> Self {
        let cell = c.roads.route_cell;
        let m = (c.size / cell).round() as usize + 1;
        let origin = DVec2::splat(-0.5 * c.size);
        let pos = |i: usize| origin + DVec2::new((i % m) as f64, (i / m) as f64) * cell;
        let z: Vec<f64> = (0..m * m).into_par_iter().map(|i| hs.at(pos(i))).collect();
        let at = |x: usize, y: usize| z[y.min(m - 1) * m + x.min(m - 1)];
        let grad = (0..m * m)
            .into_par_iter()
            .map(|i| {
                let (x, y) = (i % m, i / m);
                let (x0, x1, y0, y1) = (x.saturating_sub(1), x + 1, y.saturating_sub(1), y + 1);
                DVec2::new(
                    (at(x1, y) - at(x0, y)) / ((x1.min(m - 1) - x0) as f64 * cell),
                    (at(x, y1) - at(x, y0)) / ((y1.min(m - 1) - y0) as f64 * cell),
                )
            })
            .collect();
        let blocked = (0..m * m).into_par_iter().map(|i| wet(pos(i))).collect();
        Self { m, cell, origin, z, grad, blocked, pad: vec![0; m * m] }
    }

    /// Mark the vertices within `radius` of `centre` as the pad of farm `k`.
    fn mark_pad(&mut self, k: usize, centre: DVec2, radius: f64) {
        for (i, pad) in self.pad.iter_mut().enumerate() {
            let p = self.origin + DVec2::new((i % self.m) as f64, (i / self.m) as f64) * self.cell;
            if *pad == 0 && p.distance(centre) <= radius {
                *pad = k as u16 + 1;
            }
        }
    }

    fn pos(&self, i: usize) -> DVec2 {
        self.origin + DVec2::new((i % self.m) as f64, (i / self.m) as f64) * self.cell
    }

    fn index(&self, p: DVec2) -> usize {
        let q = ((p - self.origin) / self.cell).round();
        let x = (q.x.max(0.0) as usize).min(self.m - 1);
        let y = (q.y.max(0.0) as usize).min(self.m - 1);
        y * self.m + x
    }

    /// Target of move `d` from vertex `i`, if on the grid and not blocked (a knight move
    /// also needs the two vertices it passes free).
    fn step(&self, i: usize, d: usize) -> Option<usize> {
        let (dx, dy) = DIRS[d];
        let (x, y) = ((i % self.m) as i32 + dx, (i / self.m) as i32 + dy);
        if x < 0 || y < 0 || x >= self.m as i32 || y >= self.m as i32 {
            return None;
        }
        let j = y as usize * self.m + x as usize;
        if self.blocked[j] {
            return None;
        }
        if dx.abs() + dy.abs() == 3 {
            let (sx, sy) = (dx.signum(), dy.signum());
            let (ax, ay) = if dx.abs() == 2 { (sx, 0) } else { (0, sy) };
            let a = ((i / self.m) as i32 + ay) as usize * self.m + ((i % self.m) as i32 + ax) as usize;
            let b = ((i / self.m) as i32 + sy) as usize * self.m + ((i % self.m) as i32 + sx) as usize;
            if self.blocked[a] || self.blocked[b] {
                return None;
            }
        }
        Some(j)
    }

    /// Cost of the move `d` from `i` to `j` for a road with maximum grade `max_grade`, twenty
    /// times higher inside a farm pad other than `own`.
    fn move_cost(&self, c: &RoadsConfig, i: usize, j: usize, d: usize, max_grade: f64, own: u16) -> f64 {
        let (dx, dy) = DIRS[d];
        let dir = DVec2::new(dx as f64, dy as f64);
        let len = dir.length() * self.cell;
        let grade = (self.z[j] - self.z[i]).abs() / len;
        let side = 0.5 * (self.grad[i] + self.grad[j]).perp_dot(dir / dir.length()).abs();
        let steep = (grade - max_grade).max(0.0);
        let pad = if self.pad[j] != 0 && self.pad[j] != own { 20.0 } else { 1.0 };
        pad * len * (1.0 + c.grade_cost * grade * grade + 400.0 * steep + c.side_slope_cost * side * side)
    }
}

/// Min-heap entry: smaller cost first, then smaller state.
#[derive(PartialEq)]
struct Open(f64, u32);

impl Eq for Open {}

impl Ord for Open {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then_with(|| other.1.cmp(&self.1))
    }
}

impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Search state reused between A* runs: only the touched entries are reset.
struct Search {
    cost: Vec<f64>,
    from: Vec<u32>,
    touched: Vec<u32>,
}

impl Search {
    fn new(g: &RouteGrid) -> Self {
        let states = g.m * g.m * 16;
        Self { cost: vec![f64::INFINITY; states], from: vec![u32::MAX; states], touched: Vec::new() }
    }

    fn reset(&mut self) {
        for &s in &self.touched {
            self.cost[s as usize] = f64::INFINITY;
            self.from[s as usize] = u32::MAX;
        }
        self.touched.clear();
    }
}

/// What a road being routed may do: its grade limit and the farm pad it may cross.
struct Leg {
    max_grade: f64,
    own_pad: u16,
}

/// A* over (vertex, direction) states from `start` until `goal(vertex)`, with heuristic `h`.
/// Turning by more than 67.5° in one step is not allowed. Returns the vertices of the path.
fn astar(
    g: &RouteGrid,
    c: &RoadsConfig,
    leg: &Leg,
    ws: &mut Search,
    start: usize,
    goal: &dyn Fn(usize) -> bool,
    h: &dyn Fn(usize) -> f64,
) -> Option<Vec<usize>> {
    ws.reset();
    let Search { cost, from, touched } = ws;
    let mut heap = BinaryHeap::new();
    for d in 0..16 {
        let s = start * 16 + d;
        cost[s] = 0.0;
        touched.push(s as u32);
        heap.push(Open(h(start), s as u32));
    }
    let step_angle = std::f64::consts::TAU / 16.0;
    while let Some(Open(f, s)) = heap.pop() {
        let s = s as usize;
        let (v, d) = (s / 16, s % 16);
        let gs = cost[s];
        if f > gs + h(v) + 1e-9 {
            continue;
        }
        if goal(v) && v != start {
            let mut path = vec![v];
            let mut k = s;
            while from[k] != u32::MAX {
                k = from[k] as usize;
                path.push(k / 16);
            }
            path.reverse();
            return Some(path);
        }
        for turn in -3i32..=3 {
            let nd = (d as i32 + turn).rem_euclid(16) as usize;
            let Some(w) = g.step(v, nd) else { continue };
            let a = turn as f64 * step_angle;
            let nc = gs + g.move_cost(c, v, w, nd, leg.max_grade, leg.own_pad) + c.turn_cost * g.cell * a * a;
            let ns = w * 16 + nd;
            if nc < cost[ns] {
                if cost[ns] == f64::INFINITY {
                    touched.push(ns as u32);
                }
                cost[ns] = nc;
                from[ns] = s as u32;
                heap.push(Open(nc + h(w), ns as u32));
            }
        }
    }
    None
}

/// Chamfer distance (m) from every vertex to the nearest marked one.
fn distance_to(g: &RouteGrid, marked: &[bool]) -> Vec<f64> {
    let m = g.m;
    let mut d: Vec<f64> = marked.iter().map(|&b| if b { 0.0 } else { f64::INFINITY }).collect();
    let (a, b) = (g.cell, g.cell * std::f64::consts::SQRT_2);
    for y in 0..m {
        for x in 0..m {
            let i = y * m + x;
            let mut v = d[i];
            if x > 0 {
                v = v.min(d[i - 1] + a);
            }
            if y > 0 {
                v = v.min(d[i - m] + a);
                if x > 0 {
                    v = v.min(d[i - m - 1] + b);
                }
                if x + 1 < m {
                    v = v.min(d[i - m + 1] + b);
                }
            }
            d[i] = v;
        }
    }
    for y in (0..m).rev() {
        for x in (0..m).rev() {
            let i = y * m + x;
            let mut v = d[i];
            if x + 1 < m {
                v = v.min(d[i + 1] + a);
            }
            if y + 1 < m {
                v = v.min(d[i + m] + a);
                if x + 1 < m {
                    v = v.min(d[i + m + 1] + b);
                }
                if x > 0 {
                    v = v.min(d[i + m - 1] + b);
                }
            }
            d[i] = v;
        }
    }
    d
}

struct PlanNode {
    position: DVec2,
    kind: NodeKind,
}

struct Farm {
    centre: DVec2,
    node: u32,
}

struct RoutedPath {
    class: RoadClass,
    vertices: Vec<usize>,
}

/// Routed paths on the grid, the vertices that are nodes, and the connected farms.
struct Plan {
    paths: Vec<RoutedPath>,
    /// Grid vertex → node index.
    node_at: BTreeMap<usize, u32>,
    nodes: Vec<PlanNode>,
    farms: Vec<Farm>,
}

/// A road between two nodes before profiling: horizontal points about 1 m apart.
struct Piece {
    class: RoadClass,
    start: u32,
    end: u32,
    points: Vec<DVec2>,
}

impl Plan {
    fn node(&mut self, v: usize, position: DVec2, kind: NodeKind) -> u32 {
        if let Some(&k) = self.node_at.get(&v) {
            return k;
        }
        let k = self.nodes.len() as u32;
        self.nodes.push(PlanNode { position, kind });
        self.node_at.insert(v, k);
        k
    }

    /// Every path cut at the nodes it passes through.
    fn split(&self, g: &RouteGrid) -> Vec<Piece> {
        let mut out = Vec::new();
        for p in &self.paths {
            let mut from = 0;
            for k in 1..p.vertices.len() {
                if let Some(&end) = self.node_at.get(&p.vertices[k]) {
                    let start = self.node_at[&p.vertices[from]];
                    let mut points: Vec<DVec2> = p.vertices[from..=k].iter().map(|&v| g.pos(v)).collect();
                    points[0] = self.nodes[start as usize].position;
                    *points.last_mut().expect("two vertices") = self.nodes[end as usize].position;
                    out.push(Piece { class: p.class, start, end, points });
                    from = k;
                }
            }
        }
        out
    }
}

/// The main road across the map, then a road from each farm and a track to each field far
/// from the roads (nearest to the network first) to the nearest road built so far.
fn route_roads(
    c: &RuralConfig,
    g: &RouteGrid,
    sites: &[DVec2],
    fields: &[DVec2],
    rng: &mut autonomousim_core::rng::SimRng,
) -> Plan {
    let mut plan = Plan { paths: Vec::new(), node_at: BTreeMap::new(), nodes: Vec::new(), farms: Vec::new() };
    let r = &c.roads;
    let m = g.m;
    let mut ws = Search::new(g);
    let half = 0.5 * c.size - 2.0;
    // Main road: between opposite edges, on dry ground; a few attempts.
    let main = Leg { max_grade: r.paved.max_grade, own_pad: 0 };
    for attempt in 0..16 {
        let along_x = (rng.below(2) == 0) != (attempt % 2 == 1);
        let a = rng.range(-0.6, 0.6) * half;
        let b = rng.range(-0.6, 0.6) * half;
        let (pa, pb) = if along_x {
            (DVec2::new(-half, a), DVec2::new(half, b))
        } else {
            (DVec2::new(a, -half), DVec2::new(b, half))
        };
        let (va, vb) = (g.index(pa), g.index(pb));
        if g.blocked[va] || g.blocked[vb] {
            continue;
        }
        let target = g.pos(vb);
        let Some(path) = astar(g, r, &main, &mut ws, va, &|v| v == vb, &|v| g.pos(v).distance(target)) else {
            continue;
        };
        plan.node(va, g.pos(va), NodeKind::End);
        plan.node(vb, g.pos(vb), NodeKind::End);
        plan.paths.push(RoutedPath { class: RoadClass::Paved, vertices: path });
        break;
    }
    if plan.paths.is_empty() {
        return plan;
    }
    let mut on_road = vec![false; m * m];
    for p in &plan.paths {
        for &v in &p.vertices {
            on_road[v] = true;
        }
    }
    // Farm roads (gravel), then field tracks, each joining the nearest road built so far,
    // the one nearest to the network first. Places that cannot be reached are left out.
    let mut connect = |points: &[DVec2], class: RoadClass, min_distance: f64, plan: &mut Plan| {
        let dist = distance_to(g, &on_road);
        let mut order: Vec<(f64, usize)> = points.iter().enumerate().map(|(k, &p)| (dist[g.index(p)], k)).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, k) in order {
            let p = points[k];
            let v = g.index(p);
            if g.blocked[v] || on_road[v] {
                continue;
            }
            let dist = distance_to(g, &on_road);
            if dist[v] < min_distance {
                continue;
            }
            let (kind, own_pad) = match class {
                RoadClass::Gravel => (NodeKind::Yard, k as u16 + 1),
                _ => (NodeKind::Gate, 0),
            };
            let leg = Leg { max_grade: r.class(class).max_grade, own_pad };
            let Some(path) = astar(g, r, &leg, &mut ws, v, &|w| on_road[w], &|w| 0.92 * dist[w]) else {
                continue;
            };
            let node = plan.node(v, p, kind);
            let end = *path.last().expect("a path");
            plan.node(end, g.pos(end), NodeKind::Junction);
            for &w in &path {
                on_road[w] = true;
            }
            plan.paths.push(RoutedPath { class, vertices: path });
            if class == RoadClass::Gravel {
                plan.farms.push(Farm { centre: p, node });
            }
        }
    };
    connect(sites, RoadClass::Gravel, 0.0, &mut plan);
    connect(fields, RoadClass::Track, c.fields.track_distance, &mut plan);
    plan
}

// ------------------------------------------------------------------------------ splines

/// Centripetal Catmull–Rom through control points taken every 12 m along `path` (ends kept),
/// resampled every metre; the controls are smoothed until the curvature away from the ends
/// stays within `1 / min_radius`.
fn smooth_path(path: &[DVec2], min_radius: f64) -> Vec<DVec2> {
    let len: f64 = path.windows(2).map(|w| w[0].distance(w[1])).sum();
    let spacing = 12.0;
    let mut controls = vec![path[0]];
    let mut acc = 0.0;
    for w in path.windows(2) {
        acc += w[0].distance(w[1]);
        if acc >= spacing {
            controls.push(w[1]);
            acc = 0.0;
        }
    }
    let last = *path.last().expect("a path");
    if controls.last().is_some_and(|p| p.distance(last) < 0.5 * spacing) && controls.len() > 1 {
        controls.pop();
    }
    controls.push(last);
    let kmax = 1.0 / min_radius;
    let mut points = catmull_rom(&controls);
    for _ in 0..60 {
        if controls.len() < 3 || max_curvature(&points, 10.0_f64.min(0.25 * len)) <= kmax {
            break;
        }
        let prev = controls.clone();
        for i in 1..controls.len() - 1 {
            controls[i] = prev[i] + 0.5 * (0.5 * (prev[i - 1] + prev[i + 1]) - prev[i]);
        }
        points = catmull_rom(&controls);
    }
    points
}

/// Largest |curvature| (three points 2 m apart) more than `skip` m from both ends.
fn max_curvature(points: &[DVec2], skip: f64) -> f64 {
    let line = Polyline::new(points.iter().map(|p| p.extend(0.0)).collect());
    let len = line.length();
    let mut s = skip;
    let mut k: f64 = 0.0;
    while s <= len - skip {
        k = k.max(line.curvature_at(s).abs());
        s += 1.0;
    }
    k
}

/// Centripetal Catmull–Rom spline through `controls` (with mirrored end tangents), resampled
/// every metre by arc length.
fn catmull_rom(controls: &[DVec2]) -> Vec<DVec2> {
    if controls.len() == 2 {
        return resample(&[controls[0], controls[1]], 1.0);
    }
    let n = controls.len();
    let ext = |i: isize| -> DVec2 {
        if i < 0 {
            2.0 * controls[0] - controls[1]
        } else if i as usize >= n {
            2.0 * controls[n - 1] - controls[n - 2]
        } else {
            controls[i as usize]
        }
    };
    let mut dense = Vec::new();
    for i in 0..n - 1 {
        let (p0, p1, p2, p3) = (ext(i as isize - 1), ext(i as isize), ext(i as isize + 1), ext(i as isize + 2));
        let knot = |a: DVec2, b: DVec2| a.distance(b).sqrt().max(1e-6);
        let (t1, t2, t3) = (knot(p0, p1), knot(p1, p2), knot(p2, p3));
        let (t0, t1, t2, t3) = (0.0, t1, t1 + t2, t1 + t2 + t3);
        let steps = ((p1.distance(p2) / 0.25).ceil() as usize).max(1);
        for k in 0..steps {
            let t = t1 + (t2 - t1) * k as f64 / steps as f64;
            let a1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0));
            let a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1));
            let a3 = p2 * ((t3 - t) / (t3 - t2)) + p3 * ((t - t2) / (t3 - t2));
            let b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0));
            let b2 = a2 * ((t3 - t) / (t3 - t1)) + a3 * ((t - t1) / (t3 - t1));
            dense.push(b1 * ((t2 - t) / (t2 - t1)) + b2 * ((t - t1) / (t2 - t1)));
        }
    }
    dense.push(controls[n - 1]);
    resample(&dense, 1.0)
}

/// Points every `step` m along a polyline (at least the two ends).
fn resample(points: &[DVec2], step: f64) -> Vec<DVec2> {
    let len: f64 = points.windows(2).map(|w| w[0].distance(w[1])).sum();
    let count = ((len / step).round() as usize).max(1);
    let ds = len / count as f64;
    let mut out = Vec::with_capacity(count + 1);
    out.push(points[0]);
    let (mut seg, mut seg_start) = (0usize, 0.0);
    for k in 1..count {
        let s = k as f64 * ds;
        while seg + 2 < points.len() && seg_start + points[seg].distance(points[seg + 1]) < s {
            seg_start += points[seg].distance(points[seg + 1]);
            seg += 1;
        }
        let l = points[seg].distance(points[seg + 1]).max(1e-12);
        out.push(points[seg].lerp(points[seg + 1], ((s - seg_start) / l).clamp(0.0, 1.0)));
    }
    out.push(*points.last().expect("points"));
    out
}

// ------------------------------------------------------------------------------ profiles

/// Heights along a road: the terrain averaged over `window` m, eased onto the node heights
/// over the first and last 15 m, then limited to `max_grade`. Too steep only where the
/// nodes themselves are too far apart in height.
/// Move the heights of the free (non-yard) nodes so that every piece can climb from one end to
/// the other within 90 % of its class's grade limit over its graded length (`lengths`): a road whose ends are too far apart in
/// height could not meet both (see `profile`). Pieces pull their ends together in a fixed
/// order until all fit or the passes run out; the terrain is then cut and filled to match.
fn reach_nodes(c: &RoadsConfig, pieces: &[Piece], lengths: &[f64], node_z: &mut [f64], fixed: &[bool]) {
    for _ in 0..100 {
        let mut moved = false;
        for (p, &len) in pieces.iter().zip(lengths) {
            let (a, b) = (p.start as usize, p.end as usize);
            let allowed = 0.9 * c.class(p.class).max_grade * len;
            let excess = (node_z[b] - node_z[a]).abs() - allowed;
            if excess <= 1e-9 || (fixed[a] && fixed[b]) {
                continue;
            }
            // Towards each other: all of it on a free end facing a fixed one, else half each.
            let dir = (node_z[b] - node_z[a]).signum();
            let (ka, kb) = match (fixed[a], fixed[b]) {
                (true, _) => (0.0, 1.0),
                (_, true) => (1.0, 0.0),
                _ => (0.5, 0.5),
            };
            node_z[a] += dir * ka * excess;
            node_z[b] -= dir * kb * excess;
            moved = true;
        }
        if !moved {
            break;
        }
    }
}

/// Heights along a road from `z0` to `z1`: the terrain, smoothed over `window` m, eased into
/// the end heights and limited to `max_grade`. Where the ends are too far apart for the grade
/// (only between two yards, see `reach_nodes`), a straight ramp instead: the road meets its
/// ends and exceeds the grade.
fn profile(points: &[DVec2], hs: &Heights, z0: f64, z1: f64, max_grade: f64, window: f64) -> Vec<f64> {
    let n = points.len();
    let raw: Vec<f64> = points.iter().map(|&p| hs.at(p)).collect();
    let ds: Vec<f64> = points.windows(2).map(|w| w[0].distance(w[1])).collect();
    let mut s = vec![0.0; n];
    for i in 1..n {
        s[i] = s[i - 1] + ds[i - 1];
    }
    let len = s[n - 1];
    if (z1 - z0).abs() > max_grade * len {
        return s.iter().map(|&si| z0 + (z1 - z0) * si / len.max(1e-9)).collect();
    }
    let half = (0.5 * window / (len / (n - 1).max(1) as f64)).round() as usize;
    let mut z: Vec<f64> = (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half).min(n - 1));
            raw[a..=b].iter().sum::<f64>() / (b - a + 1) as f64
        })
        .collect();
    let ease = 15.0_f64.min(0.5 * len);
    for i in 0..n {
        let w0 = 1.0 - smoothstep(0.0, ease, s[i]);
        let w1 = 1.0 - smoothstep(0.0, ease, len - s[i]);
        z[i] = z[i] * (1.0 - w0 - w1).max(0.0) + z0 * w0 + z1 * w1;
    }
    // Keep within reach of both ends, then limit the grade forward and backward (the backward
    // pass keeps the forward pass's limit, and neither leaves the band).
    for i in 0..n {
        let lo = (z0 - max_grade * s[i]).max(z1 - max_grade * (len - s[i]));
        let hi = (z0 + max_grade * s[i]).min(z1 + max_grade * (len - s[i]));
        z[i] = z[i].clamp(lo, hi.max(lo));
    }
    z[0] = z0;
    z[n - 1] = z1;
    for i in 1..n {
        let lim = max_grade * ds[i - 1];
        z[i] = z[i].clamp(z[i - 1] - lim, z[i - 1] + lim);
    }
    for i in (0..n - 1).rev() {
        let lim = max_grade * ds[i];
        z[i] = z[i].clamp(z[i + 1] - lim, z[i + 1] + lim);
    }
    z
}

// ------------------------------------------------------------------------------ blending

/// Cut and fill the farm pads, then the road surfaces, into the terrain. The roads go last, so
/// that each keeps its surface out to its edges (plus 0.5 m) where it passes a pad; the pad
/// then falls off towards the road over the road's shoulder.
fn blend(
    c: &RuralConfig,
    heights: &mut [f32],
    n: usize,
    origin: DVec2,
    net: &RoadNetwork,
    yards: &[Yard],
    pad_half: DVec2,
) {
    let r = &c.roads;
    let max_half = [&r.paved, &r.gravel, &r.track].iter().map(|k| 0.5 * k.width).fold(0.0, f64::max);
    // Deep cuts get wide shoulders; beyond this reach nothing is touched.
    let reach = max_half + 0.5 + r.shoulder.max(15.0);
    heights.par_chunks_mut(n).enumerate().for_each(|(iy, row)| {
        let y = origin.y + iy as f64 * c.cell;
        for (ix, h) in row.iter_mut().enumerate() {
            let p = DVec2::new(origin.x + ix as f64 * c.cell, y);
            let h0 = *h as f64;
            let mut best: Option<(f64, f64)> = None; // (weight, target)
            for yard in yards {
                let out = yard.distance(p, pad_half);
                let shoulder = r.shoulder.max(1.5 * (yard.z - h0).abs());
                let w = 1.0 - smoothstep(0.0, shoulder, out);
                if w > 0.0 && best.is_none_or(|b| w > b.0) {
                    best = Some((w, yard.z));
                }
            }
            let mut h1 = best.map_or(h0, |(w, target)| h0 + w * (target - h0));
            if let Some(rp) = net.nearest(p, reach) {
                let road = &net.roads()[rp.road as usize];
                let cc = r.class(road.class);
                let half = 0.5 * road.width;
                let pr = rp.projection;
                let target = pr.point.z - cc.crown * pr.offset.abs().min(half);
                let flat = half + 0.5;
                let shoulder = r.shoulder.max(1.5 * (target - h1).abs());
                let w = 1.0 - smoothstep(flat, flat + shoulder, pr.distance);
                h1 += w * (target - h1);
            }
            *h = h1 as f32;
        }
    });
}

// ------------------------------------------------------------------------------ materials

/// What covers the land: roads, yards and their pads, parcels.
struct Landuse<'a> {
    net: &'a RoadNetwork,
    yards: &'a [Yard],
    yard_half: DVec2,
    pad_half: DVec2,
    parcels: &'a Parcels,
}

/// Material and slope (rise over run) per cell.
fn materials(
    c: &RuralConfig,
    origin: DVec2,
    heights: &[f32],
    water: &[f32],
    moisture: &[f32],
    land: &Landuse,
) -> (Vec<MaterialId>, Vec<f32>) {
    let m = &c.materials;
    let n = c.vertices();
    let cw = n - 1;
    let rock = libm::tan(m.rock_slope_deg.to_radians());
    let near = if m.shore_width > 0.0 && water.iter().any(|w| !w.is_nan()) {
        Some(nearby_water(water, cw, (m.shore_width / c.cell).ceil() as usize))
    } else {
        None
    };
    let r = &c.roads;
    let headland = c.fields.headland;
    let reach = 0.5 * r.paved.width.max(r.gravel.width).max(r.track.width) + headland;
    let inv = 1.0 / c.cell;
    let mut out = vec![MaterialId::GRASS; cw * cw];
    let mut slopes = vec![0.0f32; cw * cw];
    out.par_chunks_mut(cw).zip(slopes.par_chunks_mut(cw)).enumerate().for_each(|(cy, (row, slope_row))| {
        let y = origin.y + (cy as f64 + 0.5) * c.cell;
        for (cx, (mat, slope_out)) in row.iter_mut().zip(slope_row.iter_mut()).enumerate() {
            let i = cy * n + cx;
            let (h00, h10, h01, h11) =
                (heights[i] as f64, heights[i + 1] as f64, heights[i + n] as f64, heights[i + n + 1] as f64);
            let gx = 0.5 * (h10 - h00 + h11 - h01) * inv;
            let gy = 0.5 * (h01 - h00 + h11 - h10) * inv;
            let slope = (gx * gx + gy * gy).sqrt();
            *slope_out = slope as f32;
            let z = 0.25 * (h00 + h10 + h01 + h11);
            let p = DVec2::new(origin.x + (cx as f64 + 0.5) * c.cell, y);
            let k = cy * cw + cx;
            let wet = moisture[i].max(moisture[i + 1]).max(moisture[i + n]).max(moisture[i + n + 1]) as f64;
            let road = land.net.nearest(p, reach).map(|rp| {
                let road = &land.net.roads()[rp.road as usize];
                (road.class, rp.projection.distance - 0.5 * road.width)
            });
            *mat = if !water[k].is_nan() {
                if water[k] as f64 - z < 0.5 { MaterialId::SAND } else { MaterialId::MUD }
            } else if land.yards.iter().any(|y| y.distance(p, land.yard_half) == 0.0) {
                MaterialId::CONCRETE
            } else if let Some((class, _)) = road.filter(|r| r.1 <= 0.0) {
                match class {
                    RoadClass::Paved => MaterialId::ASPHALT,
                    RoadClass::Gravel => MaterialId::GRAVEL,
                    RoadClass::Track => MaterialId::DIRT,
                }
            } else if slope > rock {
                MaterialId::ROCK
            } else if near.as_ref().is_some_and(|l| z < l[k] as f64 + m.shore_height) {
                MaterialId::SAND
            } else if wet > m.marsh_moisture && slope < 0.05 {
                MaterialId::MUD
            } else if road.is_some_and(|r| r.1 < headland)
                || land.yards.iter().any(|y| y.distance(p, land.pad_half) == 0.0)
            {
                MaterialId::GRASS
            } else {
                let (parcel, edge) = land.parcels.locate(p);
                let kind = land.parcels.kinds[parcel];
                if edge < headland && kind != ParcelKind::Woods { MaterialId::GRASS } else { kind.material() }
            };
        }
    });
    (out, slopes)
}
