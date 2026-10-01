//! Urban maps: a city on a gentle plateau with a street network, fading into farmland.
//!
//! Pipeline:
//! 1. The wild landform (terrain noise, erosion, upsampling, lakes) with gentle relief,
//!    flattened towards a plateau under the city (its outline a wobbly circle with a fringe).
//! 2. Districts: the Voronoi cells of seeds spread over the city; downtown districts share one
//!    perturbed grid, the others have a grid of their own or organic streets.
//! 3. Streets on a planar graph (see [`layout`]): radial arterials out to the map edge (paved
//!    rural roads beyond the city) and a ring road, the grids, organic streets grown from the
//!    arterials and collectors, a clean-up, roundabouts.
//! 4. Roads: the graph's chains between junctions (and changes of class or street), smoothed
//!    within their class's minimum radius (or, where that strays from the street graph,
//!    with each corner filleted), with a cross-section per street and class (lanes,
//!    median, bike and parking lanes, sidewalks) drawn from the street's own random stream.
//! 5. Road profiles as for rural maps: terrain heights along each road, smoothed, pinned to the
//!    node heights and limited to the class's maximum grade.
//! 6. Terrain blending: carriageways (with a crown) and sidewalks (flat, level with the
//!    carriageway's edge) are cut or filled into the terrain, with shoulders beyond. Junctions
//!    are paved over as far as their lanes leave free (the area their connectors cross), and
//!    where roads meet the surface turns smoothly from one road's height to the next.
//! 7. Materials per cell: asphalt carriageways, concrete sidewalks, lake beds, rock on steep
//!    slopes, grass in the city and field parcels in the countryside.
//!
//! As for wild and rural maps, the result depends only on the configuration and the seed.

mod graph;
mod layout;
mod sites;

use crate::ProcgenError;
use crate::farmland::{FieldsConfig, ParcelKind, Parcels};
use crate::noise::smoothstep;
use crate::rural::{Heights, profile, resample, smooth_path};
use crate::terrain::{ErosionConfig, TerrainConfig};
use crate::wild::{Land, WaterConfig, landform, merge, nearby_water};
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_world::{
    BayKind, GeoOrigin, HeightGrid, MapMeta, NodeKind, ObstacleSet, Polyline, Road, RoadClass, RoadNetwork, RoadNode,
    RoadPoint, Section, StaticWorld, Zone,
};
use glam::DVec2;
use graph::{Graph, NodeId};
use layout::{Districts, Ring, Site};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
pub use sites::{FurnitureConfig, LotsConfig, ZoneConfig};
use std::f64::consts::TAU;
use std::time::Instant;

/// Bumped whenever the output for a given configuration and seed changes.
pub const URBAN_VERSION: u32 = 6;

/// The city's outline and ground.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CityConfig {
    /// Radius (m) of the city, and the width (m) of the fringe where it fades into farmland.
    pub radius: f64,
    pub fringe: f64,
    /// Largest offset of the city centre from the map centre, as a share of the map size.
    pub centre_jitter: f64,
    /// Relative amplitude of the outline's wobble.
    pub wobble: f64,
    /// Share of the terrain relief kept under the city (1 keeps it all).
    pub plateau: f64,
}

impl Default for CityConfig {
    fn default() -> Self {
        Self { radius: 380.0, fringe: 80.0, centre_jitter: 0.04, wobble: 0.08, plateau: 0.3 }
    }
}

/// How the streets of a district are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistrictKind {
    /// Part of the (one) downtown grid.
    Downtown,
    /// A grid of its own orientation and block size.
    Grid,
    /// Organic streets with branches and cul-de-sacs.
    Organic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DistrictsConfig {
    /// Mean distance between district seeds (m).
    pub spacing: f64,
    /// Seeds within this share of the city radius are downtown.
    pub downtown: f64,
    /// Share of the other districts with a grid (the rest are organic).
    pub grid_share: f64,
    /// Block sizes (m, range) of grid districts and of the downtown grid, and the jitter of
    /// each gap between grid lines (share of the block size).
    pub block: [f64; 2],
    pub downtown_block: [f64; 2],
    pub block_jitter: f64,
    /// Every n-th grid line is a collector.
    pub collector_every: u32,
    pub downtown_collector_every: u32,
    /// Organic streets: seeds every this many metres along the other streets, segment
    /// lengths (m, range), turning per segment (rad, standard deviation), chance of a branch
    /// on each side per segment, segments per street (range), snapping radius (m) to nodes
    /// ahead, least distance (m) to other streets, chance of a cul-de-sac at a street's end.
    pub organic_seed: f64,
    pub organic_step: [f64; 2],
    pub organic_turn: f64,
    pub organic_branch: f64,
    pub organic_length: [u32; 2],
    pub organic_snap: f64,
    pub organic_spacing: f64,
    pub cul_de_sac: f64,
}

impl Default for DistrictsConfig {
    fn default() -> Self {
        Self {
            spacing: 220.0,
            downtown: 0.45,
            grid_share: 0.4,
            block: [70.0, 110.0],
            downtown_block: [60.0, 90.0],
            block_jitter: 0.15,
            collector_every: 3,
            downtown_collector_every: 3,
            organic_seed: 70.0,
            organic_step: [25.0, 45.0],
            organic_turn: 0.15,
            organic_branch: 0.35,
            organic_length: [3, 8],
            organic_snap: 15.0,
            organic_spacing: 30.0,
            cul_de_sac: 0.3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RoundaboutConfig {
    /// Chance that a suitable junction (3 to 5 streets, one at least a collector, the streets
    /// at least `min_gap_deg` apart) becomes a roundabout; at most `max` of them, `spacing` m
    /// apart.
    pub share: f64,
    pub max: u32,
    pub spacing: f64,
    pub min_gap_deg: f64,
    /// Ring radius (m, range; to the centre line), lane width (m), and the spacing (m) of the
    /// graph's nodes around the ring.
    pub radius: [f64; 2],
    pub lane_width: f64,
    pub arc_step: f64,
}

impl Default for RoundaboutConfig {
    fn default() -> Self {
        Self {
            share: 0.3,
            max: 12,
            spacing: 250.0,
            min_gap_deg: 50.0,
            radius: [14.0, 20.0],
            lane_width: 5.0,
            arc_step: 6.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StreetsConfig {
    /// Segment ends snap to nodes and streets within this radius (m).
    pub snap: f64,
    /// Shortest segment (m) and smallest angle (deg) between streets at a junction.
    pub min_length: f64,
    pub min_angle_deg: f64,
    /// Shortest street (m) between two junctions; closer junctions are merged.
    pub junction_gap: f64,
    /// Dead ends are joined to the streets ahead within this reach (m) and cone (deg); dead
    /// ends shorter than `spur` m (other than cul-de-sacs) are removed.
    pub join_reach: f64,
    pub join_cone_deg: f64,
    pub spur: f64,
    /// Dead ends lie at least this far (m) from every other street (the turning space of
    /// their U-turns); closer ones are cut back.
    pub dead_end_clearance: f64,
    /// Number of radial arterials (range), their segment length (m) and meander (rad per
    /// segment, standard deviation).
    pub radials: [u32; 2],
    pub arterial_step: f64,
    pub arterial_wiggle: f64,
    /// Radius of the ring road as a share of the city radius (0: none).
    pub ring_road: f64,
    /// Streets keep this far (m) from water.
    pub water_clearance: f64,
    pub roundabouts: RoundaboutConfig,
    /// Minimum width (m) of the shoulders that blend the roads into the terrain (wider where
    /// the cut or fill is deep), and the window (m) of the moving average along a road.
    pub shoulder: f64,
    pub profile_window: f64,
}

impl Default for StreetsConfig {
    fn default() -> Self {
        Self {
            snap: 6.0,
            min_length: 15.0,
            min_angle_deg: 30.0,
            junction_gap: 30.0,
            join_reach: 60.0,
            join_cone_deg: 35.0,
            spur: 30.0,
            dead_end_clearance: 20.0,
            radials: [4, 6],
            arterial_step: 60.0,
            arterial_wiggle: 0.08,
            ring_road: 0.6,
            water_clearance: 6.0,
            roundabouts: RoundaboutConfig::default(),
            shoulder: 3.0,
            profile_window: 20.0,
        }
    }
}

/// Cross-sections and geometry limits of one street class. Each street draws its section:
/// lanes per direction (range), then a median, bike lanes and parking lanes each with its
/// chance, and one-way traffic (one direction's lanes only) with its chance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreetClassConfig {
    pub lanes: [u8; 2],
    pub lane_width: f64,
    pub median: f64,
    pub median_share: f64,
    pub bike: f64,
    pub bike_share: f64,
    pub parking: f64,
    pub parking_share: f64,
    pub one_way_share: f64,
    /// Sidewalk width (m) on each side (0: none).
    pub sidewalk: f64,
    /// Maximum longitudinal grade, minimum horizontal radius (m), cross slope.
    pub max_grade: f64,
    pub min_radius: f64,
    pub crown: f64,
}

impl StreetClassConfig {
    fn validate(&self, name: &str) -> Result<(), String> {
        let shares = [self.median_share, self.bike_share, self.parking_share, self.one_way_share];
        let widths = [self.median, self.bike, self.parking, self.sidewalk];
        let ok = (1..=4).contains(&self.lanes[0])
            && self.lanes[0] <= self.lanes[1]
            && self.lanes[1] <= 4
            && self.lane_width > 0.0
            && shares.iter().all(|s| (0.0..=1.0).contains(s))
            && widths.iter().all(|w| *w >= 0.0)
            && self.max_grade > 0.0
            && self.min_radius > 0.0
            && self.crown >= 0.0;
        if ok { Ok(()) } else { Err(format!("classes.{name}: invalid lanes, widths, shares or limits")) }
    }

    /// Draw a street's section.
    fn section(&self, rng: &mut SimRng) -> Section {
        let lanes = self.lanes[0] + rng.below(u64::from(self.lanes[1] - self.lanes[0]) + 1) as u8;
        let median = if rng.chance(self.median_share) { self.median } else { 0.0 };
        let bike = if rng.chance(self.bike_share) { self.bike } else { 0.0 };
        let parking = if rng.chance(self.parking_share) { self.parking } else { 0.0 };
        let one_way = rng.chance(self.one_way_share);
        let sw = [self.sidewalk; 2];
        if one_way {
            // One-way streets keep a single bike and parking lane, on the right.
            Section {
                lanes: [lanes, 0],
                lane_width: self.lane_width,
                median: 0.0,
                bike: [bike, 0.0],
                parking: [parking, 0.0],
                sidewalk: sw,
            }
        } else {
            Section {
                lanes: [lanes; 2],
                lane_width: self.lane_width,
                median,
                bike: [bike; 2],
                parking: [parking; 2],
                sidewalk: sw,
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StreetClassesConfig {
    pub arterial: StreetClassConfig,
    pub collector: StreetClassConfig,
    pub local: StreetClassConfig,
    /// The arterials' continuation beyond the city (class `paved`).
    pub rural: StreetClassConfig,
}

impl Default for StreetClassesConfig {
    fn default() -> Self {
        let local = StreetClassConfig {
            lanes: [1, 1],
            lane_width: 3.0,
            median: 0.0,
            median_share: 0.0,
            bike: 0.0,
            bike_share: 0.0,
            parking: 2.2,
            parking_share: 0.6,
            one_way_share: 0.2,
            sidewalk: 2.0,
            max_grade: 0.12,
            min_radius: 15.0,
            crown: 0.02,
        };
        Self {
            arterial: StreetClassConfig {
                lanes: [1, 2],
                lane_width: 3.5,
                median: 3.0,
                median_share: 0.5,
                bike: 1.8,
                bike_share: 0.3,
                parking_share: 0.2,
                one_way_share: 0.0,
                sidewalk: 3.5,
                max_grade: 0.07,
                min_radius: 60.0,
                ..local.clone()
            },
            collector: StreetClassConfig {
                lane_width: 3.25,
                bike_share: 0.4,
                bike: 1.5,
                parking_share: 0.5,
                one_way_share: 0.0,
                sidewalk: 2.5,
                max_grade: 0.1,
                min_radius: 30.0,
                ..local.clone()
            },
            rural: StreetClassConfig {
                parking_share: 0.0,
                one_way_share: 0.0,
                sidewalk: 0.0,
                max_grade: 0.08,
                min_radius: 30.0,
                ..local.clone()
            },
            local,
        }
    }
}

impl StreetClassesConfig {
    pub fn class(&self, class: RoadClass) -> &StreetClassConfig {
        match class {
            RoadClass::Arterial => &self.arterial,
            RoadClass::Collector => &self.collector,
            RoadClass::Local => &self.local,
            RoadClass::Paved | RoadClass::Gravel | RoadClass::Track => &self.rural,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UrbanMaterialsConfig {
    /// Bare rock above this slope.
    pub rock_slope_deg: f64,
}

impl Default for UrbanMaterialsConfig {
    fn default() -> Self {
        Self { rock_slope_deg: 35.0 }
    }
}

/// Everything that shapes an urban map (the seed is separate).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UrbanConfig {
    /// Edge length of the square map (m), centred on the origin.
    pub size: f64,
    /// Final grid cell (m).
    pub cell: f64,
    pub geo_origin: GeoOrigin,
    pub terrain: TerrainConfig,
    pub erosion: ErosionConfig,
    pub water: WaterConfig,
    pub city: CityConfig,
    pub districts: DistrictsConfig,
    pub streets: StreetsConfig,
    pub classes: StreetClassesConfig,
    /// Field parcels of the countryside around the city.
    pub fields: FieldsConfig,
    pub materials: UrbanMaterialsConfig,
    /// Blocks, lots and what stands on them.
    pub lots: LotsConfig,
    /// Trees, lamps and parking along the streets.
    pub furniture: FurnitureConfig,
}

impl Default for UrbanConfig {
    fn default() -> Self {
        Self::training()
    }
}

/// Named starting points for [`UrbanConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UrbanPreset {
    /// A town on 1 km², for training pools.
    Training,
    /// A city on 2×2 km.
    Showcase,
}

impl UrbanPreset {
    pub fn config(self) -> UrbanConfig {
        match self {
            Self::Training => UrbanConfig::training(),
            Self::Showcase => UrbanConfig::showcase(),
        }
    }
}

impl std::str::FromStr for UrbanPreset {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "training" => Ok(Self::Training),
            "showcase" => Ok(Self::Showcase),
            _ => Err(format!("unknown preset {s:?} (training, showcase)")),
        }
    }
}

impl UrbanConfig {
    /// Gentle relief, as for rural maps.
    fn terrain() -> TerrainConfig {
        TerrainConfig {
            relief: 60.0,
            hills_wavelength: 600.0,
            hills_amplitude: 0.35,
            mountains_wavelength: 1200.0,
            mountains_amplitude: 0.3,
            mask_wavelength: 1200.0,
            mask_threshold: 0.5,
            warp_wavelength: 500.0,
            warp_strength: 40.0,
            detail_amplitude: 0.03,
            ..TerrainConfig::default()
        }
    }

    pub fn training() -> Self {
        Self {
            size: 1024.0,
            cell: 1.0,
            geo_origin: GeoOrigin::default(),
            terrain: Self::terrain(),
            erosion: ErosionConfig::default(),
            water: WaterConfig::default(),
            city: CityConfig::default(),
            districts: DistrictsConfig::default(),
            streets: StreetsConfig::default(),
            classes: StreetClassesConfig::default(),
            fields: FieldsConfig::default(),
            materials: UrbanMaterialsConfig::default(),
            lots: LotsConfig::default(),
            furniture: FurnitureConfig::default(),
        }
    }

    pub fn showcase() -> Self {
        Self {
            size: 2048.0,
            city: CityConfig { radius: 820.0, fringe: 140.0, ..CityConfig::default() },
            streets: StreetsConfig {
                radials: [5, 6],
                roundabouts: RoundaboutConfig { max: 24, ..RoundaboutConfig::default() },
                ..StreetsConfig::default()
            },
            fields: FieldsConfig { spacing: 150.0, ..FieldsConfig::default() },
            ..Self::training()
        }
    }

    /// `preset` with `overrides` merged on top (as [`WildConfig::from_preset`](crate::WildConfig::from_preset)).
    pub fn from_preset(preset: UrbanPreset, overrides: Option<&serde_json::Value>) -> Result<Self, ProcgenError> {
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
        let range = |r: [f64; 2]| r[0] > 0.0 && r[0] <= r[1];
        let check = || -> Result<(), String> {
            if !(0.25..=8.0).contains(&self.cell) {
                return Err("cell must be in [0.25, 8] m".into());
            }
            let cells = self.size / self.cell;
            if !(cells.fract() == 0.0 && (cells as u64).is_multiple_of(2) && (64.0..=16384.0).contains(&cells)) {
                return Err("size / cell must be an even integer in [64, 16384]".into());
            }
            self.terrain.validate()?;
            self.erosion.validate()?;
            let c = &self.city;
            if !(c.radius > 0.0 && c.fringe > 0.0 && (0.0..=0.25).contains(&c.centre_jitter)) {
                return Err("city: radius and fringe must be positive, centre_jitter in [0, 0.25]".into());
            }
            if !((0.0..=0.5).contains(&c.wobble) && (0.0..=1.0).contains(&c.plateau)) {
                return Err("city: wobble in [0, 0.5], plateau in [0, 1]".into());
            }
            let d = &self.districts;
            let shares = [d.downtown, d.grid_share, d.organic_branch, d.cul_de_sac];
            if !(d.spacing > 0.0 && shares.iter().all(|s| (0.0..=1.0).contains(s))) {
                return Err("districts: spacing must be positive, shares in [0, 1]".into());
            }
            if !(range(d.block)
                && range(d.downtown_block)
                && range(d.organic_step)
                && (0.0..0.5).contains(&d.block_jitter))
            {
                return Err(
                    "districts: block sizes and organic_step need 0 < min ≤ max, block_jitter in [0, 0.5)".into()
                );
            }
            let counts = d.collector_every > 0 && d.downtown_collector_every > 0;
            if !(counts && d.organic_length[0] > 0 && d.organic_length[0] <= d.organic_length[1]) {
                return Err("districts: collector_every and organic_length must be positive".into());
            }
            if !(d.organic_seed > 0.0 && d.organic_turn >= 0.0 && d.organic_snap >= 0.0 && d.organic_spacing > 0.0) {
                return Err("districts: organic_seed and organic_spacing must be positive".into());
            }
            let s = &self.streets;
            if !(s.snap > 0.0
                && s.min_length > s.snap
                && s.junction_gap >= s.min_length
                && (0.0..90.0).contains(&s.min_angle_deg))
            {
                return Err(
                    "streets: snap > 0, min_length > snap, junction_gap >= min_length, min_angle_deg in [0, 90)".into(),
                );
            }
            if !(s.join_reach >= 0.0
                && s.join_cone_deg >= 0.0
                && s.spur >= 0.0
                && s.dead_end_clearance >= 0.0
                && s.water_clearance >= 0.0)
            {
                return Err(
                    "streets: join_reach, join_cone_deg, spur, dead_end_clearance and water_clearance must be ≥ 0"
                        .into(),
                );
            }
            if !(s.radials[0] <= s.radials[1]
                && s.radials[1] <= 12
                && s.arterial_step > 0.0
                && s.arterial_wiggle >= 0.0)
            {
                return Err("streets: radials in [0, 12] (min ≤ max), arterial_step > 0".into());
            }
            if !((0.0..1.0).contains(&s.ring_road) && s.shoulder > 0.0 && s.profile_window >= 0.0) {
                return Err("streets: ring_road in [0, 1), shoulder > 0, profile_window ≥ 0".into());
            }
            let r = &s.roundabouts;
            if !((0.0..=1.0).contains(&r.share) && range(r.radius) && r.lane_width > 0.0 && r.arc_step > 0.0) {
                return Err(
                    "streets.roundabouts: share in [0, 1], radius 0 < min ≤ max, lane_width, arc_step > 0".into()
                );
            }
            if !(r.spacing >= 0.0 && (0.0..120.0).contains(&r.min_gap_deg)) {
                return Err("streets.roundabouts: spacing ≥ 0, min_gap_deg in [0, 120)".into());
            }
            let k = &self.classes;
            for (name, class) in
                [("arterial", &k.arterial), ("collector", &k.collector), ("local", &k.local), ("rural", &k.rural)]
            {
                class.validate(name)?;
            }
            self.fields.validate()?;
            self.lots.validate()?;
            self.furniture.validate()?;
            if !(self.materials.rock_slope_deg > 0.0 && self.materials.rock_slope_deg < 90.0) {
                return Err("materials: rock_slope_deg in (0, 90)".into());
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
pub struct UrbanStats {
    /// Wall time of each stage (s).
    pub stages: Vec<(&'static str, f64)>,
    pub vertices: usize,
    pub lakes: usize,
    /// City centre and radius (m).
    pub centre: [f64; 2],
    pub radius: f64,
    /// Districts: downtown, grid, organic.
    pub districts: [usize; 3],
    /// Roads (between nodes) and their total length (m) per class: arterial, collector,
    /// local, paved (beyond the city).
    pub roads: [usize; 4],
    pub road_length: [f64; 4],
    pub nodes: usize,
    pub junctions: usize,
    pub dead_ends: usize,
    pub cul_de_sacs: usize,
    pub roundabouts: usize,
    /// One-way streets (roads), after those that cut the network apart became two-way.
    pub one_way: usize,
    pub height_range: (f64, f64),
    /// Lots per zone (in [`Zone::ALL`] order), buildings, rooftop pads, parking bays in lots
    /// and along streets, and obstacles.
    pub lots: [usize; 6],
    pub buildings: usize,
    pub pads: usize,
    pub bays: [usize; 2],
    pub obstacles: usize,
    /// Cells per material id.
    pub materials: Vec<(String, usize)>,
}

impl UrbanStats {
    fn stage(&mut self, name: &'static str, t: &mut Instant) {
        self.stages.push((name, t.elapsed().as_secs_f64()));
        *t = Instant::now();
    }

    pub fn total_seconds(&self) -> f64 {
        self.stages.iter().map(|s| s.1).sum()
    }
}

/// The network, with every one-way street next to a lane that cannot reach every other lane
/// (or be reached from it) made two-way, until each lane reaches each other one.
fn network_without_strands(
    nodes: Vec<RoadNode>,
    mut roads: Vec<Road>,
    mut sections: Vec<Section>,
    oneway_ok: &dyn Fn(usize) -> bool,
) -> Result<RoadNetwork, ProcgenError> {
    loop {
        let net = RoadNetwork::new(nodes.clone(), roads.clone())
            .and_then(|net| net.with_sections(sections.clone()))
            .map_err(|e| ProcgenError::Config(e.to_string()))?;
        let g = net.lanes();
        let mut bad = vec![false; nodes.len()];
        for l in g.stranded() {
            let lane = &g.lanes()[l as usize];
            bad[lane.from_node as usize] = true;
            bad[lane.to_node as usize] = true;
        }
        let cut: Vec<usize> = (0..roads.len())
            .filter(|&i| {
                sections[i].one_way() && oneway_ok(i) && (bad[roads[i].start as usize] || bad[roads[i].end as usize])
            })
            .collect();
        if cut.is_empty() {
            return Ok(net);
        }
        for i in cut {
            let s = sections[i];
            sections[i] = Section {
                lanes: [s.lanes[0].div_ceil(2); 2],
                median: 0.0,
                bike: [s.bike[0]; 2],
                parking: [s.parking[0]; 2],
                ..s
            };
            roads[i].width = sections[i].width();
        }
    }
}

/// Index of an urban map's class in [`UrbanStats::roads`].
fn class_index(class: RoadClass) -> usize {
    match class {
        RoadClass::Arterial => 0,
        RoadClass::Collector => 1,
        RoadClass::Local => 2,
        _ => 3,
    }
}

/// The city's outline: a circle whose radius wobbles with the direction.
struct City {
    centre: DVec2,
    radius: f64,
    fringe: f64,
    /// Amplitudes and phases of the wobble's harmonics 2 to 4.
    wobble: [(f64, f64); 3],
}

impl City {
    fn new(c: &UrbanConfig, rng: &mut SimRng) -> Self {
        let j = c.city.centre_jitter * c.size;
        let centre = DVec2::new(rng.range(-j, j), rng.range(-j, j));
        let wobble = [0, 1, 2].map(|_| (c.city.wobble * rng.range(0.3, 1.0) / 3.0_f64.sqrt(), rng.range(0.0, TAU)));
        Self { centre, radius: c.city.radius, fringe: c.city.fringe, wobble }
    }

    /// 1 in the city, 0 in the countryside, smooth over the fringe.
    fn share(&self, p: DVec2) -> f64 {
        let d = p - self.centre;
        let a = libm::atan2(d.y, d.x);
        let w: f64 =
            self.wobble.iter().enumerate().map(|(k, &(amp, ph))| amp * libm::sin((k + 2) as f64 * a + ph)).sum();
        let r = self.radius * (1.0 + w);
        1.0 - smoothstep(r, r + self.fringe, d.length())
    }
}

/// A road of the network before profiling: its plan-view points and ends.
struct Piece {
    points: Vec<DVec2>,
    class: RoadClass,
    street: u32,
    start: u32,
    end: u32,
}

/// Generate an urban map. Uses the current rayon pool; the output does not depend on its size.
pub fn generate(config: &UrbanConfig, seed: u64) -> Result<(StaticWorld, UrbanStats), ProcgenError> {
    config.validate()?;
    let c = config;
    let mut stats = UrbanStats::default();
    let mut t = Instant::now();
    let root = Seed::from_u64(seed).child("map/urban");
    let n = c.vertices();
    let origin = DVec2::splat(-0.5 * c.size);
    stats.vertices = n * n;

    // 1. Landform under a plateau.
    let city = City::new(c, &mut root.child("city").rng());
    stats.centre = city.centre.to_array();
    stats.radius = city.radius;
    let keep = c.city.plateau;
    let mid = 0.5 * c.terrain.relief;
    let shape = |p: DVec2, h: f64| {
        let m = (1.0 - keep) * city.share(p);
        h * (1.0 - m) + m * mid
    };
    let land = landform(
        &Land {
            size: c.size,
            cell: c.cell,
            terrain: &c.terrain,
            erosion: &c.erosion,
            water: &c.water,
            shape: Some(&shape),
        },
        &root,
        &mut |name| stats.stage(name, &mut t),
    );
    let (mut heights, lakes) = (land.heights, land.lakes);
    drop(land.accumulation);
    stats.lakes = lakes.count;
    let water = lakes.water;
    let k = &c.classes;
    let max_half = [&k.arterial, &k.collector, &k.local, &k.rural]
        .iter()
        .map(|s| {
            let lanes = f64::from(s.lanes[1]);
            lanes * s.lane_width + 0.5 * s.median + s.bike + s.parking + s.sidewalk
        })
        .fold(0.0, f64::max);
    let near_water = nearby_water(&water, n - 1, ((c.streets.water_clearance + max_half) / c.cell).ceil() as usize);
    let wet = |p: DVec2| {
        let q = ((p - origin) / c.cell).floor();
        let (cx, cy) = ((q.x.max(0.0) as usize).min(n - 2), (q.y.max(0.0) as usize).min(n - 2));
        near_water[cy * (n - 1) + cx] > f32::NEG_INFINITY
    };

    // 2–3. Districts and the street graph.
    let city_share = |p: DVec2| city.share(p);
    let districts = Districts::new(c, city.centre, city.radius, &mut root.child("districts").rng());
    for kind in &districts.kinds {
        stats.districts[*kind as usize] += 1;
    }
    let site =
        Site { c, centre: city.centre, radius: city.radius, city: &city_share, wet: &wet, districts: &districts };
    let (graph, rings) = layout::layout(&site, &root.child("streets"));
    stats.roundabouts = rings.len();
    stats.stage("streets", &mut t);

    // 4. Roads between the graph's junctions.
    let (pieces, node_ids, kinds) = extract(&graph, &rings);
    let pieces: Vec<Piece> = pieces
        .into_iter()
        .map(|mut p| {
            p.points = if let Some(ring) = rings.iter().find(|r| r.street == p.street) {
                arc(ring, p.points[0], *p.points.last().expect("points"))
            } else if p.points.len() > 2 {
                // Round the corners as widely as the smoothing stays near the graph's chain (it
                // must not meet another street); tighter radii first, then each corner on its
                // own as widely as it may.
                let r = k.class(p.class).min_radius;
                [r, 0.5 * r, 0.25 * r, 8.0]
                    .into_iter()
                    .map(|r| smooth_path(&p.points, r))
                    .find(|smooth| smooth.iter().all(|&q| distance_to(&p.points, q) < CHAIN_DEVIATION))
                    .unwrap_or_else(|| resample(&fillet(&p.points, r, CHAIN_DEVIATION - 0.1), 1.0))
            } else {
                resample(&p.points, 1.0)
            };
            p
        })
        .collect();
    let sections: Vec<Section> = pieces
        .iter()
        .map(|p| {
            if rings.iter().any(|r| r.street == p.street) {
                let r = &c.streets.roundabouts;
                // Counter-clockwise: the outside is on the right.
                Section {
                    lanes: [1, 0],
                    lane_width: r.lane_width,
                    median: 0.0,
                    bike: [0.0; 2],
                    parking: [0.0; 2],
                    sidewalk: [k.collector.sidewalk, 0.0],
                }
            } else {
                let mut rng = root.child("sections").child_index(u64::from(p.street)).child_index(p.class as u64).rng();
                k.class(p.class).section(&mut rng)
            }
        })
        .collect();
    stats.stage("roads", &mut t);

    // 5. Profiles.
    let hs = Heights { h: &heights, n, origin, cell: c.cell };
    let mut node_z: Vec<f64> = node_ids
        .iter()
        .map(|&g| {
            let p = graph.p(g);
            let ring = (0..8).map(|k| {
                let a = TAU * k as f64 / 8.0;
                hs.at(p + 5.0 * DVec2::new(libm::cos(a), libm::sin(a)))
            });
            (hs.at(p) + ring.sum::<f64>()) / 9.0
        })
        .collect();
    let lengths: Vec<f64> = pieces.iter().map(|p| p.points.windows(2).map(|w| w[0].distance(w[1])).sum()).collect();
    reach_nodes(c, &pieces, &lengths, &mut node_z);
    let roads: Vec<Road> = pieces
        .iter()
        .zip(&sections)
        .map(|(p, s)| {
            let cc = k.class(p.class);
            let (z0, z1) = (node_z[p.start as usize], node_z[p.end as usize]);
            let z = profile(&p.points, &hs, z0, z1, cc.max_grade, c.streets.profile_window);
            let points = p.points.iter().zip(z).map(|(q, z)| q.extend(z)).collect();
            Road { class: p.class, width: s.width(), start: p.start, end: p.end, line: Polyline::new(points) }
        })
        .collect();
    let nodes: Vec<RoadNode> = node_ids
        .iter()
        .zip(&node_z)
        .zip(&kinds)
        .map(|((&g, &z), &kind)| RoadNode { position: graph.p(g).extend(z), kind })
        .collect();
    stats.nodes = nodes.len();
    stats.junctions = nodes.iter().filter(|n| n.kind == NodeKind::Junction).count();
    stats.dead_ends = node_ids.iter().filter(|&&g| graph.degree(g) == 1).count();
    stats.cul_de_sacs =
        node_ids.iter().filter(|&&g| graph.degree(g) == 1 && graph.nodes[g as usize].cul_de_sac).count();
    for r in &roads {
        stats.roads[class_index(r.class)] += 1;
        stats.road_length[class_index(r.class)] += r.line.length();
    }
    let on_ring = |i: usize| rings.iter().any(|r| r.street == pieces[i].street);
    let network = network_without_strands(nodes, roads, sections, &|i| !on_ring(i))?;
    stats.one_way = (0..pieces.len()).filter(|&i| network.section(i).one_way() && !on_ring(i)).count();
    stats.stage("profiles", &mut t);

    // 6. Terrain blending.
    blend(c, &mut heights, n, origin, &network, max_half);
    stats.stage("blending", &mut t);

    // 7. Lots, buildings and street furniture.
    let hs = Heights { h: &heights, n, origin, cell: c.cell };
    let lot_wet = |p: DVec2| {
        let q = ((p - origin) / c.cell).floor();
        let (cx, cy) = ((q.x.max(0.0) as usize).min(n - 2), (q.y.max(0.0) as usize).min(n - 2));
        !water[cy * (n - 1) + cx].is_nan()
    };
    let town = sites::Town {
        centre: city.centre,
        radius: city.radius,
        share: &city_share,
        districts: &districts,
        wet: &lot_wet,
        rings: rings.iter().map(|r| (r.centre, r.radius)).collect(),
    };
    let built = sites::generate(c, &network, &hs, &town, &root.child("sites"));
    for lot in &built.sites.lots {
        stats.lots[Zone::ALL.iter().position(|&z| z == lot.zone).expect("zone")] += 1;
    }
    stats.buildings = built.sites.buildings.len();
    stats.pads = built.sites.pads.len();
    for b in &built.sites.bays {
        stats.bays[usize::from(b.kind == BayKind::Street)] += 1;
    }
    stats.obstacles = built.obstacles.len();
    stats.stage("sites", &mut t);

    // 8. Materials.
    let slope = |p: DVec2| {
        let d = 10.0;
        let gx = hs.at(p + DVec2::X * d) - hs.at(p - DVec2::X * d);
        let gy = hs.at(p + DVec2::Y * d) - hs.at(p - DVec2::Y * d);
        (gx * gx + gy * gy).sqrt() / (2.0 * d)
    };
    let parcels = Parcels::new(&c.fields, c.size, &slope, &root.child("parcels"));
    let materials = materials(c, origin, &heights, &water, &network, &city, &parcels, &built.zones, max_half);
    let table = MaterialTable::rural();
    let mut counts = [0usize; 256];
    for id in &materials {
        counts[id.0 as usize] += 1;
    }
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

    let mut meta = MapMeta::new("urban", "urban", seed);
    meta.generator_version = URBAN_VERSION;
    meta.geo_origin = c.geo_origin;
    let world = StaticWorld::new(meta, grid, ObstacleSet::new(built.obstacles), table)
        .with_roads(network)
        .with_sites(built.sites);
    stats.stage("obstacle index", &mut t);
    Ok((world, stats))
}

/// Farthest a road's line may run from its chain in the street graph (m).
const CHAIN_DEVIATION: f64 = 2.5;

/// `path` with each corner rounded by a circular arc (points about 1 m apart) of radius
/// `radius`, or less where the arc would come more than `deviation` from the corner or need
/// more than half of a neighbouring segment.
fn fillet(path: &[DVec2], radius: f64, deviation: f64) -> Vec<DVec2> {
    let mut out = vec![path[0]];
    for i in 1..path.len() - 1 {
        let (a, p, b) = (path[i - 1], path[i], path[i + 1]);
        let (d1, d2) = ((p - a).normalize_or_zero(), (b - p).normalize_or_zero());
        let turn = d1.perp_dot(d2).atan2(d1.dot(d2));
        let half = 0.5 * turn.abs();
        if half < 1e-3 || half > 0.5 * std::f64::consts::PI - 1e-3 {
            out.push(p);
            continue;
        }
        // Deviation from the corner r(1/cos − 1); tangent length r·tan.
        let room = 0.5 * a.distance(p).min(p.distance(b));
        let r = radius.min(deviation / (1.0 / half.cos() - 1.0)).min(room / half.tan());
        let t = r * half.tan();
        let start = p - d1 * t;
        let centre = start + d1.perp() * r * turn.signum();
        let steps = ((r * turn.abs()).ceil() as usize).max(1);
        let from = (start - centre).to_angle();
        for k in 0..=steps {
            let phi = from + turn * k as f64 / steps as f64;
            out.push(centre + DVec2::from_angle(phi) * r);
        }
    }
    out.push(*path.last().expect("a path"));
    out
}

/// Distance from `q` to the polyline `line`.
fn distance_to(line: &[DVec2], q: DVec2) -> f64 {
    line.windows(2).map(|w| graph::closest_on_segment(w[0], w[1], q).distance(q)).fold(f64::INFINITY, f64::min)
}

/// Points every metre or so along a roundabout's ring, counter-clockwise from `a` to `b`.
fn arc(ring: &Ring, a: DVec2, b: DVec2) -> Vec<DVec2> {
    let angle = |p: DVec2| libm::atan2(p.y - ring.centre.y, p.x - ring.centre.x);
    let a0 = angle(a);
    let sweep = (angle(b) - a0).rem_euclid(TAU);
    let steps = ((sweep * ring.radius).ceil() as usize).max(2);
    let mut out: Vec<DVec2> = (0..=steps)
        .map(|k| {
            let t = a0 + sweep * k as f64 / steps as f64;
            ring.centre + ring.radius * DVec2::new(libm::cos(t), libm::sin(t))
        })
        .collect();
    // The ends exactly on the nodes.
    out[0] = a;
    out[steps] = b;
    out
}

/// The graph's chains between nodes that end roads: junctions and dead ends, fixed nodes,
/// roundabout entries and changes of class or street. Returns the pieces (ring pieces running
/// counter-clockwise), the graph node of each network node, and the network nodes' kinds.
fn extract(g: &Graph, rings: &[Ring]) -> (Vec<Piece>, Vec<NodeId>, Vec<NodeKind>) {
    let is_ring = |street: u32| rings.iter().any(|r| r.street == street);
    let mut breaks: Vec<bool> = (0..g.nodes.len() as NodeId)
        .map(|v| {
            let node = &g.nodes[v as usize];
            match node.edges.as_slice() {
                [] => false,
                [e, f] => {
                    let (e, f) = (&g.edges[*e as usize], &g.edges[*f as usize]);
                    node.fixed || e.class != f.class || e.street != f.street
                }
                _ => true,
            }
        })
        .collect();
    let mut visited = vec![false; g.edges.len()];
    let mut ids: Vec<u32> = vec![u32::MAX; g.nodes.len()];
    let mut node_ids: Vec<NodeId> = Vec::new();
    let mut pieces = Vec::new();
    let mut id = |v: NodeId, node_ids: &mut Vec<NodeId>| {
        if ids[v as usize] == u32::MAX {
            ids[v as usize] = node_ids.len() as u32;
            node_ids.push(v);
        }
        ids[v as usize]
    };
    // Two passes: chains from the break nodes, then closed loops of degree-2 nodes (each
    // broken at its lowest node).
    for pass in 0..2 {
        for v in 0..g.nodes.len() as NodeId {
            if pass == 1 && !g.nodes[v as usize].edges.iter().any(|&e| !visited[e as usize]) {
                continue;
            }
            if pass == 1 {
                breaks[v as usize] = true;
            }
            if !breaks[v as usize] {
                continue;
            }
            for &e0 in &g.nodes[v as usize].edges {
                if visited[e0 as usize] {
                    continue;
                }
                let (mut cur, mut e) = (v, e0);
                let mut nodes = vec![v];
                let mut edges = vec![e0];
                loop {
                    visited[e as usize] = true;
                    cur = g.other(e, cur);
                    nodes.push(cur);
                    if breaks[cur as usize] {
                        break;
                    }
                    e = *g.nodes[cur as usize].edges.iter().find(|&&f| f != e).expect("degree 2");
                    edges.push(e);
                }
                let edge = &g.edges[e0 as usize];
                let (class, street) = (edge.class, edge.street);
                // Ring pieces run the way their edges were laid (counter-clockwise).
                if is_ring(street) && edge.a != v {
                    nodes.reverse();
                }
                // A chain back to its own start is split in two.
                let halves: Vec<&[NodeId]> = if nodes.first() == nodes.last() {
                    let mid = nodes.len() / 2;
                    vec![&nodes[..=mid], &nodes[mid..]]
                } else {
                    vec![&nodes[..]]
                };
                for h in halves {
                    let (a, b) = (h[0], h[h.len() - 1]);
                    let (start, end) = (id(a, &mut node_ids), id(b, &mut node_ids));
                    pieces.push(Piece { points: h.iter().map(|&n| g.p(n)).collect(), class, street, start, end });
                }
            }
        }
    }
    let kinds = node_ids
        .iter()
        .map(|&v| {
            let node = &g.nodes[v as usize];
            if node.edges.iter().any(|&e| is_ring(g.edges[e as usize].street)) {
                NodeKind::Roundabout
            } else if node.edges.len() == 1 {
                NodeKind::End
            } else {
                NodeKind::Junction
            }
        })
        .collect();
    (pieces, node_ids, kinds)
}

/// Move the node heights so that every road can climb from one end to the other within 90 %
/// of its class's grade limit (as for rural maps; see `rural::reach_nodes`).
fn reach_nodes(c: &UrbanConfig, pieces: &[Piece], lengths: &[f64], node_z: &mut [f64]) {
    for _ in 0..200 {
        let mut moved = false;
        for (p, &len) in pieces.iter().zip(lengths) {
            let (a, b) = (p.start as usize, p.end as usize);
            let allowed = 0.9 * c.classes.class(p.class).max_grade * len;
            let excess = (node_z[b] - node_z[a]).abs() - allowed;
            if excess <= 1e-9 || a == b {
                continue;
            }
            let dir = (node_z[b] - node_z[a]).signum();
            node_z[a] += dir * 0.5 * excess;
            node_z[b] -= dir * 0.5 * excess;
            moved = true;
        }
        if !moved {
            break;
        }
    }
}

/// Cut and fill the roads into the terrain: the carriageway with its crown, the sidewalks
/// flat at the carriageway's edge height, then a shoulder falling off to the terrain.
/// Distance (m) beyond the nearest carriageway over which the surfaces of other roads blend
/// into its own.
const JUNCTION_BLEND: f64 = 8.0;

/// Added to the distances (m) by which the road surfaces are weighted where roads meet: the
/// larger, the more gently the surface turns from one road's height to another's.
const JUNCTION_SOFTEN: f64 = 3.0;

fn blend(c: &UrbanConfig, heights: &mut [f32], n: usize, origin: DVec2, net: &RoadNetwork, max_half: f64) {
    let s = &c.streets;
    let reach = max_half + 0.5 + s.shoulder.max(15.0);
    // The junctions' areas (the discs the lanes leave free, crossed by their connectors) are
    // kept at road level like the carriageways.
    let junctions: Vec<(DVec2, f64)> = net
        .lanes()
        .junctions()
        .iter()
        .map(|j| (net.nodes()[j.node as usize].position.truncate(), j.radius + 1.0))
        .collect();
    // The surface of a road at a point, crowned.
    let surface = |rp: &RoadPoint| {
        let r = &net.roads()[rp.road as usize];
        rp.projection.point.z - c.classes.class(r.class).crown * rp.projection.distance.min(0.5 * r.width)
    };
    heights.par_chunks_mut(n).enumerate().for_each(|(iy, row)| {
        let y = origin.y + iy as f64 * c.cell;
        let mut near = Vec::new();
        for (ix, h) in row.iter_mut().enumerate() {
            let p = DVec2::new(origin.x + ix as f64 * c.cell, y);
            net.nearest_each(p, reach, &mut near);
            if near.is_empty() {
                continue;
            }
            // Distance beyond each road's carriageway, and beyond its flat band (carriageway,
            // the sidewalk on that side and half a metre); the left of a road (positive
            // offsets) is the right of travel against it.
            let beyond = |q: &RoadPoint| q.projection.distance - 0.5 * net.roads()[q.road as usize].width;
            let flat = |q: &RoadPoint| {
                let sidewalk = net.section(q.road as usize).sidewalk[usize::from(q.projection.offset > 0.0)];
                beyond(q) - sidewalk - 0.5
            };
            // The target: the nearest carriageway's surface; in and around junctions (fading
            // out over `JUNCTION_BLEND` beyond their areas), the roads' surfaces weighted by
            // the inverse square of the distance to their carriageways (plus
            // `JUNCTION_SOFTEN`), each faded out from the nearest carriageway's distance to
            // `JUNCTION_BLEND` beyond it, so that it runs smoothly from one road to the next.
            let area = junctions.iter().map(|&(c, r)| p.distance(c) - r).fold(f64::INFINITY, f64::min);
            let mix = 1.0 - smoothstep(0.0, JUNCTION_BLEND, area);
            let nearest = near
                .iter()
                .min_by(|a, b| {
                    (a.projection.distance, a.road).partial_cmp(&(b.projection.distance, b.road)).expect("finite")
                })
                .expect("a road");
            let edge = beyond(nearest).max(0.0);
            let (mut sum, mut total) = (0.0, 0.0);
            for q in &near {
                let b = beyond(q).max(0.0);
                let share = if q.road == nearest.road { 1.0 } else { mix };
                let w = share * (1.0 - smoothstep(0.0, JUNCTION_BLEND, b - edge)) / (b + JUNCTION_SOFTEN).powi(2);
                sum += w * surface(q);
                total += w;
            }
            let target = sum / total;
            let inside = area.min(near.iter().map(flat).fold(f64::INFINITY, f64::min));
            let h0 = f64::from(*h);
            let shoulder = s.shoulder.max(1.5 * (target - h0).abs());
            let w = 1.0 - smoothstep(0.0, shoulder, inside);
            *h = (h0 + w * (target - h0)) as f32;
        }
    });
}

/// Material per cell.
#[allow(clippy::too_many_arguments)]
fn materials(
    c: &UrbanConfig,
    origin: DVec2,
    heights: &[f32],
    water: &[f32],
    net: &RoadNetwork,
    city: &City,
    parcels: &Parcels,
    zones: &sites::ZoneRaster,
    max_half: f64,
) -> Vec<MaterialId> {
    let n = c.vertices();
    let cw = n - 1;
    let rock = libm::tan(c.materials.rock_slope_deg.to_radians());
    let inv = 1.0 / c.cell;
    let headland = c.fields.headland;
    let mut out = vec![MaterialId::GRASS; cw * cw];
    out.par_chunks_mut(cw).enumerate().for_each(|(cy, row)| {
        let y = origin.y + (cy as f64 + 0.5) * c.cell;
        for (cx, mat) in row.iter_mut().enumerate() {
            let i = cy * n + cx;
            let (h00, h10, h01, h11) =
                (heights[i] as f64, heights[i + 1] as f64, heights[i + n] as f64, heights[i + n + 1] as f64);
            let gx = 0.5 * (h10 - h00 + h11 - h01) * inv;
            let gy = 0.5 * (h01 - h00 + h11 - h10) * inv;
            let slope = (gx * gx + gy * gy).sqrt();
            let z = 0.25 * (h00 + h10 + h01 + h11);
            let p = DVec2::new(origin.x + (cx as f64 + 0.5) * c.cell, y);
            let k = cy * cw + cx;
            // Where the cell lies across the nearest road: carriageway, sidewalk or beyond.
            let road = net.nearest(p, max_half + headland).map(|rp| {
                let half = 0.5 * net.roads()[rp.road as usize].width;
                let sidewalk = net.section(rp.road as usize).sidewalk[usize::from(rp.projection.offset > 0.0)];
                (rp.projection.distance - half, sidewalk)
            });
            *mat = if !water[k].is_nan() {
                if water[k] as f64 - z < 0.5 { MaterialId::SAND } else { MaterialId::MUD }
            } else if road.is_some_and(|r| r.0 <= 0.0) {
                MaterialId::ASPHALT
            } else if road.is_some_and(|r| r.0 <= r.1) {
                MaterialId::CONCRETE
            } else if slope > rock {
                MaterialId::ROCK
            } else if let Some(zone) = zones.at(p) {
                match zone {
                    Zone::Downtown | Zone::Commercial => MaterialId::CONCRETE,
                    Zone::Industrial => MaterialId::GRAVEL,
                    Zone::Parking => MaterialId::ASPHALT,
                    Zone::Residential | Zone::Park => MaterialId::GRASS,
                }
            } else if city.share(p) >= 0.5 || road.is_some_and(|r| r.0 < headland) {
                MaterialId::GRASS
            } else {
                let (parcel, edge) = parcels.locate(p);
                let kind = parcels.kinds[parcel];
                if edge < headland && kind != ParcelKind::Woods { MaterialId::GRASS } else { kind.material() }
            };
        }
    });
    out
}
