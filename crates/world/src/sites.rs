//! What an urban map holds beside its roads and obstacles: lots with their zoning, buildings
//! (their shapes are obstacles), rooftop landing pads and parking bays. Rural and wild maps
//! have none.

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

/// The use of a lot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Zone {
    /// Towers with high coverage.
    Downtown,
    /// Mid-rise blocks.
    Commercial,
    /// Houses with gardens.
    Residential,
    /// Large low halls and yards.
    Industrial,
    /// Trees and grass.
    Park,
    /// A parking lot with marked bays.
    Parking,
}

impl Zone {
    pub const ALL: [Zone; 6] =
        [Zone::Downtown, Zone::Commercial, Zone::Residential, Zone::Industrial, Zone::Park, Zone::Parking];
}

/// A lot: a piece of a block with frontage on a street.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lot {
    pub zone: Zone,
    /// Centroid and area (m²).
    pub centre: DVec2,
    pub area: f64,
    /// The street it fronts and the middle of its frontage.
    pub road: u32,
    pub front: DVec2,
}

/// Roof shape of a building.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Roof {
    /// Flat, behind a parapet where tall enough; landable.
    Flat,
    /// A gable over the footprint, the ridge along its longer side.
    Gable,
}

/// A building: its bounding footprint (a rectangle; the shape may be an L, a U or a
/// courtyard within it) and heights. Its collision shapes are obstacles
/// `obstacles[0]..obstacles[1]` of the map.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Building {
    pub lot: u32,
    pub centre: DVec2,
    /// Heading of the footprint's x axis (along the street).
    pub yaw: f64,
    /// Footprint size along x and y (m).
    pub size: DVec2,
    /// Bottom of the walls (below the lowest ground under the footprint) and height of the
    /// walls above it (the flat roof or the eaves).
    pub base: f64,
    pub height: f64,
    pub storeys: u8,
    pub roof: Roof,
    pub obstacles: [u32; 2],
}

impl Building {
    /// Height of the flat roof or the eaves.
    pub fn top(&self) -> f64 {
        self.base + self.height
    }
}

/// A landing pad on a flat roof: a square of side `2·half` whose surface lies at `centre.z`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pad {
    pub centre: DVec3,
    pub yaw: f64,
    pub half: f64,
    pub building: u32,
}

/// Where a parking bay lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BayKind {
    /// In a parking lot, nose in.
    Lot,
    /// In a street's parking lane, along the traffic.
    Street,
}

/// A parking bay: a rectangle on the ground, `size` = (length, width), the length along `yaw`
/// (the heading of a car parked in it).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParkingBay {
    pub centre: DVec3,
    pub yaw: f64,
    pub size: DVec2,
    pub kind: BayKind,
}

impl ParkingBay {
    /// The point at `local` (x along the bay, y to its left) in world coordinates.
    pub fn point(&self, local: DVec2) -> DVec2 {
        self.centre.truncate() + DVec2::from_angle(self.yaw).rotate(local)
    }

    /// Corners, counter-clockwise.
    pub fn corners(&self) -> [DVec2; 4] {
        let (l, w) = (0.5 * self.size.x, 0.5 * self.size.y);
        [(-l, -w), (l, -w), (l, w), (-l, w)].map(|(x, y)| self.point(DVec2::new(x, y)))
    }
}

/// Lots, buildings, pads and bays of a map.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sites {
    pub lots: Vec<Lot>,
    pub buildings: Vec<Building>,
    pub pads: Vec<Pad>,
    pub bays: Vec<ParkingBay>,
}

impl Sites {
    pub fn is_empty(&self) -> bool {
        self.lots.is_empty() && self.buildings.is_empty() && self.pads.is_empty() && self.bays.is_empty()
    }
}
