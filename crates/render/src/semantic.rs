//! Semantic classes of the camera's class image (one byte per pixel).
//!
//! The table is part of the observation format: classes are only appended, and
//! [`SEMANTIC_VERSION`] changes with it.

use autonomousim_core::material::MaterialId;
use autonomousim_world::obstacles::tags;
use autonomousim_world::{Obstacle, ObstacleClass};

/// Version of the class table below.
pub const SEMANTIC_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SemanticClass {
    /// Nothing within the far plane.
    Sky = 0,
    /// Terrain: grass, meadow and fields.
    Grass = 1,
    /// Terrain: forest floor.
    ForestFloor = 2,
    /// Terrain: rock and scree.
    Rock = 3,
    /// Terrain: sand and mud.
    Soil = 4,
    /// Terrain: snow.
    Snow = 5,
    Water = 6,
    Road = 7,
    /// Tree trunks and branches.
    Trunk = 8,
    /// Tree crowns and shrubs.
    Canopy = 9,
    /// Rocks and boulders standing on the ground.
    Boulder = 10,
    /// Buildings, walls and other built structures.
    Building = 11,
    /// The camera's own vehicle.
    OwnVehicle = 12,
    /// Any other agent's vehicle.
    Vehicle = 13,
    /// Markers placed by the scenario, e.g. landing pads under goals (version 2).
    Marker = 14,
}

impl SemanticClass {
    pub const ALL: [SemanticClass; 15] = [
        Self::Sky,
        Self::Grass,
        Self::ForestFloor,
        Self::Rock,
        Self::Soil,
        Self::Snow,
        Self::Water,
        Self::Road,
        Self::Trunk,
        Self::Canopy,
        Self::Boulder,
        Self::Building,
        Self::OwnVehicle,
        Self::Vehicle,
        Self::Marker,
    ];

    pub fn id(self) -> u8 {
        self as u8
    }

    pub fn from_id(id: u8) -> Option<Self> {
        Self::ALL.get(id as usize).copied()
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Sky => "sky",
            Self::Grass => "grass",
            Self::ForestFloor => "forest_floor",
            Self::Rock => "rock",
            Self::Soil => "soil",
            Self::Snow => "snow",
            Self::Water => "water",
            Self::Road => "road",
            Self::Trunk => "trunk",
            Self::Canopy => "canopy",
            Self::Boulder => "boulder",
            Self::Building => "building",
            Self::OwnVehicle => "own_vehicle",
            Self::Vehicle => "vehicle",
            Self::Marker => "marker",
        }
    }
}

/// Class of terrain of material `m`.
pub fn terrain_class(m: MaterialId) -> SemanticClass {
    match m {
        MaterialId::ROCK | MaterialId::SCREE => SemanticClass::Rock,
        MaterialId::SNOW => SemanticClass::Snow,
        MaterialId::SAND | MaterialId::MUD | MaterialId::DIRT | MaterialId::PLOWED => SemanticClass::Soil,
        MaterialId::FOREST_FLOOR => SemanticClass::ForestFloor,
        MaterialId::WATER => SemanticClass::Water,
        MaterialId::ASPHALT | MaterialId::CONCRETE | MaterialId::GRAVEL => SemanticClass::Road,
        MaterialId::WOOD | MaterialId::METAL => SemanticClass::Building,
        MaterialId::FOLIAGE => SemanticClass::Canopy,
        _ => SemanticClass::Grass,
    }
}

/// Class of an obstacle: by its tag, else by its class and material.
pub fn obstacle_class(o: &Obstacle) -> SemanticClass {
    match o.tag {
        tags::TRUNK => SemanticClass::Trunk,
        tags::CANOPY | tags::CANOPY_BROADLEAF => SemanticClass::Canopy,
        tags::HEDGE if o.class == ObstacleClass::Foliage => SemanticClass::Canopy,
        tags::HEDGE => SemanticClass::Trunk,
        tags::ROCK => SemanticClass::Boulder,
        tags::PILLAR | tags::WALL | tags::FENCE | tags::BUILDING | tags::SILO => SemanticClass::Building,
        _ if o.class == ObstacleClass::Foliage => SemanticClass::Canopy,
        _ => match o.material {
            MaterialId::ROCK | MaterialId::SCREE => SemanticClass::Boulder,
            MaterialId::WOOD => SemanticClass::Trunk,
            _ => SemanticClass::Building,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_dense_and_round_trip() {
        for (i, c) in SemanticClass::ALL.iter().enumerate() {
            assert_eq!(c.id() as usize, i);
            assert_eq!(SemanticClass::from_id(c.id()), Some(*c));
        }
        assert_eq!(SemanticClass::from_id(SemanticClass::ALL.len() as u8), None);
    }
}
