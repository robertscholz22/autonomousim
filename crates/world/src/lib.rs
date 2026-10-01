//! Runtime static-world data (terrain, obstacles, roads, materials) and environment models.

pub mod environment;
pub mod geodesy;
pub mod heightgrid;
pub mod lanes;
pub mod mapfile;
pub mod obstacles;
pub mod roads;
pub mod signals;
pub mod sites;
pub mod static_world;
pub mod testworlds;
pub mod tiles;
pub mod walkways;

pub use geodesy::{GeoOrigin, Geodetic};
pub use heightgrid::HeightGrid;
pub use lanes::{Area, Connector, Crossing, Junction, JunctionKind, Lane, LaneGraph, Turn};
pub use mapfile::{MapFileError, MapHash};
pub use obstacles::{Obstacle, ObstacleClass, ObstacleSet, ObstacleShape};
pub use roads::{NodeKind, Polyline, Road, RoadClass, RoadNetwork, RoadNode, RoadPoint, Route, Section};
pub use signals::{Controller, Light, Phase};
pub use sites::{BayKind, Building, Lot, Pad, ParkingBay, Roof, Sites, Zone};
pub use static_world::{MapMeta, MapObstacles, MapTerrain, StaticWorld};
pub use tiles::{Tile, TileLayout, TileSource, TiledMap};
pub use walkways::{Place, PlaceKind, WalkEdge, WalkKind, WalkNode, WalkRoute, Walkways};
