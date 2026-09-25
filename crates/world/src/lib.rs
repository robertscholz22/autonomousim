//! Runtime static-world data (terrain, obstacles, roads, materials) and environment models.

pub mod environment;
pub mod geodesy;
pub mod heightgrid;
pub mod mapfile;
pub mod obstacles;
pub mod roads;
pub mod static_world;
pub mod testworlds;

pub use geodesy::{GeoOrigin, Geodetic};
pub use heightgrid::HeightGrid;
pub use mapfile::{MapFileError, MapHash};
pub use obstacles::{Obstacle, ObstacleClass, ObstacleSet, ObstacleShape};
pub use roads::{NodeKind, Polyline, Road, RoadClass, RoadNetwork, RoadNode, Route};
pub use static_world::{MapMeta, StaticWorld};
