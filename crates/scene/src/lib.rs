//! Renderer-independent scene meshes for the viewer (and headless cameras later): terrain
//! chunks with levels of detail, water, road ribbons, merged obstacle meshes and vehicle
//! visuals.
//!
//! Meshes are plain arrays ([`MeshData`]) in the simulator's frames (ENU for the world, FLU
//! for vehicles); renderers convert them (see `autonomousim_core::math::frames`).

pub mod mesh;
pub mod props;
pub mod roads;
pub mod terrain;

pub use mesh::MeshData;
pub use props::{MultirotorVisual, PropDetail};
pub use terrain::Chunk;
