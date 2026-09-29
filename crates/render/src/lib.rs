//! Headless rendering of [`scene`](autonomousim_scene) meshes for camera sensors.
//!
//! One pass draws three images at once (multiple render targets): RGB (sRGB-encoded bytes),
//! linear depth along the optical axis (m; 0 where nothing was hit) and a semantic class per
//! pixel ([`SemanticClass`]). Cameras are ideal pinholes ([`Intrinsics`]).
//!
//! Frames: the world is ENU and a camera's frame is FLU with the optical axis along +x, so a
//! camera mounted without rotation looks where its vehicle's nose points. Image columns grow
//! to the camera's right (−y) and rows downward (−z); pixel centres are at half-integers.
//! Every transform is composed in f64 relative to the camera before it is rounded to f32, so
//! precision does not depend on the distance from the world origin.
//!
//! Determinism: the same adapter and driver give bit-identical images. Tests and golden
//! hashes use the software rasterizer (lavapipe; [`AdapterChoice::Software`]), which does not
//! depend on the machine.

pub mod camera;
pub mod context;
pub mod renderer;
pub mod semantic;
pub mod world;

pub use camera::{CameraPose, Intrinsics};
pub use context::{AdapterChoice, GpuContext, RenderError};
pub use renderer::{Draw, Frame, GpuMesh, Renderer, Shading, View};
pub use semantic::{SemanticClass, obstacle_class, terrain_class};
pub use world::{GpuRig, GpuWorld, WorldOptions};
