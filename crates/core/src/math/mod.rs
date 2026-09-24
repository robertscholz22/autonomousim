//! Spatial algebra, rigid transforms, inertias, small linear algebra and frame conventions.

pub mod frames;
pub mod inertia;
pub mod linalg;
pub mod quat;
pub mod spatial;
pub mod transform;

#[cfg(test)]
pub(crate) mod testutil;

pub use inertia::{ArticulatedInertia, RigidInertia, outer};
pub use linalg::{DenseMatrix, Mat6, NotPositiveDefinite, SingularMatrix, mat3_max_abs, skew};
pub use spatial::{SpatialForce, SpatialMotion};
pub use transform::{Pose, Xform};
