//! Reduced-coordinate multibody dynamics (Featherstone spatial-vector algorithms).
//!
//! * [`MultibodyModel`]: immutable kinematic tree (shared between instances).
//! * [`MbState`]: generalised coordinates `(q, v)` of one instance.
//! * [`KcTable`]: suspension kinematics for [`JointType::KcTravel`].
//! * [`aba`]: forward dynamics O(n); [`rnea`]: inverse dynamics; [`crba`]: mass matrix.
//! * Gravity enters as a fictitious upward base acceleration, so link accelerations in
//!   [`AbaWorkspace::acc`] are *proper* accelerations (what an accelerometer measures, up to
//!   the `ω × v` term that converts spatial to classical acceleration).

mod aba;
mod energy;
mod integrator;
mod joint;
mod kc;
mod kinematics;
mod model;
mod rnea;

pub use aba::{AbaWorkspace, DynamicsError, aba, aba_with_kinematics, single_body_acceleration};
pub use energy::{center_of_mass_world, kinetic_energy, potential_energy, spatial_momentum_world};
pub use integrator::{
    Integrator, Rk4Workspace, midpoint_gyroscopic_update, normalize, qdot, rk4, semi_implicit_euler,
    semi_implicit_euler_with_momentum,
};
pub use joint::JointType;
pub use kc::{KcPoint, KcTable, KcTableSpec};
pub use kinematics::{KinCache, forward_kinematics};
pub use model::{Link, MbState, MultibodyModel};
pub use rnea::{crba, rnea};
