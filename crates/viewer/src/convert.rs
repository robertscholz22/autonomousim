//! Conversions between the simulator (f64, ENU/FLU) and Bevy (f32, Y up).

use autonomousim_core::math::frames::{bevy_to_enu_vec, enu_to_bevy_quat, enu_to_bevy_vec};
use autonomousim_scene::MeshData;
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// ENU (or FLU) vector → Bevy vector.
pub fn vec(v: glam::DVec3) -> Vec3 {
    Vec3::from_array(enu_to_bevy_vec(v))
}

/// Bevy vector → ENU vector.
pub fn enu(v: Vec3) -> glam::DVec3 {
    bevy_to_enu_vec(v.to_array())
}

/// ENU rotation → Bevy rotation.
pub fn quat(q: glam::DQuat) -> Quat {
    Quat::from_array(enu_to_bevy_quat(q))
}

/// A scene mesh (ENU or FLU coordinates) as a Bevy mesh. The basis change is a proper
/// rotation, so triangle winding is preserved.
pub fn mesh(m: &MeshData) -> Mesh {
    let flip = |p: &[f32; 3]| [p[0], p[2], -p[1]];
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, m.positions.iter().map(flip).collect::<Vec<_>>())
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, m.normals.iter().map(flip).collect::<Vec<_>>())
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, m.colors.clone())
        .with_inserted_indices(Indices::U32(m.indices.clone()))
}

/// ENU (or FLU) pose → Bevy transform.
pub fn transform(p: &autonomousim_core::math::Pose) -> Transform {
    Transform::from_translation(vec(p.pos)).with_rotation(quat(p.rot))
}
