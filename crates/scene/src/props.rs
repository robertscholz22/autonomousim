//! Obstacle meshes (trees, rocks, pillars, walls), merged per terrain chunk so that a map
//! with tens of thousands of obstacles needs only one draw call per chunk.

use crate::mesh::{self, MeshData, srgb};
use crate::terrain::Chunk;
use autonomousim_world::obstacles::tags;
use autonomousim_world::{HeightGrid, Obstacle, ObstacleShape, StaticWorld};
use glam::{DQuat, DVec3, Vec3};

/// Tessellation of the primitives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropDetail {
    /// Segments around cones and cylinders.
    pub segments: usize,
    /// Segments around capsules (tree trunks).
    pub capsule_segments: usize,
    /// Icosphere subdivisions (smooth-shaded).
    pub sphere_subdivisions: u32,
    /// Obstacles smaller than this (largest extent, m) are left out.
    pub min_size: f64,
}

impl Default for PropDetail {
    fn default() -> Self {
        Self { segments: 7, capsule_segments: 5, sphere_subdivisions: 1, min_size: 0.0 }
    }
}

impl PropDetail {
    /// For obstacles seen from a few hundred metres: coarse, without small rocks.
    pub fn far() -> Self {
        Self { segments: 4, capsule_segments: 3, sphere_subdivisions: 0, min_size: 1.5 }
    }
}

/// Largest extent of a shape (m).
fn extent(shape: &ObstacleShape) -> f64 {
    match shape {
        ObstacleShape::Sphere { radius } => 2.0 * radius,
        ObstacleShape::Capsule { half_height, radius } => 2.0 * (half_height + radius),
        ObstacleShape::Cylinder { half_height, radius } | ObstacleShape::Cone { half_height, radius } => {
            2.0 * half_height.max(*radius)
        }
        ObstacleShape::Cuboid { half_extents } => 2.0 * half_extents.max_element(),
        ObstacleShape::ConvexHull { points } => {
            let (lo, hi) =
                points.iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
            (hi - lo).max_element().max(0.0)
        }
    }
}

/// Base colour (sRGB) of an obstacle.
fn base_color(world: &StaticWorld, o: &Obstacle) -> [u8; 3] {
    match o.tag {
        tags::TRUNK => [92, 66, 46],
        tags::CANOPY => [44, 82, 50],
        tags::CANOPY_BROADLEAF => [82, 124, 56],
        _ => {
            let table = world.materials();
            if (o.material.0 as usize) < table.len() { table.get(o.material).color } else { [200, 0, 200] }
        }
    }
}

/// Brightness factor in [0.82, 1.18] that varies from obstacle to obstacle.
fn variation(index: usize) -> f32 {
    let mut h = (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 29;
    0.82 + 0.36 * ((h & 0xFFFF) as f32 / 65535.0)
}

/// Mesh of one obstacle in its local frame.
pub fn obstacle_mesh(shape: &ObstacleShape, color: [f32; 4], detail: PropDetail) -> MeshData {
    let s = detail.segments.max(3);
    match shape {
        ObstacleShape::Sphere { radius } => mesh::icosphere(*radius as f32, detail.sphere_subdivisions, true, color),
        // Capsules are drawn as open tubes reaching halfway into their caps (trunks end in the
        // ground and the crown).
        ObstacleShape::Capsule { half_height, radius } => {
            mesh::tube(*radius as f32, (half_height + 0.5 * radius) as f32, detail.capsule_segments.max(3), color)
        }
        ObstacleShape::Cylinder { half_height, radius } => {
            mesh::cylinder(*radius as f32, *half_height as f32, s, color)
        }
        ObstacleShape::Cone { half_height, radius } => mesh::cone(*radius as f32, *half_height as f32, s, color),
        ObstacleShape::Cuboid { half_extents } => mesh::cuboid(half_extents.as_vec3(), color),
        ObstacleShape::ConvexHull { points } => mesh::convex_hull(points, color),
    }
}

/// Index of the chunk (in [`chunks`](crate::terrain::chunks) order) that contains `p`.
pub fn chunk_index(grid: &HeightGrid, size: usize, p: DVec3) -> usize {
    let (cw, ch) = grid.cells();
    let per_row = cw.div_ceil(size);
    let q = (p.truncate() - grid.origin()) / grid.cell_size();
    let cx = (q.x.max(0.0) as usize).min(cw - 1) / size;
    let cy = (q.y.max(0.0) as usize).min(ch - 1) / size;
    cy * per_row + cx
}

/// Obstacle meshes grouped by chunk: one merged mesh per entry of `chunks` (empty where a
/// chunk has no obstacles).
pub fn props_by_chunk(world: &StaticWorld, chunks: &[Chunk], size: usize, detail: PropDetail) -> Vec<MeshData> {
    let grid = world.terrain();
    let mut out = vec![MeshData::new(); chunks.len()];
    for (i, o) in world.obstacles().obstacles().iter().enumerate() {
        if detail.min_size > 0.0 && extent(&o.shape) < detail.min_size {
            continue;
        }
        let k = chunk_index(grid, size, o.pose.pos);
        let color = srgb(base_color(world, o));
        let mut m = obstacle_mesh(&o.shape, color, detail);
        m.tint(variation(i));
        out[k].append_transformed(&m, o.pose.rot, o.pose.pos);
    }
    out
}

/// Visual of a multirotor in its body frame (FLU): hub, arms, motors and rotor discs.
pub struct MultirotorVisual {
    pub body: MeshData,
    /// Hub position and thrust axis of every rotor (body frame), for separate disc meshes.
    pub rotors: Vec<(DVec3, DVec3)>,
    pub rotor_radius: f32,
    /// Largest distance of a rotor tip from the centre (m), for camera distances.
    pub span: f32,
}

/// Build the visual of a multirotor. Arms towards the front (+x) are red, the others grey.
pub fn multirotor(def: &autonomousim_vehicles::multirotor::MultirotorDef) -> MultirotorVisual {
    let r = def.rotor.radius as f32;
    let arm = def.rotors.iter().map(|m| m.position.truncate().length()).fold(0.0, f64::max).max(1e-3) as f32;
    let mut body = MeshData::new();
    let dark = srgb([40, 42, 46]);
    let front = srgb([200, 40, 36]);
    let back = srgb([90, 94, 100]);
    // Central hub, flatter than wide.
    body.append(&mesh::cuboid(Vec3::new(0.32 * arm, 0.22 * arm, 0.1 * arm), dark));
    // A light marker on top at the front.
    let marker = mesh::cuboid(Vec3::new(0.06 * arm, 0.06 * arm, 0.03 * arm), srgb([240, 240, 90]));
    body.append_transformed(&marker, DQuat::IDENTITY, DVec3::new(0.24 * arm as f64, 0.0, 0.11 * arm as f64));
    let width = 0.05 * arm;
    for m in &def.rotors {
        let hub = m.position;
        let flat = hub.truncate();
        let len = flat.length() as f32;
        let color = if hub.x > 1e-9 { front } else { back };
        // Arm from the centre to the hub, along its direction in the xy plane.
        let arm_mesh = mesh::cuboid(Vec3::new(0.5 * len, width, 0.5 * width), color);
        let rot = DQuat::from_rotation_z(flat.y.atan2(flat.x));
        body.append_transformed(&arm_mesh, rot, (flat * 0.5).extend(hub.z - 0.02 * arm as f64));
        // Motor can.
        let motor = mesh::cylinder(0.09 * arm.max(r), 0.06 * arm.max(r), 10, dark);
        let axis_rot = DQuat::from_rotation_arc(DVec3::Z, m.axis.normalize());
        body.append_transformed(&motor, axis_rot, hub - m.axis.normalize() * 0.03 * arm as f64);
    }
    let rotors = def.rotors.iter().map(|m| (m.position, m.axis.normalize())).collect();
    MultirotorVisual { body, rotors, rotor_radius: r, span: arm + r }
}

/// Thin disc for a spinning rotor (its z axis is the rotor axis).
pub fn rotor_disc(radius: f32, color: [f32; 4]) -> MeshData {
    mesh::cylinder(radius, 0.004 * radius.max(0.05), 24, color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::chunks;
    use autonomousim_vehicles::presets;
    use autonomousim_world::testworlds;

    #[test]
    fn props_land_in_the_chunk_below_them() {
        let w = testworlds::forest_patch(120.0, 150.0, 3);
        let g = w.terrain();
        let cs = chunks(g, 32);
        let props = props_by_chunk(&w, &cs, 32, PropDetail::default());
        assert_eq!(props.len(), cs.len());
        let total: usize = props.iter().map(MeshData::triangle_count).sum();
        assert!(total > 100 * w.obstacles().len() / 10, "{total}");
        // The far level of detail has fewer triangles.
        let far: usize = props_by_chunk(&w, &cs, 32, PropDetail::far()).iter().map(MeshData::triangle_count).sum();
        assert!(far > 0 && 3 * far < 2 * total, "{far} of {total}");
        for (chunk, m) in cs.iter().zip(&props) {
            let Some((lo, hi)) = m.bounds() else { continue };
            let (clo, chi) = chunk.bounds(g);
            // Obstacles stick out of their chunk by at most a crown radius.
            assert!(lo.x as f64 > clo.x - 8.0 && (hi.x as f64) < chi.x + 8.0);
            assert!(lo.y as f64 > clo.y - 8.0 && (hi.y as f64) < chi.y + 8.0);
            assert!((hi.z as f64) > clo.z);
        }
    }

    #[test]
    fn multirotor_visual_matches_its_definition() {
        for name in ["cf2x", "iris_like"] {
            let def = presets::multirotor(name).unwrap();
            let v = multirotor(&def);
            assert_eq!(v.rotors.len(), def.rotors.len());
            let arm = def.rotors[0].position.truncate().length() as f32;
            let reach = v.body.positions.iter().map(|p| p[0].hypot(p[1])).fold(0.0, f32::max);
            assert!(reach > 0.95 * arm && reach < 1.2 * arm, "{name}: {reach} vs {arm}");
            assert!(v.span > arm);
            // The front arms are red.
            let red = v.body.colors.iter().filter(|c| c[0] > 0.5 && c[1] < 0.1).count();
            assert!(red > 0);
        }
    }
}
