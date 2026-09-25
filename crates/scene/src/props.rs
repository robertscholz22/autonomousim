//! Obstacle meshes (trees, rocks, pillars, walls), merged per terrain chunk so that a map
//! with tens of thousands of obstacles needs only one draw call per chunk.

use crate::mesh::{self, MeshData, srgb};
use crate::terrain::Chunk;
use autonomousim_core::material::MaterialId;
use autonomousim_world::obstacles::tags;
use autonomousim_world::{HeightGrid, Obstacle, ObstacleClass, ObstacleShape, StaticWorld};
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
        tags::HEDGE if o.class == ObstacleClass::Foliage => [58, 94, 46],
        tags::FENCE => [128, 104, 76],
        tags::SILO => [188, 192, 196],
        tags::BUILDING => match o.material {
            MaterialId::CONCRETE => [218, 208, 186],
            MaterialId::WOOD => [142, 98, 66],
            _ => [166, 170, 176],
        },
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

/// Roof colour of a building by its material (house, barn, shed).
fn roof_color(material: MaterialId) -> [u8; 3] {
    match material {
        MaterialId::CONCRETE => [156, 72, 52],
        MaterialId::WOOD => [104, 52, 44],
        _ => [112, 118, 124],
    }
}

/// Mesh of one obstacle in its local frame as the viewer shows it: its shape, except that
/// buildings get a gable roof (above their collision box), silos a conical cap, and fences
/// are drawn as posts and wires.
pub fn obstacle_visual(o: &Obstacle, color: [f32; 4], detail: PropDetail) -> MeshData {
    match (o.tag, &o.shape) {
        (tags::BUILDING, ObstacleShape::Cuboid { half_extents: h }) => {
            let mut m = mesh::cuboid(h.as_vec3(), color);
            // The ridge runs along the longer side; the roof overhangs by 0.3 m.
            let (long, short, turn) =
                if h.x >= h.y { (h.x, h.y, 0.0) } else { (h.y, h.x, std::f64::consts::FRAC_PI_2) };
            let roof = Vec3::new((long + 0.3) as f32, (short + 0.3) as f32, (0.5 * short) as f32);
            let roof = mesh::gable_roof(roof, srgb(roof_color(o.material)));
            m.append_transformed(&roof, DQuat::from_rotation_z(turn), DVec3::new(0.0, 0.0, h.z));
            m
        }
        (tags::SILO, ObstacleShape::Cylinder { half_height, radius }) => {
            let s = detail.segments.max(3) * 2;
            let mut m = mesh::cylinder(*radius as f32, *half_height as f32, s, color);
            let cap = mesh::cone(*radius as f32 * 1.05, 0.2 * *radius as f32, s, srgb([150, 154, 158]));
            m.append_transformed(&cap, DQuat::IDENTITY, DVec3::new(0.0, 0.0, half_height + 0.2 * radius));
            m
        }
        (tags::FENCE, ObstacleShape::Cuboid { half_extents: h }) => {
            let (long, turn) = if h.x >= h.y { (h.x, 0.0) } else { (h.y, std::f64::consts::FRAC_PI_2) };
            let rot = DQuat::from_rotation_z(turn);
            let mut m = MeshData::new();
            let post = mesh::cuboid(Vec3::new(0.06, 0.06, h.z as f32), color);
            for x in [-long + 0.06, long - 0.06] {
                m.append_transformed(&post, rot, rot * DVec3::new(x, 0.0, 0.0));
            }
            let wire = mesh::cuboid(Vec3::new(long as f32, 0.012, 0.012), srgb([150, 152, 150]));
            for z in [-0.35, 0.1, 0.55] {
                m.append_transformed(&wire, rot, DVec3::new(0.0, 0.0, z * h.z / 0.6));
            }
            m
        }
        _ => obstacle_mesh(&o.shape, color, detail),
    }
}

/// Mesh of one obstacle's shape in its local frame.
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
        // Hedges hide their woody cores.
        let hidden = o.tag == tags::HEDGE && o.class == ObstacleClass::Solid;
        if hidden || (detail.min_size > 0.0 && extent(&o.shape) < detail.min_size) {
            continue;
        }
        let k = chunk_index(grid, size, o.pose.pos);
        let color = srgb(base_color(world, o));
        let mut m = obstacle_visual(o, color, detail);
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

/// Visual of a wheeled vehicle: the body in the chassis frame (FLU) and one mesh per wheel in
/// its spinning link's frame (spin axis y), to be posed from the simulated wheels.
#[derive(Clone, Debug)]
pub struct WheeledVisual {
    pub body: MeshData,
    pub wheels: Vec<MeshData>,
    /// Largest distance of a wheel's outer edge from the chassis origin (m), for cameras.
    pub span: f32,
    /// Driver's eye point in the chassis frame (m), for the first-person camera.
    pub eye: DVec3,
    /// Per wheel with suspension: chassis-side ends of the strut and of the lower arm, in the
    /// chassis frame (m). The other ends follow the wheel centre.
    pub links: Vec<Option<[DVec3; 2]>>,
    /// Link (unit cylinder along z, from −0.5 to 0.5) scaled to the links' thickness.
    pub link: MeshData,
}

/// Build the visual of a wheeled vehicle: a box over the wheelbase and track with a red nose
/// (and a cabin on cars), dark tyres with a light marker on the rim so that spin is visible,
/// and a strut and lower arm per suspended wheel.
pub fn wheeled(def: &autonomousim_vehicles::ground::WheeledDef) -> WheeledVisual {
    let n = def.num_wheels();
    let positions: Vec<DVec3> = (0..n).map(|w| def.wheel_position(w)).collect();
    let tire = |w: usize| def.tire(w / 2);
    let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    for (w, p) in positions.iter().enumerate() {
        let r = tire(w).radius();
        lo = lo.min(*p - DVec3::new(r, 0.0, 0.0));
        hi = hi.max(*p + DVec3::new(r, 0.0, r));
    }
    let width = (0..n).map(|w| tire(w).width()).fold(0.0, f64::max);
    let radius = (0..n).map(|w| tire(w).radius()).fold(0.0, f64::max);
    // Between the wheels, from the axle line up to a little above the tyre tops.
    let half = DVec3::new(0.5 * (hi.x - lo.x), (0.5 * (hi.y - lo.y) - 0.6 * width).max(0.3 * radius), 0.4 * radius);
    let centre = DVec3::new(0.5 * (hi.x + lo.x), 0.5 * (hi.y + lo.y), lo.z + 0.2 * radius + half.z);
    let mut body = MeshData::new();
    let h = |v: DVec3| v.as_vec3();
    body.append_transformed(&mesh::cuboid(h(half), srgb([70, 110, 150])), DQuat::IDENTITY, centre);
    let nose = mesh::cuboid(h(DVec3::new(0.08 * half.x, 0.8 * half.y, 0.3 * half.z)), srgb([200, 40, 36]));
    body.append_transformed(&nose, DQuat::IDENTITY, centre + DVec3::new(half.x, 0.0, 0.5 * half.z));
    let top = centre.z + half.z;
    // Cars (wheels larger than a robot's): a cabin over the middle, a little behind centre.
    let car = radius > 0.2;
    let eye = if car {
        let cabin = DVec3::new(0.3 * half.x, 0.85 * half.y, 0.55 * radius.max(0.35));
        let at = DVec3::new(centre.x - 0.1 * half.x, centre.y, top + cabin.z);
        body.append_transformed(&mesh::cuboid(h(cabin), srgb([150, 185, 210])), DQuat::IDENTITY, at);
        DVec3::new(at.x + 0.2 * cabin.x, centre.y + 0.4 * cabin.y, top + 1.2 * cabin.z)
    } else {
        DVec3::new(centre.x + 0.8 * half.x, centre.y, top + 0.3 * half.z)
    };
    let axis = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
    let wheels = (0..n)
        .map(|w| {
            let (r, b) = (tire(w).radius() as f32, tire(w).width() as f32);
            let mut m = MeshData::new();
            m.append_transformed(&mesh::cylinder(r, 0.5 * b, 20, srgb([30, 30, 32])), axis, DVec3::ZERO);
            let marker = mesh::cuboid(Vec3::new(0.12 * r, 0.52 * b, 0.12 * r), srgb([220, 220, 220]));
            m.append_transformed(&marker, DQuat::IDENTITY, DVec3::new(0.0, 0.0, 0.7 * r as f64));
            m
        })
        .collect();
    // Strut from above the wheel, inboard, and lower arm to the body's side, below the axle line.
    let links = (0..n)
        .map(|w| {
            def.axles[w / 2].suspension.as_ref()?;
            let (p, r) = (positions[w], tire(w).radius());
            let inboard = p.y.signum() * (0.5 * width + 0.25 * r);
            let strut = DVec3::new(p.x, p.y - inboard, p.z + 0.9 * r);
            let arm = DVec3::new(p.x, (p.y - 2.0 * inboard).abs().min(half.y).copysign(p.y), p.z - 0.2 * r);
            Some([strut, arm])
        })
        .collect();
    let link = mesh::cylinder((0.06 * radius) as f32, 0.5, 8, srgb([90, 90, 95]));
    let span = positions.iter().enumerate().map(|(w, p)| p.length() + tire(w).radius()).fold(0.0, f64::max) as f32;
    WheeledVisual { body, wheels, span, eye, links, link }
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
    fn farm_obstacles_get_roofs_caps_and_wires() {
        use autonomousim_core::math::Pose;
        let color = srgb([200, 200, 200]);
        let cuboid = |h: DVec3| ObstacleShape::Cuboid { half_extents: h };
        let barn = Obstacle::solid(cuboid(DVec3::new(6.0, 11.0, 4.5)), Pose::IDENTITY, MaterialId::WOOD)
            .with_tag(tags::BUILDING);
        let m = obstacle_visual(&barn, color, PropDetail::default());
        let (lo, hi) = m.bounds().unwrap();
        // The ridge runs along the long (y) side, 0.5 × the short half-width above the walls;
        // the roof overhangs by 0.3 m.
        assert!((hi.z - 7.5).abs() < 1e-4 && (lo.z + 4.5).abs() < 1e-4, "{lo} {hi}");
        assert!((hi.x - 6.3).abs() < 1e-4 && (hi.y - 11.3).abs() < 1e-4, "{hi}");
        // Every face points away from the centre of the roof or walls.
        for t in m.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| Vec3::from_array(m.positions[i as usize]));
            let centre = if a.z.min(b.z).min(c.z) >= 4.5 - 1e-4 { Vec3::new(0.0, 0.0, 4.4) } else { Vec3::ZERO };
            assert!((b - a).cross(c - a).dot((a + b + c) / 3.0 - centre) > 0.0);
        }
        let fence =
            Obstacle::solid(cuboid(DVec3::new(0.05, 2.0, 0.6)), Pose::IDENTITY, MaterialId::WOOD).with_tag(tags::FENCE);
        let (lo, hi) = obstacle_visual(&fence, color, PropDetail::default()).bounds().unwrap();
        assert!(hi.y > 1.9 && lo.y < -1.9 && hi.x < 0.07 && hi.z <= 0.6 + 1e-4, "{lo} {hi}");
        let silo = Obstacle::solid(
            ObstacleShape::Cylinder { half_height: 6.0, radius: 2.0 },
            Pose::IDENTITY,
            MaterialId::METAL,
        )
        .with_tag(tags::SILO);
        let (_, hi) = obstacle_visual(&silo, color, PropDetail::default()).bounds().unwrap();
        assert!((hi.z - 6.8).abs() < 1e-4, "{hi}");
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

    #[test]
    fn wheeled_visual_covers_its_wheels() {
        for name in ["sedan_like", "rover_diff", "offroad_4x4"] {
            let def = presets::wheeled(name).unwrap();
            let v = wheeled(&def);
            assert_eq!(v.wheels.len(), def.num_wheels());
            let (lo, hi) = v.body.bounds().unwrap();
            let front = def.wheel_position(0).x as f32;
            assert!(hi.x > front && lo.x < def.wheel_position(def.num_wheels() - 1).x as f32, "{name}");
            // Wheels are discs about y.
            let (wlo, whi) = v.wheels[0].bounds().unwrap();
            let r = def.tire(0).radius() as f32;
            assert!((whi.z - r).abs() < 1e-3 && (wlo.x + r).abs() < 0.05 * r, "{name}: {wlo} {whi}");
            assert!(whi.y < 0.6 * def.tire(0).width() as f32);
            assert!(v.span > front);
            // Links on suspended wheels only; the eye is over the body.
            for (w, l) in v.links.iter().enumerate() {
                assert_eq!(l.is_some(), def.axles[w / 2].suspension.is_some(), "{name}");
            }
            assert!((v.eye.x as f32) < hi.x && v.eye.z > def.wheel_position(0).z, "{name}: {}", v.eye);
        }
        assert!(wheeled(&presets::wheeled("sedan_like").unwrap()).links.iter().all(Option::is_some));
    }
}
