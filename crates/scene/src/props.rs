//! Obstacle meshes (trees, rocks, pillars, walls), merged per terrain chunk so that a map
//! with tens of thousands of obstacles needs only one draw call per chunk.

use crate::mesh::{self, MeshData, srgb};
use crate::terrain::Chunk;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_vehicles::ground::tire::TireModel;
use autonomousim_world::obstacles::tags;
use autonomousim_world::{HeightGrid, Obstacle, ObstacleClass, ObstacleShape, StaticWorld};
use glam::{DMat3, DQuat, DVec2, DVec3, Vec3};

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
fn base_color(table: &MaterialTable, o: &Obstacle) -> [u8; 3] {
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
            if (o.material.0 as usize) < table.len() {
                table.get(o.material).color
            } else {
                [200, 0, 200]
            }
        }
    }
}

/// Brightness factor in [0.82, 1.18] that varies from obstacle to obstacle.
fn variation(index: u64) -> f32 {
    let mut h = index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
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
    let grid = world.grid();
    let obstacles = world.obstacle_set().obstacles().iter().enumerate();
    props_grouped(
        world.materials(),
        obstacles.map(|(i, o)| (i as u64, o, chunk_index(grid, size, o.pose.pos))),
        chunks.len(),
        detail,
        DVec3::ZERO,
    )
}

/// Obstacle meshes merged into `n` groups, relative to `anchor`: `items` are
/// `(key, obstacle, group)`; the key varies the brightness from obstacle to obstacle.
pub fn props_grouped<'a>(
    materials: &MaterialTable,
    items: impl Iterator<Item = (u64, &'a Obstacle, usize)>,
    n: usize,
    detail: PropDetail,
    anchor: DVec3,
) -> Vec<MeshData> {
    let mut out = vec![MeshData::new(); n];
    for (key, o, k) in items {
        // Hedges hide their woody cores.
        let hidden = o.tag == tags::HEDGE && o.class == ObstacleClass::Solid;
        if hidden || (detail.min_size > 0.0 && extent(&o.shape) < detail.min_size) {
            continue;
        }
        let color = srgb(base_color(materials, o));
        let mut m = obstacle_visual(o, color, detail);
        m.tint(variation(key));
        out[k].append_transformed(&m, o.pose.rot, o.pose.pos - anchor);
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

/// A hinged control surface of a [`FixedWingVisual`]: the mesh about its hinge point, turned
/// about `axis` by the deflection in pilot sense (see [`FixedWingVisual`]).
#[derive(Clone, Debug)]
pub struct SurfaceVisual {
    pub mesh: MeshData,
    /// Hinge point in the body frame (m).
    pub hinge: DVec3,
    /// Hinge axis: a positive rotation about it is the deflection that rolls right, pitches up,
    /// yaws right or (flaps) lowers the trailing edge.
    pub axis: DVec3,
    /// Which control drives it: 0 aileron, 1 elevator, 2 rudder, 3 flaps.
    pub control: usize,
}

/// Visual of a fixed-wing aircraft, derived from its definition (the preset files carry no
/// shape): fuselage from the frame colliders, the wing from the reference geometry, a tail at
/// the rear, the propeller and the wheels. Body frame FLU.
#[derive(Clone, Debug)]
pub struct FixedWingVisual {
    pub body: MeshData,
    pub surfaces: Vec<SurfaceVisual>,
    /// Propeller hub and axis (body frame) and radius (m), for a spinning disc.
    pub propeller: (DVec3, DVec3),
    pub propeller_radius: f32,
    /// Largest distance of any part from the centre of mass (m), for cameras.
    pub span: f32,
    /// Pilot's eye point (m), for the first-person camera.
    pub eye: DVec3,
}

pub fn fixed_wing(def: &autonomousim_vehicles::fixedwing::FixedWingDef) -> FixedWingVisual {
    use autonomousim_vehicles::multirotor::ColliderPart;
    let g = &def.geometry;
    let (b, c) = (g.span, g.chord);
    let frame: Vec<_> = def.colliders.iter().filter(|k| k.part == ColliderPart::Frame).collect();
    // A tractor propeller sits at the nose, a pusher's behind the wing (not necessarily at the
    // tail).
    let prop_x = def.propulsion.position.x;
    let nose = frame.iter().map(|k| k.center.x + k.radius).fold((0.5 * c).max(prop_x), f64::max);
    let tail = frame.iter().map(|k| k.center.x - k.radius).fold(-1.5 * c, f64::min);
    let length = nose - tail;
    let white = srgb([232, 234, 238]);
    let trim = srgb([200, 50, 40]);
    let grey = srgb([150, 154, 160]);
    let dark = srgb([40, 42, 46]);
    let glass = srgb([60, 90, 120]);
    let mut body = MeshData::new();
    // Fuselage: a cabin over the front half, tapering to a slim boom at the tail.
    let (w, h) = ((0.09 * b).min(0.2 * length), (0.1 * b).min(0.22 * length));
    let mid = nose - 0.45 * length;
    let section = |x: f64, w: f64, h: f64, z: f64| {
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
            .map(|(sy, sz)| DVec3::new(x, sy * 0.5 * w, z + sz * 0.5 * h))
    };
    let front: Vec<_> = section(nose, 0.6 * w, 0.6 * h, 0.0).into_iter().chain(section(mid, w, h, 0.0)).collect();
    body.append(&mesh::convex_hull(&front, white));
    let boom: Vec<_> = section(mid, w, h, 0.0).into_iter().chain(section(tail, 0.3 * w, 0.35 * h, 0.15 * h)).collect();
    body.append(&mesh::convex_hull(&boom, white));
    // Canopy over the front of the cabin.
    let eye = DVec3::new(nose - 0.3 * length, 0.0, 0.45 * h);
    let canopy = mesh::ellipsoid(Vec3::new((0.12 * length) as f32, (0.4 * w) as f32, (0.25 * h) as f32), 1, glass);
    body.append_transformed(&canopy, DQuat::IDENTITY, eye - DVec3::Z * 0.1 * h);
    // Wing about the reference point: its quarter chord there, 12 % thick.
    let r = g.aero_reference;
    let (le, te) = (r.x + 0.25 * c, r.x - 0.75 * c);
    let t = 0.06 * c;
    let fixed_te = te + 0.25 * c;
    body.append_transformed(
        &mesh::cuboid(Vec3::new((0.5 * (le - fixed_te)) as f32, (0.5 * b) as f32, t as f32), white),
        DQuat::IDENTITY,
        DVec3::new(0.5 * (le + fixed_te), 0.0, r.z),
    );
    for side in [-1.0, 1.0] {
        let tip = mesh::cuboid(Vec3::new((0.5 * c) as f32, (0.01 * b) as f32, (1.1 * t) as f32), trim);
        body.append_transformed(&tip, DQuat::IDENTITY, DVec3::new(r.x - 0.25 * c, side * 0.5 * b, r.z));
    }
    let mut surfaces = Vec::new();
    // A plate of half extents `half` just behind its hinge line.
    let flap = |chord: f64, half: Vec3, color| {
        let mut m = MeshData::new();
        m.append_transformed(&mesh::cuboid(half, color), DQuat::IDENTITY, DVec3::new(-0.5 * chord, 0.0, 0.0));
        m
    };
    // Trailing-edge surfaces: ailerons on the outer 45 %, flaps inboard of them.
    let sc = 0.25 * c;
    for side in [-1.0f64, 1.0] {
        for (from, to, control) in [(0.55, 0.98, 0), (0.12, 0.55, 3)] {
            let half = 0.25 * b * (to - from);
            let y = side * 0.5 * b * 0.5 * (from + to);
            let m = flap(sc, Vec3::new((0.5 * sc) as f32, half as f32, (0.8 * t) as f32), grey);
            // Right aileron (y < 0) up and left down rolls right: rotations about +y raise the
            // trailing edge, so the right one turns about +y, the left one about −y.
            let axis = if control == 0 { DVec3::Y * -side } else { -DVec3::Y };
            surfaces.push(SurfaceVisual { mesh: m, hinge: DVec3::new(fixed_te, y, r.z), axis, control });
        }
    }
    // Tail: stabiliser and fin with elevator and rudder on their rear 35 %.
    let (tail_span, tail_chord) = (0.34 * b, 0.65 * c);
    let fin_h = 0.13 * b;
    let tail_z = 0.15 * h;
    let hinge_x = tail + 0.35 * tail_chord;
    let stab_x = hinge_x + 0.325 * tail_chord;
    let tail_t = (0.04 * tail_chord) as f32;
    body.append_transformed(
        &mesh::cuboid(Vec3::new((0.325 * tail_chord) as f32, (0.5 * tail_span) as f32, tail_t), white),
        DQuat::IDENTITY,
        DVec3::new(stab_x, 0.0, tail_z),
    );
    body.append_transformed(
        &mesh::cuboid(Vec3::new((0.325 * tail_chord) as f32, tail_t, (0.5 * fin_h) as f32), trim),
        DQuat::IDENTITY,
        DVec3::new(stab_x, 0.0, tail_z + 0.5 * fin_h),
    );
    let ec = 0.35 * tail_chord;
    surfaces.push(SurfaceVisual {
        mesh: flap(ec, Vec3::new((0.5 * ec) as f32, (0.5 * tail_span) as f32, tail_t), grey),
        hinge: DVec3::new(hinge_x, 0.0, tail_z),
        // Trailing edge up pitches up: about +y.
        axis: DVec3::Y,
        control: 1,
    });
    surfaces.push(SurfaceVisual {
        mesh: flap(ec, Vec3::new((0.5 * ec) as f32, tail_t, (0.5 * fin_h) as f32), trim),
        hinge: DVec3::new(hinge_x, 0.0, tail_z + 0.5 * fin_h),
        // Trailing edge right (−y) yaws right: about +z.
        axis: DVec3::Z,
        control: 2,
    });
    // Propeller hub with a spinner.
    let p = &def.propulsion;
    let axis = p.axis.normalize_or(DVec3::X);
    let radius = (0.5 * p.propeller.diameter) as f32;
    let spinner = mesh::cone((0.12 * radius).max(0.3 * w as f32), 0.15 * radius.max(0.5), 12, dark);
    body.append_transformed(
        &spinner,
        DQuat::from_rotation_arc(DVec3::Z, axis),
        p.position + axis * 0.05 * radius as f64,
    );
    // Wheels on struts to the fuselage.
    for gear in &def.gear {
        // The gear position is the contact point with the strut extended.
        let wr = gear.wheel_radius as f32;
        let centre = gear.position + DVec3::Z * gear.wheel_radius;
        let wheel = mesh::cylinder(wr, 0.35 * wr, 14, dark);
        body.append_transformed(&wheel, DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2), centre);
        let top = DVec3::new(centre.x, centre.y * 0.3, -0.4 * h);
        let d = top - centre;
        if d.length() > 1e-3 {
            let strut = mesh::cylinder(0.25 * wr, 0.5 * d.length() as f32, 6, grey);
            body.append_transformed(&strut, DQuat::from_rotation_arc(DVec3::Z, d.normalize()), centre + 0.5 * d);
        }
    }
    let span = (0.5 * b).max(nose.abs()).max(tail.abs()).max(p.position.length() + radius as f64) as f32;
    FixedWingVisual { body, surfaces, propeller: (p.position, axis), propeller_radius: radius, span, eye }
}

/// A flapped surface of a [`TiltrotorVisual`]: the flap turns about `axis` through `hinge` by
/// its deflection (positive lowers the trailing edge, in the surface's own frame).
#[derive(Clone, Debug)]
pub struct TiltSurfaceVisual {
    pub mesh: MeshData,
    pub hinge: DVec3,
    pub axis: DVec3,
    /// Index of the surface in the definition (its mixing gains give the deflection).
    pub surface: usize,
}

/// A rotor pod of a [`TiltrotorVisual`], in its own frame: origin at the pivot, z along the
/// thrust axis at tilt 0; posed by `R_y(tilt)` at `pivot`. The disc sits at `offset` along z.
#[derive(Clone, Debug)]
pub struct NacelleVisual {
    pub mesh: MeshData,
    pub pivot: DVec3,
    pub offset: f64,
    pub radius: f32,
}

/// Visual of a tiltrotor, derived from its definition: a fuselage along the frame colliders
/// on the centre line, the lifting surfaces as plates with their flaps, booms joining the
/// rotor pivots on each side, landing legs, and one tilting pod per rotor. Body frame FLU.
#[derive(Clone, Debug)]
pub struct TiltrotorVisual {
    pub body: MeshData,
    pub surfaces: Vec<TiltSurfaceVisual>,
    pub nacelles: Vec<NacelleVisual>,
    /// Largest distance of any part from the centre of mass (m), for cameras.
    pub span: f32,
    /// Pilot's eye point (m), for the first-person camera.
    pub eye: DVec3,
}

pub fn tiltrotor(def: &autonomousim_vehicles::tiltrotor::TiltrotorDef) -> TiltrotorVisual {
    use autonomousim_vehicles::multirotor::ColliderPart;
    let white = srgb([232, 234, 238]);
    let trim = srgb([40, 110, 190]);
    let grey = srgb([150, 154, 160]);
    let dark = srgb([40, 42, 46]);
    let glass = srgb([60, 90, 120]);
    let mut body = MeshData::new();
    // Fuselage between the frame colliders on the centre line.
    let centre: Vec<_> =
        def.colliders.iter().filter(|k| k.part == ColliderPart::Frame && k.center.y.abs() < 0.05).collect();
    let nose = centre.iter().map(|k| k.center.x + k.radius).fold(0.3, f64::max);
    let tail = centre.iter().map(|k| k.center.x - k.radius).fold(-0.3, f64::min);
    let girth = centre.iter().map(|k| k.radius).fold(0.05, f64::max);
    let (w, h) = (1.6 * girth, 1.8 * girth);
    let mid = nose - 0.4 * (nose - tail);
    let section = |x: f64, w: f64, h: f64, z: f64| {
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
            .map(|(sy, sz)| DVec3::new(x, sy * 0.5 * w, z + sz * 0.5 * h))
    };
    let front: Vec<_> = section(nose, 0.5 * w, 0.5 * h, 0.0).into_iter().chain(section(mid, w, h, 0.0)).collect();
    body.append(&mesh::convex_hull(&front, white));
    let boom: Vec<_> = section(mid, w, h, 0.0).into_iter().chain(section(tail, 0.3 * w, 0.3 * h, 0.2 * h)).collect();
    body.append(&mesh::convex_hull(&boom, white));
    let eye = DVec3::new(nose - 0.25 * (nose - tail), 0.0, 0.35 * h);
    let dome = mesh::ellipsoid(Vec3::new((0.1 * (nose - tail)) as f32, (0.35 * w) as f32, (0.3 * h) as f32), 1, glass);
    body.append_transformed(&dome, DQuat::IDENTITY, eye - DVec3::Z * 0.1 * h);
    // Surfaces: a plate from the leading edge to the hinge (its quarter chord at `position`),
    // 10 % thick, turned by the surface's roll; the flap behind the hinge.
    let mut surfaces = Vec::new();
    let mut extent = nose.abs().max(tail.abs());
    for (i, sf) in def.surfaces.iter().enumerate() {
        let (c, b) = (sf.chord, sf.span);
        let t = (0.05 * c) as f32;
        let roll = DQuat::from_rotation_x(sf.roll);
        let cf = sf.flap.as_ref().map_or(0.0, |f| f.chord_fraction);
        let (le, hinge) = (0.25 * c, 0.25 * c - (1.0 - cf) * c);
        let color = if sf.roll.cos().abs() < 0.5 { trim } else { white };
        let plate = mesh::cuboid(Vec3::new((0.5 * (le - hinge)) as f32, (0.5 * b) as f32, t), color);
        body.append_transformed(&plate, roll, sf.position + roll * DVec3::X * 0.5 * (le + hinge));
        extent = extent.max(sf.position.length() + 0.5 * b);
        if cf > 0.0 {
            let fc = cf * c;
            let mut m = MeshData::new();
            let flap = mesh::cuboid(Vec3::new((0.5 * fc) as f32, (0.5 * b) as f32, 0.8 * t), grey);
            m.append_transformed(&flap, roll, roll * DVec3::new(-0.5 * fc, 0.0, 0.0));
            // Trailing edge down (towards −normal) is a rotation about −span.
            surfaces.push(TiltSurfaceVisual {
                mesh: m,
                hinge: sf.position + roll * DVec3::X * hinge,
                axis: roll * -DVec3::Y,
                surface: i,
            });
        }
    }
    // Booms along x joining the pivots on each side (and to the wing).
    let mut sides: Vec<(f64, f64, f64, f64)> = Vec::new();
    for r in &def.rotors {
        match sides.iter_mut().find(|s| (s.0 - r.pivot.y).abs() < 1e-3) {
            Some(s) => (s.1, s.2) = (s.1.min(r.pivot.x), s.2.max(r.pivot.x)),
            None => sides.push((r.pivot.y, r.pivot.x.min(0.0), r.pivot.x.max(0.0), r.pivot.z)),
        }
    }
    let boom_r = (0.3 * girth) as f32;
    for (y, x0, x1, z) in sides {
        let len = x1 - x0;
        if len > 1e-3 {
            let tube = mesh::cylinder(boom_r, (0.5 * len) as f32, 10, grey);
            body.append_transformed(
                &tube,
                DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2),
                DVec3::new(0.5 * (x0 + x1), y, z - 1.2 * boom_r as f64),
            );
        }
    }
    // Landing legs from the body down to the gear.
    for g in def.colliders.iter().filter(|k| k.part == ColliderPart::Gear) {
        let foot = g.center;
        let top = DVec3::new(foot.x, 0.6 * foot.y, -0.3 * h);
        let d = top - foot;
        if d.length() > 1e-3 {
            let leg = mesh::cylinder((0.6 * g.radius) as f32, (0.5 * d.length()) as f32, 6, dark);
            body.append_transformed(&leg, DQuat::from_rotation_arc(DVec3::Z, d.normalize()), foot + 0.5 * d);
        }
        body.append_transformed(&mesh::icosphere(g.radius as f32, 1, true, dark), DQuat::IDENTITY, foot);
    }
    // Pods: motor can from the pivot to the hub, spinner on top.
    let radius = (0.5 * def.propeller.diameter) as f32;
    let can = (0.12 * radius as f64).max(0.02);
    let nacelles = def
        .rotors
        .iter()
        .map(|r| {
            let mut m = MeshData::new();
            let len = r.offset.max(2.0 * can);
            m.append_transformed(
                &mesh::cylinder(can as f32, (0.5 * len) as f32, 12, trim),
                DQuat::IDENTITY,
                DVec3::Z * (r.offset - 0.5 * len),
            );
            m.append_transformed(
                &mesh::cone((0.8 * can) as f32, (0.6 * can) as f32, 12, dark),
                DQuat::IDENTITY,
                DVec3::Z * (r.offset + 0.6 * can),
            );
            extent = extent.max(r.pivot.length() + r.offset + radius as f64);
            NacelleVisual { mesh: m, pivot: r.pivot, offset: r.offset, radius }
        })
        .collect();
    TiltrotorVisual { body, surfaces, nacelles, span: extent as f32, eye }
}

/// A rotor of a [`HelicopterVisual`]: its hub and shaft frame, and one blade to be posed per
/// blade from the rotor's azimuth, tip-path-plane tilt and coning.
#[derive(Clone, Debug)]
pub struct RotorVisual {
    /// Hub (body frame, m) and the shaft frame's orientation in the body frame (z along the
    /// shaft).
    pub hub: DVec3,
    pub frame: DQuat,
    pub radius: f32,
    pub blades: u32,
    /// +1 when the rotor turns counter-clockwise seen from +z of the shaft frame, −1 otherwise.
    pub spin: f64,
    /// A blade along +x of the shaft frame from the root cut-out to the tip, its flapping hinge
    /// at the hub.
    pub blade: MeshData,
}

/// Visual of a helicopter, derived from its definition (the preset files carry no shape): a
/// cabin over the forward frame colliders with a canopy and engine cowling, a tail boom to the
/// tail rotor, the fin and tailplane from the aerodynamic surfaces, the mast and the skids from
/// the gear colliders. The rotors are separate (see [`RotorVisual`]). Body frame FLU.
#[derive(Clone, Debug)]
pub struct HelicopterVisual {
    pub body: MeshData,
    /// Main and tail rotor.
    pub rotors: [RotorVisual; 2],
    /// Largest distance of any part from the centre of mass (m), for cameras.
    pub span: f32,
    /// Pilot's eye point (m), for the first-person camera.
    pub eye: DVec3,
}

pub fn helicopter(def: &autonomousim_vehicles::rotorcraft::HelicopterDef) -> HelicopterVisual {
    use autonomousim_vehicles::multirotor::ColliderPart;
    let main = &def.main_rotor;
    let tail_hub = def.tail_rotor.hub;
    let r_main = main.rotor.radius;
    let r_tail = def.tail_rotor.rotor.radius;
    let body_color = srgb([214, 88, 40]);
    let white = srgb([232, 234, 238]);
    let grey = srgb([150, 154, 160]);
    let dark = srgb([40, 42, 46]);
    let glass = srgb([60, 90, 120]);
    let frame: Vec<_> = def.colliders.iter().filter(|k| k.part == ColliderPart::Frame).collect();
    // The cabin: the frame colliders ahead of 40 % of the way to the tail rotor (at least the
    // foremost one).
    let split = 0.4 * tail_hub.x;
    let mut cabin: Vec<_> = frame.iter().filter(|k| k.center.x > split).collect();
    if cabin.is_empty() {
        cabin.extend(frame.iter().max_by(|a, b| a.center.x.total_cmp(&b.center.x)));
    }
    // Without frame colliders: a pod of a tenth of the rotor radius.
    let fallback = autonomousim_vehicles::multirotor::ColliderDef {
        center: DVec3::ZERO,
        radius: 0.1 * r_main,
        part: ColliderPart::Frame,
    };
    let fallback = [&fallback];
    let cabin: Vec<_> = if cabin.is_empty() { fallback.to_vec() } else { cabin.into_iter().copied().collect() };
    let largest = cabin.iter().max_by(|a, b| a.radius.total_cmp(&b.radius)).unwrap();
    let r = largest.radius;
    let mut body = MeshData::new();
    // Cabin: the hull of the colliders, a little slimmer than tall.
    let dirs = mesh::icosphere(1.0, 1, false, white).positions;
    let points: Vec<DVec3> = cabin
        .iter()
        .flat_map(|k| {
            dirs.iter()
                .map(move |d| k.center + k.radius * DVec3::new(1.1 * d[0] as f64, 0.8 * d[1] as f64, d[2] as f64))
        })
        .collect();
    body.append(&mesh::convex_hull(&points, body_color));
    let front = cabin.iter().max_by(|a, b| (a.center.x + a.radius).total_cmp(&(b.center.x + b.radius))).unwrap();
    let nose = front.center.x + 1.1 * front.radius;
    let rear = cabin.iter().map(|k| k.center.x - 1.1 * k.radius).fold(f64::MAX, f64::min);
    let top = cabin.iter().map(|k| k.center.z + k.radius).fold(f64::MIN, f64::max);
    let bottom = cabin.iter().map(|k| k.center.z - k.radius).fold(f64::MAX, f64::min);
    // Canopy on the upper front of the foremost collider; the pilot sits behind it.
    let rf = front.radius;
    let canopy_at = front.center + rf * DVec3::new(0.5, 0.0, 0.4);
    let canopy = mesh::ellipsoid(Vec3::new((0.6 * rf) as f32, (0.7 * rf) as f32, (0.5 * rf) as f32), 1, glass);
    body.append_transformed(&canopy, DQuat::IDENTITY, canopy_at);
    let eye = DVec3::new(canopy_at.x - 0.2 * rf, 0.0, canopy_at.z + 0.1 * rf);
    // Engine cowling on the cabin roof, under the mast.
    let cowl = mesh::ellipsoid(Vec3::new((0.7 * r) as f32, (0.4 * r) as f32, (0.3 * r) as f32), 1, body_color);
    body.append_transformed(&cowl, DQuat::IDENTITY, DVec3::new(main.hub.x - 0.2 * r, 0.0, top - 0.1 * r));
    // Mast from the cowling up the shaft to the hub, and the hub.
    let axis = main.axis.normalize_or(DVec3::Z);
    let length = ((main.hub.z - top + 0.2 * r) / axis.z.max(0.2)).max(0.05 * r_main);
    let mast = mesh::cylinder((0.08 * r) as f32, (0.5 * length) as f32, 10, grey);
    let tilt = DQuat::from_rotation_arc(DVec3::Z, axis);
    body.append_transformed(&mast, tilt, main.hub - axis * 0.5 * length);
    let hub = mesh::cylinder((0.18 * r) as f32, (0.06 * r) as f32, 12, dark);
    body.append_transformed(&hub, tilt, main.hub);
    // Tail boom from the upper back of the cabin, tapering to the tail rotor (about half its
    // radius below the hub), and a gearbox there.
    let section = |x: f64, s: f64, z: f64| {
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
            .map(|(sy, sz)| DVec3::new(x, sy * 0.5 * s, z + sz * 0.5 * s))
    };
    let root_z = largest.center.z + 0.3 * r;
    let end = DVec3::new(tail_hub.x, 0.0, tail_hub.z - 0.5 * r_tail);
    let boom: Vec<_> =
        section(rear + 0.5 * r, 0.4 * r, root_z).into_iter().chain(section(end.x, 0.15 * r, end.z)).collect();
    body.append(&mesh::convex_hull(&boom, body_color));
    let gearbox = (tail_hub - end).length().max(0.1 * r);
    let stub: Vec<_> = section(end.x + 0.1 * r, 0.18 * r, end.z)
        .into_iter()
        .chain(section(tail_hub.x, 0.12 * r, tail_hub.z))
        .collect();
    if gearbox > 1e-6 {
        body.append(&mesh::convex_hull(&stub, body_color));
    }
    let tail_axis = def.tail_rotor.axis.normalize_or(DVec3::Y);
    let tail_hub_mesh = mesh::cylinder((0.12 * r_tail) as f32, (0.08 * r_tail) as f32, 10, dark);
    body.append_transformed(&tail_hub_mesh, DQuat::from_rotation_arc(DVec3::Z, tail_axis), tail_hub);
    // Fin and tailplane: plates of the surfaces' span and chord, rolled like them.
    for s in &def.surfaces {
        let t = (0.08 * s.chord).max(0.004);
        let plate = mesh::cuboid(Vec3::new((0.5 * s.chord) as f32, (0.5 * s.span) as f32, t as f32), white);
        body.append_transformed(&plate, DQuat::from_rotation_x(s.roll), s.position);
    }
    // Skids: a tube along each row of gear colliders (a quarter longer, turned up at the front)
    // and two cross tubes up to the cabin floor.
    let gear: Vec<_> = def.colliders.iter().filter(|k| k.part == ColliderPart::Gear).collect();
    let mut reach = 0.0f64;
    for side in [1.0f64, -1.0] {
        let row: Vec<_> = gear.iter().filter(|k| k.center.y * side > 1e-9).collect();
        let (Some(first), Some(last)) = (
            row.iter().max_by(|a, b| a.center.x.total_cmp(&b.center.x)),
            row.iter().min_by(|a, b| a.center.x.total_cmp(&b.center.x)),
        ) else {
            continue;
        };
        let rad = first.radius;
        let (a, b) = (first.center, last.center);
        let extra = 0.12 * (a.x - b.x).max(4.0 * rad);
        let (a, b) = (a + DVec3::X * extra, b - DVec3::X * extra);
        let tube = |from: DVec3, to: DVec3, radius: f64| {
            let d = to - from;
            let m = mesh::cylinder(radius as f32, (0.5 * d.length()) as f32, 8, grey);
            (m, DQuat::from_rotation_arc(DVec3::Z, d.normalize_or(DVec3::Z)), 0.5 * (from + to))
        };
        let mut add = |(m, rot, at): (MeshData, DQuat, DVec3)| body.append_transformed(&m, rot, at);
        add(tube(b, a, rad));
        let tip = a + DVec3::new(2.0 * extra, 0.0, 1.5 * extra);
        add(tube(a, tip, rad));
        for f in [0.25, 0.75] {
            let foot = b.lerp(a, f);
            add(tube(foot, DVec3::new(foot.x, 0.4 * foot.y, bottom + 0.2 * r), 0.8 * rad));
        }
        reach = reach.max(tip.length());
    }
    let blade = |rotor: &autonomousim_vehicles::rotorcraft::RotorDef| {
        let (root, tip) = (rotor.root_cutout.max(0.08) * rotor.radius, rotor.radius);
        let c = rotor.chord;
        let mut m = MeshData::new();
        let half = Vec3::new((0.5 * (tip - root)) as f32, (0.5 * c) as f32, (0.06 * c) as f32);
        m.append_transformed(&mesh::cuboid(half, dark), DQuat::IDENTITY, DVec3::new(0.5 * (root + tip), 0.0, 0.0));
        // White tips.
        let cap = Vec3::new((0.03 * tip) as f32, (0.52 * c) as f32, (0.07 * c) as f32);
        m.append_transformed(&mesh::cuboid(cap, white), DQuat::IDENTITY, DVec3::new(0.97 * tip, 0.0, 0.0));
        m
    };
    let rotor = |m: &autonomousim_vehicles::rotorcraft::RotorMount| RotorVisual {
        hub: m.hub,
        frame: DQuat::from_mat3(&m.frame()),
        radius: m.rotor.radius as f32,
        blades: m.rotor.blades,
        spin: m.rotor.spin_sign(),
        blade: blade(&m.rotor),
    };
    let span = (main.hub.length() + r_main).max(tail_hub.length() + r_tail).max(nose.abs()).max(reach) as f32;
    HelicopterVisual { body, rotors: [rotor(main), rotor(&def.tail_rotor)], span, eye }
}

/// Rotation of a rotor's shaft frame to its tip-path plane, tilted by `flap` (`[β₁c, β₁s]`
/// rad, toward +x and +y of the shaft frame).
pub fn rotor_tilt(flap: [f64; 2]) -> DQuat {
    DQuat::from_rotation_arc(DVec3::Z, DVec3::new(flap[0], flap[1], 1.0).normalize())
}

/// Rotation (in the tip-path-plane frame, turned with the rotor) of blade `k` of `count`, coned
/// up by `coning` (rad).
pub fn blade_rotation(k: u32, count: u32, coning: f64) -> DQuat {
    let azimuth = std::f64::consts::TAU * f64::from(k) / f64::from(count.max(1));
    DQuat::from_rotation_z(azimuth) * DQuat::from_rotation_y(-coning)
}

/// Visual of a wheeled vehicle: the body in the chassis frame (FLU) and one mesh per wheel in
/// its spinning link's frame (spin axis y), to be posed from the simulated wheels.
#[derive(Clone, Debug)]
pub struct WheeledVisual {
    /// The towing unit's body, in its frame.
    pub body: MeshData,
    /// Bodies of the units behind it (unit `u` at `u − 1`), each in its unit's frame.
    pub units: Vec<MeshData>,
    /// Per wheel, a mesh about its centre.
    pub wheels: Vec<MeshData>,
    /// Largest distance of a wheel's outer edge or of a body from the chassis origin, all
    /// units in line (m), for cameras.
    pub span: f32,
    /// Driver's eye point in the chassis frame (m), for the first-person camera.
    pub eye: DVec3,
    /// Reversing camera point in the last unit's frame (m): at its tail, below its top.
    pub rear_eye: DVec3,
    /// Per wheel with suspension: body-side ends of the strut and of the lower arm, in the
    /// frame of the wheel's unit (m). The other ends follow the wheel centre.
    pub links: Vec<Option<[DVec3; 2]>>,
    /// Link (unit cylinder along z, from −0.5 to 0.5) scaled to the links' thickness.
    pub link: MeshData,
    /// Single-track vehicles: the steered parts and the rider, posed from the state.
    pub single_track: Option<crate::single_track::SingleTrackVisual>,
}

/// Box between a unit's wheels: half extents and centre, in the unit's frame.
struct WheelBox {
    half: DVec3,
    centre: DVec3,
}

/// Build the visual of a wheeled vehicle: a box over the wheelbase and track with a red nose
/// (and a cabin on cars), dark tyres with a light marker on the rim so that spin is visible,
/// and a strut and lower arm per suspended wheel. Units behind the towing unit get a box over
/// their colliders (trailer bodies), a low frame between their wheels (dollies) or a bar to
/// the next joint (drawbars).
pub fn wheeled(def: &autonomousim_vehicles::ground::WheeledDef) -> WheeledVisual {
    if def.is_single_track()
        && let Some((head, front)) = def.steering_head()
    {
        return crate::single_track::visual(def, head, front);
    }
    let n = def.num_wheels();
    let positions: Vec<DVec3> = (0..n).map(|w| def.wheel_position(w)).collect();
    let tire = |w: usize| def.wheel_tire(w);
    let on = |u: usize| (0..n).filter(move |&w| def.wheel_unit(w) == u);
    let width = (0..n).map(|w| tire(w).width()).fold(0.0, f64::max);
    let radius = (0..n).map(|w| tire(w).radius()).fold(0.0, f64::max);
    // Between the wheels, from the axle line up to a little above the tyre tops.
    let wheel_box = |u: usize| {
        let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
        let mut radius = 0.0f64;
        for w in on(u) {
            let (p, r) = (positions[w], tire(w).radius());
            lo = lo.min(p - DVec3::new(r, 0.0, 0.0));
            hi = hi.max(p + DVec3::new(r, 0.0, r));
            radius = radius.max(r);
        }
        let half = DVec3::new(0.5 * (hi.x - lo.x), (0.5 * (hi.y - lo.y) - 0.6 * width).max(0.3 * radius), 0.4 * radius);
        let centre = DVec3::new(0.5 * (hi.x + lo.x), 0.5 * (hi.y + lo.y), lo.z + 0.2 * radius + half.z);
        WheelBox { half, centre }
    };
    let h = |v: DVec3| v.as_vec3();
    let WheelBox { half, centre } = wheel_box(0);
    let mut body = MeshData::new();
    body.append_transformed(&mesh::cuboid(h(half), srgb([70, 110, 150])), DQuat::IDENTITY, centre);
    let nose = mesh::cuboid(h(DVec3::new(0.08 * half.x, 0.8 * half.y, 0.3 * half.z)), srgb([200, 40, 36]));
    body.append_transformed(&nose, DQuat::IDENTITY, centre + DVec3::new(half.x, 0.0, 0.5 * half.z));
    let top = centre.z + half.z;
    // Trucks and tractors (large wheels) with colliders high above the frame: a cab over the
    // frontmost. Cars (wheels larger than a robot's): a cabin over the middle, a
    // little behind centre.
    let tall = def.colliders.iter().filter(|c| c.center.z + c.radius > top + 1.0);
    let cab = tall.max_by(|a, b| a.center.x.total_cmp(&b.center.x)).filter(|_| radius >= 0.5);
    let car = radius > 0.2;
    let eye = if let Some(c) = cab {
        let (bottom, roof) = (top - 0.1 * half.z, c.center.z + c.radius);
        let cab = DVec3::new(0.8 * c.radius, 0.95 * half.y.max(c.radius), 0.5 * (roof - bottom));
        let at = DVec3::new(c.center.x, centre.y, bottom + cab.z);
        body.append_transformed(&mesh::cuboid(h(cab), srgb([150, 185, 210])), DQuat::IDENTITY, at);
        let glass = mesh::cuboid(h(DVec3::new(0.02, 0.9 * cab.y, 0.25 * cab.z)), srgb([40, 50, 60]));
        body.append_transformed(&glass, DQuat::IDENTITY, at + DVec3::new(cab.x, 0.0, 0.45 * cab.z));
        DVec3::new(at.x + 0.3 * cab.x, centre.y + 0.4 * cab.y, at.z + 0.5 * cab.z)
    } else if car {
        let cabin = DVec3::new(0.3 * half.x, 0.85 * half.y, 0.55 * radius.max(0.35));
        let at = DVec3::new(centre.x - 0.1 * half.x, centre.y, top + cabin.z);
        body.append_transformed(&mesh::cuboid(h(cabin), srgb([150, 185, 210])), DQuat::IDENTITY, at);
        DVec3::new(at.x + 0.2 * cabin.x, centre.y + 0.4 * cabin.y, top + 1.2 * cabin.z)
    } else {
        DVec3::new(centre.x + 0.8 * half.x, centre.y, top + 0.3 * half.z)
    };
    let mut tops = vec![top];
    let mut halves = vec![half.y];
    let mut reach = DVec3::new(centre.x - half.x, 0.0, 0.0).length().max((centre + half).length());
    let units = (1..def.num_units())
        .map(|u| {
            let unit = &def.units[u - 1];
            let origin = def.unit_origin(u);
            let mut m = MeshData::new();
            let (lo, hi) = if !unit.colliders.is_empty() {
                // The body: over the colliders, a panel line at the front.
                let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
                for c in &unit.colliders {
                    lo = lo.min(c.center - DVec3::splat(c.radius));
                    hi = hi.max(c.center + DVec3::splat(c.radius));
                }
                let (half, centre) = (0.5 * (hi - lo), 0.5 * (hi + lo));
                m.append_transformed(&mesh::cuboid(h(half), srgb([185, 180, 165])), DQuat::IDENTITY, centre);
                let stripe = mesh::cuboid(h(DVec3::new(0.02, 1.01 * half.y, 0.06 * half.z)), srgb([200, 40, 36]));
                m.append_transformed(&stripe, DQuat::IDENTITY, centre - DVec3::new(0.0, 0.0, 0.8 * half.z));
                (lo, hi)
            } else if on(u).next().is_some() {
                let WheelBox { half, centre } = wheel_box(u);
                m.append_transformed(&mesh::cuboid(h(half), srgb([90, 90, 95])), DQuat::IDENTITY, centre);
                (centre - half, centre + half)
            } else {
                // A bar from the eye to each unit hanging from this one.
                let r = (0.04 * radius).max(0.03);
                let mut far = DVec3::ZERO;
                for child in def.units.iter().filter(|c| c.parent == u) {
                    let d = child.position;
                    let rot = DQuat::from_rotation_arc(DVec3::X, d.normalize_or(DVec3::X));
                    let bar = mesh::cuboid(h(DVec3::new(0.5 * d.length(), r, r)), srgb([60, 60, 65]));
                    m.append_transformed(&bar, rot, 0.5 * d);
                    far = if d.length() > far.length() { d } else { far };
                }
                (far.min(DVec3::ZERO) - DVec3::splat(r), far.max(DVec3::ZERO) + DVec3::splat(r))
            };
            tops.push(hi.z);
            halves.push(0.5 * (hi.y - lo.y));
            reach = reach.max((origin + lo).length()).max((origin + DVec3::new(lo.x, hi.y, hi.z)).length());
            m
        })
        .collect();
    let axis = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
    let wheels = (0..n)
        .map(|w| {
            let (mut r, mut b) = (tire(w).radius() as f32, tire(w).width() as f32);
            // Road wheels inside a track: the band (drawn separately) wraps them.
            if def.track.is_some() && matches!(tire(w).model, TireModel::Track(_)) {
                (r, b) = (r - track_thickness(r as f64) as f32, 0.6 * b);
            }
            let mut m = MeshData::new();
            // Dual wheels: two tyres side by side.
            let section = if def.track.is_some() { b } else { tire(w).section_width() as f32 };
            let offsets = tire(w).dual.map_or(vec![0.0], |s| vec![-0.5 * s, 0.5 * s]);
            for y in offsets {
                let tyre = mesh::cylinder(r, 0.5 * section, 20, srgb([30, 30, 32]));
                m.append_transformed(&tyre, axis, DVec3::new(0.0, y, 0.0));
            }
            let marker = mesh::cuboid(Vec3::new(0.12 * r, 0.52 * b, 0.12 * r), srgb([220, 220, 220]));
            m.append_transformed(&marker, DQuat::IDENTITY, DVec3::new(0.0, 0.0, 0.7 * r as f64));
            m
        })
        .collect();
    // Strut from above the wheel, inboard, and lower arm to the body's side, below the axle line.
    let links = (0..n)
        .map(|w| {
            def.axles[def.wheel_axle(w)].suspension.as_ref()?;
            let (p, r) = (positions[w], tire(w).radius());
            let inboard = p.y.signum() * (0.5 * width + 0.25 * r);
            let strut = DVec3::new(p.x, p.y - inboard, p.z + 0.9 * r);
            let side = halves[def.wheel_unit(w)];
            let arm = DVec3::new(p.x, (p.y - 2.0 * inboard).abs().min(side).copysign(p.y), p.z - 0.2 * r);
            Some([strut, arm])
        })
        .collect();
    let link = mesh::cylinder((0.06 * radius) as f32, 0.5, 8, srgb([90, 90, 95]));
    let wheel_reach = (0..n).map(|w| def.wheel_position_in_line(w).length() + tire(w).radius());
    let span = wheel_reach.fold(0.0, f64::max).max(if def.num_units() > 1 { reach } else { 0.0 }) as f32;
    let last = def.num_units() - 1;
    let rear_eye = def.tail() + DVec3::new(0.0, 0.0, 0.9 * tops[last]);
    WheeledVisual { body, units, wheels, span, eye, rear_eye, links, link, single_track: None }
}

/// Thickness of a track's band (m) around road wheels whose track patch has this radius
/// (road wheel plus band).
pub fn track_thickness(patch_radius: f64) -> f64 {
    (0.18 * patch_radius).clamp(0.01, 0.07)
}

/// A track's band at lateral position `y` in the chassis frame (FLU), `width` wide: the loop
/// around its sprocket, road wheels and idler, given as circles `(centre (x, z), radius)` of
/// the band's outer surface, with grousers every `pitch` metres. The band has run `travel`
/// metres (forward driving is positive: the ground run moves backwards relative to the hull).
pub fn track_band(circles: &[(DVec2, f64)], y: f64, width: f64, thickness: f64, pitch: f64, travel: f64) -> MeshData {
    let mut m = MeshData::new();
    // The loop: convex hull of the circles, counter-clockwise in (x, z).
    let mut pts: Vec<DVec2> = circles
        .iter()
        .flat_map(|&(c, r)| (0..24).map(move |k| c + r * DVec2::from_angle(std::f64::consts::TAU * k as f64 / 24.0)))
        .collect();
    let hull = hull_2d(&mut pts);
    let n = hull.len();
    if n < 3 {
        return m;
    }
    // Outward normals of the edges (i → i + 1) and at the vertices (mitred).
    let edge_normal = |i: usize| {
        let e = (hull[(i + 1) % n] - hull[i]).normalize_or_zero();
        DVec2::new(e.y, -e.x)
    };
    let inner: Vec<DVec2> = (0..n)
        .map(|i| {
            let (a, b) = (edge_normal((i + n - 1) % n), edge_normal(i));
            let v = (a + b).normalize_or(b);
            hull[i] - thickness / v.dot(b).max(0.5) * v
        })
        .collect();
    let at = |p: DVec2, side: f64| DVec3::new(p.x, y + 0.5 * side * width, p.y).as_vec3();
    let band = srgb([46, 46, 50]);
    // A quad, wound to face `out`.
    let quad = |m: &mut MeshData, q: [Vec3; 4], out: Vec3| {
        let flip = (q[1] - q[0]).cross(q[2] - q[0]).dot(out) < 0.0;
        let [a, b, c, d] = if flip { [q[0], q[3], q[2], q[1]] } else { q };
        m.push_flat_triangle(a, b, c, band);
        m.push_flat_triangle(a, c, d, band);
    };
    for i in 0..n {
        let j = (i + 1) % n;
        let o = edge_normal(i);
        let out = Vec3::new(o.x as f32, 0.0, o.y as f32);
        quad(&mut m, [at(hull[i], -1.0), at(hull[j], -1.0), at(hull[j], 1.0), at(hull[i], 1.0)], out);
        quad(&mut m, [at(inner[i], -1.0), at(inner[j], -1.0), at(inner[j], 1.0), at(inner[i], 1.0)], -out);
        for side in [-1.0, 1.0] {
            let q = [at(hull[i], side), at(hull[j], side), at(inner[j], side), at(inner[i], side)];
            quad(&mut m, q, Vec3::Y * side as f32);
        }
    }
    // Grousers at fixed places on the band, which moves against the loop's direction.
    let lengths: Vec<f64> = (0..n).map(|i| hull[i].distance(hull[(i + 1) % n])).collect();
    let total: f64 = lengths.iter().sum();
    let count = (total / pitch).floor().max(1.0) as usize;
    let spacing = total / count as f64;
    let grouser = mesh::cuboid(DVec3::new(0.2 * spacing, 0.5 * width, 0.25 * thickness).as_vec3(), srgb([92, 92, 98]));
    let (mut edge, mut start) = (0, 0.0);
    for k in 0..count {
        let s = (k as f64 * spacing - travel).rem_euclid(total);
        if s < start {
            (edge, start) = (0, 0.0);
        }
        while start + lengths[edge] < s && edge + 1 < n {
            start += lengths[edge];
            edge += 1;
        }
        let t = (hull[(edge + 1) % n] - hull[edge]).normalize_or(DVec2::X);
        let p = hull[edge] + (s - start) * t;
        let o = edge_normal(edge);
        let (t3, o3) = (DVec3::new(t.x, 0.0, t.y), DVec3::new(o.x, 0.0, o.y));
        let rot = DQuat::from_mat3(&DMat3::from_cols(t3, DVec3::Y, t3.cross(DVec3::Y)));
        m.append_transformed(&grouser, rot, DVec3::new(p.x, y, p.y) + 0.25 * thickness * o3);
    }
    m
}

/// Convex hull (Andrew's monotone chain), counter-clockwise, without repeated points.
fn hull_2d(pts: &mut [DVec2]) -> Vec<DVec2> {
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    let cross = |o: DVec2, a: DVec2, b: DVec2| (a - o).perp_dot(b - o);
    let mut hull: Vec<DVec2> = Vec::with_capacity(pts.len() + 1);
    for pass in 0..2 {
        let floor = hull.len();
        let iter: Box<dyn Iterator<Item = &DVec2>> =
            if pass == 0 { Box::new(pts.iter()) } else { Box::new(pts.iter().rev()) };
        for &p in iter {
            while hull.len() >= floor + 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 1e-12 {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull.dedup_by(|a, b| a.distance(*b) < 1e-9);
    hull
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::chunks;
    use autonomousim_vehicles::multirotor::ColliderPart;
    use autonomousim_vehicles::presets;
    use autonomousim_world::testworlds;

    #[test]
    fn track_band_wraps_the_wheels_and_moves_with_travel() {
        // Sprocket, two road wheels, idler: the loop's bottom run is flat under the road wheels.
        let circles = [
            (DVec2::new(0.0, 0.0), 0.3),
            (DVec2::new(-1.0, -0.3), 0.35),
            (DVec2::new(-2.0, -0.3), 0.35),
            (DVec2::new(-3.0, -0.1), 0.3),
        ];
        let band = |travel| track_band(&circles, 1.0, 0.4, 0.06, 0.15, travel);
        let m = band(0.0);
        let (lo, hi) = m.bounds().unwrap();
        // Grousers stand half the band's thickness out of its outer surface.
        assert!((lo.z as f64 + 0.65 + 0.03).abs() < 1e-3, "{lo}");
        assert!((lo.y - 0.8).abs() < 1e-6 && (hi.y - 1.2).abs() < 1e-6);
        assert!((lo.x as f64 + 3.3).abs() < 0.04 && (hi.x as f64 - 0.3).abs() < 0.04, "{lo} {hi}");
        assert!(m.triangle_count() > 100);
        // A band that ran a whole grouser spacing looks the same; half of one does not.
        let spacing = |m: &MeshData| {
            let mut p: Vec<[i64; 3]> = m.positions.iter().map(|v| v.map(|c| (c as f64 * 1e3).round() as i64)).collect();
            p.sort();
            p
        };
        let total: f64 = {
            let mut pts: Vec<DVec2> = circles
                .iter()
                .flat_map(|&(c, r)| {
                    (0..24).map(move |k| c + r * DVec2::from_angle(std::f64::consts::TAU * k as f64 / 24.0))
                })
                .collect();
            let h = hull_2d(&mut pts);
            (0..h.len()).map(|i| h[i].distance(h[(i + 1) % h.len()])).sum()
        };
        let pitch = total / (total / 0.15).floor();
        let same =
            |a: &[[i64; 3]], b: &[[i64; 3]]| a.iter().zip(b).all(|(p, q)| (0..3).all(|k| (p[k] - q[k]).abs() <= 1));
        assert!(same(&spacing(&m), &spacing(&band(pitch))));
        assert!(!same(&spacing(&m), &spacing(&band(0.5 * pitch))));
        // Forward travel moves the grousers on the ground run backwards (−x): the grouser
        // nearest below x = −1.5 moves by the travel.
        let under = |m: &MeshData| {
            let low: Vec<f32> = m.positions.iter().filter(|p| p[2] < -0.66).map(|p| p[0]).collect();
            low.iter().copied().filter(|x| *x < -1.5).fold(f32::NEG_INFINITY, f32::max)
        };
        let d = under(&band(0.02)) - under(&m);
        assert!((d as f64 + 0.02).abs() < 1e-3, "{d}");
    }

    #[test]
    fn props_land_in_the_chunk_below_them() {
        let w = testworlds::forest_patch(120.0, 150.0, 3);
        let g = w.grid();
        let cs = chunks(g, 32);
        let props = props_by_chunk(&w, &cs, 32, PropDetail::default());
        assert_eq!(props.len(), cs.len());
        let total: usize = props.iter().map(MeshData::triangle_count).sum();
        assert!(total > 100 * w.obstacle_set().len() / 10, "{total}");
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
    fn fixed_wing_visual_matches_its_definition() {
        for name in ["aerosonde_like", "c172_like"] {
            let def = presets::fixed_wing(name).unwrap();
            let v = fixed_wing(&def);
            // The wing reaches the span; the body spans nose to tail.
            let half = v.body.positions.iter().map(|p| p[1].abs()).fold(0.0, f32::max);
            let b = def.geometry.span as f32;
            assert!((half - 0.5 * b).abs() < 0.02 * b, "{name}: {half} vs {b}");
            let (lo, hi) = v.body.bounds().unwrap();
            let frame = def.colliders.iter().filter(|c| c.part == ColliderPart::Frame).map(|c| c.center.x);
            let (tail, nose) = (frame.clone().fold(f64::MAX, f64::min), frame.fold(f64::MIN, f64::max));
            assert!(f64::from(lo.x) <= tail && f64::from(hi.x) >= nose, "{name}: {lo} {hi}");
            // Two ailerons, two flaps, elevator and rudder; each hinged at the body.
            let count = |k: usize| v.surfaces.iter().filter(|s| s.control == k).count();
            assert_eq!([count(0), count(1), count(2), count(3)], [2, 1, 1, 2], "{name}");
            for s in &v.surfaces {
                assert!(!s.mesh.is_empty() && (s.axis.length() - 1.0).abs() < 1e-12);
                let (lo, hi) = s.mesh.bounds().unwrap();
                assert!(hi.x <= 1e-6 && lo.x < 0.0, "{name}: surfaces trail their hinge");
            }
            // The ailerons turn opposite ways.
            let ailerons: Vec<_> = v.surfaces.iter().filter(|s| s.control == 0).collect();
            assert_eq!(ailerons[0].axis, -ailerons[1].axis);
            assert!(v.propeller_radius > 0.0 && v.span >= 0.5 * b);
            assert!(v.eye.x > 0.0);
        }
    }

    #[test]
    fn helicopter_visual_matches_its_definition() {
        for name in ["bo105_like", "xcell60_like"] {
            let def = presets::helicopter(name).unwrap();
            let v = helicopter(&def);
            // The body reaches from the nose collider to the tail rotor and down to the skids.
            let (lo, hi) = v.body.bounds().unwrap();
            let frame = def.colliders.iter().filter(|c| c.part == ColliderPart::Frame);
            let nose = frame.map(|c| c.center.x + c.radius).fold(f64::MIN, f64::max);
            assert!(f64::from(hi.x) >= nose && f64::from(lo.x) <= def.tail_rotor.hub.x, "{name}: {lo} {hi}");
            let skid = def.colliders.iter().filter(|c| c.part == ColliderPart::Gear).map(|c| c.center.z);
            assert!(f64::from(lo.z) <= skid.fold(f64::MAX, f64::min), "{name}: {lo}");
            // The rotors: a blade from near the hub to the tip, the shaft frame along the axis.
            for (r, m) in v.rotors.iter().zip([&def.main_rotor, &def.tail_rotor]) {
                assert_eq!((r.blades, r.hub), (m.rotor.blades, m.hub));
                assert!(((r.frame * DVec3::Z) - m.axis.normalize()).length() < 1e-12, "{name}");
                let (blo, bhi) = r.blade.bounds().unwrap();
                assert!((bhi.x - r.radius).abs() < 0.05 * r.radius && blo.x > 0.0 && blo.x < 0.3 * r.radius, "{name}");
                assert_eq!(r.spin, m.rotor.spin_sign());
            }
            assert!(v.span as f64 >= def.main_rotor.rotor.radius && v.eye.x > 0.0, "{name}");
        }
        // Coning raises the blade tips; a positive β₁c tilts the disc toward +x.
        assert!((blade_rotation(0, 4, 0.1) * DVec3::X).z > 0.09);
        assert!(((blade_rotation(1, 4, 0.0) * DVec3::X) - DVec3::Y).length() < 1e-12);
        assert!((rotor_tilt([0.1, 0.0]) * DVec3::Z).x > 0.09 && (rotor_tilt([0.0, 0.1]) * DVec3::Z).y > 0.09);
    }

    #[test]
    fn tiltrotor_visual_matches_its_definition() {
        let def = presets::tiltrotor("quadtilt_like").unwrap();
        let v = tiltrotor(&def);
        // One pod per rotor at its pivot, the disc at the hub with the propeller's radius.
        assert_eq!(v.nacelles.len(), def.rotors.len());
        for (n, r) in v.nacelles.iter().zip(&def.rotors) {
            assert_eq!((n.pivot, n.offset), (r.pivot, r.offset));
            assert!((f64::from(n.radius) - 0.5 * def.propeller.diameter).abs() < 1e-6);
            let (lo, hi) = n.mesh.bounds().unwrap();
            assert!(lo.z < 0.0 && f64::from(hi.z) >= r.offset - 0.05, "{lo} {hi}");
        }
        // Every flapped surface has a flap visual, hinged on its trailing edge.
        let flapped: Vec<usize> = (0..def.surfaces.len()).filter(|&i| def.surfaces[i].flap.is_some()).collect();
        assert_eq!(v.surfaces.iter().map(|s| s.surface).collect::<Vec<_>>(), flapped);
        for s in &v.surfaces {
            let d = &def.surfaces[s.surface];
            assert!(s.hinge.x < d.position.x && (s.axis.length() - 1.0).abs() < 1e-12);
        }
        // The body spans the wings and the rotor pivots.
        let (lo, hi) = v.body.bounds().unwrap();
        let wing = def.surfaces.iter().map(|s| 0.5 * s.span).fold(0.0, f64::max);
        assert!(f64::from(hi.y) >= 0.95 * wing && f64::from(lo.y) <= -0.95 * wing, "{lo} {hi}");
        assert!(v.span as f64 >= wing && v.eye.x > 0.0);
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
                assert_eq!(l.is_some(), def.axles[def.wheel_axle(w)].suspension.is_some(), "{name}");
            }
            assert!((v.eye.x as f32) < hi.x && v.eye.z > def.wheel_position(0).z, "{name}: {}", v.eye);
        }
        assert!(wheeled(&presets::wheeled("sedan_like").unwrap()).links.iter().all(Option::is_some));
    }

    /// Rigs: a body per unit behind the tractor, in its frame, and a span over the whole rig.
    #[test]
    fn rig_visuals_cover_every_unit() {
        for (tractor, trailer) in [("truck_6x4", "semitrailer_3axle"), ("farm_tractor", "farm_trailer")] {
            let alone = presets::wheeled(tractor).unwrap();
            let def = alone.with_trailers(&[presets::trailer(trailer).unwrap()]).unwrap();
            let (v, v0) = (wheeled(&def), wheeled(&alone));
            assert_eq!(v.units.len(), def.num_units() - 1);
            assert_eq!(v.wheels.len(), def.num_wheels());
            // The tractor looks as without the trailer.
            assert_eq!(v.body.positions, v0.body.positions);
            let last = &v.units[def.num_units() - 2];
            let (lo, hi) = last.bounds().unwrap();
            // The body ends at the tail, the reversing camera looks from there.
            assert!((lo.x as f64 - def.tail().x).abs() < 1e-3, "{trailer}: {lo} vs {}", def.tail());
            assert!((v.rear_eye.x - def.tail().x).abs() < 1e-9 && v.rear_eye.z > 0.0 && (v.rear_eye.z as f32) < hi.z);
            let length = -(def.unit_origin(def.num_units() - 1).x + def.tail().x);
            assert!(v.span as f64 > length && v.span > 2.0 * v0.span, "{trailer}: {} vs {length}", v.span);
            for (u, m) in v.units.iter().enumerate() {
                assert!(m.triangle_count() > 0, "{trailer}: unit {}", u + 1);
            }
            if trailer == "farm_trailer" {
                // The drawbar reaches from the eye to the dolly's hinge.
                let (lo, hi) = v.units[0].bounds().unwrap();
                let hinge = def.units[1].position;
                assert!((hi.x as f64 - hinge.x.max(0.0)).abs() < 0.1 && (lo.x as f64 - hinge.x.min(0.0)).abs() < 0.1);
            }
        }
    }
}
