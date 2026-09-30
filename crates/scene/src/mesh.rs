//! Indexed triangle meshes with per-vertex normals and colours, and flat-shaded primitives.
//!
//! Positions and normals are in the frame of whoever builds the mesh (world ENU for terrain
//! and props, body FLU for vehicles); the renderer converts them. Colours are linear RGBA.

use glam::{DQuat, DVec3, Vec3};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// Linear RGBA.
    pub colors: Vec<[f32; 4]>,
    /// Counter-clockwise triangles (seen from the side the normals point to).
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn push_vertex(&mut self, p: Vec3, n: Vec3, c: [f32; 4]) -> u32 {
        let i = self.positions.len() as u32;
        self.positions.push(p.to_array());
        self.normals.push(n.to_array());
        self.colors.push(c);
        i
    }

    pub fn push_triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend_from_slice(&[a, b, c]);
    }

    /// A triangle with its own three vertices and face normal (flat shading).
    pub fn push_flat_triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]) {
        let n = (b - a).cross(c - a).normalize_or_zero();
        let i = self.push_vertex(a, n, color);
        self.push_vertex(b, n, color);
        self.push_vertex(c, n, color);
        self.push_triangle(i, i + 1, i + 2);
    }

    /// Append `other` moved by `rotation` and `translation` (in f64, then rounded).
    pub fn append_transformed(&mut self, other: &MeshData, rotation: DQuat, translation: DVec3) {
        let base = self.positions.len() as u32;
        for (p, n) in other.positions.iter().zip(&other.normals) {
            let p = rotation * DVec3::from(Vec3::from_array(*p)) + translation;
            let n = rotation * DVec3::from(Vec3::from_array(*n));
            self.positions.push(p.as_vec3().to_array());
            self.normals.push(n.as_vec3().to_array());
        }
        self.colors.extend_from_slice(&other.colors);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }

    /// Append `other` unchanged.
    pub fn append(&mut self, other: &MeshData) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.normals.extend_from_slice(&other.normals);
        self.colors.extend_from_slice(&other.colors);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }

    /// Multiply every vertex colour's RGB by `k`.
    pub fn tint(&mut self, k: f32) {
        for c in &mut self.colors {
            for v in &mut c[..3] {
                *v *= k;
            }
        }
    }

    /// Axis-aligned bounds `(min, max)`; `None` when empty.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut it = self.positions.iter().map(|p| Vec3::from_array(*p));
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
    }
}

/// sRGB-encoded 8-bit colour → linear RGBA.
pub fn srgb(c: [u8; 3]) -> [f32; 4] {
    let lin = |v: u8| {
        let s = f32::from(v) / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    [lin(c[0]), lin(c[1]), lin(c[2]), 1.0]
}

/// `color` with alpha `a`.
pub fn with_alpha(mut color: [f32; 4], a: f32) -> [f32; 4] {
    color[3] = a;
    color
}

// ------------------------------------------------------------------------------ primitives
//
// All flat-shaded, centred on the origin; axisymmetric shapes use the z axis.

fn ring(radius: f32, z: f32, segments: usize) -> Vec<Vec3> {
    (0..segments)
        .map(|k| {
            let a = std::f32::consts::TAU * k as f32 / segments as f32;
            Vec3::new(radius * a.cos(), radius * a.sin(), z)
        })
        .collect()
}

/// Cone with its base disc at `z = −half_height` and apex at `z = +half_height`.
pub fn cone(radius: f32, half_height: f32, segments: usize, color: [f32; 4]) -> MeshData {
    let mut m = MeshData::new();
    let base = ring(radius, -half_height, segments);
    let apex = Vec3::new(0.0, 0.0, half_height);
    let bottom = Vec3::new(0.0, 0.0, -half_height);
    for k in 0..segments {
        let (a, b) = (base[k], base[(k + 1) % segments]);
        m.push_flat_triangle(a, b, apex, color);
        m.push_flat_triangle(b, a, bottom, color);
    }
    m
}

/// Cylinder from `z = −half_height` to `+half_height`, with caps.
pub fn cylinder(radius: f32, half_height: f32, segments: usize, color: [f32; 4]) -> MeshData {
    let mut m = tube(radius, half_height, segments, color);
    let lo = ring(radius, -half_height, segments);
    let hi = ring(radius, half_height, segments);
    let (top, bottom) = (Vec3::new(0.0, 0.0, half_height), Vec3::new(0.0, 0.0, -half_height));
    for k in 0..segments {
        let j = (k + 1) % segments;
        m.push_flat_triangle(hi[k], hi[j], top, color);
        m.push_flat_triangle(lo[j], lo[k], bottom, color);
    }
    m
}

/// A flat landing pad of `radius` at `z = 0`, facing up: an orange disc inside a white ring
/// (the outer 30 %), with a white bar across the centre along x.
pub fn landing_pad(radius: f32) -> MeshData {
    const SEGMENTS: usize = 32;
    let (white, orange) = (srgb([235, 235, 235]), srgb([240, 110, 20]));
    let mut m = MeshData::new();
    let outer = ring(radius, 0.0, SEGMENTS);
    let inner = ring(0.7 * radius, 0.0, SEGMENTS);
    for k in 0..SEGMENTS {
        let j = (k + 1) % SEGMENTS;
        m.push_flat_triangle(inner[k], outer[k], outer[j], white);
        m.push_flat_triangle(inner[k], outer[j], inner[j], white);
        m.push_flat_triangle(Vec3::ZERO, inner[k], inner[j], orange);
    }
    // The bar lies 1 mm above the disc.
    let (a, b) = (0.5 * radius, 0.1 * radius);
    let q = |x: f32, y: f32| Vec3::new(x, y, 0.001);
    m.push_flat_triangle(q(-a, -b), q(a, -b), q(a, b), white);
    m.push_flat_triangle(q(-a, -b), q(a, b), q(-a, b), white);
    m
}

/// Open cylinder (no caps) from `z = −half_height` to `+half_height`, with smooth normals.
pub fn tube(radius: f32, half_height: f32, segments: usize, color: [f32; 4]) -> MeshData {
    let mut m = MeshData::new();
    for (p, z) in ring(radius, 0.0, segments).into_iter().flat_map(|p| [(p, -half_height), (p, half_height)]) {
        m.push_vertex(p + Vec3::Z * z, p / radius, color);
    }
    let n = segments as u32;
    for k in 0..n {
        let j = (k + 1) % n;
        m.push_triangle(2 * k, 2 * j, 2 * j + 1);
        m.push_triangle(2 * k, 2 * j + 1, 2 * k + 1);
    }
    m
}

/// Torus about the z axis: a tube of radius `minor` around the circle of radius `major` in the
/// xy plane, with smooth normals.
pub fn torus(major: f32, minor: f32, segments: usize, sides: usize, color: [f32; 4]) -> MeshData {
    let mut m = MeshData::new();
    for k in 0..segments {
        let a = std::f32::consts::TAU * k as f32 / segments as f32;
        let radial = Vec3::new(a.cos(), a.sin(), 0.0);
        for j in 0..sides {
            let b = std::f32::consts::TAU * j as f32 / sides as f32;
            let n = radial * b.cos() + Vec3::Z * b.sin();
            m.push_vertex(radial * major + n * minor, n, color);
        }
    }
    let (s, n) = (sides as u32, segments as u32);
    for k in 0..n {
        let k1 = (k + 1) % n;
        for j in 0..s {
            let j1 = (j + 1) % s;
            let (a, b, c, d) = (k * s + j, k1 * s + j, k1 * s + j1, k * s + j1);
            m.push_triangle(a, b, c);
            m.push_triangle(a, c, d);
        }
    }
    m
}

/// Ellipsoid with semi-axes `h` (a smooth icosphere scaled per axis).
pub fn ellipsoid(h: Vec3, subdivisions: u32, color: [f32; 4]) -> MeshData {
    let mut m = icosphere(1.0, subdivisions, true, color);
    for (p, n) in m.positions.iter_mut().zip(&mut m.normals) {
        *p = (Vec3::from_array(*p) * h).to_array();
        *n = (Vec3::from_array(*n) / h).normalize().to_array();
    }
    m
}

/// Box with half extents `h`.
pub fn cuboid(h: Vec3, color: [f32; 4]) -> MeshData {
    let mut m = MeshData::new();
    let c = |x: f32, y: f32, z: f32| Vec3::new(x * h.x, y * h.y, z * h.z);
    // Faces as (corner, u, v) with u × v pointing outwards.
    let faces = [
        (c(1., -1., -1.), Vec3::Y, Vec3::Z),
        (c(-1., 1., -1.), -Vec3::Y, Vec3::Z),
        (c(1., 1., -1.), -Vec3::X, Vec3::Z),
        (c(-1., -1., -1.), Vec3::X, Vec3::Z),
        (c(-1., -1., 1.), Vec3::X, Vec3::Y),
        (c(-1., 1., -1.), Vec3::X, -Vec3::Y),
    ];
    for (o, u, v) in faces {
        let (du, dv) = (u * 2.0 * h, v * 2.0 * h);
        m.push_flat_triangle(o, o + du, o + du + dv, color);
        m.push_flat_triangle(o, o + du + dv, o + dv, color);
    }
    m
}

/// A cuboid of half extents `h` whose side faces are cut at the heights (relative to the
/// centre, ascending) of `bands`: the side between `bands[k].0` and `bands[k].1` takes
/// `band_color`, the rest `color`. The same box as [`cuboid`], with more triangles.
pub fn banded_cuboid(h: Vec3, color: [f32; 4], bands: &[(f32, f32)], band_color: [f32; 4]) -> MeshData {
    let mut m = MeshData::new();
    let c = |x: f32, y: f32, z: f32| Vec3::new(x * h.x, y * h.y, z * h.z);
    // Slices of the sides, bottom to top.
    let mut slices = Vec::new();
    let mut z = -h.z;
    for &(a, b) in bands {
        let (a, b) = (a.clamp(z, h.z), b.clamp(z, h.z));
        if a > z {
            slices.push((z, a, color));
        }
        if b > a {
            slices.push((a, b, band_color));
        }
        z = z.max(b);
    }
    if h.z > z {
        slices.push((z, h.z, color));
    }
    // Sides as (bottom corner at z = 0, horizontal edge) with edge × z pointing outwards.
    let sides = [
        (c(1., -1., 0.), Vec3::Y * 2.0 * h.y),
        (c(-1., 1., 0.), -Vec3::Y * 2.0 * h.y),
        (c(1., 1., 0.), -Vec3::X * 2.0 * h.x),
        (c(-1., -1., 0.), Vec3::X * 2.0 * h.x),
    ];
    for (o, du) in sides {
        for &(z0, z1, col) in &slices {
            let (a, b) = (o + Vec3::Z * z0, o + Vec3::Z * z1);
            m.push_flat_triangle(a, a + du, b + du, col);
            m.push_flat_triangle(a, b + du, b, col);
        }
    }
    for (o, u, v) in [(c(-1., -1., 1.), Vec3::X, Vec3::Y), (c(-1., 1., -1.), Vec3::X, -Vec3::Y)] {
        let (du, dv) = (u * 2.0 * h, v * 2.0 * h);
        m.push_flat_triangle(o, o + du, o + du + dv, color);
        m.push_flat_triangle(o, o + du + dv, o + dv, color);
    }
    m
}

/// Gable roof over the rectangle `±h.x × ±h.y` at `z = 0`, its ridge along x at `z = h.z`:
/// two slopes and two gable ends, without a bottom.
pub fn gable_roof(h: Vec3, color: [f32; 4]) -> MeshData {
    let mut m = MeshData::new();
    let c = |x: f32, y: f32, z: f32| Vec3::new(x * h.x, y * h.y, z * h.z);
    for side in [-1.0, 1.0] {
        // Slope: eave to ridge, wound so that its normal points up and outwards.
        let (a, b) = (c(-side, side, 0.0), c(side, side, 0.0));
        let (r0, r1) = (c(-side, 0.0, 1.0), c(side, 0.0, 1.0));
        m.push_flat_triangle(a, r1, b, color);
        m.push_flat_triangle(a, r0, r1, color);
        // Gable end at x = side·h.x.
        m.push_flat_triangle(c(side, -side, 0.0), c(side, side, 0.0), c(side, 0.0, 1.0), color);
    }
    m
}

/// Icosphere: an icosahedron subdivided `subdivisions` times (20·4ⁿ triangles), flat-shaded
/// or `smooth` (shared vertices).
pub fn icosphere(radius: f32, subdivisions: u32, smooth: bool, color: [f32; 4]) -> MeshData {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let v = [
        [-1., t, 0.],
        [1., t, 0.],
        [-1., -t, 0.],
        [1., -t, 0.],
        [0., -1., t],
        [0., 1., t],
        [0., -1., -t],
        [0., 1., -t],
        [t, 0., -1.],
        [t, 0., 1.],
        [-t, 0., -1.],
        [-t, 0., 1.],
    ]
    .map(|p| Vec3::from_array(p).normalize());
    let faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    let mut tris: Vec<[Vec3; 3]> = faces.iter().map(|f| f.map(|i| v[i])).collect();
    for _ in 0..subdivisions {
        tris = tris
            .iter()
            .flat_map(|&[a, b, c]| {
                let (ab, bc, ca) = ((a + b).normalize(), (b + c).normalize(), (c + a).normalize());
                [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            })
            .collect();
    }
    let mut m = MeshData::new();
    if !smooth {
        for [a, b, c] in tris {
            m.push_flat_triangle(a * radius, b * radius, c * radius, color);
        }
        return m;
    }
    // Shared vertices with radial normals.
    let mut index = std::collections::HashMap::new();
    for t in tris {
        let [a, b, c] = t.map(|p| {
            *index.entry(p.to_array().map(f32::to_bits)).or_insert_with(|| m.push_vertex(p * radius, p, color))
        });
        m.push_triangle(a, b, c);
    }
    m
}

/// Convex hull of `points` (flat-shaded).
pub fn convex_hull(points: &[DVec3], color: [f32; 4]) -> MeshData {
    let (vertices, triangles) = autonomousim_core::parry::transformation::convex_hull(points);
    let mut m = MeshData::new();
    for [a, b, c] in triangles {
        let p = |i: u32| vertices[i as usize].as_vec3();
        m.push_flat_triangle(p(a), p(b), p(c), color);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Signed volume by the divergence theorem; positive when all faces point outwards.
    fn volume(m: &MeshData) -> f32 {
        m.indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let p = |i: u32| Vec3::from_array(m.positions[i as usize]);
                p(t[0]).dot(p(t[1]).cross(p(t[2]))) / 6.0
            })
            .sum()
    }

    /// Every face normal points away from the centre.
    fn outward(m: &MeshData) -> bool {
        m.indices.as_chunks::<3>().0.iter().all(|t| {
            let p = |i: u32| Vec3::from_array(m.positions[i as usize]);
            let n = Vec3::from_array(m.normals[t[0] as usize]);
            let centre = (p(t[0]) + p(t[1]) + p(t[2])) / 3.0;
            n.dot(centre) > 0.0
        })
    }

    #[test]
    fn primitives_are_closed_and_outward() {
        let c = [1.0; 4];
        let pi = std::f32::consts::PI;
        let cases = [
            (cuboid(Vec3::new(1.0, 2.0, 3.0), c), 48.0, 1e-4),
            (icosphere(2.0, 3, false, c), 4.0 / 3.0 * pi * 8.0, 0.02),
            (icosphere(2.0, 3, true, c), 4.0 / 3.0 * pi * 8.0, 0.02),
            (cylinder(1.0, 2.0, 64, c), pi * 4.0, 0.01),
            (cone(1.0, 1.5, 64, c), pi * 3.0 / 3.0, 0.01),
            (ellipsoid(Vec3::new(1.0, 2.0, 0.5), 3, c), 4.0 / 3.0 * pi, 0.02),
        ];
        for (m, expected, tol) in cases {
            assert!(outward(&m));
            let v = volume(&m);
            assert!((v / expected - 1.0).abs() < tol, "volume {v} vs {expected}");
        }
        let pts: Vec<DVec3> = [-1.0, 1.0]
            .iter()
            .flat_map(|&x| [-1.0, 1.0].map(move |y| (x, y)))
            .flat_map(|(x, y)| [-1.0, 1.0].map(move |z| DVec3::new(x, y, z)))
            .collect();
        let hull = convex_hull(&pts, c);
        assert!(outward(&hull));
        assert!((volume(&hull) - 8.0).abs() < 1e-4);
        assert_eq!(icosphere(1.0, 1, false, c).triangle_count(), 80);
        assert_eq!(icosphere(1.0, 1, true, c).vertex_count(), 42);
        // A torus faces away from its core circle and holds 2π²Rr².
        let t = torus(2.0, 0.5, 64, 32, c);
        assert!((volume(&t) / (2.0 * pi * pi * 2.0 * 0.25) - 1.0).abs() < 0.01, "{}", volume(&t));
        assert!(t.indices.as_chunks::<3>().0.iter().all(|f| {
            let p = |i: u32| Vec3::from_array(t.positions[i as usize]);
            let centre = (p(f[0]) + p(f[1]) + p(f[2])) / 3.0;
            let core = Vec3::new(centre.x, centre.y, 0.0).normalize() * 2.0;
            Vec3::from_array(t.normals[f[0] as usize]).dot(centre - core) > 0.0
        }));
        let t = tube(1.0, 1.0, 6, c);
        assert_eq!((t.vertex_count(), t.triangle_count()), (12, 12));
    }

    #[test]
    fn transforms_and_colours() {
        let mut m = MeshData::new();
        let unit = cuboid(Vec3::splat(0.5), srgb([255, 0, 0]));
        m.append_transformed(&unit, DQuat::from_rotation_z(0.3), DVec3::new(10.0, 0.0, 0.0));
        m.append(&unit);
        assert_eq!(m.triangle_count(), 24);
        let (lo, hi) = m.bounds().unwrap();
        assert!(lo.x < -0.49 && hi.x > 10.5 && hi.z < 0.51);
        // A flat-shaded box has 36 vertices; the second copy's indices are offset by them.
        assert_eq!((m.indices[36..].iter().min(), m.indices[36..].iter().max()), (Some(&36), Some(&71)));
        assert_eq!(srgb([255, 0, 0]), [1.0, 0.0, 0.0, 1.0]);
        assert!((srgb([128, 128, 128])[0] - 0.2158).abs() < 1e-3);
    }
}
