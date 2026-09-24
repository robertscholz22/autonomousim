//! Terrain and water meshes, cut into square chunks of the height grid.
//!
//! A chunk covers `size × size` cells (fewer at the far map edges). Its mesh samples every
//! `stride`-th vertex (level of detail); a skirt hangs from its border so that neighbours at
//! different strides leave no visible cracks. Vertex colours blend the materials of the
//! four cells around each vertex.

use crate::mesh::{MeshData, srgb, with_alpha};
use autonomousim_core::material::MaterialId;
use autonomousim_world::{HeightGrid, StaticWorld};
use glam::{DVec3, Vec3};

/// A chunk of the height grid: cells `[x0, x0 + nx) × [y0, y0 + ny)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Chunk {
    /// Chunk coordinates.
    pub cx: usize,
    pub cy: usize,
    /// First cell and cell counts.
    pub x0: usize,
    pub y0: usize,
    pub nx: usize,
    pub ny: usize,
}

impl Chunk {
    /// Bounding box `(min, max)` of the chunk's terrain in world coordinates.
    pub fn bounds(&self, grid: &HeightGrid) -> (DVec3, DVec3) {
        let c = grid.cell_size();
        let o = grid.origin();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for iy in self.y0..=self.y0 + self.ny {
            for ix in self.x0..=self.x0 + self.nx {
                let z = grid.vertex_height(ix, iy);
                lo = lo.min(z);
                hi = hi.max(z);
            }
        }
        let min = DVec3::new(o.x + self.x0 as f64 * c, o.y + self.y0 as f64 * c, lo);
        let max = DVec3::new(o.x + (self.x0 + self.nx) as f64 * c, o.y + (self.y0 + self.ny) as f64 * c, hi);
        (min, max)
    }
}

/// The chunks of `grid` with `size` cells per side, row by row.
pub fn chunks(grid: &HeightGrid, size: usize) -> Vec<Chunk> {
    assert!(size > 0);
    let (cw, ch) = grid.cells();
    let mut out = Vec::new();
    for (cy, y0) in (0..ch).step_by(size).enumerate() {
        for (cx, x0) in (0..cw).step_by(size).enumerate() {
            out.push(Chunk { cx, cy, x0, y0, nx: size.min(cw - x0), ny: size.min(ch - y0) });
        }
    }
    out
}

/// Sample indices `start, start + stride, …` ending exactly at `start + n`.
fn samples(start: usize, n: usize, stride: usize) -> Vec<usize> {
    let mut s: Vec<usize> = (start..start + n).step_by(stride).collect();
    s.push(start + n);
    s
}

/// Linear colour of every material of the map.
fn material_colors(world: &StaticWorld) -> Vec<[f32; 4]> {
    let table = world.materials();
    (0..table.len()).map(|i| srgb(table.get(MaterialId(i as u8)).color)).collect()
}

/// Terrain mesh of `chunk` sampling every `stride`-th vertex, with a skirt `skirt` metres deep.
pub fn terrain_chunk(world: &StaticWorld, chunk: &Chunk, stride: usize, skirt: f32) -> MeshData {
    let grid = world.terrain();
    let colors = material_colors(world);
    let (cw, ch) = grid.cells();
    let (vw, vh) = grid.dims();
    let c = grid.cell_size();
    let o = grid.origin();
    let xs = samples(chunk.x0, chunk.nx, stride.max(1));
    let ys = samples(chunk.y0, chunk.ny, stride.max(1));
    let mut m = MeshData::new();
    m.positions.reserve(xs.len() * ys.len() + 2 * (xs.len() + ys.len()));

    let pos = |ix: usize, iy: usize| {
        Vec3::new((o.x + ix as f64 * c) as f32, (o.y + iy as f64 * c) as f32, grid.vertex_height(ix, iy) as f32)
    };
    // Normal from central differences over the stride (one-sided at the map edge).
    let normal = |ix: usize, iy: usize| {
        let s = stride.max(1);
        let (x0, x1) = (ix.saturating_sub(s), (ix + s).min(vw - 1));
        let (y0, y1) = (iy.saturating_sub(s), (iy + s).min(vh - 1));
        let gx = (grid.vertex_height(x1, iy) - grid.vertex_height(x0, iy)) / ((x1 - x0) as f64 * c);
        let gy = (grid.vertex_height(ix, y1) - grid.vertex_height(ix, y0)) / ((y1 - y0) as f64 * c);
        DVec3::new(-gx, -gy, 1.0).normalize().as_vec3()
    };
    let color = |ix: usize, iy: usize| {
        let mut sum = [0.0f32; 4];
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (x, y) = ((ix + dx).saturating_sub(1).min(cw - 1), (iy + dy).saturating_sub(1).min(ch - 1));
            let k = colors.get(grid.cell_material(x, y).0 as usize).copied().unwrap_or([1.0, 0.0, 1.0, 1.0]);
            for i in 0..4 {
                sum[i] += 0.25 * k[i];
            }
        }
        sum
    };

    for &iy in &ys {
        for &ix in &xs {
            m.push_vertex(pos(ix, iy), normal(ix, iy), color(ix, iy));
        }
    }
    let w = xs.len() as u32;
    for j in 0..ys.len() as u32 - 1 {
        for i in 0..w - 1 {
            let a = j * w + i;
            // Split each quad along the diagonal that matches the height grid's triangulation.
            m.push_triangle(a, a + 1, a + w + 1);
            m.push_triangle(a, a + w + 1, a + w);
        }
    }

    // Skirts: each border edge gets a copy lowered by `skirt`, joined by a vertical strip that
    // faces outwards (edges run counter-clockwise around the chunk).
    let border = |m: &mut MeshData, ring: &[u32]| {
        let base = m.vertex_count() as u32;
        for &v in ring {
            let p = Vec3::from_array(m.positions[v as usize]) - Vec3::Z * skirt;
            let (n, col) = (Vec3::from_array(m.normals[v as usize]), m.colors[v as usize]);
            m.push_vertex(p, n, col);
        }
        for k in 0..ring.len() as u32 - 1 {
            let (a, b) = (ring[k as usize], ring[k as usize + 1]);
            let (la, lb) = (base + k, base + k + 1);
            m.push_triangle(a, lb, b);
            m.push_triangle(a, la, lb);
        }
    };
    let h = ys.len() as u32;
    let south: Vec<u32> = (0..w).collect();
    let east: Vec<u32> = (0..h).map(|j| j * w + w - 1).collect();
    let north: Vec<u32> = (0..w).rev().map(|i| (h - 1) * w + i).collect();
    let west: Vec<u32> = (0..h).rev().map(|j| j * w).collect();
    for ring in [south, east, north, west] {
        border(&mut m, &ring);
    }
    m
}

/// Water surface of `chunk`: one quad per run of wet cells with the same level in a row.
pub fn water_chunk(world: &StaticWorld, chunk: &Chunk, color: [f32; 4]) -> MeshData {
    let grid = world.terrain();
    let mut m = MeshData::new();
    if grid.water().is_none() {
        return m;
    }
    let c = grid.cell_size();
    let o = grid.origin();
    let x = |ix: usize| (o.x + ix as f64 * c) as f32;
    let y = |iy: usize| (o.y + iy as f64 * c) as f32;
    for cy in chunk.y0..chunk.y0 + chunk.ny {
        let mut cx = chunk.x0;
        let end = chunk.x0 + chunk.nx;
        while cx < end {
            let Some(level) = grid.cell_water(cx, cy) else {
                cx += 1;
                continue;
            };
            let start = cx;
            while cx < end && grid.cell_water(cx, cy) == Some(level) {
                cx += 1;
            }
            let z = level as f32;
            let i = m.push_vertex(Vec3::new(x(start), y(cy), z), Vec3::Z, color);
            m.push_vertex(Vec3::new(x(cx), y(cy), z), Vec3::Z, color);
            m.push_vertex(Vec3::new(x(cx), y(cy + 1), z), Vec3::Z, color);
            m.push_vertex(Vec3::new(x(start), y(cy + 1), z), Vec3::Z, color);
            m.push_triangle(i, i + 1, i + 2);
            m.push_triangle(i, i + 2, i + 3);
        }
    }
    m
}

/// Default water colour (linear RGBA, translucent).
pub fn water_color() -> [f32; 4] {
    with_alpha(srgb([46, 98, 132]), 0.78)
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::terrain::Terrain;
    use autonomousim_world::testworlds;

    #[test]
    fn chunks_tile_the_grid() {
        let w = testworlds::sine_hills(100.0, 3.0, 20.0);
        let g = w.terrain();
        let cs = chunks(g, 32);
        let (cw, ch) = g.cells();
        assert_eq!(cs.len(), cw.div_ceil(32) * ch.div_ceil(32));
        assert_eq!(cs.iter().map(|c| c.nx * c.ny).sum::<usize>(), cw * ch);
        let last = cs.last().unwrap();
        assert_eq!((last.x0 + last.nx, last.y0 + last.ny), (cw, ch));
    }

    #[test]
    fn chunk_meshes_follow_the_terrain() {
        let w = testworlds::sine_hills(100.0, 3.0, 20.0);
        let g = w.terrain();
        for stride in [1, 2, 4, 8, 64] {
            for chunk in chunks(g, 32) {
                let m = terrain_chunk(&w, &chunk, stride, 2.0);
                let (sx, sy) = (chunk.nx.div_ceil(stride) + 1, chunk.ny.div_ceil(stride) + 1);
                let grid_vertices = sx * sy;
                assert_eq!(m.vertex_count(), grid_vertices + 2 * (sx + sy));
                assert_eq!(m.triangle_count(), 2 * (sx - 1) * (sy - 1) + 4 * (sx - 1 + sy - 1));
                // Grid vertices lie on the terrain and normals point up.
                for (p, n) in m.positions[..grid_vertices].iter().zip(&m.normals) {
                    let z = g.height(f64::from(p[0]), f64::from(p[1]));
                    assert!((z - f64::from(p[2])).abs() < 1e-4, "{p:?} vs {z}");
                    assert!(n[2] > 0.5);
                }
                // Triangles of the surface face up.
                for t in m.indices[..6 * (sx - 1) * (sy - 1)].as_chunks::<3>().0 {
                    let p = |i: u32| Vec3::from_array(m.positions[i as usize]);
                    assert!((p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).z > 0.0);
                }
                // Skirt triangles face away from the chunk centre.
                let (lo, hi) = chunk.bounds(g);
                let centre = ((lo + hi) / 2.0).as_vec3();
                for t in m.indices[6 * (sx - 1) * (sy - 1)..].as_chunks::<3>().0 {
                    let p = |i: u32| Vec3::from_array(m.positions[i as usize]);
                    let n = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
                    let mid = (p(t[0]) + p(t[1]) + p(t[2])) / 3.0 - centre;
                    assert!(n.x * mid.x + n.y * mid.y > 0.0, "skirt faces inwards");
                }
            }
        }
    }

    #[test]
    fn water_quads_cover_the_wet_cells() {
        let w = testworlds::lake(100.0, 4.0, -1.0);
        let g = w.terrain();
        let color = water_color();
        let mut area = 0.0;
        for chunk in chunks(g, 16) {
            let m = water_chunk(&w, &chunk, color);
            for t in m.indices.as_chunks::<3>().0 {
                let p = |i: u32| Vec3::from_array(m.positions[i as usize]);
                let n = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
                assert!(n.z > 0.0 && n.x == 0.0 && n.y == 0.0);
                area += n.z / 2.0;
                assert_eq!(p(t[0]).z, -1.0);
            }
        }
        let (cw, ch) = g.cells();
        let wet = (0..ch).flat_map(|y| (0..cw).map(move |x| (x, y))).filter(|&(x, y)| g.cell_water(x, y).is_some());
        let expected = wet.count() as f64 * g.cell_size() * g.cell_size();
        assert!(expected > 100.0);
        assert!((f64::from(area) - expected).abs() < 1e-3 * expected, "{area} vs {expected}");
    }
}
