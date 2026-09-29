//! A map resident on the GPU: terrain chunks at every level of detail, water, road ribbons and
//! obstacles, each labelled with its semantic class; and vehicle rigs.

use autonomousim_core::math::Pose;
use autonomousim_scene::MeshData;
use autonomousim_scene::props::{self, PropDetail};
use autonomousim_scene::rig::{Placement, Rig};
use autonomousim_scene::terrain::{self, Chunk};
use autonomousim_scene::{mesh::srgb, roads};
use autonomousim_world::StaticWorld;
use glam::{DQuat, DVec3};

use crate::context::{GpuContext, RenderError};
use crate::renderer::{Draw, GpuMesh, View};
use crate::semantic::{SemanticClass, obstacle_class, terrain_class};

/// How a map is cut up and simplified with distance.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldOptions {
    /// Cells per chunk side.
    pub chunk_cells: usize,
    /// Terrain strides by distance from the camera to the chunk: `(max distance in m,
    /// stride)`, ascending; beyond the last, [`coarsest`](Self::coarsest).
    pub lod: Vec<(f64, usize)>,
    pub coarsest: usize,
    /// Obstacles are drawn at [`props`](Self::props) detail within this distance of the
    /// camera, at [`far_props`](Self::far_props) beyond it.
    pub detailed_props: f64,
    pub props: PropDetail,
    pub far_props: PropDetail,
}

impl Default for WorldOptions {
    fn default() -> Self {
        Self {
            chunk_cells: 32,
            lod: vec![(160.0, 1), (380.0, 2), (800.0, 4)],
            coarsest: 8,
            detailed_props: 250.0,
            // Finer than the viewer's: depth images show the facets.
            props: PropDetail { segments: 12, capsule_segments: 8, sphere_subdivisions: 2, min_size: 0.0 },
            far_props: PropDetail::default(),
        }
    }
}

impl WorldOptions {
    pub fn stride_for(&self, distance: f64) -> usize {
        self.lod.iter().find(|(d, _)| distance < *d).map_or(self.coarsest, |(_, s)| *s)
    }

    fn strides(&self) -> Vec<usize> {
        let mut s: Vec<usize> = self.lod.iter().map(|(_, s)| *s).chain([self.coarsest]).collect();
        s.sort_unstable();
        s.dedup();
        s
    }
}

struct GpuChunk {
    /// Bounds of the terrain, and of everything in the chunk.
    terrain_min: DVec3,
    terrain_max: DVec3,
    min: DVec3,
    max: DVec3,
    /// Meshes are relative to this point.
    anchor: DVec3,
    /// By stride, as [`WorldOptions::strides`].
    terrain: Vec<GpuMesh>,
    water: Option<GpuMesh>,
    props: Option<GpuMesh>,
    far_props: Option<GpuMesh>,
}

/// A monolithic map on the GPU.
pub struct GpuWorld {
    options: WorldOptions,
    strides: Vec<usize>,
    chunks: Vec<GpuChunk>,
    /// Road ribbons in world coordinates, one per chunk that has roads, with their bounds.
    roads: Vec<(GpuMesh, DVec3, DVec3)>,
    triangles: usize,
}

impl GpuWorld {
    /// Upload `world`. Tiled maps are not supported yet.
    pub fn new(ctx: &GpuContext, world: &StaticWorld, options: WorldOptions) -> Result<Self, RenderError> {
        let Some(grid) = world.terrain().grid() else {
            return Err(RenderError::Unsupported("tiled maps".into()));
        };
        let size = options.chunk_cells;
        let chunks = terrain::chunks(grid, size);
        let strides = options.strides();
        let colors = terrain::material_colors(world);
        let classes: Vec<u8> =
            (0..=u8::MAX).map(|m| terrain_class(autonomousim_core::material::MaterialId(m)).id()).collect();
        let water_color = srgb([46, 98, 132]);
        let mut triangles = 0;

        // Obstacles by chunk, each vertex with its obstacle's class.
        let materials = world.materials();
        let mut near = vec![(MeshData::new(), Vec::new()); chunks.len()];
        let mut far = near.clone();
        for (i, o) in world.obstacle_set().obstacles().iter().enumerate() {
            let k = props::chunk_index(grid, size, o.pose.pos);
            let anchor = chunk_anchor(grid, &chunks[k]);
            let class = obstacle_class(o).id();
            for (out, detail) in [(&mut near[k], options.props), (&mut far[k], options.far_props)] {
                if let Some(m) = props::shown_obstacle(materials, i as u64, o, detail) {
                    out.1.resize(out.1.len() + m.vertex_count(), class);
                    out.0.append_transformed(&m, o.pose.rot, o.pose.pos - anchor);
                }
            }
        }

        let mut gpu_chunks = Vec::with_capacity(chunks.len());
        for ((c, near), far) in chunks.iter().zip(near).zip(far) {
            let anchor = chunk_anchor(grid, c);
            let (terrain_min, terrain_max) = c.bounds(grid);
            let (mut min, mut max) = (terrain_min, terrain_max);
            let mut grow = |m: &MeshData| {
                if let Some((lo, hi)) = m.bounds() {
                    min = min.min(lo.as_dvec3() + anchor);
                    max = max.max(hi.as_dvec3() + anchor);
                }
            };
            let terrain: Vec<GpuMesh> = strides
                .iter()
                .map(|&s| {
                    let m = terrain::terrain_mesh(grid, &colors, c, s, skirt(grid.cell_size(), s), anchor);
                    let ids: Vec<u8> =
                        terrain::terrain_vertex_materials(grid, c, s).iter().map(|m| classes[m.0 as usize]).collect();
                    if s == strides[0] {
                        triangles += m.triangle_count();
                    }
                    GpuMesh::with_classes(ctx, &m, &ids)
                })
                .collect();
            let water = terrain::water_mesh(grid, c, water_color, anchor);
            grow(&water);
            grow(&near.0);
            triangles += water.triangle_count() + near.0.triangle_count();
            let upload = |m: &MeshData, ids: Option<&[u8]>, class: SemanticClass| {
                (!m.is_empty()).then(|| match ids {
                    Some(ids) => GpuMesh::with_classes(ctx, m, ids),
                    None => GpuMesh::new(ctx, m, class),
                })
            };
            gpu_chunks.push(GpuChunk {
                terrain_min,
                terrain_max,
                min,
                max,
                anchor,
                terrain,
                water: upload(&water, None, SemanticClass::Water),
                props: upload(&near.0, Some(&near.1), SemanticClass::Building),
                far_props: upload(&far.0, Some(&far.1), SemanticClass::Building),
            });
        }
        let roads = roads::roads_by_chunk(world, &chunks, size)
            .into_iter()
            .filter(|m| !m.is_empty())
            .map(|m| {
                triangles += m.triangle_count();
                let (lo, hi) = m.bounds().expect("a road mesh has vertices");
                (GpuMesh::new(ctx, &m, SemanticClass::Road), lo.as_dvec3(), hi.as_dvec3())
            })
            .collect();
        Ok(Self { options, strides, chunks: gpu_chunks, roads, triangles })
    }

    /// Triangles at full detail (terrain at the finest stride, water, obstacles, roads).
    pub fn triangle_count(&self) -> usize {
        self.triangles
    }

    pub fn options(&self) -> &WorldOptions {
        &self.options
    }

    /// Append what `view` may see: chunks in the frustum, terrain at the stride for its
    /// distance, obstacles in full or simplified detail.
    pub fn draws<'a>(&'a self, view: &View, out: &mut Vec<Draw<'a>>) {
        let eye = view.pose.position;
        for c in &self.chunks {
            if !view.may_see(c.min, c.max) {
                continue;
            }
            let at = |mesh| Draw::new(mesh, c.anchor, DQuat::IDENTITY);
            let stride = self.options.stride_for(box_distance(eye, c.terrain_min, c.terrain_max));
            let level = self.strides.iter().position(|&s| s == stride).expect("strides include every LOD stride");
            out.push(at(&c.terrain[level]));
            if let Some(w) = &c.water {
                out.push(at(w));
            }
            let props =
                if box_distance(eye, c.min, c.max) < self.options.detailed_props { &c.props } else { &c.far_props };
            if let Some(p) = props {
                out.push(at(p));
            }
        }
        for (m, lo, hi) in &self.roads {
            if view.may_see(*lo, *hi) {
                out.push(Draw::world(m));
            }
        }
    }
}

/// South-west corner of a chunk at height 0: its meshes are relative to it.
fn chunk_anchor(grid: &autonomousim_world::HeightGrid, c: &Chunk) -> DVec3 {
    let o = grid.origin();
    let cell = grid.cell_size();
    DVec3::new(o.x + c.x0 as f64 * cell, o.y + c.y0 as f64 * cell, 0.0)
}

/// Depth of the skirts of chunks with `cell`-metre cells sampled at `stride`.
fn skirt(cell: f64, stride: usize) -> f32 {
    (1.0 + 2.0 * stride as f64 * cell) as f32
}

/// Distance from `p` to the box `[min, max]`.
fn box_distance(p: DVec3, min: DVec3, max: DVec3) -> f64 {
    (p.clamp(min, max) - p).length()
}

/// A vehicle's rig on the GPU: one mesh per part.
pub struct GpuRig {
    meshes: Vec<GpuMesh>,
}

impl GpuRig {
    pub fn new(ctx: &GpuContext, rig: &Rig) -> Self {
        Self { meshes: rig.meshes.iter().map(|m| GpuMesh::new(ctx, m, SemanticClass::Vehicle)).collect() }
    }

    /// Append the parts placed at `placements` (from [`Rig::place`]) on a vehicle at `pose`,
    /// all labelled `class`.
    pub fn draws<'a>(&'a self, pose: Pose, placements: &[Placement], class: SemanticClass, out: &mut Vec<Draw<'a>>) {
        for (mesh, p) in self.meshes.iter().zip(placements) {
            let world = pose * p.pose;
            out.push(Draw::new(mesh, world.pos, world.rot).with_scale(p.scale).with_class(class));
        }
    }
}
