//! The map: terrain chunks with distance-based level of detail, water, road ribbons, merged
//! obstacle meshes (coarse ones far away), sun, sky and fog. When the simulation moves to
//! another map (a replayed episode on another map of the pool, a regenerated map) the map is
//! rebuilt.

use crate::convert;
use crate::sim::Sim;
use crate::{CameraRig, Quality};
use autonomousim_scene::MeshData;
use autonomousim_scene::props::{self, PropDetail};
use autonomousim_scene::roads;
use autonomousim_scene::terrain::{self, Chunk};
use autonomousim_world::StaticWorld;
use bevy::camera::primitives::Aabb;
use bevy::light::CascadeShadowConfigBuilder;
use bevy::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

/// Cells per chunk side.
pub const CHUNK_CELLS: usize = 64;

/// Terrain strides by distance: `(max distance in m, stride)`; beyond the last, the coarsest.
const LOD_BANDS: [(f32, usize); 3] = [(160.0, 1), (380.0, 2), (800.0, 4)];
const COARSEST: usize = 8;
/// Chunk meshes rebuilt per frame at most.
const REBUILDS_PER_FRAME: usize = 24;

#[derive(Resource)]
pub struct MapView {
    pub world: Arc<StaticWorld>,
    pub chunks: Vec<Chunk>,
    /// Distance beyond which nothing is drawn (m); fog ends there.
    pub view_distance: f32,
    /// Distance beyond which obstacles are drawn coarse (m).
    pub detail_distance: f32,
    materials: Option<MapMaterials>,
}

impl MapView {
    pub fn new(world: Arc<StaticWorld>, quality: Quality) -> Self {
        let chunks = terrain::chunks(world.terrain(), CHUNK_CELLS);
        Self {
            world,
            chunks,
            view_distance: quality.view_distance(),
            detail_distance: quality.prop_detail_distance(),
            materials: None,
        }
    }
}

#[derive(Clone)]
struct MapMaterials {
    terrain: Handle<StandardMaterial>,
    water: Handle<StandardMaterial>,
    props: Handle<StandardMaterial>,
    roads: Handle<StandardMaterial>,
}

/// Everything that belongs to the current map.
#[derive(Component)]
pub struct MapEntity;

#[derive(Component)]
pub struct TerrainChunk {
    index: usize,
    stride: usize,
    /// Bounds in Bevy coordinates.
    min: Vec3,
    max: Vec3,
}

/// Props and water of a chunk; hidden beyond the view distance.
#[derive(Component)]
pub struct ChunkPart {
    min: Vec3,
    max: Vec3,
}

/// Obstacles of a chunk at one level of detail: shown nearer than the detail distance, or
/// beyond it.
#[derive(Component)]
pub struct PropLod {
    near: bool,
}

fn stride_for(distance: f32) -> usize {
    LOD_BANDS.iter().find(|(d, _)| distance < *d).map_or(COARSEST, |(_, s)| *s)
}

fn skirt(world: &StaticWorld, stride: usize) -> f32 {
    (1.0 + 2.0 * stride as f64 * world.terrain().cell_size()) as f32
}

/// Distance from `p` to the box `[min, max]`.
fn box_distance(p: Vec3, min: Vec3, max: Vec3) -> f32 {
    (p.clamp(min, max) - p).length()
}

/// Bounds of a mesh in Bevy coordinates.
fn bevy_bounds(m: &MeshData) -> Option<(Vec3, Vec3)> {
    let (lo, hi) = m.bounds()?;
    let (a, b) = (convert::vec(lo.as_dvec3()), convert::vec(hi.as_dvec3()));
    Some((a.min(b), a.max(b)))
}

fn aabb(min: Vec3, max: Vec3) -> Aabb {
    Aabb::from_min_max(min, max)
}

/// Sun and ambient light (the same for every map).
pub fn spawn_lights(mut commands: Commands, view: Res<MapView>, quality: Res<Quality>) {
    let (min, max) = view.world.extent();
    let centre = convert::vec(((min + max) / 2.0).extend(view.world.terrain().height_range().1));
    // Sun from the south-west, 40° high.
    let (elevation, azimuth) = (40f32.to_radians(), 225f32.to_radians());
    let to_sun = glam::DVec3::new(
        f64::from(elevation.cos() * azimuth.cos()),
        f64::from(elevation.cos() * azimuth.sin()),
        f64::from(elevation.sin()),
    );
    let shadows = quality.shadows();
    let (num_cascades, maximum_distance) = quality.shadow_cascades();
    commands.spawn((
        DirectionalLight { illuminance: 9_000.0, shadow_maps_enabled: shadows, ..default() },
        Transform::from_translation(centre).looking_to(-convert::vec(to_sun), Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades,
            minimum_distance: 0.1,
            first_cascade_far_bound: 30.0,
            maximum_distance,
            overlap_proportion: 0.2,
        }
        .build(),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.75, 0.82, 1.0),
        brightness: 900.0,
        ..default()
    });
}

pub fn spawn_map(
    mut commands: Commands,
    mut view: ResMut<MapView>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let m = MapMaterials {
        terrain: materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.95,
            reflectance: 0.15,
            ..default()
        }),
        water: materials.add(StandardMaterial {
            base_color: Color::WHITE,
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.08,
            reflectance: 0.6,
            ..default()
        }),
        props: materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.9,
            reflectance: 0.2,
            ..default()
        }),
        // Drawn in front of the terrain they lie a few centimetres above.
        roads: materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.85,
            reflectance: 0.1,
            depth_bias: 50.0,
            ..default()
        }),
    };
    view.materials = Some(m);
    build_map(&mut commands, &view, &mut meshes);
}

/// Rebuild the map when the simulation has moved to another one.
pub fn sync_map(
    mut commands: Commands,
    sim: Res<Sim>,
    mut view: ResMut<MapView>,
    mut meshes: ResMut<Assets<Mesh>>,
    old: Query<Entity, With<MapEntity>>,
) {
    let map = sim.world.map();
    if Arc::ptr_eq(&view.world, map) {
        return;
    }
    for e in &old {
        commands.entity(e).despawn();
    }
    view.world = map.clone();
    view.chunks = terrain::chunks(map.terrain(), CHUNK_CELLS);
    build_map(&mut commands, &view, &mut meshes);
}

/// Terrain (at the coarsest level; [`update_lod`] refines it around the camera), water and
/// obstacles at both levels of detail, one entity each per chunk.
fn build_map(commands: &mut Commands, view: &MapView, meshes: &mut Assets<Mesh>) {
    let world = &*view.world;
    let materials = view.materials.clone().expect("map materials");
    let start = std::time::Instant::now();
    let built: Vec<(MeshData, MeshData)> = view
        .chunks
        .par_iter()
        .map(|c| {
            let t = terrain::terrain_chunk(world, c, COARSEST, skirt(world, COARSEST));
            (t, terrain::water_chunk(world, c, terrain::water_color()))
        })
        .collect();
    let ((mut near, mut far), road_meshes) = rayon::join(
        || {
            rayon::join(
                || props::props_by_chunk(world, &view.chunks, CHUNK_CELLS, PropDetail::default()),
                || props::props_by_chunk(world, &view.chunks, CHUNK_CELLS, PropDetail::far()),
            )
        },
        || roads::roads_by_chunk(world, &view.chunks, CHUNK_CELLS),
    );
    let mut road_triangles = 0;
    // Roads are drawn where obstacles are drawn in detail.
    for r in road_meshes.into_iter().filter(|r| !r.is_empty()) {
        road_triangles += r.triangle_count();
        let (rmin, rmax) = bevy_bounds(&r).unwrap();
        commands.spawn((
            Mesh3d(meshes.add(convert::mesh(&r))),
            MeshMaterial3d(materials.roads.clone()),
            aabb(rmin, rmax),
            ChunkPart { min: rmin, max: rmax },
            PropLod { near: true },
            bevy::light::NotShadowCaster,
            MapEntity,
        ));
    }

    let (mut triangles, mut prop_triangles, mut far_triangles) = (0, 0, 0);
    for (i, (c, (t, w))) in view.chunks.iter().zip(built).enumerate() {
        let (lo, hi) = c.bounds(world.terrain());
        let (a, b) = (convert::vec(lo), convert::vec(hi));
        let (cmin, cmax) = (a.min(b), a.max(b));
        let (mmin, mmax) = bevy_bounds(&t).unwrap_or((cmin, cmax));
        triangles += t.triangle_count();
        commands.spawn((
            Mesh3d(meshes.add(convert::mesh(&t))),
            MeshMaterial3d(materials.terrain.clone()),
            aabb(mmin, mmax),
            TerrainChunk { index: i, stride: COARSEST, min: cmin, max: cmax },
            MapEntity,
        ));
        if !w.is_empty() {
            let (wmin, wmax) = bevy_bounds(&w).unwrap();
            commands.spawn((
                Mesh3d(meshes.add(convert::mesh(&w))),
                MeshMaterial3d(materials.water.clone()),
                aabb(wmin, wmax),
                ChunkPart { min: wmin, max: wmax },
                bevy::light::NotShadowCaster,
                MapEntity,
            ));
        }
        for (lod, p) in [(true, std::mem::take(&mut near[i])), (false, std::mem::take(&mut far[i]))] {
            if p.is_empty() {
                continue;
            }
            if lod {
                prop_triangles += p.triangle_count();
            } else {
                far_triangles += p.triangle_count();
            }
            let (pmin, pmax) = bevy_bounds(&p).unwrap();
            commands.spawn((
                Mesh3d(meshes.add(convert::mesh(&p))),
                MeshMaterial3d(materials.props.clone()),
                aabb(pmin, pmax),
                ChunkPart { min: pmin, max: pmax },
                PropLod { near: lod },
                if lod { Visibility::Inherited } else { Visibility::Hidden },
                MapEntity,
            ));
        }
    }
    info!(
        "map {}: {} chunks, {} terrain triangles at the coarsest level, {} road triangles, {} obstacle triangles ({} far) ({:.2} s)",
        world.meta.name,
        view.chunks.len(),
        triangles,
        road_triangles,
        prop_triangles,
        far_triangles,
        start.elapsed().as_secs_f64()
    );
}

/// Refine or coarsen terrain chunks by their distance to the camera (nearest first, a
/// bounded number per frame) and hide everything beyond the view distance.
pub fn update_lod(
    view: Res<MapView>,
    camera: Query<&GlobalTransform, With<CameraRig>>,
    mut chunks: Query<(Entity, &mut TerrainChunk, &mut Mesh3d, &mut Aabb, &mut Visibility)>,
    mut parts: Query<(&ChunkPart, Option<&PropLod>, &mut Visibility), Without<TerrainChunk>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Ok(cam) = camera.single() else { return };
    let eye = cam.translation();
    let world = &*view.world;
    let far = view.view_distance;

    // (distance, entity, chunk, stride)
    let mut todo: Vec<(f32, Entity, usize, usize)> = Vec::new();
    for (entity, chunk, _, _, mut vis) in &mut chunks {
        let d = box_distance(eye, chunk.min, chunk.max);
        vis.set_if_neq(if d > far { Visibility::Hidden } else { Visibility::Inherited });
        // Hysteresis: coarsen only once clearly past the band.
        let fine = stride_for(d);
        let coarse = stride_for(d / 1.15);
        let target = if fine < chunk.stride {
            fine
        } else if coarse > chunk.stride {
            coarse
        } else {
            chunk.stride
        };
        if target != chunk.stride && d <= far {
            todo.push((d, entity, chunk.index, target));
        }
    }
    todo.sort_by(|a, b| a.0.total_cmp(&b.0));
    todo.truncate(REBUILDS_PER_FRAME);
    let built: Vec<MeshData> = todo
        .par_iter()
        .map(|&(_, _, i, s)| terrain::terrain_chunk(world, &view.chunks[i], s, skirt(world, s)))
        .collect();
    for (&(_, entity, _, s), m) in todo.iter().zip(built) {
        let Ok((_, mut chunk, mut mesh, mut bounds, _)) = chunks.get_mut(entity) else { continue };
        if let Some((lo, hi)) = bevy_bounds(&m) {
            *bounds = aabb(lo, hi);
        }
        mesh.0 = meshes.add(convert::mesh(&m));
        chunk.stride = s;
    }

    for (part, lod, mut vis) in &mut parts {
        let d = box_distance(eye, part.min, part.max);
        let shown = d <= far && lod.is_none_or(|l| l.near == (d < view.detail_distance));
        vis.set_if_neq(if shown { Visibility::Inherited } else { Visibility::Hidden });
    }
}
