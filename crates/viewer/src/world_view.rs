//! The map: terrain chunks with distance-based level of detail, water, road ribbons, merged
//! obstacle meshes (coarse ones far away), sun, sky and fog. When the simulation moves to
//! another map (a replayed episode on another map of the pool, a regenerated map) the map is
//! rebuilt.
//!
//! Tiled (large) maps are streamed: the coarse layer is drawn everywhere as one chunk per
//! tile, and the detail tiles near the camera are generated and meshed on worker threads and
//! replace their coarse chunks once ready. The view distance grows with the camera's height.
//!
//! Every map entity has an [`Anchor`] (ENU) and is placed relative to the
//! [`RenderOrigin`], which follows the camera in 1 km steps.

use crate::convert::{self, RenderOrigin};
use crate::sim::Sim;
use crate::{CameraRig, Quality};
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::terrain::Terrain;
use autonomousim_scene::MeshData;
use autonomousim_scene::mesh::srgb;
use autonomousim_scene::props::{self, PropDetail};
use autonomousim_scene::roads;
use autonomousim_scene::terrain::{self, Chunk};
use autonomousim_world::{HeightGrid, StaticWorld, Tile, TiledMap};
use bevy::camera::primitives::Aabb;
use bevy::light::CascadeShadowConfigBuilder;
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::prelude::*;
use glam::{DVec2, DVec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

/// Cells per chunk side.
pub const CHUNK_CELLS: usize = 64;

/// Terrain strides by distance: `(max distance in m, stride)`; beyond the last, the coarsest.
const LOD_BANDS: [(f64, usize); 3] = [(160.0, 1), (380.0, 2), (800.0, 4)];
/// The same for the coarse layer of tiled maps (8 m cells).
const COARSE_BANDS: [(f64, usize); 3] = [(1500.0, 1), (3000.0, 2), (6000.0, 4)];
const COARSEST: usize = 8;
/// Chunk meshes rebuilt per frame at most.
const REBUILDS_PER_FRAME: usize = 24;
/// The render origin moves when the camera is farther than this from it (m, per axis).
pub const RECENTER: f64 = 1000.0;
/// View distance never exceeds this (m).
const MAX_VIEW: f64 = 12_000.0;
/// Detail tiles generated or meshed at once.
const MAX_PENDING: usize = 4;
/// Detail tiles are dropped beyond this multiple of the detail radius.
const UNLOAD: f64 = 1.25;
/// Colour (sRGB) of forest seen from afar, between the conifer and broadleaf crowns.
const CANOPY: [u8; 3] = [58, 98, 52];

#[derive(Resource)]
pub struct MapView {
    pub world: Arc<StaticWorld>,
    /// Chunks of a monolithic map (none for tiled maps).
    pub chunks: Vec<Chunk>,
    /// Distance beyond which nothing is drawn at ground level (m); it grows with height.
    pub view_distance: f32,
    /// The current view distance (m); fog ends there.
    pub far: f32,
    /// Distance beyond which obstacles are drawn coarse (m).
    pub detail_distance: f32,
    materials: Option<MapMaterials>,
    tiled: Option<Streamer>,
    quality: Quality,
}

impl MapView {
    pub fn new(world: Arc<StaticWorld>, quality: Quality) -> Self {
        let tiled = world.terrain().tiled().map(|t| Streamer::new(t.clone(), &world, quality));
        let chunks = match world.terrain().grid() {
            Some(g) => terrain::chunks(g, CHUNK_CELLS),
            None => Vec::new(),
        };
        let view_distance = if tiled.is_some() { quality.far_view_distance() } else { quality.view_distance() };
        Self {
            world,
            chunks,
            view_distance,
            far: view_distance,
            detail_distance: quality.prop_detail_distance(),
            materials: None,
            tiled,
            quality,
        }
    }

    /// Ground or water surface height at `(x, y)`, without generating tiles: on tiled maps
    /// from the detail tiles on screen, else from the coarse layer.
    pub fn surface_height(&self, x: f64, y: f64) -> f64 {
        let Some(s) = &self.tiled else { return self.world.surface_height(x, y) };
        let key = s.map.layout().tile_at(x, y);
        let surface = |t: &dyn Terrain| {
            let h = t.height(x, y);
            t.water_level(x, y).map_or(h, |w| w.max(h))
        };
        match s.loaded.get(&key) {
            Some(d) => surface(&d.tile.grid),
            None => surface(s.map.coarse()),
        }
    }

    /// Number of detail tiles on screen and being built (tiled maps).
    pub fn streamed_tiles(&self) -> Option<(usize, usize)> {
        self.tiled.as_ref().map(|s| (s.loaded.len(), s.pending.len()))
    }
}

/// Streaming state of a tiled map.
struct Streamer {
    map: Arc<TiledMap>,
    colors: Arc<Vec<[f32; 4]>>,
    /// Colours of the coarse layer: forest floor there stands for the forest on it.
    coarse_colors: Vec<[f32; 4]>,
    materials: Arc<MaterialTable>,
    /// One chunk of the coarse grid per tile (index `ty · tiles.0 + tx`).
    coarse_chunks: Vec<Chunk>,
    /// Detail tiles are shown within this distance of the camera (m).
    radius: f64,
    loaded: HashMap<(u32, u32), DetailTile>,
    pending: HashSet<(u32, u32)>,
    pool: Arc<rayon::ThreadPool>,
    sender: Sender<BuiltTile>,
    receiver: Mutex<Receiver<BuiltTile>>,
}

impl Streamer {
    fn new(map: Arc<TiledMap>, world: &StaticWorld, quality: Quality) -> Self {
        let layout = *map.layout();
        let coarse = map.coarse();
        let per_tile = (layout.tile_size / coarse.cell_size()).round() as usize;
        let mut coarse_chunks = Vec::with_capacity((layout.tiles.0 * layout.tiles.1) as usize);
        for ty in 0..layout.tiles.1 {
            for tx in 0..layout.tiles.0 {
                let (lo, _) = layout.core(tx, ty);
                let q = (lo - coarse.origin()) / coarse.cell_size();
                coarse_chunks.push(Chunk {
                    cx: tx as usize,
                    cy: ty as usize,
                    x0: q.x.round() as usize,
                    y0: q.y.round() as usize,
                    nx: per_tile,
                    ny: per_tile,
                });
            }
        }
        let (sender, receiver) = std::sync::mpsc::channel();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .thread_name(|i| format!("tiles-{i}"))
            .build()
            .expect("tile worker threads");
        let colors = terrain::material_colors(world);
        let mut coarse_colors = colors.clone();
        if let Some(c) = coarse_colors.get_mut(MaterialId::FOREST_FLOOR.0 as usize) {
            *c = srgb(CANOPY);
        }
        Self {
            map,
            colors: Arc::new(colors),
            coarse_colors,
            materials: Arc::new(world.materials().clone()),
            coarse_chunks,
            radius: quality.tile_radius(),
            loaded: HashMap::new(),
            pending: HashSet::new(),
            pool: Arc::new(pool),
            sender,
            receiver: Mutex::new(receiver),
        }
    }

    /// Horizontal distance from `p` to the core of tile `key`.
    fn tile_distance(&self, key: (u32, u32), p: DVec2) -> f64 {
        let (lo, hi) = self.map.layout().core(key.0, key.1);
        (p.clamp(lo, hi) - p).length()
    }
}

/// A detail tile on screen.
struct DetailTile {
    tile: Arc<Tile>,
    chunks: Vec<Chunk>,
    anchor: DVec3,
}

/// Meshes of a detail tile, built on a worker thread.
struct BuiltTile {
    key: (u32, u32),
    tile: Arc<Tile>,
    anchor: DVec3,
    chunks: Vec<BuiltChunk>,
}

struct BuiltChunk {
    chunk: Chunk,
    /// Terrain bounds (ENU).
    min: DVec3,
    max: DVec3,
    stride: usize,
    terrain: MeshData,
    water: MeshData,
    near: MeshData,
    far: MeshData,
}

#[derive(Clone)]
struct MapMaterials {
    terrain: Handle<StandardMaterial>,
    water: Handle<StandardMaterial>,
    props: Handle<StandardMaterial>,
    roads: Handle<StandardMaterial>,
}

/// Everything that belongs to the current map.
#[derive(Component, Clone, Copy)]
pub struct MapEntity;

/// Where an entity's mesh origin is (ENU); its transform places it relative to the
/// [`RenderOrigin`].
#[derive(Component, Clone, Copy)]
pub struct Anchor(pub DVec3);

/// Which grid a terrain chunk samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layer {
    /// The grid of a monolithic map.
    Grid,
    /// The coarse layer of a tiled map.
    Coarse,
    /// A detail tile.
    Detail((u32, u32)),
}

#[derive(Component)]
pub struct TerrainChunk {
    layer: Layer,
    index: usize,
    stride: usize,
    /// Bounds (ENU).
    min: DVec3,
    max: DVec3,
}

/// Props and water of a chunk; hidden beyond the view distance.
#[derive(Component)]
pub struct ChunkPart {
    min: DVec3,
    max: DVec3,
}

/// Obstacles of a chunk at one level of detail: shown nearer than the detail distance, or
/// beyond it.
#[derive(Component)]
pub struct PropLod {
    near: bool,
}

/// Part of the coarse layer standing for a tile; hidden while the tile is shown in detail.
#[derive(Component)]
pub struct CoarseOf(pub (u32, u32));

/// Part of a detail tile.
#[derive(Component, Clone, Copy)]
pub struct DetailOf(pub (u32, u32));

fn stride_for(layer: Layer, distance: f64) -> usize {
    let bands = if layer == Layer::Coarse { &COARSE_BANDS } else { &LOD_BANDS };
    bands.iter().find(|(d, _)| distance < *d).map_or(COARSEST, |(_, s)| *s)
}

/// Depth of the skirts of chunks with `cell`-metre cells sampled at `stride`.
fn skirt(cell: f64, stride: usize) -> f32 {
    (1.0 + 2.0 * stride as f64 * cell) as f32
}

/// Distance from `p` to the box `[min, max]`.
fn box_distance(p: DVec3, min: DVec3, max: DVec3) -> f64 {
    (p.clamp(min, max) - p).length()
}

/// Bounds of a mesh in Bevy coordinates (relative to its anchor).
fn bevy_bounds(m: &MeshData) -> Option<(Vec3, Vec3)> {
    let (lo, hi) = m.bounds()?;
    let (a, b) = (convert::vec(lo.as_dvec3()), convert::vec(hi.as_dvec3()));
    Some((a.min(b), a.max(b)))
}

fn aabb(min: Vec3, max: Vec3) -> Aabb {
    Aabb::from_min_max(min, max)
}

/// Sun and ambient light (the same for every map).
pub fn spawn_lights(mut commands: Commands, quality: Res<Quality>) {
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
        Transform::default().looking_to(-convert::vec(to_sun), Vec3::Y),
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
    origin: Res<RenderOrigin>,
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
    build_map(&mut commands, &view, *origin, &mut meshes);
}

/// Rebuild the map when the simulation has moved to another one.
pub fn sync_map(
    mut commands: Commands,
    sim: Res<Sim>,
    mut view: ResMut<MapView>,
    origin: Res<RenderOrigin>,
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
    let materials = view.materials.take();
    *view = MapView::new(map.clone(), view.quality);
    view.materials = materials;
    build_map(&mut commands, &view, *origin, &mut meshes);
}

/// Move the render origin to the camera once it is more than [`RECENTER`] away, and every
/// map entity with it (vehicles, the camera and gizmos are placed from the origin each
/// frame).
pub fn recenter(
    mut origin: ResMut<RenderOrigin>,
    camera: Query<&CameraRig>,
    mut anchored: Query<(&Anchor, &mut Transform)>,
) {
    let Ok(rig) = camera.single() else { return };
    let d = rig.eye - origin.0;
    if d.x.abs().max(d.y.abs()) <= RECENTER {
        return;
    }
    let snap = |v: f64| (v / RECENTER).round() * RECENTER;
    origin.0 = DVec3::new(snap(rig.eye.x), snap(rig.eye.y), 0.0);
    for (a, mut t) in &mut anchored {
        t.translation = origin.pos(a.0);
    }
}

/// Terrain (at the coarsest level; [`update_lod`] refines it around the camera), water and
/// obstacles at both levels of detail, one entity each per chunk. Tiled maps start with the
/// coarse layer; [`stream_tiles`] adds the detail tiles.
fn build_map(commands: &mut Commands, view: &MapView, origin: RenderOrigin, meshes: &mut Assets<Mesh>) {
    if let Some(s) = &view.tiled {
        build_coarse(commands, view, s, origin, meshes);
        return;
    }
    let world = &*view.world;
    let grid = world.grid();
    let materials = view.materials.clone().expect("map materials");
    let start = std::time::Instant::now();
    let at_origin = (Anchor(DVec3::ZERO), Transform::from_translation(origin.pos(DVec3::ZERO)));
    let built: Vec<(MeshData, MeshData)> = view
        .chunks
        .par_iter()
        .map(|c| {
            let t = terrain::terrain_chunk(world, c, COARSEST, skirt(grid.cell_size(), COARSEST));
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
    let enu_bounds = |m: &MeshData| m.bounds().map(|(lo, hi)| (lo.as_dvec3(), hi.as_dvec3()));
    let mut road_triangles = 0;
    // Roads are drawn where obstacles are drawn in detail.
    for r in road_meshes.into_iter().filter(|r| !r.is_empty()).map(|r| r.mesh) {
        road_triangles += r.triangle_count();
        let (rmin, rmax) = bevy_bounds(&r).unwrap();
        let (min, max) = enu_bounds(&r).unwrap();
        commands.spawn((
            Mesh3d(meshes.add(convert::mesh(&r))),
            MeshMaterial3d(materials.roads.clone()),
            aabb(rmin, rmax),
            ChunkPart { min, max },
            PropLod { near: true },
            bevy::light::NotShadowCaster,
            MapEntity,
            at_origin,
        ));
    }

    let (mut triangles, mut prop_triangles, mut far_triangles) = (0, 0, 0);
    for (i, (c, (t, w))) in view.chunks.iter().zip(built).enumerate() {
        let (min, max) = c.bounds(grid);
        let (mmin, mmax) = bevy_bounds(&t).unwrap();
        triangles += t.triangle_count();
        commands.spawn((
            Mesh3d(meshes.add(convert::mesh(&t))),
            MeshMaterial3d(materials.terrain.clone()),
            aabb(mmin, mmax),
            TerrainChunk { layer: Layer::Grid, index: i, stride: COARSEST, min, max },
            MapEntity,
            at_origin,
        ));
        if !w.is_empty() {
            let (wmin, wmax) = bevy_bounds(&w).unwrap();
            let (min, max) = enu_bounds(&w).unwrap();
            commands.spawn((
                Mesh3d(meshes.add(convert::mesh(&w))),
                MeshMaterial3d(materials.water.clone()),
                aabb(wmin, wmax),
                ChunkPart { min, max },
                bevy::light::NotShadowCaster,
                MapEntity,
                at_origin,
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
            let (min, max) = enu_bounds(&p).unwrap();
            commands.spawn((
                Mesh3d(meshes.add(convert::mesh(&p))),
                MeshMaterial3d(materials.props.clone()),
                aabb(pmin, pmax),
                ChunkPart { min, max },
                PropLod { near: lod },
                if lod { Visibility::Inherited } else { Visibility::Hidden },
                MapEntity,
                at_origin,
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

/// The coarse layer of a tiled map: terrain and water, one chunk per tile.
fn build_coarse(
    commands: &mut Commands,
    view: &MapView,
    s: &Streamer,
    origin: RenderOrigin,
    meshes: &mut Assets<Mesh>,
) {
    let materials = view.materials.clone().expect("map materials");
    let start = std::time::Instant::now();
    let coarse = s.map.coarse();
    let tiles = s.map.layout().tiles;
    let built: Vec<(DVec3, (DVec3, DVec3), MeshData, MeshData)> = s
        .coarse_chunks
        .par_iter()
        .map(|c| {
            let (min, max) = c.bounds(coarse);
            let anchor = DVec3::new(min.x, min.y, 0.0);
            let t = terrain::terrain_mesh(
                coarse,
                &s.coarse_colors,
                c,
                COARSEST,
                skirt(coarse.cell_size(), COARSEST),
                anchor,
            );
            (anchor, (min, max), t, terrain::water_mesh(coarse, c, terrain::water_color(), anchor))
        })
        .collect();
    for (i, (anchor, (min, max), t, w)) in built.into_iter().enumerate() {
        let key = ((i % tiles.0 as usize) as u32, (i / tiles.0 as usize) as u32);
        let place = (Anchor(anchor), Transform::from_translation(origin.pos(anchor)));
        let (mmin, mmax) = bevy_bounds(&t).unwrap();
        commands.spawn((
            Mesh3d(meshes.add(convert::mesh(&t))),
            MeshMaterial3d(materials.terrain.clone()),
            aabb(mmin, mmax),
            TerrainChunk { layer: Layer::Coarse, index: i, stride: COARSEST, min, max },
            CoarseOf(key),
            MapEntity,
            place,
        ));
        if !w.is_empty() {
            let (wmin, wmax) = bevy_bounds(&w).unwrap();
            commands.spawn((
                Mesh3d(meshes.add(convert::mesh(&w))),
                MeshMaterial3d(materials.water.clone()),
                aabb(wmin, wmax),
                ChunkPart { min, max },
                CoarseOf(key),
                bevy::light::NotShadowCaster,
                MapEntity,
                place,
            ));
        }
    }
    info!(
        "map {}: tiled, {} × {} tiles, coarse layer meshed in {:.2} s",
        view.world.meta.name,
        tiles.0,
        tiles.1,
        start.elapsed().as_secs_f64()
    );
}

/// Meshes of detail tile `key` (on a worker thread); terrain at the stride for `distance`.
/// What a tile job needs: the map, material colours and table.
struct TileJob {
    map: Arc<TiledMap>,
    colors: Arc<Vec<[f32; 4]>>,
    materials: Arc<MaterialTable>,
}

fn build_tile(job: &TileJob, key: (u32, u32), eye: DVec3) -> BuiltTile {
    let TileJob { map, colors, materials } = job;
    let layout = map.layout();
    let tile = map.tile(key.0, key.1);
    let grid = &tile.grid;
    let cell = grid.cell_size();
    let (lo, _) = layout.core(key.0, key.1);
    let anchor = DVec3::new(lo.x, lo.y, 0.0);
    let ring = ((lo.x - grid.origin().x) / cell).round() as usize;
    let per_tile = (layout.tile_size / cell).round() as usize;
    let per_side = per_tile.div_ceil(CHUNK_CELLS);
    let chunks: Vec<Chunk> = (0..per_side * per_side)
        .map(|i| {
            let (cx, cy) = (i % per_side, i / per_side);
            let (x0, y0) = (cx * CHUNK_CELLS, cy * CHUNK_CELLS);
            Chunk {
                cx,
                cy,
                x0: ring + x0,
                y0: ring + y0,
                nx: CHUNK_CELLS.min(per_tile - x0),
                ny: CHUNK_CELLS.min(per_tile - y0),
            }
        })
        .collect();
    // Obstacles anchored in this tile (the others are drawn with their own tiles), by chunk.
    let chunk_of = |p: DVec3| {
        let q = ((p.truncate() - lo) / cell).max(DVec2::ZERO);
        let (x, y) = ((q.x as usize).min(per_tile - 1), (q.y as usize).min(per_tile - 1));
        (y / CHUNK_CELLS) * per_side + x / CHUNK_CELLS
    };
    let owned: Vec<(u64, &autonomousim_world::Obstacle, usize)> = tile
        .obstacles
        .obstacles()
        .iter()
        .zip(&tile.ids)
        .filter(|(_, id)| map.source().owner(**id) == key)
        .map(|(o, id)| (u64::from(*id), o, chunk_of(o.pose.pos)))
        .collect();
    let mut near = props::props_grouped(materials, owned.iter().copied(), chunks.len(), PropDetail::default(), anchor);
    let mut far = props::props_grouped(materials, owned.iter().copied(), chunks.len(), PropDetail::far(), anchor);
    let built = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let (min, max) = c.bounds(grid);
            let stride = stride_for(Layer::Detail(key), box_distance(eye, min, max));
            BuiltChunk {
                chunk: *c,
                min,
                max,
                stride,
                terrain: terrain::terrain_mesh(grid, colors, c, stride, skirt(cell, stride), anchor),
                water: terrain::water_mesh(grid, c, terrain::water_color(), anchor),
                near: std::mem::take(&mut near[i]),
                far: std::mem::take(&mut far[i]),
            }
        })
        .collect();
    BuiltTile { key, tile, anchor, chunks: built }
}

/// Tiled maps: spawn the detail tiles that are ready, request the nearest missing ones within
/// the detail radius, and drop those far behind.
pub fn stream_tiles(
    mut commands: Commands,
    mut view: ResMut<MapView>,
    origin: Res<RenderOrigin>,
    camera: Query<&CameraRig>,
    detail: Query<(Entity, &DetailOf)>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Ok(rig) = camera.single() else { return };
    let eye = rig.eye;
    let materials = view.materials.clone();
    let Some(s) = &mut view.tiled else { return };
    let Some(materials) = materials else { return };

    // Finished tiles.
    let ready: Vec<BuiltTile> = s.receiver.lock().expect("tile channel").try_iter().collect();
    for b in ready {
        s.pending.remove(&b.key);
        if s.tile_distance(b.key, eye.truncate()) > UNLOAD * s.radius || s.loaded.contains_key(&b.key) {
            continue;
        }
        let place = (Anchor(b.anchor), Transform::from_translation(origin.pos(b.anchor)));
        let mut chunks = Vec::with_capacity(b.chunks.len());
        for (i, c) in b.chunks.into_iter().enumerate() {
            chunks.push(c.chunk);
            let tag = (DetailOf(b.key), MapEntity, place);
            if let Some((lo, hi)) = bevy_bounds(&c.terrain) {
                commands.spawn((
                    Mesh3d(meshes.add(convert::mesh(&c.terrain))),
                    MeshMaterial3d(materials.terrain.clone()),
                    aabb(lo, hi),
                    TerrainChunk { layer: Layer::Detail(b.key), index: i, stride: c.stride, min: c.min, max: c.max },
                    tag,
                ));
            }
            // Water and obstacles; bounds (ENU) from the meshes, which are relative to the anchor.
            let parts = [
                (&c.water, &materials.water, None),
                (&c.near, &materials.props, Some(true)),
                (&c.far, &materials.props, Some(false)),
            ];
            for (m, material, lod) in parts {
                let (Some((lo, hi)), Some((mlo, mhi))) = (bevy_bounds(m), m.bounds()) else { continue };
                let (min, max) = (b.anchor + mlo.as_dvec3(), b.anchor + mhi.as_dvec3());
                let mut e = commands.spawn((
                    Mesh3d(meshes.add(convert::mesh(m))),
                    MeshMaterial3d(material.clone()),
                    aabb(lo, hi),
                    ChunkPart { min, max },
                    tag,
                ));
                match lod {
                    Some(near) => {
                        e.insert((PropLod { near }, if near { Visibility::Inherited } else { Visibility::Hidden }));
                    }
                    None => {
                        e.insert(bevy::light::NotShadowCaster);
                    }
                }
            }
        }
        s.loaded.insert(b.key, DetailTile { tile: b.tile, chunks, anchor: b.anchor });
    }

    // Tiles far behind.
    let far: Vec<(u32, u32)> =
        s.loaded.keys().copied().filter(|&k| s.tile_distance(k, eye.truncate()) > UNLOAD * s.radius).collect();
    if !far.is_empty() {
        for k in &far {
            s.loaded.remove(k);
        }
        let far: HashSet<(u32, u32)> = far.into_iter().collect();
        for (e, d) in &detail {
            if far.contains(&d.0) {
                commands.entity(e).despawn();
            }
        }
    }

    // The nearest missing tiles.
    if s.pending.len() >= MAX_PENDING {
        return;
    }
    let layout = *s.map.layout();
    let r = s.radius;
    let (a, b) = (layout.tile_at(eye.x - r, eye.y - r), layout.tile_at(eye.x + r, eye.y + r));
    let mut wanted: Vec<(f64, (u32, u32))> = (a.1..=b.1)
        .flat_map(|ty| (a.0..=b.0).map(move |tx| (tx, ty)))
        .filter(|k| !s.loaded.contains_key(k) && !s.pending.contains(k))
        .map(|k| (s.tile_distance(k, eye.truncate()), k))
        .filter(|(d, _)| *d <= r)
        .collect();
    wanted.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
    for (_, key) in wanted.into_iter().take(MAX_PENDING - s.pending.len()) {
        s.pending.insert(key);
        let shared = TileJob { map: s.map.clone(), colors: s.colors.clone(), materials: s.materials.clone() };
        let sender = s.sender.clone();
        s.pool.spawn(move || {
            // The viewer may have moved on to another map; then nobody listens.
            let _ = sender.send(build_tile(&shared, key, eye));
        });
    }
}

/// Grow the view distance with the camera's height above the ground, and the fog and far
/// plane with it.
pub fn update_view_distance(
    mut view: ResMut<MapView>,
    mut camera: Query<(&CameraRig, &mut DistanceFog, &mut Projection)>,
) {
    let Ok((rig, mut fog, mut projection)) = camera.single_mut() else { return };
    let agl = (rig.eye.z - view.surface_height(rig.eye.x, rig.eye.y)).max(0.0);
    let base = f64::from(view.view_distance);
    let per_metre = if view.tiled.is_some() { 10.0 } else { 4.0 };
    let far = (base + per_metre * agl).min(MAX_VIEW.max(base)) as f32;
    if (far - view.far).abs() < 0.02 * view.far {
        return;
    }
    view.far = far;
    fog.falloff = FogFalloff::Linear { start: 0.3 * far, end: far };
    if let Projection::Perspective(p) = &mut *projection {
        p.far = 1.5 * far;
    }
}

/// Refine or coarsen terrain chunks by their distance to the camera (nearest first, a
/// bounded number per frame) and hide everything beyond the view distance, and coarse
/// chunks whose tile is shown in detail.
#[allow(clippy::type_complexity)]
pub fn update_lod(
    view: Res<MapView>,
    camera: Query<&CameraRig>,
    mut chunks: Query<(Entity, &mut TerrainChunk, &mut Mesh3d, &mut Aabb, &mut Visibility, Option<&CoarseOf>)>,
    mut parts: Query<(&ChunkPart, Option<&PropLod>, Option<&CoarseOf>, &mut Visibility), Without<TerrainChunk>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Ok(rig) = camera.single() else { return };
    let eye = rig.eye;
    let world = &*view.world;
    let far = f64::from(view.far);
    let detailed = |c: Option<&CoarseOf>| match (&view.tiled, c) {
        (Some(s), Some(c)) => s.loaded.contains_key(&c.0),
        _ => false,
    };

    // (distance, entity, layer, chunk, stride)
    let mut todo: Vec<(f64, Entity, Layer, usize, usize)> = Vec::new();
    for (entity, chunk, _, _, mut vis, coarse) in &mut chunks {
        let d = box_distance(eye, chunk.min, chunk.max);
        let shown = d <= far && !detailed(coarse);
        vis.set_if_neq(if shown { Visibility::Inherited } else { Visibility::Hidden });
        // Hysteresis: coarsen only once clearly past the band.
        let fine = stride_for(chunk.layer, d);
        let coarse = stride_for(chunk.layer, d / 1.15);
        let target = if fine < chunk.stride {
            fine
        } else if coarse > chunk.stride {
            coarse
        } else {
            chunk.stride
        };
        if target != chunk.stride && shown {
            todo.push((d, entity, chunk.layer, chunk.index, target));
        }
    }
    todo.sort_by(|a, b| a.0.total_cmp(&b.0));
    todo.truncate(REBUILDS_PER_FRAME);
    let built: Vec<Option<MeshData>> = todo
        .par_iter()
        .map(|&(_, _, layer, i, s)| match layer {
            Layer::Grid => {
                let cell = world.grid().cell_size();
                Some(terrain::terrain_chunk(world, &view.chunks[i], s, skirt(cell, s)))
            }
            Layer::Coarse => {
                let st = view.tiled.as_ref()?;
                let (g, c) = (st.map.coarse(), &st.coarse_chunks[i]);
                let (min, _) = c.bounds(g);
                let anchor = DVec3::new(min.x, min.y, 0.0);
                Some(terrain::terrain_mesh(g, &st.coarse_colors, c, s, skirt(g.cell_size(), s), anchor))
            }
            Layer::Detail(key) => {
                let st = view.tiled.as_ref()?;
                let d = st.loaded.get(&key)?;
                let g: &HeightGrid = &d.tile.grid;
                Some(terrain::terrain_mesh(g, &st.colors, &d.chunks[i], s, skirt(g.cell_size(), s), d.anchor))
            }
        })
        .collect();
    for (&(_, entity, _, _, s), m) in todo.iter().zip(built) {
        let Some(m) = m else { continue };
        let Ok((_, mut chunk, mut mesh, mut bounds, _, _)) = chunks.get_mut(entity) else { continue };
        if let Some((lo, hi)) = bevy_bounds(&m) {
            *bounds = aabb(lo, hi);
        }
        mesh.0 = meshes.add(convert::mesh(&m));
        chunk.stride = s;
    }

    let detail_distance = f64::from(view.detail_distance);
    for (part, lod, coarse, mut vis) in &mut parts {
        let d = box_distance(eye, part.min, part.max);
        let shown = d <= far && !detailed(coarse) && lod.is_none_or(|l| l.near == (d < detail_distance));
        vis.set_if_neq(if shown { Visibility::Inherited } else { Visibility::Hidden });
    }
}

/// The landing pad under the current goal of agent `.0` (groups with `goals.pad`), as the
/// cameras see it.
#[derive(Component)]
pub struct PadVisual(usize);

pub fn spawn_pads(
    mut commands: Commands,
    sim: Res<Sim>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let groups = &sim.world.scenario().groups;
    if groups.iter().all(|g| g.spec.goals.pad <= 0.0) {
        return;
    }
    let mesh = meshes.add(convert::mesh(&autonomousim_scene::mesh::landing_pad(1.0)));
    // Two-sided, as the cameras draw it.
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.9,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    for (i, a) in sim.world.agents().iter().enumerate() {
        if groups[a.group].spec.goals.pad > 0.0 {
            // Meshes live in the render world only, so the bounds are given (the unit pad).
            let bounds = aabb(Vec3::new(-1.0, -0.01, -1.0), Vec3::new(1.0, 0.01, 1.0));
            commands.spawn((
                PadVisual(i),
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                bounds,
                Visibility::Hidden,
            ));
        }
    }
}

/// Pads lie on the ground under the goals, tilted with it, 2 cm up (as in
/// `sim::camera`); hidden while their agent is disabled or has no goal.
pub fn sync_pads(
    sim: Res<Sim>,
    origin: Res<RenderOrigin>,
    mut pads: Query<(&PadVisual, &mut Transform, &mut Visibility)>,
) {
    for (pad, mut t, mut visible) in &mut pads {
        let Some(a) = sim.world.agents().get(pad.0) else { continue };
        if a.disabled || a.goals.is_empty() {
            *visible = Visibility::Hidden;
            continue;
        }
        let g = a.goal().position;
        let (h, n) = sim.world.map().terrain().height_normal(g.x, g.y);
        let r = sim.world.scenario().groups[a.group].spec.goals.pad;
        let pose = autonomousim_core::math::Pose::new(
            DVec3::new(g.x, g.y, h) + 0.02 * n,
            glam::DQuat::from_rotation_arc(DVec3::Z, n),
        );
        *t = origin.transform(&pose).with_scale(Vec3::new(r as f32, 1.0, r as f32));
        *visible = Visibility::Inherited;
    }
}
