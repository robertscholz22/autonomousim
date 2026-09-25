//! Map files and content hashes.
//!
//! A map file is `MAGIC ‖ format version (u32 LE) ‖ content hash (32 bytes) ‖ zstd(postcard(map)
//! ‖ postcard(roads))`. The map part holds the metadata, the height grid (heights, cell
//! materials, water), the obstacles in their stored order and the material table; the road part
//! (format 2 on, written only for maps with roads) the road nodes and polylines. The height
//! pyramid, the obstacle BVH and the road grid are rebuilt on load. Format 1 files (no roads)
//! still load.
//!
//! The **content hash** is blake3 over a domain tag and the same postcard encoding (the roads
//! only when there are any, so maps without roads keep their format-1 hashes), so two maps
//! have equal hashes exactly when every stored value is bit-identical. Generators use it for
//! golden tests; recordings store it to check that a replay rebuilt the same map.

use crate::heightgrid::HeightGrid;
use crate::obstacles::{Obstacle, ObstacleClass, ObstacleSet, ObstacleShape};
use crate::roads::{RoadNetwork, RoadsData};
use crate::static_world::{MapMeta, StaticWorld};
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::{Read, Write};
use std::path::Path;
use std::str::FromStr;

const MAGIC: &[u8; 8] = b"AUTOSIMM";
const FORMAT_VERSION: u32 = 2;
const HASH_DOMAIN: &[u8] = b"autonomousim map v1";

/// blake3 content hash of a map (see the module docs).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MapHash(pub [u8; 32]);

impl MapHash {
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Display for MapHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

impl fmt::Debug for MapHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MapHash({})", self.hex())
    }
}

impl FromStr for MapHash {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.len() != 64 || !s.is_ascii() {
            return Err(format!("a map hash has 64 hex digits, got {s:?}"));
        }
        let mut out = [0u8; 32];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|e| format!("{s:?}: {e}"))?;
        }
        Ok(Self(out))
    }
}

impl Serialize for MapHash {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.hex())
    }
}

impl<'de> Deserialize<'de> for MapHash {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MapFileError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not an autonomousim map file")]
    BadMagic,
    #[error("unsupported map file version {0} (this build reads 1 to {FORMAT_VERSION})")]
    Version(u32),
    #[error("corrupt map file: {0}")]
    Corrupt(String),
    #[error("map content hash mismatch: file says {stored}, content is {actual}")]
    HashMismatch { stored: MapHash, actual: MapHash },
}

/// Borrowed view used for writing and hashing.
#[derive(Serialize)]
struct MapRef<'a> {
    meta: &'a MapMeta,
    origin: DVec2,
    cell: f64,
    nx: u32,
    ny: u32,
    heights: &'a [f32],
    materials: &'a [MaterialId],
    water: Option<&'a [f32]>,
    obstacles: ObstaclesRef<'a>,
    material_table: &'a MaterialTable,
}

// Obstacles are stored as externally tagged records: postcard is not self-describing, so it
// cannot read the internally tagged `ObstacleShape` of configuration files. The variant order
// of `ShapeRef` and `ShapeRec` is the file format.

struct ObstaclesRef<'a>(&'a [Obstacle]);

impl Serialize for ObstaclesRef<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(self.0.iter().map(|o| ObstacleRef {
            shape: match &o.shape {
                ObstacleShape::Sphere { radius } => ShapeRef::Sphere(*radius),
                ObstacleShape::Capsule { half_height, radius } => ShapeRef::Capsule(*half_height, *radius),
                ObstacleShape::Cylinder { half_height, radius } => ShapeRef::Cylinder(*half_height, *radius),
                ObstacleShape::Cone { half_height, radius } => ShapeRef::Cone(*half_height, *radius),
                ObstacleShape::Cuboid { half_extents } => ShapeRef::Cuboid(*half_extents),
                ObstacleShape::ConvexHull { points } => ShapeRef::ConvexHull(points),
            },
            pose: &o.pose,
            solid: o.class == ObstacleClass::Solid,
            material: o.material,
            tag: o.tag,
        }))
    }
}

#[derive(Serialize)]
struct ObstacleRef<'a> {
    shape: ShapeRef<'a>,
    pose: &'a Pose,
    solid: bool,
    material: MaterialId,
    tag: u16,
}

#[derive(Serialize)]
enum ShapeRef<'a> {
    Sphere(f64),
    Capsule(f64, f64),
    Cylinder(f64, f64),
    Cone(f64, f64),
    Cuboid(DVec3),
    ConvexHull(&'a [DVec3]),
}

#[derive(Deserialize)]
struct ObstacleRec {
    shape: ShapeRec,
    pose: Pose,
    solid: bool,
    material: MaterialId,
    tag: u16,
}

#[derive(Deserialize)]
enum ShapeRec {
    Sphere(f64),
    Capsule(f64, f64),
    Cylinder(f64, f64),
    Cone(f64, f64),
    Cuboid(DVec3),
    ConvexHull(Vec<DVec3>),
}

impl From<ObstacleRec> for Obstacle {
    fn from(r: ObstacleRec) -> Self {
        let shape = match r.shape {
            ShapeRec::Sphere(radius) => ObstacleShape::Sphere { radius },
            ShapeRec::Capsule(half_height, radius) => ObstacleShape::Capsule { half_height, radius },
            ShapeRec::Cylinder(half_height, radius) => ObstacleShape::Cylinder { half_height, radius },
            ShapeRec::Cone(half_height, radius) => ObstacleShape::Cone { half_height, radius },
            ShapeRec::Cuboid(half_extents) => ObstacleShape::Cuboid { half_extents },
            ShapeRec::ConvexHull(points) => ObstacleShape::ConvexHull { points },
        };
        let class = if r.solid { ObstacleClass::Solid } else { ObstacleClass::Foliage };
        Obstacle { shape, pose: r.pose, class, material: r.material, tag: r.tag }
    }
}

/// Owned form used for reading (same field order and types as [`MapRef`]).
#[derive(Deserialize)]
struct MapData {
    meta: MapMeta,
    origin: DVec2,
    cell: f64,
    nx: u32,
    ny: u32,
    heights: Vec<f32>,
    materials: Vec<MaterialId>,
    water: Option<Vec<f32>>,
    obstacles: Vec<ObstacleRec>,
    material_table: MaterialTable,
}

fn view(world: &StaticWorld) -> MapRef<'_> {
    let t = world.terrain();
    let (nx, ny) = t.dims();
    MapRef {
        meta: &world.meta,
        origin: t.origin(),
        cell: t.cell_size(),
        nx: nx as u32,
        ny: ny as u32,
        heights: t.heights(),
        materials: t.materials(),
        water: t.water(),
        obstacles: ObstaclesRef(world.obstacles().obstacles()),
        material_table: world.materials(),
    }
}

/// Content hash of a map.
pub fn content_hash(world: &StaticWorld) -> MapHash {
    let mut h = blake3::Hasher::new();
    h.update(HASH_DOMAIN);
    postcard::to_io(&view(world), &mut h).expect("hashing cannot fail");
    if !world.roads().is_empty() {
        postcard::to_io(&world.roads().stored(), &mut h).expect("hashing cannot fail");
    }
    MapHash(*h.finalize().as_bytes())
}

/// Write a map file (zstd level `level`, 3 is a good default); returns the content hash.
pub fn write(world: &StaticWorld, w: impl Write, level: i32) -> Result<MapHash, MapFileError> {
    let hash = content_hash(world);
    let mut w = w;
    w.write_all(MAGIC)?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    w.write_all(&hash.0)?;
    let enc = zstd::Encoder::new(w, level)?;
    let mut enc = postcard::to_io(&view(world), enc).map_err(|e| MapFileError::Corrupt(e.to_string()))?;
    if !world.roads().is_empty() {
        enc = postcard::to_io(&world.roads().stored(), enc).map_err(|e| MapFileError::Corrupt(e.to_string()))?;
    }
    enc.finish()?.flush()?;
    Ok(hash)
}

/// Read a map file; the content hash is recomputed and checked.
pub fn read(r: impl Read) -> Result<(StaticWorld, MapHash), MapFileError> {
    let mut r = r;
    let mut header = [0u8; 8 + 4 + 32];
    r.read_exact(&mut header).map_err(|_| MapFileError::BadMagic)?;
    if &header[..8] != MAGIC {
        return Err(MapFileError::BadMagic);
    }
    let version = u32::from_le_bytes(header[8..12].try_into().expect("4 bytes"));
    if !(1..=FORMAT_VERSION).contains(&version) {
        return Err(MapFileError::Version(version));
    }
    let stored = MapHash(header[12..44].try_into().expect("32 bytes"));
    let bytes = zstd::decode_all(r)?;
    let (d, rest): (MapData, _) =
        postcard::take_from_bytes(&bytes).map_err(|e| MapFileError::Corrupt(e.to_string()))?;
    let roads = if rest.is_empty() {
        RoadNetwork::default()
    } else {
        let r: RoadsData = postcard::from_bytes(rest).map_err(|e| MapFileError::Corrupt(e.to_string()))?;
        RoadNetwork::try_from(r).map_err(MapFileError::Corrupt)?
    };
    let (nx, ny) = (d.nx as usize, d.ny as usize);
    let cells = nx.saturating_sub(1) * ny.saturating_sub(1);
    let valid = nx >= 2
        && ny >= 2
        && d.cell > 0.0
        && d.heights.len() == nx * ny
        && d.materials.len() == cells
        && d.water.as_ref().is_none_or(|w| w.len() == cells)
        && d.heights.iter().all(|h| h.is_finite());
    if !valid {
        return Err(MapFileError::Corrupt("inconsistent height grid".into()));
    }
    let mut terrain = HeightGrid::new(d.origin, d.cell, nx, ny, d.heights, d.materials);
    if let Some(w) = d.water {
        terrain = terrain.with_water(w);
    }
    let world = StaticWorld::new(
        d.meta,
        terrain,
        ObstacleSet::new(d.obstacles.into_iter().map(Obstacle::from).collect()),
        d.material_table,
    )
    .with_roads(roads);
    let actual = content_hash(&world);
    if actual != stored {
        return Err(MapFileError::HashMismatch { stored, actual });
    }
    Ok((world, actual))
}

/// Write a map file atomically (a temporary file renamed into place).
pub fn save(world: &StaticWorld, path: impl AsRef<Path>) -> Result<MapHash, MapFileError> {
    let path = path.as_ref();
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp{}-{n}", std::process::id()));
    let result = (|| {
        let file = std::fs::File::create(&tmp)?;
        let hash = write(world, std::io::BufWriter::new(file), 3)?;
        std::fs::rename(&tmp, path)?;
        Ok(hash)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

pub fn load(path: impl AsRef<Path>) -> Result<(StaticWorld, MapHash), MapFileError> {
    read(std::io::BufReader::new(std::fs::File::open(path)?))
}

impl StaticWorld {
    /// See [`content_hash`].
    pub fn content_hash(&self) -> MapHash {
        content_hash(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testworlds;

    #[test]
    fn round_trip_preserves_content() {
        let w = testworlds::lake(120.0, 3.0, -1.0);
        let f = testworlds::forest_patch(100.0, 150.0, 3);
        assert_ne!(w.content_hash(), f.content_hash());
        for world in [w, f] {
            let mut buf = Vec::new();
            let h = write(&world, &mut buf, 3).unwrap();
            assert_eq!(h, world.content_hash());
            let (back, h2) = read(&buf[..]).unwrap();
            assert_eq!(h2, h);
            assert_eq!(back.terrain().heights(), world.terrain().heights());
            // Dry cells are NaN: compare bit patterns.
            let bits = |w: &StaticWorld| w.terrain().water().map(|w| w.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            assert_eq!(bits(&back), bits(&world));
            assert_eq!(back.obstacles().obstacles(), world.obstacles().obstacles());
            assert_eq!(back.meta, world.meta);
            // Corruption is detected.
            let n = buf.len();
            buf[20] ^= 1;
            assert!(matches!(read(&buf[..]), Err(MapFileError::HashMismatch { .. })));
            buf[20] ^= 1;
            buf[n - 3] ^= 0x55;
            assert!(read(&buf[..]).is_err());
            assert!(matches!(read(&b"nonsense"[..]), Err(MapFileError::BadMagic)));
        }
        let h = testworlds::flat(10.0).content_hash();
        assert_eq!(h.to_string().parse::<MapHash>().unwrap(), h);
        assert_eq!(serde_json::from_str::<MapHash>(&serde_json::to_string(&h).unwrap()).unwrap(), h);
    }

    #[test]
    fn roads_round_trip_and_old_files_load() {
        use crate::roads::{NodeKind, Polyline, Road, RoadClass, RoadNode};
        let plain = testworlds::lake(120.0, 3.0, -1.0);
        let before = plain.content_hash();
        let nodes = vec![
            RoadNode { position: DVec3::new(-50.0, -50.0, 0.0), kind: NodeKind::End },
            RoadNode { position: DVec3::new(50.0, -40.0, 0.0), kind: NodeKind::Yard },
        ];
        let line =
            Polyline::new((0..=100).map(|i| DVec3::new(-50.0 + i as f64, -50.0 + 0.1 * i as f64, 0.0)).collect());
        let road = Road { class: RoadClass::Gravel, width: 4.0, start: 0, end: 1, line };
        let with = plain.clone().with_roads(RoadNetwork::new(nodes, vec![road]).unwrap());
        // Roads change the hash; a map without them keeps its old one.
        assert_ne!(with.content_hash(), before);
        assert_eq!(plain.clone().with_roads(RoadNetwork::default()).content_hash(), before);
        let mut buf = Vec::new();
        write(&with, &mut buf, 3).unwrap();
        let (back, h) = read(&buf[..]).unwrap();
        assert_eq!(h, with.content_hash());
        assert_eq!(back.roads(), with.roads());
        assert!(back.roads().on_road(DVec2::new(0.0, -45.0)).is_some());
        // A format-1 file: the same bytes as a map without roads, under version 1.
        let mut old = Vec::new();
        write(&plain, &mut old, 3).unwrap();
        old[8..12].copy_from_slice(&1u32.to_le_bytes());
        let (back, h) = read(&old[..]).unwrap();
        assert_eq!(h, before);
        assert!(back.roads().is_empty());
        old[8..12].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(read(&old[..]), Err(MapFileError::Version(3))));
    }
}
