//! Road ribbons: the road network drawn as strips just above the terrain, with lane markings
//! on paved roads and wheel ruts on tracks, merged per terrain chunk.
//!
//! The terrain under a road already carries its material; the ribbons add crisp edges and
//! markings. Every strip vertex sits a few centimetres above the terrain below it, so the
//! ribbons follow the crown and the blended shoulders. Paved roads lie highest, so they
//! cover the gravel roads and tracks that join them.

use crate::mesh::{MeshData, srgb};
use crate::props::chunk_index;
use crate::terrain::Chunk;
use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::Terrain;
use autonomousim_world::{Polyline, Road, RoadClass, StaticWorld};
use glam::DVec2;

/// Height of a road surface above the terrain (m), by class.
fn lift(class: RoadClass) -> f64 {
    match class {
        RoadClass::Paved => 0.06,
        RoadClass::Gravel => 0.05,
        RoadClass::Track => 0.04,
    }
}

/// Markings lie this much above their road's surface (m).
const MARKING_LIFT: f64 = 0.015;
const MARKING: [u8; 3] = [232, 232, 226];
/// Centre-line dashes: length and period along the road (m).
const DASH: f64 = 3.0;
const DASH_PERIOD: f64 = 9.0;

/// A strip along a road: lateral offsets (m, + left) of its long edges, sampled at the
/// given interior offsets too so it follows a crowned surface.
struct Layer {
    /// Ascending (right to left).
    offsets: Vec<f64>,
    lift: f64,
    color: [f32; 4],
}

/// Unit vector to the left of heading `h`.
fn left(h: f64) -> DVec2 {
    DVec2::new(-h.sin(), h.cos())
}

/// Offsets of a narrow strip between `a` and `b`, right to left.
fn band(a: f64, b: f64) -> Vec<f64> {
    vec![a.min(b), a.max(b)]
}

/// The layers of a road: its surface, then markings or ruts.
fn layers(world: &StaticWorld, road: &Road) -> Vec<Layer> {
    let table = world.materials();
    let color = |id: MaterialId, k: f32| {
        let mut c = srgb(if (id.0 as usize) < table.len() { table.get(id).color } else { [128, 128, 128] });
        c[..3].iter_mut().for_each(|v| *v *= k);
        c
    };
    let w = 0.5 * road.width;
    let base = lift(road.class);
    let span = |a: f64, b: f64, n: usize| (0..=n).map(|k| a + (b - a) * k as f64 / n as f64).collect::<Vec<_>>();
    match road.class {
        RoadClass::Paved => {
            let edge = |side: f64| Layer {
                offsets: band(side * (w - 0.35), side * (w - 0.2)),
                lift: base + MARKING_LIFT,
                color: srgb(MARKING),
            };
            vec![Layer { offsets: span(-w, w, 4), lift: base, color: srgb([64, 66, 70]) }, edge(-1.0), edge(1.0)]
        }
        RoadClass::Gravel => vec![Layer { offsets: span(-w, w, 2), lift: base, color: color(MaterialId::GRAVEL, 1.0) }],
        RoadClass::Track => {
            let rut = |side: f64| Layer {
                offsets: band(side * 0.55, side * 1.05),
                lift: base,
                color: color(MaterialId::DIRT, 0.7),
            };
            vec![rut(-1.0), rut(1.0)]
        }
    }
}

/// Append a strip along `line` between the stations of `points` (indices into the line's
/// points) to `m`.
fn strip(m: &mut MeshData, world: &StaticWorld, line: &Polyline, points: &[usize], layer: &Layer) {
    let pts = line.points();
    let n = pts.len();
    let cols = layer.offsets.len();
    let base = m.vertex_count() as u32;
    for &i in points {
        let (a, b) = (pts[i.saturating_sub(1)], pts[(i + 1).min(n - 1)]);
        let d = (b - a).truncate();
        let l = left(d.y.atan2(d.x));
        for &off in &layer.offsets {
            let xy = pts[i].truncate() + off * l;
            let (h, normal) = world.terrain().height_normal(xy.x, xy.y);
            m.push_vertex(xy.extend(h + layer.lift).as_vec3(), normal.as_vec3(), layer.color);
        }
    }
    for r in 0..points.len().saturating_sub(1) as u32 {
        for c in 0..cols as u32 - 1 {
            // Rows run along the road, columns from right to left: counter-clockwise from above.
            let (i00, i01) = (base + r * cols as u32 + c, base + r * cols as u32 + c + 1);
            let (i10, i11) = (i00 + cols as u32, i01 + cols as u32);
            m.push_triangle(i00, i10, i11);
            m.push_triangle(i00, i11, i01);
        }
    }
}

/// Road ribbons grouped by chunk: one merged mesh per entry of `chunks` (empty where no road
/// passes). A road is cut where it crosses into another chunk.
pub fn roads_by_chunk(world: &StaticWorld, chunks: &[Chunk], size: usize) -> Vec<MeshData> {
    let grid = world.terrain();
    let mut out = vec![MeshData::new(); chunks.len()];
    for road in world.roads().roads() {
        let line = &road.line;
        let pts = line.points();
        if pts.len() < 2 {
            continue;
        }
        let layers = layers(world, road);
        // Runs of segments whose midpoints lie in the same chunk.
        let chunk_of = |i: usize| chunk_index(grid, size, 0.5 * (pts[i] + pts[i + 1]));
        let mut start = 0;
        while start + 1 < pts.len() {
            let k = chunk_of(start);
            let mut end = start + 1;
            while end + 1 < pts.len() && chunk_of(end) == k {
                end += 1;
            }
            let run: Vec<usize> = (start..=end).collect();
            for layer in &layers {
                strip(&mut out[k], world, line, &run, layer);
            }
            start = end;
        }
        // Centre-line dashes on paved roads.
        if road.class == RoadClass::Paved {
            let dash =
                Layer { offsets: vec![-0.06, 0.06], lift: lift(road.class) + MARKING_LIFT, color: srgb(MARKING) };
            let stations = station_indices(line);
            let mut s = 0.5 * DASH_PERIOD;
            while s + DASH < line.length() {
                let run: Vec<usize> =
                    stations.iter().copied().filter(|&(_, st)| st >= s && st <= s + DASH).map(|(i, _)| i).collect();
                if run.len() >= 2 {
                    let k = chunk_index(grid, size, pts[run[0]]);
                    strip(&mut out[k], world, line, &run, &dash);
                }
                s += DASH_PERIOD;
            }
        }
    }
    out
}

/// Every point of `line` with its station.
fn station_indices(line: &Polyline) -> Vec<(usize, f64)> {
    let pts = line.points();
    let mut s = 0.0;
    let mut out = Vec::with_capacity(pts.len());
    for (i, p) in pts.iter().enumerate() {
        if i > 0 {
            s += (p.truncate() - pts[i - 1].truncate()).length();
        }
        out.push((i, s));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::chunks;
    use autonomousim_world::{NodeKind, RoadNetwork, RoadNode, testworlds};
    use glam::{DVec3, Vec3};

    /// Lowest height of `m` above the terrain (m).
    fn min_clearance(world: &StaticWorld, m: &MeshData) -> f64 {
        m.positions
            .iter()
            .map(|p| {
                let p = Vec3::from_array(*p).as_dvec3();
                p.z - world.terrain().height(p.x, p.y)
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// A flat world with a paved road and a track crossing it.
    fn world() -> StaticWorld {
        let w = testworlds::flat(200.0);
        let line = |a: DVec2, b: DVec2| {
            let n = (a.distance(b)).ceil() as usize;
            Polyline::new((0..=n).map(|k| a.lerp(b, k as f64 / n as f64).extend(0.0)).collect())
        };
        let node = |x: f64, y: f64| RoadNode { position: DVec3::new(x, y, 0.0), kind: NodeKind::End };
        let nodes = vec![node(-90.0, 0.0), node(90.0, 0.0), node(0.0, -60.0), node(0.0, 60.0)];
        let roads = vec![
            Road {
                class: RoadClass::Paved,
                width: 6.0,
                start: 0,
                end: 1,
                line: line(DVec2::new(-90.0, 0.0), DVec2::new(90.0, 0.0)),
            },
            Road {
                class: RoadClass::Track,
                width: 3.0,
                start: 2,
                end: 3,
                line: line(DVec2::new(0.0, -60.0), DVec2::new(0.0, 60.0)),
            },
        ];
        w.with_roads(RoadNetwork::new(nodes, roads).unwrap())
    }

    #[test]
    fn ribbons_cover_the_roads_just_above_the_terrain() {
        let w = world();
        let cs = chunks(w.terrain(), 32);
        let meshes = roads_by_chunk(&w, &cs, 32);
        assert_eq!(meshes.len(), cs.len());
        let all = meshes.iter().fold(MeshData::new(), |mut m, c| {
            m.append(c);
            m
        });
        assert!(min_clearance(&w, &all) > 0.035);
        // Faces point up.
        for t in all.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| Vec3::from_array(all.positions[i as usize]));
            assert!((b - a).cross(c - a).z > 0.0, "{a} {b} {c}");
        }
        // The paved road is covered edge to edge along its whole length: its area.
        let area: f32 = all
            .indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| Vec3::from_array(all.positions[i as usize]));
                if a.y.abs() <= 3.01 && b.y.abs() <= 3.01 && c.y.abs() <= 3.01 && (a.z - 0.06).abs() < 1e-4 {
                    0.5 * (b - a).cross(c - a).length()
                } else {
                    0.0
                }
            })
            .sum();
        assert!((area - 180.0 * 6.0).abs() < 1.0, "{area}");
        // Every chunk the roads cross has ribbons, the others none.
        for (c, m) in cs.iter().zip(&meshes) {
            let (lo, hi) = c.bounds(w.terrain());
            let crossed = (lo.y < 0.0 && hi.y > 0.0 && lo.x < 90.0 && hi.x > -90.0)
                || (lo.x < 0.0 && hi.x > 0.0 && hi.y > -60.0 && lo.y < 60.0);
            assert_eq!(!m.is_empty(), crossed, "chunk {c:?}");
        }
    }
}
