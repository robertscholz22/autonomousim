//! Small hand-made maps for unit tests, controller tuning and benchmarks. All are centred on
//! the origin and deterministic.

use crate::heightgrid::HeightGrid;
use crate::obstacles::{Obstacle, ObstacleSet, ObstacleShape};
use crate::roads::{NodeKind, Polyline, Road, RoadClass, RoadNetwork, RoadNode, Section};
use crate::static_world::{MapMeta, StaticWorld};
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use autonomousim_core::rng::Seed;
use autonomousim_core::terrain::Terrain;
use glam::{DVec2, DVec3};

pub use crate::obstacles::tags;

fn grid(size: f64, cell: f64, height: impl Fn(f64, f64) -> f64, material: MaterialId) -> HeightGrid {
    let n = (size / cell).round() as usize + 1;
    HeightGrid::from_fn(DVec2::splat(-0.5 * size), cell, n, n, height, |_, _| material)
}

fn world(name: &str, terrain: HeightGrid, obstacles: Vec<Obstacle>) -> StaticWorld {
    StaticWorld::new(
        MapMeta::new(name, &format!("testworld/{name}"), 0),
        terrain,
        ObstacleSet::new(obstacles),
        MaterialTable::standard(),
    )
}

/// Conifer standing on the ground at `base`: a solid capsule trunk and a foliage cone.
pub fn tree(base: DVec3, height: f64) -> [Obstacle; 2] {
    let trunk_r = 0.025 * height;
    let trunk_hh = 0.2 * height;
    let crown_hh = 0.3 * height;
    let trunk = Obstacle::solid(
        ObstacleShape::Capsule { half_height: trunk_hh, radius: trunk_r },
        Pose::from_translation(base + DVec3::Z * trunk_hh),
        MaterialId::WOOD,
    )
    .with_tag(tags::TRUNK);
    let canopy = Obstacle::foliage(
        ObstacleShape::Cone { half_height: crown_hh, radius: 0.25 * height },
        Pose::from_translation(base + DVec3::Z * (height - crown_hh)),
    )
    .with_tag(tags::CANOPY);
    [trunk, canopy]
}

/// Flat grass plane at `z = 0`.
pub fn flat(size: f64) -> StaticWorld {
    world("flat", grid(size, 2.0, |_, _| 0.0, MaterialId::GRASS), Vec::new())
}

/// Flat asphalt with a circular road of centre-line `radius` (m) round the origin, in two
/// halves (counter-clockwise from (r, 0) over the top, then back under the bottom) meeting at
/// two through nodes: `lanes[0]` lanes counter-clockwise and `lanes[1]` clockwise, each
/// `lane_width` wide, of road class `class` (its speed limit). For traffic tests.
pub fn ring(radius: f64, lanes: [u8; 2], lane_width: f64, class: RoadClass) -> StaticWorld {
    let size = 2.0 * (radius + 60.0);
    let terrain = grid(size, 2.0, |_, _| 0.0, MaterialId::ASPHALT);
    let n = ((std::f64::consts::PI * radius / 2.0).ceil() as usize).max(8);
    let arc = |from: f64| {
        Polyline::new(
            (0..=n)
                .map(|k| {
                    let a = from + std::f64::consts::PI * k as f64 / n as f64;
                    DVec3::new(radius * a.cos(), radius * a.sin(), 0.0)
                })
                .collect(),
        )
    };
    let section = Section { lanes, lane_width, median: 0.0, bike: [0.0; 2], parking: [0.0; 2], sidewalk: [0.0; 2] };
    let node = |x: f64| RoadNode { position: DVec3::new(x, 0.0, 0.0), kind: NodeKind::Junction };
    let road = |start, end, from| Road { class, width: section.width(), start, end, line: arc(from) };
    let net =
        RoadNetwork::new(vec![node(radius), node(-radius)], vec![road(0, 1, 0.0), road(1, 0, std::f64::consts::PI)])
            .and_then(|n| n.with_sections(vec![section; 2]))
            .expect("a valid ring");
    world("ring", terrain, Vec::new()).with_roads(net)
}

/// Plane rising along +x with slope `angle` (radians), height 0 at the origin.
pub fn incline(size: f64, angle: f64, material: MaterialId) -> StaticWorld {
    let s = angle.tan();
    world("incline", grid(size, 2.0, |x, _| s * x, material), Vec::new())
}

/// Smooth rolling hills `a·sin(2πx/λ)·cos(2πy/λ)`.
pub fn sine_hills(size: f64, amplitude: f64, wavelength: f64) -> StaticWorld {
    let k = std::f64::consts::TAU / wavelength;
    world(
        "sine_hills",
        grid(size, 1.0, |x, y| amplitude * (k * x).sin() * (k * y).cos(), MaterialId::GRASS),
        Vec::new(),
    )
}

/// Flat sand plain with a paraboloid basin of the given depth (radius `size/4`) filled to
/// `water_level` (≤ 0).
pub fn lake(size: f64, depth: f64, water_level: f64) -> StaticWorld {
    let r = 0.25 * size;
    let terrain = grid(size, 2.0, |x, y| -depth * (1.0 - (x * x + y * y) / (r * r)).max(0.0), MaterialId::SAND);
    let (w, h) = terrain.cells();
    let mut water = vec![f32::NAN; w * h];
    let (o, c) = (terrain.origin(), terrain.cell_size());
    for cy in 0..h {
        for cx in 0..w {
            let p = o + DVec2::new(cx as f64 + 0.5, cy as f64 + 0.5) * c;
            if terrain.height(p.x, p.y) < water_level {
                water[cy * w + cx] = water_level as f32;
            }
        }
    }
    world("lake", terrain.with_water(water), Vec::new())
}

/// Flat plane with a single 10 m tree at the origin (trunk radius 0.25 m, crown from 4 m to
/// 10 m with base radius 2.5 m).
pub fn single_tree() -> StaticWorld {
    world("single_tree", grid(100.0, 2.0, |_, _| 0.0, MaterialId::GRASS), tree(DVec3::ZERO, 10.0).to_vec())
}

/// Flat plane with an `n × n` grid of vertical cylinders spaced `spacing` apart.
pub fn pillars(n: usize, spacing: f64, radius: f64, height: f64) -> StaticWorld {
    let size = (n as f64 + 2.0) * spacing;
    let off = 0.5 * (n as f64 - 1.0) * spacing;
    let obstacles = (0..n * n)
        .map(|k| {
            let p = DVec3::new((k % n) as f64 * spacing - off, (k / n) as f64 * spacing - off, 0.5 * height);
            Obstacle::solid(
                ObstacleShape::Cylinder { half_height: 0.5 * height, radius },
                Pose::from_translation(p),
                MaterialId::CONCRETE,
            )
            .with_tag(tags::PILLAR)
        })
        .collect();
    world("pillars", grid(size, 2.0, |_, _| 0.0, MaterialId::CONCRETE), obstacles)
}

/// Square arena of inner half-width `half` enclosed by 0.2 m thick walls of the given height.
pub fn walled_arena(half: f64, height: f64) -> StaticWorld {
    let t = 0.1;
    let wall = |c: DVec3, he: DVec3| {
        Obstacle::solid(ObstacleShape::Cuboid { half_extents: he }, Pose::from_translation(c), MaterialId::CONCRETE)
            .with_tag(tags::WALL)
    };
    let hz = 0.5 * height;
    let obstacles = vec![
        wall(DVec3::new(half + t, 0.0, hz), DVec3::new(t, half + 2.0 * t, hz)),
        wall(DVec3::new(-half - t, 0.0, hz), DVec3::new(t, half + 2.0 * t, hz)),
        wall(DVec3::new(0.0, half + t, hz), DVec3::new(half, t, hz)),
        wall(DVec3::new(0.0, -half - t, hz), DVec3::new(half, t, hz)),
    ];
    world("walled_arena", grid(2.0 * half + 20.0, 2.0, |_, _| 0.0, MaterialId::CONCRETE), obstacles)
}

/// Gentle hills covered by trees (`density` per hectare, at least 3 m apart, heights 8–25 m)
/// and a few boulders.
pub fn forest_patch(size: f64, density: f64, seed: u64) -> StaticWorld {
    let root = Seed::from_u64(seed).child("testworld/forest_patch");
    let k = std::f64::consts::TAU / 120.0;
    let terrain = grid(size, 1.0, |x, y| 3.0 * (k * x).sin() * (k * y + 0.7).cos(), MaterialId::FOREST_FLOOR);
    let half = 0.5 * size - 2.0;
    let target = (density * size * size / 10_000.0).round() as usize;
    let mut rng = root.child("trees").rng();
    let mut placed: Vec<DVec2> = Vec::with_capacity(target);
    let mut obstacles = Vec::with_capacity(2 * target + 16);
    for _ in 0..target * 20 {
        if placed.len() == target {
            break;
        }
        let p = DVec2::new(rng.range(-half, half), rng.range(-half, half));
        let height = rng.range(8.0, 25.0);
        if placed.iter().any(|q| q.distance_squared(p) < 9.0) {
            continue;
        }
        placed.push(p);
        obstacles.extend(tree(p.extend(terrain.height(p.x, p.y)), height));
    }
    let mut rng = root.child("rocks").rng();
    for _ in 0..(size * size / 40_000.0).ceil() as usize {
        let p = DVec2::new(rng.range(-half, half), rng.range(-half, half));
        let r = rng.range(0.4, 1.5);
        let center = p.extend(terrain.height(p.x, p.y) + 0.3 * r);
        obstacles.push(
            Obstacle::solid(ObstacleShape::Sphere { radius: r }, Pose::from_translation(center), MaterialId::ROCK)
                .with_tag(tags::ROCK),
        );
    }
    let mut w = world("forest_patch", terrain, obstacles);
    w.meta.seed = seed;
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::geometry::{HitKind, HitMask, Ray, StaticGeometry};

    #[test]
    fn incline_is_exact() {
        let w = incline(100.0, 0.2, MaterialId::ROCK);
        let (h, n) = w.terrain().height_normal(7.3, -4.1);
        assert!((h - 7.3 * 0.2f64.tan()).abs() < 1e-5);
        assert!((n - DVec3::new(-0.2f64.sin(), 0.0, 0.2f64.cos())).length() < 1e-6);
    }

    #[test]
    fn forest_is_deterministic_and_grounded() {
        let a = forest_patch(200.0, 150.0, 7);
        let b = forest_patch(200.0, 150.0, 7);
        let c = forest_patch(200.0, 150.0, 8);
        assert_eq!(a.obstacle_set().obstacles(), b.obstacle_set().obstacles());
        assert_ne!(a.obstacle_set().obstacles(), c.obstacle_set().obstacles());
        let trunks: Vec<_> = a.obstacle_set().obstacles().iter().filter(|o| o.tag == tags::TRUNK).collect();
        assert_eq!(trunks.len(), 600);
        for o in trunks {
            let ObstacleShape::Capsule { half_height, .. } = o.shape else { panic!() };
            let base = o.pose.pos.z - half_height;
            assert!((base - a.terrain().height(o.pose.pos.x, o.pose.pos.y)).abs() < 1e-5);
        }
    }

    #[test]
    fn pillars_and_arena_block_rays() {
        let w = pillars(4, 10.0, 0.5, 8.0);
        assert_eq!(w.obstacle_set().len(), 16);
        let hit = w.raycast(&Ray::new(DVec3::new(-30.0, -15.0, 2.0), DVec3::X), 100.0, HitMask::ALL).unwrap();
        assert!(matches!(hit.kind, HitKind::Solid(0)));
        assert!((hit.point.x + 15.5).abs() < 1e-9);

        let w = walled_arena(10.0, 3.0);
        for dir in [DVec3::X, -DVec3::X, DVec3::Y, -DVec3::Y, DVec3::new(1.0, 1.0, 0.0)] {
            let hit = w.raycast(&Ray::new(DVec3::new(0.0, 0.0, 1.0), dir), 100.0, HitMask::ALL).unwrap();
            assert!(matches!(hit.kind, HitKind::Solid(_)), "{dir}");
        }
        assert!(
            (w.obstacles().nearest_distance(DVec3::new(9.0, 0.0, 1.0), 5.0, HitMask::SOLID).unwrap() - 1.0).abs()
                < 1e-9
        );
    }

    #[test]
    fn ring_lanes_close_the_loop() {
        let w = ring(40.0, [2, 1], 3.5, RoadClass::Local);
        let g = w.roads().lanes();
        assert_eq!(g.lanes().len(), 6);
        for (k, l) in g.lanes().iter().enumerate() {
            // Through nodes: each lane continues straight into the next half's.
            let straight =
                l.successors.iter().filter(|&&c| g.connectors()[c as usize].turn == crate::lanes::Turn::Straight);
            eprintln!(
                "lane {k} dir {} index {} length {:.2} successors {:?}",
                l.dir,
                l.index,
                l.line.length(),
                l.successors
            );
            assert_eq!(straight.count(), 1, "lane {k}");
        }
        // Counter-clockwise inner lane plus its connectors: a full circle at its offset.
        let inner = &g.lanes()[0];
        let r = inner.line.points()[0].truncate().length();
        let c = &g.connectors()[inner.successors[0] as usize];
        let next = &g.lanes()[c.to as usize];
        let c2 = &g.connectors()[next.successors[0] as usize];
        let total = inner.line.length() + c.line.length() + next.line.length() + c2.line.length();
        assert!((total - std::f64::consts::TAU * r).abs() < 0.5, "{total} vs {}", std::f64::consts::TAU * r);
    }
}
