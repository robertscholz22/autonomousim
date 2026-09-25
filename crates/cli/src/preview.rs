//! Top-down preview image of a map: material colours with hill shading, water, trees and rocks.

use autonomousim_world::ObstacleShape;
use autonomousim_world::StaticWorld;
use autonomousim_world::obstacles::{ObstacleClass, tags};
use glam::DVec3;
use std::io::Write;
use std::path::Path;

/// Write a binary PPM with one pixel per `stride` cells (north up).
pub fn write_ppm(world: &StaticWorld, stride: usize, path: &Path) -> std::io::Result<()> {
    let t = world.terrain();
    let (cw, ch) = t.cells();
    let (w, h) = (cw / stride, ch / stride);
    let cell = t.cell_size() * stride as f64;
    let mut rgb = vec![0u8; w * h * 3];
    let light = glam::DVec3::new(-1.0, 1.0, 1.5).normalize();
    for py in 0..h {
        for px in 0..w {
            let (cx, cy) = (px * stride, py * stride);
            let z = |x: usize, y: usize| t.vertex_height(x.min(cw), y.min(ch));
            let gx = (z(cx + stride, cy) - z(cx, cy)) / cell;
            let gy = (z(cx, cy + stride) - z(cx, cy)) / cell;
            let n = glam::DVec3::new(-gx, -gy, 1.0).normalize();
            let shade = 0.35 + 0.65 * n.dot(light).max(0.0);
            let mut c = world.material(t.cell_material(cx, cy)).color.map(|v| v as f64);
            if let Some(level) = t.cell_water(cx, cy) {
                let depth = (level - 0.25 * (z(cx, cy) + z(cx + 1, cy) + z(cx, cy + 1) + z(cx + 1, cy + 1))).max(0.0);
                let k = (depth / 4.0).min(1.0);
                c = [40.0 - 20.0 * k, 110.0 - 50.0 * k, 160.0 - 40.0 * k];
            } else {
                c = c.map(|v| v * shade);
            }
            let i = ((h - 1 - py) * w + px) * 3;
            for k in 0..3 {
                rgb[i + k] = c[k].clamp(0.0, 255.0) as u8;
            }
        }
    }
    let origin = t.origin();
    for o in world.obstacles().obstacles() {
        let footprint = match (o.tag, &o.shape) {
            (tags::HEDGE | tags::FENCE | tags::BUILDING, ObstacleShape::Cuboid { half_extents: he }) => {
                Some((*he, [40u8, 90, 35]))
            }
            (tags::SILO, ObstacleShape::Cylinder { radius, .. }) => {
                Some((DVec3::new(*radius, *radius, 0.0), [200, 200, 205]))
            }
            _ => None,
        };
        if let Some((he, mut color)) = footprint {
            if o.class == ObstacleClass::Solid && o.tag == tags::HEDGE {
                continue;
            }
            match o.tag {
                tags::FENCE => color = [140, 100, 60],
                tags::BUILDING => color = [150, 60, 45],
                _ => {}
            }
            let (nx, ny) = ((he.x / 0.25).ceil() as i32, (he.y / 0.25).ceil() as i32);
            for i in -nx..=nx {
                for j in -ny..=ny {
                    let local = DVec3::new(he.x * i as f64 / nx.max(1) as f64, he.y * j as f64 / ny.max(1) as f64, 0.0);
                    let q = (o.pose.transform_point(local).truncate() - origin) / cell;
                    let (x, y) = (q.x as isize, q.y as isize);
                    if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                        let k = ((h - 1 - y as usize) * w + x as usize) * 3;
                        rgb[k..k + 3].copy_from_slice(&color);
                    }
                }
            }
            continue;
        }
        let (color, r) = match o.tag {
            tags::CANOPY => ([20u8, 70, 30], 1),
            tags::CANOPY_BROADLEAF => ([60, 120, 40], 1),
            tags::ROCK => ([150, 150, 150], 0),
            _ => continue,
        };
        let q = (o.pose.pos.truncate() - origin) / cell;
        let (px, py) = (q.x as isize, q.y as isize);
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (px + dx, py + dy);
                if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                    let i = ((h - 1 - y as usize) * w + x as usize) * 3;
                    rgb[i..i + 3].copy_from_slice(&color);
                }
            }
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(&rgb)?;
    f.flush()
}
