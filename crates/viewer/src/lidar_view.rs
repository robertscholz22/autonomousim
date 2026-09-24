//! The followed agent's latest LiDAR scan in 2D (V or `--lidar-view`): a top view around the
//! sensor with the heading up, and for ring patterns a range image with one row per ring
//! (highest on top) and one column per azimuth (ahead in the middle, left on the left). The
//! colours are those of the hits drawn in the scene: red near, through yellow, to green far;
//! dark grey is a beam without a return.

use crate::hud::Hud;
use crate::overlay::{Scan, latest_scan, range_color};
use crate::sim::Sim;
use autonomousim_core::math::quat::yaw;
use autonomousim_sensors::BeamPattern;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use glam::{DQuat, DVec3};

/// Width of the views (points).
const WIDTH: f32 = 260.0;
/// Returns drawn in the top view at most.
const MAX_POINTS: usize = 4096;
const BACKGROUND: egui::Color32 = egui::Color32::from_gray(18);
const NO_RETURN: egui::Color32 = egui::Color32::from_gray(45);
const GOAL: egui::Color32 = egui::Color32::from_rgb(255, 204, 26);

#[derive(Resource, Default)]
pub struct LidarView {
    pub visible: bool,
}

fn color32(c: Color) -> egui::Color32 {
    let c = c.to_srgba();
    let u = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    egui::Color32::from_rgb(u(c.red), u(c.green), u(c.blue))
}

/// The range image of a ring pattern: its size `[columns, rows]` and the range of each pixel,
/// row by row from the top. Rows are the rings from the highest elevation down. Columns run
/// from behind the sensor over its left, ahead in the middle and its right back to behind (a
/// full turn), or from the left edge of the field of view to the right edge. `None` for
/// explicit beams or when the ranges do not match the pattern.
pub fn range_image(pattern: &BeamPattern, ranges: &[Option<f64>]) -> Option<([usize; 2], Vec<Option<f64>>)> {
    let BeamPattern::Rings { elevations, azimuths, azimuth_fov } = pattern else { return None };
    let (rows, cols) = (elevations.len(), *azimuths as usize);
    if cols == 0 || rows * cols != ranges.len() {
        return None;
    }
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| elevations[b].total_cmp(&elevations[a]));
    let full = (azimuth_fov - 360.0).abs() < 1e-9;
    let mut out = Vec::with_capacity(rows * cols);
    for ring in order {
        for c in 0..cols {
            // The beams of a ring turn counter-clockwise (to the left): from ahead on a full
            // turn, from the right edge otherwise.
            let k = if full { (cols / 2 + cols - c) % cols } else { cols - 1 - c };
            out.push(ranges[ring * cols + k]);
        }
    }
    Some(([cols, rows], out))
}

/// Spacing of the range rings in the top view: at most four rings.
fn ring_step(max: f64) -> f64 {
    [1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0].into_iter().find(|s| max / s <= 4.0).unwrap_or(500.0)
}

pub fn lidar_view(
    mut contexts: EguiContexts,
    mut view: ResMut<LidarView>,
    hud: Res<Hud>,
    sim: Res<Sim>,
    keys: Res<ButtonInput<KeyCode>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    if !ctx.egui_wants_keyboard_input() && keys.just_pressed(KeyCode::KeyV) {
        view.visible = !view.visible;
    }
    if !view.visible || !hud.visible {
        return Ok(());
    }
    let agent = sim.pilot;
    let scan = latest_scan(&sim, agent);
    let a = sim.world.agent(agent);
    let goal = (!a.goals.is_empty()).then(|| a.goal().position);
    // Above the replay timeline.
    let bottom = if sim.replay.is_some() { -84.0 } else { -8.0 };
    egui::Window::new("LiDAR")
        .anchor(egui::Align2::RIGHT_BOTTOM, [-8.0, bottom])
        .default_width(WIDTH)
        .resizable(false)
        .collapsible(true)
        .show(ctx, |ui| {
            let Some(scan) = scan else {
                ui.label(format!("agent {agent}: no LiDAR scan"));
                return;
            };
            let config = scan.config;
            let layout = match &config.pattern {
                BeamPattern::Rings { elevations, azimuths, .. } => format!("{} × {azimuths} beams", elevations.len()),
                BeamPattern::Beams { directions } => format!("{} beams", directions.len()),
            };
            ui.label(format!("{layout} · {:.0} m · {} Hz", config.max_range, config.rate_hz));
            let returns: Vec<f64> = scan.ranges.iter().flatten().copied().collect();
            let nearest = returns.iter().copied().fold(f64::INFINITY, f64::min);
            let nearest = if nearest.is_finite() { format!(" · nearest {nearest:.1} m") } else { String::new() };
            ui.label(format!("{} of {} returns{nearest}", returns.len(), scan.ranges.len()));
            top_view(ui, &scan, goal);
            range_image_view(ui, &scan);
        });
    Ok(())
}

/// Returns around the sensor seen from above, heading up, with range rings and the current
/// goal (on the edge when out of range).
fn top_view(ui: &mut egui::Ui, scan: &Scan, goal: Option<DVec3>) {
    let (response, painter) = ui.allocate_painter(egui::vec2(WIDTH, WIDTH), egui::Sense::hover());
    let rect = response.rect;
    let center = rect.center();
    let max = scan.config.max_range;
    let scale = (0.5 * WIDTH - 6.0) / max as f32;
    painter.rect_filled(rect, 4.0, BACKGROUND);
    let step = ring_step(max);
    let mut d = step;
    while d <= max + 1e-9 {
        painter.circle_stroke(center, d as f32 * scale, egui::Stroke::new(1.0, egui::Color32::from_gray(60)));
        d += step;
    }
    let small = egui::FontId::proportional(11.0);
    painter.text(
        rect.left_top() + egui::vec2(5.0, 3.0),
        egui::Align2::LEFT_TOP,
        format!("rings {step} m"),
        small,
        egui::Color32::GRAY,
    );

    // Heading frame: yaw only, so the view does not tilt with the vehicle. Ahead is up and
    // left is left.
    let heading = DQuat::from_rotation_z(-yaw(scan.pose.rot));
    let to_heading = heading * scan.pose.rot;
    let to_screen = |p: DVec3| center + egui::vec2(-p.y as f32, -p.x as f32) * scale;
    let hits: Vec<(DVec3, f64)> =
        scan.ranges.iter().zip(scan.directions).filter_map(|(r, d)| r.map(|r| (to_heading * (*d * r), r))).collect();
    let stride = hits.len().div_ceil(MAX_POINTS).max(1);
    let radius = if hits.len() <= 512 { 2.5 } else { 1.5 };
    for &(p, r) in hits.iter().step_by(stride) {
        painter.circle_filled(to_screen(p), radius, color32(range_color(r, max)));
    }

    let s = 6.0;
    let nose = [egui::vec2(0.0, -1.4 * s), egui::vec2(-0.8 * s, s), egui::vec2(0.8 * s, s)];
    painter.add(egui::Shape::convex_polygon(
        nose.map(|v| center + v).to_vec(),
        egui::Color32::WHITE,
        egui::Stroke::NONE,
    ));

    if let Some(goal) = goal {
        let rel = (heading * (goal - scan.pose.pos)).truncate();
        let distance = rel.length();
        let inside = distance <= max;
        let p = to_screen((if inside { rel } else { rel * (max / distance) }).extend(0.0));
        if inside {
            painter.circle_filled(p, 5.0, GOAL);
        } else {
            painter.circle_stroke(p, 5.0, egui::Stroke::new(2.0, GOAL));
        }
    }
}

/// The range image of a ring pattern, with the directions marked below it. It is drawn as
/// coloured cells rather than a texture: bevy_egui turns every full texture update into a new
/// image asset, which is missing from the frame it is created in, so the image flickered with
/// each scan. Beams narrower than a point share a cell, which shows the nearest return.
fn range_image_view(ui: &mut egui::Ui, scan: &Scan) {
    let BeamPattern::Rings { azimuth_fov, .. } = scan.config.pattern else {
        ui.label("explicit beams: no range image");
        return;
    };
    let Some(([cols, rows], ranges)) = range_image(&scan.config.pattern, &scan.ranges) else { return };
    let max = scan.config.max_range;
    let row = (80.0 / rows as f32).clamp(3.0, 10.0);
    let height = row * rows as f32;
    let (response, painter) = ui.allocate_painter(egui::vec2(WIDTH, height + 16.0), egui::Sense::hover());
    let image = egui::Rect::from_min_size(response.rect.min, egui::vec2(WIDTH, height));
    let cells = cols.min(WIDTH as usize);
    let width = WIDTH / cells as f32;
    let mut mesh = egui::Mesh::default();
    for (y, ring) in ranges.chunks(cols).enumerate() {
        for cell in 0..cells {
            let beams = &ring[cell * cols / cells..(cell + 1) * cols / cells];
            let nearest = beams.iter().flatten().copied().reduce(f64::min);
            let color = nearest.map_or(NO_RETURN, |r| color32(range_color(r, max)));
            let min = image.min + egui::vec2(cell as f32 * width, y as f32 * row);
            mesh.add_colored_rect(egui::Rect::from_min_size(min, egui::vec2(width, row)), color);
        }
    }
    painter.add(egui::Shape::mesh(mesh));

    // Marks at the centres of the columns of these azimuths (degrees, to the left of ahead).
    let n = cols as f64;
    let full = (azimuth_fov - 360.0).abs() < 1e-9;
    let half = 0.5 * azimuth_fov;
    let column = |a: f64| {
        if full {
            0.5 * n - a * n / 360.0
        } else if cols > 1 {
            n - 1.0 - (a + half) * (n - 1.0) / azimuth_fov
        } else {
            0.0
        }
    };
    let marks: Vec<(f64, String)> = if full {
        [(180.0, "behind"), (90.0, "left"), (0.0, "ahead"), (-90.0, "right"), (-180.0, "behind")]
            .map(|(a, s)| (a, s.to_owned()))
            .to_vec()
    } else {
        vec![(half, format!("left {half:.0}°")), (0.0, "ahead".to_owned()), (-half, format!("right {half:.0}°"))]
    };
    let small = egui::FontId::proportional(11.0);
    for (a, label) in marks {
        let x = ((column(a) + 0.5) / n).clamp(0.0, 1.0) as f32;
        let top = egui::pos2(image.left() + x * WIDTH, image.bottom());
        painter.line_segment([top, top + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.0, egui::Color32::GRAY));
        let align = match x {
            x if x < 0.1 => egui::Align2::LEFT_TOP,
            x if x > 0.9 => egui::Align2::RIGHT_TOP,
            _ => egui::Align2::CENTER_TOP,
        };
        painter.text(top + egui::vec2(0.0, 3.0), align, label, small.clone(), egui::Color32::GRAY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_control::multirotor::ActionMode;
    use autonomousim_core::rng::Seed;
    use autonomousim_sensors::{LidarConfig, SensorConfig, SensorSpec};
    use autonomousim_sim::scenario::{MapSource, Testworld, VehicleRef};
    use autonomousim_sim::{GroupSpec, Scenario, WorldInstance};
    use std::sync::Arc;

    /// Every pixel names its beam, so the layout can be checked against the beam directions.
    fn beams(pattern: &BeamPattern) -> ([usize; 2], Vec<usize>, Vec<DVec3>) {
        let directions = pattern.directions();
        let ranges: Vec<Option<f64>> = (0..directions.len()).map(|i| Some(i as f64)).collect();
        let (size, image) = range_image(pattern, &ranges).unwrap();
        (size, image.into_iter().map(|r| r.unwrap() as usize).collect(), directions)
    }

    #[test]
    fn full_turn_range_image_has_ahead_in_the_middle_and_the_highest_ring_on_top() {
        let pattern = BeamPattern::Rings { elevations: vec![-10.0, 20.0, 5.0], azimuths: 8, azimuth_fov: 360.0 };
        let (size, image, dirs) = beams(&pattern);
        assert_eq!(size, [8, 3]);
        let pixel = |row: usize, col: usize| dirs[image[row * 8 + col]];
        // Rows from the highest ring down.
        for (row, el) in [20.0f64, 5.0, -10.0].into_iter().enumerate() {
            assert!((pixel(row, 0).z - el.to_radians().sin()).abs() < 1e-12);
        }
        // Ahead in the middle, the left side left of it, the right side right of it, behind
        // at the edges.
        assert!(pixel(0, 4).x > 0.9 && pixel(0, 4).y.abs() < 1e-9);
        assert!(pixel(0, 2).y > 0.9 && pixel(0, 6).y < -0.9);
        assert!(pixel(0, 0).x < -0.9 && pixel(0, 3).y > 0.0 && pixel(0, 5).y < 0.0);
    }

    #[test]
    fn partial_range_image_runs_from_the_left_edge_to_the_right() {
        let pattern = BeamPattern::Rings { elevations: vec![0.0], azimuths: 5, azimuth_fov: 90.0 };
        let (size, image, dirs) = beams(&pattern);
        assert_eq!(size, [5, 1]);
        let az = |col: usize| dirs[image[col]].y.atan2(dirs[image[col]].x).to_degrees();
        let angles: Vec<f64> = (0..5).map(az).collect();
        for (a, b) in angles.iter().zip([45.0, 22.5, 0.0, -22.5, -45.0]) {
            assert!((a - b).abs() < 1e-9, "{angles:?}");
        }
    }

    #[test]
    fn explicit_beams_and_mismatched_scans_have_no_range_image() {
        let beams = BeamPattern::Beams { directions: vec![DVec3::X, DVec3::Y] };
        assert!(range_image(&beams, &[None, None]).is_none());
        let rings = BeamPattern::rings(2, -10.0, 10.0, 4, 360.0);
        assert!(range_image(&rings, &[None; 7]).is_none());
        assert_eq!(ring_step(40.0), 10.0);
        assert_eq!(ring_step(100.0), 25.0);
    }

    /// Live, the scan comes from the pilot's sensor: over flat ground the rings below the
    /// horizon (the bottom rows) all see the ground and the ones above it nothing.
    #[test]
    fn live_scan_over_flat_ground() {
        let lidar = SensorSpec { name: "lidar".into(), config: SensorConfig::Lidar(LidarConfig::rl64()) };
        let sc = Scenario {
            map: MapSource::Testworld(Testworld::Flat { size: 200.0 }),
            groups: vec![GroupSpec {
                vehicle: VehicleRef::Name("iris_like".into()),
                action_mode: Some(ActionMode::Velocity.into()),
                sensors: vec![lidar],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut sim = Sim::new(WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(1)));
        for _ in 0..12 {
            sim.advance(1.0 / 60.0);
        }
        let scan = latest_scan(&sim, 0).unwrap();
        let (size, image) = range_image(&scan.config.pattern, &scan.ranges).unwrap();
        assert_eq!(size, [16, 4]);
        let (above, below) = image.split_at(32);
        assert!(above.iter().all(Option::is_none) && below.iter().all(Option::is_some), "{image:?}");
        // Each return lies on the ground (z = 0), whatever the attitude.
        for (r, d) in scan.ranges.iter().zip(scan.directions) {
            if let Some(r) = r {
                let z = scan.pose.transform_point(*d * *r).z;
                assert!(z.abs() < 0.05, "return {r} m along {d} ends at z = {z}");
            }
        }
    }
}
