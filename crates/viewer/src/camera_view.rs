//! The followed agent's camera sensor (I): the image its policy sees, as RGB, depth or
//! semantic classes (K cycles), and its view frustum in the scene. Live, the panel shows the
//! sensor's visible frame (noise and latency included); in a replay the camera is rendered
//! again from the recorded state at the playback time.

use crate::convert::RenderOrigin;
use crate::hud::Hud;
use crate::sim::Sim;
use autonomousim_core::geometry::{HitMask, Ray};
use autonomousim_core::math::Pose;
use autonomousim_render::{Intrinsics, SemanticClass};
use autonomousim_sensors::{CameraConfig, CameraImage, Sensor};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

/// Width of the image (points).
const WIDTH: f32 = 256.0;
/// Pixels drawn per axis at most (larger images are subsampled).
const MAX_CELLS: u32 = 128;
/// The frustum's rays end on the ground or obstacles, or at this distance (m).
const FRUSTUM_RANGE: f64 = 40.0;
const FRUSTUM: Color = Color::srgb(1.0, 0.55, 0.1);
const SKY: egui::Color32 = egui::Color32::from_rgb(20, 24, 48);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Output {
    #[default]
    Rgb,
    Depth,
    Semantic,
}

impl Output {
    const ALL: [Self; 3] = [Self::Rgb, Self::Depth, Self::Semantic];

    fn name(self) -> &'static str {
        match self {
            Self::Rgb => "RGB",
            Self::Depth => "depth",
            Self::Semantic => "semantic",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Rgb => Self::Depth,
            Self::Depth => Self::Semantic,
            Self::Semantic => Self::Rgb,
        }
    }
}

#[derive(Resource, Default)]
pub struct CameraView {
    pub visible: bool,
    pub output: Output,
    /// Which of the followed agent's cameras (in sensor order).
    pub camera: usize,
}

/// The followed agent's cameras: sensor index, name and configuration.
fn cameras(sim: &Sim, agent: usize) -> Vec<(usize, String, CameraConfig)> {
    let a = sim.world.agent(agent);
    let spec = &sim.world.scenario().groups[a.group].spec;
    a.sensors
        .iter()
        .enumerate()
        .filter_map(|(k, s)| match s {
            Sensor::Camera(c) => Some((k, spec.sensors[k].name.clone(), c.config().clone())),
            _ => None,
        })
        .collect()
}

/// Colour of a semantic class for display.
pub fn class_color(class: u8) -> egui::Color32 {
    const PALETTE: [[u8; 3]; 15] = [
        [20, 24, 48],    // sky
        [96, 160, 64],   // grass
        [70, 100, 40],   // forest floor
        [128, 128, 128], // rock
        [150, 110, 70],  // soil
        [240, 240, 250], // snow
        [40, 90, 200],   // water
        [60, 60, 66],    // road
        [110, 70, 40],   // trunk
        [30, 140, 60],   // canopy
        [170, 150, 130], // boulder
        [200, 60, 60],   // building
        [255, 220, 0],   // own vehicle
        [230, 40, 200],  // vehicle
        [255, 128, 0],   // marker
    ];
    let [r, g, b] = PALETTE.get(usize::from(class)).copied().unwrap_or([255, 255, 255]);
    egui::Color32::from_rgb(r, g, b)
}

/// Depth shade: white near, dark far over `[0, max]`; sky (0) dark blue.
fn depth_color(d: f32, max: f32) -> egui::Color32 {
    if d <= 0.0 {
        return SKY;
    }
    let v = (255.0 * (1.0 - (d / max).clamp(0.0, 1.0)).powf(1.5)).round() as u8;
    egui::Color32::from_gray(v.max(16))
}

/// Draw `image` in `output` as cells of at most [`MAX_CELLS`] per axis. Drawn as a mesh
/// rather than a texture: bevy_egui turns every texture update into a new image asset, which
/// flickers (see the LiDAR view).
fn image_view(ui: &mut egui::Ui, image: &CameraImage, output: Output) -> f32 {
    let (w, h) = (image.width.max(1), image.height.max(1));
    let size = egui::vec2(WIDTH, WIDTH * h as f32 / w as f32);
    let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let rect = response.rect;
    let (cols, rows) = (w.min(MAX_CELLS), h.min(MAX_CELLS));
    let cell = egui::vec2(size.x / cols as f32, size.y / rows as f32);
    let max_depth = image.depth.iter().copied().fold(0.0f32, f32::max).max(1.0);
    let mut mesh = egui::Mesh::default();
    for r in 0..rows {
        for c in 0..cols {
            let (x, y) = ((c * w / cols) as usize, (r * h / rows) as usize);
            let i = y * w as usize + x;
            let color = match output {
                Output::Rgb => {
                    let p = &image.rgb[3 * i..3 * i + 3];
                    egui::Color32::from_rgb(p[0], p[1], p[2])
                }
                Output::Depth => depth_color(image.depth[i], max_depth),
                Output::Semantic => class_color(image.class[i]),
            };
            let min = rect.min + egui::vec2(c as f32 * cell.x, r as f32 * cell.y);
            mesh.add_colored_rect(egui::Rect::from_min_size(min, cell), color);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
    max_depth
}

pub fn camera_view(
    mut contexts: EguiContexts,
    mut view: ResMut<CameraView>,
    hud: Res<Hud>,
    mut sim: ResMut<Sim>,
    keys: Res<ButtonInput<KeyCode>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    if !ctx.egui_wants_keyboard_input() {
        if keys.just_pressed(KeyCode::KeyI) {
            view.visible = !view.visible;
        }
        if keys.just_pressed(KeyCode::KeyK) {
            view.output = view.output.next();
        }
    }
    if !view.visible || !hud.visible {
        return Ok(());
    }
    let agent = sim.pilot;
    let list = cameras(&sim, agent);
    view.camera = view.camera.min(list.len().saturating_sub(1));
    let chosen = list.get(view.camera).cloned();
    let image = chosen.as_ref().and_then(|(k, ..)| sim.camera_image(agent, *k).map(|(t, i)| (t, i.clone())));
    let paused = sim.paused;
    // Above the replay timeline.
    let bottom = if sim.replay.is_some() { -84.0 } else { -8.0 };
    egui::Window::new("Camera")
        .anchor(egui::Align2::LEFT_BOTTOM, [8.0, bottom])
        .default_width(WIDTH)
        .resizable(false)
        .collapsible(true)
        .show(ctx, |ui| {
            let Some((_, name, config)) = &chosen else {
                ui.label(format!("agent {agent}: no camera"));
                return;
            };
            ui.horizontal(|ui| {
                if list.len() > 1 {
                    egui::ComboBox::from_id_salt("camera sensor").selected_text(name.as_str()).show_ui(ui, |ui| {
                        for (k, (_, n, _)) in list.iter().enumerate() {
                            ui.selectable_value(&mut view.camera, k, n.as_str());
                        }
                    });
                } else {
                    ui.label(name.as_str());
                }
                for o in Output::ALL {
                    ui.selectable_value(&mut view.output, o, o.name());
                }
            });
            ui.label(format!(
                "{}×{} · {:.0}° · {} Hz · latency {} frame{}",
                config.width,
                config.height,
                config.fov_deg,
                config.rate_hz,
                config.latency,
                if config.latency == 1 { "" } else { "s" }
            ));
            let Some((time, image)) = &image else {
                ui.label(if paused { "no frame yet (paused)" } else { "no frame yet" });
                return;
            };
            let max_depth = image_view(ui, image, view.output);
            match view.output {
                Output::Rgb => ui.label(format!("frame of t = {time:.2} s")),
                Output::Depth => ui.label(format!("frame of t = {time:.2} s · white 0 m → dark {max_depth:.0} m")),
                Output::Semantic => {
                    let mut seen = [false; 256];
                    image.class.iter().for_each(|&c| seen[usize::from(c)] = true);
                    ui.horizontal_wrapped(|ui| {
                        for class in SemanticClass::ALL.iter().filter(|c| seen[**c as usize]) {
                            ui.colored_label(class_color(*class as u8), class.name());
                        }
                    })
                    .response
                }
            };
        });
    Ok(())
}

/// The frustum of the shown camera, from the followed agent's drawn pose to the ground (or
/// [`FRUSTUM_RANGE`]).
pub fn draw_frustum(
    view: Res<CameraView>,
    hud: Res<Hud>,
    sim: Res<Sim>,
    origin: Res<RenderOrigin>,
    mut gizmos: Gizmos,
) {
    if !view.visible || !hud.visible {
        return;
    }
    let agent = sim.pilot;
    let a = sim.world.agent(agent);
    if a.disabled {
        return;
    }
    let Some((_, _, config)) = cameras(&sim, agent).into_iter().nth(view.camera) else { return };
    // The camera relative to the body, moved with the drawn (interpolated) pose.
    let body = a.vehicle.pose();
    let mount = config.mount.world_pose(&a.kinematics());
    let rel = Pose::new(body.rot.inverse() * (mount.pos - body.pos), body.rot.inverse() * mount.rot);
    let drawn = sim.render_pose(agent);
    let cam = Pose::new(drawn.pos + drawn.rot * rel.pos, drawn.rot * rel.rot);
    let intrinsics = Intrinsics::new(config.width, config.height, config.fov_deg.to_radians());
    let (w, h) = (f64::from(config.width), f64::from(config.height));
    let map = sim.world.map();
    let corners: Vec<Vec3> = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        .into_iter()
        .map(|(u, v)| {
            let dir = cam.rot * intrinsics.ray(u, v).normalize();
            let reach = map
                .raycast(&Ray::new(cam.pos, dir), FRUSTUM_RANGE, HitMask::TERRAIN | HitMask::WATER | HitMask::SOLID)
                .map_or(FRUSTUM_RANGE, |hit| hit.toi);
            origin.pos(cam.pos + dir * reach)
        })
        .collect();
    let apex = origin.pos(cam.pos);
    for (k, &c) in corners.iter().enumerate() {
        gizmos.line(apex, c, FRUSTUM);
        gizmos.line(c, corners[(k + 1) % 4], FRUSTUM);
    }
}
