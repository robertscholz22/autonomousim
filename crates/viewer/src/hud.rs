//! Head-up display: flight data, events, rotor speeds (or a ground vehicle's powertrain,
//! pedals and per-wheel load, slip and force), simulation controls and key help; a map seed
//! control (live), a timeline (replay), and plots of the followed agent. The LiDAR view is in
//! `lidar_view`.

use crate::Regenerate;
use crate::camera::CameraRig;
use crate::history::History;
use crate::sim::Sim;
use crate::world_view::MapView;
use autonomousim_core::math::quat::yaw;
use autonomousim_sim::Events;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, Points, VLine};

#[derive(Resource)]
pub struct Hud {
    pub visible: bool,
    pub help: bool,
    pub plots: bool,
}

const HELP: &str = "W/S  forward/back     A/D  left/right\n\
                    Space/Shift  up/down  Q/E  yaw\n\
                    M  pilot mode         -/=  max speed\n\
                    R  reset episode      Tab  next agent\n\
                    P  pause   [/]  time scale   C  camera\n\
                    mouse drag  look      wheel  zoom\n\
                    O  goals/trails       G  plots\n\
                    L  LiDAR hits         V  LiDAR view\n\
                    H  hide HUD   F1  help   Esc  quit";

const GROUND_HELP: &str = "W/S  throttle / brake, reverse   A/D  steer\n\
                           Space  handbrake      R  reset episode\n\
                           Tab  next agent       P  pause   [/]  time scale\n\
                           C  camera             mouse drag  look   wheel  zoom\n\
                           O  goals/trails       G  plots\n\
                           L  LiDAR hits         V  LiDAR view\n\
                           H  hide HUD   F1  help   Esc  quit\n\
                           gamepad: RT/LT pedals, left stick steers, A handbrake";

/// Shown above [`HELP`] while a policy flies.
const POLICY_HELP: &str = "T  take over the followed agent / hand it back";

const REPLAY_HELP: &str = "P  play/pause         [/]  speed\n\
                           ←/→  ∓1 s   Shift+←/→  one sample\n\
                           N/B  next/previous episode\n\
                           Home/R  start of episode\n\
                           Tab  next agent       C  camera\n\
                           mouse drag  look      wheel  zoom\n\
                           O  goals/trails       G  plots\n\
                           L  LiDAR hits         V  LiDAR view\n\
                           H  hide HUD   F1  help   Esc  quit";

/// Colours of the x, y and z components in the plots.
const AXES: [egui::Color32; 3] = [
    egui::Color32::from_rgb(230, 90, 80),
    egui::Color32::from_rgb(110, 200, 90),
    egui::Color32::from_rgb(90, 150, 240),
];

#[allow(clippy::too_many_arguments)]
pub fn hud(
    mut contexts: EguiContexts,
    mut hud: ResMut<Hud>,
    mut sim: ResMut<Sim>,
    regen: Option<ResMut<Regenerate>>,
    history: Res<History>,
    view: Res<MapView>,
    keys: Res<ButtonInput<KeyCode>>,
    diagnostics: Res<DiagnosticsStore>,
    camera: Query<&CameraRig>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    if !ctx.egui_wants_keyboard_input() {
        if keys.just_pressed(KeyCode::KeyH) {
            hud.visible = !hud.visible;
        }
        if keys.just_pressed(KeyCode::F1) {
            hud.help = !hud.help;
        }
        if keys.just_pressed(KeyCode::KeyG) {
            hud.plots = !hud.plots;
        }
    }
    if !hud.visible {
        return Ok(());
    }
    let fps = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS).and_then(|d| d.smoothed()).unwrap_or(0.0);
    let camera_mode = camera.single().map(|c| c.mode.name()).unwrap_or("-");
    status_window(ctx, &hud, &sim, regen, &view, fps, camera_mode);
    if sim.replay.is_some() {
        timeline(ctx, &mut sim);
    }
    if hud.plots {
        plots(ctx, &sim, &history);
    }
    Ok(())
}

fn status_window(
    ctx: &egui::Context,
    hud: &Hud,
    sim: &Sim,
    regen: Option<ResMut<Regenerate>>,
    view: &MapView,
    fps: f64,
    camera_mode: &str,
) {
    let world = &sim.world;
    let agent = world.agent(sim.pilot);
    let v = &agent.vehicle;
    let vel = v.lin_vel_world();
    let pos = v.position();
    let meta = &view.world.meta;
    egui::Window::new("autonomousim")
        .anchor(egui::Align2::LEFT_TOP, [8.0, 8.0])
        .resizable(false)
        .collapsible(true)
        .show(ctx, |ui| {
            ui.label(format!("map {} · seed {} · {} maps in pool", meta.name, meta.seed, world.scenario().maps.len()));
            let (episode, episodes) = sim.episode();
            let of = episodes.map_or(String::new(), |n| format!(" of {n}"));
            ui.label(format!("agent {} ({}) · episode {episode}{of}", sim.pilot, v.name()));
            if let Some(a) = &sim.autopilot {
                ui.label(format!("policy {}", a.name));
                let who = if sim.manual_agent().is_some() { "you fly · T: hand back" } else { "T: take over" };
                ui.label(who);
                if a.flown > 0 {
                    ui.label(format!(
                        "{} of {} episodes reached the last goal ({:.0} %)",
                        a.finished,
                        a.flown,
                        100.0 * a.finished as f64 / a.flown as f64
                    ));
                }
            }
            if let Some(mut regen) = regen
                && let Some(current) = regen.map_seed()
            {
                ui.horizontal(|ui| {
                    let mut seed = regen.seed;
                    ui.label("map seed");
                    ui.add(egui::DragValue::new(&mut seed).speed(0.2));
                    regen.seed = seed;
                    if regen.pending.is_some() {
                        ui.spinner();
                        ui.label("generating…");
                    } else if ui
                        .add_enabled(seed != current && !sim.is_recording(), egui::Button::new("generate"))
                        .on_disabled_hover_text("the recording holds this map")
                        .clicked()
                    {
                        regen.start(seed);
                    }
                });
                if let Some(e) = &regen.error {
                    ui.colored_label(egui::Color32::from_rgb(230, 80, 60), e);
                }
            }
            ui.separator();
            egui::Grid::new("flight").num_columns(2).show(ui, |ui| {
                let row = |ui: &mut egui::Ui, k: &str, v: String| {
                    ui.label(k);
                    ui.monospace(v);
                    ui.end_row();
                };
                row(ui, "time", format!("{:8.2} s", sim.time()));
                row(ui, "position", format!("{:7.1} {:7.1} {:6.1} m", pos.x, pos.y, pos.z));
                if let Some(w) = v.as_wheeled() {
                    let forward = w.lin_vel_body().x;
                    row(ui, "speed", format!("{:5.1} km/h  ({forward:+5.1} m/s)", 3.6 * vel.length()));
                } else {
                    row(ui, "height", format!("{:6.1} m AGL", agent.agl_now(world.map())));
                    row(ui, "speed", format!("{:5.1} m/s  climb {:+5.1}", vel.truncate().length(), vel.z));
                }
                row(ui, "heading", format!("{:5.0}°", yaw(v.orientation()).to_degrees()));
                if !agent.goals.is_empty() {
                    let g = agent.goal();
                    let n = agent.goals.len();
                    let which =
                        if n > 1 { format!("  ({} of {n})", agent.goal_index.min(n - 1) + 1) } else { String::new() };
                    row(ui, "goal", format!("{:6.2} m{which}", (pos - g.position).length()));
                }
                if sim.replay.is_none() && sim.manual_agent().is_none() {
                    let a: Vec<String> = agent.action.iter().map(|x| format!("{x:+.2}")).collect();
                    row(ui, "policy", a.join(" "));
                } else if sim.replay.is_none() && v.as_wheeled().is_some() {
                    let hb = if sim.handbrake { "  handbrake" } else { "" };
                    let who = if sim.drive_command.is_some() { "demo" } else { "keys" };
                    row(ui, "driver", format!("{who}  pedal {:+.1}  steer {:+.2}{hb}", sim.stick[0], sim.steer));
                } else if sim.replay.is_none() {
                    let [f, l, u, y] = sim.stick;
                    row(ui, "pilot", format!("{}  {f:+.1} {l:+.1} {u:+.1} {y:+.1}", sim.pilot_mode.name()));
                    row(ui, "max speed", format!("{:4.1} m/s", sim.max_speed));
                }
            });
            if let Some(m) = v.as_multirotor() {
                let (_, max) = m.speed_range();
                ui.horizontal(|ui| {
                    ui.label("rotors");
                    for &w in m.motor_speeds() {
                        ui.add(egui::ProgressBar::new((w / max) as f32).desired_width(40.0));
                    }
                });
            }
            if let Some(w) = v.as_wheeled() {
                ground_status(ui, sim, w);
            }
            let latched = sim.latched[sim.pilot];
            let now = agent.events;
            let names: Vec<&str> = latched.names().collect();
            let text = if names.is_empty() { "–".to_owned() } else { names.join(", ") };
            let color = if latched.intersects(Events::TERMINAL) {
                egui::Color32::from_rgb(230, 80, 60)
            } else if now.is_empty() {
                ui.visuals().text_color()
            } else {
                egui::Color32::from_rgb(230, 190, 60)
            };
            ui.horizontal(|ui| {
                ui.label("events");
                ui.colored_label(color, text);
            });
            ui.separator();
            let state = match (sim.paused, sim.replay.is_some()) {
                (true, _) => "paused",
                (false, true) => "playing",
                (false, false) => "running",
            };
            ui.label(format!(
                "{state} · ×{} · real time ×{:.2} · {fps:.0} fps · camera {camera_mode}",
                sim.time_scale, sim.real_time_factor
            ));
            if sim.is_recording() {
                ui.colored_label(egui::Color32::from_rgb(230, 80, 60), "● recording");
            }
            if hud.help {
                ui.separator();
                if sim.autopilot.is_some() {
                    ui.monospace(POLICY_HELP);
                }
                let help = match (sim.replay.is_some(), v.as_wheeled().is_some()) {
                    (true, _) => REPLAY_HELP,
                    (false, true) => GROUND_HELP,
                    (false, false) => HELP,
                };
                ui.monospace(help);
            }
        });
}

/// Wheel names: FL, FR, RL, RR for two axles, else axle number and side.
fn wheel_name(w: usize, axles: usize) -> String {
    let side = if w.is_multiple_of(2) { "L" } else { "R" };
    match (axles, w / 2) {
        (2, 0) => format!("F{side}"),
        (2, _) => format!("R{side}"),
        (_, a) => format!("{}{side}", a + 1),
    }
}

/// Gear, engine speed, steering and the driver's pedals, and a table of the wheels: tyre load,
/// suspension travel, slip, forces and torques.
fn ground_status(ui: &mut egui::Ui, sim: &Sim, w: &autonomousim_vehicles::ground::Wheeled) {
    let agent = sim.world.agent(sim.pilot);
    let p = w.powertrain();
    let gear = match p.gear {
        0 => "–".to_owned(),
        -1 => "R".to_owned(),
        g => g.to_string(),
    };
    ui.label(format!(
        "steering {:+5.1}° · gear {gear} · {:5.0} rpm",
        w.steering_angle().to_degrees(),
        p.engine_speed * 30.0 / std::f64::consts::PI
    ));
    if sim.replay.is_none()
        && let Some(c) = agent.controller.as_ground()
    {
        let input = c.last_input();
        ui.horizontal(|ui| {
            ui.label("throttle");
            ui.add(egui::ProgressBar::new(input.throttle.abs() as f32).desired_width(60.0));
            ui.label("brake");
            ui.add(egui::ProgressBar::new(input.brake as f32).desired_width(60.0));
            if input.parking {
                ui.label("P");
            }
        });
    }
    let axles = w.def().axles.len();
    egui::Grid::new("wheels").num_columns(8).striped(true).show(ui, |ui| {
        for h in ["", "load kN", "travel mm", "κ", "α °", "Fx kN", "Fy kN", "drive/brake N·m"] {
            ui.label(egui::RichText::new(h).small());
        }
        ui.end_row();
        for (k, s) in w.wheels().enumerate() {
            let t = &s.tire;
            ui.label(wheel_name(k, axles));
            ui.monospace(format!("{:5.2}", t.fz / 1e3));
            ui.monospace(format!("{:+5.0}", s.travel * 1e3));
            ui.monospace(format!("{:+5.2}", t.kappa));
            ui.monospace(format!("{:+5.1}", t.tan_alpha.atan().to_degrees()));
            ui.monospace(format!("{:+5.2}", t.fx / 1e3));
            ui.monospace(format!("{:+5.2}", t.fy / 1e3));
            ui.monospace(format!("{:+5.0}/{:4.0}", s.drive_torque, s.brake_torque.abs()));
            ui.end_row();
        }
    });
}

/// Replay controls at the bottom: play/pause, episode, time slider and speed.
fn timeline(ctx: &egui::Context, sim: &mut Sim) {
    let width = (ctx.content_rect().width() - 32.0).clamp(300.0, 900.0);
    let mut paused = sim.paused;
    let mut time_scale = sim.time_scale;
    let Some(r) = sim.replay.as_mut() else { return };
    egui::Window::new("timeline")
        .title_bar(false)
        .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -8.0])
        .resizable(false)
        .fixed_size([width, 0.0])
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("⏮").on_hover_text("previous episode (B)").clicked() {
                    let n = r.recording.episodes.len();
                    r.set_episode((r.episode + n - 1) % n);
                }
                if ui.button(if paused { "▶" } else { "⏸" }).on_hover_text("play/pause (P)").clicked() {
                    paused = !paused;
                }
                if ui.button("⏭").on_hover_text("next episode (N)").clicked() {
                    let n = r.recording.episodes.len();
                    r.set_episode((r.episode + 1) % n);
                }
                let mut episode = r.episode;
                egui::ComboBox::from_id_salt("episode").selected_text(format!("episode {}", episode + 1)).show_ui(
                    ui,
                    |ui| {
                        for (i, ep) in r.recording.episodes.iter().enumerate() {
                            ui.selectable_value(&mut episode, i, format!("{} · {:.1} s", i + 1, ep.duration()));
                        }
                    },
                );
                if episode != r.episode {
                    r.set_episode(episode);
                }
                ui.monospace(format!("{:6.2} / {:.2} s", r.time, r.duration()));
                egui::ComboBox::from_id_salt("speed").selected_text(format!("×{time_scale}")).width(60.0).show_ui(
                    ui,
                    |ui| {
                        for s in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] {
                            ui.selectable_value(&mut time_scale, s, format!("×{s}"));
                        }
                    },
                );
                ui.checkbox(&mut r.looping, "loop");
            });
            let mut t = r.time;
            let duration = r.duration();
            ui.spacing_mut().slider_width = width - 16.0;
            if ui.add(egui::Slider::new(&mut t, 0.0..=duration).show_value(false)).changed() {
                r.seek(t);
            }
        });
    sim.paused = paused;
    sim.time_scale = time_scale;
}

fn line<'a>(name: &str, points: Vec<[f64; 2]>, color: egui::Color32) -> Line<'a> {
    Line::new(name, PlotPoints::from(points)).color(color)
}

/// Colours of the wheels in the tyre plots.
const WHEELS: [egui::Color32; 8] = [
    egui::Color32::from_rgb(230, 90, 80),
    egui::Color32::from_rgb(240, 170, 60),
    egui::Color32::from_rgb(90, 150, 240),
    egui::Color32::from_rgb(110, 200, 90),
    egui::Color32::from_rgb(190, 110, 220),
    egui::Color32::from_rgb(80, 200, 200),
    egui::Color32::from_rgb(200, 200, 90),
    egui::Color32::from_rgb(200, 200, 200),
];

/// Height (sideslip for ground vehicles) and goal distance, and setpoint against the
/// measured value, over time; for ground vehicles also every tyre's force over its load
/// against its slip.
fn plots(ctx: &egui::Context, sim: &Sim, history: &History) {
    let samples = &history.samples;
    let now = sim.time();
    let cursor = sim.replay.is_some();
    let series = |f: &dyn Fn(&crate::history::Sample) -> f64| -> Vec<[f64; 2]> {
        samples.iter().map(|s| [s.time, f(s)]).filter(|p| p[1].is_finite()).collect()
    };
    egui::Window::new("plots")
        .anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0])
        .default_width(420.0)
        .collapsible(true)
        .show(ctx, |ui| {
            let ground = history.tracking == crate::history::Tracking::Ground;
            Plot::new("height").height(150.0).legend(Legend::default()).link_axis("t", [true, false]).show(ui, |p| {
                if ground {
                    p.line(line("sideslip (°)", series(&|s| s.sideslip), AXES[2]));
                } else {
                    p.line(line("height AGL (m)", series(&|s| s.agl), AXES[2]));
                }
                p.line(line("goal distance (m)", series(&|s| s.goal_distance), egui::Color32::from_rgb(240, 200, 40)));
                if cursor {
                    p.vline(VLine::new("now", now).color(egui::Color32::GRAY));
                }
            });
            let tracking = history.tracking;
            ui.label(format!("{}: setpoint dashed, measured solid", tracking.label()));
            Plot::new("tracking").height(170.0).legend(Legend::default()).link_axis("t", [true, false]).show(ui, |p| {
                for (k, name) in tracking.axes().iter().enumerate() {
                    let actual = series(&|s| s.actual[k]);
                    let command = series(&|s| s.command[k]);
                    if !actual.is_empty() {
                        p.line(line(name, actual, AXES[k]));
                    }
                    if !command.is_empty() {
                        p.line(line(name, command, AXES[k]).style(LineStyle::dashed_dense()));
                    }
                }
                if cursor {
                    p.vline(VLine::new("now", now).color(egui::Color32::GRAY));
                }
            });
            if ground {
                tyre_plots(ui, sim, history);
            }
        });
}

/// Force over load against slip for every tyre: lateral against the slip angle, longitudinal
/// against κ. Live: the history window; replay: the episode up to the playback time.
fn tyre_plots(ui: &mut egui::Ui, sim: &Sim, history: &History) {
    let now = sim.time();
    let upto: Vec<&crate::history::Sample> = history.samples.iter().filter(|s| s.time <= now + 1e-9).collect();
    let n = upto.iter().map(|s| s.wheels.len()).max().unwrap_or(0);
    let axles = n / 2;
    let scatter = |x: fn(&crate::history::WheelSample) -> f64, y: fn(&crate::history::WheelSample) -> f64| {
        let upto = &upto;
        move |p: &mut egui_plot::PlotUi| {
            for k in 0..n {
                let pts: Vec<[f64; 2]> = upto
                    .iter()
                    .filter_map(|s| s.wheels.get(k))
                    .map(|w| [x(w), y(w)])
                    .filter(|q| q[0].is_finite() && q[1].is_finite())
                    .collect();
                p.points(Points::new(wheel_name(k, axles), pts).radius(1.5).color(WHEELS[k % WHEELS.len()]));
            }
        }
    };
    ui.label("tyres: lateral force / load against slip angle (°)");
    Plot::new("fy_alpha").height(140.0).legend(Legend::default()).show(ui, scatter(|w| w.alpha, |w| w.fy));
    ui.label("tyres: longitudinal force / load against slip ratio κ");
    Plot::new("fx_kappa").height(140.0).legend(Legend::default()).show(ui, scatter(|w| w.kappa, |w| w.fx));
}
