//! Head-up display: flight data, events, rotor speeds (or a ground vehicle's powertrain,
//! pedals and per-wheel load, slip and force; or an aircraft's air data, controls and
//! artificial horizon), simulation controls and key help; a map seed
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
                    O  goals/trails/lanes G  plots   I/K  camera image/output\n\
                    L  LiDAR hits         V  LiDAR view\n\
                    H  hide HUD   F1  help   Esc  quit";

const GROUND_HELP: &str = "W/S  throttle / brake, reverse   A/D  steer\n\
                           Space  handbrake      R  reset episode\n\
                           Tab  next agent       P  pause   [/]  time scale\n\
                           C  camera             mouse drag  look   wheel  zoom\n\
                           O  goals/trails/lanes G  plots   I/K  camera image/output\n\
                           L  LiDAR hits         V  LiDAR view\n\
                           H  hide HUD   F1  help   Esc  quit\n\
                           gamepad: RT/LT pedals, left stick steers, A handbrake";

const RIDE_HELP: &str = "W/S  speed setpoint up/down (vk)   A/D  turn\n\
                         Space  stop           R  reset episode\n\
                         Tab  next agent       P  pause   [/]  time scale\n\
                         C  camera             mouse drag  look   wheel  zoom\n\
                         O  goals/trails/lanes G  plots   I/K  camera image/output\n\
                         L  LiDAR hits         V  LiDAR view\n\
                         H  hide HUD   F1  help   Esc  quit";

const FLIGHT_HELP: &str = "attitude  A/D bank   W/S pitch   Space/Shift airspeed\n\
                           guidance  A/D turn   Space/Shift climb   W/S airspeed\n\
                           rates     A/D roll   W/S pitch   Q/E yaw   Space/Shift throttle\n\
                           M  pilot mode         R  reset episode\n\
                           Tab  next agent       P  pause   [/]  time scale\n\
                           C  camera             mouse drag  look   wheel  zoom\n\
                           O  goals/trails/lanes G  plots   I/K  camera image/output\n\
                           H  hide HUD   F1  help   Esc  quit";

const HELI_HELP: &str = "velocity  W/S forward/back   A/D left/right   Space/Shift climb   Q/E yaw\n\
                         attitude  W/S pitch   A/D bank   Space/Shift collective   Q/E yaw\n\
                         rates     W/S pitch   A/D roll   Space/Shift collective   Q/E yaw\n\
                         M  pilot mode         R  reset episode\n\
                         Tab  next agent       P  pause   [/]  time scale\n\
                         C  camera             mouse drag  look   wheel  zoom\n\
                         O  goals/trails/lanes G  plots   I/K  camera image/output\n\
                         H  hide HUD   F1  help   Esc  quit";

const TILT_HELP: &str = "velocity  W/S speed setpoint   A/D sideways (hover) / turn (wing)   Space/Shift climb   Q/E yaw\n\
                         attitude  W/S airspeed setpoint   A/D bank   Space/Shift climb   Q/E yaw\n\
                         M  pilot mode         R  reset episode\n\
                         Tab  next agent       P  pause   [/]  time scale\n\
                         C  camera             mouse drag  look   wheel  zoom\n\
                         O  goals/trails/lanes G  plots   I/K  camera image/output\n\
                         H  hide HUD   F1  help   Esc  quit";

/// Shown above [`HELP`] while a policy flies.
const POLICY_HELP: &str = "T  take over the followed agent / hand it back";

const REPLAY_HELP: &str = "P  play/pause         [/]  speed\n\
                           ←/→  ∓1 s   Shift+←/→  one sample\n\
                           N/B  next/previous episode\n\
                           Home/R  start of episode\n\
                           Tab  next agent       C  camera\n\
                           mouse drag  look      wheel  zoom\n\
                           O  goals/trails/lanes G  plots   I/K  camera image/output\n\
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
            if let Some((shown, building)) = view.streamed_tiles() {
                ui.label(format!("detail tiles {shown} (+{building} building) · view {:.1} km", view.far / 1000.0));
            }
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
                    if a.agent_crashes > 0 {
                        ui.label(format!("{} hit another agent", a.agent_crashes));
                    }
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
                } else if sim.replay.is_none() && sim.riding() && sim.drive_command.is_none() {
                    let full = sim.ride_map().map_or((0.0, 0.0), |m| (m.speed(), m.curvature()));
                    let speed = sim.ride_speed * full.0;
                    row(
                        ui,
                        "rider",
                        format!("keys  vk  {speed:4.1} m/s  turn {:+.2} (≤ {:.2} 1/m)", sim.steer, full.1),
                    );
                } else if sim.replay.is_none() && v.as_wheeled().is_some() {
                    let hb = if sim.handbrake { "  handbrake" } else { "" };
                    let who = if sim.drive_command.is_some() { "demo" } else { "keys" };
                    row(ui, "driver", format!("{who}  pedal {:+.1}  steer {:+.2}{hb}", sim.stick[0], sim.steer));
                } else if sim.replay.is_none() {
                    let [f, l, u, y] = sim.stick;
                    row(ui, "pilot", format!("{}  {f:+.1} {l:+.1} {u:+.1} {y:+.1}", sim.pilot_mode_name()));
                    if v.as_multirotor().is_some() {
                        row(ui, "max speed", format!("{:4.1} m/s", sim.max_speed));
                    }
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
            if let Some(f) = v.as_fixed_wing() {
                flight_status(ui, sim, f);
            }
            if let Some(h) = v.as_helicopter() {
                heli_status(ui, h);
            }
            if let Some(t) = v.as_tiltrotor() {
                tilt_status(ui, sim, t, agent.controller.as_tiltrotor().map(|c| c.schedule()));
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
                    (false, false) if v.as_fixed_wing().is_some() => FLIGHT_HELP,
                    (false, false) if v.as_helicopter().is_some() => HELI_HELP,
                    (false, false) if v.as_tiltrotor().is_some() => TILT_HELP,
                    (false, true) if sim.riding() => RIDE_HELP,
                    (false, true) => GROUND_HELP,
                    (false, false) => HELP,
                };
                ui.monospace(help);
            }
        });
}

/// An aircraft's air data, engine, controls and attitude: airspeed (and the keys' setpoint), α
/// and β (red past the stall angle), throttle and propeller speed, surface deflections, a stall
/// warning and an artificial horizon.
fn flight_status(ui: &mut egui::Ui, sim: &Sim, f: &autonomousim_vehicles::fixedwing::FixedWing) {
    let flow = f.flow();
    let (hi, lo) = f.stall_angles();
    let stalled = f.stalled();
    let red = egui::Color32::from_rgb(230, 80, 60);
    let setpoint = match sim.airspeed {
        Some(v) if sim.replay.is_none() && sim.manual_agent().is_some() => format!("  (set {v:4.1})"),
        _ => String::new(),
    };
    ui.label(format!("airspeed {:5.1} m/s{setpoint}", flow.airspeed));
    ui.horizontal(|ui| {
        let alpha = format!("α {:+5.1}°", flow.alpha.to_degrees());
        if flow.alpha > 0.85 * hi || flow.alpha < 0.85 * lo {
            ui.colored_label(red, alpha);
        } else {
            ui.label(alpha);
        }
        ui.label(format!("β {:+5.1}°", flow.beta.to_degrees()));
        if stalled {
            ui.colored_label(red, egui::RichText::new("STALL").strong());
        }
    });
    let throttle = f.input().throttle;
    ui.horizontal(|ui| {
        ui.label("throttle");
        ui.add(egui::ProgressBar::new(throttle as f32).desired_width(80.0).text(format!("{:3.0} %", 100.0 * throttle)));
        ui.label(format!("{:5.0} rpm", f.rotor_speed() * 30.0 / std::f64::consts::PI));
    });
    let [a, e, r, flap] = f.surfaces().map(f64::to_degrees);
    ui.label(format!("aileron {a:+5.1}° elevator {e:+5.1}° rudder {r:+5.1}° flaps {flap:4.1}°"));
    let (roll, pitch, _) = autonomousim_control::fixedwing::euler(f.orientation());
    horizon(ui, roll, pitch);
}

/// A helicopter's rotor, engine, air data, controls and attitude: rotor speed in per cent of
/// the governed speed (red below 90 %), engine power against its limit, airspeed and climb
/// rate, the pilot inputs, the tip-path plane's tilt and an artificial horizon.
fn heli_status(ui: &mut egui::Ui, h: &autonomousim_vehicles::rotorcraft::Helicopter) {
    let d = h.def();
    let red = egui::Color32::from_rgb(230, 80, 60);
    let rpm = h.rotor_speed() / d.engine.rated_speed;
    ui.horizontal(|ui| {
        ui.label("rotor");
        let bar = egui::ProgressBar::new((rpm / 1.2).clamp(0.0, 1.0) as f32).desired_width(80.0);
        ui.add(bar.text(format!("{:3.0} %", 100.0 * rpm)));
        if rpm < 0.9 {
            ui.colored_label(red, egui::RichText::new("LOW ROTOR").strong());
        }
    });
    ui.horizontal(|ui| {
        ui.label("power");
        let power = h.engine_power();
        let bar = egui::ProgressBar::new((power / d.engine.max_power).clamp(0.0, 1.0) as f32).desired_width(80.0);
        ui.add(bar.text(format!("{:.1} kW", 1e-3 * power)));
    });
    ui.label(format!("airspeed {:5.1} m/s  climb {:+5.1} m/s", h.flow().airspeed, h.lin_vel_world().z));
    ui.horizontal(|ui| {
        for (name, u) in ["coll", "lon", "lat", "ped"].into_iter().zip(h.input().to_array()) {
            ui.label(name);
            ui.add(egui::ProgressBar::new((0.5 * (u + 1.0)) as f32).desired_width(48.0).text(format!("{u:+.2}")));
        }
    });
    let [b1c, b1s] = h.main_rotor_state().flap.map(f64::to_degrees);
    ui.label(format!("disc tilt  forward {b1c:+4.1}°  left {b1s:+4.1}°"));
    let (roll, pitch, _) = autonomousim_control::fixedwing::euler(h.orientation());
    horizon(ui, roll, pitch);
}

/// A tiltrotor's air data, rotors, mounts and attitude: airspeed (and the keys' setpoint), α
/// (red near the wing's stall) and β, climb rate, electric power, throttle per rotor, the
/// mount tilts against the schedule's tilt at the airspeed and the flight state (hover,
/// converting or on the wing), the conversion corridor and an artificial horizon.
fn tilt_status(
    ui: &mut egui::Ui,
    sim: &Sim,
    t: &autonomousim_vehicles::tiltrotor::Tiltrotor,
    schedule: Option<&autonomousim_control::tiltrotor::TiltSchedule>,
) {
    let red = egui::Color32::from_rgb(230, 80, 60);
    let flow = t.flow();
    let setpoint = match sim.airspeed {
        Some(v) if sim.replay.is_none() && sim.manual_agent().is_some() => format!("  (set {v:+5.1})"),
        _ => String::new(),
    };
    ui.label(format!("airspeed {:5.1} m/s{setpoint}  climb {:+5.1} m/s", flow.airspeed, t.lin_vel_world().z));
    let stall = t
        .def()
        .surfaces
        .iter()
        .filter(|s| s.roll.cos().abs() > 0.5)
        .map(|s| s.alpha_stall - s.incidence)
        .fold(f64::INFINITY, f64::min);
    ui.horizontal(|ui| {
        let alpha = format!("α {:+5.1}°", flow.alpha.to_degrees());
        if flow.airspeed > 5.0 && flow.alpha > 0.85 * stall {
            ui.colored_label(red, alpha);
        } else {
            ui.label(alpha);
        }
        ui.label(format!("β {:+5.1}°", flow.beta.to_degrees()));
        if flow.airspeed > 5.0 && flow.alpha > stall {
            ui.colored_label(red, egui::RichText::new("STALL").strong());
        }
        ui.label(format!("power {:.2} kW", 1e-3 * t.electric_power()));
    });
    ui.horizontal(|ui| {
        ui.label("throttle");
        for &u in &t.input().throttle[..t.rotor_count()] {
            ui.add(egui::ProgressBar::new(u as f32).desired_width(40.0));
        }
    });
    let tilts = t.tilts();
    let tilt = tilts.iter().sum::<f64>() / tilts.len().max(1) as f64;
    let mounts: Vec<String> = tilts.iter().map(|x| format!("{:3.0}", x.to_degrees())).collect();
    match schedule {
        Some(s) => {
            let wing = s.wing_share(flow.airspeed);
            let state = if wing < 0.05 {
                "hover"
            } else if wing > 0.95 {
                "wing"
            } else {
                "converting"
            };
            ui.label(format!(
                "tilt {}°  scheduled {:3.0}°  · {state} ({:3.0} % wing)",
                mounts.join(" "),
                s.tilt(flow.airspeed).to_degrees(),
                100.0 * wing
            ));
            corridor(ui, s, t.def().controls.tilt.max, flow.airspeed, tilt);
        }
        None => {
            ui.label(format!("tilt {}°", mounts.join(" ")));
        }
    }
    let (roll, pitch, _) = autonomousim_control::fixedwing::euler(t.orientation());
    horizon(ui, roll, pitch);
}

/// The conversion corridor: tilt (up, 0 to `max_tilt`) against airspeed (right), the band of
/// tilts at which level flight is feasible (swept up to 1.4·V_s; faster, the rotors stay
/// forward), the stall speed, the schedule's tilt and the aircraft's point.
fn corridor(
    ui: &mut egui::Ui,
    s: &autonomousim_control::tiltrotor::TiltSchedule,
    max_tilt: f64,
    airspeed: f64,
    tilt: f64,
) {
    const W: f32 = 220.0;
    const H: f32 = 90.0;
    let (response, painter) = ui.allocate_painter(egui::vec2(W, H), egui::Sense::hover());
    let rect = response.rect;
    let corridor = s.corridor();
    let top = corridor.last().map_or(s.max_speed(), |p| p.speed).max(s.max_speed()).max(1.0);
    let max_tilt = max_tilt.max(0.1);
    let at = |v: f64, x: f64| {
        let u = (v / top).clamp(0.0, 1.0) as f32;
        let w = (x / max_tilt).clamp(0.0, 1.0) as f32;
        egui::pos2(rect.left() + u * rect.width(), rect.bottom() - w * rect.height())
    };
    painter.rect_filled(rect, 3.0, egui::Color32::from_gray(30));
    // One mesh without feathering, so that the quads between sweep speeds join without seams.
    let band = egui::Color32::from_rgb(55, 100, 70);
    let mut mesh = egui::Mesh::default();
    for pair in corridor.windows(2) {
        if let (Some((a0, a1)), Some((b0, b1))) = (pair[0].tilt, pair[1].tilt) {
            let k = mesh.vertices.len() as u32;
            for p in [at(pair[0].speed, a0), at(pair[1].speed, b0), at(pair[1].speed, b1), at(pair[0].speed, a1)] {
                mesh.colored_vertex(p, band);
            }
            mesh.add_triangle(k, k + 1, k + 2);
            mesh.add_triangle(k, k + 2, k + 3);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
    let line: Vec<egui::Pos2> = s.points().iter().map(|p| at(p.speed, p.tilt)).collect();
    painter.add(egui::Shape::line(line, egui::Stroke::new(1.5, egui::Color32::WHITE)));
    let vs = s.stall_speed();
    painter.line_segment(
        [at(vs, 0.0), at(vs, max_tilt)],
        egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(230, 80, 60, 160)),
    );
    painter.circle_filled(at(airspeed, tilt), 4.0, egui::Color32::from_rgb(250, 210, 40));
    painter.text(
        rect.right_bottom() + egui::vec2(-4.0, -2.0),
        egui::Align2::RIGHT_BOTTOM,
        format!("tilt vs airspeed 0–{top:.0} m/s"),
        egui::FontId::monospace(10.0),
        egui::Color32::LIGHT_GRAY,
    );
}

/// Artificial horizon: sky and ground split by the horizon, rotated by the bank (right bank
/// raises its right end) and moved down as the nose rises; a pitch ladder every 10° and the
/// aircraft symbol fixed in the middle.
fn horizon(ui: &mut egui::Ui, roll: f64, pitch: f64) {
    const SIZE: f32 = 130.0;
    // Screen pixels per radian of pitch.
    let k = SIZE / 60f32.to_radians();
    let (response, painter) = ui.allocate_painter(egui::vec2(SIZE, SIZE), egui::Sense::hover());
    let rect = response.rect;
    let c = rect.center();
    let (sin, cos) = (roll as f32).sin_cos();
    // Horizon frame (x right, y down, origin on the horizon) → screen.
    let offset = pitch as f32 * k;
    let at = |x: f32, y: f32| {
        let y = y + offset;
        c + egui::vec2(x * cos + y * sin, -x * sin + y * cos)
    };
    let l = 2.0 * SIZE;
    painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(70, 130, 200));
    let ground = vec![at(-l, 0.0), at(l, 0.0), at(l, l), at(-l, l)];
    painter.add(egui::Shape::convex_polygon(ground, egui::Color32::from_rgb(140, 95, 50), egui::Stroke::NONE));
    let white = egui::Stroke::new(1.5, egui::Color32::WHITE);
    painter.line_segment([at(-l, 0.0), at(l, 0.0)], white);
    for deg in [-20i32, -10, 10, 20] {
        let y = -(deg as f32).to_radians() * k;
        let w = if deg.abs() == 10 { 0.15 * SIZE } else { 0.25 * SIZE };
        painter.line_segment([at(-w, y), at(w, y)], egui::Stroke::new(1.0, egui::Color32::WHITE));
    }
    let yellow = egui::Stroke::new(3.0, egui::Color32::from_rgb(250, 210, 40));
    painter.line_segment([c + egui::vec2(-0.3 * SIZE, 0.0), c + egui::vec2(-0.08 * SIZE, 0.0)], yellow);
    painter.line_segment([c + egui::vec2(0.08 * SIZE, 0.0), c + egui::vec2(0.3 * SIZE, 0.0)], yellow);
    painter.circle_filled(c, 3.0, yellow.color);
    painter.text(
        rect.left_bottom() + egui::vec2(4.0, -4.0),
        egui::Align2::LEFT_BOTTOM,
        format!("bank {:+3.0}° pitch {:+3.0}°", roll.to_degrees(), pitch.to_degrees()),
        egui::FontId::monospace(10.0),
        egui::Color32::WHITE,
    );
}

/// Wheel names: FL, FR, RL, RR for two axles, else axle number and side (none for a single
/// wheel: F and R on a bicycle).
fn wheel_name(def: &autonomousim_vehicles::ground::WheeledDef, w: usize) -> String {
    let a = def.wheel_axle(w);
    let side = match (def.axles[a].is_single(), def.wheel_side(w)) {
        (true, _) => "",
        (false, 0) => "L",
        _ => "R",
    };
    match (def.axles.len(), a) {
        (2, 0) => format!("F{side}"),
        (2, _) => format!("R{side}"),
        (_, a) => format!("{}{side}", a + 1),
    }
}

/// Gear, engine speed, steering, the trailers' articulation and the driver's pedals, and a
/// table of the wheels: tyre load, suspension travel, slip, forces and torques.
fn ground_status(ui: &mut egui::Ui, sim: &Sim, w: &autonomousim_vehicles::ground::Wheeled) {
    let agent = sim.world.agent(sim.pilot);
    let p = w.powertrain();
    let gear = match p.gear {
        0 => "–".to_owned(),
        -1 => "R".to_owned(),
        g => g.to_string(),
    };
    let rpm = p.engine_speed * 30.0 / std::f64::consts::PI;
    if let Some((head, k)) = w.def().steering_head().filter(|_| w.def().is_single_track()) {
        single_track_status(ui, w, head, k, &gear, rpm);
    } else {
        ui.label(format!("steering {:+5.1}° · gear {gear} · {rpm:5.0} rpm", w.steering_angle().to_degrees()));
    }
    // Articulation of each trailer (and dolly) against the unit ahead, red near a jackknife.
    let limit = sim.world.scenario().spec.events.ground.jackknife_deg;
    for u in 1..w.num_units() {
        let unit = &w.def().units[u - 1];
        if matches!(unit.joint, autonomousim_vehicles::ground::UnitJoint::Hinge) {
            continue;
        }
        let (angle, rate) = w.articulation(u);
        let deg = angle.to_degrees();
        let text = format!("{} {deg:+6.1}° ({:+5.1}°/s)", unit.name, rate.to_degrees());
        let color = if deg.abs() > 0.8 * limit {
            egui::Color32::from_rgb(230, 80, 60)
        } else if deg.abs() > 0.5 * limit {
            egui::Color32::from_rgb(230, 180, 60)
        } else {
            ui.visuals().text_color()
        };
        ui.colored_label(color, text);
    }
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
    track_status(ui, w);
    egui::Grid::new("wheels").num_columns(8).striped(true).show(ui, |ui| {
        for h in ["", "load kN", "travel mm", "κ", "α °", "Fx kN", "Fy kN", "drive/brake N·m"] {
            ui.label(egui::RichText::new(h).small());
        }
        ui.end_row();
        for (k, s) in w.wheels().enumerate() {
            let t = &s.tire;
            ui.label(wheel_name(w.def(), k));
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

/// Roll (amber beyond 30°, red beyond 45°), the steering head's angle, rate and
/// torque (the rider's and the damper's, against the rider's limit), the rider's lean, the
/// feet and the gear.
fn single_track_status(
    ui: &mut egui::Ui,
    w: &autonomousim_vehicles::ground::Wheeled,
    head: &autonomousim_vehicles::ground::SteeringHeadDef,
    k: usize,
    gear: &str,
    rpm: f64,
) {
    let (_, _, roll) = w.pose().rot.to_euler(glam::EulerRot::ZYX);
    let (steer, rate) = w.steering_head().unwrap_or((0.0, 0.0));
    let torque = w.wheel(k).steer_torque;
    let deg = roll.to_degrees();
    let color = match deg.abs() {
        d if d > 45.0 => egui::Color32::from_rgb(230, 80, 60),
        d if d > 30.0 => egui::Color32::from_rgb(230, 180, 60),
        _ => ui.visuals().text_color(),
    };
    ui.horizontal(|ui| {
        ui.label("roll");
        ui.colored_label(color, egui::RichText::new(format!("{deg:+5.1}°")).monospace());
        let (lean, _) = w.rider_lean();
        if w.def().rider.is_some() {
            ui.label("rider lean");
            ui.monospace(format!("{:+5.1}°", lean.to_degrees()));
        }
        ui.label(if w.feet_down() { "feet down" } else { "feet up" });
    });
    ui.horizontal(|ui| {
        ui.label("steer");
        ui.monospace(format!("{:+5.1}° {:+6.1}°/s", steer.to_degrees(), rate.to_degrees()));
        ui.label("torque");
        let share = (torque / head.max_torque).clamp(-1.0, 1.0) as f32;
        ui.add(egui::ProgressBar::new(share.abs()).desired_width(50.0));
        ui.monospace(format!("{torque:+5.1} N·m"));
    });
    ui.label(format!("gear {gear} · {rpm:5.0} rpm"));
}

/// Tracked vehicles, per side: band speed (the road wheels' mean rim speed), slip (mean κ of
/// the loaded road wheels), sinkage into soft soil and load.
fn track_status(ui: &mut egui::Ui, w: &autonomousim_vehicles::ground::Wheeled) {
    use autonomousim_vehicles::ground::tire::TireModel;
    let def = w.def();
    if def.track.is_none() {
        return;
    }
    let patch = |k: usize| match &def.wheel_tire(k).model {
        TireModel::Track(p) => Some(p.radius),
        _ => None,
    };
    egui::Grid::new("tracks").num_columns(5).striped(true).show(ui, |ui| {
        for h in ["track", "band m/s", "slip", "sinkage mm", "load kN"] {
            ui.label(egui::RichText::new(h).small());
        }
        ui.end_row();
        for (side, name) in ["left", "right"].into_iter().enumerate() {
            let wheels: Vec<(usize, f64)> = (0..w.num_wheels())
                .filter(|&k| def.wheel_side(k) == side)
                .filter_map(|k| Some((k, patch(k)?)))
                .collect();
            if wheels.is_empty() {
                continue;
            }
            let band = wheels.iter().map(|&(k, r)| w.wheel(k).spin * r).sum::<f64>() / wheels.len() as f64;
            let loaded: Vec<&autonomousim_vehicles::ground::tire::TireForces> =
                wheels.iter().map(|&(k, _)| &w.wheel(k).tire).filter(|t| t.fz > 0.0).collect();
            let mean = |f: &dyn Fn(&autonomousim_vehicles::ground::tire::TireForces) -> f64| {
                if loaded.is_empty() { 0.0 } else { loaded.iter().map(|t| f(t)).sum::<f64>() / loaded.len() as f64 }
            };
            ui.label(name);
            ui.monospace(format!("{band:+6.2}"));
            ui.monospace(format!("{:+5.2}", mean(&|t| t.kappa)));
            ui.monospace(format!("{:5.0}", mean(&|t| t.sinkage) * 1e3));
            ui.monospace(format!("{:6.1}", loaded.iter().map(|t| t.fz).sum::<f64>() / 1e3));
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
    let Some(def) = sim.world.agent(sim.pilot).vehicle.as_wheeled().map(|w| w.def()) else { return };
    let n = n.min(def.num_wheels());
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
                p.points(Points::new(wheel_name(def, k), pts).radius(1.5).color(WHEELS[k % WHEELS.len()]));
            }
        }
    };
    ui.label("tyres: lateral force / load against slip angle (°)");
    Plot::new("fy_alpha").height(140.0).legend(Legend::default()).show(ui, scatter(|w| w.alpha, |w| w.fy));
    ui.label("tyres: longitudinal force / load against slip ratio κ");
    Plot::new("fx_kappa").height(140.0).legend(Legend::default()).show(ui, scatter(|w| w.kappa, |w| w.fx));
}
