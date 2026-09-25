//! Native viewer for autonomousim: flies a vehicle through a generated map in real time, or
//! plays back a recording.
//!
//! ```text
//! cargo run -p autonomousim-viewer --release -- --preset showcase --seed 0 --vehicle iris_like
//! cargo run -p autonomousim-viewer --release -- --preset offroad --vehicle offroad_4x4 --record drive.mcap
//! cargo run -p autonomousim-viewer --release -- --map rural --vehicle truck_6x4 --trailer semitrailer_3axle
//! cargo run -p autonomousim-viewer --release -- --scenario assets/scenarios/forest.toml
//! cargo run -p autonomousim-viewer --release -- replay recordings/run.mcap --episode 2
//! cargo run -p autonomousim-viewer --release -- policy runs/run/policy.json --agents 4
//! ```
//!
//! Live, the simulation runs in-process at its physics rate from a fixed-step accumulator and
//! the first agent is flown or driven from the keyboard (or a gamepad), optionally recorded to
//! MCAP; a replay rebuilds the recorded
//! maps, checks their hashes and puts the agents where the recording has them; `policy` flies
//! the agents with a trained policy (from `examples/export_policy.py`) in its task's scenario
//! on a new map. F1 shows the keys.

mod autopilot;
mod camera;
mod convert;
mod history;
mod hud;
mod lidar_view;
mod overlay;
mod replay;
mod sim;
mod vehicle_view;
mod world_view;

use anyhow::{Context, bail};
use autonomousim_control::ground::{GroundActionMode, GroundSetpoint};
use autonomousim_control::multirotor::ActionMode;
use autonomousim_core::math::quat::yaw;
use autonomousim_core::rng::Seed;
use autonomousim_procgen::{RuralPreset, WildPreset};
use autonomousim_sim::policy::PolicyFile;
use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::scenario::{GoalKind, GoalSpec, MapSource, RuralMaps, SpawnSpec, VehicleRef, WildMaps};
use autonomousim_sim::{CompiledScenario, Events, GroupSpec, Scenario, WorldInstance};
use autonomousim_vehicles::{Vehicle, VehicleDef};
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::{PresentMode, WindowResolution};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use camera::CameraRig;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;

#[derive(Parser, Debug)]
#[command(version, about = "autonomousim viewer: fly through a generated map or play back a recording")]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    live: LiveArgs,
    #[command(flatten)]
    display: DisplayArgs,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Fly through a generated map (the default).
    Live {
        #[command(flatten)]
        live: LiveArgs,
        #[command(flatten)]
        display: DisplayArgs,
    },
    /// Play back an MCAP recording (from `eval_record.py` or `Recorder`).
    Replay {
        file: PathBuf,
        /// Episode to start with (from 1).
        #[arg(long, default_value_t = 1)]
        episode: usize,
        /// Stop at the end of the last episode instead of starting over.
        #[arg(long)]
        once: bool,
        #[command(flatten)]
        display: DisplayArgs,
    },
    /// Fly a trained policy (exported by `examples/export_policy.py`) in its task's scenario.
    Policy {
        file: PathBuf,
        /// Map seed (the training maps are generated from the scenario's seed, 0 by default).
        #[arg(long, default_value_t = 1000)]
        map_seed: u64,
        /// Agents flown by the policy (default: as trained).
        #[arg(long)]
        agents: Option<usize>,
        /// Episode seed (spawn positions and goals).
        #[arg(long, default_value_t = 0)]
        episode_seed: u64,
        /// Generate the map without the map cache.
        #[arg(long)]
        no_cache: bool,
        /// Record the session to this MCAP file.
        #[arg(long)]
        record: Option<PathBuf>,
        #[command(flatten)]
        display: DisplayArgs,
    },
}

#[derive(Args, Debug)]
struct LiveArgs {
    /// Scenario file (TOML or JSON); replaces the map and vehicle options.
    #[arg(long)]
    scenario: Option<PathBuf>,
    /// Map generator.
    #[arg(long, value_enum, default_value_t = MapKind::Wild)]
    map: MapKind,
    /// Map preset: `training`, `showcase` or (wild maps) `offroad`.
    #[arg(long, default_value = "showcase")]
    preset: String,
    /// Map seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Map size (m), overriding the preset.
    #[arg(long)]
    size: Option<f64>,
    /// Vehicle preset name or TOML file.
    #[arg(long, default_value = "iris_like")]
    vehicle: String,
    /// Trailer preset name or TOML file towed by a ground vehicle (repeat for a road train).
    #[arg(long)]
    trailer: Vec<String>,
    /// Mean wind from the west (m/s).
    #[arg(long, default_value_t = 0.0)]
    wind: f64,
    /// Episode seed (spawn position).
    #[arg(long, default_value_t = 0)]
    episode_seed: u64,
    /// Generate the map without the map cache.
    #[arg(long)]
    no_cache: bool,
    /// Fly forward on your own, or drive from place to place along drivable paths (for demos
    /// and performance checks).
    #[arg(long)]
    demo: bool,
    /// Record the session to this MCAP file (replay it with `replay`).
    #[arg(long)]
    record: Option<PathBuf>,
}

/// Map generators of live mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum MapKind {
    Wild,
    Rural,
}

#[derive(Args, Debug, Clone)]
struct DisplayArgs {
    #[arg(long, value_enum, default_value_t = Quality::Medium)]
    quality: Quality,
    /// Window size in physical pixels, e.g. `1920x1080` (default: 1600×900 logical pixels).
    #[arg(long, value_parser = parse_window)]
    window: Option<(u32, u32)>,
    /// Render as fast as possible instead of at the display rate (always on with
    /// `--screenshot`, since compositors throttle hidden windows to about one frame per
    /// second under vsync).
    #[arg(long)]
    no_vsync: bool,
    /// Show the plots from the start (G toggles them).
    #[arg(long)]
    plots: bool,
    /// Show the LiDAR view from the start (V toggles it).
    #[arg(long)]
    lidar_view: bool,
    /// Save a screenshot (PNG) after `--frames` frames and exit.
    #[arg(long)]
    screenshot: Option<PathBuf>,
    #[arg(long, default_value_t = 120)]
    frames: u32,
}

/// Rendering presets.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Quality {
    Low,
    Medium,
    High,
}

impl Quality {
    fn shadows(self) -> bool {
        self != Quality::Low
    }

    /// Shadow cascades and the distance they cover (m). Shadows are the largest cost on the
    /// Iris Xe: at 1080p the medium preset runs at 61 fps with two cascades to 150 m and
    /// 4× MSAA, and at 80 fps with one cascade to 100 m and 2× MSAA.
    fn shadow_cascades(self) -> (usize, f32) {
        match self {
            Quality::High => (2, 150.0),
            _ => (1, 100.0),
        }
    }

    fn view_distance(self) -> f32 {
        match self {
            Quality::Low => 500.0,
            Quality::Medium => 900.0,
            Quality::High => 1600.0,
        }
    }

    /// Distance beyond which obstacles are drawn coarse (m).
    fn prop_detail_distance(self) -> f32 {
        match self {
            Quality::Low => 250.0,
            Quality::Medium => 400.0,
            Quality::High => 700.0,
        }
    }

    fn msaa(self) -> Msaa {
        match self {
            Quality::Low => Msaa::Off,
            Quality::Medium => Msaa::Sample2,
            Quality::High => Msaa::Sample4,
        }
    }
}

fn parse_window(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s.split_once('x').ok_or("expected WIDTHxHEIGHT")?;
    Ok((w.parse().map_err(|e| format!("{e}"))?, h.parse().map_err(|e| format!("{e}"))?))
}

fn scenario(args: &LiveArgs) -> anyhow::Result<Scenario> {
    if let Some(path) = &args.scenario {
        return Scenario::load(path).with_context(|| format!("loading {}", path.display()));
    }
    let config = args.size.map(|s| serde_json::json!({ "size": s }));
    let vehicle = VehicleRef::Name(args.vehicle.clone());
    let ground = matches!(vehicle.resolve()?, VehicleDef::Wheeled(_));
    if !ground && !args.trailer.is_empty() {
        bail!("only ground vehicles tow trailers");
    }
    let rural = args.map == MapKind::Rural;
    let group = if ground {
        // On rural maps: start in a lane with a route to a farm yard.
        let (spawn, goals) = if rural {
            let goals = GoalSpec { kind: GoalKind::Route, distance: [150.0, 400.0], radius: 5.0, ..Default::default() };
            (SpawnSpec { on_road: true, min_separation: 10.0, ..Default::default() }, goals)
        } else {
            (SpawnSpec { margin: 60.0, ..Default::default() }, GoalSpec::default())
        };
        GroupSpec {
            name: "driver".into(),
            vehicle,
            action_mode: Some(GroundActionMode::Raw.into()),
            trailers: args.trailer.clone(),
            spawn,
            goals,
            disable_on_terminal: false,
            ..Default::default()
        }
    } else {
        GroupSpec {
            name: "pilot".into(),
            vehicle,
            action_mode: Some(ActionMode::Velocity.into()),
            spawn: SpawnSpec { agl: [2.0, 2.0], clearance: 4.0, margin: 60.0, ..Default::default() },
            disable_on_terminal: false,
            ..Default::default()
        }
    };
    let mut sc = Scenario {
        name: "viewer".into(),
        map: match args.map {
            MapKind::Wild => MapSource::Wild(WildMaps {
                seed: args.seed,
                count: 1,
                preset: args.preset.parse::<WildPreset>().map_err(anyhow::Error::msg)?,
                config,
                cache: !args.no_cache,
            }),
            MapKind::Rural => MapSource::Rural(RuralMaps {
                seed: args.seed,
                count: 1,
                preset: args.preset.parse::<RuralPreset>().map_err(anyhow::Error::msg)?,
                config,
                cache: !args.no_cache,
            }),
        },
        groups: vec![group],
        ..Default::default()
    };
    sc.environment.wind.mean = glam::DVec2::new(args.wind, 0.0);
    Ok(sc)
}

/// The scenario of a policy file: its task's scenario on one map of `map_seed`, with `agents`
/// in the policy's group.
fn policy_scenario(file: &PolicyFile, map_seed: u64, agents: Option<usize>, cache: bool) -> anyhow::Result<Scenario> {
    let mut sc = file.scenario.clone();
    sc.map.set_pool(map_seed, 1, cache);
    let group = sc.groups.iter_mut().find(|g| g.name == file.group);
    let group = group.with_context(|| format!("the scenario has no agent group {:?}", file.group))?;
    if let Some(n) = agents {
        group.count = n.max(1);
    }
    Ok(sc)
}

#[derive(Resource)]
struct Capture {
    path: Option<PathBuf>,
    /// Frames of the run and frames left.
    total: u32,
    frames: u32,
    demo: bool,
    /// Start of the measured frames (after a warm-up) and their count.
    measured: Option<(std::time::Instant, u32)>,
}

/// Frames not counted in the fps of a screenshot run (start-up, shader compilation).
const WARM_UP: u32 = 120;

/// The live scenario, from which new maps are generated in the background (HUD).
#[derive(Resource)]
pub struct Regenerate {
    pub scenario: Scenario,
    /// Map seed being generated or shown.
    pub seed: u64,
    pub pending: Option<JoinHandle<anyhow::Result<CompiledScenario>>>,
    pub error: Option<String>,
}

impl Regenerate {
    /// The seed of generated maps, if the scenario generates them.
    pub fn map_seed(&self) -> Option<u64> {
        self.scenario.map.seed()
    }

    /// Start generating the maps of `seed` on a background thread.
    pub fn start(&mut self, seed: u64) {
        let mut sc = self.scenario.clone();
        sc.map.set_seed(seed);
        self.seed = seed;
        self.error = None;
        self.pending = Some(std::thread::spawn(move || Ok(sc.compile()?)));
    }
}

/// Swap in the new maps once they are ready.
fn finish_regenerate(mut regen: ResMut<Regenerate>, mut sim: ResMut<sim::Sim>) {
    if !regen.pending.as_ref().is_some_and(|h| h.is_finished()) {
        return;
    }
    let handle = regen.pending.take().unwrap();
    match handle.join().map_err(|_| anyhow::anyhow!("map generation panicked")).and_then(|r| r) {
        Ok(compiled) => {
            let seed = regen.seed;
            regen.scenario.map.set_seed(seed);
            let world = WorldInstance::new(Arc::new(compiled), Seed::from_u64(sim.episodes));
            sim.set_world(world);
        }
        Err(e) => regen.error = Some(format!("{e:#}")),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let start = std::time::Instant::now();
    let (sim, display, regen, demo, title) = match cli.command {
        Some(Command::Replay { file, episode, once, display }) => {
            let recording = Recording::read(&file).with_context(|| format!("reading {}", file.display()))?;
            if recording.episodes.is_empty() {
                bail!("{} holds no episodes", file.display());
            }
            let compiled = Arc::new(recording.compile().context("rebuilding the recorded scenario")?);
            let world = WorldInstance::new(compiled, Seed::from_u64(0));
            println!(
                "{}: {} episodes, {} agents, {} maps rebuilt and checked",
                file.display(),
                recording.episodes.len(),
                recording.agents.len(),
                recording.map_hashes.len()
            );
            let mut replay = replay::Replay::new(recording, episode.saturating_sub(1));
            replay.looping = !once;
            let title = format!("autonomousim · replay {}", file.file_name().unwrap_or_default().to_string_lossy());
            (sim::Sim::replay(world, replay), display, None, false, title)
        }
        Some(Command::Policy { file, map_seed, agents, episode_seed, no_cache, record, display }) => {
            let policy = PolicyFile::read(&file)?;
            let sc = policy_scenario(&policy, map_seed, agents, !no_cache)?;
            let compiled = Arc::new(sc.clone().compile().context("building the policy's scenario")?);
            let world = WorldInstance::new(compiled, Seed::from_u64(episode_seed));
            let autopilot = autopilot::Autopilot::new(&policy, &world).context("loading the policy")?;
            println!(
                "{}: {} policy for {} ({} layers, checked against PyTorch)",
                file.display(),
                policy.algo,
                policy.env_id,
                policy.layers.len()
            );
            let mut sim = sim::Sim::new(world);
            sim.autopilot = Some(autopilot);
            start_recording(&mut sim, record.as_ref())?;
            let regen = Regenerate { seed: map_seed, scenario: sc, pending: None, error: None };
            let title = format!("autonomousim · policy {}", policy.name);
            (sim, display, Some(regen), false, title)
        }
        command => {
            let (live, display) = match command {
                Some(Command::Live { live, display }) => (live, display),
                _ => (cli.live, cli.display),
            };
            let sc = scenario(&live)?;
            let compiled = Arc::new(sc.clone().compile().context("building the scenario")?);
            let world = WorldInstance::new(compiled, Seed::from_u64(live.episode_seed));
            let regen = Regenerate { seed: live.seed, scenario: sc, pending: None, error: None };
            let mut sim = sim::Sim::new(world);
            start_recording(&mut sim, live.record.as_ref())?;
            (sim, display, Some(regen), live.demo, "autonomousim".to_owned())
        }
    };
    let map = sim.world.map().clone();
    println!(
        "map {} (seed {}, {:.0} m) ready in {:.2} s",
        map.meta.name,
        map.meta.seed,
        (map.extent().1 - map.extent().0).x,
        start.elapsed().as_secs_f64()
    );
    let quality = display.quality;
    let sky = Color::srgb(0.62, 0.75, 0.88);

    let mut app = App::new();
    app.insert_resource(ClearColor(sky))
        .insert_resource(quality)
        .insert_resource(world_view::MapView::new(map, quality))
        .insert_resource(history::History::default())
        .insert_resource(overlay::Overlay::default())
        .insert_resource(sim)
        .insert_resource(DemoRoute::default())
        .insert_resource(hud::Hud { visible: true, help: display.screenshot.is_none(), plots: display.plots })
        .insert_resource(lidar_view::LidarView { visible: display.lidar_view })
        .insert_resource(Capture {
            path: display.screenshot.clone(),
            total: display.frames,
            frames: display.frames,
            demo,
            measured: None,
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title,
                resolution: match display.window {
                    Some((w, h)) => WindowResolution::new(w, h).with_scale_factor_override(1.0),
                    None => (1600u32, 900u32).into(),
                },
                present_mode: if display.no_vsync || display.screenshot.is_some() {
                    PresentMode::AutoNoVsync
                } else {
                    PresentMode::AutoVsync
                },
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(EguiPlugin::default())
        .add_systems(
            Startup,
            (world_view::spawn_lights, world_view::spawn_map, vehicle_view::spawn_vehicles, spawn_camera),
        )
        .add_systems(
            Update,
            (
                sim::pilot_input,
                sim::replay_input,
                demo_pilot,
                sim::step,
                history::record,
                world_view::sync_map,
                vehicle_view::sync_vehicles,
                camera::update_camera,
                world_view::update_lod,
                overlay::draw,
                capture,
                quit_on_escape,
            )
                .chain(),
        )
        .add_systems(EguiPrimaryContextPass, (hud::hud, lidar_view::lidar_view).chain())
        .add_systems(Last, finish_recording);
    if let Some(regen) = regen {
        app.insert_resource(regen).add_systems(Update, finish_regenerate.before(sim::step));
    }
    app.run();
    Ok(())
}

fn spawn_camera(mut commands: Commands, sim: Res<sim::Sim>, view: Res<world_view::MapView>, quality: Res<Quality>) {
    let v = &sim.world.agent(sim.pilot).vehicle;
    let heading = yaw(v.orientation());
    let (span, rig) = match v {
        Vehicle::Multirotor(m) => {
            let span = autonomousim_scene::props::multirotor(m.def()).span;
            (span, CameraRig::new(f64::from(span), heading))
        }
        Vehicle::Wheeled(w) => {
            let visual = autonomousim_scene::props::wheeled(w.def());
            let mut rig = CameraRig::ground(f64::from(visual.span), heading, visual.eye, visual.rear_eye);
            if w.num_units() > 1 {
                // Look over the trailers at the tractor.
                rig.pitch = 0.35;
            }
            (visual.span, rig)
        }
    };
    let far = view.view_distance;
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 70f32.to_radians(),
            near: (0.05 * span).clamp(0.005, 0.1),
            far: 1.5 * far,
            ..default()
        }),
        DistanceFog {
            color: Color::srgb(0.62, 0.75, 0.88),
            directional_light_color: Color::srgba(1.0, 0.95, 0.85, 0.5),
            directional_light_exponent: 30.0,
            falloff: FogFalloff::Linear { start: 0.3 * far, end: far },
        },
        quality.msaa(),
        rig,
        Transform::default(),
    ));
}

/// Record the session to `path`, if given.
fn start_recording(sim: &mut sim::Sim, path: Option<&PathBuf>) -> anyhow::Result<()> {
    if let Some(path) = path {
        let config = RecorderConfig { lidar: true, ..Default::default() };
        let recorder = Recorder::create(path, config).with_context(|| format!("creating {}", path.display()))?;
        sim.record(recorder);
        println!("recording to {}", path.display());
    }
    Ok(())
}

/// Close the recording when the viewer exits.
fn finish_recording(mut exits: MessageReader<AppExit>, mut sim: ResMut<sim::Sim>) {
    if exits.read().next().is_some()
        && sim.is_recording()
        && let Err(e) = sim.finish_recording()
    {
        error!("finishing the recording: {e:#}");
    }
}

/// With `--demo`: fly forward at 8 m/s, turning slowly, 40 m above the ground (above the
/// tallest trees); ground vehicles drive to random reachable places along drivable paths.
fn demo_pilot(capture: Res<Capture>, mut sim: ResMut<sim::Sim>, mut route: ResMut<DemoRoute>) {
    if !capture.demo {
        return;
    }
    if sim.world.agent(sim.pilot).vehicle.family() == autonomousim_vehicles::Family::Wheeled {
        drive_demo(&mut sim, &mut route);
        return;
    }
    let agl = sim.world.agent(sim.pilot).agl_now(sim.world.map());
    let climb = (40.0 - agl).clamp(-2.0, 3.0);
    sim.pilot_mode = sim::PilotMode::Velocity;
    sim.stick = [8.0 / sim.max_speed, 0.0, climb / sim.max_climb, 0.1 / sim.max_yaw_rate];
}

/// The demo driver's route: grid path points still ahead, and its random state.
#[derive(Resource, Default)]
struct DemoRoute {
    path: Vec<glam::DVec2>,
    rng: u64,
}

impl DemoRoute {
    /// Uniform in [0, 1) (xorshift).
    fn uniform(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Pure pursuit of a point 8 m ahead along a drivable grid path to a random place 80 m or more
/// away, at 8 m/s (4 m/s while turning); a new episode after a terminal event or getting
/// stuck.
fn drive_demo(sim: &mut sim::Sim, route: &mut DemoRoute) {
    const LOOKAHEAD: f64 = 8.0;
    let i = sim.pilot;
    if sim.latched[i].intersects(Events(Events::TERMINAL.0 | Events::STUCK.0)) {
        sim.reset();
        route.path.clear();
    }
    let agent = sim.world.agent(i);
    let p = agent.vehicle.position().truncate();
    if let Some(lane) = agent.route.clone() {
        drive_route(sim, &lane);
        return;
    }
    let grid = sim.world.scenario().groups[agent.group].drive[sim.world.map_index()].clone();
    if route.path.len() <= 1 && route.path.first().is_none_or(|q| q.distance(p) < LOOKAHEAD) {
        if route.rng == 0 {
            route.rng = 0x9e37_79b9_7f4a_7c15;
        }
        let (lo, hi) = sim.world.map().extent();
        route.path = (0..200)
            .find_map(|_| {
                let q = lo + (hi - lo) * glam::DVec2::new(route.uniform(), route.uniform());
                (q.distance(p) > 80.0 && grid.reachable(p, q)).then(|| grid.path(p, q)).flatten()
            })
            .unwrap_or_default();
    }
    while route.path.len() > 1 && route.path[0].distance(p) < LOOKAHEAD {
        route.path.remove(0);
    }
    let Some(&target) = route.path.first() else {
        sim.drive_command = Some(GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 });
        return;
    };
    let rel = glam::DVec2::from_angle(-yaw(agent.vehicle.orientation())).rotate(target - p);
    let curvature = 2.0 * rel.y / rel.length_squared().max(1.0);
    let speed = if rel.x > rel.y.abs() { 8.0 } else { 4.0 };
    sim.drive_command = Some(GroundSetpoint::SpeedCurvature { speed, curvature });
}

/// Pure pursuit along the pilot's route lane, 8 m ahead, at up to 12 m/s (slower in bends,
/// for 2 m/s² of lateral acceleration); a new episode at the end of the route.
fn drive_route(sim: &mut sim::Sim, lane: &autonomousim_world::Polyline) {
    const LOOKAHEAD: f64 = 8.0;
    let agent = sim.world.agent(sim.pilot);
    if agent.goal_index >= agent.goals.len() {
        sim.reset();
        return;
    }
    let (p, y) = (agent.vehicle.position().truncate(), yaw(agent.vehicle.orientation()));
    let follow = autonomousim_sim::lane::Follow::new(Some(lane), sim.world.map(), p, y).expect("a route");
    let rel = glam::DVec2::from_angle(-y).rotate(follow.point(LOOKAHEAD) - p);
    let curvature = 2.0 * rel.y / rel.length_squared().max(1.0);
    let bend = [0.0, 10.0, 20.0, 40.0].map(|a| follow.curvature(a).abs()).into_iter().fold(0.0, f64::max);
    let speed = (2.0 / bend.max(1e-3)).sqrt().min(12.0);
    sim.drive_command = Some(GroundSetpoint::SpeedCurvature { speed, curvature });
}

fn capture(
    mut commands: Commands,
    mut capture: ResMut<Capture>,
    diagnostics: Res<DiagnosticsStore>,
    sim: Res<sim::Sim>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(path) = capture.path.clone() else { return };
    capture.frames = capture.frames.saturating_sub(1);
    let warm = capture.frames + WARM_UP <= capture.total;
    match &mut capture.measured {
        Some((_, n)) => *n += 1,
        None if warm => capture.measured = Some((std::time::Instant::now(), 0)),
        None => {}
    }
    if capture.frames == 10 {
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
    } else if capture.frames == 0 {
        let recent = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS).and_then(|d| d.average()).unwrap_or(0.0);
        let mean = capture.measured.map_or(f64::NAN, |(t, n)| f64::from(n) / t.elapsed().as_secs_f64());
        let p = sim.render_pose(sim.pilot).pos;
        info!("{mean:.1} fps after warm-up, {recent:.1} over the last frames; t = {:.1} s at {p:.1}", sim.time());
        exit.write(AppExit::Success);
    }
}

fn quit_on_escape(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;
    use world_view::MapEntity;

    #[test]
    fn live_is_the_default_command() {
        let cli = Cli::try_parse_from(["viewer", "--seed", "3", "--quality", "low"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!((cli.live.seed, cli.display.quality), (3, Quality::Low));
        let cli = Cli::try_parse_from(["viewer", "live", "--seed", "4"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Live { live, .. }) if live.seed == 4));
        let cli = Cli::try_parse_from(["viewer", "replay", "a.mcap", "--episode", "2", "--window", "640x480"]).unwrap();
        let Some(Command::Replay { file, episode, once, display }) = cli.command else { panic!() };
        assert_eq!((file, episode, once, display.window), ("a.mcap".into(), 2, false, Some((640, 480))));
        assert!(!display.lidar_view);
        let cli = Cli::try_parse_from(["viewer", "replay", "a.mcap", "--lidar-view", "--plots"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Replay { display, .. }) if display.lidar_view && display.plots));
        // Live options do not apply to a replay.
        assert!(Cli::try_parse_from(["viewer", "--seed", "3", "replay", "a.mcap"]).is_err());
        assert!(Cli::try_parse_from(["viewer", "replay", "a.mcap", "--seed", "3"]).is_err());
        let cli = Cli::try_parse_from(["viewer", "policy", "p.json", "--agents", "3", "--lidar-view"]).unwrap();
        let Some(Command::Policy { file, map_seed, agents, display, .. }) = cli.command else { panic!() };
        assert_eq!((file, map_seed, agents, display.lidar_view), ("p.json".into(), 1000, Some(3), true));
        // Trailers for ground vehicles only.
        let rig = ["viewer", "--vehicle", "truck_6x4", "--trailer", "semitrailer_3axle"];
        let sc = scenario(&Cli::parse_from(rig).live).unwrap();
        assert_eq!(sc.groups[0].trailers, ["semitrailer_3axle"]);
        let drone = ["viewer", "--vehicle", "cf2x", "--trailer", "semitrailer_3axle"];
        assert!(scenario(&Cli::parse_from(drone).live).is_err());
    }

    /// A new map seed is generated in the background, swapped into the simulation, and the
    /// map entities are rebuilt for it.
    #[test]
    fn regenerated_maps_replace_the_shown_one() {
        let cli = Cli::parse_from(["viewer", "--preset", "training", "--size", "128", "--no-cache", "--seed", "1"]);
        let sc = scenario(&cli.live).unwrap();
        let world = WorldInstance::new(Arc::new(sc.clone().compile().unwrap()), Seed::from_u64(0));
        let first = world.map().clone();
        let mut app = World::new();
        app.insert_resource(world_view::MapView::new(first.clone(), Quality::Low));
        app.insert_resource(sim::Sim::new(world));
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());
        app.run_system_once(world_view::spawn_map).unwrap();
        let before: Vec<Entity> = app.query_filtered::<Entity, With<MapEntity>>().iter(&app).collect();
        assert!(!before.is_empty());

        let mut regen = Regenerate { seed: 1, scenario: sc, pending: None, error: None };
        regen.start(2);
        while !regen.pending.as_ref().unwrap().is_finished() {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        app.insert_resource(regen);
        app.run_system_once(finish_regenerate).unwrap();
        let sim = app.resource::<sim::Sim>();
        assert_eq!((sim.world.map().meta.seed, sim.episodes), (2, 2));
        assert!(!Arc::ptr_eq(sim.world.map(), &first));
        assert!(app.resource::<Regenerate>().pending.is_none());
        assert_eq!(app.resource::<Regenerate>().map_seed(), Some(2));

        app.run_system_once(world_view::sync_map).unwrap();
        let view = app.resource::<world_view::MapView>();
        assert!(Arc::ptr_eq(&view.world, app.resource::<sim::Sim>().world.map()));
        let after: Vec<Entity> = app.query_filtered::<Entity, With<MapEntity>>().iter(&app).collect();
        assert!(!after.is_empty() && before.iter().all(|e| app.get_entity(*e).is_err()));
    }

    /// A wheeled vehicle gets a driver group; `--record` writes the drive, wheels included.
    #[test]
    fn a_recorded_drive_plays_back() {
        let path = std::env::temp_dir().join(format!("autonomousim-viewer-drive-{}.mcap", std::process::id()));
        let path_arg = path.to_str().unwrap();
        let args = ["viewer", "--preset", "offroad", "--size", "128", "--no-cache", "--vehicle", "offroad_4x4"];
        let cli = Cli::try_parse_from(args.into_iter().chain(["--record", path_arg])).unwrap();
        assert_eq!(cli.live.record.as_deref(), Some(path.as_path()));
        let sc = scenario(&cli.live).unwrap();
        assert_eq!(sc.groups[0].name, "driver");
        let mut sim = sim::Sim::new(WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(0)));
        start_recording(&mut sim, cli.live.record.as_ref()).unwrap();
        assert!(sim.is_recording());
        sim.stick = [1.0, 0.5, 0.0, 0.0];
        for _ in 0..60 {
            sim.advance(1.0 / 60.0);
        }
        sim.finish_recording().unwrap();
        assert!(!sim.is_recording());
        let recording = Recording::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let states = &recording.episodes[0].states[0];
        assert!(states.len() >= 45, "{}", states.len());
        let last = states.last().unwrap();
        assert_eq!(last.wheels.len(), 4);
        assert!(last.wheels.iter().all(|w| w.spin_angle > 0.1 && w.load > 0.0), "{:?}", last.wheels);
    }
}
