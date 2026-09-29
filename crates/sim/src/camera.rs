//! Rendering camera sensors: the frames due in a world are rendered on the GPU and handed to
//! the cameras, which add noise and delay them (see [`autonomousim_sensors::Camera`]).
//!
//! Rendering is not part of the physics tick: after a policy step (or a reset),
//! [`Cameras::capture`] renders every camera due at the world's current tick, from the poses
//! at that tick, and [`WorldInstance::deliver`] passes the frames on. [`BatchSim`](crate::BatchSim)
//! renders the cameras of all its worlds in one GPU submission ([`Cameras::capture_batch`]);
//! a single world uses [`Cameras::update`].
//!
//! A frame shows the world's map, water, roads and obstacles, and every active agent, the
//! camera's own vehicle labelled [`SemanticClass::OwnVehicle`] and the others
//! [`SemanticClass::Vehicle`], and the landing pads under the current goals of the groups
//! with `goals.pad` ([`SemanticClass::Marker`]). Frames are bit-identical for the same state on the same GPU
//! and driver.
//!
//! The process has one GPU context, created on first use with the adapter named by
//! `AUTONOMOUSIM_RENDER_ADAPTER` (see [`AdapterChoice::from_env`]), or with the one given to
//! [`use_adapter`] before.

use crate::SimError;
use crate::scenario::CompiledScenario;
use crate::world::WorldInstance;
use autonomousim_core::terrain::Terrain;
use autonomousim_render::{
    AdapterChoice, CameraPose, Draw, GpuContext, GpuMesh, GpuRig, GpuWorld, Intrinsics, Job, Renderer, SemanticClass,
    Shading, View, WorldOptions,
};
use autonomousim_scene::rig::{Placement, Rig};
use autonomousim_sensors::{CameraImage, Sensor, SensorConfig};
use autonomousim_vehicles::Vehicle;
use glam::{DQuat, DVec3};
use std::sync::{Arc, OnceLock};

static GPU: OnceLock<Result<Arc<GpuContext>, String>> = OnceLock::new();

/// Create the process's GPU context with `choice`, unless it exists already; false if it
/// did (with whatever adapter it was created with).
pub fn use_adapter(choice: &AdapterChoice) -> bool {
    GPU.get().is_none() && GPU.set(GpuContext::new(choice).map(Arc::new).map_err(|e| e.to_string())).is_ok()
}

/// The process's GPU context.
pub fn gpu() -> Result<Arc<GpuContext>, SimError> {
    GPU.get_or_init(|| GpuContext::from_env().map(Arc::new).map_err(|e| e.to_string()))
        .clone()
        .map_err(SimError::Render)
}

/// Whether any group of `scenario` carries a camera.
pub fn has_cameras(scenario: &CompiledScenario) -> bool {
    scenario.groups.iter().any(|g| g.spec.sensors.iter().any(|s| matches!(s.config, SensorConfig::Camera(_))))
}

/// A frame for camera `sensor` of `agent`, rendered at the world's current tick.
#[derive(Clone, Debug)]
pub struct Capture {
    pub agent: usize,
    pub sensor: usize,
    pub image: CameraImage,
}

/// Renders the cameras of the worlds of one scenario.
pub struct Cameras {
    ctx: Arc<GpuContext>,
    renderer: Renderer,
    options: WorldOptions,
    shading: Shading,
    /// The maps of the scenario's pool, uploaded when first seen.
    maps: Vec<Option<GpuWorld>>,
    /// Per group.
    rigs: Vec<(Rig, GpuRig)>,
    /// Per agent, reused.
    placements: Vec<Vec<Placement>>,
    /// Pad radius per group (0: none).
    pads: Vec<f64>,
    /// A pad of unit radius.
    pad: GpuMesh,
}

impl Cameras {
    pub fn new(ctx: Arc<GpuContext>, scenario: &CompiledScenario) -> Self {
        let rigs = scenario
            .groups
            .iter()
            .map(|g| {
                let rig = Rig::new(&Vehicle::new(&g.def, scenario.clock.dt()));
                let gpu = GpuRig::new(&ctx, &rig);
                (rig, gpu)
            })
            .collect();
        Self {
            renderer: Renderer::new(&ctx),
            options: WorldOptions::default(),
            shading: Shading::default(),
            maps: (0..scenario.maps.len()).map(|_| None).collect(),
            rigs,
            placements: vec![Vec::new(); scenario.num_agents()],
            pads: scenario.groups.iter().map(|g| g.spec.goals.pad).collect(),
            pad: GpuMesh::new(&ctx, &autonomousim_scene::mesh::landing_pad(1.0), SemanticClass::Marker),
            ctx,
        }
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Render the cameras of active agents that are due at `world`'s tick into `out`.
    pub fn capture(&mut self, world: &WorldInstance, out: &mut Vec<Capture>) -> Result<(), SimError> {
        self.capture_batch(&mut [(world, out)])
    }

    /// Render the cameras due in each world into its list, all in one GPU submission.
    pub fn capture_batch(&mut self, worlds: &mut [(&WorldInstance, &mut Vec<Capture>)]) -> Result<(), SimError> {
        let due = |w: &WorldInstance| {
            let tick = w.clock().tick;
            w.agents()
                .iter()
                .any(|a| !a.disabled && a.sensors.iter().any(|s| matches!(s, Sensor::Camera(c) if c.is_due(tick))))
        };
        let Self { ctx, renderer, options, shading, maps, rigs, placements, pads, pad } = self;
        for (w, _) in worlds.iter().filter(|(w, _)| due(w)) {
            let map = &mut maps[w.map_index()];
            if map.is_none() {
                *map = Some(GpuWorld::new(ctx, w.map(), options.clone()).map_err(|e| SimError::Render(e.to_string()))?);
            }
        }
        let mut draws: Vec<Draw> = Vec::new();
        let mut jobs = Vec::new();
        // World, agent and sensor of each job.
        let mut owners = Vec::new();
        for (wi, (w, _)) in worlds.iter().enumerate().filter(|(_, (w, _))| due(w)) {
            let tick = w.clock().tick;
            let map = maps[w.map_index()].as_ref().expect("uploaded above");
            let agents = w.agents();
            // Pads lie on the ground under the goals, tilted with it, 2 cm up.
            let pad_draws: Vec<Draw> = agents
                .iter()
                .filter(|a| !a.disabled && pads[a.group] > 0.0 && !a.goals.is_empty())
                .map(|a| {
                    let g = a.goal().position;
                    let (h, n) = w.map().terrain().height_normal(g.x, g.y);
                    let r = pads[a.group];
                    Draw::new(pad, DVec3::new(g.x, g.y, h) + 0.02 * n, DQuat::from_rotation_arc(DVec3::Z, n))
                        .with_scale(DVec3::new(r, r, 1.0))
                })
                .collect();
            for (a, p) in agents.iter().zip(placements.iter_mut()) {
                p.clear();
                if !a.disabled {
                    rigs[a.group].0.place(&a.vehicle, p);
                }
            }
            for (i, a) in agents.iter().enumerate().filter(|(_, a)| !a.disabled) {
                for (k, s) in a.sensors.iter().enumerate() {
                    let Sensor::Camera(cam) = s else { continue };
                    if !cam.is_due(tick) {
                        continue;
                    }
                    let c = cam.config();
                    let pose = c.mount.world_pose(&a.kinematics());
                    let view = View {
                        pose: CameraPose::new(pose.pos, pose.rot),
                        intrinsics: Intrinsics {
                            near: c.near,
                            far: c.far,
                            ..Intrinsics::new(c.width, c.height, c.fov_deg.to_radians())
                        },
                        shading: *shading,
                    };
                    let first = draws.len();
                    map.draws(&view, &mut draws);
                    draws.extend(pad_draws.iter().copied());
                    for (j, b) in agents.iter().enumerate().filter(|(_, b)| !b.disabled) {
                        let class = if j == i { SemanticClass::OwnVehicle } else { SemanticClass::Vehicle };
                        rigs[b.group].1.draws(b.vehicle.pose(), &placements[j], class, &mut draws);
                    }
                    jobs.push(Job { view, draws: first..draws.len() });
                    owners.push((wi, i, k));
                }
            }
        }
        let frames = renderer.render_batch(ctx, &jobs, &draws).map_err(|e| SimError::Render(e.to_string()))?;
        for ((wi, agent, sensor), frame) in owners.into_iter().zip(frames) {
            let image = CameraImage {
                width: frame.width,
                height: frame.height,
                rgb: frame.rgb,
                depth: frame.depth,
                class: frame.class,
            };
            worlds[wi].1.push(Capture { agent, sensor, image });
        }
        Ok(())
    }

    /// Render and deliver the frames due in `world`.
    pub fn update(&mut self, world: &mut WorldInstance) -> Result<(), SimError> {
        let mut frames = Vec::new();
        self.capture(world, &mut frames)?;
        world.deliver(frames);
        Ok(())
    }
}
