//! Rendering camera sensors: the frames due in a world are rendered on the GPU and handed to
//! the cameras, which add noise and delay them (see [`autonomousim_sensors::Camera`]).
//!
//! Rendering is not part of the physics tick: after a policy step (or a reset),
//! [`Cameras::capture`] renders every camera due at the world's current tick, from the poses
//! at that tick, and [`WorldInstance::deliver`] passes the frames on. [`BatchSim`](crate::BatchSim)
//! does this for all its worlds; a single world uses [`Cameras::update`].
//!
//! A frame shows the world's map, water, roads and obstacles, and every active agent, the
//! camera's own vehicle labelled [`SemanticClass::OwnVehicle`] and the others
//! [`SemanticClass::Vehicle`]. Frames are bit-identical for the same state on the same GPU
//! and driver.
//!
//! The process has one GPU context, created on first use with the adapter named by
//! `AUTONOMOUSIM_RENDER_ADAPTER` (see [`AdapterChoice::from_env`]), or with the one given to
//! [`use_adapter`] before.

use crate::SimError;
use crate::scenario::CompiledScenario;
use crate::world::WorldInstance;
use autonomousim_render::{
    AdapterChoice, CameraPose, Draw, GpuContext, GpuRig, GpuWorld, Intrinsics, Renderer, SemanticClass, Shading, View,
    WorldOptions,
};
use autonomousim_scene::rig::{Placement, Rig};
use autonomousim_sensors::{CameraImage, Sensor, SensorConfig};
use autonomousim_vehicles::Vehicle;
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
            ctx,
        }
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Render the cameras of active agents that are due at `world`'s tick into `out`.
    pub fn capture(&mut self, world: &WorldInstance, out: &mut Vec<Capture>) -> Result<(), SimError> {
        let tick = world.clock().tick;
        let due = |s: &Sensor| matches!(s, Sensor::Camera(c) if c.is_due(tick));
        let agents = world.agents();
        if !agents.iter().any(|a| !a.disabled && a.sensors.iter().any(due)) {
            return Ok(());
        }
        let Self { ctx, renderer, options, shading, maps, rigs, placements } = self;
        let map = &mut maps[world.map_index()];
        if map.is_none() {
            *map = Some(GpuWorld::new(ctx, world.map(), options.clone()).map_err(|e| SimError::Render(e.to_string()))?);
        }
        let map = map.as_ref().expect("uploaded above");
        for (a, p) in agents.iter().zip(placements.iter_mut()) {
            p.clear();
            if !a.disabled {
                rigs[a.group].0.place(&a.vehicle, p);
            }
        }
        let mut draws: Vec<Draw> = Vec::new();
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
                draws.clear();
                map.draws(&view, &mut draws);
                for (j, b) in agents.iter().enumerate().filter(|(_, b)| !b.disabled) {
                    let class = if j == i { SemanticClass::OwnVehicle } else { SemanticClass::Vehicle };
                    rigs[b.group].1.draws(b.vehicle.pose(), &placements[j], class, &mut draws);
                }
                let frame = renderer.render(ctx, &view, &draws).map_err(|e| SimError::Render(e.to_string()))?;
                let image = CameraImage {
                    width: frame.width,
                    height: frame.height,
                    rgb: frame.rgb,
                    depth: frame.depth,
                    class: frame.class,
                };
                out.push(Capture { agent: i, sensor: k, image });
            }
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
