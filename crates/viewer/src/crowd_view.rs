//! Pedestrians: a low-poly figure per pedestrian of the crowd, scaled to its height, walking
//! with a cycle paced by its speed (see [`figures`]); the ones hit lie on the ground.

use crate::convert::{self, RenderOrigin};
use crate::sim::Sim;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::from_yaw;
use autonomousim_scene::figures;
use autonomousim_sim::pedestrians::PedState;
use bevy::light::NotShadowCaster;
use bevy::prelude::*;
use glam::{DQuat, DVec3};

/// One pedestrian's figure: its index in the crowd, and the cycles it has walked.
#[derive(Component)]
pub struct Figure {
    index: usize,
    phase: f64,
    frame: usize,
}

/// Figure meshes per outfit and frame, the material, and the simulation time drawn last.
#[derive(Resource, Default)]
pub struct Figures {
    meshes: Vec<[Handle<Mesh>; figures::FRAMES]>,
    material: Option<Handle<StandardMaterial>>,
    time: f64,
}

/// Speed below which a pedestrian stands (m/s).
const STANDING: f64 = 0.1;

/// Figures farther from the camera are hidden (a few pixels tall at 1080p), and those farther
/// than [`SHADOW_RANGE`] cast no shadows (m).
const DRAW_RANGE: f32 = 250.0;
const SHADOW_RANGE: f32 = 60.0;

/// A figure's entity and the components posed each frame.
type FigureParts = (
    Entity,
    &'static mut Figure,
    &'static mut Transform,
    &'static mut Mesh3d,
    &'static mut Visibility,
    Has<NotShadowCaster>,
);

/// Keep a figure per pedestrian (respawned when the crowd's size changes) and pose them.
pub fn sync_crowd(
    mut commands: Commands,
    sim: Res<Sim>,
    origin: Res<RenderOrigin>,
    mut figs: ResMut<Figures>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    camera: Query<&Transform, (With<Camera3d>, Without<Figure>)>,
    mut query: Query<FigureParts>,
) {
    let peds = &sim.world.crowd().peds;
    if query.iter().len() != peds.len() {
        for (e, ..) in &query {
            commands.entity(e).despawn();
        }
        if figs.meshes.is_empty() {
            figs.meshes = (0..figures::OUTFITS)
                .map(|k| std::array::from_fn(|f| meshes.add(convert::mesh(&figures::pedestrian(k, f)))))
                .collect();
        }
        let material = figs
            .material
            .get_or_insert_with(|| materials.add(StandardMaterial { perceptual_roughness: 0.9, ..default() }))
            .clone();
        for index in 0..peds.len() {
            commands.spawn((
                Figure { index, phase: 0.0, frame: 0 },
                Mesh3d(figs.meshes[index % figures::OUTFITS][0].clone()),
                MeshMaterial3d(material.clone()),
                Transform::default(),
                Visibility::Hidden,
            ));
        }
        figs.time = sim.time();
        return;
    }
    let t = sim.time();
    // Paused, or a replay restarted: no steps.
    let dt = (t - figs.time).clamp(0.0, 1.0);
    figs.time = t;
    let eye = camera.single().map_or(Vec3::ZERO, |c| c.translation);
    for (e, mut fig, mut tf, mut mesh, mut vis, shadowless) in &mut query {
        let Some(p) = peds.get(fig.index) else { continue };
        let at = origin.pos(DVec3::new(p.pos.x, p.pos.y, p.z));
        let d = at.distance(eye);
        if d > DRAW_RANGE {
            vis.set_if_neq(Visibility::Hidden);
            continue;
        }
        vis.set_if_neq(Visibility::Inherited);
        match (d > SHADOW_RANGE, shadowless) {
            (true, false) => _ = commands.entity(e).insert(NotShadowCaster),
            (false, true) => _ = commands.entity(e).remove::<NotShadowCaster>(),
            _ => {}
        }
        let scale = p.height / figures::HEIGHT;
        let speed = p.vel.length();
        let (frame, rot) = if p.state == PedState::Hit {
            // Lying on its back.
            (0, from_yaw(p.heading) * DQuat::from_rotation_y(-std::f64::consts::FRAC_PI_2))
        } else if speed < STANDING {
            fig.phase = 0.0;
            (0, from_yaw(p.heading))
        } else {
            fig.phase += speed * dt / (figures::CYCLE_LENGTH * scale);
            (figures::frame(fig.phase), from_yaw(p.heading))
        };
        let pose = Pose::new(DVec3::new(p.pos.x, p.pos.y, p.z), rot);
        tf.set_if_neq(origin.transform(&pose).with_scale(Vec3::splat(scale as f32)));
        if frame != fig.frame {
            fig.frame = frame;
            mesh.0 = figs.meshes[fig.index % figures::OUTFITS][frame].clone();
        }
    }
}
