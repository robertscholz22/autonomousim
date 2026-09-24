//! Vehicle visuals: body mesh and rotor discs whose opacity follows the rotor speed.

use crate::convert;
use crate::sim::Sim;
use autonomousim_scene::mesh::srgb;
use autonomousim_scene::props;
use bevy::prelude::*;

/// Root entity of agent `0`'s visual.
#[derive(Component)]
pub struct VehicleVisual(pub usize);

#[derive(Component)]
pub struct RotorDisc {
    agent: usize,
    rotor: usize,
}

pub fn spawn_vehicles(
    mut commands: Commands,
    sim: Res<Sim>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let body_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.5,
        metallic: 0.1,
        ..default()
    });
    for (i, agent) in sim.world.agents().iter().enumerate() {
        let def = agent.vehicle.def();
        let v = props::multirotor(def);
        let disc = meshes.add(convert::mesh(&props::rotor_disc(v.rotor_radius, [1.0; 4])));
        let pose = sim.render_pose(i);
        commands
            .spawn((
                VehicleVisual(i),
                Transform::from_translation(convert::vec(pose.pos)).with_rotation(convert::quat(pose.rot)),
                Visibility::default(),
            ))
            .with_children(|parent| {
                parent.spawn((Mesh3d(meshes.add(convert::mesh(&v.body))), MeshMaterial3d(body_material.clone())));
                for (k, (hub, axis)) in v.rotors.iter().enumerate() {
                    let rotation = glam::DQuat::from_rotation_arc(glam::DVec3::Z, *axis);
                    let spin = def.rotors[k].spin.sign();
                    let tint = if spin > 0.0 { [150, 200, 255] } else { [255, 200, 150] };
                    let c = srgb(tint);
                    let material = materials.add(StandardMaterial {
                        base_color: Color::linear_rgba(c[0], c[1], c[2], 0.3),
                        alpha_mode: AlphaMode::Blend,
                        cull_mode: None,
                        double_sided: true,
                        unlit: true,
                        ..default()
                    });
                    parent.spawn((
                        Mesh3d(disc.clone()),
                        MeshMaterial3d(material),
                        Transform::from_translation(convert::vec(*hub + *axis * (0.02 * v.span as f64)))
                            .with_rotation(convert::quat(rotation)),
                        RotorDisc { agent: i, rotor: k },
                        bevy::light::NotShadowCaster,
                    ));
                }
            });
    }
}

pub fn sync_vehicles(
    sim: Res<Sim>,
    mut roots: Query<(&VehicleVisual, &mut Transform)>,
    discs: Query<(&RotorDisc, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (v, mut t) in &mut roots {
        let pose = sim.render_pose(v.0);
        t.translation = convert::vec(pose.pos);
        t.rotation = convert::quat(pose.rot);
    }
    for (d, m) in &discs {
        let vehicle = &sim.world.agent(d.agent).vehicle;
        let (_, max) = vehicle.speed_range();
        let omega = vehicle.motor_speeds().get(d.rotor).copied().unwrap_or(0.0);
        let alpha = (0.08 + 0.4 * (omega / max.max(1.0))).clamp(0.05, 0.5) as f32;
        if let Some(mut material) = materials.get_mut(&m.0) {
            let c = material.base_color.to_linear();
            if (c.alpha - alpha).abs() > 0.01 {
                material.base_color = Color::linear_rgba(c.red, c.green, c.blue, alpha);
            }
        }
    }
}
