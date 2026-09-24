//! Vehicle visuals: a multirotor's body and rotor discs whose opacity follows the rotor speed;
//! a wheeled vehicle's body, wheels posed from the simulated steering, travel and spin, and a
//! strut and lower arm per suspended wheel that follow the wheel.

use crate::convert;
use crate::sim::Sim;
use autonomousim_scene::mesh::srgb;
use autonomousim_scene::props;
use autonomousim_vehicles::Vehicle;
use bevy::prelude::*;

/// Root entity of agent `0`'s visual.
#[derive(Component)]
pub struct VehicleVisual(pub usize);

#[derive(Component)]
pub struct RotorDisc {
    agent: usize,
    rotor: usize,
}

#[derive(Component)]
pub struct WheelVisual {
    agent: usize,
    wheel: usize,
}

/// A suspension link from a chassis point (chassis frame) to the centre of a wheel.
#[derive(Component)]
pub struct LinkVisual {
    agent: usize,
    wheel: usize,
    mount: glam::DVec3,
}

/// Transform (in the chassis frame) of a unit link (a cylinder along z from −0.5 to 0.5)
/// stretched from `a` to `b`.
fn link_transform(a: glam::DVec3, b: glam::DVec3) -> Transform {
    let d = b - a;
    let rot = glam::DQuat::from_rotation_arc(glam::DVec3::Z, d.normalize_or(glam::DVec3::Z));
    Transform::from_translation(convert::vec(0.5 * (a + b))).with_rotation(convert::quat(rot)).with_scale(Vec3::new(
        1.0,
        d.length().max(1e-3) as f32,
        1.0,
    ))
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
        let pose = sim.render_pose(i);
        let root = (
            VehicleVisual(i),
            Transform::from_translation(convert::vec(pose.pos)).with_rotation(convert::quat(pose.rot)),
            Visibility::default(),
        );
        let def = match &agent.vehicle {
            Vehicle::Multirotor(m) => m.def(),
            Vehicle::Wheeled(w) => {
                let v = props::wheeled(w.def());
                let chassis = w.pose();
                let link = meshes.add(convert::mesh(&v.link));
                commands.spawn(root).with_children(|parent| {
                    parent.spawn((Mesh3d(meshes.add(convert::mesh(&v.body))), MeshMaterial3d(body_material.clone())));
                    for (k, mounts) in v.links.iter().enumerate() {
                        let centre = (chassis.inverse() * w.wheel_pose(k)).pos;
                        for &mount in mounts.iter().flatten() {
                            parent.spawn((
                                Mesh3d(link.clone()),
                                MeshMaterial3d(body_material.clone()),
                                link_transform(mount, centre),
                                LinkVisual { agent: i, wheel: k, mount },
                            ));
                        }
                    }
                    for (k, mesh) in v.wheels.iter().enumerate() {
                        let local = chassis.inverse() * w.wheel_pose(k);
                        parent.spawn((
                            Mesh3d(meshes.add(convert::mesh(mesh))),
                            MeshMaterial3d(body_material.clone()),
                            Transform::from_translation(convert::vec(local.pos))
                                .with_rotation(convert::quat(local.rot)),
                            WheelVisual { agent: i, wheel: k },
                        ));
                    }
                });
                continue;
            }
        };
        let v = props::multirotor(def);
        let disc = meshes.add(convert::mesh(&props::rotor_disc(v.rotor_radius, [1.0; 4])));
        commands.spawn(root).with_children(|parent| {
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

#[allow(clippy::type_complexity)]
pub fn sync_vehicles(
    sim: Res<Sim>,
    mut roots: Query<(&VehicleVisual, &mut Transform), (Without<WheelVisual>, Without<LinkVisual>)>,
    mut wheels: Query<(&WheelVisual, &mut Transform), (Without<VehicleVisual>, Without<LinkVisual>)>,
    mut links: Query<(&LinkVisual, &mut Transform), (Without<VehicleVisual>, Without<WheelVisual>)>,
    discs: Query<(&RotorDisc, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (v, mut t) in &mut roots {
        let pose = sim.render_pose(v.0);
        t.translation = convert::vec(pose.pos);
        t.rotation = convert::quat(pose.rot);
    }
    for (wv, mut t) in &mut wheels {
        let Some(w) = sim.world.agent(wv.agent).vehicle.as_wheeled() else { continue };
        let local = w.pose().inverse() * w.wheel_pose(wv.wheel);
        t.translation = convert::vec(local.pos);
        t.rotation = convert::quat(local.rot);
    }
    for (l, mut t) in &mut links {
        let Some(w) = sim.world.agent(l.agent).vehicle.as_wheeled() else { continue };
        *t = link_transform(l.mount, (w.pose().inverse() * w.wheel_pose(l.wheel)).pos);
    }
    for (d, m) in &discs {
        let Some(vehicle) = sim.world.agent(d.agent).vehicle.as_multirotor() else { continue };
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
