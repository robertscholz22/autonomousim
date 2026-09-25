//! Vehicle visuals: a multirotor's body and rotor discs whose opacity follows the rotor speed;
//! a wheeled vehicle's body, wheels posed from the simulated steering, travel and spin, and a
//! strut and lower arm per suspended wheel that follow the wheel. The units behind a tractor
//! (trailers, dollies, drawbars) are children of the root posed from the simulated joints, and
//! carry their own wheels and links.

use crate::convert;
use crate::sim::Sim;
use autonomousim_core::math::Pose;
use autonomousim_scene::mesh::srgb;
use autonomousim_scene::props;
use autonomousim_vehicles::Vehicle;
use autonomousim_vehicles::ground::Wheeled;
use bevy::prelude::*;

/// Root entity of agent `0`'s visual.
#[derive(Component)]
pub struct VehicleVisual(pub usize);

#[derive(Component)]
pub struct RotorDisc {
    agent: usize,
    rotor: usize,
}

/// Unit `unit` (≥ 1) of a wheeled agent; its transform is relative to the towing unit.
#[derive(Component)]
pub struct UnitVisual {
    agent: usize,
    unit: usize,
}

/// A wheel; its transform is relative to its unit.
#[derive(Component)]
pub struct WheelVisual {
    agent: usize,
    wheel: usize,
}

/// A suspension link from a point of the wheel's unit (its frame) to the centre of a wheel.
#[derive(Component)]
pub struct LinkVisual {
    agent: usize,
    wheel: usize,
    mount: glam::DVec3,
}

/// Transform (in a unit's frame) of a unit link (a cylinder along z from −0.5 to 0.5)
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

/// Pose of unit `u` relative to the towing unit. Unit and wheel poses are those of the last
/// step's start, so they are related to each other rather than to the current chassis pose.
fn unit_local(w: &Wheeled, u: usize) -> Pose {
    w.unit_pose(0).inverse() * w.unit_pose(u)
}

/// Pose of wheel `k` relative to its unit.
fn wheel_local(w: &Wheeled, k: usize) -> Pose {
    w.unit_pose(w.def().wheel_unit(k)).inverse() * w.wheel_pose(k)
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
                let link = meshes.add(convert::mesh(&v.link));
                let root = commands.spawn(root).id();
                let mut parents = vec![root];
                commands
                    .entity(root)
                    .with_child((Mesh3d(meshes.add(convert::mesh(&v.body))), MeshMaterial3d(body_material.clone())));
                for (u, mesh) in v.units.iter().enumerate() {
                    let unit = commands
                        .spawn((
                            Mesh3d(meshes.add(convert::mesh(mesh))),
                            MeshMaterial3d(body_material.clone()),
                            convert::transform(&unit_local(w, u + 1)),
                            UnitVisual { agent: i, unit: u + 1 },
                        ))
                        .id();
                    commands.entity(root).add_child(unit);
                    parents.push(unit);
                }
                for (k, mounts) in v.links.iter().enumerate() {
                    let centre = wheel_local(w, k).pos;
                    for &mount in mounts.iter().flatten() {
                        commands.entity(parents[w.def().wheel_unit(k)]).with_child((
                            Mesh3d(link.clone()),
                            MeshMaterial3d(body_material.clone()),
                            link_transform(mount, centre),
                            LinkVisual { agent: i, wheel: k, mount },
                        ));
                    }
                }
                for (k, mesh) in v.wheels.iter().enumerate() {
                    commands.entity(parents[w.def().wheel_unit(k)]).with_child((
                        Mesh3d(meshes.add(convert::mesh(mesh))),
                        MeshMaterial3d(body_material.clone()),
                        convert::transform(&wheel_local(w, k)),
                        WheelVisual { agent: i, wheel: k },
                    ));
                }
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
    mut roots: Query<
        (&VehicleVisual, &mut Transform),
        (Without<WheelVisual>, Without<LinkVisual>, Without<UnitVisual>),
    >,
    mut units: Query<
        (&UnitVisual, &mut Transform),
        (Without<VehicleVisual>, Without<WheelVisual>, Without<LinkVisual>),
    >,
    mut wheels: Query<
        (&WheelVisual, &mut Transform),
        (Without<VehicleVisual>, Without<LinkVisual>, Without<UnitVisual>),
    >,
    mut links: Query<
        (&LinkVisual, &mut Transform),
        (Without<VehicleVisual>, Without<WheelVisual>, Without<UnitVisual>),
    >,
    discs: Query<(&RotorDisc, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (v, mut t) in &mut roots {
        *t = convert::transform(&sim.render_pose(v.0));
    }
    for (uv, mut t) in &mut units {
        let Some(w) = sim.world.agent(uv.agent).vehicle.as_wheeled() else { continue };
        *t = convert::transform(&unit_local(w, uv.unit));
    }
    for (wv, mut t) in &mut wheels {
        let Some(w) = sim.world.agent(wv.agent).vehicle.as_wheeled() else { continue };
        *t = convert::transform(&wheel_local(w, wv.wheel));
    }
    for (l, mut t) in &mut links {
        let Some(w) = sim.world.agent(l.agent).vehicle.as_wheeled() else { continue };
        *t = link_transform(l.mount, wheel_local(w, l.wheel).pos);
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

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_control::ground::GroundSetpoint;
    use autonomousim_core::rng::Seed;
    use autonomousim_sim::scenario::{GroupSpec, MapSource, Testworld, VehicleRef};
    use autonomousim_sim::{Scenario, WorldInstance};
    use bevy::ecs::system::RunSystemOnce;
    use std::sync::Arc;

    /// A farm rig: the drawbar, dolly and trailer are children of the tractor posed from the
    /// joints, the wheels children of their units; composed, every wheel sits where the
    /// simulation has it.
    #[test]
    fn trailer_units_carry_their_wheels() {
        let sc = Scenario {
            map: MapSource::Testworld(Testworld::Flat { size: 300.0 }),
            groups: vec![GroupSpec {
                vehicle: VehicleRef::Name("farm_tractor".into()),
                trailers: vec!["farm_trailer".into()],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut world = WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(1));
        world.set_command(0, GroundSetpoint::SpeedCurvature { speed: 3.0, curvature: 0.1 });
        for _ in 0..(8.0 / world.scenario().policy_dt()) as usize {
            world.step();
        }
        let mut app = World::new();
        app.insert_resource(Sim::new(world));
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());
        app.run_system_once(spawn_vehicles).unwrap();
        app.run_system_once(sync_vehicles).unwrap();
        let units: Vec<(Entity, usize)> =
            app.query::<(Entity, &UnitVisual)>().iter(&app).map(|(e, u)| (e, u.unit)).collect();
        assert_eq!(units.len(), 3);
        let pose = |t: &Transform| {
            Pose::new(
                convert::enu(t.translation),
                autonomousim_core::math::frames::bevy_to_enu_quat(t.rotation.to_array()),
            )
        };
        let root = pose(app.query::<(&VehicleVisual, &Transform)>().single(&app).unwrap().1);
        let unit_poses: Vec<(Entity, usize, Pose)> =
            units.iter().map(|&(e, k)| (e, k, root * pose(app.get::<Transform>(e).unwrap()))).collect();
        let wheels: Vec<(usize, Pose, Entity)> = app
            .query::<(&WheelVisual, &Transform, &ChildOf)>()
            .iter(&app)
            .map(|(wv, t, parent)| (wv.wheel, pose(t), parent.parent()))
            .collect();
        let sim = app.resource::<Sim>();
        let w = sim.world.agent(0).vehicle.as_wheeled().unwrap();
        assert!(w.articulation(3).0.abs() > 0.05, "{:?}", w.articulation(3));
        let chassis = root * w.unit_pose(0).inverse();
        let mut seen = 0;
        for (k, local, parent) in wheels {
            let u = w.def().wheel_unit(k);
            let unit = match unit_poses.iter().find(|(e, _, _)| *e == parent) {
                Some(&(_, j, p)) => {
                    assert_eq!(j, u);
                    p
                }
                None => {
                    assert_eq!(u, 0);
                    root
                }
            };
            let (shown, want) = (unit * local, chassis * w.wheel_pose(k));
            assert!((shown.pos - want.pos).length() < 1e-4, "wheel {k}: {} vs {}", shown.pos, want.pos);
            seen += 1;
        }
        assert_eq!(seen, w.num_wheels());
    }
}
