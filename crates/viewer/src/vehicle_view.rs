//! Vehicle visuals: a multirotor's body and rotor discs whose opacity follows the rotor speed;
//! a wheeled vehicle's body, wheels posed from the simulated steering, travel and spin, and a
//! strut and lower arm per suspended wheel that follow the wheel; a tracked vehicle's band on
//! each side, wrapped around its sprocket, road wheels and idler and running with the road
//! wheels. The units behind a tractor
//! (trailers, dollies, drawbars) are children of the root posed from the simulated joints, and
//! carry their own wheels and links.

use crate::convert::{self, RenderOrigin};
use crate::sim::Sim;
use autonomousim_core::math::Pose;
use autonomousim_scene::mesh::srgb;
use autonomousim_scene::props;
use autonomousim_scene::single_track::{Limb, SingleTrackVisual};
use autonomousim_vehicles::Vehicle;
use autonomousim_vehicles::ground::Wheeled;
use autonomousim_vehicles::ground::tire::TireModel;
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

/// A posed part of a single-track vehicle or its rider, in the chassis frame.
#[derive(Component)]
pub struct RiderPart {
    agent: usize,
    kind: PartKind,
}

#[derive(Clone, Copy)]
enum PartKind {
    /// Fork tubes, clamps and handlebar, turned about the steering axis.
    Steered,
    /// The rider's upper body, leaned about the hip.
    Torso,
    /// A limb on a side (+1 left, −1 right).
    Limb(Limb, f64),
    Boot(f64),
}

/// The band of a track (side 0: left, 1: right), rebuilt in the chassis frame every frame.
#[derive(Component)]
pub struct TrackVisual {
    agent: usize,
    side: usize,
    mesh: Handle<Mesh>,
}

/// Band of side `side` of a tracked vehicle around its sprocket, idler and road wheels as
/// they stand, advanced by the road wheels' mean spin; `None` without tracks.
fn track_band(w: &Wheeled, side: usize) -> Option<autonomousim_scene::MeshData> {
    let def = w.def();
    let track = def.track.as_ref()?;
    let wheels: Vec<usize> = (0..w.num_wheels())
        .filter(|&k| {
            def.wheel_side(k) == side
                && def.wheel_unit(k) == 0
                && matches!(def.wheel_tire(k).model, TireModel::Track(_))
        })
        .collect();
    let TireModel::Track(patch) = &def.tire(*wheels.first()? / 2).model else { return None };
    let flat = |p: glam::DVec3| glam::DVec2::new(p.x, p.z);
    let mut circles: Vec<(glam::DVec2, f64)> =
        wheels.iter().map(|&k| (flat(wheel_local(w, k).pos), patch.radius)).collect();
    circles.extend(track.sprocket.iter().chain(&track.idler).map(|r| (flat(r.position), r.radius)));
    let y = def.wheel_position(wheels[0]).y;
    let spin = wheels.iter().map(|&k| w.wheel(k).spin_angle).sum::<f64>() / wheels.len() as f64;
    let thickness = props::track_thickness(patch.radius);
    Some(props::track_band(&circles, y, patch.width, thickness, 0.25 * patch.length, spin * patch.radius))
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
    origin: Res<RenderOrigin>,
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
        let root = (VehicleVisual(i), origin.transform(&pose), Visibility::default());
        let def = match &agent.vehicle {
            Vehicle::Multirotor(m) => m.def(),
            Vehicle::FixedWing(f) => {
                // Placeholder until the aircraft get their own visuals: wing, fuselage and
                // wheels as boxes and spheres.
                let d = f.def();
                let g = &d.geometry;
                let length = d.colliders.iter().map(|c| c.center.x).fold(0.0f64, f64::max)
                    - d.colliders.iter().map(|c| c.center.x).fold(0.0f64, f64::min);
                let wing = Cuboid::new(g.chord as f32, g.span as f32, (0.06 * g.chord) as f32);
                let body = Cuboid::new(length.max(g.chord) as f32, (0.12 * g.span) as f32, (0.12 * g.span) as f32);
                commands.spawn(root).with_children(|parent| {
                    parent.spawn((
                        Mesh3d(meshes.add(wing)),
                        MeshMaterial3d(body_material.clone()),
                        Transform::from_translation(convert::vec(g.aero_reference)),
                    ));
                    parent.spawn((Mesh3d(meshes.add(body)), MeshMaterial3d(body_material.clone())));
                    for gear in &d.gear {
                        parent.spawn((
                            Mesh3d(meshes.add(Sphere::new(gear.wheel_radius as f32))),
                            MeshMaterial3d(body_material.clone()),
                            Transform::from_translation(convert::vec(gear.position)),
                        ));
                    }
                });
                continue;
            }
            Vehicle::Wheeled(w) => {
                let v = props::wheeled(w.def());
                let link = meshes.add(convert::mesh(&v.link));
                let root = commands.spawn(root).id();
                if let Some(s) = &v.single_track {
                    commands.entity(root).insert(SingleTrack(Box::new(s.clone())));
                }
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
                for side in 0..2 {
                    let Some(band) = track_band(w, side) else { break };
                    let mesh = meshes.add(convert::mesh(&band));
                    commands.entity(root).with_child((
                        Mesh3d(mesh.clone()),
                        MeshMaterial3d(body_material.clone()),
                        Transform::IDENTITY,
                        TrackVisual { agent: i, side, mesh },
                    ));
                }
                if let Some(s) = &v.single_track {
                    let mut part = |kind: PartKind, mesh: &autonomousim_scene::MeshData| {
                        commands.entity(root).with_child((
                            Mesh3d(meshes.add(convert::mesh(mesh))),
                            MeshMaterial3d(body_material.clone()),
                            part_transform(s, w, kind),
                            RiderPart { agent: i, kind },
                        ));
                    };
                    part(PartKind::Steered, &s.steered);
                    if w.def().rider.is_some() {
                        part(PartKind::Torso, &s.torso);
                    }
                    let limbs = [Limb::ForkSlider, Limb::UpperArm, Limb::Forearm, Limb::Thigh, Limb::Shin];
                    for side in [1.0, -1.0] {
                        for limb in limbs {
                            if limb == Limb::ForkSlider || w.def().rider.is_some() {
                                part(PartKind::Limb(limb, side), &s.limbs[limb as usize]);
                            }
                        }
                        if w.def().rider.is_some() {
                            part(PartKind::Boot(side), &s.boot);
                        }
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
    origin: Res<RenderOrigin>,
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
        *t = origin.transform(&sim.render_pose(v.0));
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

/// The posable parts of a single-track vehicle, on its root.
#[derive(Component)]
pub struct SingleTrack(Box<SingleTrackVisual>);

/// Transform (chassis frame) of part `kind` of single-track vehicle `w`.
fn part_transform(s: &SingleTrackVisual, w: &Wheeled, kind: PartKind) -> Transform {
    let steer = w.steering_head().map_or(0.0, |(angle, _)| angle);
    let lean = w.rider_lean().0;
    match kind {
        PartKind::Steered => convert::transform(&Pose::new(s.pivot, s.steer_rotation(steer))),
        PartKind::Torso => convert::transform(&Pose::new(s.hip, SingleTrackVisual::lean_rotation(lean))),
        PartKind::Limb(limb, side) => {
            let [a, b] = match limb {
                Limb::ForkSlider => s.fork_slider(side, wheel_local(w, s.front_wheel).pos, steer),
                Limb::UpperArm | Limb::Forearm => {
                    let [shoulder, elbow, hand] = s.arm_joints(side, steer, lean);
                    if limb == Limb::UpperArm { [shoulder, elbow] } else { [elbow, hand] }
                }
                Limb::Thigh | Limb::Shin => {
                    let [hip, knee, foot] = s.leg_joints(side, w.feet_down());
                    if limb == Limb::Thigh { [hip, knee] } else { [knee, foot] }
                }
            };
            link_transform(a, b)
        }
        PartKind::Boot(side) => {
            let [.., foot] = s.leg_joints(side, w.feet_down());
            Transform::from_translation(convert::vec(foot))
        }
    }
}

/// Pose the fork, the rider's upper body and limbs from the steering angle, lean and feet.
pub fn sync_riders(
    sim: Res<Sim>,
    roots: Query<&SingleTrack>,
    mut parts: Query<(&RiderPart, &ChildOf, &mut Transform)>,
) {
    for (p, parent, mut t) in &mut parts {
        let (Ok(s), Some(w)) = (roots.get(parent.parent()), sim.world.agent(p.agent).vehicle.as_wheeled()) else {
            continue;
        };
        *t = part_transform(&s.0, w, p.kind);
    }
}

/// Rebuild the track bands from the road wheels' travel and spin.
pub fn sync_tracks(sim: Res<Sim>, tracks: Query<&TrackVisual>, mut meshes: ResMut<Assets<Mesh>>) {
    for t in &tracks {
        let Some(w) = sim.world.agent(t.agent).vehicle.as_wheeled() else { continue };
        if let (Some(band), Some(mut mesh)) = (track_band(w, t.side), meshes.get_mut(&t.mesh)) {
            *mesh = convert::mesh(&band);
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

    /// A motorcycle in a turn, shown steered and leaned: the fork turns with the steering head, the torso
    /// leans with the rider, the fork sliders stand beside the front wheel and the hands and
    /// feet stay where the handlebar and pegs are.
    #[test]
    fn rider_parts_follow_the_state() {
        let sc = Scenario {
            map: MapSource::Testworld(Testworld::Flat { size: 1000.0 }),
            groups: vec![GroupSpec { vehicle: VehicleRef::Name("motorcycle_sport".into()), ..Default::default() }],
            ..Default::default()
        };
        let mut world = WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(1));
        for k in 0..(8.0 / world.scenario().policy_dt()) as usize {
            let curvature = if k > 80 { 0.03 } else { 0.0 };
            world.set_command(0, GroundSetpoint::SpeedCurvature { speed: 10.0, curvature });
            world.step();
        }
        // Shown with more steering and lean than the turn needs.
        let v = world.agent_mut(0).vehicle.as_wheeled_mut().unwrap();
        let init = autonomousim_vehicles::ground::WheeledInit {
            pose: v.pose(),
            lin_vel_world: v.lin_vel_world(),
            ang_vel_body: v.ang_vel_body(),
        };
        let mut wheels: Vec<autonomousim_vehicles::ground::WheelState> = v.wheels().copied().collect();
        let front = v.def().steering_head().unwrap().1;
        wheels[front].steer = 0.2;
        let powertrain = v.powertrain();
        v.show(&init, &[0.25], 0.2, &wheels, powertrain);
        let mut app = World::new();
        app.insert_resource(Sim::new(world));
        app.init_resource::<RenderOrigin>();
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());
        app.run_system_once(spawn_vehicles).unwrap();
        app.run_system_once(sync_riders).unwrap();
        let parts: Vec<(PartKind, Transform)> =
            app.query::<(&RiderPart, &Transform)>().iter(&app).map(|(p, t)| (p.kind, *t)).collect();
        let sim = app.resource::<Sim>();
        let w = sim.world.agent(0).vehicle.as_wheeled().unwrap();
        let (steer, lean) = (w.steering_head().unwrap().0, w.rider_lean().0);
        assert!((steer - 0.2).abs() < 1e-12 && (lean - 0.25).abs() < 1e-12, "steer {steer}, lean {lean}");
        let s = props::wheeled(w.def()).single_track.unwrap();
        let pose = |t: &Transform| {
            Pose::new(
                convert::enu(t.translation),
                autonomousim_core::math::frames::bevy_to_enu_quat(t.rotation.to_array()),
            )
        };
        let mut kinds = 0;
        for (kind, t) in parts {
            let p = pose(&t);
            match kind {
                PartKind::Steered => {
                    assert!((p.pos - s.pivot).length() < 1e-5);
                    assert!((p.rot * s.axis - s.axis).length() < 1e-5);
                    assert!(((p.rot * glam::DVec3::X).y - steer.sin() * s.axis.z).abs() < 1e-3);
                }
                PartKind::Torso => {
                    assert!((p.pos - s.hip).length() < 1e-5);
                    assert!(((p.rot * glam::DVec3::Z).y + lean.sin()).abs() < 1e-5);
                }
                PartKind::Limb(Limb::ForkSlider, side) => {
                    let wheel = wheel_local(w, s.front_wheel).pos;
                    let bottom = p.pos - p.rot * glam::DVec3::Z * (0.5 * s.fork_leg);
                    assert!(((bottom - wheel).length() - s.fork_offset).abs() < 1e-4, "{side}");
                }
                PartKind::Limb(Limb::Forearm, side) => {
                    let [.., hand] = s.arm_joints(side, steer, lean);
                    let end = p.pos + p.rot * glam::DVec3::Z * (0.5 * f64::from(t.scale.y));
                    assert!((end - hand).length() < 1e-4);
                }
                PartKind::Boot(side) => {
                    let foot = if w.feet_down() { s.foot_down } else { s.foot_up };
                    assert!((p.pos - foot * glam::DVec3::new(1.0, side, 1.0)).length() < 1e-5);
                }
                _ => {}
            }
            kinds += 1;
        }
        // The fork and the torso; per side a slider, four limbs and a boot.
        assert_eq!(kinds, 2 + 2 * 6);
    }

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
        app.init_resource::<RenderOrigin>();
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
