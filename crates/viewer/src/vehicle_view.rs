//! Vehicle visuals: a multirotor's body and rotor discs whose opacity follows the rotor speed;
//! a wheeled vehicle's body, wheels posed from the simulated steering, travel and spin, and a
//! strut and lower arm per suspended wheel that follow the wheel; a tracked vehicle's band on
//! each side, wrapped around its sprocket, road wheels and idler and running with the road
//! wheels; an aircraft's airframe with its ailerons, flaps, elevator and rudder turned by the
//! simulated (or recorded) deflections and a propeller disc; a helicopter's airframe with its
//! rotor heads turning, tilted with the tip-path plane, and the blades coned; a tiltrotor's
//! airframe with its flaps turned by the mixed channel deflections and its rotor pods tilted
//! with their mounts. The units behind
//! a tractor (trailers, dollies, drawbars) are children of the root posed from the simulated
//! joints, and carry their own wheels and links.

use crate::convert::{self, RenderOrigin};
use crate::sim::Sim;
use autonomousim_core::math::Pose;
use autonomousim_scene::mesh::srgb;
use autonomousim_scene::props;
use autonomousim_scene::single_track::{Limb, SingleTrackVisual};
use autonomousim_vehicles::Vehicle;
use autonomousim_vehicles::fixedwing::FixedWing;
use autonomousim_vehicles::ground::Wheeled;
use autonomousim_vehicles::ground::tire::TireModel;
use autonomousim_vehicles::rotorcraft::Helicopter;
use bevy::prelude::*;

/// Root entity of agent `0`'s visual.
#[derive(Component)]
pub struct VehicleVisual(pub usize);

#[derive(Component)]
pub struct RotorDisc {
    agent: usize,
    rotor: usize,
}

/// A control surface of a fixed-wing agent, turned about its hinge by the deflection.
#[derive(Component)]
pub struct SurfacePart {
    agent: usize,
    visual: props::SurfaceVisual,
}

/// A surface's transform at `angle` (rad, pilot sense; see [`props::SurfaceVisual`]).
fn surface_transform(s: &props::SurfaceVisual, angle: f64) -> Transform {
    convert::transform(&Pose { pos: s.hinge, rot: glam::DQuat::from_axis_angle(s.axis, angle) })
}

/// Deflection of `s` in pilot sense (rad): positive commands roll right, pitch up and yaw
/// right whatever the model's sign convention; flaps positive down.
fn surface_angle(f: &FixedWing, s: &props::SurfaceVisual) -> f64 {
    let d = f.surfaces()[s.control];
    if s.control < 3 { d * f.control_signs()[s.control] } else { d }
}

/// A flapped surface of a tiltrotor agent, turned about its hinge by its mixed deflection.
#[derive(Component)]
pub struct TiltSurfacePart {
    agent: usize,
    visual: props::TiltSurfaceVisual,
}

fn tilt_surface_transform(s: &props::TiltSurfaceVisual, angle: f64) -> Transform {
    convert::transform(&Pose { pos: s.hinge, rot: glam::DQuat::from_axis_angle(s.axis, angle) })
}

/// A tiltrotor's rotor pod, turned about the body y axis through its pivot by the mount tilt.
#[derive(Component)]
pub struct Nacelle {
    agent: usize,
    rotor: usize,
    pivot: glam::DVec3,
}

fn nacelle_transform(pivot: glam::DVec3, tilt: f64) -> Transform {
    convert::transform(&Pose { pos: pivot, rot: glam::DQuat::from_rotation_y(tilt) })
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

/// Fastest apparent rotor turn (rad/s): faster rotors are drawn turning at this rate, since
/// blades at the true speed alias at frame rates.
const MAX_SHOWN_SPIN: f64 = 9.0;

/// A helicopter's rotor head (0 main, 1 tail): at the hub, the shaft frame tilted with the
/// tip-path plane and turned by the shown azimuth; its blades and disc are children.
#[derive(Component)]
pub struct RotorHead {
    agent: usize,
    rotor: usize,
    hub: glam::DVec3,
    frame: glam::DQuat,
    spin: f64,
    /// Shown azimuth (rad).
    azimuth: f64,
}

/// Blade `Some(k)` of `count` of a [`RotorHead`], coned up; `None` for the disc, raised to
/// the blades' mean height.
#[derive(Component)]
pub struct Blade {
    agent: usize,
    rotor: usize,
    blade: Option<u32>,
    count: u32,
    radius: f32,
}

/// Tip-path-plane tilt `[β₁c, β₁s]` and coning (rad) of rotor 0 (main) or 1 (tail).
fn flapping(h: &Helicopter, rotor: usize) -> ([f64; 2], f64) {
    if rotor == 0 {
        (h.main_rotor_state().flap, h.loads().main.coning)
    } else {
        (h.tail_rotor_state().flap, h.loads().tail.coning)
    }
}

/// Transform (body frame) of a rotor head: at the hub, the shaft frame tilted with the
/// tip-path plane and turned to `azimuth` in the rotor's sense.
fn rotor_head_transform(r: &props::RotorVisual, azimuth: f64, flap: [f64; 2]) -> Transform {
    head_transform(r.hub, r.frame, r.spin * azimuth, flap)
}

fn head_transform(hub: glam::DVec3, frame: glam::DQuat, angle: f64, flap: [f64; 2]) -> Transform {
    convert::transform(&Pose { pos: hub, rot: frame * props::rotor_tilt(flap) * glam::DQuat::from_rotation_z(angle) })
}

/// Transform (head frame) of blade `k` of `count`, coned up by `coning` (rad).
fn blade_transform(k: u32, count: u32, coning: f64) -> Transform {
    convert::transform(&Pose { pos: glam::DVec3::ZERO, rot: props::blade_rotation(k, count, coning) })
}

/// Turn the helicopters' rotor heads by the shown rotor speed over the frame's simulated time,
/// tilt them with the tip-path plane and cone the blades.
/// Turn the tiltrotors' flaps by their mixed deflections and their pods by the mount tilts.
pub fn sync_tiltrotors(
    sim: Res<Sim>,
    mut surfaces: Query<(&TiltSurfacePart, &mut Transform), Without<Nacelle>>,
    mut pods: Query<(&Nacelle, &mut Transform), Without<TiltSurfacePart>>,
) {
    for (sp, mut t) in &mut surfaces {
        let Some(v) = sim.world.agent(sp.agent).vehicle.as_tiltrotor() else { continue };
        let gains = v.surface_gains().get(sp.visual.surface).copied().unwrap_or_default();
        let c = v.channels();
        *t = tilt_surface_transform(&sp.visual, gains[0] * c[0] + gains[1] * c[1] + gains[2] * c[2]);
    }
    for (n, mut t) in &mut pods {
        let Some(v) = sim.world.agent(n.agent).vehicle.as_tiltrotor() else { continue };
        *t = nacelle_transform(n.pivot, v.tilts().get(n.rotor).copied().unwrap_or(0.0));
    }
}

pub fn sync_rotors(
    time: Res<Time>,
    sim: Res<Sim>,
    mut heads: Query<(&mut RotorHead, &mut Transform), Without<Blade>>,
    mut blades: Query<(&Blade, &mut Transform), Without<RotorHead>>,
) {
    let dt = if sim.paused { 0.0 } else { f64::from(time.delta_secs()) * sim.time_scale };
    for (mut head, mut t) in &mut heads {
        let Some(h) = sim.world.agent(head.agent).vehicle.as_helicopter() else { continue };
        let omega = if head.rotor == 0 { h.rotor_speed() } else { h.tail_rotor_state().omega };
        head.azimuth = (head.azimuth + dt * omega.min(MAX_SHOWN_SPIN)).rem_euclid(std::f64::consts::TAU);
        *t = head_transform(head.hub, head.frame, head.spin * head.azimuth, flapping(h, head.rotor).0);
    }
    for (b, mut t) in &mut blades {
        let Some(h) = sim.world.agent(b.agent).vehicle.as_helicopter() else { continue };
        let coning = flapping(h, b.rotor).1;
        *t = match b.blade {
            Some(k) => blade_transform(k, b.count, coning),
            None => {
                Transform::from_translation(convert::vec(glam::DVec3::Z * (0.6 * f64::from(b.radius) * coning.sin())))
            }
        };
    }
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
                let v = props::fixed_wing(f.def());
                let root = commands.spawn(root).id();
                commands
                    .entity(root)
                    .with_child((Mesh3d(meshes.add(convert::mesh(&v.body))), MeshMaterial3d(body_material.clone())));
                for s in &v.surfaces {
                    commands.entity(root).with_child((
                        Mesh3d(meshes.add(convert::mesh(&s.mesh))),
                        MeshMaterial3d(body_material.clone()),
                        surface_transform(s, 0.0),
                        SurfacePart { agent: i, visual: s.clone() },
                    ));
                }
                let (hub, axis) = v.propeller;
                let material = materials.add(StandardMaterial {
                    base_color: Color::linear_rgba(0.1, 0.1, 0.1, 0.3),
                    alpha_mode: AlphaMode::Blend,
                    cull_mode: None,
                    double_sided: true,
                    unlit: true,
                    ..default()
                });
                commands.entity(root).with_child((
                    Mesh3d(meshes.add(convert::mesh(&props::rotor_disc(v.propeller_radius, [1.0; 4])))),
                    MeshMaterial3d(material),
                    Transform::from_translation(convert::vec(hub))
                        .with_rotation(convert::quat(glam::DQuat::from_rotation_arc(glam::DVec3::Z, axis))),
                    RotorDisc { agent: i, rotor: 0 },
                    bevy::light::NotShadowCaster,
                ));
                continue;
            }
            Vehicle::Tiltrotor(t) => {
                let v = props::tiltrotor(t.def());
                let root = commands.spawn(root).id();
                commands
                    .entity(root)
                    .with_child((Mesh3d(meshes.add(convert::mesh(&v.body))), MeshMaterial3d(body_material.clone())));
                for s in &v.surfaces {
                    commands.entity(root).with_child((
                        Mesh3d(meshes.add(convert::mesh(&s.mesh))),
                        MeshMaterial3d(body_material.clone()),
                        tilt_surface_transform(s, 0.0),
                        TiltSurfacePart { agent: i, visual: s.clone() },
                    ));
                }
                for (k, n) in v.nacelles.iter().enumerate() {
                    let material = materials.add(StandardMaterial {
                        base_color: Color::linear_rgba(0.1, 0.1, 0.1, 0.3),
                        alpha_mode: AlphaMode::Blend,
                        cull_mode: None,
                        double_sided: true,
                        unlit: true,
                        ..default()
                    });
                    let pod = commands
                        .spawn((
                            nacelle_transform(n.pivot, t.tilts().get(k).copied().unwrap_or(0.0)),
                            Visibility::default(),
                            Nacelle { agent: i, rotor: k, pivot: n.pivot },
                        ))
                        .id();
                    commands.entity(root).add_child(pod);
                    commands.entity(pod).with_child((
                        Mesh3d(meshes.add(convert::mesh(&n.mesh))),
                        MeshMaterial3d(body_material.clone()),
                    ));
                    commands.entity(pod).with_child((
                        Mesh3d(meshes.add(convert::mesh(&props::rotor_disc(n.radius, [1.0; 4])))),
                        MeshMaterial3d(material),
                        Transform::from_translation(convert::vec(glam::DVec3::Z * n.offset)),
                        RotorDisc { agent: i, rotor: k },
                        bevy::light::NotShadowCaster,
                    ));
                }
                continue;
            }
            Vehicle::Helicopter(h) => {
                let v = props::helicopter(h.def());
                let root = commands.spawn(root).id();
                commands
                    .entity(root)
                    .with_child((Mesh3d(meshes.add(convert::mesh(&v.body))), MeshMaterial3d(body_material.clone())));
                let blade_material = materials.add(StandardMaterial {
                    base_color: Color::srgb(0.16, 0.17, 0.18),
                    perceptual_roughness: 0.6,
                    ..default()
                });
                for (k, r) in v.rotors.iter().enumerate() {
                    let material = materials.add(StandardMaterial {
                        base_color: Color::linear_rgba(0.1, 0.1, 0.1, 0.3),
                        alpha_mode: AlphaMode::Blend,
                        cull_mode: None,
                        double_sided: true,
                        unlit: true,
                        ..default()
                    });
                    let head = commands
                        .spawn((
                            rotor_head_transform(r, 0.0, [0.0; 2]),
                            Visibility::default(),
                            RotorHead { agent: i, rotor: k, hub: r.hub, frame: r.frame, spin: r.spin, azimuth: 0.0 },
                        ))
                        .id();
                    commands.entity(root).add_child(head);
                    let blade =
                        |b: Option<u32>| Blade { agent: i, rotor: k, blade: b, count: r.blades, radius: r.radius };
                    commands.entity(head).with_child((
                        Mesh3d(meshes.add(convert::mesh(&props::rotor_disc(r.radius, [1.0; 4])))),
                        MeshMaterial3d(material),
                        Transform::IDENTITY,
                        RotorDisc { agent: i, rotor: k },
                        blade(None),
                        bevy::light::NotShadowCaster,
                    ));
                    let mesh = meshes.add(convert::mesh(&r.blade));
                    for b in 0..r.blades {
                        commands.entity(head).with_child((
                            Mesh3d(mesh.clone()),
                            MeshMaterial3d(blade_material.clone()),
                            blade_transform(b, r.blades, 0.0),
                            blade(Some(b)),
                        ));
                    }
                }
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
    mut surfaces: Query<
        (&SurfacePart, &mut Transform),
        (Without<VehicleVisual>, Without<WheelVisual>, Without<LinkVisual>, Without<UnitVisual>),
    >,
    discs: Query<(&RotorDisc, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (sp, mut t) in &mut surfaces {
        let Some(f) = sim.world.agent(sp.agent).vehicle.as_fixed_wing() else { continue };
        *t = surface_transform(&sp.visual, surface_angle(f, &sp.visual));
    }

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
        let vehicle = &sim.world.agent(d.agent).vehicle;
        let (omega, max) = if let Some(v) = vehicle.as_multirotor() {
            (v.motor_speeds().get(d.rotor).copied().unwrap_or(0.0), v.speed_range().1)
        } else if let Some(f) = vehicle.as_fixed_wing() {
            (f.rotor_speed(), f.full_throttle_speed())
        } else if let Some(h) = vehicle.as_helicopter() {
            (h.rotor_speed(), h.def().engine.rated_speed)
        } else if let Some(v) = vehicle.as_tiltrotor() {
            (v.rotor_speeds().get(d.rotor).copied().unwrap_or(0.0), v.full_throttle_speed())
        } else {
            continue;
        };
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

    /// A helicopter in forward flight: a head per rotor with its blades and disc; the main
    /// rotor head's axis is the shaft tilted with the simulated tip-path plane, and the blades
    /// are coned up.
    #[test]
    fn rotor_heads_follow_the_flapping() {
        use autonomousim_control::rotorcraft::HelicopterSetpoint;
        let sc = Scenario {
            map: MapSource::Testworld(Testworld::Flat { size: 2000.0 }),
            groups: vec![GroupSpec {
                vehicle: VehicleRef::Name("bo105_like".into()),
                spawn: autonomousim_sim::scenario::SpawnSpec { agl: [30.0, 30.0], ..Default::default() },
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut world = WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(1));
        let setpoint = HelicopterSetpoint::Velocity { velocity: glam::DVec3::new(15.0, 0.0, 0.0), yaw_rate: 0.0 };
        for _ in 0..(10.0 / world.scenario().policy_dt()) as usize {
            world.set_command(0, setpoint);
            world.step();
        }
        let mut app = World::new();
        app.insert_resource(Sim::new(world));
        app.init_resource::<RenderOrigin>();
        app.init_resource::<Time>();
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());
        app.run_system_once(spawn_vehicles).unwrap();
        app.run_system_once(sync_rotors).unwrap();
        let rot = |t: &Transform| autonomousim_core::math::frames::bevy_to_enu_quat(t.rotation.to_array());
        let heads: Vec<(usize, glam::DQuat)> =
            app.query::<(&RotorHead, &Transform)>().iter(&app).map(|(h, t)| (h.rotor, rot(t))).collect();
        let blades: Vec<(usize, Option<u32>, glam::DQuat)> =
            app.query::<(&Blade, &Transform)>().iter(&app).map(|(b, t)| (b.rotor, b.blade, rot(t))).collect();
        let sim = app.resource::<Sim>();
        let h = sim.world.agent(0).vehicle.as_helicopter().unwrap();
        let d = h.def();
        assert_eq!(heads.len(), 2);
        let count = |r: usize| blades.iter().filter(|b| b.0 == r && b.1.is_some()).count() as u32;
        assert_eq!([count(0), count(1)], [d.main_rotor.rotor.blades, d.tail_rotor.rotor.blades]);
        assert_eq!(blades.iter().filter(|b| b.1.is_none()).count(), 2);
        let flap = h.main_rotor_state().flap;
        assert!(flap[0].abs() > 1e-3, "{flap:?}");
        let (_, main) = heads.iter().find(|h| h.0 == 0).unwrap();
        let frame = glam::DQuat::from_mat3(&d.main_rotor.frame());
        let want = frame * props::rotor_tilt(flap) * glam::DVec3::Z;
        assert!((*main * glam::DVec3::Z - want).length() < 1e-5);
        let coning = h.loads().main.coning;
        assert!(coning > 0.01);
        for (_, _, q) in blades.iter().filter(|b| b.0 == 0 && b.1.is_some()) {
            assert!(((*q * glam::DVec3::X).z - coning.sin()).abs() < 1e-5);
        }
    }

    /// A tiltrotor shown mid-conversion with deflected controls: each pod turns about its
    /// pivot by its mount's tilt (the thrust axis tips forward), and each flap by the mixed
    /// channel deflections.
    #[test]
    fn tilt_pods_and_flaps_follow_the_state() {
        let sc = Scenario {
            map: MapSource::Testworld(Testworld::Flat { size: 2000.0 }),
            groups: vec![GroupSpec {
                vehicle: VehicleRef::Name("quadtilt_like".into()),
                spawn: autonomousim_sim::scenario::SpawnSpec { agl: [30.0, 30.0], ..Default::default() },
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut world = WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(1));
        let t = world.agent_mut(0).vehicle.as_tiltrotor_mut().unwrap();
        let n = t.rotor_count();
        let tilts: Vec<f64> = (0..n).map(|k| 0.3 + 0.2 * k as f64).collect();
        t.show(&autonomousim_vehicles::tiltrotor::TiltrotorDisplay {
            rotor_speeds: vec![300.0; n],
            throttles: vec![0.5; n],
            tilts: tilts.clone(),
            channels: [0.1, -0.15, 0.05],
            electric_power: 100.0,
            airspeed: 10.0,
            alpha: 0.05,
            beta: 0.0,
        });
        let mut app = World::new();
        app.insert_resource(Sim::new(world));
        app.init_resource::<RenderOrigin>();
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());
        app.run_system_once(spawn_vehicles).unwrap();
        app.run_system_once(sync_tiltrotors).unwrap();
        let pose = |t: &Transform| {
            Pose::new(
                convert::enu(t.translation),
                autonomousim_core::math::frames::bevy_to_enu_quat(t.rotation.to_array()),
            )
        };
        let pods: Vec<(usize, glam::DVec3, Pose)> =
            app.query::<(&Nacelle, &Transform)>().iter(&app).map(|(n, t)| (n.rotor, n.pivot, pose(t))).collect();
        let flaps: Vec<(props::TiltSurfaceVisual, Pose)> = app
            .query::<(&TiltSurfacePart, &Transform)>()
            .iter(&app)
            .map(|(s, t)| (s.visual.clone(), pose(t)))
            .collect();
        let sim = app.resource::<Sim>();
        let t = sim.world.agent(0).vehicle.as_tiltrotor().unwrap();
        assert_eq!(pods.len(), n);
        for (k, pivot, p) in pods {
            assert!((p.pos - pivot).length() < 1e-5 && (pivot - t.def().rotors[k].pivot).length() < 1e-12);
            let axis = p.rot * glam::DVec3::Z;
            assert!((axis - glam::DVec3::new(tilts[k].sin(), 0.0, tilts[k].cos())).length() < 1e-5, "{k}: {axis}");
        }
        assert!(!flaps.is_empty());
        for (v, p) in flaps {
            let g = t.surface_gains()[v.surface];
            let angle = g[0] * 0.1 - g[1] * 0.15 + g[2] * 0.05;
            let (axis, got) = p.rot.to_axis_angle();
            let got = if axis.dot(v.axis) < 0.0 { -got } else { got };
            assert!((got - angle).abs() < 1e-5 && (p.pos - v.hinge).length() < 1e-5, "{}: {got} vs {angle}", v.surface);
        }
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
