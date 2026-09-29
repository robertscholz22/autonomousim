//! Vehicle rigs: a vehicle's visual as a flat list of rigid parts, each placed in the vehicle's
//! frame from the simulated state (control surfaces, rotor pods and heads, trailer units,
//! wheels, suspension links, a rider), for renderers that draw meshes at poses.
//!
//! This is the viewer's posing without its scene graph: every placement is composed down to
//! the vehicle's root frame (the frame of [`Vehicle::pose`]). What the rig leaves out:
//! translucent rotor discs, and animation that needs time rather than state (a helicopter's
//! blades stand at azimuth 0, track bands keep the shape they had when the rig was built).

use crate::mesh::MeshData;
use crate::props::{self, NacelleVisual, RotorVisual, SurfaceVisual, TiltSurfaceVisual};
use crate::single_track::{Limb, SingleTrackVisual};
use autonomousim_core::math::Pose;
use autonomousim_vehicles::Vehicle;
use autonomousim_vehicles::fixedwing::FixedWing;
use autonomousim_vehicles::ground::Wheeled;
use autonomousim_vehicles::ground::tire::TireModel;
use autonomousim_vehicles::rotorcraft::Helicopter;
use glam::{DQuat, DVec2, DVec3};

/// Where a part is drawn: its mesh scaled (in the mesh frame), then posed in the vehicle frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub pose: Pose,
    pub scale: DVec3,
}

impl Placement {
    pub fn new(pose: Pose) -> Self {
        Self { pose, scale: DVec3::ONE }
    }

    /// A unit link (a mesh along z from −0.5 to 0.5) stretched from `a` to `b`.
    pub fn link(a: DVec3, b: DVec3) -> Self {
        let d = b - a;
        let rot = DQuat::from_rotation_arc(DVec3::Z, d.normalize_or(DVec3::Z));
        Self { pose: Pose::new(0.5 * (a + b), rot), scale: DVec3::new(1.0, 1.0, d.length().max(1e-3)) }
    }

    /// This placement seen from a frame in which its parent sits at `parent`.
    fn under(self, parent: Pose) -> Self {
        Self { pose: parent * self.pose, scale: self.scale }
    }
}

#[derive(Clone, Debug)]
enum Part {
    /// Fixed in the vehicle frame.
    Fixed(Pose),
    Surface(SurfaceVisual),
    TiltSurface(TiltSurfaceVisual),
    Nacelle {
        rotor: usize,
        pivot: DVec3,
    },
    Blade {
        rotor: usize,
        hub: DVec3,
        frame: DQuat,
        blade: u32,
        count: u32,
    },
    Unit(usize),
    Wheel(usize),
    Link {
        wheel: usize,
        mount: DVec3,
    },
    Steered,
    Torso,
    Limb(Limb, f64),
    Boot(f64),
}

/// The parts of one vehicle; [`place`](Self::place) poses them from the vehicle's state.
#[derive(Clone, Debug)]
pub struct Rig {
    /// One mesh per part, in the part's frame.
    pub meshes: Vec<MeshData>,
    parts: Vec<Part>,
    single_track: Option<Box<SingleTrackVisual>>,
}

impl Rig {
    /// The rig of `v` (its definition, and its state for the parts fixed at build time).
    pub fn new(v: &Vehicle) -> Self {
        let mut rig = Self { meshes: Vec::new(), parts: Vec::new(), single_track: None };
        match v {
            Vehicle::Multirotor(m) => rig.push(props::multirotor(m.def()).body, Part::Fixed(Pose::IDENTITY)),
            Vehicle::FixedWing(f) => {
                let visual = props::fixed_wing(f.def());
                rig.push(visual.body, Part::Fixed(Pose::IDENTITY));
                for s in visual.surfaces {
                    rig.push(s.mesh.clone(), Part::Surface(s));
                }
            }
            Vehicle::Tiltrotor(t) => {
                let visual = props::tiltrotor(t.def());
                rig.push(visual.body, Part::Fixed(Pose::IDENTITY));
                for s in visual.surfaces {
                    rig.push(s.mesh.clone(), Part::TiltSurface(s));
                }
                for (rotor, NacelleVisual { mesh, pivot, .. }) in visual.nacelles.into_iter().enumerate() {
                    rig.push(mesh, Part::Nacelle { rotor, pivot });
                }
            }
            Vehicle::Helicopter(h) => {
                let visual = props::helicopter(h.def());
                rig.push(visual.body, Part::Fixed(Pose::IDENTITY));
                for (rotor, RotorVisual { hub, frame, blades, blade, .. }) in visual.rotors.into_iter().enumerate() {
                    for k in 0..blades {
                        rig.push(blade.clone(), Part::Blade { rotor, hub, frame, blade: k, count: blades });
                    }
                }
            }
            Vehicle::Wheeled(w) => {
                let visual = props::wheeled(w.def());
                rig.push(visual.body, Part::Fixed(Pose::IDENTITY));
                for (u, mesh) in visual.units.into_iter().enumerate() {
                    rig.push(mesh, Part::Unit(u + 1));
                }
                for (wheel, mounts) in visual.links.iter().enumerate() {
                    for &mount in mounts.iter().flatten() {
                        rig.push(visual.link.clone(), Part::Link { wheel, mount });
                    }
                }
                for side in 0..2 {
                    let Some(band) = track_band(w, side) else { break };
                    rig.push(band, Part::Fixed(Pose::IDENTITY));
                }
                if let Some(s) = visual.single_track {
                    let rider = w.def().rider.is_some();
                    rig.push(s.steered.clone(), Part::Steered);
                    if rider {
                        rig.push(s.torso.clone(), Part::Torso);
                    }
                    let limbs = [Limb::ForkSlider, Limb::UpperArm, Limb::Forearm, Limb::Thigh, Limb::Shin];
                    for side in [1.0, -1.0] {
                        for limb in limbs {
                            if limb == Limb::ForkSlider || rider {
                                rig.push(s.limbs[limb as usize].clone(), Part::Limb(limb, side));
                            }
                        }
                        if rider {
                            rig.push(s.boot.clone(), Part::Boot(side));
                        }
                    }
                    rig.single_track = Some(Box::new(s));
                }
                for (k, mesh) in visual.wheels.into_iter().enumerate() {
                    rig.push(mesh, Part::Wheel(k));
                }
            }
        }
        rig
    }

    fn push(&mut self, mesh: MeshData, part: Part) {
        self.meshes.push(mesh);
        self.parts.push(part);
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Placements of all parts (in [`meshes`](Self::meshes) order) in the frame of
    /// `v.pose()`, from `v`'s state. `v` must be the vehicle the rig was built for.
    pub fn place(&self, v: &Vehicle, out: &mut Vec<Placement>) {
        out.clear();
        out.extend(self.parts.iter().map(|p| self.place_part(v, p)));
    }

    fn place_part(&self, v: &Vehicle, part: &Part) -> Placement {
        let at = |pos: DVec3, rot: DQuat| Placement::new(Pose::new(pos, rot));
        match (part, v) {
            (Part::Fixed(pose), _) => Placement::new(*pose),
            (Part::Surface(s), Vehicle::FixedWing(f)) => {
                at(s.hinge, DQuat::from_axis_angle(s.axis, surface_angle(f, s)))
            }
            (Part::TiltSurface(s), Vehicle::Tiltrotor(t)) => {
                let g = t.surface_gains().get(s.surface).copied().unwrap_or_default();
                let c = t.channels();
                at(s.hinge, DQuat::from_axis_angle(s.axis, g[0] * c[0] + g[1] * c[1] + g[2] * c[2]))
            }
            (Part::Nacelle { rotor, pivot }, Vehicle::Tiltrotor(t)) => {
                at(*pivot, DQuat::from_rotation_y(t.tilts().get(*rotor).copied().unwrap_or(0.0)))
            }
            (&Part::Blade { rotor, hub, frame, blade, count }, Vehicle::Helicopter(h)) => {
                let (flap, coning) = flapping(h, rotor);
                at(hub, frame * props::rotor_tilt(flap) * props::blade_rotation(blade, count, coning))
            }
            (Part::Unit(u), Vehicle::Wheeled(w)) => Placement::new(unit_local(w, *u)),
            (Part::Wheel(k), Vehicle::Wheeled(w)) => Placement::new(wheel_in_root(w, *k)),
            (&Part::Link { wheel, mount }, Vehicle::Wheeled(w)) => {
                let unit = unit_local(w, w.def().wheel_unit(wheel));
                Placement::link(mount, wheel_local(w, wheel).pos).under(unit)
            }
            (part, Vehicle::Wheeled(w)) => {
                let s = self.single_track.as_deref().expect("single-track parts come with their visual");
                rider_part(s, w, part)
            }
            _ => panic!("rig placed for another vehicle than it was built for"),
        }
    }
}

/// Deflection of `s` in pilot sense (rad): positive commands roll right, pitch up and yaw
/// right whatever the model's sign convention; flaps positive down.
pub fn surface_angle(f: &FixedWing, s: &SurfaceVisual) -> f64 {
    let d = f.surfaces()[s.control];
    if s.control < 3 { d * f.control_signs()[s.control] } else { d }
}

/// Tip-path-plane tilt `[β₁c, β₁s]` and coning (rad) of rotor 0 (main) or 1 (tail).
pub fn flapping(h: &Helicopter, rotor: usize) -> ([f64; 2], f64) {
    if rotor == 0 {
        (h.main_rotor_state().flap, h.loads().main.coning)
    } else {
        (h.tail_rotor_state().flap, h.loads().tail.coning)
    }
}

/// Pose of unit `u` relative to the towing unit. Unit and wheel poses are those of the last
/// step's start, so they are related to each other rather than to the current chassis pose.
pub fn unit_local(w: &Wheeled, u: usize) -> Pose {
    w.unit_pose(0).inverse() * w.unit_pose(u)
}

/// Pose of wheel `k` relative to its unit.
pub fn wheel_local(w: &Wheeled, k: usize) -> Pose {
    w.unit_pose(w.def().wheel_unit(k)).inverse() * w.wheel_pose(k)
}

/// Pose of wheel `k` in the vehicle frame.
fn wheel_in_root(w: &Wheeled, k: usize) -> Pose {
    unit_local(w, w.def().wheel_unit(k)) * wheel_local(w, k)
}

/// Band of side `side` of a tracked vehicle around its sprocket, idler and road wheels as
/// they stand, advanced by the road wheels' mean spin; `None` without tracks.
pub fn track_band(w: &Wheeled, side: usize) -> Option<MeshData> {
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
    let flat = |p: DVec3| DVec2::new(p.x, p.z);
    let mut circles: Vec<(DVec2, f64)> = wheels.iter().map(|&k| (flat(wheel_local(w, k).pos), patch.radius)).collect();
    circles.extend(track.sprocket.iter().chain(&track.idler).map(|r| (flat(r.position), r.radius)));
    let y = def.wheel_position(wheels[0]).y;
    let spin = wheels.iter().map(|&k| w.wheel(k).spin_angle).sum::<f64>() / wheels.len() as f64;
    let thickness = props::track_thickness(patch.radius);
    Some(props::track_band(&circles, y, patch.width, thickness, 0.25 * patch.length, spin * patch.radius))
}

/// Placement (chassis frame) of a single-track vehicle's fork or a part of its rider.
fn rider_part(s: &SingleTrackVisual, w: &Wheeled, part: &Part) -> Placement {
    let steer = w.steering_head().map_or(0.0, |(angle, _)| angle);
    let lean = w.rider_lean().0;
    match *part {
        Part::Steered => Placement::new(Pose::new(s.pivot, s.steer_rotation(steer))),
        Part::Torso => Placement::new(Pose::new(s.hip, SingleTrackVisual::lean_rotation(lean))),
        Part::Limb(limb, side) => {
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
            Placement::link(a, b)
        }
        Part::Boot(side) => {
            let [.., foot] = s.leg_joints(side, w.feet_down());
            Placement::new(Pose::from_translation(foot))
        }
        _ => unreachable!("not a rider part"),
    }
}
