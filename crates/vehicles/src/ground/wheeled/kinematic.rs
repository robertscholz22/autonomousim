//! Kinematic driving of a wheeled vehicle: a cheap stand-in for the multibody model (traffic
//! NPCs away from the learning agents), which keeps the full model's state up to date so that
//! it can take over (promotion) or hand back (demotion) at any step.
//!
//! **Model**: a kinematic bicycle about the turn-centre axle (the unsteered axles' mean, see
//! [`WheeledDef::steer_reference`](crate::ground::WheeledDef::steer_reference)): that point
//! moves along the heading at `speed`, and the vehicle yaws at `speed·tan(share·δ)/L`, where
//! `δ` is the bicycle steering angle (rate-limited as the full model's) and `L`, `share` the
//! wheelbase and steering share of the most steered axle, as the ground controller has them.
//! Side drives (no steering) yaw at a commanded rate instead. The speed (and a side drive's yaw
//! rate) follows its target in first order, within acceleration and deceleration limits.
//!
//! **Lateral dynamics**: single-unit vehicles with steering add the linear single-track model
//! above walking pace ([`DYNAMIC_SPEED`]): the lateral velocity `v` of the turn-centre point
//! and the yaw rate `r` obey `m (v̇ + x_g ṙ + u r) = ΣF`, `I ṙ = Σ (x_k − x_g) F_k` with
//! `F_k = C_k (δ_k − (v + x_k r)/u)` per wheel (`x` along the chassis from the turn-centre
//! point, `x_g` the centre of mass, `C_k` the tyre's cornering stiffness at nominal load,
//! `δ_k` its steering angle), integrated implicitly. So the vehicle lags its steering and
//! understeers as the multibody model does; like the ground controller, the steering is then
//! trimmed by the integral of the curvature error.
//!
//! **Units behind** (trailers, dollies) follow the hitch kinematics: a unit whose axle point
//! lies `L_t` behind its joint yaws at `θ̇ = v_J·ŷ/L_t` (`v_J` the joint's velocity, `ŷ` the
//! unit's lateral axis), so its axle never slides sideways. Units without wheels, and hinged
//! ones, move rigidly with the unit ahead.
//!
//! **Ground following**: the towing unit sits at its static rest pose in the plane fitted
//! through the ground under its wheels; each unit behind takes the roll of the ground across
//! its wheels, and the pitch about its joint that puts its axle at its rest height above the
//! ground. Suspension travel stays at the static travel, the wheels roll with the ground.
//!
//! No forces act on a kinematic vehicle and it computes no contacts.

use super::{WheelState, Wheeled, wheel_angle};
use crate::ground::def::SteerMode;
use crate::ground::units::{UnitJoint, yaw_pitch_roll};
use autonomousim_core::dynamics::forward_kinematics;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::{wrap_angle, yaw};
use autonomousim_core::terrain::Terrain;
use glam::{DMat2, DMat3, DQuat, DVec2, DVec3};
use smallvec::SmallVec;

/// Speed below which a yaw-rate target is read as a curvature at this speed (m/s).
const CRAWL: f64 = 0.5;

/// Speed (m/s) from which the lateral dynamics act (below: the kinematic bicycle).
pub const DYNAMIC_SPEED: f64 = 1.0;

/// Speed (m/s) above which the steering trim integrates (as the ground controller's).
const TRIM_SPEED: f64 = 2.0;

/// Limits of the kinematic speed (and side drives' yaw-rate) response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KinematicLimits {
    /// Time constant of the first-order response (s).
    pub time_constant: f64,
    /// Largest speed-up and slow-down (m/s²).
    pub max_accel: f64,
    pub max_decel: f64,
    /// Gain (1/s) of the steering trim on the curvature error, and its limit (rad), as the
    /// ground controller's `curvature_integral` and `curvature_correction`.
    pub trim_gain: f64,
    pub trim_limit: f64,
}

impl Default for KinematicLimits {
    fn default() -> Self {
        Self { time_constant: 0.5, max_accel: 3.0, max_decel: 6.0, trim_gain: 1.0, trim_limit: 0.15 }
    }
}

/// What a kinematic vehicle is asked to do (as the ground controller's setpoints).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KinematicTarget {
    /// Forward speed (m/s) and path curvature (1/m, positive left).
    SpeedCurvature { speed: f64, curvature: f64 },
    /// Forward speed (m/s) and yaw rate (rad/s).
    SpeedYawRate { speed: f64, yaw_rate: f64 },
}

/// State of a kinematically driven vehicle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KinematicState {
    /// Chassis frame origin (horizontal position) and heading (rad).
    pub xy: DVec2,
    pub yaw: f64,
    /// Forward and lateral speed of the turn-centre point (m/s) and the yaw rate (rad/s).
    pub speed: f64,
    pub lateral: f64,
    pub yaw_rate: f64,
    /// Bicycle steering angle (rad), and the trim added to its target (rad).
    pub steer: f64,
    pub trim: f64,
    /// Heading of each unit behind the towing unit (rad), unit 1 first.
    pub unit_yaw: SmallVec<[f64; 2]>,
}

/// A unit's geometry for the kinematics, from the vehicle at rest.
#[derive(Clone, Debug)]
struct KinUnit {
    /// Wheels on the unit.
    wheels: Vec<usize>,
    /// Axle point (unit frame: the mean of its unsteered wheels' centres, on the centre line);
    /// `None` for units that move rigidly with the unit ahead.
    axle: Option<DVec3>,
    /// Height of the axle point above the ground at rest (m).
    axle_height: f64,
    /// Rotation relative to the unit ahead at rest, and the unit's pitch at rest (rad).
    rest_joint: DQuat,
    rest_pitch: f64,
}

/// Geometry of the kinematic model, fixed per vehicle.
#[derive(Clone, Debug)]
pub(super) struct KinematicGeometry {
    /// Chassis pose at rest on flat ground at the origin, heading +x.
    rest: Pose,
    /// Turn-centre point on the chassis x axis (m).
    reference: f64,
    /// Wheelbase (m) and steering share of the bicycle model; `None` for side drives.
    steering: Option<(f64, f64)>,
    units: Vec<KinUnit>,
    /// Static suspension travel (m) and rolling radius (m) per wheel.
    travel: Vec<f64>,
    radius: Vec<f64>,
    /// The single-track model's parameters (single-unit vehicles with steering).
    lateral: Option<Lateral>,
}

/// Parameters of the linear single-track model.
#[derive(Clone, Debug)]
struct Lateral {
    /// Mass (kg), yaw inertia about the centre of mass (kg·m²) and its position ahead of the
    /// turn-centre point (m).
    mass: f64,
    inertia: f64,
    x_g: f64,
    /// Per wheel on the towing unit: its position ahead of the turn-centre point (m) and its
    /// cornering stiffness (N/rad).
    wheels: Vec<(usize, f64, f64)>,
}

/// Planar motion of one unit: its frame origin, heading, yaw rate and origin velocity.
#[derive(Clone, Copy, Debug)]
struct Planar {
    origin: DVec2,
    yaw: f64,
    rate: f64,
    velocity: DVec2,
}

impl Planar {
    fn velocity_at(&self, p: DVec2) -> DVec2 {
        let r = p - self.origin;
        self.velocity + self.rate * DVec2::new(-r.y, r.x)
    }
}

fn heading(yaw: f64) -> DVec2 {
    DVec2::from_angle(yaw)
}

impl KinematicGeometry {
    /// A placeholder until the vehicle is built.
    pub(super) fn empty() -> Self {
        Self {
            rest: Pose::IDENTITY,
            reference: 0.0,
            steering: None,
            units: Vec::new(),
            travel: Vec::new(),
            radius: Vec::new(),
            lateral: None,
        }
    }

    /// From vehicle `w` placed at its rest pose at the origin (as [`Wheeled::new`] leaves it).
    pub(super) fn new(w: &Wheeled) -> Self {
        let def = &*w.def;
        let kin = &w.ws.kin;
        let pose = |u: usize| kin.pose[w.units[u].link];
        let steering = def.steering.and_then(|_| {
            let axle = def
                .axles
                .iter()
                .filter(|a| a.share() != 0.0)
                .max_by(|a, b| a.share().abs().total_cmp(&b.share().abs()))?;
            let wheelbase = axle.position.x - def.steer_reference();
            (wheelbase.abs() > 1e-6).then_some((wheelbase, axle.share()))
        });
        let units = (0..w.units.len())
            .map(|u| {
                let wheels: Vec<usize> = (0..def.num_wheels()).filter(|&k| def.wheel_unit(k) == u).collect();
                let unsteered: Vec<usize> = wheels
                    .iter()
                    .copied()
                    .filter(|&k| {
                        let a = &def.axles[def.wheel_axle(k)];
                        a.share() == 0.0 && matches!(a.steer_mode, SteerMode::Ackermann)
                    })
                    .collect();
                let on = if unsteered.is_empty() { &wheels } else { &unsteered };
                let rigid = u == 0 || on.is_empty() || matches!(def.units[u - 1].joint, UnitJoint::Hinge);
                let axle = (!rigid).then(|| {
                    let mean = on.iter().map(|&k| def.wheel_position(k)).sum::<DVec3>() / on.len() as f64;
                    DVec3::new(mean.x, 0.0, mean.z)
                });
                // Units whose axle is not behind their joint cannot trail: rigid.
                let axle = axle.filter(|a| a.x < -0.1);
                let (rest_joint, rest_pitch) = if u == 0 {
                    (DQuat::IDENTITY, 0.0)
                } else {
                    let parent = def.units[u - 1].parent;
                    ((pose(parent).rot.inverse() * pose(u).rot).normalize(), yaw_pitch_roll(pose(u).rot)[1])
                };
                KinUnit {
                    wheels,
                    axle_height: axle.map_or(0.0, |a| pose(u).transform_point(a).z),
                    axle,
                    rest_joint,
                    rest_pitch,
                }
            })
            .collect();
        let st = def.rest_state();
        let n = def.num_wheels();
        let reference = def.steer_reference();
        let lateral = (steering.is_some() && w.units.len() == 1).then(|| {
            // Mass properties of the whole tree in the chassis frame.
            let chassis = w.pose();
            let links = w.model.links();
            let (mut mass, mut moment) = (0.0, DVec3::ZERO);
            for (l, link) in links.iter().enumerate() {
                let c = chassis.inverse_transform_point(kin.pose[l].transform_point(link.inertia.com));
                mass += link.inertia.mass;
                moment += link.inertia.mass * c;
            }
            let com = moment / mass;
            let mut inertia = 0.0;
            for (l, link) in links.iter().enumerate() {
                let rot = DMat3::from_quat(chassis.rot.inverse() * kin.pose[l].rot);
                let i = rot * link.inertia.i_com * rot.transpose();
                let c = chassis.inverse_transform_point(kin.pose[l].transform_point(link.inertia.com)) - com;
                inertia += i.z_axis.z + link.inertia.mass * (c.x * c.x + c.y * c.y);
            }
            let wheels = (0..n)
                .map(|k| {
                    let stiffness = def.wheel_tire(k).initial_state().stiffness[1];
                    (k, def.wheel_position(k).x - reference, stiffness)
                })
                .collect();
            Lateral { mass, inertia, x_g: com.x - reference, wheels }
        });
        Self {
            rest: w.pose(),
            reference,
            steering,
            units,
            travel: (0..n).map(|k| st.map_or(0.0, |s| s.travel[k])).collect(),
            radius: (0..n).map(|k| def.wheel_tire(k).radius() - st.map_or(0.0, |s| s.deflection[k].max(0.0))).collect(),
            lateral,
        }
    }
}

impl Wheeled {
    /// Whether the vehicle can be driven kinematically (not single-track).
    pub fn supports_kinematic(&self) -> bool {
        !self.def.is_single_track() && self.head.is_none()
    }

    /// The kinematic state of the vehicle as it is now (for driving it on kinematically):
    /// the chassis position and heading, the turn-centre point's forward speed, the yaw rate,
    /// the steering angle and the units' headings.
    pub fn kinematic_state(&self) -> KinematicState {
        let poses = self.current_poses();
        let pose = self.pose();
        let psi = yaw(pose.rot);
        let omega = pose.rot * self.ang_vel_body();
        let reference = pose.transform_point(DVec3::X * self.kinematic.reference);
        let v = self.lin_vel_world() + omega.cross(reference - pose.pos);
        let e = heading(psi);
        KinematicState {
            xy: pose.pos.truncate(),
            yaw: psi,
            speed: v.truncate().dot(e),
            lateral: if self.kinematic.lateral.is_some() { v.truncate().dot(e.perp()) } else { 0.0 },
            yaw_rate: omega.z,
            steer: self.steer_angle,
            trim: 0.0,
            unit_yaw: (1..self.units.len()).map(|u| yaw(poses.unit(u).rot)).collect(),
        }
    }

    /// Planar motion of every unit for state `k`.
    fn planar(&self, k: &KinematicState) -> SmallVec<[Planar; 4]> {
        let g = &self.kinematic;
        let e = heading(k.yaw);
        let reference = k.xy + e * g.reference;
        let r = reference - k.xy;
        let v0 = e * k.speed + e.perp() * k.lateral - k.yaw_rate * DVec2::new(-r.y, r.x);
        let mut out: SmallVec<[Planar; 4]> = SmallVec::new();
        out.push(Planar { origin: k.xy, yaw: k.yaw, rate: k.yaw_rate, velocity: v0 });
        for u in 1..self.units.len() {
            let unit = &self.def.units[u - 1];
            let p = out[unit.parent];
            let joint = p.origin + DVec2::from_angle(p.yaw).rotate(unit.position.truncate());
            let v = p.velocity_at(joint);
            let (yaw, rate) = match g.units[u].axle {
                Some(a) => {
                    let yaw = k.unit_yaw[u - 1];
                    let lateral = heading(yaw).perp();
                    (yaw, v.dot(lateral) / -a.x)
                }
                None => (p.yaw, p.rate),
            };
            out.push(Planar { origin: joint, yaw, rate, velocity: v });
        }
        out
    }

    /// Advance kinematic state `k` by one step towards `target` within `limits`, and place the
    /// vehicle there (see [`place_kinematic`](Self::place_kinematic)).
    pub fn step_kinematic<T: Terrain + ?Sized>(
        &mut self,
        k: &mut KinematicState,
        target: KinematicTarget,
        limits: &KinematicLimits,
        terrain: &T,
    ) {
        let dt = self.dt;
        let finite = |x: f64| if x.is_finite() { x } else { 0.0 };
        let v_ref = match target {
            KinematicTarget::SpeedCurvature { speed, .. } | KinematicTarget::SpeedYawRate { speed, .. } => {
                finite(speed)
            }
        };
        // Speed: first order, within the limits (deceleration while slowing down).
        let raw = (v_ref - k.speed) / limits.time_constant;
        let slowing = raw * k.speed < 0.0;
        let a = if slowing {
            raw.clamp(-limits.max_decel, limits.max_decel)
        } else {
            raw.clamp(-limits.max_accel, limits.max_accel)
        };
        let mut speed = k.speed + a * dt;
        if (speed - v_ref) * (k.speed - v_ref) < 0.0 {
            speed = v_ref;
        }
        // The units behind turn with the motion of the step's start.
        let rates: SmallVec<[f64; 4]> = self.planar(k).iter().map(|p| p.rate).collect();
        // Turning: steering (rate-limited to the bicycle angle for the curvature), or the yaw
        // rate of a side drive.
        let (yaw_rate, lateral) = match (self.kinematic.steering, self.def.steering) {
            (Some((wheelbase, share)), Some(s)) => {
                let curvature = match target {
                    KinematicTarget::SpeedCurvature { curvature, .. } => finite(curvature),
                    KinematicTarget::SpeedYawRate { yaw_rate, .. } => {
                        finite(yaw_rate) / speed.abs().max(v_ref.abs()).max(CRAWL)
                    }
                };
                if speed.abs() > TRIM_SPEED && self.kinematic.lateral.is_some() {
                    let error = curvature - k.yaw_rate / speed;
                    k.trim = (k.trim + limits.trim_gain * wheelbase * error * dt)
                        .clamp(-limits.trim_limit, limits.trim_limit);
                }
                let delta = ((curvature * wheelbase).atan() / share + k.trim).clamp(-s.max_angle, s.max_angle);
                k.steer += (delta - k.steer).clamp(-s.rate * dt, s.rate * dt);
                match &self.kinematic.lateral {
                    Some(lat) if speed.abs() >= DYNAMIC_SPEED => self.single_track(lat, k, speed, dt),
                    _ => (speed * (share * k.steer).tan() / wheelbase, 0.0),
                }
            }
            _ => {
                let r_ref = match target {
                    KinematicTarget::SpeedCurvature { speed, curvature } => finite(speed) * finite(curvature),
                    KinematicTarget::SpeedYawRate { yaw_rate, .. } => finite(yaw_rate),
                };
                (k.yaw_rate + (r_ref - k.yaw_rate) * (1.0 - (-dt / limits.time_constant).exp()), 0.0)
            }
        };
        // The turn-centre point moves along the mean heading of the step.
        let reference = k.xy + heading(k.yaw) * self.kinematic.reference;
        let mid = k.yaw + 0.5 * yaw_rate * dt;
        let reference = reference
            + heading(mid) * (0.5 * (k.speed + speed) * dt)
            + heading(mid).perp() * (0.5 * (k.lateral + lateral) * dt);
        k.yaw = wrap_angle(k.yaw + yaw_rate * dt);
        k.xy = reference - heading(k.yaw) * self.kinematic.reference;
        k.speed = speed;
        k.lateral = lateral;
        k.yaw_rate = yaw_rate;
        for (u, y) in k.unit_yaw.iter_mut().enumerate() {
            *y = wrap_angle(*y + rates[u + 1] * dt);
        }
        for c in &self.corners {
            self.state.q[c.spin.0] += self.state.v[c.spin.1] * dt;
        }
        self.place_kinematic(k, terrain);
    }

    /// One implicit Euler step of the single-track model at forward speed `u`: the new yaw rate
    /// and lateral speed of the turn-centre point.
    fn single_track(&self, lat: &Lateral, k: &KinematicState, u: f64, dt: f64) -> (f64, f64) {
        let ackermann = self.def.steering.map_or(0.0, |s| s.ackermann);
        // Forces F0 − F1 v − F2 r and moments M0 − M1 v − M2 r about the centre of mass.
        let (mut f, mut m) = ([0.0; 3], [0.0; 3]);
        for &(w, x, c) in &lat.wheels {
            let corner = &self.corners[w];
            let delta = match corner.steer {
                Some(_) => {
                    let a = wheel_angle(corner.law.angle(k.steer), corner.wheelbase, corner.lateral, ackermann);
                    corner.independent.map_or(a, |(lock, _)| a.clamp(-lock, lock))
                }
                None => 0.0,
            };
            let arm = x - lat.x_g;
            let row = [c * delta, c / u, c * x / u];
            for i in 0..3 {
                f[i] += row[i];
                m[i] += arm * row[i];
            }
        }
        let (mass, inertia, x_g) = (lat.mass, lat.inertia, lat.x_g);
        // ṡ = A s + b for s = (v, r).
        let a = DMat2::from_cols(
            DVec2::new(-f[1] / mass + x_g * m[1] / inertia, -m[1] / inertia),
            DVec2::new(-f[2] / mass - u + x_g * m[2] / inertia, -m[2] / inertia),
        );
        let b = DVec2::new(f[0] / mass - x_g * m[0] / inertia, m[0] / inertia);
        let s = DVec2::new(k.lateral, k.yaw_rate);
        let next = (DMat2::IDENTITY - a * dt).inverse() * (s + b * dt);
        (next.y, next.x)
    }

    /// Place the vehicle at kinematic state `k` on `terrain` (see the module documentation):
    /// poses, velocities, steering, static suspension travel and rolling wheels (their spin
    /// angles are kept). Units that move rigidly get the heading of the unit ahead in `k`.
    pub fn place_kinematic<T: Terrain + ?Sized>(&mut self, k: &mut KinematicState, terrain: &T) {
        let g = self.kinematic.clone();
        let def = self.def.clone();
        let planar = self.planar(k);
        for (u, p) in planar.iter().enumerate().skip(1) {
            k.unit_yaw[u - 1] = p.yaw;
        }
        // The towing unit: its rest pose in the plane fitted through the ground under its
        // wheels (and its origin).
        let turn = DQuat::from_rotation_z(k.yaw);
        let points: SmallVec<[DVec2; 8]> =
            g.units[0].wheels.iter().map(|&w| (turn * def.wheel_position(w)).truncate()).chain([DVec2::ZERO]).collect();
        let (a, b, c) = fit_plane(terrain, k.xy, &points);
        let normal = DVec3::new(-b, -c, 1.0).normalize();
        let forward = turn * DVec3::X;
        let x = (forward - normal * forward.dot(normal)).normalize();
        let frame = DQuat::from_mat3(&DMat3::from_cols(x, normal.cross(x), normal));
        let pose0 = Pose::new(k.xy.extend(a) + frame * g.rest.pos, (frame * g.rest.rot).normalize());
        let v_xy = planar[0].velocity;
        let v0 = v_xy.extend(-(normal.x * v_xy.x + normal.y * v_xy.y) / normal.z);
        let omega0 = DVec3::Z * k.yaw_rate;
        self.state.q[..3].copy_from_slice(&pose0.pos.to_array());
        self.state.q[3..7].copy_from_slice(&pose0.rot.to_array());
        self.state.v[..3].copy_from_slice(&(pose0.rot.inverse() * omega0).to_array());
        self.state.v[3..6].copy_from_slice(&(pose0.rot.inverse() * v0).to_array());
        // Units behind, in order (each after the unit it hangs from).
        let mut rots: SmallVec<[DQuat; 4]> = SmallVec::new();
        rots.push(pose0.rot);
        let mut origins: SmallVec<[DVec3; 4]> = SmallVec::new();
        origins.push(pose0.pos);
        for u in 1..self.units.len() {
            let unit = &def.units[u - 1];
            let (rot_p, parent) = (rots[unit.parent], unit.parent);
            let joint = origins[parent] + rot_p * unit.position;
            let ku = &g.units[u];
            let theta = planar[u].yaw;
            let rot = match (ku.axle, unit.joint) {
                (None, _) | (_, UnitJoint::Hinge) => rot_p * ku.rest_joint,
                (Some(_), UnitJoint::Turntable) => rot_p * DQuat::from_rotation_z(wrap_angle(theta - yaw(rot_p))),
                (Some(axle), UnitJoint::Coupling(_)) => {
                    let turn = DQuat::from_rotation_z(theta);
                    // Roll: the ground's slope across the unit's wheels.
                    let pts: SmallVec<[(f64, f64); 8]> = ku
                        .wheels
                        .iter()
                        .map(|&w| {
                            let p = def.wheel_position(w);
                            let at = joint.truncate() + (turn * p).truncate();
                            (p.y, terrain.height(at.x, at.y))
                        })
                        .collect();
                    let roll = lateral_slope(&pts).atan();
                    // Pitch about the joint that puts the axle at its rest height.
                    let at = joint.truncate() + (turn * axle).truncate();
                    let target = terrain.height(at.x, at.y) + ku.axle_height - joint.z;
                    let z = roll.cos() * axle.z;
                    let mut pitch = ku.rest_pitch;
                    for _ in 0..4 {
                        let f = -axle.x * pitch.sin() + z * pitch.cos() - target;
                        let df = -axle.x * pitch.cos() - z * pitch.sin();
                        if df.abs() < 1e-9 {
                            break;
                        }
                        pitch = (pitch - f / df).clamp(-1.0, 1.0);
                    }
                    turn * DQuat::from_rotation_y(pitch) * DQuat::from_rotation_x(roll)
                }
            };
            let rel = (rot_p.inverse() * rot).normalize();
            let links = self.units[u];
            let rate = planar[u].rate - planar[parent].rate;
            match unit.joint {
                UnitJoint::Coupling(_) => {
                    self.state.q[links.q..links.q + 4].copy_from_slice(&rel.to_array());
                    let w = rot.inverse() * (DVec3::Z * rate);
                    self.state.v[links.v..links.v + 3].copy_from_slice(&w.to_array());
                }
                UnitJoint::Turntable => {
                    self.state.q[links.q] = yaw_pitch_roll(rel)[0];
                    self.state.v[links.v] = rate;
                }
                UnitJoint::Hinge => {
                    self.state.q[links.q] = yaw_pitch_roll(rel)[1];
                    self.state.v[links.v] = 0.0;
                }
            }
            rots.push(rot);
            origins.push(joint);
        }
        // Wheels: static travel, steering, rolling with their unit.
        self.steer_angle = k.steer;
        let ackermann = def.steering.map_or(0.0, |s| s.ackermann);
        for i in 0..self.corners.len() {
            let c = &self.corners[i];
            if let Some((q, v)) = c.travel {
                (self.state.q[q], self.state.v[v]) = (g.travel[i], 0.0);
            }
            let unit = def.wheel_unit(i);
            let angle = match (c.independent, c.forced) {
                (Some((lock, _)), _) => {
                    wheel_angle(c.law.angle(k.steer), c.wheelbase, c.lateral, ackermann).clamp(-lock, lock)
                }
                (_, Some((gain, u))) => gain * self.articulation(u).0,
                _ => wheel_angle(c.law.angle(k.steer), c.wheelbase, c.lateral, ackermann),
            };
            let c = &mut self.corners[i];
            if c.independent.is_some() {
                c.steer_cmd = angle;
            }
            if let Some((q, v)) = c.steer {
                (self.state.q[q], self.state.v[v]) = (angle, 0.0);
            }
            let p = planar[unit];
            let centre = origins[unit].truncate() + (rots[unit] * def.wheel_position(i)).truncate();
            let forward = p.velocity_at(centre).dot(heading(p.yaw));
            self.state.v[c.spin.1] = forward / g.radius[i];
            c.out = WheelState {
                travel: c.travel.map_or(0.0, |_| g.travel[i]),
                steer: angle,
                spin_angle: self.state.q[c.spin.0],
                spin: self.state.v[c.spin.1],
                ..WheelState::default()
            };
        }
        self.specific_force = pose0.rot.inverse() * DVec3::Z * super::super::def::STANDARD_GRAVITY;
        self.ang_acc = DVec3::ZERO;
        self.contacts.clear();
        self.f_ext.fill(autonomousim_core::math::SpatialForce::ZERO);
        forward_kinematics(&self.model, &self.state.q, &self.state.v, &mut self.ws.kin);
    }

    /// Hand the vehicle back to the dynamics after kinematic driving (promotion): the pose,
    /// velocities, steering and wheel speeds stay as placed; tyres, brakes, bands, the
    /// powertrain (in the gear reached accelerating to the wheel speeds, see
    /// [`Powertrain::reset_cruising`](crate::ground::Powertrain::reset_cruising)) and the
    /// contact state start afresh.
    pub fn resume_dynamics(&mut self) {
        for (w, c) in self.corners.iter_mut().enumerate() {
            c.tire = self.def.wheel_tire(w).initial_state();
            c.brake.reset();
            c.brake_delay.fill(0.0);
            c.brake_level = 0.0;
        }
        for b in &mut self.bands {
            b.reset();
        }
        for b in self.steer_brakes.iter_mut().flat_map(|(b, _)| b) {
            b.reset();
        }
        self.head_torque = 0.0;
        let spin: Vec<f64> = self.corners.iter().map(|c| self.state.v[c.spin.1]).collect();
        self.powertrain.reset_cruising(&spin);
        self.cache.clear();
        self.contacts.clear();
        self.f_ext.fill(autonomousim_core::math::SpatialForce::ZERO);
        forward_kinematics(&self.model, &self.state.q, &self.state.v, &mut self.ws.kin);
    }
}

/// Least squares `z = a + b·x + c·y` through the ground at `xy + p` for offsets `p` (the
/// ground's slope at `xy` where they are degenerate).
fn fit_plane<T: Terrain + ?Sized>(terrain: &T, xy: DVec2, points: &[DVec2]) -> (f64, f64, f64) {
    let mut m = DMat3::ZERO;
    let mut rhs = DVec3::ZERO;
    for d in points {
        let z = terrain.height(xy.x + d.x, xy.y + d.y);
        let row = DVec3::new(1.0, d.x, d.y);
        m += DMat3::from_cols(row * row.x, row * row.y, row * row.z);
        rhs += row * z;
    }
    if m.determinant().abs() > 1e-9 {
        let s = m.inverse() * rhs;
        (s.x, s.y, s.z)
    } else {
        let (h, n) = terrain.height_normal(xy.x, xy.y);
        (h, -n.x / n.z, -n.y / n.z)
    }
}

/// Slope `dz/dy` of the least-squares line through `(y, z)` (0 when the `y` do not spread).
fn lateral_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let (my, mz) = (points.iter().map(|p| p.0).sum::<f64>() / n, points.iter().map(|p| p.1).sum::<f64>() / n);
    let syy: f64 = points.iter().map(|p| (p.0 - my).powi(2)).sum();
    let syz: f64 = points.iter().map(|p| (p.0 - my) * (p.1 - mz)).sum();
    if syy > 1e-9 { syz / syy } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets;
    use autonomousim_core::terrain::FlatTerrain;

    const FLAT: FlatTerrain = FlatTerrain { height: 0.0, material: autonomousim_core::material::MaterialId(0) };

    fn car() -> Wheeled {
        Wheeled::new(std::sync::Arc::new(presets::wheeled("sedan_like").expect("preset")), 1e-3)
    }

    #[test]
    fn rest_placement_is_the_rest_pose() {
        let mut w = car();
        let rest = w.pose();
        let mut k = w.kinematic_state();
        w.place_kinematic(&mut k, &FLAT);
        assert!(w.pose().pos.distance(rest.pos) < 1e-12 && w.pose().rot.angle_between(rest.rot) < 1e-9);
    }

    #[test]
    fn steady_turn_has_the_commanded_curvature() {
        let mut w = car();
        let mut k = w.kinematic_state();
        let target = KinematicTarget::SpeedCurvature { speed: 10.0, curvature: 0.02 };
        for _ in 0..10_000 {
            w.step_kinematic(&mut k, target, &KinematicLimits::default(), &FLAT);
        }
        assert!((k.speed - 10.0).abs() < 1e-6, "{}", k.speed);
        assert!((k.yaw_rate - 0.2).abs() < 1e-4, "{}", k.yaw_rate);
        // The wheels roll without slip and the body velocity matches the state.
        assert!((w.kinematic_state().speed - 10.0).abs() < 1e-6);
    }
}
