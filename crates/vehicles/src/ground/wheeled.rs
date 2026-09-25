//! Wheeled vehicle instance: the multibody tree, suspension, steering, brakes, powertrain,
//! tyres, aerodynamic drag and chassis contacts, stepped in phases like the multirotor.
//!
//! Tree per wheel: chassis (free) → carrier (`KcTravel`, if sprung) → steering knuckle
//! (massless revolute about the carrier's z axis, prescribed, if steered) → wheel (revolute
//! about the lateral axis). Suspension, anti-roll bar, drive and brake forces are generalised
//! forces on the travel and spin coordinates, so their reactions reach the chassis through the
//! joints. Steering angles are prescribed trajectories: each step drives the knuckle exactly
//! to its target angle, and the joint torque this needs is reported.

use super::def::{SteerMode, WheeledDef, deflection_at};
use super::powertrain::{Coupling, DriveInput, Powertrain, PowertrainStatus};
use super::tire::{Surface, TireForces, TireState, WheelMotion};
use crate::multirotor::AirData;
use autonomousim_core::contact::{
    ContactCache, ContactModel, ContactPoint, ContactScratch, SphereCollider, StaticScene, compute_contacts,
};
use autonomousim_core::dynamics::{
    AbaWorkspace, DynamicsError, JointType, MbState, MultibodyModel, aba_with_kinematics, forward_kinematics,
    semi_implicit_euler,
};
use autonomousim_core::math::{Pose, RigidInertia, SpatialForce};
use glam::{DMat3, DQuat, DVec3};
use std::sync::Arc;

/// Everything outside the vehicle that one physics step needs.
#[derive(Clone, Copy)]
pub struct GroundStepEnv<'a> {
    pub scene: StaticScene<'a>,
    pub gravity: DVec3,
    pub air: AirData,
}

/// Initial conditions for [`Wheeled::reset`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheeledInit {
    /// Chassis frame pose.
    pub pose: Pose,
    /// Linear velocity in the world frame (m/s); the wheels start rolling with it.
    pub lin_vel_world: DVec3,
    /// Angular velocity in the chassis frame (rad/s).
    pub ang_vel_body: DVec3,
}

/// Per-wheel outputs of the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WheelState {
    /// Suspension travel (m, bump positive) and its rate.
    pub travel: f64,
    pub travel_rate: f64,
    /// Steering angle (rad) and the torque the steering applied (N·m).
    pub steer: f64,
    pub steer_torque: f64,
    /// Spin angle (rad) and rate (rad/s) relative to the carrier.
    pub spin_angle: f64,
    pub spin: f64,
    /// Drive torque (powertrain and differentials) and brake torque on the wheel (N·m).
    pub drive_torque: f64,
    pub brake_torque: f64,
    pub tire: TireForces,
}

#[derive(Clone, Debug)]
struct Corner {
    wheel: usize,
    /// Link whose angular velocity the (non-spinning) carrier has.
    carrier: usize,
    travel: Option<(usize, usize)>,
    steer: Option<(usize, usize)>,
    spin: (usize, usize),
    /// Steering share and the wheel's position for the Ackermann geometry.
    steer_share: f64,
    /// Lock (rad) and rate (rad/s) of independent steering, and its rate-limited angle.
    independent: Option<(f64, f64)>,
    steer_cmd: f64,
    wheelbase: f64,
    lateral: f64,
    preload: f64,
    brake: Coupling,
    tire: TireState,
    out: WheelState,
}

/// One simulated wheeled vehicle.
///
/// Step phases: [`begin_step`](Self::begin_step) → [`apply_drive`](Self::apply_drive) /
/// [`apply_tires`](Self::apply_tires) / [`apply_contacts`](Self::apply_contacts) /
/// [`apply_force`](Self::apply_force) → [`finish_step`](Self::finish_step).
/// [`step`](Self::step) runs them all.
#[derive(Clone, Debug)]
pub struct Wheeled {
    def: Arc<WheeledDef>,
    dt: f64,
    model: MultibodyModel,
    mass: f64,
    corners: Vec<Corner>,
    powertrain: Powertrain,
    colliders: Vec<SphereCollider>,
    contact: ContactModel,
    // State.
    pub state: MbState,
    /// Steering angle of the equivalent bicycle (rad).
    steer_angle: f64,
    /// Specific force and angular acceleration of the chassis frame over the last step.
    specific_force: DVec3,
    ang_acc: DVec3,
    // Step buffers.
    ws: AbaWorkspace,
    tau: Vec<f64>,
    f_ext: Vec<SpatialForce>,
    spin: Vec<f64>,
    drive: Vec<f64>,
    brake: Vec<f64>,
    cache: ContactCache,
    scratch: ContactScratch,
    contacts: Vec<ContactPoint>,
}

impl Wheeled {
    /// Instance for physics step `dt`, at its static rest pose at the origin (or the design
    /// pose for vehicles without a static solution).
    pub fn new(def: Arc<WheeledDef>, dt: f64) -> Self {
        assert!(dt > 0.0, "dt must be positive");
        let c = &def.chassis;
        let mut model = MultibodyModel::new();
        model.add_link(
            "chassis",
            None,
            JointType::Free,
            Pose::IDENTITY,
            RigidInertia::new(c.mass, c.com, DMat3::from_diagonal(c.inertia)),
        );
        let inertia = def.spin_inertia();
        // Reference for the Ackermann geometry: the axles without a steering share (or all).
        let unsteered: Vec<f64> = def.axles.iter().filter(|a| a.steer == 0.0).map(|a| a.position.x).collect();
        let reference = if unsteered.is_empty() {
            def.axles.iter().map(|a| a.position.x).sum::<f64>() / def.axles.len() as f64
        } else {
            unsteered.iter().sum::<f64>() / unsteered.len() as f64
        };
        let mut corners = Vec::with_capacity(def.num_wheels());
        for (w, (axle, side)) in def.wheels().enumerate() {
            let p = def.wheel_position(w);
            let (mut parent, mut frame) = (0, Pose::from_translation(p));
            let mut travel = None;
            if let Some(s) = &axle.suspension {
                let table = s.table(side).expect("validated");
                let link = model.add_link(
                    format!("carrier_{w}"),
                    Some(0),
                    JointType::KcTravel(Arc::new(table)),
                    frame,
                    RigidInertia::diag(s.carrier_mass, s.carrier_inertia),
                );
                travel = Some((model.q_offset(link), model.v_offset(link)));
                (parent, frame) = (link, Pose::IDENTITY);
            }
            let carrier = parent;
            let mut steer = None;
            if axle.is_steered() {
                let link = model.add_link(
                    format!("knuckle_{w}"),
                    Some(parent),
                    JointType::Revolute { axis: DVec3::Z },
                    frame,
                    RigidInertia::ZERO,
                );
                model.set_prescribed(link, true);
                steer = Some((model.q_offset(link), model.v_offset(link)));
                (parent, frame) = (link, Pose::IDENTITY);
            }
            let mut moments = axle.wheel.inertia;
            moments.y = inertia[w];
            let wheel = model.add_link(
                format!("wheel_{w}"),
                Some(parent),
                JointType::Revolute { axis: DVec3::Y },
                frame,
                RigidInertia::diag(axle.wheel.mass, moments),
            );
            corners.push(Corner {
                wheel,
                carrier,
                travel,
                steer,
                spin: (model.q_offset(wheel), model.v_offset(wheel)),
                steer_share: axle.steer,
                independent: match axle.steer_mode {
                    SteerMode::Independent { max_angle, rate } => Some((max_angle, rate)),
                    SteerMode::Ackermann => None,
                },
                steer_cmd: 0.0,
                wheelbase: p.x - reference,
                lateral: p.y,
                preload: def.preload(w),
                brake: Coupling::new(&[w], &[], &inertia, dt),
                tire: def.tire(w / 2).initial_state(),
                out: WheelState::default(),
            });
        }
        let n = model.num_links();
        let mass = def.total_mass();
        let mut s = Self {
            dt,
            ws: AbaWorkspace::new(&model),
            state: model.neutral_state(),
            tau: vec![0.0; model.nv()],
            f_ext: vec![SpatialForce::ZERO; n],
            spin: vec![0.0; def.num_wheels()],
            drive: vec![0.0; def.num_wheels()],
            brake: vec![0.0; def.num_wheels()],
            powertrain: Powertrain::new(&def.powertrain, &inertia, dt),
            colliders: def.sphere_colliders(),
            contact: def.contact.model(mass, dt),
            model,
            mass,
            corners,
            steer_angle: 0.0,
            specific_force: DVec3::ZERO,
            ang_acc: DVec3::ZERO,
            cache: ContactCache::default(),
            scratch: ContactScratch::default(),
            contacts: Vec::new(),
            def,
        };
        let init = s.rest(DVec3::ZERO, 0.0, 0.0);
        s.reset(&init);
        s
    }

    pub fn def(&self) -> &WheeledDef {
        &self.def
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    pub fn model(&self) -> &MultibodyModel {
        &self.model
    }

    /// The static rest pose on flat ground at height `ground.z` below `ground`, heading `yaw`,
    /// moving forward at `speed` (m/s). Falls back to the design pose with nominal tyre
    /// deflection for vehicles without a static solution.
    pub fn rest(&self, ground: DVec3, yaw: f64, speed: f64) -> WheeledInit {
        let heading = DQuat::from_rotation_z(yaw);
        let (height, attitude) = match self.def.static_state(super::def::STANDARD_GRAVITY) {
            Ok(st) => (st.height, st.rotation()),
            Err(_) => {
                let t = self.def.tire(0);
                (t.radius() - deflection_at(t, t.nominal_load()) - self.def.axles[0].position.z, DQuat::IDENTITY)
            }
        };
        let rot = heading * attitude;
        WheeledInit {
            pose: Pose::new(ground + DVec3::Z * height, rot),
            // Along the ground, not along the (pitched) chassis x axis.
            lin_vel_world: heading * DVec3::X * speed,
            ang_vel_body: DVec3::ZERO,
        }
    }

    /// Place the vehicle (suspension at its static travel, wheels rolling with the chassis)
    /// and clear all transient state.
    pub fn reset(&mut self, init: &WheeledInit) {
        let rot = init.pose.rot.normalize();
        self.state = self.model.neutral_state();
        self.state.q[..3].copy_from_slice(&init.pose.pos.to_array());
        self.state.q[3..7].copy_from_slice(&rot.to_array());
        self.state.v[..3].copy_from_slice(&init.ang_vel_body.to_array());
        let v_body = rot.inverse() * init.lin_vel_world;
        self.state.v[3..6].copy_from_slice(&v_body.to_array());
        let st = self.def.static_state(super::def::STANDARD_GRAVITY).ok();
        for (w, c) in self.corners.iter_mut().enumerate() {
            if let (Some((q, _)), Some(st)) = (c.travel, &st) {
                self.state.q[q] = st.travel[w];
            }
            let tire = self.def.tire(w / 2);
            let deflection = st.as_ref().map_or(0.0, |s| s.deflection[w]);
            let v_wheel = v_body + init.ang_vel_body.cross(self.def.wheel_position(w));
            self.state.v[c.spin.1] = v_wheel.x / (tire.radius() - deflection);
            c.tire = tire.initial_state();
            c.brake.reset();
            c.steer_cmd = 0.0;
            c.out = WheelState::default();
        }
        self.steer_angle = 0.0;
        self.specific_force = rot.inverse() * DVec3::Z * super::def::STANDARD_GRAVITY;
        self.ang_acc = DVec3::ZERO;
        let spin: Vec<f64> = self.corners.iter().map(|c| self.state.v[c.spin.1]).collect();
        self.powertrain.reset(&spin);
        self.cache.clear();
        self.contacts.clear();
        self.f_ext.fill(SpatialForce::ZERO);
        forward_kinematics(&self.model, &self.state.q, &self.state.v, &mut self.ws.kin);
    }

    /// Show a recorded state instead of simulating: place the vehicle (as [`reset`](Self::reset))
    /// with the bicycle steering angle `steering`, the wheels' travel, steering and spin angles
    /// and outputs from `wheels` (one per wheel; fewer leave the rest as placed), and the
    /// powertrain status `powertrain`.
    pub fn show(&mut self, init: &WheeledInit, steering: f64, wheels: &[WheelState], powertrain: PowertrainStatus) {
        self.reset(init);
        for (c, w) in self.corners.iter_mut().zip(wheels) {
            if let Some((q, v)) = c.travel {
                (self.state.q[q], self.state.v[v]) = (w.travel, w.travel_rate);
            }
            if let Some((q, _)) = c.steer {
                self.state.q[q] = w.steer;
            }
            (self.state.q[c.spin.0], self.state.v[c.spin.1]) = (w.spin_angle, w.spin);
            c.out = *w;
        }
        self.steer_angle = steering;
        self.powertrain.set_status(powertrain);
        forward_kinematics(&self.model, &self.state.q, &self.state.v, &mut self.ws.kin);
    }

    // ---------------------------------------------------------------- step phases

    /// Forward kinematics and cleared force accumulators.
    pub fn begin_step(&mut self) {
        forward_kinematics(&self.model, &self.state.q, &self.state.v, &mut self.ws.kin);
        self.f_ext.fill(SpatialForce::ZERO);
        self.tau.fill(0.0);
        self.contacts.clear();
    }

    /// Steering, suspension, anti-roll bars, powertrain, differentials, brakes and chassis
    /// drag for driver input `input`.
    pub fn apply_drive(&mut self, input: &DriveInput, air: &AirData) {
        let input = input.clamped();
        let dt = self.dt;
        // Steering: rate-limited bicycle angle; wheel angles from the Ackermann geometry.
        if let Some(s) = &self.def.steering {
            let target = input.steering * s.max_angle;
            self.steer_angle += (target - self.steer_angle).clamp(-s.rate * dt, s.rate * dt);
        }
        let ackermann = self.def.steering.map_or(0.0, |s| s.ackermann);
        for (w, c) in self.corners.iter_mut().enumerate() {
            let Some((lock, rate)) = c.independent else { continue };
            let target = match &input.wheels {
                Some(wc) => wc.steer[w] * lock,
                None => {
                    wheel_angle(self.steer_angle * c.steer_share, c.wheelbase, c.lateral, ackermann).clamp(-lock, lock)
                }
            };
            c.steer_cmd += (target - c.steer_cmd).clamp(-rate * dt, rate * dt);
        }
        // Suspension springs, stops, dampers and anti-roll bars.
        for (k, c) in self.corners.iter().enumerate() {
            let (Some((q, v)), Some(susp)) = (c.travel, &self.def.axles[k / 2].suspension) else { continue };
            let (s, ds) = (self.state.q[q], self.state.v[v]);
            self.tau[v] -= susp.spring_force(s, c.preload) + susp.damper_force(ds);
            if k % 2 == 0 && susp.anti_roll > 0.0 {
                let other = &self.corners[k + 1];
                if let Some((q2, v2)) = other.travel {
                    let f = susp.anti_roll * (s - self.state.q[q2]);
                    self.tau[v] -= f;
                    self.tau[v2] += f;
                }
            }
        }
        // Drive and brake torques on the spin coordinates.
        for (w, c) in self.corners.iter().enumerate() {
            self.spin[w] = self.state.v[c.spin.1];
        }
        self.drive.fill(0.0);
        self.brake.fill(0.0);
        self.powertrain.step(&input, &self.spin, dt, &mut self.drive);
        for (w, c) in self.corners.iter_mut().enumerate() {
            let b = &self.def.axles[w / 2].brake;
            let pedal = (input.wheels.map_or(input.brake, |wc| wc.brake[w]) + input.wheel_brake[w]).min(1.0);
            let limit = pedal * b.max_torque + if input.parking { b.parking_torque } else { 0.0 };
            c.brake.step(limit, &self.spin, dt, &mut self.brake);
            self.tau[c.spin.1] += self.drive[w] + self.brake[w];
        }
        // Chassis drag at the centre of mass.
        let drag = self.def.chassis.drag_area;
        if drag != DVec3::ZERO {
            let v_air = self.lin_vel_body() - self.orientation().inverse() * air.wind;
            let f = -0.5 * air.density * drag * v_air.abs() * v_air;
            self.f_ext[0] += SpatialForce::from_force_at_point(f, self.def.chassis.com);
        }
    }

    /// Tyre forces on terrain `scene.terrain`, with friction and rolling resistance from the
    /// material under each wheel.
    pub fn apply_tires(&mut self, scene: &StaticScene) {
        let kin = &self.ws.kin;
        for (w, c) in self.corners.iter_mut().enumerate() {
            let tire = self.def.tire(w / 2);
            let pose = kin.pose[c.wheel];
            let motion = WheelMotion {
                center: pose.pos,
                axis: pose.rot * DVec3::Y,
                velocity: kin.point_velocity_world(c.wheel, DVec3::ZERO),
                carrier_angvel: kin.angular_velocity_world(c.carrier),
                spin: self.state.v[c.spin.1],
            };
            let contact = tire.contact(scene.terrain, &motion);
            let surface = contact.as_ref().map_or(Surface::REFERENCE, |r| Surface::of(scene.materials.get(r.material)));
            let f = tire.step(&mut c.tire, contact.as_ref(), &motion, surface, self.dt);
            self.f_ext[c.wheel] +=
                SpatialForce::new(pose.inverse_transform_vector(f.torque), pose.inverse_transform_vector(f.force));
            c.out.tire = f;
        }
    }

    /// Penalty contacts of the chassis colliders with the static world.
    pub fn apply_contacts(&mut self, scene: &StaticScene) {
        compute_contacts(
            scene,
            &self.ws.kin,
            &self.colliders,
            &self.contact,
            self.dt,
            &mut self.cache,
            &mut self.scratch,
            &mut self.f_ext,
            &mut self.contacts,
        );
    }

    /// Add an external force (world frame) acting on the chassis at a world point.
    pub fn apply_force(&mut self, force: DVec3, point: DVec3) {
        let pose = self.ws.kin.pose[0];
        self.f_ext[0] += SpatialForce::from_force_at_point(
            pose.inverse_transform_vector(force),
            pose.inverse_transform_point(point),
        );
    }

    /// Forward dynamics (steering prescribed) and integration.
    pub fn finish_step(&mut self, gravity: DVec3) -> Result<(), DynamicsError> {
        let dt = self.dt;
        let ackermann = self.def.steering.map_or(0.0, |s| s.ackermann);
        for c in &self.corners {
            if let Some((q, v)) = c.steer {
                let target = match c.independent {
                    Some(_) => c.steer_cmd,
                    None => wheel_angle(self.steer_angle * c.steer_share, c.wheelbase, c.lateral, ackermann),
                };
                self.ws.qdd[v] = ((target - self.state.q[q]) / dt - self.state.v[v]) / dt;
            }
        }
        aba_with_kinematics(&self.model, &self.tau, &self.f_ext, gravity, &mut self.ws)?;
        // Classical acceleration of the chassis origin: spatial acceleration plus ω × v.
        let qdd = &self.ws.qdd;
        let (w, v) = (self.ang_vel_body(), self.lin_vel_body());
        let accel = DVec3::new(qdd[3], qdd[4], qdd[5]) + w.cross(v);
        self.specific_force = accel - self.orientation().inverse() * gravity;
        self.ang_acc = DVec3::new(qdd[0], qdd[1], qdd[2]);
        semi_implicit_euler(&self.model, &mut self.state, &self.ws.qdd, dt);
        for (w, c) in self.corners.iter_mut().enumerate() {
            let o = &mut c.out;
            if let Some((q, v)) = c.travel {
                (o.travel, o.travel_rate) = (self.state.q[q], self.state.v[v]);
            }
            if let Some((q, v)) = c.steer {
                (o.steer, o.steer_torque) = (self.state.q[q], self.ws.tau_prescribed[v]);
            }
            (o.spin_angle, o.spin) = (self.state.q[c.spin.0], self.state.v[c.spin.1]);
            (o.drive_torque, o.brake_torque) = (self.drive[w], self.brake[w]);
        }
        Ok(())
    }

    /// One full physics step.
    pub fn step(&mut self, input: &DriveInput, env: &GroundStepEnv) -> Result<(), DynamicsError> {
        self.begin_step();
        self.apply_drive(input, &env.air);
        self.apply_tires(&env.scene);
        self.apply_contacts(&env.scene);
        self.finish_step(env.gravity)
    }

    // ---------------------------------------------------------------- state access

    pub fn position(&self) -> DVec3 {
        DVec3::from_slice(&self.state.q[..3])
    }

    /// Chassis → world rotation.
    pub fn orientation(&self) -> DQuat {
        DQuat::from_slice(&self.state.q[3..7])
    }

    pub fn pose(&self) -> Pose {
        Pose::new(self.position(), self.orientation())
    }

    pub fn ang_vel_body(&self) -> DVec3 {
        DVec3::from_slice(&self.state.v[..3])
    }

    /// Velocity of the chassis frame origin, chassis axes.
    pub fn lin_vel_body(&self) -> DVec3 {
        DVec3::from_slice(&self.state.v[3..6])
    }

    pub fn lin_vel_world(&self) -> DVec3 {
        self.orientation() * self.lin_vel_body()
    }

    /// Forward speed (m/s) of the chassis frame origin.
    pub fn speed(&self) -> f64 {
        self.lin_vel_body().x
    }

    /// Total mass (kg).
    pub fn mass(&self) -> f64 {
        self.mass
    }

    pub fn num_wheels(&self) -> usize {
        self.corners.len()
    }

    pub fn wheel(&self, w: usize) -> &WheelState {
        &self.corners[w].out
    }

    pub fn wheels(&self) -> impl Iterator<Item = &WheelState> {
        self.corners.iter().map(|c| &c.out)
    }

    /// World pose of wheel `w`'s spinning link (centre and orientation).
    pub fn wheel_pose(&self, w: usize) -> Pose {
        self.ws.kin.pose[self.corners[w].wheel]
    }

    /// Unloaded tyre radius of wheel `w` (m).
    pub fn wheel_radius(&self, w: usize) -> f64 {
        self.def.tire(w / 2).radius()
    }

    /// Bicycle steering angle (rad).
    pub fn steering_angle(&self) -> f64 {
        self.steer_angle
    }

    pub fn powertrain(&self) -> PowertrainStatus {
        self.powertrain.status()
    }

    pub fn powertrain_mut(&mut self) -> &mut Powertrain {
        &mut self.powertrain
    }

    /// Active chassis contacts of the last step.
    pub fn contacts(&self) -> &[ContactPoint] {
        &self.contacts
    }

    pub fn colliders(&self) -> &[SphereCollider] {
        &self.colliders
    }

    /// Specific force (accelerometer reading at the chassis frame origin, chassis axes) of the
    /// last step: acceleration minus gravity.
    pub fn specific_force_body(&self) -> DVec3 {
        self.specific_force
    }

    /// Angular acceleration of the chassis (chassis axes) over the last step.
    pub fn ang_acc_body(&self) -> DVec3 {
        self.ang_acc
    }
}

/// Angle (rad) of a wheel `wheelbase` ahead of the turn-centre axle and `lateral` to the left,
/// for bicycle angle `delta`: `ackermann` blends from parallel steer (0) to the exact
/// Ackermann angle (1), `tan δ_w = L tan δ / (L − y tan δ)`.
pub fn wheel_angle(delta: f64, wheelbase: f64, lateral: f64, ackermann: f64) -> f64 {
    if ackermann == 0.0 || wheelbase == 0.0 {
        return delta;
    }
    let t = delta.tan();
    let exact = (wheelbase * t).atan2(wheelbase - lateral * t);
    // atan2 picks the branch of the right sign for either direction of travel of the axle.
    let exact = if wheelbase > 0.0 { exact } else { exact - std::f64::consts::PI.copysign(exact) };
    delta + ackermann * (exact - delta)
}

#[cfg(test)]
mod tests {
    use super::wheel_angle;

    #[test]
    fn ackermann_inner_wheel_turns_more() {
        let (l, y, d) = (2.8, 0.8, 0.3f64);
        let (left, right) = (wheel_angle(d, l, y, 1.0), wheel_angle(d, l, -y, 1.0));
        assert!(left > d && right < d);
        // Both wheels' normals meet on the rear-axle line at the same turn centre.
        let r = l / d.tan();
        assert!(((l / left.tan() + y) - r).abs() < 1e-12 && ((l / right.tan() - y) - r).abs() < 1e-12);
        assert_eq!(wheel_angle(d, l, y, 0.0), d);
        assert!((wheel_angle(-d, l, y, 1.0) + wheel_angle(d, l, -y, 1.0)).abs() < 1e-12);
        assert!(wheel_angle(0.0, l, y, 1.0).abs() < 1e-15);
    }
}
