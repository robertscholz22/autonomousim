//! Tiltrotor instance: parameters, state and per-step force model.
//!
//! Each step the surface and tilt servos move; every rotor gives its thrust and torque for the
//! axial flow at its hub (the fixed-wing propeller model: `C_T(J)`, `C_Q(J)` on an electric
//! motor, with Cheeseman–Bennett ground effect on the thrust) and an in-plane drag
//! `−K_d·ω·v_⊥` for the flow across its disc; the motors drive the rotor speeds. The body
//! carries the thrusts and in-plane forces at the hubs, the reaction of each motor torque
//! `−s·Q_m·axis`, the moment of tilting a spinning rotor `−s·I·ω·ȧxis`, the rotors' gyroscopic
//! moment, the lifting surfaces and the fuselage drag.

use super::def::{MAX_ROTORS, TiltMount, TiltrotorDef};
use crate::aero::{AirData, AirFlow, GroundPlane, ground_effect};
use crate::fixedwing::{Propulsion, PropulsionOutput};
use crate::multirotor::StepEnv;
use autonomousim_core::contact::{
    ContactCache, ContactModel, ContactPoint, ContactScratch, SphereCollider, StaticScene, compute_contacts,
};
use autonomousim_core::dynamics::{
    AbaWorkspace, DynamicsError, JointType, MbState, MultibodyModel, aba_with_kinematics, forward_kinematics,
    semi_implicit_euler_with_momentum,
};
use autonomousim_core::math::{Pose, SpatialForce};
use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Pilot inputs: rotor throttles in [0, 1], mount tilts (rad, 0 up, π/2 forward; clamped to
/// the mounts' range), and aileron, elevator and rudder in [−1, 1] (roll right, nose up, nose
/// right). Entries beyond the vehicle's rotors are ignored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TiltrotorInput {
    pub throttle: [f64; MAX_ROTORS],
    pub tilt: [f64; MAX_ROTORS],
    pub aileron: f64,
    pub elevator: f64,
    pub rudder: f64,
}

impl TiltrotorInput {
    /// Within range; non-finite values read as 0.
    pub fn clamped(self) -> Self {
        let f = |x: f64, lo: f64| if x.is_finite() { x.clamp(lo, 1.0) } else { 0.0 };
        Self {
            throttle: self.throttle.map(|t| f(t, 0.0)),
            tilt: self.tilt.map(|t| if t.is_finite() { t } else { 0.0 }),
            aileron: f(self.aileron, -1.0),
            elevator: f(self.elevator, -1.0),
            rudder: f(self.rudder, -1.0),
        }
    }

    /// All throttles at `throttle` and all tilts at `tilt`, surfaces centred.
    pub fn uniform(throttle: f64, tilt: f64) -> Self {
        Self { throttle: [throttle; MAX_ROTORS], tilt: [tilt; MAX_ROTORS], ..Self::default() }
    }
}

/// Initial conditions for [`Tiltrotor::reset`]: servos already at the controls, and the
/// rotors at their steady speed for the controls' throttles in the initial flow (in still air
/// of the given density) unless `rotor_speed` gives them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TiltrotorInit {
    pub pose: Pose,
    /// Linear velocity in the world frame (m/s).
    pub lin_vel_world: DVec3,
    /// Angular velocity in the body frame (rad/s).
    pub ang_vel_body: DVec3,
    /// Controls held from the start; also the vehicle's hold input.
    pub controls: TiltrotorInput,
    pub rotor_speed: Option<[f64; MAX_ROTORS]>,
    /// Air density the rotors are settled in (kg/m³).
    pub density: f64,
}

impl TiltrotorInit {
    /// At rest at `pose`, rotors stopped and pointing up.
    pub fn at_rest(pose: Pose) -> Self {
        Self {
            pose,
            lin_vel_world: DVec3::ZERO,
            ang_vel_body: DVec3::ZERO,
            controls: TiltrotorInput::default(),
            rotor_speed: None,
            density: AirData::default().density,
        }
    }
}

/// Aerodynamic loads on the airframe for one state (body frame, about the centre of mass),
/// without the motor torque reactions: total force and moment, and per rotor the thrust (N,
/// with ground effect), the propeller torque (N·m) and the axial speed into the disc (m/s).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TiltrotorLoads {
    pub force: DVec3,
    pub moment: DVec3,
    pub thrust: [f64; MAX_ROTORS],
    pub prop_torque: [f64; MAX_ROTORS],
    pub v_axial: [f64; MAX_ROTORS],
}

/// Loads of one rotor (body frame, about the centre of mass): thrust (N, with ground effect),
/// propeller torque (N·m), axial speed into the disc (m/s), force and moment.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RotorLoad {
    pub thrust: f64,
    pub torque: f64,
    pub v_axial: f64,
    pub force: DVec3,
    pub moment: DVec3,
}

/// Recorded state that [`Tiltrotor::show`] displays.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TiltrotorDisplay {
    pub rotor_speeds: Vec<f64>,
    pub throttles: Vec<f64>,
    pub tilts: Vec<f64>,
    pub channels: [f64; 3],
    pub electric_power: f64,
    pub airspeed: f64,
    pub alpha: f64,
    pub beta: f64,
}

/// One simulated tiltrotor.
///
/// Step phases: [`begin_step`](Self::begin_step) → [`apply_controls`](Self::apply_controls) /
/// [`apply_contacts`](Self::apply_contacts) / [`apply_force`](Self::apply_force) →
/// [`finish_step`](Self::finish_step). [`step`](Self::step) runs them all.
#[derive(Clone, Debug)]
pub struct Tiltrotor {
    def: Arc<TiltrotorDef>,
    dt: f64,
    body: MultibodyModel,
    mass: f64,
    inertia: DMat3,
    colliders: Vec<SphereCollider>,
    contact: ContactModel,
    propulsion: Propulsion,
    /// Mixing gains of each surface.
    gains: Vec<[f64; 3]>,
    rotors: usize,
    // State.
    pub state: MbState,
    omega: [f64; MAX_ROTORS],
    tilt: [f64; MAX_ROTORS],
    /// Aileron, elevator and rudder channel deflections (rad).
    channels: [f64; 3],
    hold: TiltrotorInput,
    input: TiltrotorInput,
    // Step buffers and outputs.
    tilt_rate: [f64; MAX_ROTORS],
    omega_next: [f64; MAX_ROTORS],
    outputs: [PropulsionOutput; MAX_ROTORS],
    flow: AirFlow,
    loads: TiltrotorLoads,
    ws: AbaWorkspace,
    f_ext: [SpatialForce; 1],
    h_rotor: DVec3,
    ang_acc: DVec3,
    cache: ContactCache,
    scratch: ContactScratch,
    contacts: Vec<ContactPoint>,
}

impl Tiltrotor {
    /// Instance for physics step `dt`, at rest at the origin.
    pub fn new(def: Arc<TiltrotorDef>, dt: f64) -> Self {
        assert!(dt > 0.0, "dt must be positive");
        let mut body = MultibodyModel::new();
        body.add_link("body", None, JointType::Free, Pose::IDENTITY, def.body.rigid_inertia());
        let mut s = Self {
            dt,
            ws: AbaWorkspace::new(&body),
            state: body.neutral_state(),
            body,
            mass: def.body.mass,
            inertia: def.body.inertia_matrix(),
            colliders: def.sphere_colliders(),
            contact: def.contact.model(def.body.mass, dt),
            propulsion: def.propulsion(),
            gains: def.mixing_gains(),
            rotors: def.rotors.len(),
            omega: [0.0; MAX_ROTORS],
            tilt: [0.0; MAX_ROTORS],
            channels: [0.0; 3],
            hold: TiltrotorInput::default(),
            input: TiltrotorInput::default(),
            tilt_rate: [0.0; MAX_ROTORS],
            omega_next: [0.0; MAX_ROTORS],
            outputs: [PropulsionOutput::default(); MAX_ROTORS],
            flow: AirFlow::default(),
            loads: TiltrotorLoads::default(),
            f_ext: [SpatialForce::ZERO],
            h_rotor: DVec3::ZERO,
            ang_acc: DVec3::ZERO,
            cache: ContactCache::default(),
            scratch: ContactScratch::default(),
            contacts: Vec::new(),
            def,
        };
        s.reset(&TiltrotorInit::at_rest(Pose::IDENTITY));
        s
    }

    pub fn def(&self) -> &TiltrotorDef {
        &self.def
    }

    pub fn shared_def(&self) -> &Arc<TiltrotorDef> {
        &self.def
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Number of rotors.
    pub fn rotor_count(&self) -> usize {
        self.rotors
    }

    pub fn propulsion(&self) -> &Propulsion {
        &self.propulsion
    }

    /// Place the tiltrotor, set the controls and servos, spin up the rotors and clear contacts
    /// and outputs.
    pub fn reset(&mut self, init: &TiltrotorInit) {
        let rot = init.pose.rot.normalize();
        self.state.q[..3].copy_from_slice(&init.pose.pos.to_array());
        self.state.q[3..7].copy_from_slice(&rot.to_array());
        self.state.v[..3].copy_from_slice(&init.ang_vel_body.to_array());
        let v_body = rot.inverse() * init.lin_vel_world;
        self.state.v[3..6].copy_from_slice(&v_body.to_array());
        let controls = init.controls.clamped();
        self.hold = controls;
        self.input = controls;
        self.channels = self.channel_targets(&controls);
        for k in 0..MAX_ROTORS {
            self.tilt[k] = if k < self.rotors { self.def.mount_tilt(k, controls.tilt[k]) } else { 0.0 };
        }
        let air = AirData { density: init.density, ..AirData::default() };
        let flow = AirFlow::new(&air, v_body, init.ang_vel_body);
        for k in 0..self.rotors {
            let axis = TiltMount::axis(self.tilt[k]);
            let v_axial = flow.at(self.def.rotors[k].hub(self.tilt[k])).dot(axis);
            self.omega[k] = match init.rotor_speed {
                Some(w) => w[k].max(0.0),
                None => self.propulsion.steady_omega(controls.throttle[k], v_axial, init.density, 0.0),
            };
        }
        self.omega_next = self.omega;
        self.tilt_rate = [0.0; MAX_ROTORS];
        self.outputs = [PropulsionOutput::default(); MAX_ROTORS];
        self.flow = flow;
        self.loads = self.airframe_loads(&flow, &self.omega, &self.tilt, &self.channels, &[1.0; MAX_ROTORS]);
        self.h_rotor = self.rotor_momentum(&self.omega, &self.tilt);
        self.ang_acc = DVec3::ZERO;
        self.cache.clear();
        self.contacts.clear();
        self.f_ext = [SpatialForce::ZERO];
        forward_kinematics(&self.body, &self.state.q, &self.state.v, &mut self.ws.kin);
    }

    /// Channel deflections (rad) the controls ask for: aileron, elevator, rudder.
    pub fn channel_targets(&self, c: &TiltrotorInput) -> [f64; 3] {
        let k = &self.def.controls;
        [k.aileron.pitch(c.aileron), k.elevator.pitch(c.elevator), k.rudder.pitch(c.rudder)]
    }

    /// Channel deflections for inputs continued linearly beyond their range (for trimming).
    pub(super) fn channels_linear(&self, aileron: f64, elevator: f64, rudder: f64) -> [f64; 3] {
        let k = &self.def.controls;
        let lin = |ch: &crate::rotorcraft::PitchChannel, w: f64| 0.5 * (ch.min + ch.max) + 0.5 * w * (ch.max - ch.min);
        [lin(&k.aileron, aileron), lin(&k.elevator, elevator), lin(&k.rudder, rudder)]
    }

    fn rotor_momentum(&self, omega: &[f64; MAX_ROTORS], tilt: &[f64; MAX_ROTORS]) -> DVec3 {
        let i = self.propulsion.inertia;
        (0..self.rotors).map(|k| TiltMount::axis(tilt[k]) * (self.def.rotors[k].sense * i * omega[k])).sum()
    }

    /// Loads (see [`TiltrotorLoads`]) for rotor speeds `omega`, tilts `tilt`, channel
    /// deflections `channels` (rad) and ground-effect thrust ratios `ge`.
    pub fn airframe_loads(
        &self,
        flow: &AirFlow,
        omega: &[f64; MAX_ROTORS],
        tilt: &[f64; MAX_ROTORS],
        channels: &[f64; 3],
        ge: &[f64; MAX_ROTORS],
    ) -> TiltrotorLoads {
        let mut l = TiltrotorLoads::default();
        for k in 0..self.rotors {
            let r = self.rotor_load(k, flow, omega[k], tilt[k], ge[k]);
            l.force += r.force;
            l.moment += r.moment;
            l.thrust[k] = r.thrust;
            l.prop_torque[k] = r.torque;
            l.v_axial[k] = r.v_axial;
        }
        let (f, m) = self.aero_loads(flow, channels);
        l.force += f;
        l.moment += m;
        l
    }

    /// Loads of rotor `k` turning at `omega` (rad/s) on its mount at tilt `tilt` (rad), with
    /// ground-effect thrust ratio `ge`: thrust along the axis and in-plane drag at the hub.
    pub fn rotor_load(&self, k: usize, flow: &AirFlow, omega: f64, tilt: f64, ge: f64) -> RotorLoad {
        let axis = TiltMount::axis(tilt);
        let hub = self.def.rotors[k].hub(tilt);
        let v = flow.at(hub);
        let v_axial = v.dot(axis);
        let (thrust, torque) = self.propulsion.prop_loads(omega, v_axial, flow.density);
        let thrust = if thrust > 0.0 { thrust * ge } else { thrust };
        let force = axis * thrust - (v - axis * v_axial) * (self.def.rotor_drag * omega);
        RotorLoad { thrust, torque, v_axial, force, moment: hub.cross(force) }
    }

    /// Force and moment of the lifting surfaces at channel deflections `channels` (rad) and
    /// of the fuselage.
    pub fn aero_loads(&self, flow: &AirFlow, channels: &[f64; 3]) -> (DVec3, DVec3) {
        let (mut force, mut moment) = (DVec3::ZERO, DVec3::ZERO);
        for (s, g) in self.def.surfaces.iter().zip(&self.gains) {
            let d = g[0] * channels[0] + g[1] * channels[1] + g[2] * channels[2];
            let (f, m) = s.wrench(flow, d);
            force += f;
            moment += m;
        }
        let fus = &self.def.fuselage;
        let v = flow.at(fus.position);
        let drag = -0.5 * flow.density * v.length() * (fus.drag_area * v);
        (force + drag, moment + fus.position.cross(drag))
    }

    /// Mixing gains `[aileron, elevator, rudder]` of each surface.
    pub fn surface_gains(&self) -> &[[f64; 3]] {
        &self.gains
    }

    /// Rotor speed (rad/s) at which a rotor gives `thrust` (N) at axial speed `v_axial` (m/s)
    /// in air of density `rho`, by Newton's method from `guess`.
    pub fn rotor_speed_for(&self, thrust: f64, v_axial: f64, rho: f64, guess: f64) -> f64 {
        let t = |w: f64| self.propulsion.prop_loads(w, v_axial, rho).0;
        let mut w = guess.max(10.0);
        for _ in 0..6 {
            let h = 1e-3 * w + 0.1;
            let slope = (t(w + h) - t(w - h)) / (2.0 * h);
            if !(slope.is_finite() && slope > 1e-9) {
                break;
            }
            let next = (w - (t(w) - thrust) / slope).clamp(0.5 * w, 2.0 * w);
            let done = (next - w).abs() < 1e-6 * w;
            w = next;
            if done {
                break;
            }
        }
        w
    }

    /// Throttle that drives a rotor turning at `omega` toward `target` (rad/s) with time
    /// constant `lag` (s) at axial speed `v_axial` in air of density `rho`: the motor torque
    /// carries the propeller torque plus `I·(target − ω)/lag`, so it holds `target` once there.
    /// Unclamped.
    pub fn throttle_for(&self, omega: f64, target: f64, v_axial: f64, rho: f64, lag: f64) -> f64 {
        let m = &self.def.motor;
        let k = 60.0 / (2.0 * std::f64::consts::PI * m.kv);
        let q = self.propulsion.prop_loads(omega, v_axial, rho).1;
        let i = (q + self.propulsion.inertia * (target - omega) / lag) / k + m.no_load_current;
        (k * omega + i * m.resistance) / m.voltage.unwrap_or(1.0)
    }

    /// Steady thrust (N) of one rotor at full throttle at axial speed `v_axial` (m/s).
    pub fn max_thrust(&self, v_axial: f64, rho: f64) -> f64 {
        let w = self.propulsion.steady_omega(1.0, v_axial, rho, 0.0);
        self.propulsion.prop_loads(w, v_axial, rho).0
    }

    /// Reaction on the airframe of the motor torques `q_motor` (body frame).
    fn motor_reaction(&self, tilt: &[f64; MAX_ROTORS], q_motor: &[f64; MAX_ROTORS]) -> DVec3 {
        (0..self.rotors).map(|k| -TiltMount::axis(tilt[k]) * (self.def.rotors[k].sense * q_motor[k])).sum()
    }

    /// Total non-gravitational force and moment (body frame) with the rotors turning steadily
    /// (motor torque = propeller torque) at `omega`, out of ground effect.
    pub fn steady_wrench(
        &self,
        flow: &AirFlow,
        omega: &[f64; MAX_ROTORS],
        tilt: &[f64; MAX_ROTORS],
        channels: &[f64; 3],
    ) -> (DVec3, DVec3, TiltrotorLoads) {
        let loads = self.airframe_loads(flow, omega, tilt, channels, &[1.0; MAX_ROTORS]);
        (loads.force, loads.moment + self.motor_reaction(tilt, &loads.prop_torque), loads)
    }

    // ---------------------------------------------------------------- step phases

    /// Forward kinematics and cleared force accumulators.
    pub fn begin_step(&mut self) {
        forward_kinematics(&self.body, &self.state.q, &self.state.v, &mut self.ws.kin);
        self.f_ext = [SpatialForce::ZERO];
        self.contacts.clear();
    }

    /// Servos, rotors, motors, surfaces and fuselage for the pilot input. `ground` is the
    /// surface below (ground effect).
    pub fn apply_controls(&mut self, input: &TiltrotorInput, air: &AirData, ground: Option<&GroundPlane>) {
        let input = input.clamped();
        self.input = input;
        let targets = self.channel_targets(&input);
        let k = &self.def.controls;
        for (i, ch) in k.surface_channels().into_iter().enumerate() {
            self.channels[i] = ch.servo(self.channels[i], targets[i], self.dt);
        }
        for r in 0..self.rotors {
            let target = self.def.mount_tilt(r, input.tilt[r]);
            let next = if self.def.rotors[r].tilting { k.tilt.servo(self.tilt[r], target, self.dt) } else { target };
            self.tilt_rate[r] = (next - self.tilt[r]) / self.dt;
            self.tilt[r] = next;
        }
        let (pos, rot) = (self.position(), self.orientation());
        let flow = air.flow(rot, self.lin_vel_world(), self.ang_vel_body());
        let radius = self.def.rotor_radius();
        let mut ge = [1.0; MAX_ROTORS];
        if let Some(g) = ground {
            for (r, m) in self.def.rotors.iter().enumerate() {
                let z = g.distance_along(pos + rot * m.hub(self.tilt[r]), rot * TiltMount::axis(self.tilt[r]));
                if z.is_finite() {
                    ge[r] = ground_effect(z, radius);
                }
            }
        }
        let loads = self.airframe_loads(&flow, &self.omega, &self.tilt, &self.channels, &ge);

        // Motors (the new speeds take effect in finish_step), their reactions and the moment
        // of tilting the spinning rotors.
        let i = self.propulsion.inertia;
        let mut q_motor = [0.0; MAX_ROTORS];
        let mut tilting = DVec3::ZERO;
        for r in 0..self.rotors {
            let out = self.propulsion.step(
                self.omega[r],
                input.throttle[r],
                loads.v_axial[r],
                air.density,
                true,
                0.0,
                self.dt,
            );
            q_motor[r] = out.engine_torque;
            self.omega_next[r] = out.omega;
            self.outputs[r] = PropulsionOutput { thrust: loads.thrust[r], ..out };
            let (s, c) = self.tilt[r].sin_cos();
            let axis_dot = DVec3::new(c, 0.0, -s) * self.tilt_rate[r];
            tilting -= axis_dot * (self.def.rotors[r].sense * i * self.omega[r]);
        }
        self.h_rotor = self.rotor_momentum(&self.omega, &self.tilt);
        let reaction = self.motor_reaction(&self.tilt, &q_motor);
        self.f_ext[0] += SpatialForce::new(loads.moment + reaction + tilting, loads.force);
        self.flow = flow;
        self.loads = loads;
    }

    /// Penalty contacts of the colliders with the static world.
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

    /// Add an external force (world frame) acting at a world point.
    pub fn apply_force(&mut self, force: DVec3, point: DVec3) {
        let pose = self.ws.kin.pose[0];
        self.f_ext[0] += SpatialForce::from_force_at_point(
            pose.inverse_transform_vector(force),
            pose.inverse_transform_point(point),
        );
    }

    /// Forward dynamics and integration of the body and rotor speeds.
    pub fn finish_step(&mut self, gravity: DVec3) -> Result<(), DynamicsError> {
        aba_with_kinematics(&self.body, &[0.0; 6], &self.f_ext, gravity, &mut self.ws)?;
        let w1 = self.ang_vel_body();
        semi_implicit_euler_with_momentum(&self.body, &mut self.state, &self.ws.qdd, self.dt, self.h_rotor);
        self.ang_acc = (self.ang_vel_body() - w1) / self.dt;
        self.omega = self.omega_next;
        Ok(())
    }

    /// One full physics step.
    pub fn step(&mut self, input: &TiltrotorInput, env: &StepEnv) -> Result<(), DynamicsError> {
        self.begin_step();
        self.apply_controls(input, &env.air, env.ground.as_ref());
        if let Some(scene) = &env.scene {
            self.apply_contacts(scene);
        }
        self.finish_step(env.gravity)
    }

    // ---------------------------------------------------------------- state access

    pub fn position(&self) -> DVec3 {
        DVec3::from_slice(&self.state.q[..3])
    }

    /// Body → world rotation.
    pub fn orientation(&self) -> DQuat {
        DQuat::from_slice(&self.state.q[3..7])
    }

    pub fn pose(&self) -> Pose {
        Pose::new(self.position(), self.orientation())
    }

    pub fn ang_vel_body(&self) -> DVec3 {
        DVec3::from_slice(&self.state.v[..3])
    }

    pub fn lin_vel_body(&self) -> DVec3 {
        DVec3::from_slice(&self.state.v[3..6])
    }

    pub fn lin_vel_world(&self) -> DVec3 {
        self.orientation() * self.lin_vel_body()
    }

    pub fn mass(&self) -> f64 {
        self.mass
    }

    /// Inertia matrix about the centre of mass (body frame).
    pub fn inertia(&self) -> DMat3 {
        self.inertia
    }

    /// Rotor speeds (rad/s).
    pub fn rotor_speeds(&self) -> &[f64] {
        &self.omega[..self.rotors]
    }

    /// Mount tilts (rad, 0 up, π/2 forward).
    pub fn tilts(&self) -> &[f64] {
        &self.tilt[..self.rotors]
    }

    /// Aileron, elevator and rudder channel deflections (rad).
    pub fn channels(&self) -> [f64; 3] {
        self.channels
    }

    /// Propulsion outputs of the last step per rotor (thrust with ground effect, torques,
    /// current, electric power).
    pub fn rotor_outputs(&self) -> &[PropulsionOutput] {
        &self.outputs[..self.rotors]
    }

    /// Show recorded state without stepping (replay): rotor speeds (rad/s), throttles, mount
    /// tilts (rad), surface channel deflections (rad), electric power of all motors (W, shared
    /// evenly), airspeed (m/s), angle of attack and sideslip (rad). Entries beyond the
    /// vehicle's rotors are ignored; missing ones leave the state as it is.
    pub fn show(&mut self, d: &TiltrotorDisplay) {
        let n = self.rotors;
        for (k, &w) in d.rotor_speeds.iter().take(n).enumerate() {
            self.omega[k] = w.max(0.0);
            self.omega_next[k] = self.omega[k];
        }
        for (k, &x) in d.tilts.iter().take(n).enumerate() {
            self.tilt[k] = x;
        }
        for (k, &x) in d.throttles.iter().take(n).enumerate() {
            self.input.throttle[k] = x;
        }
        self.channels = d.channels;
        for o in &mut self.outputs[..n] {
            o.electric_power = d.electric_power / n as f64;
        }
        self.h_rotor = self.rotor_momentum(&self.omega, &self.tilt);
        self.flow.airspeed = d.airspeed;
        self.flow.alpha = d.alpha;
        self.flow.beta = d.beta;
    }

    /// Static rotor speed at full throttle at sea level (rad/s), a scale for displays.
    pub fn full_throttle_speed(&self) -> f64 {
        self.propulsion.steady_omega(1.0, 0.0, crate::aero::SEA_LEVEL_DENSITY, 0.0)
    }

    /// Angular momentum of the spinning rotors (body frame, N·m·s).
    pub fn rotor_momentum_body(&self) -> DVec3 {
        self.h_rotor
    }

    /// Electric power drawn by all motors in the last step (W).
    pub fn electric_power(&self) -> f64 {
        self.rotor_outputs().iter().map(|o| o.electric_power).sum()
    }

    /// Input of the last step.
    pub fn input(&self) -> &TiltrotorInput {
        &self.input
    }

    /// Input the tiltrotor was reset with.
    pub fn hold_input(&self) -> &TiltrotorInput {
        &self.hold
    }

    /// Air-relative flow at the centre of mass in the last step.
    pub fn flow(&self) -> &AirFlow {
        &self.flow
    }

    /// Loads of the last step.
    pub fn loads(&self) -> &TiltrotorLoads {
        &self.loads
    }

    pub fn contacts(&self) -> &[ContactPoint] {
        &self.contacts
    }

    pub fn colliders(&self) -> &[SphereCollider] {
        &self.colliders
    }

    /// Specific force (accelerometer reading at the centre of mass, body frame) of the last
    /// step.
    pub fn specific_force_body(&self) -> DVec3 {
        self.f_ext[0].lin / self.mass
    }

    /// Total non-gravitational wrench of the last step (body frame, about the centre of mass).
    pub fn external_wrench(&self) -> SpatialForce {
        self.f_ext[0]
    }

    pub fn ang_acc_body(&self) -> DVec3 {
        self.ang_acc
    }

    /// Mechanical energy: kinetic (body and rotors) plus potential for gravity `g` (J).
    pub fn energy(&self, g: f64) -> f64 {
        let (v, w) = (self.lin_vel_body(), self.ang_vel_body());
        let rotors: f64 = self.rotor_speeds().iter().map(|o| 0.5 * self.propulsion.inertia * o * o).sum();
        0.5 * self.mass * v.length_squared()
            + 0.5 * w.dot(self.inertia * w)
            + rotors
            + self.mass * g * self.position().z
    }

    /// Power of the external forces of the last step on the airframe, plus the net torque on
    /// each rotor times its speed (W): the rate of change of [`energy`](Self::energy) apart
    /// from gravity (with the mounts at rest).
    pub fn external_power(&self) -> f64 {
        let f = self.f_ext[0];
        let rotors: f64 = (0..self.rotors)
            .map(|r| (self.outputs[r].engine_torque - self.outputs[r].prop_torque) * self.omega[r])
            .sum();
        f.lin.dot(self.lin_vel_body()) + f.ang.dot(self.ang_vel_body()) + rotors
    }
}
