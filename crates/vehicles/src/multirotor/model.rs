//! Multirotor instance: parameters (possibly randomised), state and per-step force model.

use super::MAX_ROTORS;
use super::aero::{AirData, GroundPlane, ground_effect};
use super::battery::BatteryState;
use super::def::MultirotorDef;
use autonomousim_core::contact::{
    ContactCache, ContactModel, ContactPoint, ContactScratch, SphereCollider, StaticScene, compute_contacts,
};
use autonomousim_core::dynamics::{
    AbaWorkspace, DynamicsError, JointType, MbState, MultibodyModel, aba_with_kinematics, forward_kinematics,
    semi_implicit_euler_with_momentum,
};
use autonomousim_core::math::{Pose, RigidInertia, SpatialForce};
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Arc;

type PerRotor<T> = SmallVec<[T; MAX_ROTORS]>;

/// Multiplicative parameter perturbations for domain randomisation (all 1 = nominal).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MultirotorScales {
    pub mass: f64,
    pub inertia: DVec3,
    /// Per-rotor thrust coefficient scales; missing entries are 1.
    pub k_thrust: Vec<f64>,
    pub k_torque: f64,
    pub motor_tau: f64,
    pub body_drag: f64,
    pub rotor_drag: f64,
}

impl Default for MultirotorScales {
    fn default() -> Self {
        Self {
            mass: 1.0,
            inertia: DVec3::ONE,
            k_thrust: Vec::new(),
            k_torque: 1.0,
            motor_tau: 1.0,
            body_drag: 1.0,
            rotor_drag: 1.0,
        }
    }
}

/// Runtime constants of one rotor (body frame, density-normalised at the reference density).
#[derive(Clone, Copy, Debug)]
struct Rotor {
    position: DVec3,
    axis: DVec3,
    spin: f64,
    radius: f64,
    k_thrust: f64,
    k_torque: f64,
    inertia: f64,
    drag: f64,
    rolling_moment: f64,
    omega_min: f64,
    omega_max: f64,
    /// `exp(−dt/τ)` for spin-up and spin-down.
    decay_up: f64,
    decay_down: f64,
}

/// Initial conditions for [`Multirotor::reset`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InitialState {
    pub pose: Pose,
    /// Linear velocity in the world frame (m/s).
    pub lin_vel_world: DVec3,
    /// Angular velocity in the body frame (rad/s).
    pub ang_vel_body: DVec3,
    pub motors: MotorInit,
    /// Battery state of charge (ignored without a battery).
    pub soc: f64,
}

impl InitialState {
    pub fn at(pose: Pose, motors: MotorInit) -> Self {
        Self { pose, lin_vel_world: DVec3::ZERO, ang_vel_body: DVec3::ZERO, motors, soc: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotorInit {
    /// All rotors at idle speed.
    Idle,
    /// All rotors at the given speed (rad/s).
    Speed(f64),
}

/// Everything outside the vehicle that one physics step needs.
#[derive(Clone, Copy)]
pub struct StepEnv<'a> {
    pub scene: Option<StaticScene<'a>>,
    pub gravity: DVec3,
    pub air: AirData,
    pub ground: Option<GroundPlane>,
}

/// One simulated multirotor.
///
/// A physics step is split into phases so that the simulation can add forces between agents:
/// [`begin_step`](Self::begin_step) → [`apply_rotors`](Self::apply_rotors) /
/// [`apply_contacts`](Self::apply_contacts) / [`apply_force`](Self::apply_force) →
/// [`finish_step`](Self::finish_step). [`step`](Self::step) runs them all.
#[derive(Clone, Debug)]
pub struct Multirotor {
    def: Arc<MultirotorDef>,
    dt: f64,
    // Parameters (nominal or randomised).
    body: MultibodyModel,
    mass: f64,
    rotors: PerRotor<Rotor>,
    reference_density: f64,
    drag_area: DVec3,
    angular_damping: DVec3,
    colliders: Vec<SphereCollider>,
    contact: ContactModel,
    // State.
    pub state: MbState,
    motor: PerRotor<f64>,
    battery: Option<BatteryState>,
    // Step buffers and outputs.
    ws: AbaWorkspace,
    f_ext: [SpatialForce; 1],
    motor_next: PerRotor<f64>,
    thrust: PerRotor<f64>,
    /// Rotor angular momentum (body frame) for the implicit gyroscopic term.
    h_rotors: DVec3,
    shaft_power: f64,
    ang_acc: DVec3,
    cache: ContactCache,
    scratch: ContactScratch,
    contacts: Vec<ContactPoint>,
}

impl Multirotor {
    /// Instance with nominal parameters for physics step `dt`, at rest at the origin with
    /// idle motors.
    pub fn new(def: Arc<MultirotorDef>, dt: f64) -> Self {
        assert!(dt > 0.0, "dt must be positive");
        let mut body = MultibodyModel::new();
        body.add_link(
            "body",
            None,
            JointType::Free,
            Pose::IDENTITY,
            RigidInertia::diag(def.body.mass, def.body.inertia),
        );
        let n = def.rotors.len();
        let mut s = Self {
            dt,
            ws: AbaWorkspace::new(&body),
            state: body.neutral_state(),
            body,
            mass: def.body.mass,
            rotors: SmallVec::new(),
            reference_density: def.rotor.reference_density,
            drag_area: def.body.drag_area,
            angular_damping: def.body.angular_damping,
            colliders: def.sphere_colliders(),
            contact: def.contact.model(def.body.mass, dt),
            motor: SmallVec::from_elem(0.0, n),
            battery: def.battery.as_ref().map(|b| b.state(1.0)),
            f_ext: [SpatialForce::ZERO],
            motor_next: SmallVec::from_elem(0.0, n),
            thrust: SmallVec::from_elem(0.0, n),
            h_rotors: DVec3::ZERO,
            shaft_power: 0.0,
            ang_acc: DVec3::ZERO,
            cache: ContactCache::default(),
            scratch: ContactScratch::default(),
            contacts: Vec::new(),
            def,
        };
        s.set_scales(&MultirotorScales::default());
        s.reset(&InitialState::at(Pose::IDENTITY, MotorInit::Idle));
        s
    }

    pub fn def(&self) -> &MultirotorDef {
        &self.def
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Rebuild the parameters from the definition with the given perturbations.
    pub fn set_scales(&mut self, s: &MultirotorScales) {
        let d = &self.def;
        self.mass = d.body.mass * s.mass;
        let inertia = RigidInertia::diag(self.mass, d.body.inertia * s.mass * s.inertia);
        let mut body = MultibodyModel::new();
        body.add_link("body", None, JointType::Free, Pose::IDENTITY, inertia);
        self.body = body;
        let p = &d.rotor;
        let decay = |tau: f64| (-self.dt / (tau * s.motor_tau)).exp();
        self.rotors = d
            .rotors
            .iter()
            .enumerate()
            .map(|(i, m)| Rotor {
                position: m.position,
                axis: m.axis,
                spin: m.spin.sign(),
                radius: p.radius,
                k_thrust: p.k_thrust * s.k_thrust.get(i).copied().unwrap_or(1.0),
                k_torque: p.k_torque * s.k_torque,
                inertia: p.inertia,
                drag: p.drag * s.rotor_drag,
                rolling_moment: p.rolling_moment,
                omega_min: p.omega_min,
                omega_max: p.omega_max,
                decay_up: decay(p.tau_up),
                decay_down: decay(p.tau_down),
            })
            .collect();
        self.drag_area = d.body.drag_area * s.body_drag;
        self.contact = d.contact.model(self.mass, self.dt);
    }

    /// Place the vehicle and clear contacts and transient outputs.
    pub fn reset(&mut self, init: &InitialState) {
        let rot = init.pose.rot.normalize();
        self.state.q[..3].copy_from_slice(&init.pose.pos.to_array());
        self.state.q[3..7].copy_from_slice(&rot.to_array());
        self.state.v[..3].copy_from_slice(&init.ang_vel_body.to_array());
        self.state.v[3..6].copy_from_slice(&(rot.inverse() * init.lin_vel_world).to_array());
        for (w, r) in self.motor.iter_mut().zip(&self.rotors) {
            *w = match init.motors {
                MotorInit::Idle => r.omega_min,
                MotorInit::Speed(x) => x.clamp(r.omega_min, r.omega_max),
            };
        }
        self.battery = self.def.battery.as_ref().map(|b| b.state(init.soc));
        self.thrust.fill(0.0);
        self.h_rotors = DVec3::ZERO;
        self.shaft_power = 0.0;
        self.ang_acc = DVec3::ZERO;
        self.cache.clear();
        self.contacts.clear();
        self.f_ext = [SpatialForce::ZERO];
        forward_kinematics(&self.body, &self.state.q, &self.state.v, &mut self.ws.kin);
    }

    // ---------------------------------------------------------------- step phases

    /// Forward kinematics and cleared force accumulators.
    pub fn begin_step(&mut self) {
        forward_kinematics(&self.body, &self.state.q, &self.state.v, &mut self.ws.kin);
        self.f_ext = [SpatialForce::ZERO];
        self.contacts.clear();
    }

    /// Rotor thrust, drag torque, spin-up reaction, rotor drag and body drag, given the
    /// commanded rotor speeds (rad/s, clamped to the feasible range). Also advances the motor
    /// model (exact first-order response over the step; takes effect in
    /// [`finish_step`](Self::finish_step)). The rotor gyroscopic torque `−ω × Σ s·J_r·ω_r·a` is
    /// applied implicitly by the integrator.
    pub fn apply_rotors(&mut self, omega_cmd: &[f64], air: &AirData, ground: Option<&GroundPlane>) {
        debug_assert_eq!(omega_cmd.len(), self.rotors.len());
        let (pos, rot) = (self.position(), self.orientation());
        let w_b = self.ang_vel_body();
        let v_air = self.lin_vel_body() - rot.inverse() * air.wind;
        let rho = air.density / self.reference_density;
        let top = self.speed_limit_scale();
        let (mut force, mut torque, mut h_rotors, mut power) = (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO, 0.0);
        for (i, r) in self.rotors.iter().enumerate() {
            let omega = self.motor[i];
            let cmd = omega_cmd[i].clamp(r.omega_min, r.omega_max * top);
            let decay = if cmd > omega { r.decay_up } else { r.decay_down };
            let next = cmd + (omega - cmd) * decay;
            let omega_dot = (next - omega) / self.dt;
            self.motor_next[i] = next;

            let w2 = omega * omega;
            let ge =
                ground.map_or(1.0, |g| ground_effect(g.distance_along(pos + rot * r.position, rot * r.axis), r.radius));
            let thrust = r.k_thrust * rho * w2 * ge;
            let v_hub = v_air + w_b.cross(r.position);
            let v_perp = v_hub - r.axis * r.axis.dot(v_hub);
            let f = r.axis * thrust - v_perp * (omega * r.drag * rho);
            let drag_torque = r.k_torque * rho * w2;
            force += f;
            torque += r.position.cross(f)
                - r.axis * (r.spin * (drag_torque + r.inertia * omega_dot))
                - v_perp * (omega * r.spin * r.rolling_moment * rho);
            h_rotors += r.axis * (r.spin * r.inertia * omega);
            power += drag_torque * omega;
            self.thrust[i] = thrust;
        }
        self.h_rotors = h_rotors;
        force -= 0.5 * air.density * self.drag_area * v_air.abs() * v_air;
        torque -= self.angular_damping * w_b;
        self.f_ext[0] += SpatialForce::new(torque, force);
        self.shaft_power = power;
    }

    /// Penalty contacts with the static world (terrain, obstacles, foliage).
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

    /// Add an external force (world frame) acting at a world point, e.g. from another agent.
    pub fn apply_force(&mut self, force: DVec3, point: DVec3) {
        let pose = self.ws.kin.pose[0];
        self.f_ext[0] += SpatialForce::from_force_at_point(
            pose.inverse_transform_vector(force),
            pose.inverse_transform_point(point),
        );
    }

    /// Forward dynamics and integration of body, motors and battery.
    pub fn finish_step(&mut self, gravity: DVec3) -> Result<(), DynamicsError> {
        aba_with_kinematics(&self.body, &[0.0; 6], &self.f_ext, gravity, &mut self.ws)?;
        let w1 = self.ang_vel_body();
        semi_implicit_euler_with_momentum(&self.body, &mut self.state, &self.ws.qdd, self.dt, self.h_rotors);
        self.ang_acc = (self.ang_vel_body() - w1) / self.dt;
        self.motor.copy_from_slice(&self.motor_next);
        if let (Some(def), Some(state)) = (&self.def.battery, &mut self.battery) {
            def.step(state, self.shaft_power, self.dt);
        }
        Ok(())
    }

    /// One full physics step with commanded rotor speeds `omega_cmd` (rad/s).
    pub fn step(&mut self, omega_cmd: &[f64], env: &StepEnv) -> Result<(), DynamicsError> {
        self.begin_step();
        self.apply_rotors(omega_cmd, &env.air, env.ground.as_ref());
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

    /// Principal moments of inertia (after randomisation).
    pub fn inertia(&self) -> DVec3 {
        let i = self.body.link(0).inertia.i_origin();
        DVec3::new(i.x_axis.x, i.y_axis.y, i.z_axis.z)
    }

    pub fn num_rotors(&self) -> usize {
        self.rotors.len()
    }

    /// Current rotor speeds (rad/s, non-negative; the spin sense is in the definition).
    pub fn motor_speeds(&self) -> &[f64] {
        &self.motor
    }

    pub fn set_motor_speeds(&mut self, omega: &[f64]) {
        self.motor.copy_from_slice(omega);
    }

    /// Rotor thrusts (N) of the last step.
    pub fn thrusts(&self) -> &[f64] {
        &self.thrust
    }

    /// Feasible rotor speed range `[min, max]` right now (the top speed sags with the battery).
    pub fn speed_range(&self) -> (f64, f64) {
        let r = &self.rotors[0];
        (r.omega_min, r.omega_max * self.speed_limit_scale())
    }

    fn speed_limit_scale(&self) -> f64 {
        match (&self.def.battery, &self.battery) {
            (Some(d), Some(s)) => s.voltage / d.full_voltage(),
            _ => 1.0,
        }
    }

    pub fn battery(&self) -> Option<&BatteryState> {
        self.battery.as_ref()
    }

    /// Active contacts of the last step.
    pub fn contacts(&self) -> &[ContactPoint] {
        &self.contacts
    }

    pub fn colliders(&self) -> &[SphereCollider] {
        &self.colliders
    }

    /// Specific force (accelerometer reading at the centre of mass, body frame) of the last
    /// step: all non-gravitational forces over mass.
    pub fn specific_force_body(&self) -> DVec3 {
        self.f_ext[0].lin / self.mass
    }

    /// Total non-gravitational wrench of the last step (body frame, about the centre of mass),
    /// excluding the rotor gyroscopic torque.
    pub fn external_wrench(&self) -> SpatialForce {
        self.f_ext[0]
    }

    /// Angular momentum of the rotors (body frame) at the start of the last step.
    pub fn rotor_momentum(&self) -> DVec3 {
        self.h_rotors
    }

    /// Mean angular acceleration (body frame) over the last step.
    pub fn ang_acc_body(&self) -> DVec3 {
        self.ang_acc
    }
}
