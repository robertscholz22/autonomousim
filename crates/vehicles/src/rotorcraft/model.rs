//! Helicopter instance: parameters, state and per-step force model.
//!
//! Each step the swashplate servos move, the main and tail rotors (the tail geared to the main)
//! give their loads for the flow at their hubs, the governor sets the engine torque, and the
//! drive train `I·Ω̇ = Q_engine − Q_main − n·Q_tail` sets the rotor speed. The body carries the
//! rotor forces and hub moments, the fuselage drag, the fins, the reaction of the torque driving
//! each rotor `−(Q + I_rΩ̇_r)·s·axis`, and the rotors' gyroscopic moment.

use super::def::{HelicopterDef, PitchChannel};
use super::rotor::{Rotor, RotorInput, RotorLoads, RotorState};
use crate::aero::{AirData, AirFlow, GroundPlane};
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

/// Damping ratio of the governor loop.
const GOVERNOR_ZETA: f64 = 0.7;

/// Rotor speed (fraction of rated) below which the engine's power limit no longer grows its
/// torque limit.
const MIN_POWER_SPEED: f64 = 0.2;

/// Normalised pilot inputs in [−1, 1]: collective up, cyclic stick forward (nose down) and
/// right (roll right), pedal right (nose right).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HelicopterInput {
    pub collective: f64,
    pub longitudinal: f64,
    pub lateral: f64,
    pub pedal: f64,
}

impl HelicopterInput {
    /// Within range; non-finite values read as 0.
    pub fn clamped(self) -> Self {
        let f = |x: f64| if x.is_finite() { x.clamp(-1.0, 1.0) } else { 0.0 };
        Self {
            collective: f(self.collective),
            longitudinal: f(self.longitudinal),
            lateral: f(self.lateral),
            pedal: f(self.pedal),
        }
    }

    pub fn to_array(self) -> [f64; 4] {
        [self.collective, self.longitudinal, self.lateral, self.pedal]
    }

    pub fn from_array(a: [f64; 4]) -> Self {
        Self { collective: a[0], longitudinal: a[1], lateral: a[2], pedal: a[3] }
    }
}

/// Initial conditions for [`Helicopter::reset`]. The rotors start with their flapping and
/// inflow settled for the initial flow (in still air of the given density) and the engine
/// delivering the torque they need.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HelicopterInit {
    pub pose: Pose,
    /// Linear velocity in the world frame (m/s).
    pub lin_vel_world: DVec3,
    /// Angular velocity in the body frame (rad/s).
    pub ang_vel_body: DVec3,
    /// Controls held from the start (servos already there); also the vehicle's hold input.
    pub controls: HelicopterInput,
    /// Main rotor speed (rad/s); the governed speed when absent.
    pub rotor_speed: Option<f64>,
    /// Air density the rotors are settled in (kg/m³).
    pub density: f64,
}

impl HelicopterInit {
    /// At rest at `pose`, rotors turning at the governed speed with the collective at flat
    /// pitch (zero, or the nearest end of its travel).
    pub fn at_rest(def: &HelicopterDef, pose: Pose) -> Self {
        let collective = def.controls.collective.input(0.0).clamp(-1.0, 1.0);
        Self {
            pose,
            lin_vel_world: DVec3::ZERO,
            ang_vel_body: DVec3::ZERO,
            controls: HelicopterInput { collective, ..Default::default() },
            rotor_speed: None,
            density: AirData::default().density,
        }
    }
}

/// Aerodynamic loads on the airframe for one state (body frame, about the centre of mass),
/// without the drive torque reactions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HelicopterLoads {
    pub main: RotorLoads,
    pub tail: RotorLoads,
    pub force: DVec3,
    pub moment: DVec3,
}

/// One simulated helicopter.
///
/// Step phases: [`begin_step`](Self::begin_step) → [`apply_controls`](Self::apply_controls) /
/// [`apply_contacts`](Self::apply_contacts) / [`apply_force`](Self::apply_force) →
/// [`finish_step`](Self::finish_step). [`step`](Self::step) runs them all.
#[derive(Clone, Debug)]
pub struct Helicopter {
    def: Arc<HelicopterDef>,
    dt: f64,
    body: MultibodyModel,
    mass: f64,
    inertia: DMat3,
    colliders: Vec<SphereCollider>,
    contact: ContactModel,
    main: Rotor,
    tail: Rotor,
    /// Shaft → body frames.
    main_frame: DMat3,
    tail_frame: DMat3,
    /// Tail collective input sign that yaws the nose right.
    pedal_sign: f64,
    drive_inertia: f64,
    governor: (f64, f64),
    // State.
    pub state: MbState,
    main_state: RotorState,
    tail_state: RotorState,
    /// Collective, longitudinal and lateral cyclic amplitude, tail collective (rad).
    pitches: [f64; 4],
    engine_torque: f64,
    governor_integral: f64,
    engine_running: bool,
    hold: HelicopterInput,
    input: HelicopterInput,
    // Step buffers and outputs.
    flow: AirFlow,
    loads: HelicopterLoads,
    omega_dot: f64,
    omega_next: f64,
    ws: AbaWorkspace,
    f_ext: [SpatialForce; 1],
    h_rotor: DVec3,
    ang_acc: DVec3,
    cache: ContactCache,
    scratch: ContactScratch,
    contacts: Vec<ContactPoint>,
}

impl Helicopter {
    /// Instance for physics step `dt`, at rest at the origin.
    pub fn new(def: Arc<HelicopterDef>, dt: f64) -> Self {
        assert!(dt > 0.0, "dt must be positive");
        let mut body = MultibodyModel::new();
        body.add_link("body", None, JointType::Free, Pose::IDENTITY, def.body.rigid_inertia());
        let (main, tail) = def.rotors();
        let t = &def.tail_rotor;
        let yaw_per_thrust = t.hub.cross(t.axis).z;
        let i = def.drive_inertia();
        let w = def.engine.governor_bandwidth;
        let mut s = Self {
            dt,
            ws: AbaWorkspace::new(&body),
            state: body.neutral_state(),
            body,
            mass: def.body.mass,
            inertia: def.body.inertia_matrix(),
            colliders: def.sphere_colliders(),
            contact: def.contact.model(def.body.mass, dt),
            main,
            tail,
            main_frame: def.main_rotor.frame(),
            tail_frame: def.tail_rotor.frame(),
            pedal_sign: if yaw_per_thrust > 0.0 { -1.0 } else { 1.0 },
            drive_inertia: i,
            governor: (2.0 * GOVERNOR_ZETA * w * i, w * w * i),
            main_state: RotorState::default(),
            tail_state: RotorState::default(),
            pitches: [0.0; 4],
            engine_torque: 0.0,
            governor_integral: 0.0,
            engine_running: true,
            hold: HelicopterInput::default(),
            input: HelicopterInput::default(),
            flow: AirFlow::default(),
            loads: HelicopterLoads::default(),
            omega_dot: 0.0,
            omega_next: 0.0,
            f_ext: [SpatialForce::ZERO],
            h_rotor: DVec3::ZERO,
            ang_acc: DVec3::ZERO,
            cache: ContactCache::default(),
            scratch: ContactScratch::default(),
            contacts: Vec::new(),
            def,
        };
        let rest = HelicopterInit::at_rest(&s.def, Pose::IDENTITY);
        s.reset(&rest);
        s
    }

    pub fn def(&self) -> &HelicopterDef {
        &self.def
    }

    pub fn shared_def(&self) -> &Arc<HelicopterDef> {
        &self.def
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Place the helicopter, set the controls, settle the rotors and clear contacts and outputs.
    pub fn reset(&mut self, init: &HelicopterInit) {
        let rot = init.pose.rot.normalize();
        self.state.q[..3].copy_from_slice(&init.pose.pos.to_array());
        self.state.q[3..7].copy_from_slice(&rot.to_array());
        self.state.v[..3].copy_from_slice(&init.ang_vel_body.to_array());
        let v_body = rot.inverse() * init.lin_vel_world;
        self.state.v[3..6].copy_from_slice(&v_body.to_array());
        let controls = init.controls.clamped();
        self.hold = controls;
        self.input = controls;
        self.pitches = self.targets(&controls);
        let omega = init.rotor_speed.unwrap_or(self.def.engine.rated_speed).max(0.0);
        let air = AirData { density: init.density, ..AirData::default() };
        let flow = AirFlow::new(&air, v_body, init.ang_vel_body);
        let (loads, main, tail) = self.settled(&flow, omega, &self.pitches);
        self.main_state = main;
        self.tail_state = tail;
        let q = (loads.main.torque + self.def.tail_gear_ratio * loads.tail.torque).clamp(0.0, self.torque_limit(omega));
        self.engine_torque = q;
        self.governor_integral = q;
        self.engine_running = true;
        self.omega_next = omega;
        self.omega_dot = 0.0;
        self.flow = flow;
        self.loads = loads;
        self.h_rotor = self.rotor_momentum(omega);
        self.ang_acc = DVec3::ZERO;
        self.cache.clear();
        self.contacts.clear();
        self.f_ext = [SpatialForce::ZERO];
        forward_kinematics(&self.body, &self.state.q, &self.state.v, &mut self.ws.kin);
    }

    /// Blade pitches (rad) the controls ask for: collective, cyclic amplitudes, tail collective.
    pub fn targets(&self, c: &HelicopterInput) -> [f64; 4] {
        let k = &self.def.controls;
        [
            k.collective.pitch(c.collective),
            k.longitudinal.pitch(c.longitudinal),
            k.lateral.pitch(c.lateral),
            k.pedal.pitch(self.pedal_sign * c.pedal),
        ]
    }

    /// Blade pitches for inputs continued linearly beyond their range (for trimming).
    pub(super) fn pitches_linear(&self, c: &HelicopterInput) -> [f64; 4] {
        let k = &self.def.controls;
        let lin = |ch: &PitchChannel, w: f64| 0.5 * (ch.min + ch.max) + 0.5 * w * (ch.max - ch.min);
        [
            lin(&k.collective, c.collective),
            lin(&k.longitudinal, c.longitudinal),
            lin(&k.lateral, c.lateral),
            lin(&k.pedal, self.pedal_sign * c.pedal),
        ]
    }

    /// Main rotor cyclic `(θ₁c, θ₁s)` for the longitudinal and lateral amplitudes: phased 90°
    /// ahead of the tilt they ask for (forward and right), for the rotor's direction.
    fn cyclic(&self, longitudinal: f64, lateral: f64) -> [f64; 2] {
        let s = self.def.main_rotor.rotor.spin_sign();
        [-s * lateral, -s * longitudinal]
    }

    /// Stop or restart the engine (the rotor freewheels while it is stopped).
    pub fn set_engine_running(&mut self, running: bool) {
        self.engine_running = running;
        if !running {
            self.governor_integral = 0.0;
        }
    }

    /// Engine torque limit at main rotor speed `omega` (N·m).
    fn torque_limit(&self, omega: f64) -> f64 {
        let e = &self.def.engine;
        e.max_torque().min(e.max_power / omega.max(MIN_POWER_SPEED * e.rated_speed))
    }

    fn rotor_momentum(&self, omega: f64) -> DVec3 {
        let (m, t) = (&self.def.main_rotor, &self.def.tail_rotor);
        let n = self.def.tail_gear_ratio;
        m.axis * (m.rotor.spin_sign() * m.rotor.hub_inertia() * omega)
            + t.axis * (t.rotor.spin_sign() * t.rotor.hub_inertia() * n * omega)
    }

    /// Reaction on the airframe of the torques driving the rotors while the drive train
    /// accelerates at `omega_dot` (body frame).
    fn drive_reaction(&self, loads: &HelicopterLoads, omega_dot: f64) -> DVec3 {
        let (m, t) = (&self.def.main_rotor, &self.def.tail_rotor);
        let n = self.def.tail_gear_ratio;
        let q_main = loads.main.torque + m.rotor.polar_inertia() * omega_dot;
        let q_tail = loads.tail.torque + t.rotor.polar_inertia() * n * omega_dot;
        -(m.axis * (m.rotor.spin_sign() * q_main) + t.axis * (t.rotor.spin_sign() * q_tail))
    }

    /// Rotor inputs for the flow at the hubs.
    fn rotor_inputs(&self, flow: &AirFlow, pitches: &[f64; 4], ground: Option<f64>) -> (RotorInput, RotorInput) {
        let (m, t) = (&self.def.main_rotor, &self.def.tail_rotor);
        let (mt, tt) = (self.main_frame.transpose(), self.tail_frame.transpose());
        let main = RotorInput {
            velocity: mt * flow.at(m.hub),
            rates: mt * flow.rates,
            density: flow.density,
            collective: pitches[0],
            cyclic: self.cyclic(pitches[1], pitches[2]),
            ground_distance: ground,
        };
        let tail = RotorInput {
            velocity: tt * flow.at(t.hub),
            rates: tt * flow.rates,
            density: flow.density,
            collective: pitches[3],
            cyclic: [0.0; 2],
            ground_distance: None,
        };
        (main, tail)
    }

    /// Airframe loads from the rotor loads, fuselage and fins.
    fn airframe_loads(&self, flow: &AirFlow, main: RotorLoads, tail: RotorLoads) -> HelicopterLoads {
        let (m, t) = (&self.def.main_rotor, &self.def.tail_rotor);
        let f_main = self.main_frame * main.force;
        let f_tail = self.tail_frame * tail.force;
        let mut force = f_main + f_tail;
        let mut moment =
            m.hub.cross(f_main) + self.main_frame * main.moment + t.hub.cross(f_tail) + self.tail_frame * tail.moment;
        let fus = &self.def.fuselage;
        let v = flow.at(fus.position);
        let drag = -0.5 * flow.density * v.length() * (fus.drag_area * v);
        force += drag;
        moment += fus.position.cross(drag);
        for s in &self.def.surfaces {
            let (f, mo) = s.wrench(flow, 0.0);
            force += f;
            moment += mo;
        }
        HelicopterLoads { main, tail, force, moment }
    }

    /// Loads with both rotors settled (steady flapping and inflow) at main rotor speed `omega`,
    /// out of ground effect, and the settled rotor states.
    pub fn settled(&self, flow: &AirFlow, omega: f64, pitches: &[f64; 4]) -> (HelicopterLoads, RotorState, RotorState) {
        let (mi, ti) = self.rotor_inputs(flow, pitches, None);
        let (main, ms) = self.main.settled(omega, &mi);
        let (tail, ts) = self.tail.settled(self.def.tail_gear_ratio * omega, &ti);
        (self.airframe_loads(flow, main, tail), ms, ts)
    }

    /// Total non-gravitational force and moment (body frame) in steady flight at a fixed state:
    /// rotors settled at `omega`, the rotor speed steady (drive torque = rotor torque).
    pub fn steady_wrench(&self, flow: &AirFlow, omega: f64, pitches: &[f64; 4]) -> (DVec3, DVec3, HelicopterLoads) {
        let (loads, _, _) = self.settled(flow, omega, pitches);
        (loads.force, loads.moment + self.drive_reaction(&loads, 0.0), loads)
    }

    // ---------------------------------------------------------------- step phases

    /// Forward kinematics and cleared force accumulators.
    pub fn begin_step(&mut self) {
        forward_kinematics(&self.body, &self.state.q, &self.state.v, &mut self.ws.kin);
        self.f_ext = [SpatialForce::ZERO];
        self.contacts.clear();
    }

    /// Servos, rotors, drive train, fuselage and fins for the pilot input. `ground` is the
    /// surface below (ground effect).
    pub fn apply_controls(&mut self, input: &HelicopterInput, air: &AirData, ground: Option<&GroundPlane>) {
        let input = input.clamped();
        self.input = input;
        let targets = self.targets(&input);
        let k = &self.def.controls;
        for (i, ch) in [&k.collective, &k.longitudinal, &k.lateral, &k.pedal].into_iter().enumerate() {
            self.pitches[i] = ch.servo(self.pitches[i], targets[i], self.dt);
        }
        let (pos, rot) = (self.position(), self.orientation());
        let flow = air.flow(rot, self.lin_vel_world(), self.ang_vel_body());
        let m = &self.def.main_rotor;
        let ground = ground.map(|g| g.distance_along(pos + rot * m.hub, rot * m.axis));
        let n = self.def.tail_gear_ratio;
        let omega = self.main_state.omega;
        self.tail_state.omega = n * omega;
        let (mi, ti) = self.rotor_inputs(&flow, &self.pitches, ground);
        let main = self.main.loads(&self.main_state, &mi);
        let tail = self.tail.loads(&self.tail_state, &ti);
        let loads = self.airframe_loads(&flow, main, tail);

        // Governor and engine, then the drive train.
        let limit = if self.engine_running { self.torque_limit(omega) } else { 0.0 };
        let (kp, ki) = self.governor;
        let err = self.def.engine.rated_speed - omega;
        if self.engine_running {
            self.governor_integral = (self.governor_integral + ki * err * self.dt).clamp(0.0, limit);
        }
        let demand = (kp * err + self.governor_integral).clamp(0.0, limit);
        let lag = self.def.engine.lag;
        self.engine_torque =
            if lag > 0.0 { demand + (self.engine_torque - demand) * (-self.dt / lag).exp() } else { demand };
        self.engine_torque = self.engine_torque.min(limit);
        let load = main.torque + n * tail.torque;
        self.omega_dot = (self.engine_torque - load) / self.drive_inertia;
        self.omega_next = (omega + self.dt * self.omega_dot).max(0.0);
        self.h_rotor = self.rotor_momentum(omega);

        let reaction = self.drive_reaction(&loads, self.omega_dot);
        self.f_ext[0] += SpatialForce::new(loads.moment + reaction, loads.force);
        self.flow = flow;
        self.loads = loads;
    }

    /// Penalty contacts of the colliders (skids, airframe, rotor discs) with the static world.
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

    /// Forward dynamics and integration of the body, flapping and rotor speed.
    pub fn finish_step(&mut self, gravity: DVec3) -> Result<(), DynamicsError> {
        aba_with_kinematics(&self.body, &[0.0; 6], &self.f_ext, gravity, &mut self.ws)?;
        let w1 = self.ang_vel_body();
        semi_implicit_euler_with_momentum(&self.body, &mut self.state, &self.ws.qdd, self.dt, self.h_rotor);
        self.ang_acc = (self.ang_vel_body() - w1) / self.dt;
        self.main.advance_flapping(&mut self.main_state, &self.loads.main, self.dt);
        self.tail.advance_flapping(&mut self.tail_state, &self.loads.tail, self.dt);
        self.main_state.omega = self.omega_next;
        self.tail_state.omega = self.def.tail_gear_ratio * self.omega_next;
        Ok(())
    }

    /// One full physics step.
    pub fn step(&mut self, input: &HelicopterInput, env: &StepEnv) -> Result<(), DynamicsError> {
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

    pub fn main_rotor(&self) -> &Rotor {
        &self.main
    }

    pub fn tail_rotor(&self) -> &Rotor {
        &self.tail
    }

    pub fn main_rotor_state(&self) -> &RotorState {
        &self.main_state
    }

    pub fn tail_rotor_state(&self) -> &RotorState {
        &self.tail_state
    }

    /// Main rotor speed (rad/s).
    pub fn rotor_speed(&self) -> f64 {
        self.main_state.omega
    }

    /// Main rotor speed rate of the last step (rad/s²).
    pub fn rotor_acceleration(&self) -> f64 {
        self.omega_dot
    }

    /// Engine torque at the main rotor shaft (N·m) and the power it delivers (W).
    pub fn engine_torque(&self) -> f64 {
        self.engine_torque
    }

    pub fn engine_power(&self) -> f64 {
        self.engine_torque * self.main_state.omega
    }

    pub fn engine_running(&self) -> bool {
        self.engine_running
    }

    /// Collective, longitudinal and lateral cyclic amplitude and tail collective (rad).
    pub fn pitches(&self) -> [f64; 4] {
        self.pitches
    }

    /// Input of the last step.
    pub fn input(&self) -> &HelicopterInput {
        &self.input
    }

    /// Input the helicopter was reset with.
    pub fn hold_input(&self) -> &HelicopterInput {
        &self.hold
    }

    /// Air-relative flow at the centre of mass in the last step.
    pub fn flow(&self) -> &AirFlow {
        &self.flow
    }

    /// Loads of the last step.
    pub fn loads(&self) -> &HelicopterLoads {
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
}
