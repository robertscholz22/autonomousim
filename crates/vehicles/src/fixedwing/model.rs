//! Fixed-wing instance: parameters, state and per-step force model.

use super::aero::{AeroForces, AeroInput};
use super::def::{FixedWingDef, SurfaceDef};
use super::gear::{self, WheelContact};
use super::propulsion::{Propulsion, PropulsionOutput};
use crate::aero::{AirData, AirFlow, GroundPlane};
use crate::multirotor::{BatteryState, StepEnv};
use autonomousim_core::contact::{
    ContactCache, ContactModel, ContactPoint, ContactScratch, SphereCollider, StaticScene, compute_contacts,
};
use autonomousim_core::dynamics::{
    AbaWorkspace, DynamicsError, JointType, MbState, MultibodyModel, aba_with_kinematics, forward_kinematics,
    semi_implicit_euler_with_momentum,
};
use autonomousim_core::geometry::HitKind;
use autonomousim_core::math::{Pose, SpatialForce};
use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Height over span reported out of ground effect.
const FREE_AIR_HEIGHT: f64 = 10.0;
/// Airspeed below which α̇ is taken as zero and no stall is reported (m/s).
const MIN_FLOW_SPEED: f64 = 1.0;
/// Bound on the finite-difference α̇ (rad/s).
const MAX_ALPHA_DOT: f64 = 5.0;

/// Normalised pilot inputs: aileron, elevator and rudder in [−1, 1] (roll right, nose up,
/// nose right and nose wheel right); throttle, flap and brake in [0, 1].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FixedWingInput {
    pub aileron: f64,
    pub elevator: f64,
    pub rudder: f64,
    pub throttle: f64,
    #[serde(default)]
    pub flap: f64,
    #[serde(default)]
    pub brake: f64,
}

impl FixedWingInput {
    /// Within range; non-finite values read as 0.
    pub fn clamped(self) -> Self {
        let f = |x: f64, lo: f64| if x.is_finite() { x.clamp(lo, 1.0) } else { 0.0 };
        Self {
            aileron: f(self.aileron, -1.0),
            elevator: f(self.elevator, -1.0),
            rudder: f(self.rudder, -1.0),
            throttle: f(self.throttle, 0.0),
            flap: f(self.flap, 0.0),
            brake: f(self.brake, 0.0),
        }
    }
}

/// Initial conditions for [`FixedWing::reset`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedWingInit {
    pub pose: Pose,
    /// Linear velocity in the world frame (m/s).
    pub lin_vel_world: DVec3,
    /// Angular velocity in the body frame (rad/s).
    pub ang_vel_body: DVec3,
    /// Controls held from the start (surfaces already deflected); also the vehicle's hold
    /// input.
    pub controls: FixedWingInput,
    /// Rotor speed (rad/s); `None`: steady at the controls' throttle for the initial airspeed
    /// at sea level.
    pub rotor_speed: Option<f64>,
    /// Battery state of charge (ignored without a battery).
    pub soc: f64,
}

impl FixedWingInit {
    /// At rest at `pose` with the engine idling and the brakes set.
    pub fn at_rest(pose: Pose) -> Self {
        Self {
            pose,
            lin_vel_world: DVec3::ZERO,
            ang_vel_body: DVec3::ZERO,
            controls: FixedWingInput { brake: 1.0, ..Default::default() },
            rotor_speed: None,
            soc: 1.0,
        }
    }
}

/// One simulated fixed-wing aircraft.
///
/// Step phases: [`begin_step`](Self::begin_step) → [`apply_controls`](Self::apply_controls)
/// (servos, propulsion, aerodynamics) / [`apply_gear`](Self::apply_gear) /
/// [`apply_contacts`](Self::apply_contacts) / [`apply_force`](Self::apply_force) →
/// [`finish_step`](Self::finish_step). [`step`](Self::step) runs them all.
#[derive(Clone, Debug)]
pub struct FixedWing {
    def: Arc<FixedWingDef>,
    dt: f64,
    body: MultibodyModel,
    mass: f64,
    inertia: DMat3,
    colliders: Vec<SphereCollider>,
    /// Airframe colliders come first; the gear wheels follow.
    frame_colliders: usize,
    contact: ContactModel,
    propulsion: Propulsion,
    signs: [f64; 3],
    stall_angles: (f64, f64),
    // State.
    pub state: MbState,
    /// Aileron, elevator, rudder and flap deflections (rad).
    surfaces: [f64; 4],
    rotor: f64,
    battery: Option<BatteryState>,
    engine_running: bool,
    stall_state: f64,
    alpha_prev: Option<f64>,
    hold: FixedWingInput,
    input: FixedWingInput,
    // Step buffers and outputs.
    flow: AirFlow,
    aero: AeroForces,
    prop: PropulsionOutput,
    wheels: Vec<Option<WheelContact>>,
    ws: AbaWorkspace,
    f_ext: [SpatialForce; 1],
    rotor_next: f64,
    h_rotor: DVec3,
    ang_acc: DVec3,
    cache: ContactCache,
    scratch: ContactScratch,
    contacts: Vec<ContactPoint>,
}

/// First-order servo with a rate limit.
fn servo(x: f64, target: f64, s: &SurfaceDef, dt: f64) -> f64 {
    let next = if s.tau > 0.0 { target + (x - target) * (-dt / s.tau).exp() } else { target };
    if s.rate > 0.0 { x + (next - x).clamp(-s.rate * dt, s.rate * dt) } else { next }
}

impl FixedWing {
    /// Instance for physics step `dt`, at rest at the origin with the engine idling.
    pub fn new(def: Arc<FixedWingDef>, dt: f64) -> Self {
        assert!(dt > 0.0, "dt must be positive");
        let mut body = MultibodyModel::new();
        body.add_link("body", None, JointType::Free, Pose::IDENTITY, def.body.rigid_inertia());
        let colliders = def.sphere_colliders();
        let mut s = Self {
            dt,
            ws: AbaWorkspace::new(&body),
            state: body.neutral_state(),
            body,
            mass: def.body.mass,
            inertia: def.body.inertia_matrix(),
            frame_colliders: def.colliders.len(),
            colliders,
            contact: def.contact.model(def.body.mass, dt),
            propulsion: Propulsion::new(&def.propulsion),
            signs: def.control_signs(),
            stall_angles: def.stall_angles(),
            surfaces: [0.0; 4],
            rotor: 0.0,
            battery: def.battery.as_ref().map(|b| b.state(1.0)),
            engine_running: true,
            stall_state: 0.0,
            alpha_prev: None,
            hold: FixedWingInput::default(),
            input: FixedWingInput::default(),
            flow: AirFlow::default(),
            aero: AeroForces::default(),
            prop: PropulsionOutput::default(),
            wheels: vec![None; def.gear.len()],
            f_ext: [SpatialForce::ZERO],
            rotor_next: 0.0,
            h_rotor: DVec3::ZERO,
            ang_acc: DVec3::ZERO,
            cache: ContactCache::default(),
            scratch: ContactScratch::default(),
            contacts: Vec::new(),
            def,
        };
        s.reset(&FixedWingInit::at_rest(Pose::IDENTITY));
        s
    }

    pub fn def(&self) -> &FixedWingDef {
        &self.def
    }

    pub fn shared_def(&self) -> &Arc<FixedWingDef> {
        &self.def
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Place the aircraft, set the controls and rotor, and clear contacts and outputs.
    pub fn reset(&mut self, init: &FixedWingInit) {
        let rot = init.pose.rot.normalize();
        self.state.q[..3].copy_from_slice(&init.pose.pos.to_array());
        self.state.q[3..7].copy_from_slice(&rot.to_array());
        self.state.v[..3].copy_from_slice(&init.ang_vel_body.to_array());
        let v_body = rot.inverse() * init.lin_vel_world;
        self.state.v[3..6].copy_from_slice(&v_body.to_array());
        let controls = init.controls.clamped();
        self.hold = controls;
        self.input = controls;
        self.surfaces = self.targets(&controls);
        self.battery = self.def.battery.as_ref().map(|b| b.state(init.soc));
        let supply = self.supply_voltage();
        self.rotor = init.rotor_speed.unwrap_or_else(|| {
            let v_axial = v_body.dot(self.propulsion.axis);
            self.propulsion.steady_omega(controls.throttle, v_axial, crate::aero::SEA_LEVEL_DENSITY, supply)
        });
        self.rotor_next = self.rotor;
        self.engine_running = true;
        self.stall_state = 0.0;
        self.alpha_prev = None;
        self.flow = AirFlow::default();
        self.aero = AeroForces::default();
        self.prop = PropulsionOutput { omega: self.rotor, ..Default::default() };
        self.wheels.fill(None);
        self.h_rotor = DVec3::ZERO;
        self.ang_acc = DVec3::ZERO;
        self.cache.clear();
        self.contacts.clear();
        self.f_ext = [SpatialForce::ZERO];
        forward_kinematics(&self.body, &self.state.q, &self.state.v, &mut self.ws.kin);
    }

    /// Surface deflections (rad) the controls ask for.
    pub fn targets(&self, c: &FixedWingInput) -> [f64; 4] {
        let k = &self.def.controls;
        [
            k.aileron.deflection(self.signs[0] * c.aileron),
            k.elevator.deflection(self.signs[1] * c.elevator),
            k.rudder.deflection(self.signs[2] * c.rudder),
            k.flap.deflection(c.flap),
        ]
    }

    /// Battery voltage, or 0 for the motor's own supply voltage.
    fn supply_voltage(&self) -> f64 {
        self.battery.as_ref().map_or(0.0, |b| b.voltage)
    }

    /// Stop or restart the engine (an electric motor gets no voltage while stopped).
    pub fn set_engine_running(&mut self, running: bool) {
        self.engine_running = running;
    }

    // ---------------------------------------------------------------- step phases

    /// Forward kinematics and cleared force accumulators.
    pub fn begin_step(&mut self) {
        forward_kinematics(&self.body, &self.state.q, &self.state.v, &mut self.ws.kin);
        self.f_ext = [SpatialForce::ZERO];
        self.contacts.clear();
    }

    /// Servos, propulsion and aerodynamics for the pilot input. `ground` is the surface below
    /// (ground effect, when within about a span).
    pub fn apply_controls(&mut self, input: &FixedWingInput, air: &AirData, ground: Option<&GroundPlane>) {
        let input = input.clamped();
        self.input = input;
        let targets = self.targets(&input);
        for (k, s) in self.def.controls.surfaces().into_iter().enumerate() {
            self.surfaces[k] = servo(self.surfaces[k], targets[k], s, self.dt);
        }
        let (pos, rot) = (self.position(), self.orientation());
        let flow = air.flow(rot, self.lin_vel_world(), self.ang_vel_body());

        // Propulsion (loads at the current rotor speed; the new speed takes effect in
        // finish_step).
        let p = &self.propulsion;
        let v_axial = flow.at(p.position).dot(p.axis);
        let out = p.step(
            self.rotor,
            input.throttle,
            v_axial,
            air.density,
            self.engine_running,
            self.supply_voltage(),
            self.dt,
        );
        self.rotor_next = out.omega;
        self.h_rotor = p.axis * (p.sense * p.inertia * self.rotor);
        let thrust = p.axis * out.thrust;
        let prop_moment = p.position.cross(thrust) - p.axis * (p.sense * out.engine_torque);

        // Aerodynamics.
        let geo = &self.def.geometry;
        let wash = v_axial + 2.0 * out.induced_velocity;
        let height_over_span = ground.map_or(FREE_AIR_HEIGHT, |g| {
            (g.normal.dot(pos + rot * geo.aero_reference - g.point) / geo.span).clamp(0.0, FREE_AIR_HEIGHT)
        });
        let alpha_dot = match self.alpha_prev {
            Some(a) if flow.airspeed > MIN_FLOW_SPEED => {
                ((flow.alpha - a) / self.dt).clamp(-MAX_ALPHA_DOT, MAX_ALPHA_DOT)
            }
            _ => 0.0,
        };
        self.alpha_prev = Some(flow.alpha);
        if let Some([off, on]) = self.def.aero.stall_hysteresis() {
            if flow.alpha > on {
                self.stall_state = 1.0;
            } else if flow.alpha < off {
                self.stall_state = 0.0;
            }
        }
        let x = self.aero_input(&flow, alpha_dot, height_over_span, 0.5 * air.density * wash * wash);
        let aero = self.def.aero.forces(geo, &x);
        self.f_ext[0] += SpatialForce::new(aero.moment + prop_moment, aero.force + thrust);
        self.flow = flow;
        self.aero = aero;
        self.prop = out;
    }

    fn aero_input(&self, flow: &AirFlow, alpha_dot: f64, height_over_span: f64, wash_pressure: f64) -> AeroInput {
        let s = &self.surfaces;
        AeroInput {
            alpha: flow.alpha,
            beta: flow.beta,
            airspeed: flow.airspeed,
            mach: flow.mach,
            p: flow.rates.x,
            q: -flow.rates.y,
            r: -flow.rates.z,
            alpha_dot,
            aileron: s[0],
            elevator: s[1],
            rudder: s[2],
            flap: s[3],
            height_over_span,
            stall: self.stall_state,
            dynamic_pressure: flow.dynamic_pressure,
            wash_pressure,
        }
    }

    /// Landing gear on the terrain (and water, which only reports a contact).
    pub fn apply_gear(&mut self, scene: &StaticScene) {
        let (pose, v, w) = (self.pose(), self.lin_vel_world(), self.ang_vel_body());
        for (i, g) in self.def.gear.iter().enumerate() {
            let p = pose.transform_point(g.position);
            let vp = v + pose.rot * w.cross(g.position);
            let steer = -g.max_steer * self.input.rudder;
            let heading = pose.rot * DVec3::new(steer.cos(), steer.sin(), 0.0);
            let c = gear::contact(g, p, vp, heading, self.input.brake, scene.terrain);
            if let Some(c) = &c {
                let f = c.normal * c.normal_force + c.friction;
                if !c.water {
                    self.f_ext[0] += SpatialForce::from_force_at_point(
                        pose.inverse_transform_vector(f),
                        pose.inverse_transform_point(p),
                    );
                }
                self.contacts.push(ContactPoint {
                    collider: (self.frame_colliders + i) as u16,
                    kind: if c.water { HitKind::Water } else { HitKind::Terrain },
                    material: c.material,
                    point: c.point,
                    normal: c.normal,
                    depth: c.compression,
                    normal_velocity: c.normal_velocity,
                    normal_force: c.normal_force,
                    friction: c.friction,
                    slipping: false,
                });
            }
            self.wheels[i] = c;
        }
    }

    /// Penalty contacts of the airframe colliders with the static world.
    pub fn apply_contacts(&mut self, scene: &StaticScene) {
        compute_contacts(
            scene,
            &self.ws.kin,
            &self.colliders[..self.frame_colliders],
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

    /// Forward dynamics and integration of body, rotor and battery.
    pub fn finish_step(&mut self, gravity: DVec3) -> Result<(), DynamicsError> {
        aba_with_kinematics(&self.body, &[0.0; 6], &self.f_ext, gravity, &mut self.ws)?;
        let w1 = self.ang_vel_body();
        semi_implicit_euler_with_momentum(&self.body, &mut self.state, &self.ws.qdd, self.dt, self.h_rotor);
        self.ang_acc = (self.ang_vel_body() - w1) / self.dt;
        self.rotor = self.rotor_next;
        if let (Some(def), Some(state)) = (&self.def.battery, &mut self.battery) {
            def.step_electrical(state, self.prop.electric_power, self.dt);
        }
        Ok(())
    }

    /// One full physics step.
    pub fn step(&mut self, input: &FixedWingInput, env: &StepEnv) -> Result<(), DynamicsError> {
        self.begin_step();
        self.apply_controls(input, &env.air, env.ground.as_ref());
        if let Some(scene) = &env.scene {
            self.apply_gear(scene);
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

    pub fn propulsion(&self) -> &Propulsion {
        &self.propulsion
    }

    /// Deflection signs of aileron, elevator and rudder for positive commands.
    pub fn control_signs(&self) -> [f64; 3] {
        self.signs
    }

    /// Stall angles of attack (positive, negative; rad).
    pub fn stall_angles(&self) -> (f64, f64) {
        self.stall_angles
    }

    /// Aileron, elevator, rudder and flap deflections (rad).
    pub fn surfaces(&self) -> [f64; 4] {
        self.surfaces
    }

    /// Input of the last step.
    pub fn input(&self) -> &FixedWingInput {
        &self.input
    }

    /// Input the aircraft was reset with (trim, or idle with the brakes set).
    pub fn hold_input(&self) -> &FixedWingInput {
        &self.hold
    }

    /// Rotor speed (rad/s).
    pub fn rotor_speed(&self) -> f64 {
        self.rotor
    }

    pub fn set_rotor_speed(&mut self, omega: f64) {
        self.rotor = omega.max(0.0);
        self.rotor_next = self.rotor;
    }

    /// Air-relative flow at the centre of mass in the last step.
    pub fn flow(&self) -> &AirFlow {
        &self.flow
    }

    /// Aerodynamic loads of the last step.
    pub fn aero(&self) -> &AeroForces {
        &self.aero
    }

    /// Propulsion loads of the last step.
    pub fn propulsion_output(&self) -> &PropulsionOutput {
        &self.prop
    }

    /// Ground contact of each gear wheel in the last step.
    pub fn wheels(&self) -> &[Option<WheelContact>] {
        &self.wheels
    }

    /// Whether any wheel carries load.
    pub fn weight_on_wheels(&self) -> bool {
        self.wheels.iter().flatten().any(|c| c.normal_force > 0.0)
    }

    /// Angle of attack beyond a stall angle in flight (not on the wheels).
    pub fn stalled(&self) -> bool {
        let a = self.flow.alpha;
        self.flow.airspeed > MIN_FLOW_SPEED
            && !self.weight_on_wheels()
            && (a > self.stall_angles.0 || a < self.stall_angles.1)
    }

    pub fn battery(&self) -> Option<&BatteryState> {
        self.battery.as_ref()
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

    /// Mechanical energy: kinetic (body and rotor) plus potential for gravity `g` (J).
    pub fn energy(&self, g: f64) -> f64 {
        let (v, w) = (self.lin_vel_body(), self.ang_vel_body());
        0.5 * self.mass * v.length_squared()
            + 0.5 * w.dot(self.inertia * w)
            + 0.5 * self.propulsion.inertia * self.rotor * self.rotor
            + self.mass * g * self.position().z
    }

    /// Power of the external forces of the last step on the airframe, plus the net torque on
    /// the rotor times its speed (W): the rate of change of [`energy`](Self::energy) apart
    /// from gravity.
    pub fn external_power(&self) -> f64 {
        let f = self.f_ext[0];
        f.lin.dot(self.lin_vel_body())
            + f.ang.dot(self.ang_vel_body())
            + (self.prop.engine_torque - self.prop.prop_torque) * self.rotor
    }

    /// Aerodynamic and propulsive loads (FLU body frame, about the centre of mass) at a fixed
    /// state, for trimming and linearising: body velocity `v_body` and body rates `rates` in
    /// still air of density `rho`, deflections `surfaces`, throttle, rotor speed `omega`; out of
    /// ground effect. Returns the force, moment and the net rotor torque.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn static_loads(
        &self,
        v_body: DVec3,
        rates: DVec3,
        rho: f64,
        surfaces: [f64; 4],
        throttle: f64,
        omega: f64,
    ) -> (DVec3, DVec3, f64) {
        let air = AirData { density: rho, ..AirData::default() };
        let flow = AirFlow::new(&air, v_body, rates);
        let p = &self.propulsion;
        let v_axial = v_body.dot(p.axis);
        let supply = self.def.battery.as_ref().map_or(0.0, |b| b.full_voltage());
        let (thrust, q_prop) = p.prop_loads(omega, v_axial, rho);
        let (a, b) = p.engine_torque(throttle, omega, rho / crate::aero::SEA_LEVEL_DENSITY, true, supply);
        let q_eng = a - b * omega;
        let vi = p.induced_velocity(thrust, v_axial, rho);
        let wash = v_axial + 2.0 * vi;
        let x = AeroInput {
            aileron: surfaces[0],
            elevator: surfaces[1],
            rudder: surfaces[2],
            flap: surfaces[3],
            // Trim on the unstalled branch whatever the aircraft's current state.
            stall: 0.0,
            ..self.aero_input(&flow, 0.0, FREE_AIR_HEIGHT, 0.5 * rho * wash * wash)
        };
        let aero = self.def.aero.forces(&self.def.geometry, &x);
        let f = aero.force + p.axis * thrust;
        let m = aero.moment + p.position.cross(p.axis * thrust) - p.axis * (p.sense * q_eng);
        (f, m, q_eng - q_prop)
    }
}
