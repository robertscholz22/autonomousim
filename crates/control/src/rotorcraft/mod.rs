//! Helicopters: normalised action modes and the controller between setpoints and the
//! helicopter's [`HelicopterInput`].
//!
//! | Mode | Components | Meaning at ±1 |
//! |---|---|---|
//! | `sticks` | collective, longitudinal, lateral, pedal | full travel (+ collective up, stick forward, stick right, right pedal) |
//! | `rates` | roll, pitch, yaw rate, collective | ±`rates` about the body FLU axes; collective as in `sticks` |
//! | `attitude` | roll, pitch, yaw rate, collective | ±`roll` (+ right wing down), ±`pitch` (+ nose up), ±`yaw_rate` (+ counter-clockwise); collective as in `sticks` |
//! | `velocity` (default) | vx, vy, vz, yaw rate | heading-frame velocity (+ forward, left, up): vx from −`speed[1]` to +`speed[0]`, vy ±`speed[1]`, vz ±`speed[2]`; ±`yaw_rate` |
//!
//! Components outside `[−1, 1]` are clipped and non-finite ones read as 0. `rates`,
//! `attitude` and `velocity` share their names with the multirotor's and aircraft's modes;
//! a group resolves them by its vehicle.
//!
//! The controller is gain-scheduled on forward airspeed from trims and linear models of the
//! helicopter ([`Helicopter::trim`], [`Helicopter::linearize`]) at a grid of speeds: a rate
//! loop that inverts the linear angular-acceleration model (with PI correction, and a lead on
//! the demand against the flapping lag the quasi-steady model omits), an attitude loop on the
//! Euler angles that holds the heading the yaw rate integrates to, a velocity loop that tilts
//! from the trim attitude and drives the collective through the heave model, and a position
//! loop for scripted hover.

use crate::ControlError;
use crate::fixedwing::euler;
use autonomousim_core::math::quat::wrap_angle;
use autonomousim_vehicles::rotorcraft::{Helicopter, HelicopterDef, HelicopterInput, HelicopterLinear as L};
use glam::{DMat3, DQuat, DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HelicopterActionMode {
    /// Collective, longitudinal and lateral cyclic, pedals.
    Sticks,
    /// Body rates and collective.
    Rates,
    /// Roll, pitch, yaw rate and collective.
    Attitude,
    /// Heading-frame velocity and yaw rate.
    #[default]
    Velocity,
}

impl HelicopterActionMode {
    pub const ALL: [HelicopterActionMode; 4] = [
        HelicopterActionMode::Sticks,
        HelicopterActionMode::Rates,
        HelicopterActionMode::Attitude,
        HelicopterActionMode::Velocity,
    ];

    pub fn name(self) -> &'static str {
        match self {
            HelicopterActionMode::Sticks => "sticks",
            HelicopterActionMode::Rates => "rates",
            HelicopterActionMode::Attitude => "attitude",
            HelicopterActionMode::Velocity => "velocity",
        }
    }

    /// Action length.
    pub fn dim(self) -> usize {
        4
    }
}

impl fmt::Display for HelicopterActionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for HelicopterActionMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL.into_iter().find(|m| m.name() == s).ok_or_else(|| format!("unknown helicopter action mode {s:?}"))
    }
}

/// What a helicopter's controller tracks. Angles, rates and velocities in FLU/ENU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HelicopterSetpoint {
    /// Pilot inputs, passed through.
    Sticks(HelicopterInput),
    /// Body rates (rad/s) and the pilot's collective (−1…1).
    Rates { rates: DVec3, collective: f64 },
    /// Roll (+ right wing down) and pitch (+ nose up) angles (rad), yaw rate (rad/s,
    /// counter-clockwise) and the pilot's collective.
    Attitude { roll: f64, pitch: f64, yaw_rate: f64, collective: f64 },
    /// Velocity over ground in the heading frame (m/s; forward, left, up) and yaw rate.
    Velocity { velocity: DVec3, yaw_rate: f64 },
    /// Hover at a world position (m) with a heading (rad).
    Position { position: DVec3, yaw: f64 },
}

/// Everything configurable about [`HelicopterController`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HelicopterConfig {
    /// Rate-loop bandwidth (rad/s); default: `0.5/(flap_lead·τ_flap + τ_servo)`, the lags
    /// left after the lead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_bandwidth: Option<f64>,
    /// Each outer loop's bandwidth is the inner one's over this ratio (attitude, velocity,
    /// position).
    pub loop_ratio: f64,
    /// Heave (climb-rate) bandwidth over the rate bandwidth.
    pub heave_ratio: f64,
    /// Lead on the demanded roll and pitch acceleration against the hover flapping lag
    /// τ_flap, `(τs + 1)/(ατs + 1)` with α = `flap_lead` (1: no lead).
    pub flap_lead: f64,
    /// Largest roll and pitch away from the trim attitude the velocity loop asks for (rad).
    pub max_tilt: f64,
    /// Largest speed the position loop asks for (m/s, horizontal and vertical).
    pub max_speed: [f64; 2],
    /// Air density and gravity of the schedule's trims.
    pub design_density: f64,
    pub gravity: f64,
}

impl Default for HelicopterConfig {
    fn default() -> Self {
        Self {
            rate_bandwidth: None,
            loop_ratio: 4.0,
            heave_ratio: 0.3,
            flap_lead: 0.25,
            max_tilt: 25f64.to_radians(),
            max_speed: [5.0, 2.0],
            design_density: 1.225,
            gravity: 9.80665,
        }
    }
}

/// Largest heading error the attitude loop acts on (rad).
const HEADING_LAG: f64 = 0.5;

/// Time (s) the velocity loop takes on the ground to lower the collective to zero thrust.
const SETTLE_TIME: f64 = 1.0;

/// One point of the gain schedule.
#[derive(Clone, Copy, Debug)]
struct Point {
    speed: f64,
    /// Trim inputs, pitch and roll.
    controls: [f64; 4],
    pitch: f64,
    roll: f64,
    /// Body velocity through the air.
    velocity: DVec3,
    /// Angular acceleration per body rate, per body velocity, per `[lateral, longitudinal,
    /// pedal]` and per collective.
    rate: DMat3,
    speed_coupling: DMat3,
    cyclic: DMat3,
    collective: DVec3,
    /// Heave acceleration (body w) per collective and per body w.
    heave: f64,
    heave_damping: f64,
}

impl Point {
    fn lerp(&self, o: &Point, t: f64) -> Point {
        let m = |a: DMat3, b: DMat3| a * (1.0 - t) + b * t;
        let s = |a: f64, b: f64| a + (b - a) * t;
        let mut controls = self.controls;
        for (c, o) in controls.iter_mut().zip(o.controls) {
            *c = s(*c, o);
        }
        Point {
            speed: s(self.speed, o.speed),
            controls,
            pitch: s(self.pitch, o.pitch),
            roll: s(self.roll, o.roll),
            velocity: self.velocity.lerp(o.velocity, t),
            rate: m(self.rate, o.rate),
            speed_coupling: m(self.speed_coupling, o.speed_coupling),
            cyclic: m(self.cyclic, o.cyclic),
            collective: self.collective.lerp(o.collective, t),
            heave: s(self.heave, o.heave),
            heave_damping: s(self.heave_damping, o.heave_damping),
        }
    }
}

/// Setpoint → helicopter input.
#[derive(Clone, Debug)]
pub struct HelicopterController {
    config: HelicopterConfig,
    dt: f64,
    schedule: Vec<Point>,
    /// Rate, attitude, velocity, position and heave bandwidths (rad/s).
    k_rate: f64,
    k_att: f64,
    k_vel: f64,
    k_pos: f64,
    k_heave: f64,
    /// Integrals: angular acceleration (rad/s²), horizontal and vertical acceleration (m/s²).
    i_rate: DVec3,
    i_vel: DVec3,
    /// Heading reference of the attitude loop (rad); `None` until it first runs.
    heading: Option<f64>,
    /// Hover flapping time constant (s) and the lead filter's state (lateral, longitudinal).
    tau_flap: f64,
    lead: Option<DVec2>,
    /// Touching the ground in the last update.
    grounded: bool,
    /// Share (0…1) of the way from the velocity loop's collective to the zero-thrust one,
    /// growing on the ground while no climb is asked for.
    settle: f64,
}

impl HelicopterController {
    pub fn new(def: &Arc<HelicopterDef>, dt: f64, config: &HelicopterConfig) -> Result<Self, ControlError> {
        let fail = |m: String| ControlError::InvalidConfig(format!("{}: {m}", def.name));
        def.validate().map_err(|e| fail(e.to_string()))?;
        if dt.is_nan()
            || dt <= 0.0
            || config.loop_ratio < 1.0
            || config.heave_ratio <= 0.0
            || config.flap_lead.is_nan()
            || config.flap_lead <= 0.0
        {
            return Err(fail(format!("helicopter controller step {dt}, config {config:?}")));
        }
        let heli = Helicopter::new(def.clone(), dt);
        let (rho, g) = (config.design_density, config.gravity);
        // Speeds up to the fastest trim, in steps of half the hover induced velocity.
        let r = &def.main_rotor.rotor;
        let v_h = (heli.mass() * g / (2.0 * rho * r.disc_area())).sqrt();
        let mut schedule = Vec::new();
        for k in 0..60 {
            let speed = 0.5 * v_h * f64::from(k);
            let Ok(t) = heli.trim(speed, rho, g) else { break };
            let lin = heli.linearize(&t);
            let col = |j: usize| DVec3::new(lin.b[L::P][j], lin.b[L::Q][j], lin.b[L::R][j]);
            let a = |j: usize| DVec3::new(lin.a[L::P][j], lin.a[L::Q][j], lin.a[L::R][j]);
            let rate = DMat3::from_cols(a(L::P), a(L::Q), a(L::R));
            let speed_coupling = DMat3::from_cols(a(L::U), a(L::V), a(L::W));
            let cyclic = DMat3::from_cols(col(2), col(1), col(3));
            if cyclic.determinant().abs() < 1e-12 || lin.b[L::W][0] <= 0.0 {
                return Err(fail(format!("no control authority at {speed} m/s")));
            }
            schedule.push(Point {
                speed,
                controls: t.controls.to_array(),
                pitch: t.pitch,
                roll: t.roll,
                velocity: t.velocity_body,
                rate,
                speed_coupling,
                cyclic,
                collective: col(0),
                heave: lin.b[L::W][0],
                heave_damping: lin.a[L::W][L::W],
            });
        }
        if schedule.is_empty() {
            return Err(fail("no hover trim".into()));
        }
        let hover = heli.trim(0.0, rho, g).expect("hover trim");
        let tau_flap = hover.loads.main.time_constant;
        let c = &def.controls;
        let tau_servo =
            [c.collective.tau, c.longitudinal.tau, c.lateral.tau, c.pedal.tau].into_iter().fold(0.0, f64::max);
        let lagging = if config.flap_lead < 1.0 { config.flap_lead * tau_flap } else { tau_flap };
        let k_rate = config.rate_bandwidth.unwrap_or(0.5 / (lagging + tau_servo));
        let k_att = k_rate / config.loop_ratio;
        let k_vel = k_att / config.loop_ratio;
        let k_pos = k_vel / config.loop_ratio;
        let k_heave = k_rate * config.heave_ratio;
        Ok(Self {
            config: config.clone(),
            dt,
            schedule,
            k_rate,
            k_att,
            k_vel,
            k_pos,
            k_heave,
            tau_flap,
            i_rate: DVec3::ZERO,
            i_vel: DVec3::ZERO,
            heading: None,
            lead: None,
            grounded: false,
            settle: 0.0,
        })
    }

    pub fn config(&self) -> &HelicopterConfig {
        &self.config
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Fastest trimmed speed of the schedule (m/s).
    pub fn max_speed(&self) -> f64 {
        self.schedule.last().map_or(0.0, |p| p.speed)
    }

    /// Rate, attitude, velocity, position and heave bandwidths (rad/s).
    pub fn bandwidths(&self) -> [f64; 5] {
        [self.k_rate, self.k_att, self.k_vel, self.k_pos, self.k_heave]
    }

    /// Clear the integrators.
    pub fn reset(&mut self) {
        self.i_rate = DVec3::ZERO;
        self.i_vel = DVec3::ZERO;
        self.heading = None;
        self.lead = None;
        self.settle = 0.0;
    }

    /// Trim inputs, pitch and roll at forward airspeed `speed` (m/s).
    pub fn trim_at(&self, speed: f64) -> (HelicopterInput, f64, f64) {
        let p = self.point(speed);
        (HelicopterInput::from_array(p.controls), p.pitch, p.roll)
    }

    fn point(&self, speed: f64) -> Point {
        let s = &self.schedule;
        let step = if s.len() > 1 { s[1].speed } else { 1.0 };
        let x = (speed.max(0.0) / step).min((s.len() - 1) as f64);
        let i = (x.floor() as usize).min(s.len().saturating_sub(2));
        if s.len() == 1 {
            return s[0];
        }
        s[i].lerp(&s[i + 1], x - i as f64)
    }

    /// Pilot input for `setpoint`.
    pub fn update(&mut self, setpoint: &HelicopterSetpoint, h: &Helicopter) -> HelicopterInput {
        let point = self.point(h.flow().velocity.x);
        // On the ground the skids hold the attitude: the loops would wind up against them and
        // roll the helicopter over as a skid unloads. Clear the rate and horizontal integrators
        // and hold the attitude as it stands (the vertical integrator keeps it down after a
        // descent until the pilot climbs).
        self.grounded = !h.contacts().is_empty();
        if self.grounded {
            self.i_rate = DVec3::ZERO;
            self.i_vel = DVec3::new(0.0, 0.0, self.i_vel.z);
        }
        match *setpoint {
            HelicopterSetpoint::Sticks(input) => input.clamped(),
            HelicopterSetpoint::Rates { rates, collective } => {
                self.heading = None;
                self.rate_loop(&point, h, rates, collective)
            }
            HelicopterSetpoint::Attitude { roll, pitch, yaw_rate, collective } => {
                let rates = self.attitude_loop(h, roll, pitch, yaw_rate, None);
                self.rate_loop(&point, h, rates, collective)
            }
            HelicopterSetpoint::Velocity { velocity, yaw_rate } => self.velocity_loop(h, velocity, yaw_rate, None),
            HelicopterSetpoint::Position { position, yaw } => {
                let heading = euler(h.orientation()).2;
                let err = position - h.position();
                let [vh, vz] = self.config.max_speed;
                let horizontal = (err.truncate() * self.k_pos).clamp_length_max(vh);
                let vertical = (err.z * self.k_heave / self.config.loop_ratio).clamp(-vz, vz);
                let v = DQuat::from_rotation_z(-heading) * horizontal.extend(vertical);
                self.velocity_loop(h, v, 0.0, Some(yaw))
            }
        }
    }

    /// Body rate references for Euler-angle targets and a yaw rate, holding the heading the
    /// yaw rate integrates to (or `heading`): a rate loop alone drifts where the fin is
    /// directionally unstable (sideways and backward flight).
    fn attitude_loop(&mut self, h: &Helicopter, roll: f64, pitch: f64, yaw_rate: f64, heading: Option<f64>) -> DVec3 {
        let (phi, theta, psi) = euler(h.orientation());
        let (roll, pitch) = if self.grounded { (phi, theta) } else { (roll, pitch) };
        let reference = heading.unwrap_or_else(|| self.heading.unwrap_or(psi) + yaw_rate * self.dt);
        // Hold the reference within reach so that it does not wind up.
        let lag = wrap_angle(reference - psi).clamp(-HEADING_LAG, HEADING_LAG);
        self.heading = Some(wrap_angle(psi + lag));
        let yaw_rate = yaw_rate + self.k_att * lag;
        let phi_dot = self.k_att * wrap_angle(roll - phi);
        let theta_dot = self.k_att * (pitch - theta);
        // Body rates of Euler rates for R_z(ψ)·R_y(−θ)·R_x(φ).
        DVec3::new(
            phi_dot + yaw_rate * theta.sin(),
            -theta_dot * phi.cos() + yaw_rate * theta.cos() * phi.sin(),
            theta_dot * phi.sin() + yaw_rate * theta.cos() * phi.cos(),
        )
    }

    /// Cyclic and pedal from the inverted angular-acceleration model; `collective` passes.
    fn rate_loop(&mut self, p: &Point, h: &Helicopter, rates: DVec3, collective: f64) -> HelicopterInput {
        let w = h.ang_vel_body();
        let err = rates - w;
        let wanted = err * self.k_rate + self.i_rate;
        let d_col = collective - p.controls[0];
        // Lead on the demanded roll and pitch acceleration, (τs + 1)/(ατs + 1), against the
        // tip-path plane's lag behind the cyclic. The terms that cancel the rotor's own response
        // to the rates, the airspeed and the collective go without: they lag alike.
        let now = wanted.truncate();
        let alpha = self.config.flap_lead;
        let led = if alpha < 1.0 {
            let lagged = self.lead.unwrap_or(now);
            let next = lagged + (now - lagged) * (self.dt / (alpha * self.tau_flap)).min(1.0);
            if next.is_finite() {
                self.lead = Some(next);
            }
            now / alpha + lagged * (1.0 - 1.0 / alpha)
        } else {
            now
        };
        let dv = h.flow().velocity - p.velocity;
        let demand = led.extend(wanted.z) - p.rate * w - p.speed_coupling * dv - p.collective * d_col;
        let cyclic = p.cyclic.inverse() * demand;
        let raw = [p.controls[2] + cyclic.x, p.controls[1] + cyclic.y, p.controls[3] + cyclic.z];
        let clamped = raw.map(|u| u.clamp(-1.0, 1.0));
        // Integrate (k²/4 per unit error: critically damped) unless a control saturates.
        if raw == clamped {
            let next = self.i_rate + err * (0.25 * self.k_rate * self.k_rate * self.dt);
            if next.is_finite() {
                self.i_rate = next;
            }
        }
        HelicopterInput { collective, longitudinal: clamped[1], lateral: clamped[0], pedal: clamped[2] }.clamped()
    }

    /// Heading-frame velocity: tilt from the trim attitude for horizontal acceleration, and
    /// the collective through the heave model for vertical acceleration.
    fn velocity_loop(
        &mut self,
        h: &Helicopter,
        velocity: DVec3,
        yaw_rate: f64,
        heading: Option<f64>,
    ) -> HelicopterInput {
        let g = self.config.gravity;
        let (phi, theta, psi) = euler(h.orientation());
        let v = DQuat::from_rotation_z(-psi) * h.lin_vel_world();
        let err = velocity - v;
        let k = DVec3::new(self.k_vel, self.k_vel, self.k_heave);
        let acc = err * k + self.i_vel;
        // Trim attitude and collective at the airspeed the reference velocity gives: scheduled
        // on the measured airspeed they would feed speed back positively where the trim
        // attitude steepens with speed.
        let p = &self.point(h.flow().velocity.x + velocity.x - v.x);
        let tilt = self.config.max_tilt;
        // Forward acceleration pitches the nose down, leftward rolls left.
        let pitch = p.pitch - (acc.x / g).atan().clamp(-tilt, tilt);
        let roll = p.roll - (acc.y / g).atan().clamp(-tilt, tilt);
        let lift = (theta.cos() * phi.cos()).max(0.5);
        // Heave inverted about the trim: the climb rate's damping offsets the demand.
        let raw_col = p.controls[0] + (acc.z - p.heave_damping * v.z) / (p.heave * lift);
        // Settled on the ground (no climb asked for), the collective goes down to zero thrust
        // over `SETTLE_TIME`: with the rotor still carrying the weight the skids would skate
        // and a skid that catches would roll the helicopter over.
        self.settle =
            if self.grounded && velocity.z <= 0.0 { (self.settle + self.dt / SETTLE_TIME).min(1.0) } else { 0.0 };
        let flat = p.controls[0] - g / (p.heave * lift);
        let collective = (raw_col + self.settle * (flat.min(raw_col) - raw_col)).clamp(-1.0, 1.0);
        let rates = self.attitude_loop(h, roll, pitch, yaw_rate, heading);
        let out = self.rate_loop(p, h, rates, collective);
        // Integrate (k²/4 per unit error) up to 0.3 g, holding the vertical while the
        // collective saturates.
        let mut next =
            self.i_vel + DVec3::new(err.x * k.x * k.x, err.y * k.y * k.y, err.z * k.z * k.z) * (0.25 * self.dt);
        if raw_col != collective {
            next.z = self.i_vel.z;
        }
        let lim = 0.3 * g;
        let next = DVec3::new(next.x.clamp(-lim, lim), next.y.clamp(-lim, lim), next.z.clamp(-lim, lim));
        if next.is_finite() {
            self.i_vel = next;
        }
        out
    }
}

/// Setpoint ranges of the normalised action modes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HelicopterActionLimits {
    /// Body rates at ±1 in `rates` (rad/s).
    pub rates: DVec3,
    /// Roll and pitch at ±1 in `attitude` (rad).
    pub roll: f64,
    pub pitch: f64,
    /// Yaw rate at ±1 in `attitude` and `velocity` (rad/s).
    pub yaw_rate: f64,
    /// `velocity` speeds (m/s): forward at +1, sideways (and backward at −1), vertical;
    /// default: 80 % of the fastest trim, twice the hover induced velocity, and the hover
    /// induced velocity over two.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<[f64; 3]>,
}

impl Default for HelicopterActionLimits {
    fn default() -> Self {
        Self {
            rates: DVec3::new(1.5, 1.0, 1.0),
            roll: 30f64.to_radians(),
            pitch: 20f64.to_radians(),
            yaw_rate: 0.5,
            speed: None,
        }
    }
}

/// Maps normalised actions of one [`HelicopterActionMode`] to setpoints.
#[derive(Clone, Debug)]
pub struct HelicopterActionMap {
    mode: HelicopterActionMode,
    limits: HelicopterActionLimits,
    speed: [f64; 3],
}

impl HelicopterActionMap {
    /// Map for `mode`, with the default speeds from the controller's schedule and `def`.
    pub fn new(
        mode: HelicopterActionMode,
        limits: &HelicopterActionLimits,
        def: &HelicopterDef,
        controller: &HelicopterController,
    ) -> Result<Self, ControlError> {
        let speed = limits.speed.unwrap_or_else(|| {
            let g = controller.config().gravity;
            let rho = controller.config().design_density;
            let v_h = (def.body.mass * g / (2.0 * rho * def.main_rotor.rotor.disc_area())).sqrt();
            [0.8 * controller.max_speed(), 2.0 * v_h, 0.5 * v_h]
        });
        if !speed.iter().all(|s| s.is_finite() && *s >= 0.0) {
            return Err(ControlError::InvalidConfig(format!("helicopter action speeds {speed:?}")));
        }
        Ok(Self { mode, limits: limits.clone(), speed })
    }

    pub fn mode(&self) -> HelicopterActionMode {
        self.mode
    }

    pub fn dim(&self) -> usize {
        self.mode.dim()
    }

    pub fn limits(&self) -> &HelicopterActionLimits {
        &self.limits
    }

    /// `velocity` speeds (forward, sideways and backward, vertical; m/s).
    pub fn speeds(&self) -> [f64; 3] {
        self.speed
    }

    /// Setpoint for `action` (length [`dim`](Self::dim), already in `[−1, 1]`).
    pub fn setpoint(&self, action: &[f64]) -> HelicopterSetpoint {
        let l = &self.limits;
        let a = [action[0], action[1], action[2], action[3]];
        match self.mode {
            HelicopterActionMode::Sticks => HelicopterSetpoint::Sticks(HelicopterInput::from_array(a)),
            HelicopterActionMode::Rates => {
                HelicopterSetpoint::Rates { rates: DVec3::new(a[0], a[1], a[2]) * l.rates, collective: a[3] }
            }
            HelicopterActionMode::Attitude => HelicopterSetpoint::Attitude {
                roll: a[0] * l.roll,
                pitch: a[1] * l.pitch,
                yaw_rate: a[2] * l.yaw_rate,
                collective: a[3],
            },
            HelicopterActionMode::Velocity => {
                let [fwd, side, vert] = self.speed;
                let vx = if a[0] >= 0.0 { a[0] * fwd } else { a[0] * side };
                HelicopterSetpoint::Velocity {
                    velocity: DVec3::new(vx, a[1] * side, a[2] * vert),
                    yaw_rate: a[3] * l.yaw_rate,
                }
            }
        }
    }
}
