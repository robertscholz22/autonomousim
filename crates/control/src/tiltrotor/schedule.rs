//! The conversion schedule: mount tilt against airspeed, inside the conversion corridor, with
//! the level-flight trim at each point.

use crate::ControlError;
use autonomousim_vehicles::tiltrotor::{CorridorPoint, Tiltrotor, TiltrotorTrim, TrimLimits};

/// Speed step of the schedule (m/s).
const STEP: f64 = 1.0;
/// Tilt step of the corridor sweep (rad).
const TILT_STEP: f64 = 0.035;
/// Distance kept from the corridor's edges where it is wide enough (rad).
const MARGIN: f64 = 0.05;
/// Fastest speed tried (m/s).
const MAX_SPEED: f64 = 60.0;

/// Mount tilt and level-flight trim at one speed.
#[derive(Clone, Debug)]
pub struct SchedulePoint {
    pub speed: f64,
    pub tilt: f64,
    pub trim: TiltrotorTrim,
}

/// Mount tilt against airspeed: a smooth ramp from rotors up at `0.3·V_s` to rotors forward at
/// `1.2·V_s` (`V_s` the wing-borne stall speed), clamped into the conversion corridor and kept
/// from falling with speed; beyond the corridor sweep the rotors stay forward up to the
/// fastest speed that trims.
#[derive(Clone, Debug)]
pub struct TiltSchedule {
    points: Vec<SchedulePoint>,
    /// The conversion corridor the schedule was fitted into (up to 1.4·V_s).
    corridor: Vec<CorridorPoint>,
    stall_speed: f64,
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

impl TiltSchedule {
    pub fn new(t: &Tiltrotor, limits: &TrimLimits, rho: f64, g: f64) -> Result<Self, ControlError> {
        let def = t.def();
        let fail = |m: String| ControlError::InvalidConfig(format!("{}: {m}", def.name));
        let range = &def.controls.tilt;
        let vs = def.stall_speed(rho, g);
        let sweep_top = (1.4 * vs / STEP).ceil() * STEP;
        let speeds: Vec<f64> = (0..).map(|i| f64::from(i) * STEP).take_while(|v| *v <= sweep_top).collect();
        let corridor = t.corridor(&speeds, TILT_STEP, limits, rho, g);
        let ramp = |v: f64| range.max * smoothstep((v - 0.3 * vs) / (0.9 * vs));
        let mut points: Vec<SchedulePoint> = Vec::new();
        let mut last_tilt = f64::NEG_INFINITY;
        for c in &corridor {
            let Some((lo, hi)) = c.tilt else { break };
            // The margin keeps off the corridor's edges, not the ends of the mounts' travel.
            let (a, b) = if hi - lo > 2.0 * MARGIN {
                let a = if lo > range.min + 1e-9 { lo + MARGIN } else { lo };
                let b = if hi < range.max - 1e-9 { hi - MARGIN } else { hi };
                (a, b)
            } else {
                (lo, hi)
            };
            let tilt = ramp(c.speed).clamp(a, b).max(last_tilt).min(hi);
            let trim = match points.last() {
                Some(p) => t.trim_near(c.speed, tilt, rho, g, &p.trim).or_else(|_| t.trim(c.speed, tilt, rho, g)),
                None => t.trim(c.speed, tilt, rho, g),
            };
            let Ok(trim) = trim else { break };
            last_tilt = tilt;
            points.push(SchedulePoint { speed: c.speed, tilt, trim });
        }
        if points.is_empty() {
            return Err(fail("no hover trim for the schedule".into()));
        }
        // Rotors forward beyond the sweep, as long as the aircraft trims within its limits.
        let mut speed = points.last().map_or(0.0, |p| p.speed) + STEP;
        while speed <= MAX_SPEED && last_tilt >= range.max - 1e-9 {
            let near = &points.last().expect("points").trim;
            match t.trim_near(speed, range.max, rho, g, near) {
                Ok(trim) if trim.feasible(def, limits) => {
                    points.push(SchedulePoint { speed, tilt: range.max, trim });
                }
                _ => break,
            }
            speed += STEP;
        }
        Ok(Self { points, corridor, stall_speed: vs })
    }

    pub fn points(&self) -> &[SchedulePoint] {
        &self.points
    }

    /// Feasible tilt range against speed, as swept for the schedule (up to 1.4·V_s; the
    /// rotors-forward points beyond it are feasible trims too).
    pub fn corridor(&self) -> &[CorridorPoint] {
        &self.corridor
    }

    /// Share of the manoeuvring on the wing at airspeed `speed` (m/s): 0 up to 0.9·V_s, 1 from
    /// 1.3·V_s, smooth between.
    pub fn wing_share(&self, speed: f64) -> f64 {
        smoothstep((speed - 0.9 * self.stall_speed) / (0.4 * self.stall_speed))
    }

    /// Wing-borne stall speed (m/s).
    pub fn stall_speed(&self) -> f64 {
        self.stall_speed
    }

    /// Fastest scheduled speed (m/s).
    pub fn max_speed(&self) -> f64 {
        self.points.last().map_or(0.0, |p| p.speed)
    }

    /// Scheduled tilt at airspeed `speed` (m/s; linear between points, held beyond them).
    pub fn tilt(&self, speed: f64) -> f64 {
        let (i, f) = self.locate(speed);
        match self.points.get(i + 1) {
            Some(n) => self.points[i].tilt + (n.tilt - self.points[i].tilt) * f,
            None => self.points[i].tilt,
        }
    }

    /// Scheduled trim pitch (rad) at airspeed `speed`.
    pub fn pitch(&self, speed: f64) -> f64 {
        let (i, f) = self.locate(speed);
        match self.points.get(i + 1) {
            Some(n) => self.points[i].trim.pitch + (n.trim.pitch - self.points[i].trim.pitch) * f,
            None => self.points[i].trim.pitch,
        }
    }

    fn locate(&self, speed: f64) -> (usize, f64) {
        let x = (speed.max(0.0) / STEP).min((self.points.len() - 1) as f64);
        let i = (x.floor() as usize).min(self.points.len() - 1);
        (i, x - i as f64)
    }
}
