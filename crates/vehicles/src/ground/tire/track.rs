//! Track patch: the part of a track's ground run under one road wheel, used in place of a tyre.
//!
//! The patch carries the road wheel's load (a pad spring under the wheel) over the road-wheel
//! pitch `L` and the track width `b`. Its shear follows Janosi–Hanamoto (Wong, *Theory of
//! Ground Vehicles*, §2.4): a shoe on the ground shears the ground by `j`, the displacement
//! accumulated since it was laid down, and carries `τ = τ_max·(1 − e^(−|j|/K))` along `j`,
//! with `τ_max = μ·p` on rigid ground.
//!
//! The patch is split into [`TRACK_CELLS`] cells of length `h = L/N` along the band. Each cell
//! holds the mean shear vector `m` of the shoes over it; the shoes move through the patch with
//! the band (speed `V_b = ω·r` relative to the hull, rearward when driving forward), so
//!
//! ```text
//! dm/dt = −V_s − (|V_b|/h)·(m − m_in)
//! ```
//!
//! with the slip velocity `V_s` of the cell (longitudinal: ground speed against band speed;
//! lateral: side-slip including the yaw rate's share at the cell) and `m_in` the upstream
//! cell's shear: the neighbouring patch's end cell, or a mirror of the first cell (`−m`) at
//! the track's leading end, where fresh shoes are laid down unsheared. In steady slip `i` the
//! cells then sit at `j = i·x` of their centres, `x` the distance from the leading end, and
//! the patches sum to Wong's integral `F = A·τ_max·(1 − K/(iℓ)·(1 − e^(−iℓ/K)))` over the
//! track's contact length `ℓ`; skid steering's turning resistance comes from the cells'
//! lateral shear without a separate moment.
//!
//! A cell's stress acts along its shear while the shoes stick, and against its sliding velocity
//! once they slide (Wong and Chiang's skid-steering theory, Wong §7.3): a braked track sliding
//! longitudinally then carries little lateral stress, which a stress along the shear, whose
//! longitudinal part saturates, would overstate. The direction blends between the two over
//! shears of 0.5–1.5 `K` and sliding speeds of 1–5 cm/s, so that a parked vehicle's shear
//! spring keeps holding. The transport term is integrated implicitly (in
//! flow order), so any band speed is stable. At standstill the shear acts as a spring and a
//! parked vehicle holds without creep; below `vxlow` a damping `−k·V_s` (a corner mass on the
//! shear spring at ratio 0.7) is added to the shear, as the tyres' low-speed damping. The
//! shear is capped at `10 K`, where the stress is saturated.
//!
//! The internal resistance of the running gear is a rolling-resistance moment
//! `−f·F_z·r` against the band's motion, independent of the surface.

use super::model::{Surface, TireForces, TireState, WheelMotion};
use super::road::RoadContact;
use autonomousim_core::material::Soil;
use serde::{Deserialize, Serialize};

/// Shear cells per patch.
pub const TRACK_CELLS: usize = 8;
/// Cap of the shear displacement, in shear moduli `K`.
const SHEAR_LIMIT: f64 = 10.0;
/// Shears (in `K`) and sliding speeds (m/s) over which a cell's stress turns from the shear's
/// direction (sticking) to against the sliding velocity (sliding); both must be exceeded.
const STICKING_SHEAR: f64 = 0.5;
const SLIDING_SHEAR: f64 = 1.5;
const STICKING_SPEED: f64 = 0.01;
const SLIDING_SPEED: f64 = 0.05;
/// Band speed (m/s) over which the internal resistance changes sign.
const ROLLING_SMOOTHING: f64 = 0.05;
/// Damping ratio of a corner mass on the shear spring at standstill: higher than the tyres'
/// 0.25, since close to the friction limit the saturating shear law leaves little of it.
const LOW_SPEED_DAMPING_RATIO: f64 = 0.7;
const GRAVITY: f64 = 9.80665;

/// Parameters of a track patch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackPatch {
    /// Road-wheel centre to the ground side of the unloaded track (m): the road wheel's radius
    /// plus the track's thickness.
    pub radius: f64,
    /// Track width `b` (m).
    pub width: f64,
    /// Patch length (m): the road-wheel pitch, so the patches tile the track's ground run.
    pub length: f64,
    /// Vertical stiffness (N/m) and damping (N s/m) of the pads under the road wheel.
    pub vertical_stiffness: f64,
    #[serde(default)]
    pub vertical_damping: f64,
    /// Shear deformation modulus `K` (m) on rigid ground.
    pub shear_modulus: f64,
    /// Friction coefficient of the track on the reference surface (asphalt).
    pub mu: f64,
    /// Internal rolling-resistance coefficient of the running gear.
    #[serde(default = "default_rolling_resistance")]
    pub rolling_resistance: f64,
    /// Design load per road wheel (N); 0 (the default) shares the vehicle's weight evenly.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub nominal_load: f64,
    /// Speed below which the low-speed shear damping acts (m/s).
    #[serde(default = "default_vxlow")]
    pub vxlow: f64,
}

fn default_rolling_resistance() -> f64 {
    0.03
}

fn default_vxlow() -> f64 {
    1.0
}

fn is_zero(x: &f64) -> bool {
    *x == 0.0
}

impl TrackPatch {
    /// Check that the parameters are physical.
    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            ("radius", self.radius),
            ("width", self.width),
            ("length", self.length),
            ("vertical_stiffness", self.vertical_stiffness),
            ("shear_modulus", self.shear_modulus),
            ("mu", self.mu),
            ("vxlow", self.vxlow),
        ];
        for (name, v) in positive {
            if !(v > 0.0 && v.is_finite()) {
                return Err(format!("track {name} must be positive, got {v}"));
            }
        }
        for (name, v) in [
            ("vertical_damping", self.vertical_damping),
            ("rolling_resistance", self.rolling_resistance),
            ("nominal_load", self.nominal_load),
        ] {
            if !(v >= 0.0 && v.is_finite()) {
                return Err(format!("track {name} must be non-negative, got {v}"));
            }
        }
        Ok(())
    }

    /// Advance the shear cells by `dt` on the road plane `c` and return the patch's wrench on
    /// the road wheel.
    pub(super) fn step(
        &self,
        state: &mut TireState,
        c: &RoadContact,
        motion: &WheelMotion,
        surface: Surface,
        dt: f64,
    ) -> TireForces {
        let rho = self.radius - c.loaded_radius;
        let arm = c.point - motion.center;
        let vc = motion.velocity + motion.carrier_angvel.cross(arm);
        let (vx, vy) = (vc.dot(c.x), vc.dot(c.y));
        let yaw_rate = motion.carrier_angvel.dot(c.normal);
        let f0 = self.vertical_stiffness * rho - self.vertical_damping * motion.velocity.dot(c.normal);
        // Upstream in the soil's flow past the patch: the front neighbour when moving forward.
        let up = usize::from(vx < 0.0);
        let (fz, sinkage) = match surface.soil {
            None => {
                state.sinkage = 0.0;
                (f0.max(0.0), 0.0)
            }
            Some(soil) => {
                let z = self.sink(&soil, f0, state.sinkage, state.rut_in[up].unwrap_or(0.0), dt * vx.abs());
                state.sinkage = z;
                ((f0 - self.vertical_stiffness * z).max(0.0), z)
            }
        };
        let re = (c.loaded_radius + sinkage).max(0.5 * self.radius);
        let band = motion.spin * re;
        let vsx = vx - band;

        let n = TRACK_CELLS;
        let h = self.length / n as f64;
        let rate = dt * band.abs() / h;
        let (k, cell_limit, mu) = match surface.soil {
            None => (self.shear_modulus, self.mu * surface.mu_scale * fz / n as f64, self.mu),
            // Mohr–Coulomb over the cell: c·A + F_z·tan φ.
            Some(soil) if fz > 0.0 => {
                let limit = soil.cohesion * self.width * h + fz * soil.friction_angle.tan() / n as f64;
                (soil.shear_modulus, limit, limit * n as f64 / fz)
            }
            Some(soil) => (soil.shear_modulus, 0.0, soil.friction_angle.tan()),
        };
        let speed = vx.abs().max(band.abs());
        let k_low =
            if speed < self.vxlow { 0.5 * (1.0 + (std::f64::consts::PI * speed / self.vxlow).cos()) } else { 0.0 };
        let damping = k_low * 2.0 * LOW_SPEED_DAMPING_RATIO * (k / (mu * GRAVITY)).sqrt();
        let limit = SHEAR_LIMIT * k;

        let forward = band >= 0.0;
        let mut inflow = state.inflow[usize::from(!forward)];
        let (mut fx, mut fy, mut mz) = (0.0, 0.0, 0.0);
        let (mut su, mut sv) = (0.0, 0.0);
        for i in 0..n {
            // Cell 0 is the front one; the band carries the shoes from the front when driving
            // forward.
            let cell = if forward { i } else { n - 1 - i };
            let offset = 0.5 * self.length - (cell as f64 + 0.5) * h;
            let vy_cell = vy + yaw_rate * offset;
            let m = state.shear[cell];
            let m_in = inflow.unwrap_or([-m[0], -m[1]]);
            let mut new = [
                (m[0] - dt * vsx + rate * m_in[0]) / (1.0 + rate),
                (m[1] - dt * vy_cell + rate * m_in[1]) / (1.0 + rate),
            ];
            let j = new[0].hypot(new[1]);
            if j > limit {
                new = [new[0] * limit / j, new[1] * limit / j];
            }
            state.shear[cell] = new;
            inflow = Some(new);
            su += new[0];
            sv += new[1];
            let e = [new[0] - damping * vsx, new[1] - damping * vy_cell];
            let j = e[0].hypot(e[1]);
            if j > 0.0 {
                // Along the shear when sticking, against the sliding velocity when sliding.
                let slide = vsx.hypot(vy_cell);
                let w = ((slide - STICKING_SPEED) / (SLIDING_SPEED - STICKING_SPEED)).clamp(0.0, 1.0)
                    * ((j / k - STICKING_SHEAR) / (SLIDING_SHEAR - STICKING_SHEAR)).clamp(0.0, 1.0);
                let mut d = [(1.0 - w) * e[0] / j, (1.0 - w) * e[1] / j];
                if slide > 0.0 {
                    d = [d[0] - w * vsx / slide, d[1] - w * vy_cell / slide];
                }
                let norm = d[0].hypot(d[1]);
                let d = if norm > 1e-9 { [d[0] / norm, d[1] / norm] } else { [e[0] / j, e[1] / j] };
                let f = cell_limit * (1.0 - (-j / k).exp());
                fx += f * d[0];
                fy += f * d[1];
                mz += f * d[1] * offset;
            }
        }
        state.u = su / n as f64;
        state.v = sv / n as f64;

        // Soil: the patch compacts the rut the patches ahead left, and pushes soil ahead of it,
        // from their depth to its own. The resistance acts on the running gear, not on the
        // band: the sprocket overcomes it through the shear.
        let mut resistance = 0.0;
        if let Some(soil) = surface.soil
            && fz > 0.0
        {
            let z_in = state.rut_in[up].unwrap_or(0.0);
            let bulldozing = soil.bulldozing(self.width, sinkage, GRAVITY) - soil.bulldozing(self.width, z_in, GRAVITY);
            resistance =
                (soil.compaction(self.width, z_in, sinkage) + bulldozing.max(0.0)) * (vx / ROLLING_SMOOTHING).tanh();
        }
        let my = -self.rolling_resistance * fz * re * (band / ROLLING_SMOOTHING).tanh();
        let point = c.point - c.normal * sinkage;
        let shear = c.x * fx + c.y * fy + c.normal * fz;
        let moment = c.y * my + c.normal * mz;
        fx -= resistance;
        TireForces {
            force: shear - c.x * resistance,
            torque: moment + (point - motion.center).cross(shear),
            point,
            fx,
            fy,
            fz,
            mx: 0.0,
            my,
            mz,
            kappa: -vsx / vx.abs().max(band.abs()).max(0.1),
            tan_alpha: vy / vx.abs().max(0.1),
            deflection: rho - sinkage,
            vx,
            vy,
            vsx,
            sinkage,
        }
    }

    /// Sinkage (m) of the patch into `soil` under the force `f0` its pads would carry on rigid
    /// ground (N), in series with the pads. The rut under the patch, `rut` of the last step,
    /// is carried away by the ground's motion (`travel` m in this step) and replaced by the rut
    /// `rut_in` the patches ahead left. The soil holds up to the pressure that compacted the rut
    /// (unloading and reloading are taken as rigid), and yields along Bekker's law beyond it.
    fn sink(&self, soil: &Soil, f0: f64, rut: f64, rut_in: f64, travel: f64) -> f64 {
        let r = travel / self.length;
        let floor = (rut + r * rut_in) / (1.0 + r);
        let (kv, kz, n) = (self.vertical_stiffness, self.width * self.length * soil.modulus(self.width), soil.n);
        if f0 - kv * floor <= kz * floor.powf(n) {
            return floor;
        }
        // k_v·z + k_z·z^n = f0, increasing in z: safeguarded Newton.
        let (mut lo, mut hi) = (floor, f0 / kv);
        let mut z = (f0 / kz).powf(1.0 / n).clamp(lo, hi);
        for _ in 0..40 {
            let zn = z.powf(n);
            let g = kv * z + kz * zn - f0;
            if g > 0.0 {
                hi = z;
            } else {
                lo = z;
            }
            let mut next = z - g / (kv + kz * n * zn / z.max(1e-12));
            if !(next > lo && next < hi) {
                next = 0.5 * (lo + hi);
            }
            if (next - z).abs() < 1e-10 {
                return next;
            }
            z = next;
        }
        z
    }
}
