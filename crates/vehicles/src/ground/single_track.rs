//! Linearised dynamics of single-track vehicles: the Whipple bicycle model of Meijaard,
//! Papadopoulos, Ruina & Schwab (2007), "Linearized dynamics equations for the balance and
//! steer of a bicycle: a benchmark and review", Proc. R. Soc. A 463.
//!
//! In the paper's frame (x forward, y right, z down; origin at the rear contact point) the
//! lean `φ` of the rear frame and the steer angle `δ` (positive to the right) obey
//!
//! ```text
//! M q̈ + v C₁ q̇ + (g K₀ + v² K₂) q = f,   q = (φ, δ), f = (lean torque, steer torque)
//! ```
//!
//! for knife-edge wheels rolling without slip at forward speed `v`. The four bodies are the
//! rear frame B (with the rider locked to it), the front frame H (fork and handlebar) and the
//! wheels R and F, taken from a [`WheeledDef`] at its design pose.

use super::{WheeledDef, inertia_tensor};
use glam::{DMat3, DMat4, DVec3, DVec4};

/// Parameters of the benchmark bicycle (Meijaard et al. 2007, Table 1), in the paper's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhippleParams {
    /// Wheelbase `w`, trail `c` (m) and steer axis tilt `λ` from the vertical (rad).
    pub wheelbase: f64,
    pub trail: f64,
    pub steer_axis_tilt: f64,
    /// Rear wheel: radius, mass, moments of inertia about its diameter and its spin axis.
    pub rear_radius: f64,
    pub rear_mass: f64,
    pub rear_ixx: f64,
    pub rear_iyy: f64,
    /// Rear frame B: centre of mass (x, z), mass and inertia tensor about it.
    pub body: RigidBody,
    /// Front frame H: centre of mass (x, z), mass and inertia tensor about it.
    pub front: RigidBody,
    /// Front wheel: radius, mass, moments of inertia about its diameter and its spin axis.
    pub front_radius: f64,
    pub front_mass: f64,
    pub front_ixx: f64,
    pub front_iyy: f64,
}

/// A rigid body in the paper's frame: mass, centre of mass and inertia tensor about it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidBody {
    pub mass: f64,
    pub com: DVec3,
    pub inertia: DMat3,
}

impl RigidBody {
    pub const ZERO: Self = Self { mass: 0.0, com: DVec3::ZERO, inertia: DMat3::ZERO };

    /// The two bodies as one.
    pub fn merge(self, other: Self) -> Self {
        let mass = self.mass + other.mass;
        if mass == 0.0 {
            return Self::ZERO;
        }
        let com = (self.com * self.mass + other.com * other.mass) / mass;
        let shifted = |b: Self| {
            let d = b.com - com;
            b.inertia + (DMat3::IDENTITY * d.length_squared() - outer(d, d)) * b.mass
        };
        Self { mass, com, inertia: shifted(self) + shifted(other) }
    }
}

fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// The linearised equations' matrices (row major; rows: lean, steer).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhippleMatrices {
    pub m: [[f64; 2]; 2],
    pub c1: [[f64; 2]; 2],
    pub k0: [[f64; 2]; 2],
    pub k2: [[f64; 2]; 2],
}

impl WhippleParams {
    /// The benchmark bicycle of Meijaard et al. (2007), Table 1.
    pub fn benchmark() -> Self {
        let pi = std::f64::consts::PI;
        let tensor =
            |xx: f64, yy: f64, zz: f64, xz: f64| inertia_tensor(DVec3::new(xx, yy, zz), DVec3::new(0.0, xz, 0.0));
        Self {
            wheelbase: 1.02,
            trail: 0.08,
            steer_axis_tilt: pi / 10.0,
            rear_radius: 0.3,
            rear_mass: 2.0,
            rear_ixx: 0.0603,
            rear_iyy: 0.12,
            body: RigidBody { mass: 85.0, com: DVec3::new(0.3, 0.0, -0.9), inertia: tensor(9.2, 11.0, 2.8, 2.4) },
            front: RigidBody {
                mass: 4.0,
                com: DVec3::new(0.9, 0.0, -0.7),
                inertia: tensor(0.05892, 0.06, 0.00708, -0.00756),
            },
            front_radius: 0.35,
            front_mass: 3.0,
            front_ixx: 0.1405,
            front_iyy: 0.28,
        }
    }

    /// The Whipple bicycle of a single-track `def` at its design pose, the rider locked
    /// upright. The rear frame lumps the chassis, the rider and the rear wheel's carrier; the
    /// front frame the steered body and the front wheel's carrier. The wheels spin with their
    /// driveline share and are assumed symmetric about the spin axis (`I_zz = I_xx`). The
    /// ground is under the rear wheel at its tyre's unloaded radius.
    pub fn from_def(def: &WheeledDef) -> Result<Self, String> {
        let (head, front_wheel) = def.steering_head().ok_or("no steering head")?;
        if !def.is_single_track() || def.num_wheels() != 2 || def.num_units() != 1 || front_wheel != 1 {
            return Err("the Whipple model needs one rear wheel and one steered front wheel".into());
        }
        let spin = def.spin_inertia();
        let (rear, front) = (&def.axles[0], &def.axles[1]);
        let rear_radius = def.tire(0).radius();
        let front_radius = def.tire(1).radius();
        let ground = rear.position.z - rear_radius;
        let origin = DVec3::new(rear.position.x, 0.0, ground);
        // Chassis frame (x forward, y left, z up) to the paper's (y right, z down).
        let flip = DMat3::from_diagonal(DVec3::new(1.0, -1.0, -1.0));
        let body = |mass: f64, com: DVec3, inertia: DMat3| RigidBody {
            mass,
            com: flip * (com - origin),
            inertia: flip * inertia * flip,
        };
        let carrier = |a: &super::AxleDef| {
            a.suspension
                .as_ref()
                .map_or(RigidBody::ZERO, |s| body(s.carrier_mass, a.position, DMat3::from_diagonal(s.carrier_inertia)))
        };
        let mut b = body(def.chassis.mass, def.chassis.com, def.chassis.inertia_tensor()).merge(carrier(rear));
        if let Some(r) = &def.rider {
            b = b.merge(body(r.mass, r.com, inertia_tensor(r.inertia, r.products)));
        }
        let h = body(head.mass, head.com, inertia_tensor(head.inertia, head.products)).merge(carrier(front));
        let lambda = head.angle;
        Ok(Self {
            wheelbase: front.position.x - rear.position.x,
            trail: (front_radius * lambda.sin() - head.offset) / lambda.cos(),
            steer_axis_tilt: lambda,
            rear_radius,
            rear_mass: rear.wheel.mass,
            rear_ixx: rear.wheel.inertia.x,
            rear_iyy: spin[0],
            body: b,
            front: h,
            front_radius,
            front_mass: front.wheel.mass,
            front_ixx: front.wheel.inertia.x,
            front_iyy: spin[1],
        })
    }

    /// `M`, `C₁`, `K₀`, `K₂` (Meijaard et al. 2007, Appendix A).
    pub fn matrices(&self) -> WhippleMatrices {
        let (w, c, lambda) = (self.wheelbase, self.trail, self.steer_axis_tilt);
        let (sl, cl) = lambda.sin_cos();
        let (rr, mr, irxx, iryy) = (self.rear_radius, self.rear_mass, self.rear_ixx, self.rear_iyy);
        let (rf, mf, ifxx, ifyy) = (self.front_radius, self.front_mass, self.front_ixx, self.front_iyy);
        let (mb, xb, zb, ib) = (self.body.mass, self.body.com.x, self.body.com.z, self.body.inertia);
        let (mh, xh, zh, ih) = (self.front.mass, self.front.com.x, self.front.com.z, self.front.inertia);
        let (ibxx, ibxz, ibzz) = (ib.x_axis.x, ib.z_axis.x, ib.z_axis.z);
        let (ihxx, ihxz, ihzz) = (ih.x_axis.x, ih.z_axis.x, ih.z_axis.z);

        // The whole bicycle.
        let mt = mr + mb + mh + mf;
        let xt = (xb * mb + xh * mh + w * mf) / mt;
        let zt = (-rr * mr + zb * mb + zh * mh - rf * mf) / mt;
        let itxx = irxx + ibxx + ihxx + ifxx + mr * rr * rr + mb * zb * zb + mh * zh * zh + mf * rf * rf;
        let itxz = ibxz + ihxz - mb * xb * zb - mh * xh * zh + mf * w * rf;
        let (irzz, ifzz) = (irxx, ifxx);
        let itzz = irzz + ibzz + ihzz + ifzz + mb * xb * xb + mh * xh * xh + mf * w * w;
        // The front assembly.
        let ma = mh + mf;
        let xa = (xh * mh + w * mf) / ma;
        let za = (zh * mh - rf * mf) / ma;
        let iaxx = ihxx + ifxx + mh * (zh - za).powi(2) + mf * (rf + za).powi(2);
        let iaxz = ihxz - mh * (xh - xa) * (zh - za) + mf * (w - xa) * (rf + za);
        let iazz = ihzz + ifzz + mh * (xh - xa).powi(2) + mf * (w - xa).powi(2);
        let ua = (xa - w - c) * cl - za * sl;
        let iall = ma * ua * ua + iaxx * sl * sl + 2.0 * iaxz * sl * cl + iazz * cl * cl;
        let ialx = -ma * ua * za + iaxx * sl + iaxz * cl;
        let ialz = ma * ua * xa + iaxz * sl + iazz * cl;
        let mu = c / w * cl;
        let (sr, sf) = (iryy / rr, ifyy / rf);
        let st = sr + sf;
        let sa = ma * ua + mu * mt * xt;

        let m = [[itxx, ialx + mu * itxz], [ialx + mu * itxz, iall + 2.0 * mu * ialz + mu * mu * itzz]];
        let k0 = [[mt * zt, -sa], [-sa, -sa * sl]];
        let k2 = [[0.0, (st - mt * zt) / w * cl], [0.0, (sa + sf * sl) / w * cl]];
        let c1 = [
            [0.0, mu * st + sf * cl + itxz / w * cl - mu * mt * zt],
            [-(mu * st + sf * cl), ialz / w * cl + mu * (sa + itzz / w * cl)],
        ];
        WhippleMatrices { m, c1, k0, k2 }
    }
}

impl WhippleMatrices {
    /// State matrix `A` of `ẋ = A x + B f` for `x = (φ, δ, φ̇, δ̇)` at speed `v` and gravity `g`.
    pub fn state_matrix(&self, v: f64, g: f64) -> DMat4 {
        let inv = inverse2(self.m);
        let k = add2(scale2(self.k0, g), scale2(self.k2, v * v));
        let a21 = scale2(mul2(inv, k), -1.0);
        let a22 = scale2(mul2(inv, self.c1), -v);
        DMat4::from_cols(
            DVec4::new(0.0, 0.0, a21[0][0], a21[1][0]),
            DVec4::new(0.0, 0.0, a21[0][1], a21[1][1]),
            DVec4::new(1.0, 0.0, a22[0][0], a22[1][0]),
            DVec4::new(0.0, 1.0, a22[0][1], a22[1][1]),
        )
    }

    /// Input matrix `B` (columns: lean torque, steer torque).
    pub fn input_matrix(&self) -> [DVec4; 2] {
        let inv = inverse2(self.m);
        [DVec4::new(0.0, 0.0, inv[0][0], inv[1][0]), DVec4::new(0.0, 0.0, inv[0][1], inv[1][1])]
    }

    /// Eigenvalues at speed `v`, sorted by real part.
    pub fn eigenvalues(&self, v: f64, g: f64) -> [Complex; 4] {
        eigenvalues4(&self.state_matrix(v, g))
    }

    /// The speed range with all eigenvalues' real parts negative, searched up to `v_max`: from
    /// the weave speed to the capsize speed for a benchmark-like bicycle.
    pub fn stable_speeds(&self, g: f64, v_max: f64) -> Option<(f64, f64)> {
        let growth = |v: f64| self.eigenvalues(v, g)[3].re;
        let steps = (v_max / 0.05).ceil() as usize;
        let mut crossings = Vec::new();
        for i in 0..steps {
            let (a, b) = (i as f64 * 0.05, (i + 1) as f64 * 0.05);
            if (growth(a) < 0.0) != (growth(b) < 0.0) {
                crossings.push(bisect(growth, a, b));
            }
        }
        match crossings[..] {
            [low, high, ..] if growth(0.5 * (low + high)) < 0.0 => Some((low, high)),
            [low] if growth(v_max) < 0.0 => Some((low, v_max)),
            _ => None,
        }
    }
}

fn bisect(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let fa = f(a) < 0.0;
    for _ in 0..60 {
        let m = 0.5 * (a + b);
        if (f(m) < 0.0) == fa { a = m } else { b = m }
    }
    0.5 * (a + b)
}

fn inverse2(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]
}

fn mul2(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let e = |i: usize, j: usize| a[i][0] * b[0][j] + a[i][1] * b[1][j];
    [[e(0, 0), e(0, 1)], [e(1, 0), e(1, 1)]]
}

fn add2(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [[a[0][0] + b[0][0], a[0][1] + b[0][1]], [a[1][0] + b[1][0], a[1][1] + b[1][1]]]
}

fn scale2(a: [[f64; 2]; 2], s: f64) -> [[f64; 2]; 2] {
    [[a[0][0] * s, a[0][1] * s], [a[1][0] * s, a[1][1] * s]]
}

/// A complex number.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Natural logarithm (principal branch).
    pub fn ln(self) -> Self {
        Self::new(self.abs().ln(), self.im.atan2(self.re))
    }

    fn mul(self, o: Self) -> Self {
        Self::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }

    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }

    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Complex::sub(self, o)
    }
}

/// Eigenvalues of a 4×4 matrix, sorted by real part (then imaginary part): the roots of its
/// characteristic polynomial (Faddeev–LeVerrier), found by Durand–Kerner and polished by
/// Newton steps.
pub fn eigenvalues4(a: &DMat4) -> [Complex; 4] {
    // Characteristic polynomial s⁴ + c₃ s³ + c₂ s² + c₁ s + c₀.
    let trace = |m: &DMat4| m.x_axis.x + m.y_axis.y + m.z_axis.z + m.w_axis.w;
    let mut coeffs = [0.0; 4];
    let mut mk = DMat4::IDENTITY;
    for k in 1..=4 {
        let am = *a * mk;
        let c = -trace(&am) / k as f64;
        coeffs[4 - k] = c;
        mk = am + DMat4::IDENTITY * c;
    }
    let poly = |z: Complex| {
        let mut p = Complex::new(1.0, 0.0);
        for &k in coeffs.iter().rev() {
            p = p.mul(z);
            p.re += k;
        }
        p
    };
    let deriv = |z: Complex| {
        let d = [coeffs[1], 2.0 * coeffs[2], 3.0 * coeffs[3], 4.0];
        let mut p = Complex::new(0.0, 0.0);
        for &k in d.iter().rev() {
            p = p.mul(z);
            p.re += k;
        }
        p
    };
    let scale = 1.0 + coeffs.iter().map(|c| c.abs()).fold(0.0, f64::max);
    let seed = Complex::new(0.4, 0.9);
    let mut roots = [Complex::new(1.0, 0.0); 4];
    for i in 0..4 {
        let mut z = Complex::new(scale, 0.0);
        for _ in 0..i {
            z = z.mul(seed);
        }
        roots[i] = z;
    }
    for _ in 0..500 {
        let mut change = 0.0f64;
        for i in 0..4 {
            let mut den = Complex::new(1.0, 0.0);
            for j in 0..4 {
                if i != j {
                    den = den.mul(roots[i] - roots[j]);
                }
            }
            let step = poly(roots[i]).div(den);
            roots[i] = roots[i] - step;
            change = change.max(step.abs());
        }
        if change < 1e-14 * scale {
            break;
        }
    }
    for r in &mut roots {
        for _ in 0..3 {
            let d = deriv(*r);
            if d.abs() > 0.0 {
                let step = poly(*r).div(d);
                if step.abs().is_finite() {
                    *r = *r - step;
                }
            }
        }
        // Real roots stay real.
        if r.im.abs() < 1e-9 * (1.0 + r.re.abs()) {
            r.im = 0.0;
        }
    }
    roots.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    roots
}
