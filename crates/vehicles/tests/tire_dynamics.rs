//! Tyre force element on a single vehicle corner (a point mass carrying one wheel): transient
//! slip against its steady state, relaxation in distance, standstill on a slope and free
//! rolling.

use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::{FlatTerrain, PlaneTerrain, Terrain};
use autonomousim_vehicles::ground::tire::{
    FialaParams, MfInput, MfParams, REFERENCE_ROLLING_RESISTANCE, Surface, Tire, TireForces, TireModel, TireState,
    WheelMotion,
};
use glam::DVec3;
use std::path::PathBuf;

const G: f64 = 9.80665;

fn mf(name: &str) -> Tire {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    Tire::magic_formula(MfParams::read(root.join(format!("assets/tires/{name}.tir"))).unwrap(), None).unwrap()
}

fn robot() -> Tire {
    Tire::fiala(
        toml::from_str::<FialaParams>(
            "radius = 0.1\nwidth = 0.05\nvertical_stiffness = 4e4\nvertical_damping = 50\nslip_stiffness = 800\n\
             cornering_stiffness = 600\nmu = 0.9\nnominal_load = 42\nrelaxation_x = 0.05\nrelaxation_y = 0.06\n",
        )
        .unwrap(),
    )
    .unwrap()
}

fn tyres() -> Vec<(&'static str, Tire)> {
    vec![("Sedan", mf("Sedan_Pac02Tire")), ("HMMWV", mf("HMMWV_Pac02Tire")), ("Fiala", robot())]
}

/// A point mass with one wheel whose axis stays fixed, damped normal to the road like a
/// suspension (the tyre's own vertical damping is small).
struct Corner {
    tire: Tire,
    state: TireState,
    mass: f64,
    inertia: f64,
    axis: DVec3,
    pos: DVec3,
    vel: DVec3,
    spin: f64,
    braked: bool,
}

impl Corner {
    fn new(tire: Tire, axis: DVec3) -> Self {
        let mass = tire.nominal_load() / G;
        let r = tire.radius();
        Self {
            state: tire.initial_state(),
            inertia: 0.05 * mass * r * r,
            tire,
            mass,
            axis,
            pos: DVec3::ZERO,
            vel: DVec3::ZERO,
            spin: 0.0,
            braked: false,
        }
    }

    /// Place the wheel centre a loaded radius above `ground` (normal `n`) at the static load.
    fn place(&mut self, ground: DVec3, n: DVec3) {
        let (mut lo, mut hi) = (0.0, 0.5 * self.tire.radius());
        let load = self.mass * G * n.z;
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if self.tire.vertical_force(mid) < load { lo = mid } else { hi = mid }
        }
        self.pos = ground + n * (self.tire.radius() - lo);
    }

    fn step(&mut self, terrain: &dyn Terrain, dt: f64) -> TireForces {
        let motion = WheelMotion {
            center: self.pos,
            axis: self.axis,
            velocity: self.vel,
            carrier_angvel: DVec3::ZERO,
            spin: self.spin,
        };
        let contact = self.tire.contact(terrain, &motion);
        let f = self.tire.step(&mut self.state, contact.as_ref(), &motion, Surface::REFERENCE, dt);
        let mut force = f.force - DVec3::Z * self.mass * G;
        if let Some(c) = contact {
            let omega = (f.fz.max(1.0) / (self.mass * 0.05)).sqrt();
            force -= c.normal * c.normal.dot(self.vel) * self.mass * omega;
        }
        self.vel += force / self.mass * dt;
        self.pos += self.vel * dt;
        self.spin = if self.braked { 0.0 } else { self.spin + f.torque.dot(self.axis) / self.inertia * dt };
        f
    }
}

#[test]
fn transient_slip_settles_on_the_steady_state() {
    let flat = FlatTerrain::new(0.0, MaterialId::ASPHALT);
    for (name, tire) in tyres() {
        let r = tire.radius();
        let mut state = tire.initial_state();
        let (vx, vy) = (15.0, -0.6);
        let mut center = DVec3::new(0.0, 0.0, r - 0.02 * r);
        let mut f = TireForces::default();
        for _ in 0..4000 {
            let motion = WheelMotion {
                center,
                axis: DVec3::Y,
                velocity: DVec3::new(vx, vy, 0.0),
                carrier_angvel: DVec3::ZERO,
                spin: 1.06 * vx / r,
            };
            let contact = tire.contact(&flat, &motion);
            f = tire.step(&mut state, contact.as_ref(), &motion, Surface::REFERENCE, 5e-4);
            center += motion.velocity * 5e-4;
        }
        let (kappa, tan_alpha) = (-f.vsx / f.vx.abs(), f.vy / f.vx.abs());
        assert!((f.kappa - kappa).abs() < 1e-9 && (f.tan_alpha - tan_alpha).abs() < 1e-9, "{name}: {f:?}");
        assert!(kappa > 0.01 && tan_alpha < -0.03 && f.fx > 0.0 && f.fy > 0.0, "{name}: {f:?}");
        let (fx, fy) = match &tire.model {
            TireModel::MagicFormula(p) => {
                let mut input = MfInput::new(f.fz, kappa, tan_alpha.atan(), 0.0, vx);
                input.cos_alpha = 1.0 / (1.0 + tan_alpha * tan_alpha).sqrt();
                let o = p.eval(&input);
                (o.fx, o.fy)
            }
            TireModel::Fiala(p) => {
                let o = p.eval(f.fz, kappa, tan_alpha, p.half_length(f.deflection), 1.0);
                (o.fx, o.fy)
            }
        };
        assert!((f.fx - fx).abs() < 1e-9 * f.fz && (f.fy - fy).abs() < 1e-9 * f.fz, "{name}");
    }
}

#[test]
fn relaxation_is_first_order_in_travelled_distance() {
    let flat = FlatTerrain::new(0.0, MaterialId::ASPHALT);
    for (name, tire) in tyres() {
        let r = tire.radius();
        let sigma = tire.initial_state().sigma[1];
        for vx in [3.0, 25.0] {
            let mut state = tire.initial_state();
            let dt = 1e-2 * sigma / vx;
            let mut center = DVec3::new(0.0, 0.0, r - 0.02 * r);
            // Step to tan α = 0.02 at t = 0; after travelling σ the slip reaches 1 − 1/e.
            let mut travelled = 0.0;
            let mut f = TireForces::default();
            while travelled < sigma {
                let motion = WheelMotion {
                    center,
                    axis: DVec3::Y,
                    velocity: DVec3::new(vx, 0.02 * vx, 0.0),
                    carrier_angvel: DVec3::ZERO,
                    spin: vx / r,
                };
                let contact = tire.contact(&flat, &motion);
                f = tire.step(&mut state, contact.as_ref(), &motion, Surface::REFERENCE, dt);
                center += motion.velocity * dt;
                travelled += vx * dt;
            }
            let fraction = f.tan_alpha / 0.02;
            println!("{name} at {vx} m/s: σ_α {:.3} m, fraction {fraction:.4}, Fy {:.0} N", state.sigma[1], f.fy);
            // σ follows the load, which differs slightly from nominal here.
            let expected = 1.0 - (-travelled / state.sigma[1]).exp();
            assert!((fraction - expected).abs() < 0.01, "{name} at {vx} m/s: {fraction} vs {expected}");
        }
    }
}

#[test]
fn braked_wheel_holds_on_a_20_degree_slope_without_creep() {
    let angle = 20f64.to_radians();
    let slope = PlaneTerrain::incline(angle, MaterialId::ASPHALT);
    let n = slope.normal;
    // Facing uphill (slope along the contact x axis) and across it (along y).
    for (heading, axis) in [("uphill", DVec3::Y), ("across", DVec3::X)] {
        for (name, tire) in tyres() {
            let mut c = Corner::new(tire, axis);
            c.braked = true;
            c.place(DVec3::ZERO, n);
            let dt = 1e-3;
            let mut settled = DVec3::ZERO;
            let mut peak_late = 0.0f64;
            for k in 0..10_000 {
                c.step(&slope, dt);
                if k == 2_000 {
                    settled = c.pos;
                }
                if k > 2_000 {
                    peak_late = peak_late.max(c.vel.length());
                }
            }
            let creep = (c.pos - settled).length();
            let f = c.step(&slope, dt);
            println!(
                "{name} {heading}: creep {creep:.2e} m, late speed {peak_late:.2e} m/s, Fz {:.0} N, Fx {:.0} N, Fy {:.0} N",
                f.fz, f.fx, f.fy
            );
            assert!(creep < 1e-3, "{name} {heading}: crept {creep} m in 8 s");
            assert!(peak_late < 1e-3, "{name} {heading}: still moving at {peak_late} m/s");
            let slide = c.pos.dot(n.cross(axis).normalize());
            assert!(slide.abs() < 0.05, "{name} {heading}: slid {slide} m");
        }
    }
}

#[test]
fn free_rolling_decays_to_rolling_resistance() {
    let flat = FlatTerrain::new(0.0, MaterialId::ASPHALT);
    for (name, tire) in tyres() {
        let mut c = Corner::new(tire, DVec3::Y);
        c.place(DVec3::ZERO, DVec3::Z);
        let r0 = c.tire.radius();
        c.vel = DVec3::new(10.0, 0.0, 0.0);
        c.spin = 10.0 / (r0 - 0.01 * r0);
        let dt = 1e-3;
        let mut f = TireForces::default();
        let mut v1 = 0.0;
        for k in 0..3_000 {
            f = c.step(&flat, dt);
            if k == 999 {
                v1 = c.vel.x;
            }
        }
        let decel = (c.vel.x - v1) / 2.0;
        // Steady rolling: m a = F_x, I a / R_e = −R_l F_x + M_y.
        let coefficient = match &c.tire.model {
            TireModel::MagicFormula(p) if p.qsy1 != 0.0 => p.qsy1,
            _ => REFERENCE_ROLLING_RESISTANCE,
        };
        let my = -coefficient * f.fz * r0;
        let rl = r0 - f.deflection;
        let re = (c.vel.x / c.spin).abs();
        let expected = my / (c.inertia / re + rl * c.mass);
        println!(
            "{name}: deceleration {decel:.4} m/s² (expected {expected:.4}), Fz {:.0} N, M_y {:.2} N m",
            f.fz, f.my
        );
        assert!(decel < 0.0 && (decel / expected - 1.0).abs() < 0.03, "{name}: {decel} vs {expected} m/s²");
        assert!((f.my - my).abs() < 1e-3 * my.abs(), "{name}: M_y {} vs {my}", f.my);
    }
}

#[test]
fn surface_scales_follow_the_material_table() {
    let table = autonomousim_core::material::MaterialTable::standard();
    assert_eq!(Surface::of(table.get(MaterialId::ASPHALT)), Surface::REFERENCE);
    let mud = Surface::of(table.get(MaterialId::MUD));
    assert!(mud.mu_scale < 0.5 && mud.rolling_scale > 5.0);
}
