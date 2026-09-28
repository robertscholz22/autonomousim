//! Tiltrotor family: the preset, hover trim against momentum theory, cruise trim against a
//! component build-up of the same surfaces, the conversion corridor, energy bookkeeping,
//! control signs and ground effect.

use autonomousim_vehicles::aero::GroundPlane;
use autonomousim_vehicles::multirotor::{AirData, StepEnv};
use autonomousim_vehicles::presets;
use autonomousim_vehicles::tiltrotor::*;
use autonomousim_vehicles::{VehicleDef, VehicleError};
use glam::DVec3;
use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

const DT: f64 = 0.002;
const G: f64 = 9.80665;
const RHO: f64 = 1.225;
const CRUISE: f64 = 20.0;
// The preset's forward tilt limit (π/2 as written in the TOML).
#[allow(clippy::approx_constant)]
const FORWARD: f64 = 1.5708;

fn def() -> TiltrotorDef {
    presets::tiltrotor("quadtilt_like").unwrap()
}

fn aircraft() -> Tiltrotor {
    Tiltrotor::new(Arc::new(def()), DT)
}

fn free_air() -> StepEnv<'static> {
    StepEnv { scene: None, gravity: DVec3::new(0.0, 0.0, -G), air: AirData::default(), ground: None }
}

fn start(t: &mut Tiltrotor, trim: &TiltrotorTrim) {
    t.reset(&trim.init(DVec3::new(0.0, 0.0, 300.0), 0.0));
}

#[test]
fn preset_loads_and_validates() {
    let d = def();
    assert_eq!(d.rotors.len(), 4);
    assert!((d.span() - 2.2).abs() < 1e-12);
    assert!(d.gear_height().unwrap() > 0.2);
    // Flaps-up stall near 11.5 m/s (the preset's description).
    let vs = d.stall_speed(RHO, G);
    assert!((10.5..12.5).contains(&vs), "stall speed {vs}");
    let back: TiltrotorDef = toml::from_str(&toml::to_string(&d).unwrap()).unwrap();
    assert_eq!(back, d);
    assert!(matches!(presets::get("quadtilt_like").unwrap(), VehicleDef::Tiltrotor(_)));
    assert!(presets::tiltrotor("iris_like").is_err());

    let invalid = |d: &TiltrotorDef| matches!(d.validate(), Err(VehicleError::Invalid(_)));
    let mut bad = def();
    bad.controls.mixing[0].surface = "canard".into();
    assert!(invalid(&bad));
    let mut bad = def();
    bad.rotors.push(bad.rotors[0].clone());
    assert!(invalid(&bad));
    let mut bad = def();
    bad.motor.voltage = None;
    assert!(invalid(&bad));
    let mut bad = def();
    bad.rotors[1].sense = 0.5;
    assert!(invalid(&bad));
}

/// Hover: the rotors carry the weight at the speed momentum theory and the propeller's static
/// coefficients give, with a figure of merit `C_T^{3/2}/(√(π/2)·C_P)`; the model holds still
/// at the trim.
#[test]
fn hover_trim_against_momentum_theory() {
    let mut t = aircraft();
    let d = def();
    let trim = t.trim(0.0, 0.0, RHO, G).unwrap();
    let weight = t.mass() * G;
    assert!(trim.pitch.abs() < 1e-9 && trim.controls.elevator.abs() < 1e-9, "{trim:?}");
    assert!((trim.total_thrust() / weight - 1.0).abs() < 1e-9);
    let area = PI * d.propeller.diameter.powi(2) / 4.0;
    let ideal: f64 = trim.thrust.iter().map(|&th| th * (th / (2.0 * RHO * area)).sqrt()).sum();
    let fm = ideal / trim.shaft_power;
    let (ct, cq) = (0.09357f64, 0.005230f64);
    let expected = ct.powf(1.5) / ((PI / 2.0).sqrt() * 2.0 * PI * cq);
    assert!((fm / expected - 1.0).abs() < 1e-6, "figure of merit {fm} vs {expected}");
    assert!((0.6..0.8).contains(&fm));
    // Motor efficiency and throttle margin of a sensible design.
    let eta = trim.shaft_power / trim.electric_power;
    assert!((0.8..0.92).contains(&eta), "motor efficiency {eta}");
    assert!(trim.controls.throttle.iter().all(|th| (0.5..0.7).contains(th)), "{:?}", trim.controls.throttle);

    start(&mut t, &trim);
    let env = free_air();
    for _ in 0..3 {
        t.step(&trim.controls, &env).unwrap();
    }
    assert!((t.specific_force_body() - DVec3::Z * G).length() < 1e-6 * G, "{}", t.specific_force_body());
    assert!(t.ang_acc_body().length() < 1e-6, "{}", t.ang_acc_body());
    for (w, w0) in t.rotor_speeds().iter().zip(trim.rotor_speed) {
        assert!((w - w0).abs() < 1e-6 * w0, "{w} vs {w0}");
    }
    assert!((t.electric_power() - trim.electric_power).abs() < 1e-3 * trim.electric_power);
}

/// Thin-aerofoil flap effectiveness and moment slope for a chord fraction.
fn flap(cf: f64) -> (f64, f64) {
    let t = (2.0 * cf - 1.0).acos();
    (1.0 - (t - t.sin()) / PI, -0.5 * t.sin() * (1.0 - t.cos()))
}

/// Level cruise with the rotors forward from the same surfaces built up by hand: Helmbold lift
/// slopes, parabolic polars, thin-aerofoil flaps, fuselage drag, the thrust on its line and the
/// rotors' normal force `K_d·ω·V·sin α` at rotor speeds `omega`. Returns pitch, total thrust
/// and elevator input.
fn component_buildup(d: &TiltrotorDef, v: f64, omega: &[f64]) -> [f64; 3] {
    let q = 0.5 * RHO * v * v;
    let weight = d.body.mass * G;
    let wrench = |x: [f64; 3]| -> [f64; 3] {
        let [alpha, thrust, elevator] = x;
        let (sa, ca) = alpha.sin_cos();
        let (lift_dir, drag_dir) = (DVec3::new(sa, 0.0, ca), DVec3::new(-ca, 0.0, sa));
        let (mut f, mut m) = (DVec3::ZERO, DVec3::ZERO);
        for s in &d.surfaces {
            let ar = s.aspect_ratio();
            let a = 2.0 * PI * ar / (2.0 + (ar * ar + 4.0).sqrt());
            if s.roll != 0.0 {
                // The fin sees the flow along its chord: profile drag only.
                let fd = drag_dir * (0.5 * RHO * (v * ca).powi(2) * s.area * s.cd0);
                f += fd;
                m += s.position.cross(DVec3::new(fd.x, 0.0, 0.0));
                continue;
            }
            let delta = if s.name == "tailplane" { -elevator * d.controls.elevator.max } else { 0.0 };
            let (tau, cm_flap) = s.flap.as_ref().map_or((0.0, 0.0), |fl| flap(fl.chord_fraction));
            let cl = s.cl0 + a * (alpha + s.incidence + tau * delta);
            let cd = s.cd0 + cl * cl / (PI * s.oswald * ar);
            let cm = s.cm0 + cm_flap * delta * a / (2.0 * PI);
            let fs = (lift_dir * cl + drag_dir * cd) * (q * s.area);
            f += fs;
            m += s.position.cross(fs) + DVec3::new(0.0, -q * s.area * s.chord * cm, 0.0);
        }
        let vb = DVec3::new(v * ca, 0.0, -v * sa);
        f += -0.5 * RHO * v * (d.fuselage.drag_area * vb);
        for (r, w) in d.rotors.iter().zip(omega) {
            let hub = r.pivot + DVec3::X * r.offset;
            let ft = DVec3::new(thrust / d.rotors.len() as f64, 0.0, d.rotor_drag * w * v * sa);
            f += ft;
            m += hub.cross(ft);
        }
        let f = f - DVec3::new(weight * sa, 0.0, weight * ca);
        [f.x / weight, f.z / weight, m.y / weight]
    };
    // Newton with a forward-difference Jacobian (3 × 3, Cramer's rule).
    let mut x = [0.0, 5.0, 0.0];
    for _ in 0..50 {
        let r = wrench(x);
        let mut j = [[0.0; 3]; 3];
        for c in 0..3 {
            let mut xp = x;
            xp[c] += 1e-7;
            let rp = wrench(xp);
            for i in 0..3 {
                j[i][c] = (rp[i] - r[i]) / 1e-7;
            }
        }
        let det = |a: &[[f64; 3]; 3]| {
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        };
        let d0 = det(&j);
        for c in 0..3 {
            let mut jc = j;
            for i in 0..3 {
                jc[i][c] = r[i];
            }
            x[c] -= det(&jc) / d0;
        }
    }
    x
}

#[test]
fn cruise_trim_against_component_buildup() {
    let d = def();
    let mut t = aircraft();
    for v in [15.0, CRUISE, 25.0] {
        let trim = t.trim(v, FORWARD, RHO, G).unwrap();
        let [alpha, thrust, elevator] = component_buildup(&d, v, &trim.rotor_speed);
        assert!((trim.pitch - alpha).abs() < 0.02f64.to_radians(), "{v}: pitch {} vs {alpha}", trim.pitch);
        assert!((trim.total_thrust() / thrust - 1.0).abs() < 0.005, "{v}: thrust {} vs {thrust}", trim.total_thrust());
        assert!(
            (trim.controls.elevator - elevator).abs() < 0.01,
            "{v}: elevator {} vs {elevator}",
            trim.controls.elevator
        );
        // Rotors forward: they turn alike, and cruise takes far less power than hover.
        let w = trim.rotor_speed;
        assert!((w[0] - w[2]).abs() < 0.02 * w[0], "{w:?}");
        assert!(trim.feasible(&d, &TrimLimits::default()), "{v}: {trim:?}");
    }
    let hover = t.trim(0.0, 0.0, RHO, G).unwrap();
    let cruise = t.trim(CRUISE, FORWARD, RHO, G).unwrap();
    assert!(
        cruise.electric_power < 0.5 * hover.electric_power,
        "{} vs {}",
        cruise.electric_power,
        hover.electric_power
    );

    // The model holds the cruise trim.
    start(&mut t, &cruise);
    let env = free_air();
    for _ in 0..3 {
        t.step(&cruise.controls, &env).unwrap();
    }
    let (s, c) = cruise.pitch.sin_cos();
    assert!((t.specific_force_body() - DVec3::new(s, 0.0, c) * G).length() < 1e-6 * G);
    assert!(t.ang_acc_body().length() < 1e-5, "{}", t.ang_acc_body());
}

/// Feasible tilt against airspeed: near vertical in hover, all the way forward once the wing
/// carries the weight, and no longer vertical at speed (the rotors run out of throttle holding
/// the nose down).
#[test]
fn conversion_corridor() {
    let d = def();
    let t = aircraft();
    let speeds: Vec<f64> = (0..=15).map(|i| 2.0 * i as f64).collect();
    let corridor = t.corridor(&speeds, 2f64.to_radians(), &TrimLimits::default(), RHO, G);
    for c in &corridor {
        assert!(c.tilt.is_some() && c.contiguous, "{c:?}");
    }
    let range = |v: f64| corridor.iter().find(|c| c.speed == v).unwrap().tilt.unwrap();
    let (lo, hi) = range(0.0);
    assert!(lo <= 0.0 && hi > 0.0 && hi < 20f64.to_radians(), "hover {lo} {hi}");
    let vs = d.stall_speed(RHO, G);
    for c in &corridor {
        let (lo, hi) = c.tilt.unwrap();
        // Rotors fully forward only above the stall speed.
        assert_eq!(hi >= FORWARD - 1e-9, c.speed > vs, "{c:?}");
        // Rotors vertical only at low speed.
        assert_eq!(lo <= 0.0, c.speed < 20.0, "{c:?}");
    }
    // The corridor opens as speed builds and its floor rises at high speed.
    for w in corridor.windows(2) {
        let ((lo0, hi0), (lo1, hi1)) = (w[0].tilt.unwrap(), w[1].tilt.unwrap());
        if w[1].speed >= 4.0 && w[1].speed <= 12.0 {
            assert!(hi1 >= hi0, "{w:?}");
        }
        if w[1].speed >= 16.0 {
            assert!(lo1 > lo0, "{w:?}");
        }
    }
    assert!(range(30.0).0 > 5f64.to_radians());
}

/// Energy: the change of kinetic, rotor and potential energy equals the work of the external
/// forces and the net rotor torques, with throttle and elevator moving and the mounts fixed.
#[test]
fn energy_balance() {
    let mut t = aircraft();
    let trim = t.trim(CRUISE, FORWARD, RHO, G).unwrap();
    start(&mut t, &trim);
    let env = free_air();
    let e0 = t.energy(G);
    let (mut work, mut scale) = (0.0, 0.0);
    for i in 0..(10.0 / DT) as usize {
        let s = (2.0 * PI * i as f64 * DT / 3.0).sin();
        let mut u = trim.controls;
        for th in &mut u.throttle {
            *th += 0.15 * s;
        }
        u.elevator += 0.1 * s;
        t.step(&u, &env).unwrap();
        work += t.external_power() * DT;
        scale += t.external_power().abs() * DT;
    }
    let change = t.energy(G) - e0;
    assert!(change.abs() > 10.0, "{change}");
    assert!((change - work).abs() < 0.01 * scale, "energy changed {change} J, work {work} J (scale {scale})");
}

/// Response over `time` to a change of the trim controls, relative to the trim alone: angular
/// velocity (body frame).
fn response(trim: &TiltrotorTrim, time: f64, change: impl Fn(&mut TiltrotorInput)) -> DVec3 {
    let run = |u: &TiltrotorInput| {
        let mut t = aircraft();
        start(&mut t, trim);
        for _ in 0..(time / DT) as usize {
            t.step(u, &free_air()).unwrap();
        }
        t.ang_vel_body()
    };
    let mut u = trim.controls;
    change(&mut u);
    run(&u) - run(&trim.controls)
}

#[test]
fn control_signs() {
    let t = aircraft();
    // Hover: differential throttle and tilt; roll right is +x, nose up −y, nose right −z.
    let hover = t.trim(0.0, 0.0, RHO, G).unwrap();
    let w = response(&hover, 0.3, |u| {
        u.throttle[0] += 0.05;
        u.throttle[1] += 0.05;
    });
    assert!(w.y < -0.05 && w.x.abs() < 1e-6, "front throttle {w}");
    let w = response(&hover, 0.3, |u| {
        u.throttle[0] += 0.05;
        u.throttle[2] += 0.05;
    });
    assert!(w.x > 0.05, "left throttle {w}");
    let w = response(&hover, 0.5, |u| {
        u.tilt[0] += 0.1;
        u.tilt[2] += 0.1;
        u.tilt[1] -= 0.1;
        u.tilt[3] -= 0.1;
    });
    assert!(w.z < -0.05, "differential tilt {w}");
    // Clockwise rotors (seen from above: front left, rear right) speeding up turn the airframe
    // counter-clockwise.
    let w = response(&hover, 0.5, |u| {
        u.throttle[0] += 0.05;
        u.throttle[3] += 0.05;
        u.throttle[1] -= 0.05;
        u.throttle[2] -= 0.05;
    });
    assert!(w.z > 0.01, "yaw torque {w}");

    // Cruise: surfaces.
    let cruise = t.trim(CRUISE, FORWARD, RHO, G).unwrap();
    let w = response(&cruise, 0.2, |u| u.aileron += 0.3);
    assert!(w.x > 0.1, "aileron {w}");
    let w = response(&cruise, 0.2, |u| u.elevator += 0.3);
    assert!(w.y < -0.05, "elevator {w}");
    let w = response(&cruise, 0.2, |u| u.rudder += 0.3);
    assert!(w.z < -0.02, "rudder {w}");
}

/// Tilting the mounts moves the rotors with their servo, and the tilt is clamped to the range.
#[test]
fn tilt_servo_and_range() {
    let d = def();
    let mut t = aircraft();
    let hover = t.trim(0.0, 0.0, RHO, G).unwrap();
    start(&mut t, &hover);
    let mut u = hover.controls;
    u.tilt = [3.0; MAX_ROTORS];
    t.step(&u, &free_air()).unwrap();
    // Rate-limited.
    assert!(t.tilts().iter().all(|&x| (x - d.controls.tilt.rate * DT).abs() < 1e-12), "{:?}", t.tilts());
    for _ in 0..(3.0 / DT) as usize {
        t.step(&u, &free_air()).unwrap();
    }
    assert!(t.tilts().iter().all(|&x| (x - d.controls.tilt.max).abs() < 1e-9), "{:?}", t.tilts());
    assert!(d.controls.tilt.max >= FRAC_PI_2 - 1e-4);
}

/// Near the ground the rotors gain the Cheeseman–Bennett thrust ratio.
#[test]
fn ground_effect() {
    let d = def();
    let mut t = aircraft();
    let hover = t.trim(0.0, 0.0, RHO, G).unwrap();
    start(&mut t, &hover);
    let r = d.rotor_radius();
    let hub = t.position() + d.rotors[0].hub(0.0);
    let ground = GroundPlane { point: hub - DVec3::Z * r, normal: DVec3::Z };
    t.begin_step();
    t.apply_controls(&hover.controls, &AirData::default(), Some(&ground));
    let ratio = t.loads().thrust[0] / hover.thrust[0];
    assert!((ratio - 16.0 / 15.0).abs() < 1e-9, "{ratio}");
}
