//! Truck and farm presets: they load, their static loads and steering locks match Chrono's
//! (fixtures/chrono/truck_*.json, tools/gen_chrono_truck_fixtures.py), they settle at their
//! static state, drive and brake straight, and the truck features behave: dual tyres,
//! degressive dampers, geometric multi-axle steering, forced trailer steering and air-brake
//! lag.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::presets;
use glam::DVec3;
use serde_json::Value;
use std::sync::Arc;

const DT: f64 = 1e-3;
const G: f64 = STANDARD_GRAVITY;

fn fixture(name: &str) -> Value {
    let path = format!("{}/../../fixtures/chrono/truck_{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn rig(vehicle: &str, trailer: Option<&str>) -> WheeledDef {
    let d = presets::wheeled(vehicle).unwrap();
    match trailer {
        Some(t) => d.with_trailers(&[presets::trailer(t).unwrap()]).unwrap(),
        None => d,
    }
}

const RIGS: [(&str, Option<&str>); 5] = [
    ("truck_6x4", None),
    ("truck_6x4", Some("semitrailer_3axle")),
    ("truck_8x8", None),
    ("farm_tractor", None),
    ("farm_tractor", Some("farm_trailer")),
];

struct Flat {
    terrain: FlatTerrain,
    materials: MaterialTable,
}

impl Flat {
    fn new() -> Self {
        Self { terrain: FlatTerrain::new(0.0, MaterialId::ASPHALT), materials: MaterialTable::standard() }
    }

    fn step(&self, v: &mut Wheeled, input: &DriveInput) {
        let env = GroundStepEnv {
            scene: StaticScene { terrain: &self.terrain, obstacles: &NoObstacles, materials: &self.materials },
            gravity: DVec3::new(0.0, 0.0, -G),
            air: AirData::default(),
        };
        v.step(input, &env).unwrap();
    }

    fn run(&self, v: &mut Wheeled, input: &DriveInput, seconds: f64) {
        for _ in 0..(seconds / DT).round() as usize {
            self.step(v, input);
        }
    }
}

fn at_rest(d: WheeledDef, speed: f64) -> Wheeled {
    let mut v = Wheeled::new(Arc::new(d), DT);
    let init = v.rest(DVec3::ZERO, 0.0, speed);
    v.reset(&init);
    v
}

#[test]
fn presets_load() {
    for name in ["truck_6x4", "truck_8x8", "farm_tractor"] {
        let d = presets::wheeled(name).unwrap();
        assert_eq!(d.name, name);
        assert!(!d.source.is_empty());
    }
    assert_eq!(presets::trailer_names().collect::<Vec<_>>(), ["semitrailer_3axle", "farm_trailer"]);
    for (vehicle, trailer) in RIGS {
        let d = rig(vehicle, trailer);
        let v = Wheeled::new(Arc::new(d.clone()), DT);
        let expected = match (vehicle, trailer) {
            ("truck_6x4", None) => 6,
            ("truck_6x4", _) => 12,
            ("truck_8x8", _) => 8,
            ("farm_tractor", None) => 4,
            _ => 8,
        };
        assert_eq!(v.num_wheels(), expected, "{vehicle} {trailer:?}");
    }
    // The trucks' masses are Chrono's (the trailer's mass there counts the chassis only).
    for (d, fx) in [(rig("truck_6x4", None), "kraz_tractor"), (rig("truck_8x8", None), "man_10t")] {
        let chrono = nums(&fixture(fx)["design"]["mass"])[0];
        assert!((d.total_mass() / chrono - 1.0).abs() < 1e-3, "{fx}: {} vs {chrono}", d.total_mass());
    }
    // Dual rear wheels on the tractor, singles on the semitrailer.
    let d = rig("truck_6x4", Some("semitrailer_3axle"));
    assert_eq!(d.axles.iter().map(|a| a.dual.is_some()).collect::<Vec<_>>(), [false, true, true, false, false, false]);
}

/// Per-wheel static loads and the units' pitch against Chrono, for the tractor alone and with
/// its semitrailer (each vehicle's springs carry it at design height, as Chrono's do).
#[test]
fn static_loads_match_chrono() {
    for (d, fx) in [
        (rig("truck_6x4", None), "kraz_tractor"),
        (rig("truck_6x4", Some("semitrailer_3axle")), "kraz_rig"),
        (rig("truck_8x8", None), "man_10t"),
    ] {
        let st = d.rest_state().expect("a static state");
        let chrono = fixture(fx);
        let units = chrono["static"].as_array().unwrap();
        let mut w = 0;
        for (u, unit) in units.iter().enumerate() {
            for load in nums(&unit["loads"]) {
                assert_eq!(d.wheel_unit(w), u);
                assert!((st.loads[w] / load - 1.0).abs() < 0.01, "{fx} wheel {w}: {} vs {load}", st.loads[w]);
                w += 1;
            }
        }
        assert_eq!(w, d.num_wheels());
        let pitch = unit_pitch(st, 0);
        let expected = units[0]["pitch"].as_f64().unwrap();
        assert!((pitch - expected).abs() < 1e-3, "{fx}: pitch {pitch} vs {expected}");
        if units.len() > 1 {
            let pitch = unit_pitch(st, 1);
            let expected = units[1]["pitch"].as_f64().unwrap();
            assert!((pitch - expected).abs() < 2e-3, "{fx}: trailer pitch {pitch} vs {expected}");
        }
    }
}

/// A unit's pitch in the world (the towing unit's plus the joints' down the chain).
fn unit_pitch(st: &StaticState, u: usize) -> f64 {
    st.pitch + st.joints[..u].iter().map(|j| j[0]).sum::<f64>()
}

/// Full steering, standing: the road-wheel angles of the steered axles against Chrono's. The
/// 8×8's second axle steers by the geometric (Ackermann) share of the first, where Chrono's
/// linkage steers it by its own ratio (larger, not a common turn centre).
#[test]
fn steering_locks_match_chrono() {
    let flat = Flat::new();
    for (name, fx) in [("truck_6x4", "kraz_tractor"), ("truck_8x8", "man_10t")] {
        let mut v = at_rest(rig(name, None), 0.0);
        flat.run(&mut v, &DriveInput { steering: 1.0, ..Default::default() }, 4.0);
        let angles = nums(&fixture(fx)["lock"][0]);
        for side in 0..2 {
            let (ours, chrono) = (v.wheel(side).steer.to_degrees(), angles[side].to_degrees());
            assert!((ours - chrono).abs() < 0.4, "{name} side {side}: {ours:.2}° vs {chrono:.2}°");
        }
    }
    // Geometric steering: the second axle's bicycle angle points at the first axle's turn
    // centre on the unsteered axles' mean line, and its wheels take their Ackermann angles
    // about it.
    let d = rig("truck_8x8", None);
    assert!(d.axles[1].is_geometric() && d.axles[2].share() == 0.0);
    let (x0, x1, xr) = (d.axles[0].position.x, d.axles[1].position.x, d.steer_reference());
    let ratio = (x1 - xr) / (x0 - xr);
    assert!((d.axles[1].share() / d.axles[0].share() - ratio).abs() < 1e-12);
    let ackermann = d.steering.as_ref().unwrap().ackermann;
    let mut v = at_rest(d.clone(), 0.0);
    for steering in [0.3, -1.0] {
        flat.run(&mut v, &DriveInput { steering, ..Default::default() }, 8.0);
        let lead = d.axles[0].share() * v.steering_angle();
        let second = (ratio * lead.tan()).atan();
        for side in 0..2 {
            let lateral = d.wheel_position(side).y;
            let expected =
                [wheel_angle(lead, x0 - xr, lateral, ackermann), wheel_angle(second, x1 - xr, lateral, ackermann)];
            for (axle, e) in expected.into_iter().enumerate() {
                let got = v.wheel(2 * axle + side).steer;
                assert!((got - e).abs() < 1e-6, "steering {steering}, axle {axle} side {side}: {got} vs {e}");
            }
        }
    }
}

/// Released at the static state, every rig stays there.
#[test]
fn rigs_settle_at_the_static_state() {
    let flat = Flat::new();
    for (vehicle, trailer) in RIGS {
        let d = rig(vehicle, trailer);
        let st = d.rest_state().unwrap().clone();
        let mut v = at_rest(d, 0.0);
        flat.run(&mut v, &DriveInput { brake: 1.0, parking: true, ..Default::default() }, 3.0);
        let name = format!("{vehicle} {trailer:?}");
        assert!(v.lin_vel_world().length() < 1e-3, "{name}: still moving");
        assert!(v.contacts().is_empty(), "{name}: body on the ground");
        assert!((v.position().z - st.height).abs() < 1e-3, "{name}: height {} vs {}", v.position().z, st.height);
        for (k, wh) in v.wheels().enumerate() {
            assert!(
                (wh.tire.fz / st.loads[k] - 1.0).abs() < 0.01,
                "{name} wheel {k}: {} vs {}",
                wh.tire.fz,
                st.loads[k]
            );
        }
    }
}

/// Under throttle every rig speeds up in a straight line, and it brakes to a stop in line.
#[test]
fn rigs_drive_and_brake_straight() {
    let flat = Flat::new();
    for (vehicle, trailer) in RIGS {
        let name = format!("{vehicle} {trailer:?}");
        let d = rig(vehicle, trailer);
        let units = d.num_units();
        let mut v = at_rest(d, 5.0);
        flat.run(&mut v, &DriveInput { throttle: 0.6, ..Default::default() }, 10.0);
        assert!(v.speed() > 7.0, "{name}: speed {}", v.speed());
        // The truck tyre's lateral offsets (conicity, ply steer) pull the rigs gently aside.
        let straight = |v: &Wheeled, distance: f64| {
            let tol = 0.01 * distance;
            let last = v.unit_pose(units - 1);
            assert!(v.position().y.abs() < tol && last.pos.y.abs() < tol, "{name}: {} / {}", v.position(), last.pos);
            let heading = (v.orientation() * DVec3::X).truncate();
            assert!(heading.y.abs() < 0.01, "{name}: heading {heading}");
            for u in 1..units {
                assert!(v.articulation(u).0.abs() < 2e-3, "{name} unit {u}: {:?}", v.articulation(u));
            }
        };
        straight(&v, v.position().x);
        let x = v.position().x;
        flat.run(&mut v, &DriveInput { brake: 1.0, ..Default::default() }, 8.0);
        assert!(v.speed() < 0.05, "{name}: still at {}", v.speed());
        assert!(v.position().x - x > 5.0, "{name}: stopped in {} m", v.position().x - x);
        straight(&v, v.position().x);
    }
}

/// Air brakes: the torque follows the pedal after a dead time, then with a first-order lag.
#[test]
fn air_brakes_lag_the_pedal() {
    let flat = Flat::new();
    let d = rig("truck_6x4", None);
    let (max, delay, tau) = (d.axles[0].brake.max_torque, d.axles[0].brake.delay, d.axles[0].brake.time_constant);
    assert!(delay > 0.0 && tau > 0.0);
    let mut v = at_rest(d, 15.0);
    let pedal = 0.4;
    let input = DriveInput { brake: pedal, ..Default::default() };
    let mut torque = vec![];
    for _ in 0..(1.5 / DT) as usize {
        flat.step(&mut v, &input);
        torque.push(v.wheel(0).brake_torque.abs());
    }
    let at = |t: f64| torque[(t / DT).round() as usize - 1];
    assert!(at(delay - 0.01) < 1e-6 * max, "brakes before the delay: {}", at(delay - 0.01));
    for k in [1.0f64, 2.0, 4.0] {
        let expected = pedal * max * (1.0 - (-k).exp());
        let got = at(delay + k * tau);
        assert!((got / expected - 1.0).abs() < 0.02, "after {k} τ: {got} vs {expected}");
    }
}

/// A semitrailer axle steered by articulation turns by the gain times the articulation angle,
/// against the turn, and pulls the trailer's rear to the inside of the unsteered rig's path.
#[test]
fn articulation_steers_trailer_axles() {
    let flat = Flat::new();
    let gain = 0.5;
    let mut steered = presets::trailer("semitrailer_3axle").unwrap();
    steered.axles[2].steer_mode = SteerMode::Articulation(gain);
    let mut radius = vec![];
    for trailer in [presets::trailer("semitrailer_3axle").unwrap(), steered] {
        let d = presets::wheeled("truck_6x4").unwrap().with_trailers(&[trailer]).unwrap();
        let mut v = at_rest(d, 5.0);
        let input = DriveInput { steering: 0.5, throttle: 0.15, ..Default::default() };
        flat.run(&mut v, &input, 20.0);
        let (art, _) = v.articulation(1);
        assert!(art < -0.1, "articulation {art}");
        let rear = v.wheel(10).steer;
        if v.def().axles[5].is_steered() {
            assert!((rear - gain * art).abs() < 0.01, "rear axle {rear} vs {}", gain * art);
            assert!(v.wheel(6).steer == 0.0);
        } else {
            assert_eq!(rear, 0.0);
        }
        // The turn centre from three points of the steady front axle path, and the radii of the
        // front axle's and the trailer's rear axle's paths about it.
        let axle = |v: &Wheeled, w: usize| ((v.wheel_pose(w).pos + v.wheel_pose(w + 1).pos) / 2.0).truncate();
        let mut path = vec![axle(&v, 0)];
        for _ in 0..2 {
            flat.run(&mut v, &input, 1.5);
            path.push(axle(&v, 0));
        }
        let centre = circumcentre(path[0], path[1], path[2]);
        radius.push(((axle(&v, 0) - centre).length(), (axle(&v, 10) - centre).length()));
    }
    // Forced steering: the trailer's rear runs further out, closer to the tractor's path
    // (whose radius changes a little too, as the steered trailer pushes the tractor round).
    let [(front, plain), (front_s, steered)] = radius[..] else { unreachable!() };

    assert!((front - front_s).abs() < 0.1 * front, "front radii {front} vs {front_s}");
    let (off, off_s) = (front - plain, front_s - steered);
    assert!(off > 0.3 && off_s < 0.8 * off, "offtracking {off_s} vs {off}");
}

fn circumcentre(a: glam::DVec2, b: glam::DVec2, c: glam::DVec2) -> glam::DVec2 {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    let (a2, b2, c2) = (a.length_squared(), b.length_squared(), c.length_squared());
    glam::DVec2::new(
        (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d,
        (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d,
    )
}

/// A dual wheel is two tyres at half the load each: twice the single tyre's stiffness and
/// nominal load, over the pair's overall width; the truck dampers are degressive.
#[test]
fn dual_tyres_and_degressive_dampers() {
    let d = rig("truck_6x4", None);
    let dual = d.tire(1);
    let spacing = d.axles[1].dual.unwrap();
    assert_eq!(dual.dual, Some(spacing));
    let mut single = dual.clone();
    single.dual = None;
    assert_eq!(dual.count(), 2.0);
    assert!((dual.nominal_load() - 2.0 * single.nominal_load()).abs() < 1e-9);
    for x in [0.005, 0.02, 0.04] {
        assert!((dual.vertical_force(x) - 2.0 * single.vertical_force(x)).abs() < 1e-6 * dual.vertical_force(x));
    }
    assert!((dual.width() - (single.width() + spacing)).abs() < 1e-12);
    assert_eq!(dual.section_width(), single.width());
    let s = d.axles[0].suspension.as_ref().unwrap();
    let damper = &s.damper;
    assert!(damper.degressivity_bump > 0.0 && damper.degressivity_rebound > 0.0);
    for v in [0.05, 0.3, 1.0] {
        let bump = damper.bump * v / (1.0 + damper.degressivity_bump * v);
        let rebound = -damper.rebound * v / (1.0 + damper.degressivity_rebound * v);
        assert!((s.damper_force(v) - bump).abs() < 1e-9 * bump);
        assert!((s.damper_force(-v) - rebound).abs() < 1e-9 * -rebound);
    }
    // Force still grows with speed, ever more slowly.
    assert!(s.damper_force(1.0) > s.damper_force(0.5) && s.damper_force(0.5) > 0.5 * s.damper_force(1.0));
}
