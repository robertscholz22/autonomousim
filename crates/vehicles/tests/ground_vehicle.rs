//! Wheeled vehicle checks: presets, static equilibrium against the definition and against
//! Chrono::Vehicle, holding on a 30 % slope, full-throttle runs, coast-down and braking against
//! Chrono (fixtures/chrono/vehicle_*.json, from tools/gen_chrono_vehicle_fixtures.py), and the
//! robots' drives.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use autonomousim_core::terrain::{FlatTerrain, PlaneTerrain, Terrain};
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::{VehicleDef, presets};
use glam::{DQuat, DVec3};
use serde_json::Value;
use std::sync::Arc;

const DT: f64 = 1e-3;
/// Chrono's gravity.
const G: f64 = 9.81;
/// Preset name and Chrono fixture.
const CARS: [(&str, &str); 2] = [("sedan_like", "sedan"), ("offroad_4x4", "hmmwv")];

fn def(name: &str) -> WheeledDef {
    presets::wheeled(name).unwrap()
}

/// A vehicle without aerodynamic drag (Chrono models none).
fn vehicle(name: &str) -> Wheeled {
    let mut d = def(name);
    d.chassis.drag_area = DVec3::ZERO;
    Wheeled::new(Arc::new(d), DT)
}

fn fixture(name: &str) -> Value {
    let path = format!("{}/../../fixtures/chrono/vehicle_{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

struct World {
    terrain: Box<dyn Terrain>,
    materials: MaterialTable,
}

impl World {
    fn new(terrain: impl Terrain + 'static) -> Self {
        Self { terrain: Box::new(terrain), materials: MaterialTable::standard() }
    }

    fn flat() -> Self {
        Self::new(FlatTerrain::new(0.0, MaterialId::ASPHALT))
    }

    /// Flat asphalt without rolling resistance: Chrono's Pac02 tyres have none when the
    /// `.tir` file gives no `QSY` coefficients (ours fall back to the surface's).
    fn chrono() -> Self {
        let mut materials = MaterialTable::standard();
        let mut m = materials.get(MaterialId::ASPHALT).clone();
        m.name = "asphalt_no_rolling".into();
        m.rolling_resistance = 0.0;
        let id = materials.push(m);
        Self { terrain: Box::new(FlatTerrain::new(0.0, id)), materials }
    }

    fn run(&self, v: &mut Wheeled, input: &DriveInput, seconds: f64) {
        let env = GroundStepEnv {
            scene: StaticScene { terrain: self.terrain.as_ref(), obstacles: &NoObstacles, materials: &self.materials },
            gravity: DVec3::new(0.0, 0.0, -G),
            air: AirData::default(),
        };
        for _ in 0..(seconds / DT).round() as usize {
            v.step(input, &env).unwrap();
        }
    }
}

fn brake() -> DriveInput {
    DriveInput { brake: 1.0, ..Default::default() }
}

fn throttle(t: f64) -> DriveInput {
    DriveInput { throttle: t, ..Default::default() }
}

fn at_rest(name: &str) -> Wheeled {
    let mut v = vehicle(name);
    let init = v.rest(DVec3::ZERO, 0.0, 0.0);
    v.reset(&init);
    v
}

#[test]
fn presets_load() {
    for (name, mass) in
        [("sedan_like", 1671.261), ("offroad_4x4", 2573.122), ("rover_diff", 17.0), ("rover_skid", 50.0)]
    {
        let VehicleDef::Wheeled(d) = presets::get(name).unwrap() else { panic!("{name} is not wheeled") };
        assert_eq!(d.name, name);
        assert!(!d.source.is_empty());
        assert!((d.total_mass() - mass).abs() < 1e-6, "{name}: {}", d.total_mass());
        let v = Wheeled::new(Arc::new(d), DT);
        assert_eq!(v.num_wheels(), if name == "rover_diff" { 2 } else { 4 });
    }
    for (name, chrono) in CARS {
        assert!((def(name).total_mass() - num(&fixture(chrono)["design"]["mass"])).abs() < 0.05, "{name}");
    }
}

/// The static state the definition predicts is where the simulated vehicle settles; with
/// automatic spring preloads, at zero travel (the design ride height).
#[test]
fn settles_at_the_static_state_of_the_definition() {
    let w = World::flat();
    for name in ["sedan_like", "offroad_4x4", "rover_skid"] {
        let mut v = at_rest(name);
        let st = v.def().static_state(G).unwrap();
        w.run(&mut v, &brake(), 3.0);
        assert!(v.lin_vel_world().length() < 1e-3, "{name}: still moving");
        assert!(v.contacts().is_empty(), "{name}: body on the ground");
        let (_, pitch, roll) = v.orientation().to_euler(glam::EulerRot::ZYX);
        assert!((v.position().z - st.height).abs() < 5e-4, "{name}: height {} vs {}", v.position().z, st.height);
        assert!(
            (pitch - st.pitch).abs() < 2e-4 && (roll - st.roll).abs() < 2e-4,
            "{name}: pitch {pitch} vs {}",
            st.pitch
        );
        for (k, wh) in v.wheels().enumerate() {
            assert!(
                (wh.tire.fz / st.loads[k] - 1.0).abs() < 5e-3,
                "{name} wheel {k}: load {} vs {}",
                wh.tire.fz,
                st.loads[k]
            );
            // The mirrored tyres' lateral shifts (conicity) push the two sides apart a little,
            // which the suspension geometry turns into a fraction of a millimetre of travel.
            assert!(
                (wh.travel - st.travel[k]).abs() < 5e-4,
                "{name} wheel {k}: travel {} vs {}",
                wh.travel,
                st.travel[k]
            );
        }
        let sprung_auto =
            v.def().axles.iter().all(|a| a.suspension.as_ref().is_some_and(|s| s.spring.travel.is_empty()));
        if sprung_auto {
            assert!(st.travel.iter().all(|t| t.abs() < 1e-4), "{name}: {:?}", st.travel);
        }
        let total: f64 = st.loads.iter().sum();
        assert!((total / (v.mass() * G) - 1.0).abs() < 1e-9);
    }
}

#[test]
fn static_state_matches_chrono() {
    for (name, chrono) in CARS {
        let fx = fixture(chrono);
        let s = &fx["static"];
        let st = def(name).static_state(G).unwrap();
        assert!((st.height - num(&s["ref_z"])).abs() < 2e-3, "{name}: ride height {} vs {}", st.height, s["ref_z"]);
        assert!((st.pitch - num(&s["pitch"])).abs() < 1e-3, "{name}: pitch {} vs {}", st.pitch, s["pitch"]);
        let loads = nums(&s["loads"]);
        for axle in 0..2 {
            let (ours, theirs) = (st.loads[2 * axle] + st.loads[2 * axle + 1], loads[2 * axle] + loads[2 * axle + 1]);
            assert!((ours / theirs - 1.0).abs() < 5e-3, "{name} axle {axle}: load {ours} vs {theirs}");
        }
    }
}

#[test]
fn braked_vehicles_hold_on_a_30_percent_slope() {
    let angle = 0.3f64.atan();
    let w = World::new(PlaneTerrain::incline(angle, MaterialId::ASPHALT));
    let tilt = DQuat::from_rotation_y(-angle);
    for (name, input) in [
        ("sedan_like", brake()),
        ("offroad_4x4", brake()),
        ("rover_skid", DriveInput { parking: true, ..Default::default() }),
    ] {
        // Facing uphill and downhill.
        for yaw in [0.0, std::f64::consts::PI] {
            let place = |v: &mut Wheeled| {
                let rest = v.rest(DVec3::ZERO, yaw, 0.0);
                v.reset(&WheeledInit { pose: Pose::new(tilt * rest.pose.pos, tilt * rest.pose.rot), ..rest });
            };
            let mut v = Wheeled::new(Arc::new(def(name)), DT);
            place(&mut v);
            w.run(&mut v, &input, 3.0);
            let p = v.position();
            w.run(&mut v, &input, 5.0);
            let drift = (v.position() - p).length();
            assert!(drift < 1e-4, "{name} yaw {yaw}: crept {drift} m in 5 s");
            assert!(v.contacts().is_empty(), "{name}: body on the ground");
            // Released, it rolls away.
            let mut free = Wheeled::new(Arc::new(def(name)), DT);
            place(&mut free);
            let p0 = free.position();
            w.run(&mut free, &DriveInput::default(), 3.0);
            assert!((free.position() - p0).length() > 0.3, "{name} yaw {yaw}: unbraked but held");
        }
    }
}

/// Full throttle from rest: speeds and gears against Chrono. The launch differs (no clutch in
/// either model; our transient tyre lets the wheels spin up more), so the comparison allows
/// ~10 %; beyond ~15 s Chrono's engine keeps its map torque at the speed limit, where ours cuts
/// fuel.
#[test]
fn full_throttle_run_matches_chrono() {
    let w = World::chrono();
    for (name, chrono) in CARS {
        let fx = fixture(chrono);
        let reference: Vec<&Value> = fx["accel"].as_array().unwrap().iter().collect();
        let at = |t: f64| reference.iter().find(|s| (num(&s["t"]) - t).abs() < 1e-6).unwrap();
        let t100_chrono = num(&reference.iter().find(|s| num(&s["speed"]) >= 100.0 / 3.6).unwrap()["t"]);
        let mut v = at_rest(name);
        w.run(&mut v, &brake(), 1.0);
        let mut t100 = None;
        for k in 1..=300 {
            w.run(&mut v, &throttle(1.0), 0.05);
            let t = k as f64 * 0.05;
            if t100.is_none() && v.speed() >= 100.0 / 3.6 {
                t100 = Some(t);
            }
            if k % 100 == 0 {
                let s = at(t);
                let (ours, theirs) = (v.speed(), num(&s["speed"]));
                assert!((ours / theirs - 1.0).abs() < 0.08, "{name} at {t} s: {ours} vs {theirs} m/s");
                assert!((v.powertrain().gear as i64 - s["gear"].as_i64().unwrap()).abs() <= 1, "{name} at {t} s: gear");
            }
        }
        let t100 = t100.unwrap();
        assert!((t100 / t100_chrono - 1.0).abs() < 0.12, "{name}: 0-100 km/h in {t100} s vs {t100_chrono} s");
    }
}

/// Released throttle in gear from Chrono's initial speed and gear: engine drag through the
/// driveline and tyre losses.
#[test]
fn coast_down_matches_chrono() {
    let w = World::chrono();
    for (name, chrono) in CARS {
        let fx = fixture(chrono);
        let run = fx["coast"].as_array().unwrap();
        let (first, last) = (&run[0], run.last().unwrap());
        let mut v = vehicle(name);
        let init = v.rest(DVec3::ZERO, 0.0, num(&first["speed"]));
        v.reset(&init);
        v.powertrain_mut().set_gear(first["gear"].as_i64().unwrap() as i32);
        let x0 = v.position().x;
        w.run(&mut v, &throttle(0.0), num(&last["t"]));
        let (speed, distance) = (v.speed(), v.position().x - x0);
        let (speed_c, distance_c) = (num(&last["speed"]), num(&last["x"]) - num(&first["x"]));
        assert!((speed - speed_c).abs() < 0.4, "{name}: {speed} vs {speed_c} m/s after 30 s");
        assert!((distance / distance_c - 1.0).abs() < 0.02, "{name}: {distance} vs {distance_c} m");
        assert_eq!(v.powertrain().gear as i64, last["gear"].as_i64().unwrap(), "{name}");
    }
}

/// Full braking from speed. The sedan's stopping distance matches Chrono. The 4×4 stops
/// later: its wheels lock, and our Pac02 (checked against MFeval) falls to μ ≈ 0.55 at locked
/// wheels where Chrono's keeps ~0.9.
#[test]
fn braking_distance() {
    let w = World::chrono();
    for (name, chrono) in CARS {
        let fx = fixture(chrono);
        let run = fx["brake"].as_array().unwrap();
        let first = &run[0];
        let stop = run.iter().find(|s| num(&s["speed"]) < 0.1).unwrap();
        let distance_c = num(&stop["x"]) - num(&first["x"]);
        let mut v = vehicle(name);
        let init = v.rest(DVec3::ZERO, 0.0, num(&first["speed"]));
        v.reset(&init);
        v.powertrain_mut().set_gear(first["gear"].as_i64().unwrap() as i32);
        let x0 = v.position().x;
        w.run(&mut v, &brake(), 6.0);
        let distance = v.position().x - x0;
        assert!(v.speed() < 0.05, "{name}: not stopped");
        let mean_decel = num(&first["speed"]).powi(2) / (2.0 * distance);
        assert!(mean_decel > 0.55 * G, "{name}: {mean_decel} m/s²");
        if name == "sedan_like" {
            assert!((distance / distance_c - 1.0).abs() < 0.1, "{name}: {distance} vs {distance_c} m");
        }
    }
}

#[test]
fn robots_drive_straight_and_turn_on_the_spot() {
    let w = World::flat();
    // Preset, top speed range (m/s), minimum spot-turn rate (rad/s).
    for (name, speed, turn) in [("rover_diff", (1.4, 1.8), 5.0), ("rover_skid", (0.9, 1.2), 1.0)] {
        let mut v = Wheeled::new(Arc::new(def(name)), DT);
        let init = v.rest(DVec3::ZERO, 0.0, 0.0);
        v.reset(&init);
        w.run(&mut v, &DriveInput::default(), 1.0);
        // On its tyres; the diff-drive's casters just clear of the ground.
        let load: f64 = v.wheels().map(|w| w.tire.fz).sum();
        assert!((load / (v.mass() * G) - 1.0).abs() < 1e-3, "{name}: tyres carry {load} N");
        assert!(v.contacts().is_empty(), "{name}");
        w.run(&mut v, &throttle(1.0), 4.0);
        let u = v.lin_vel_body().x;
        assert!(u > speed.0 && u < speed.1, "{name}: top speed {u}");
        assert!(v.position().y.abs() < 1e-6 && v.ang_vel_body().z.abs() < 1e-9, "{name}: not straight");
        // Full differential command, no throttle: counter-clockwise on the spot.
        w.run(&mut v, &DriveInput { yaw: 1.0, ..Default::default() }, 2.0);
        let p = v.position();
        w.run(&mut v, &DriveInput { yaw: 1.0, ..Default::default() }, 2.0);
        assert!(v.ang_vel_body().z > turn, "{name}: yaw rate {}", v.ang_vel_body().z);
        assert!((v.position() - p).length() < 0.02, "{name}: drifted while turning");
        // Released, the motors' back-EMF stops it.
        w.run(&mut v, &DriveInput::default(), 2.0);
        assert!(v.ang_vel_body().length() < 0.05 && v.lin_vel_world().length() < 0.05, "{name}: still moving");
    }
}

#[test]
fn steering_follows_the_rate_limit_and_ackermann() {
    let w = World::flat();
    let mut v = at_rest("offroad_4x4");
    let s = v.def().steering.unwrap();
    let input = DriveInput { steering: 1.0, ..Default::default() };
    w.run(&mut v, &input, 0.1);
    assert!((v.steering_angle() - 0.1 * s.rate).abs() < 1e-9, "{}", v.steering_angle());
    w.run(&mut v, &input, 1.0);
    assert!((v.steering_angle() - s.max_angle).abs() < 1e-12);
    // Turning left: the inner (left) front wheel turns more; Chrono's HMMWV has 30.6° and
    // 24.1° at full lock.
    let (inner, outer) = (v.wheel(0).steer, v.wheel(1).steer);
    assert!((inner - 30.6f64.to_radians()).abs() < 0.02, "inner {}", inner.to_degrees());
    assert!((outer - 24.1f64.to_radians()).abs() < 0.02, "outer {}", outer.to_degrees());
    assert_eq!(v.wheel(2).steer, 0.0);
}

/// A recorded state shown on another instance poses the wheels and reports the outputs as the
/// simulated vehicle did.
#[test]
fn shown_state_matches_the_simulated_one() {
    let world = World::flat();
    let mut v = vehicle("sedan_like");
    let init = v.rest(DVec3::ZERO, 0.3, 10.0);
    v.reset(&init);
    world.run(&mut v, &DriveInput { steering: 0.3, throttle: 0.2, ..Default::default() }, 2.0);
    let wheels: Vec<WheelState> = v.wheels().copied().collect();
    let mut shown = vehicle("sedan_like");
    let state = WheeledInit { pose: v.pose(), lin_vel_world: v.lin_vel_world(), ang_vel_body: v.ang_vel_body() };
    shown.show(&state, &v.joints(), v.steering_angle(), &wheels, v.powertrain());
    // Kinematics of the current state (a step computes them at its start).
    v.begin_step();
    for w in 0..v.num_wheels() {
        let (a, b) = (v.wheel_pose(w), shown.wheel_pose(w));
        assert!((a.pos - b.pos).length() < 1e-9 && a.rot.angle_between(b.rot) < 1e-9, "wheel {w}");
        assert_eq!(shown.wheel(w), v.wheel(w));
    }
    assert!(v.wheel(0).steer.abs() > 0.05 && v.wheel(0).travel != v.wheel(1).travel);
    assert_eq!((shown.steering_angle(), shown.powertrain()), (v.steering_angle(), v.powertrain()));
}
