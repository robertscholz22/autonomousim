//! Articulated vehicles: a test tractor with a semitrailer and with a drawbar trailer (dolly,
//! drawbar and body on a turntable). The composition builds the right tree, the energy solver
//! agrees with the lever-rule solver on two-axle vehicles and with the statics of the rigs,
//! and the rigs settle at their static state, roll straight and follow a turn.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::presets;
use glam::DVec3;
use std::sync::Arc;

const DT: f64 = 1e-3;
const G: f64 = STANDARD_GRAVITY;

const TYRE: &str = r#"
[axles.tire.fiala]
radius = 0.5
width = 0.3
vertical_stiffness = 1.2e6
vertical_damping = 3000.0
slip_stiffness = 3.0e5
cornering_stiffness = 2.0e5
mu = 0.8
nominal_load = 25000.0
relaxation_x = 0.3
relaxation_y = 0.5
"#;

fn axle(position: [f64; 3], extra: &str) -> String {
    format!(
        r#"
[[axles]]
position = {position:?}
wheel = {{ mass = 100.0, inertia = [10.0, 20.0, 10.0] }}
brake = {{ max_torque = 15000.0, parking_torque = 15000.0 }}
{extra}
[axles.suspension]
carrier_mass = 150.0
carrier_inertia = [20.0, 20.0, 20.0]
spring = {{ rate = 4.0e5 }}
damper = {{ bump = 3.0e4, rebound = 3.0e4 }}
bump_stop = {{ travel = 0.1, stiffness = 5.0e6 }}
rebound_stop = {{ travel = -0.1, stiffness = 5.0e6 }}
{TYRE}"#
    )
}

/// A 4×2 tractor: steered front axle, driven rear axle, the hitch as given.
fn tractor(hitch: &str) -> WheeledDef {
    let toml = format!(
        r#"
name = "test_tractor"
[chassis]
mass = 6000.0
com = [-1.4, 0.0, 0.6]
inertia = [3000.0, 20000.0, 20000.0]
drag_area = [5.0, 10.0, 10.0]
[steering]
max_angle = 0.6
rate = 0.6
ackermann = 1.0
{}{}
[powertrain]
type = "electric"
[[powertrain.motors]]
wheels = [2, 3]
max_torque = 800.0
max_power = 250000.0
ratio = 0.04
time_constant = 0.05
coupling = "open"
[[colliders]]
center = [-1.5, 0.0, 0.3]
radius = 0.4
{hitch}
"#,
        axle([0.0, 1.0, 0.0], "steer = 1.0"),
        axle([-3.8, 0.9, 0.0], ""),
    );
    WheeledDef::from_toml(&toml).unwrap()
}

fn semitractor() -> WheeledDef {
    tractor("[hitch]\nkind = \"fifth_wheel\"\nposition = [-3.3, 0.0, 0.6]")
}

fn drawtractor() -> WheeledDef {
    tractor("[hitch]\nkind = \"drawbar\"\nposition = [-4.6, 0.0, 0.0]")
}

/// A tandem-axle semitrailer, kingpin at the origin.
fn semitrailer() -> TrailerDef {
    let toml = format!(
        r#"
type = "trailer"
name = "test_semitrailer"
[coupling]
kind = "fifth_wheel"
position = [0.0, 0.0, 0.0]
[chassis]
mass = 8000.0
com = [-5.5, 0.0, 0.8]
inertia = [8000.0, 90000.0, 90000.0]
drag_area = [2.0, 20.0, 20.0]
[[colliders]]
center = [-5.0, 0.0, 0.2]
radius = 0.4
{}{}"#,
        axle([-9.0, 0.9, -0.6], ""),
        axle([-10.3, 0.9, -0.6], ""),
    );
    TrailerDef::from_toml(&toml).unwrap()
}

/// A drawbar trailer: dolly (frame at the dolly axle) and a body with one rear axle whose
/// frame coincides with the dolly's at design.
fn drawbar_trailer() -> TrailerDef {
    let toml = format!(
        r#"
type = "trailer"
name = "test_drawbar"
[coupling]
kind = "drawbar"
position = [3.0, 0.0, 0.1]
[chassis]
mass = 6000.0
com = [-2.5, 0.0, 1.0]
inertia = [4000.0, 15000.0, 15000.0]
{}
[dolly]
chassis = {{ mass = 600.0, com = [0.0, 0.0, 0.3], inertia = [300.0, 300.0, 400.0] }}
hinge = [0.8, 0.0, 0.0]
drawbar_mass = 80.0
turntable = [0.0, 0.0, 0.5]
mount = [0.0, 0.0, 0.5]
{}"#,
        axle([-5.0, 0.9, 0.0], ""),
        axle([0.0, 0.9, 0.0], "").replace("[[axles]]", "[[dolly.axles]]").replace("[axles.", "[dolly.axles."),
    );
    TrailerDef::from_toml(&toml).unwrap()
}

fn semi_rig() -> WheeledDef {
    semitractor().with_trailers(&[semitrailer()]).unwrap()
}

fn drawbar_rig() -> WheeledDef {
    drawtractor().with_trailers(&[drawbar_trailer()]).unwrap()
}

struct Flat {
    terrain: FlatTerrain,
    materials: MaterialTable,
}

impl Flat {
    fn new() -> Self {
        Self { terrain: FlatTerrain::new(0.0, MaterialId::ASPHALT), materials: MaterialTable::standard() }
    }

    fn run(&self, v: &mut Wheeled, input: &DriveInput, seconds: f64) {
        let env = GroundStepEnv {
            scene: StaticScene { terrain: &self.terrain, obstacles: &NoObstacles, materials: &self.materials },
            gravity: DVec3::new(0.0, 0.0, -G),
            air: AirData::default(),
        };
        for _ in 0..(seconds / DT).round() as usize {
            v.step(input, &env).unwrap();
        }
    }
}

#[test]
fn composition_builds_the_units() {
    let semi = semi_rig();
    assert_eq!(semi.num_units(), 2);
    assert_eq!(semi.num_wheels(), 8);
    assert_eq!(semi.axles.iter().map(|a| a.unit).collect::<Vec<_>>(), [0, 0, 1, 1]);
    // Positions move to the kingpin; the unit hangs at the fifth wheel.
    assert_eq!(semi.units[0].position, DVec3::new(-3.3, 0.0, 0.6));
    let draw = drawbar_rig();
    let names: Vec<&str> = draw.units.iter().map(|u| u.name.as_str()).collect();
    assert_eq!(names, ["test_drawbar_drawbar", "test_drawbar_dolly", "test_drawbar"]);
    assert_eq!(draw.axles.iter().map(|a| a.unit).collect::<Vec<_>>(), [0, 0, 2, 3]);
    assert!(matches!(draw.units[1].joint, UnitJoint::Hinge) && matches!(draw.units[2].joint, UnitJoint::Turntable));
    // The dolly's frame sits at the hinge, the body's at the turntable.
    assert_eq!(draw.units[1].position, DVec3::new(-2.2, 0.0, -0.1));
    assert_eq!(draw.units[2].position, DVec3::new(-0.8, 0.0, 0.5));
    assert_eq!(draw.wheel_position(4), DVec3::new(-0.8, 0.9, 0.0));
    for d in [&semi, &draw] {
        let v = Wheeled::new(Arc::new(d.clone()), DT);
        for u in 1..d.num_units() {
            assert_eq!(v.model().link(d.unit_link(u)).name, d.units[u - 1].name);
            assert_eq!(v.unit_link(u), d.unit_link(u));
        }
        let colliders = d.sphere_colliders();
        assert!(colliders.iter().any(|c| c.link != 0) || d.units.iter().all(|u| u.colliders.is_empty()));
        // Recordings keep the composed definition.
        let json = serde_json::to_string(d).unwrap();
        let mut back: WheeledDef = serde_json::from_str(&json).unwrap();
        back.finish().unwrap();
        assert_eq!(&back, d);
    }
    // Single units serialise as before.
    assert!(!serde_json::to_string(&presets::wheeled("sedan_like").unwrap()).unwrap().contains("units"));
    // Mismatched couplings are refused.
    assert!(drawtractor().with_trailers(&[semitrailer()]).is_err());
    assert!(semitractor().with_trailers(&[drawbar_trailer()]).is_err());
    assert!(presets::wheeled("sedan_like").unwrap().with_trailers(&[semitrailer()]).is_err());
}

/// The energy solver finds the lever-rule solver's equilibrium on two-axle vehicles.
#[test]
fn solvers_agree_on_two_axle_vehicles() {
    for name in ["sedan_like", "offroad_4x4", "rover_skid"] {
        let d = presets::wheeled(name).unwrap();
        let (a, b) = (d.static_state(G).unwrap(), d.static_state_general(G).unwrap());
        let close = |x: f64, y: f64, tol: f64| (x - y).abs() < tol;
        assert!(close(a.height, b.height, 1e-6), "{name}: height {} vs {}", a.height, b.height);
        assert!(close(a.pitch, b.pitch, 1e-6) && close(a.roll, b.roll, 1e-6), "{name}: {a:?} vs {b:?}");
        for w in 0..d.num_wheels() {
            assert!(close(a.loads[w], b.loads[w], 1e-5 * a.loads[w]), "{name} {w}: {} vs {}", a.loads[w], b.loads[w]);
            assert!(close(a.travel[w], b.travel[w], 1e-6), "{name} {w}: {} vs {}", a.travel[w], b.travel[w]);
        }
    }
}

/// Loads carry the weight, the semitrailer's load splits between the kingpin and its tandem
/// by the lever rule, and the tractor carries the kingpin load.
#[test]
fn rig_statics() {
    for d in [semi_rig(), drawbar_rig()] {
        let st = d.rest_state().expect("a static state");
        let total: f64 = st.loads.iter().sum();
        assert!((total / (d.total_mass() * G) - 1.0).abs() < 1e-9, "{}: {total}", d.units[0].name);
        // Automatic preloads: every spring at zero travel.
        assert!(st.travel.iter().all(|t| t.abs() < 1e-9), "{:?}", st.travel);
    }
    let d = semi_rig();
    let st = d.rest_state().unwrap();
    let on = |u: usize| (0..8).filter(|&w| d.wheel_unit(w) == u).map(|w| st.loads[w]).sum::<f64>();
    // Kingpin load by the lever rule about the tandem's centre (the trailer barely pitches).
    let trailer = d.unit_mass(1) * G;
    let com = (d.units[0].chassis.com * d.units[0].chassis.mass
        + (4..8).map(|w| d.wheel_position(w) * d.unsprung_mass(w)).sum::<DVec3>())
        / d.unit_mass(1);
    let tandem = -9.65;
    let kingpin = trailer * (com.x - tandem) / (0.0 - tandem);
    assert!(((trailer - on(1)) / kingpin - 1.0).abs() < 0.01, "kingpin {} vs {kingpin}", trailer - on(1));
    assert!((on(0) - d.unit_mass(0) * G - kingpin).abs() < 0.01 * kingpin);
    // The tandem's axles share the load roughly (without load-equalising suspension, the
    // trailer's pitch shifts load between them).
    let (a, b) = (st.loads[4] + st.loads[5], st.loads[6] + st.loads[7]);
    assert!((a / b - 1.0).abs() < 0.2, "tandem {a} / {b}");
    assert!(st.joints[0][0].abs() < 0.02 && st.joints[0][1].abs() < 1e-9, "{:?}", st.joints);
}

fn at_rest(d: WheeledDef) -> Wheeled {
    let mut v = Wheeled::new(Arc::new(d), DT);
    let init = v.rest(DVec3::ZERO, 0.0, 0.0);
    v.reset(&init);
    v
}

/// Released at the static state, the rigs stay there.
#[test]
fn rigs_settle_at_the_static_state() {
    let flat = Flat::new();
    for d in [semi_rig(), drawbar_rig()] {
        let st = d.rest_state().unwrap().clone();
        let mut v = at_rest(d);
        let brake = DriveInput { brake: 1.0, ..Default::default() };
        flat.run(&mut v, &brake, 3.0);
        assert!(v.lin_vel_world().length() < 1e-3, "still moving");
        assert!(v.contacts().is_empty(), "body on the ground");
        assert!((v.position().z - st.height).abs() < 5e-4, "height {} vs {}", v.position().z, st.height);
        for (k, wh) in v.wheels().enumerate() {
            assert!((wh.tire.fz / st.loads[k] - 1.0).abs() < 5e-3, "wheel {k}: {} vs {}", wh.tire.fz, st.loads[k]);
        }
        for u in 1..v.num_units() {
            assert!(v.articulation(u).0.abs() < 1e-4, "unit {u}: {:?}", v.articulation(u));
        }
    }
}

/// Driving straight, the trailers follow in line; in a left turn they swing out to the right
/// of the tractor (negative articulation) and their axles run inside the tractor's path.
#[test]
fn rigs_roll_straight_and_turn() {
    let flat = Flat::new();
    for d in [semi_rig(), drawbar_rig()] {
        let units = d.num_units();
        let mut v = at_rest(d);
        flat.run(&mut v, &DriveInput { throttle: 0.3, ..Default::default() }, 8.0);
        assert!(v.speed() > 3.0, "speed {}", v.speed());
        let last = v.unit_pose(units - 1);
        assert!(v.position().y.abs() < 0.05 && last.pos.y.abs() < 0.05, "{} / {}", v.position(), last.pos);
        for u in 1..units {
            assert!(v.articulation(u).0.abs() < 1e-3, "unit {u}: {:?}", v.articulation(u));
        }
        flat.run(&mut v, &DriveInput { steering: 0.5, ..Default::default() }, 6.0);
        let (angle, _) = v.articulation(1);
        assert!(angle < -0.05, "articulation {angle}");
    }
}
