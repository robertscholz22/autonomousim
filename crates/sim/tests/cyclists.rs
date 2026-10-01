//! Cyclists in traffic (M8c step 3): bicycles driven by `traffic` drivers keep right (in bike
//! lanes where there are), cars pass them with room or overtake them through the oncoming
//! lane, and hybrid cyclists promoted to full physics ride on without falling.

use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::math::Pose;
use autonomousim_core::rng::Seed;
use autonomousim_sim::driver::Driver;
use autonomousim_sim::events::Events;
use autonomousim_sim::traffic_driver::{Elem, TrafficDriver, keep_right};
use autonomousim_sim::{CompiledScenario, Scenario, WorldInstance};
use glam::{DVec2, DVec3, EulerRot};
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

fn traffic(w: &WorldInstance, i: usize) -> &TrafficDriver {
    match w.agent(i).driver.as_ref() {
        Some(Driver::Traffic(d)) => d,
        _ => panic!("a traffic driver"),
    }
}

fn speed(w: &WorldInstance, i: usize) -> f64 {
    w.agent(i).vehicle.lin_vel_world().truncate().length()
}

fn roll(w: &WorldInstance, i: usize) -> f64 {
    w.agent(i).vehicle.orientation().to_euler(EulerRot::ZYX).2
}

/// Groups of `cars` sedans and `bikes` bicycles (physics `physics`) driven by `traffic`
/// drivers; hybrid bicycles are promoted within 100 m of a learning sedan, which stands.
fn groups(cars: usize, bikes: usize, physics: &str, separation: f64) -> String {
    let (hybrid, ego) = if physics == "hybrid" {
        (
            "hybrid = { promote = 100.0, demote = 120.0 }",
            "[[groups]]\nname = \"ego\"\nvehicle = \"sedan_like\"\nspawn = { on_ground = true, on_road = true }\n",
        )
    } else {
        ("", "")
    };
    format!(
        r#"
        [[groups]]
        name = "cars"
        count = {cars}
        vehicle = "sedan_like"
        physics = "kinematic"
        driver = {{ type = "traffic" }}
        spawn = {{ on_ground = true, on_road = true, min_separation = {separation} }}
        disable_on_terminal = false
        [[groups]]
        name = "bikes"
        count = {bikes}
        vehicle = "bicycle_city"
        physics = "{physics}"
        {hybrid}
        driver = {{ type = "traffic", speed = [4.0, 6.0], lateral_accel = 1.5 }}
        spawn = {{ on_ground = true, on_road = true, min_separation = {separation} }}
        disable_on_terminal = false
        {ego}
        "#
    )
}

#[derive(Debug, Default)]
struct Stats {
    crashes: u32,
    red_lights: u32,
    /// Falls of cyclists (counted once each until they rise or respawn).
    falls: u32,
    overtakes: u32,
    /// Closest any car came to any cyclist (m, centre to centre).
    nearest: f64,
    /// Mean speed of the cyclists (m/s), and their largest distance from where they ride (m,
    /// on lanes, 10 m clear of the ends, after the first 10 s).
    bike_speed: f64,
    off_line: f64,
    /// Promotions of cyclists, and their largest lean in full physics (rad).
    promotions: u32,
    full_roll: f64,
}

/// `minutes` of traffic (the first `cars` agents cars, the next `bikes` cyclists), a learning
/// car (if any, after them) standing.
fn run(toml: &str, seed: u64, cars: usize, bikes: usize, minutes: f64) -> Stats {
    let mut w = WorldInstance::new(compile(toml), Seed::from_u64(seed));
    let map = w.map().clone();
    let g = map.roads().lanes();
    let n = cars + bikes;
    let learner = w.agents().len() > n;
    let mut st = Stats { nearest: f64::INFINITY, ..Stats::default() };
    let steps = (25.0 * 60.0 * minutes) as usize;
    let mut sum = 0.0;
    let mut fallen = vec![false; n];
    let mut kinematic: Vec<bool> = (0..n).map(|i| w.agent(i).is_kinematic()).collect();
    for step in 1..=steps {
        if learner {
            w.set_command(n, GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 });
        }
        w.step();
        for i in 0..n {
            let e = w.agent(i).events;
            st.crashes += e.contains(Events::CRASH_AGENT) as u32;
            st.red_lights += e.contains(Events::RED_LIGHT) as u32;
            let was = std::mem::replace(&mut fallen[i], e.contains(Events::ROLLOVER));
            st.falls += (fallen[i] && !was) as u32;
        }
        for b in cars..n {
            let kin = w.agent(b).is_kinematic();
            st.promotions += (kinematic[b] && !kin) as u32;
            kinematic[b] = kin;
            if !kin {
                st.full_roll = st.full_roll.max(roll(&w, b).abs());
            }
            sum += speed(&w, b);
            let p = w.agent(b).vehicle.position();
            for c in 0..cars {
                st.nearest = st.nearest.min(w.agent(c).vehicle.position().distance(p));
            }
            if step > 250
                && let Some(place) = traffic(&w, b).place
                && let Elem::Lane(l) = place.elem
            {
                let line = &g.lanes()[l as usize].line;
                let pr = line.project(p.truncate());
                if pr.station > 10.0 && pr.station < line.length() - 10.0 && !kinked(line, pr.station) {
                    st.off_line = st.off_line.max((pr.offset - keep_right(&map, l)).abs());
                }
            }
        }
    }
    st.bike_speed = sum / (steps * bikes) as f64;
    st.overtakes = (0..cars).map(|i| traffic(&w, i).overtakes).sum();
    st
}

/// Whether the lane bends tighter than a 10 m radius within 10 m behind or 5 m ahead of
/// `station` (a kinked street: the cyclist's line inside it is tighter than they can ride at
/// speed, so they swing wide and rejoin it).
fn kinked(line: &autonomousim_world::Polyline, station: f64) -> bool {
    (-10..=5).any(|d| line.curvature_at(station + f64::from(d)).abs() > 0.1)
}

/// Urban training map `seed`.
fn urban(seed: u64, cars: usize, bikes: usize, physics: &str, hz: u32) -> String {
    format!(
        "physics_hz = {hz}\npolicy_hz = 25\nmap = {{ type = \"urban\", seed = {seed}, count = 1 }}\n{}",
        groups(cars, bikes, physics, 20.0)
    )
}

/// A ring road of radius 150 m with one lane each way (lanes of about 470 m).
fn ring(cars: usize, bikes: usize, physics: &str) -> String {
    format!(
        "physics_hz = 500\npolicy_hz = 25\nmap = {{ type = \"testworld\", kind = \"ring\", radius = 150.0, \
         lanes = [1, 1], class = \"local\" }}\n{}",
        groups(cars, bikes, physics, 30.0)
    )
}

/// On a long two-way road, cars overtake the cyclists through the oncoming lane, never
/// closer than 2 m (centre to centre: about 1 m between the car's side and the handlebar).
#[test]
fn cars_overtake_cyclists_with_room() {
    let st = run(&ring(8, 6, "kinematic"), 1, 8, 6, 10.0);
    println!("{st:?}");
    assert!(st.overtakes >= 10, "overtakes {}", st.overtakes);
    assert_eq!(st.crashes, 0);
    assert!(st.nearest > 2.0, "nearest {}", st.nearest);
    assert!(st.bike_speed > 3.5, "cyclists held up: {}", st.bike_speed);
    assert!(st.off_line < 0.3, "off their line by {}", st.off_line);
}

/// The same with the cyclists in full physics: they balance, steer and are overtaken.
#[test]
fn cyclists_in_full_physics_ride_and_are_overtaken() {
    let st = run(&ring(8, 6, "full"), 2, 8, 6, 3.0);
    println!("{st:?}");
    assert_eq!(st.falls, 0);
    assert_eq!(st.crashes, 0);
    assert!(st.overtakes >= 3, "overtakes {}", st.overtakes);
    assert!(st.off_line < 0.5, "off their line by {}", st.off_line);
}

/// In urban traffic cyclists keep right (in bike lanes where there are), stop at red lights,
/// and nobody collides.
#[test]
fn cyclists_keep_right_in_urban_traffic() {
    let st = run(&urban(1, 50, 30, "kinematic", 200), 1, 50, 30, 3.0);
    println!("{st:?}");
    assert_eq!((st.crashes, st.red_lights, st.falls), (0, 0, 0));
    assert!(st.off_line < 0.6, "off their line by {}", st.off_line);
    assert!(st.bike_speed > 2.0, "cyclists held up: {}", st.bike_speed);
}

/// Hybrid cyclists near the learning car ride in full physics: promoted mid-ride (often in
/// a turn), they ride on without falling.
#[test]
fn promoted_cyclists_ride_on() {
    let st = run(&urban(1, 50, 30, "hybrid", 500), 1, 50, 30, 3.0);
    println!("{st:?}");
    assert!(st.promotions >= 10, "promotions {}", st.promotions);
    assert_eq!((st.crashes, st.red_lights, st.falls), (0, 0, 0));
    assert!(st.full_roll < 0.5, "leaned {}", st.full_roll);
}

/// The step 3 acceptance run: 30 minutes on each of the first four training maps with hybrid
/// cyclists.
#[test]
#[ignore]
fn cyclists_in_traffic_for_half_an_hour() {
    for seed in 1..=4 {
        let st = run(&urban(seed, 50, 30, "hybrid", 500), seed, 50, 30, 30.0);
        println!("seed {seed}: {st:?}");
        assert!(st.promotions > 0, "seed {seed}: no promotions");
        assert_eq!((st.crashes, st.red_lights, st.falls), (0, 0, 0), "seed {seed}");
        assert!(st.off_line < 0.8, "seed {seed}: off their line by {}", st.off_line);
    }
}

/// A hybrid bicycle in a steady kinematic turn, promoted to full physics: it takes over
/// leaning into the turn and holds it.
#[test]
fn a_bicycle_promoted_in_a_turn_holds_it() {
    let toml = r#"
        physics_hz = 500
        policy_hz = 25
        map = { type = "testworld", kind = "flat", size = 600.0 }
        [[groups]]
        name = "bike"
        vehicle = "bicycle_city"
        physics = "hybrid"
        hybrid = { promote = 20.0, demote = 60.0 }
        action_mode = "vk"
        spawn = { region = [[-250.0, 0.0], [-250.0, 0.0]], yaw_deg = [0.0, 0.0] }
        [[groups]]
        name = "ego"
        vehicle = "sedan_like"
        spawn = { region = [[200.0, 200.0], [200.0, 200.0]], yaw_deg = [0.0, 0.0] }
    "#;
    let (v, k) = (5.0, 0.08);
    let mut w = WorldInstance::new(compile(toml), Seed::from_u64(2));
    let drive = |w: &mut WorldInstance| {
        w.set_command(0, GroundSetpoint::SpeedCurvature { speed: v, curvature: k });
        w.set_command(1, GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 });
        w.step();
    };
    for _ in 0..150 {
        drive(&mut w);
    }
    assert!(w.agent(0).is_kinematic());
    // (Leaning left: negative roll.)
    let lean = roll(&w, 0);
    assert!(lean < -0.1, "kinematic lean {lean}");
    // The learning car pulls up beside it.
    let p = w.agent(0).vehicle.position();
    let pose = w.agent(1).vehicle.pose();
    w.place_agent(
        1,
        Pose::new((p.truncate() + DVec2::new(10.0, 8.0)).extend(pose.pos.z), pose.rot),
        DVec3::ZERO,
        DVec3::ZERO,
    );
    w.switch_physics();
    assert!(!w.agent(0).is_kinematic());
    let mut worst: f64 = 0.0;
    for _ in 0..150 {
        drive(&mut w);
        assert!(!w.agent(0).events.contains(Events::ROLLOVER), "fell");
        worst = worst.max((roll(&w, 0) - lean).abs());
    }
    let curvature = (w.agent(0).vehicle.orientation() * w.agent(0).vehicle.ang_vel_body()).z / speed(&w, 0);
    println!("kinematic lean {lean:.3}, largest change {worst:.3}, curvature {curvature:.4}");
    assert!(!w.agent(0).is_kinematic(), "demoted");
    assert!(worst < 0.1, "lean swung by {worst}");
    assert!((speed(&w, 0) - v).abs() < 0.3 && (curvature - k).abs() < 0.01, "curvature {curvature}");
}

/// Two-wheelers drive only in traffic (as cyclists), in any physics mode.
#[test]
fn two_wheelers_are_only_traffic_cyclists() {
    let scenario = |driver: &str, physics: &str| {
        let toml = format!(
            r#"
            map = {{ type = "urban", seed = 1, count = 1 }}
            [[groups]]
            name = "bikes"
            vehicle = "bicycle_city"
            physics = "{physics}"
            driver = {{ type = "{driver}" }}
            spawn = {{ on_ground = true, on_road = true }}
            "#
        );
        Scenario::from_toml(&toml).unwrap().compile().map(|_| ())
    };
    for physics in ["full", "kinematic", "hybrid"] {
        assert!(scenario("traffic", physics).is_ok(), "{physics}");
    }
    let err = scenario("road", "full").unwrap_err().to_string();
    assert!(err.contains("not two-wheelers"), "{err}");
}
