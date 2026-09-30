//! Trucks and buses among urban traffic (M8b step 4): tractor-semitrailers and rigid trucks
//! swing wide at junctions and keep within their offtracking; buses keep their loops and
//! dwell at their stops.

use autonomousim_core::rng::Seed;
use autonomousim_sim::driver::Driver;
use autonomousim_sim::events::Events;
use autonomousim_sim::traffic_driver::{Elem, TrafficDriver};
use autonomousim_sim::{CompiledScenario, Scenario, WorldInstance};
use glam::DVec2;
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

/// Heavy-vehicle driver settings: slower, gentler and keeping longer gaps than cars.
const HEAVY: &str = r#"type = "traffic", speed_factor = [0.8, 0.9], headway = [1.5, 2.5], accel = [0.5, 0.8], decel = [1.5, 2.0], lateral_accel = 1.5, safe_decel = 3.0"#;

/// `cars` sedans, `rigs` tractor-semitrailers and `rigid` bobtail tractors, all kinematic
/// traffic NPCs, on urban training map `seed`.
fn trucks_toml(seed: u64, cars: usize, rigs: usize, rigid: usize) -> String {
    format!(
        r#"
        physics_hz = 200
        policy_hz = 25
        map = {{ type = "urban", seed = {seed}, count = 1 }}
        [[groups]]
        name = "rig"
        count = {rigs}
        vehicle = "truck_6x4"
        trailers = ["semitrailer_3axle"]
        physics = "kinematic"
        driver = {{ {HEAVY} }}
        spawn = {{ on_ground = true, on_road = true, min_separation = 40.0 }}
        disable_on_terminal = false
        [[groups]]
        name = "rigid"
        count = {rigid}
        vehicle = "truck_6x4"
        physics = "kinematic"
        driver = {{ {HEAVY} }}
        spawn = {{ on_ground = true, on_road = true, min_separation = 40.0 }}
        disable_on_terminal = false
        [[groups]]
        name = "car"
        count = {cars}
        vehicle = "sedan_like"
        physics = "kinematic"
        driver = {{ type = "traffic" }}
        spawn = {{ on_ground = true, on_road = true, min_separation = 20.0 }}
        disable_on_terminal = false
        "#
    )
}

/// Centre of the rearmost axle of agent `i`'s last unit (world, horizontal).
fn last_axle(w: &WorldInstance, i: usize) -> DVec2 {
    let v = w.agent(i).vehicle.as_wheeled().unwrap();
    let d = v.def();
    let last = d.num_units() - 1;
    let wheels: Vec<usize> = (0..d.num_wheels()).filter(|&k| d.wheel_unit(k) == last).collect();
    let x = wheels.iter().map(|&k| d.wheel_position(k).x).fold(f64::INFINITY, f64::min);
    let rear: Vec<usize> = wheels.into_iter().filter(|&k| (d.wheel_position(k).x - x).abs() < 1e-6).collect();
    rear.iter().map(|&k| v.wheel_pose(k).pos.truncate()).sum::<DVec2>() / rear.len() as f64
}

#[derive(Debug, Default)]
struct TruckStats {
    crashes: usize,
    respawns: u32,
    /// Largest distance of a truck's last axle from its own path beyond the allowance: its
    /// steady-state offtracking on the tightest bend there, how much wider its lock takes
    /// that bend, + 0.5 m (not during lane changes).
    worst_excess: f64,
    /// Largest distance of a truck's last axle from its path (m).
    worst_offset: f64,
    /// Junction entries by the trucks.
    entries: usize,
    /// Truck steps off the lane graph.
    lost: usize,
}

fn run_trucks(seed: u64, cars: usize, rigs: usize, rigid: usize, minutes: f64) -> TruckStats {
    let mut w = WorldInstance::new(compile(&trucks_toml(seed, cars, rigs, rigid)), Seed::from_u64(seed));
    let g = w.map().roads().lanes().clone();
    let trucks = rigs + rigid;
    let mut st = TruckStats::default();
    let mut last = vec![None; trucks];
    let steps = (25.0 * 60.0 * minutes) as usize;
    for _ in 0..steps {
        w.step();
        for i in 0..trucks + cars {
            if w.agent(i).events.contains(Events::CRASH_AGENT) {
                st.crashes += 1;
                if st.crashes <= 6 {
                    eprintln!(
                        "crash at step {}: {i} at {:.1?} place {:?}",
                        w.steps(),
                        w.agent(i).vehicle.position().truncate(),
                        traffic(&w, i).place
                    );
                }
            }
        }
        for i in 0..trucks {
            let d = traffic(&w, i);
            let Some(p) = d.place else {
                if st.lost == 0 {
                    eprintln!(
                        "truck {i} lost at step {} at {:.1?}",
                        w.steps(),
                        w.agent(i).vehicle.position().truncate()
                    );
                }
                st.lost += 1;
                continue;
            };
            if let (Some(Elem::Lane(_)), Elem::Connector(_)) = (last[i], p.elem) {
                st.entries += 1;
            }
            last[i] = Some(p.elem);
            // Not during lane changes and while the last axle follows over.
            if d.change.is_some() || d.since_change < 8.0 {
                continue;
            }
            let elems: Vec<Elem> = d.trail.iter().copied().chain([p.elem]).collect();
            let axle = last_axle(&w, i);
            let offset = elems.iter().map(|e| e.line(&g).project(axle).distance).fold(f64::INFINITY, f64::min);
            let k = elems.iter().map(|e| max_bend(e.line(&g))).fold(0.0, f64::max);
            let l = d.geometry().tracking;
            let r = 1.0 / k.max(1e-6);
            let off = if r > l { r - (r * r - l * l).sqrt() } else { r };
            st.worst_offset = st.worst_offset.max(offset);
            // Bends tighter than its lock (up to the network's margin) it takes wide.
            let kmax = d.geometry().max_curvature;
            let wide = if k > kmax { 1.0 / kmax - 1.0 / k } else { 0.0 };
            st.worst_excess = st.worst_excess.max(offset - off - wide - 0.5);
        }
    }
    st.respawns = (0..trucks + cars).map(|i| traffic(&w, i).respawns).sum();
    st
}

fn max_bend(line: &autonomousim_world::Polyline) -> f64 {
    let n = line.length().ceil() as usize;
    (0..=n).map(|k| line.curvature_at(k as f64).abs()).fold(0.0, f64::max)
}

#[test]
fn trucks_turn_within_their_offtracking() {
    for seed in [1, 2] {
        let st = run_trucks(seed, 30, 4, 2, 10.0);
        eprintln!("seed {seed}: {st:?}");
        assert_eq!(st.crashes, 0, "seed {seed}");
        assert!(st.worst_excess <= 0.0, "seed {seed}: {st:?}");
        assert!(st.entries > 30, "seed {seed}: {st:?}");
    }
}

/// `buses` city buses on a line each and `cars` sedans, kinematic traffic NPCs, on urban
/// training map `seed`.
fn buses_toml(seed: u64, cars: usize, buses: usize) -> String {
    format!(
        r#"
        physics_hz = 200
        policy_hz = 25
        map = {{ type = "urban", seed = {seed}, count = 1 }}
        [[groups]]
        name = "bus"
        count = {buses}
        vehicle = "bus_city"
        physics = "kinematic"
        driver = {{ {HEAVY}, bus = {{ length = [1500.0, 3000.0], stop_spacing = 300.0, dwell = [10.0, 30.0] }} }}
        spawn = {{ on_ground = true, on_road = true, min_separation = 40.0 }}
        disable_on_terminal = false
        [[groups]]
        name = "car"
        count = {cars}
        vehicle = "sedan_like"
        physics = "kinematic"
        driver = {{ type = "traffic" }}
        spawn = {{ on_ground = true, on_road = true, min_separation = 20.0 }}
        disable_on_terminal = false
        "#
    )
}

#[derive(Debug, Default)]
struct BusStats {
    crashes: usize,
    respawns: u32,
    /// New loops drawn after leaving one, over all buses.
    reroutes: u32,
    /// Connectors taken along their loops, stops served and passed by, and the shortest and
    /// longest stand at a stop (s).
    taken: u32,
    served: u32,
    missed: u32,
    dwell: [f64; 2],
    /// Stops per km of loop, over all loops drawn at the start.
    stops_per_km: f64,
    /// Shortest and longest loop drawn at the start (m).
    length: [f64; 2],
}

fn run_buses(seed: u64, cars: usize, buses: usize, minutes: f64) -> BusStats {
    let mut w = WorldInstance::new(compile(&buses_toml(seed, cars, buses)), Seed::from_u64(seed));
    let mut st = BusStats { dwell: [f64::INFINITY, 0.0], length: [f64::INFINITY, 0.0], ..Default::default() };
    let (mut stops, mut km) = (0, 0.0);
    for i in 0..buses {
        let r = traffic(&w, i).bus.as_ref().expect("a loop");
        stops += r.stops.iter().flatten().count();
        km += r.length / 1000.0;
        st.length = [st.length[0].min(r.length), st.length[1].max(r.length)];
    }
    st.stops_per_km = stops as f64 / km;
    // Per bus as of the step before: its stand, and its counts (connectors taken, stops served
    // and missed, reroutes).
    let mut last = vec![(None::<(f64, f64)>, [0u32; 4]); buses];
    let steps = (25.0 * 60.0 * minutes) as usize;
    for _ in 0..steps {
        w.step();
        st.crashes += (0..buses + cars).filter(|&i| w.agent(i).events.contains(Events::CRASH_AGENT)).count();
        for (i, (stand, before)) in last.iter_mut().enumerate() {
            let d = traffic(&w, i);
            let Some(r) = d.bus.as_ref() else { continue };
            let now = [r.taken, r.stops_served, r.stops_missed, d.reroutes];
            // A stand ended (at the stop served): how long it was.
            if let (Some((stood, _)), None) = (*stand, r.dwelling)
                && now[1] == before[1] + 1
            {
                st.dwell = [st.dwell[0].min(stood), st.dwell[1].max(stood)];
            }
            st.reroutes += now[3].saturating_sub(before[3]);
            // (A respawn draws a new loop: its counts start again.)
            if now[..3].iter().zip(&before[..3]).all(|(n, b)| n >= b) {
                st.taken += now[0] - before[0];
                st.served += now[1] - before[1];
                st.missed += now[2] - before[2];
            }
            *stand = r.dwelling;
            *before = now;
        }
    }
    st.respawns = (0..buses + cars).map(|i| traffic(&w, i).respawns).sum();
    st
}

#[test]
fn buses_keep_their_loops_and_dwell_at_stops() {
    for seed in [1, 2] {
        let st = run_buses(seed, 30, 4, 10.0);
        eprintln!("seed {seed}: {st:?}");
        assert_eq!(st.crashes, 0, "seed {seed}");
        assert!(st.reroutes <= 1, "seed {seed}: {st:?}");
        assert!(st.taken > 40, "seed {seed}: {st:?}");
        assert!(st.served > 8, "seed {seed}: {st:?}");
        // (Only after leaving the loop.)
        assert!(st.missed <= st.reroutes, "seed {seed}: {st:?}");
        // Stands within the drawn range (a policy step's rounding over).
        assert!(st.dwell[0] >= 10.0 && st.dwell[1] <= 30.0 + 0.05, "seed {seed}: {st:?}");
        assert!(st.stops_per_km > 1.5 && st.stops_per_km < 4.0, "seed {seed}: {st:?}");
        assert!(st.length[0] > 1000.0 && st.length[1] < 4000.0, "seed {seed}: {st:?}");
    }
}
