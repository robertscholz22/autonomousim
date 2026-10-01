//! Pedestrians (M8c step 2): lanes form in bidirectional corridor flow under the social
//! force, free walking speeds match the drawn desired speeds, and on urban maps with NPC
//! traffic no pedestrian is hit and none crosses on red unless jaywalking.

use autonomousim_core::rng::Seed;
use autonomousim_sim::events::Events;
use autonomousim_sim::pedestrians::{PedState, SocialForce};
use autonomousim_sim::{CompiledScenario, Scenario, WorldInstance};
use glam::DVec2;
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

/// Lane order of a corridor: in lateral strips of 0.5 m, the share of pedestrians that walk
/// the strip's majority direction minus the minority, over all.
fn lane_order(peds: &[(DVec2, f64)], width: f64) -> f64 {
    let strips = (width / 0.5).round() as usize;
    let mut count = vec![(0i32, 0i32); strips];
    for &(p, dir) in peds {
        let k = (((p.y + 0.5 * width) / 0.5).floor().max(0.0) as usize).min(strips - 1);
        if dir > 0.0 {
            count[k].0 += 1;
        } else {
            count[k].1 += 1;
        }
    }
    count.iter().map(|&(a, b)| (a - b).abs()).sum::<i32>() as f64 / peds.len() as f64
}

/// Bidirectional flow in a periodic corridor 40 m long and 4 m wide: lanes form.
#[test]
fn bidirectional_corridor_flow_forms_lanes() {
    let sf = SocialForce::default();
    let (length, width, radius, dt) = (40.0, 4.0, 0.25, 0.04);
    let mut rng = Seed::from_u64(7).rng();
    let n = 40;
    // (position, velocity, desired speed, direction)
    let mut peds: Vec<(DVec2, DVec2, f64, f64)> = Vec::new();
    while peds.len() < n {
        let p = DVec2::new(rng.uniform() * length, (rng.uniform() - 0.5) * (width - 2.0 * radius));
        if peds.iter().any(|q| q.0.distance(p) < 0.6) {
            continue;
        }
        let dir = if peds.len().is_multiple_of(2) { 1.0 } else { -1.0 };
        let v0 = (1.34 + 0.26 * rng.normal()).clamp(0.5, 2.5);
        peds.push((p, DVec2::ZERO, v0, dir));
    }
    let order =
        |peds: &[(DVec2, DVec2, f64, f64)]| lane_order(&peds.iter().map(|p| (p.0, p.3)).collect::<Vec<_>>(), width);
    let start = order(&peds);
    let (mut late, mut samples, mut flow) = (0.0, 0, 0.0);
    let steps = (120.0 / dt) as usize;
    for step in 0..steps {
        let snap = peds.clone();
        for (i, p) in peds.iter_mut().enumerate() {
            let mut a = sf.driving(p.1, DVec2::new(p.3 * p.2, 0.0));
            for (j, q) in snap.iter().enumerate() {
                if i == j {
                    continue;
                }
                // The nearest periodic image.
                let mut d = q.0 - p.0;
                d.x -= length * (d.x / length).round();
                a += sf.interaction(p.0, p.1, p.0 + d, q.1);
            }
            p.1 += a * dt;
            if p.1.length() > 1.3 * p.2 {
                p.1 *= 1.3 * p.2 / p.1.length();
            }
        }
        for p in &mut peds {
            p.0 += p.1 * dt;
            p.0.x = p.0.x.rem_euclid(length);
            let half = 0.5 * width - radius;
            if p.0.y.abs() > half {
                p.0.y = half * p.0.y.signum();
                p.1.y = 0.0;
            }
        }
        if step >= steps - (30.0 / dt) as usize {
            late += order(&peds);
            samples += 1;
            flow += peds.iter().map(|p| p.1.x * p.3).sum::<f64>() / n as f64;
        }
    }
    let late = late / samples as f64;
    let flow = flow / samples as f64;
    eprintln!("lane order: start {start:.2}, last 30 s {late:.2}; speed along {flow:.2} m/s");
    assert!(late > 0.8 && late > start + 0.3, "lane order {start:.2} → {late:.2}");
    assert!(flow > 1.0, "speed along the walking direction {flow:.2} m/s");
}

fn urban(seed: u64, npcs: usize, peds: usize, extra: &str) -> String {
    format!(
        r#"
        physics_hz = 200
        policy_hz = 25
        map = {{ type = "urban", seed = {seed}, count = 1 }}
        [pedestrians]
        count = {peds}
        {extra}
        [[groups]]
        name = "npc"
        count = {npcs}
        vehicle = "sedan_like"
        physics = "kinematic"
        driver = {{ type = "traffic" }}
        spawn = {{ on_ground = true, on_road = true, min_separation = 20.0 }}
        disable_on_terminal = false
        [[groups]]
        name = "parked"
        count = 20
        vehicle = "sedan_like"
        physics = "kinematic"
        driver = {{ type = "parked" }}
        spawn = {{ on_ground = true, in_bays = true }}
        disable_on_terminal = false
        "#
    )
}

#[derive(Debug, Default)]
struct Run {
    hits: u64,
    red_starts: u64,
    crossings: u64,
    arrivals: u64,
    respawns: u64,
    crashes: u64,
    red_lights: u64,
    /// Ratio of free walking speed to desired speed (mean), and the desired speeds' mean and
    /// standard deviation.
    free_ratio: f64,
    desired: (f64, f64),
}

/// `minutes` of `npcs` NPC cars, 20 parked and `peds` pedestrians on urban map `seed`.
fn run(seed: u64, npcs: usize, peds: usize, minutes: f64, extra: &str) -> Run {
    let mut w = WorldInstance::new(compile(&urban(seed, npcs, peds, extra)), Seed::from_u64(seed));
    let mut r = Run::default();
    let (mut ratio, mut free) = (0.0, 0usize);
    for step in 0..(25.0 * 60.0 * minutes) as usize {
        w.step();
        for i in 0..npcs + 20 {
            let e = w.agent(i).events;
            r.crashes += u64::from(e.contains(Events::CRASH_AGENT));
            r.red_lights += u64::from(e.contains(Events::RED_LIGHT));
        }
        // Free walking: on a walkway, no one within 3 m.
        if step % 25 == 0 {
            let c = w.crowd();
            for (i, p) in c.peds.iter().enumerate() {
                if p.state != PedState::Walking {
                    continue;
                }
                let alone = c.peds.iter().enumerate().all(|(j, q)| j == i || q.pos.distance(p.pos) > 3.0);
                if alone && p.vel.length() > 0.2 {
                    ratio += p.vel.length() / p.speed;
                    free += 1;
                }
            }
        }
    }
    let c = w.crowd();
    let st = c.stats;
    (r.hits, r.red_starts, r.crossings, r.arrivals, r.respawns) =
        (st.hits, st.red_starts, st.crossings, st.arrivals, st.respawns);
    r.free_ratio = ratio / free.max(1) as f64;
    let n = c.peds.len() as f64;
    let mean = c.peds.iter().map(|p| p.speed).sum::<f64>() / n;
    let var = c.peds.iter().map(|p| (p.speed - mean).powi(2)).sum::<f64>() / (n - 1.0);
    r.desired = (mean, var.sqrt());
    assert_eq!(c.peds.len(), peds, "the count holds");
    eprintln!("seed {seed}: {r:?}");
    r
}

fn check(seed: u64, r: &Run, minutes: f64) {
    assert_eq!(r.hits, 0, "seed {seed}: pedestrians hit");
    assert_eq!(r.red_starts, 0, "seed {seed}: crossings started on red");
    assert_eq!((r.crashes, r.red_lights), (0, 0), "seed {seed}: NPC crashes and red lights");
    assert!(r.crossings as f64 > 20.0 * minutes, "seed {seed}: {} crossings", r.crossings);
    assert!(r.arrivals as f64 > 5.0 * minutes, "seed {seed}: {} arrivals", r.arrivals);
}

#[test]
fn pedestrians_walk_with_traffic() {
    for seed in [1, 2] {
        let r = run(seed, 50, 200, 5.0, "");
        check(seed, &r, 5.0);
        // Free speeds are the desired speeds, drawn from 1.34 ± 0.26 m/s.
        assert!((r.free_ratio - 1.0).abs() < 0.05, "seed {seed}: free / desired speed {:.3}", r.free_ratio);
        assert!((r.desired.0 - 1.34).abs() < 0.06 && (r.desired.1 - 0.26).abs() < 0.05, "{:?}", r.desired);
    }
}

/// The step 2 acceptance run: 30 minutes on each of the first four training maps.
#[test]
#[ignore]
fn pedestrians_with_traffic_for_half_an_hour() {
    for seed in 1..=4 {
        let r = run(seed, 50, 200, 30.0, "");
        check(seed, &r, 30.0);
    }
}

#[test]
fn jaywalkers_cross_on_red_but_are_not_hit() {
    let r = run(3, 50, 200, 3.0, "jaywalk = 1.0");
    assert!(r.red_starts > 0, "no crossing on red");
    assert_eq!(r.hits, 0, "pedestrians hit");
}

#[test]
fn vehicles_hitting_pedestrians_raise_the_event() {
    // A pedestrian standing in an NPC's lane away from crossings is not seen by its driver.
    let mut w = WorldInstance::new(compile(&urban(1, 1, 1, "")), Seed::from_u64(1));
    for _ in 0..50 {
        w.step();
    }
    let car = w.agent(0).vehicle.pose();
    let ahead = car.pos.truncate() + car.rot.mul_vec3(glam::DVec3::X).truncate().normalize() * 12.0;
    w.crowd_mut().peds[0].stand_at(ahead, 100.0);
    let mut hit = false;
    for _ in 0..100 {
        w.step();
        hit |= w.agent(0).events.contains(Events::PEDESTRIAN_HIT);
    }
    assert!(hit, "no hit event");
    assert_eq!(w.crowd().stats.hits, 1);
    assert_eq!(w.crowd().peds[0].state, PedState::Hit);
}

#[test]
fn crowds_repeat() {
    let sc = compile(&urban(2, 10, 100, ""));
    let mut a = WorldInstance::new(sc.clone(), Seed::from_u64(5));
    let mut b = WorldInstance::new(sc, Seed::from_u64(5));
    for _ in 0..250 {
        a.step();
        b.step();
    }
    assert_eq!(a.state_hash(), b.state_hash());
    let snap = a.snapshot();
    for _ in 0..50 {
        a.step();
    }
    let h = a.state_hash();
    a.restore(&snap);
    for _ in 0..50 {
        a.step();
    }
    assert_eq!(a.state_hash(), h);
}

/// A drone over an urban map with a level LiDAR beam, and one pedestrian.
const SEEN: &str = r#"
    physics_hz = 500
    policy_hz = 50
    map = { type = "urban", seed = 1, count = 1 }
    [pedestrians]
    count = 1
    [[groups]]
    name = "drone"
    vehicle = "iris_like"
    action_mode = "velocity"
    spawn = { agl = [1.0, 1.0] }
    sensors = [
        { name = "lidar", type = "lidar", noise = 0.0, pattern = { type = "rings", elevations = [0.0], azimuths = 1, azimuth_fov = 360.0 } },
    ]
"#;

/// The drone 5 m west of the pedestrian, facing it, 1 m over the ground.
fn face_pedestrian(w: &mut WorldInstance) {
    let p = w.crowd().peds[0].clone();
    let at = p.pos - DVec2::new(5.0, 0.0);
    let pose = autonomousim_core::math::Pose::new(at.extend(p.z + 1.0), glam::DQuat::IDENTITY);
    w.place_agent(0, pose, glam::DVec3::ZERO, glam::DVec3::ZERO);
    let pos = p.pos;
    w.crowd_mut().peds[0].stand_at(pos, 100.0);
}

#[test]
fn lidar_sees_pedestrians() {
    let mut w = WorldInstance::new(compile(SEEN), Seed::from_u64(1));
    face_pedestrian(&mut w);
    for _ in 0..6 {
        w.step();
    }
    let scan = match &w.agent(0).sensors[0] {
        autonomousim_sensors::Sensor::Lidar(l) => l.latest().unwrap().clone(),
        _ => unreachable!(),
    };
    assert_eq!(scan.kinds[0], autonomousim_sensors::lidar::ReturnKind::Pedestrian);
    let gap = w.crowd().peds[0].pos.distance(w.agent(0).vehicle.position().truncate()) - w.crowd().peds[0].radius;
    assert!((f64::from(scan.ranges[0]) - gap).abs() < 0.1, "range {} for {gap:.2}", scan.ranges[0]);
}

/// A car on an urban map that sees the nearest pedestrians, and the road users near it.
const OBSERVED: &str = r#"
    physics_hz = 200
    policy_hz = 25
    map = { type = "urban", seed = 1, count = 1 }
    [pedestrians]
    count = 3
    [[groups]]
    name = "ego"
    vehicle = "sedan_like"
    physics = "kinematic"
    action_mode = "vk"
    spawn = { on_ground = true, on_road = true }
    obs = [{ term = "pedestrians", count = 3 }, { term = "traffic", count = 2 }]
    disable_on_terminal = false
"#;

/// The `pedestrians` term sees the pedestrians within its range, nearest first; `traffic` only
/// those on the carriageway (here: one standing there, hit), as pedestrians.
#[test]
fn pedestrians_and_traffic_terms_see_pedestrians() {
    let sc = compile(OBSERVED);
    assert_eq!(sc.groups[0].obs_dim(), 3 * 6 + 2 * 15);
    let mut w = WorldInstance::new(sc, Seed::from_u64(3));
    let pose = w.agent(0).vehicle.pose();
    let (me, yaw) = (pose.pos.truncate(), autonomousim_core::math::quat::yaw(pose.rot));
    let at = |x: f64, y: f64| me + DVec2::from_angle(yaw).rotate(DVec2::new(x, y));
    let on_road = |p: DVec2| w.map().roads().on_road(p).is_some();
    let (hit, near, far) = (at(6.0, 0.5), at(12.0, -3.0), at(40.0, 0.0));
    let flags = [f32::from(u8::from(on_road(hit))), f32::from(u8::from(on_road(near)))];
    let peds = &mut w.crowd_mut().peds;
    peds[0].stand_at(near, 100.0);
    peds[1].stand_at(far, 100.0);
    peds[2].stand_at(hit, 100.0);
    peds[2].state = PedState::Hit;
    let mut o = vec![0.0f32; 48];
    w.observe(0, &mut o);
    // Nearest first: the one hit, then the one 12 m ahead; the far one is out of range.
    assert!((o[0] - 6.0).abs() < 1e-4 && (o[1] - 0.5).abs() < 1e-4, "{o:?}");
    assert_eq!(&o[2..6], &[0.0, 0.0, flags[0], 1.0]);
    assert!((o[6] - 12.0).abs() < 1e-4 && (o[7] + 3.0).abs() < 1e-4, "{o:?}");
    assert_eq!(&o[8..12], &[0.0, 0.0, flags[1], 1.0]);
    assert!(o[12..18].iter().all(|&v| v == 0.0), "{o:?}");
    // `traffic`: the one hit only, a pedestrian in no lane.
    let t = &o[18..];
    assert!((t[0] - 6.0).abs() < 1e-4 && (t[1] - 0.5).abs() < 1e-4, "{t:?}");
    assert_eq!(&t[8..15], &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    assert!(t[15..].iter().all(|&v| v == 0.0), "{t:?}");
}

/// The crowd is recorded to `/pedestrians` at its rate (and at resets), rounded to cm and mrad.
#[test]
fn crowds_are_recorded() {
    use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
    let sc = compile(&urban(2, 2, 50, ""));
    let path = std::env::temp_dir().join(format!("autonomousim-crowd-{}.mcap", std::process::id()));
    let mut rec = Recorder::create(&path, RecorderConfig { pedestrian_hz: 5, ..Default::default() }).unwrap();
    let mut w = WorldInstance::new(sc, Seed::from_u64(4));
    rec.on_reset(&w);
    let mut live = vec![(w.time(), w.crowd().peds.clone())];
    for _ in 0..50 {
        w.step_with(&mut |w| {
            rec.on_tick(w);
            if w.clock().tick.is_multiple_of(40) {
                live.push((w.time(), w.crowd().peds.clone()));
            }
        });
    }
    rec.finish().unwrap();
    let recording = Recording::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(recording.pedestrian_hz, 5);
    let frames = &recording.episodes[0].pedestrians;
    assert_eq!(frames.len(), live.len());
    for (f, (time, peds)) in frames.iter().zip(&live) {
        assert_eq!(f.time, *time);
        assert_eq!(f.len(), 50);
        for (i, p) in peds.iter().enumerate() {
            let r = f.get(i).unwrap();
            assert!((r.position - p.pos.extend(p.z)).abs().max_element() <= 0.005 + 1e-9);
            assert!((r.velocity - p.vel).abs().max_element() <= 0.005 + 1e-9);
            assert!((r.heading - p.heading).abs() <= 5e-4 + 1e-12 && (r.height - p.height).abs() <= 0.005 + 1e-9);
            assert_eq!(r.state, p.state);
        }
    }
    assert!(frames.first().unwrap().position != frames.last().unwrap().position, "the crowd moved");
}
