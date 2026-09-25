//! Ground vehicles with trailers in the simulation: spawning, articulation state and terms,
//! jackknife events, contacts on trailers and recordings.

use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::yaw;
use autonomousim_core::rng::Seed;
use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, CompiledScenario, Events, STATE_DIM, STATE_FIELDS, Scenario, WorldInstance};
use autonomousim_vehicles::ground::Wheeled;
use glam::DVec3;
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

fn rig(trailer: &str, extra: &str) -> Arc<CompiledScenario> {
    let (vehicle, trailer) = match trailer {
        "semitrailer_3axle" => ("truck_6x4", trailer),
        _ => ("farm_tractor", trailer),
    };
    compile(&format!(
        r#"
        name = "rig"
        map = {{ type = "testworld", kind = "flat", size = 400.0 }}
        {extra}
        [[groups]]
        name = "rig"
        vehicle = "{vehicle}"
        trailers = ["{trailer}"]
        action_mode = "vk"
        obs = [ {{ term = "articulation" }}, {{ term = "trailer_goal" }}, {{ term = "speed" }} ]
        "#
    ))
}

fn state(w: &WorldInstance) -> Vec<f64> {
    let mut s = vec![0.0; STATE_DIM];
    w.write_state(0, &mut s);
    s
}

fn field(name: &str) -> usize {
    STATE_FIELDS.iter().take_while(|(n, _)| *n != name).map(|(_, d)| d).sum()
}

fn wheeled(w: &WorldInstance) -> &Wheeled {
    w.agent(0).vehicle.as_wheeled().unwrap()
}

fn drive(w: &mut WorldInstance, speed: f64, curvature: f64, seconds: f64) {
    w.set_command(0, GroundSetpoint::SpeedCurvature { speed, curvature });
    for _ in 0..(seconds / w.scenario().policy_dt()).round() as usize {
        w.step();
        let e = w.agent(0).events;
        assert!(!e.is_terminal(), "{e:?}");
    }
}

#[test]
fn rigs_spawn_drive_and_articulate() {
    for trailer in ["semitrailer_3axle", "farm_trailer"] {
        let sc = rig(trailer, "");
        let g = &sc.groups[0];
        let d = g.def.as_wheeled().unwrap();
        assert_eq!(d.units.len(), if trailer == "farm_trailer" { 3 } else { 1 });
        // The trailer counts for the spawn clearance.
        assert!(g.radius > 8.0, "{trailer}: radius {}", g.radius);
        let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(4));
        drive(&mut w, 0.0, 0.0, 2.0);
        let s = state(&w);
        let (art, tail) = (field("articulation"), field("tail"));
        assert!(s[art].abs() < 0.01 && s[art + 1].abs() < 0.01, "{trailer}: {:?}", &s[art..art + 2]);
        // The tail is behind the tractor, in line.
        let v = &w.agent(0).vehicle;
        let back = v.pose().inverse_transform_point(DVec3::new(s[tail], s[tail + 1], v.position().z));
        assert!(back.x < -6.0 && back.y.abs() < 0.1, "{trailer}: tail {back:?}");
        assert!((s[tail + 2] - yaw(v.orientation())).abs() < 0.01);

        drive(&mut w, 3.0, 0.08, 15.0);
        let s = state(&w);
        let w0 = wheeled(&w);
        let arts: Vec<(f64, f64)> = w0.articulations().collect();
        assert!(s[art] < -0.1, "{trailer}: articulation {}", s[art]);
        assert_eq!(s[art], arts[0].0);
        if trailer == "farm_trailer" {
            // Drawbar and body both follow the turn.
            assert!(s[art + 1] < -0.05 && s[art + 1] == arts[1].0, "{:?}", &s[art..art + 2]);
        } else {
            assert_eq!(s[art + 1], 0.0);
        }
        // The terms read the same angles, and the goal relative to the tail.
        let mut obs = vec![0.0; g.obs_dim()];
        w.observe(0, &mut obs);
        assert_eq!(obs.len(), 9);
        assert_eq!(obs[0], arts[0].0 as f32);
        assert_eq!(obs[2], arts[0].1 as f32);
        let tp = w0.tail_pose();
        let goal = w.agent(0).goal();
        let rel = glam::DQuat::from_rotation_z(yaw(tp.rot)).inverse() * (goal.position - tp.pos);
        assert!((f64::from(obs[4]) - rel.x).abs() < 1e-3 && (f64::from(obs[5]) - rel.y).abs() < 1e-3);
    }
}

#[test]
fn a_tight_turn_jackknifes() {
    let sc = rig("semitrailer_3axle", "events = { ground = { jackknife_deg = 15.0 } }");
    let mut w = WorldInstance::new(sc, Seed::from_u64(1));
    w.set_command(0, GroundSetpoint::SpeedCurvature { speed: 3.0, curvature: 0.12 });
    let mut hit = false;
    for _ in 0..(30.0 / w.scenario().policy_dt()) as usize {
        w.step();
        let e = w.agent(0).events;
        if e.contains(Events::JACKKNIFE) {
            assert!(e.is_terminal() && w.agent(0).disabled, "{e:?}");
            hit = true;
            break;
        }
    }
    assert!(hit, "no jackknife, articulation {:?}", wheeled(&w).articulation(1));
    assert!(wheeled(&w).articulation(1).0.abs() > 15f64.to_radians());
}

/// A car parked against the trailer's wheels pushes the trailer, not the tractor.
#[test]
fn contacts_act_on_the_trailer() {
    let sc = compile(
        r#"
        name = "rig_and_car"
        map = { type = "testworld", kind = "flat", size = 300.0 }
        [[groups]]
        name = "rig"
        vehicle = "truck_6x4"
        trailers = ["semitrailer_3axle"]
        disable_on_terminal = false
        [[groups]]
        name = "car"
        vehicle = "offroad_4x4"
        disable_on_terminal = false
        spawn = { min_separation = 40.0 }
        "#,
    );
    let mut w = WorldInstance::new(sc, Seed::from_u64(2));
    for k in 0..2 {
        w.set_command(k, GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 });
    }
    for _ in 0..25 {
        w.step();
    }
    let trailer_link = wheeled(&w).unit_link(1) as u16;
    let rot = wheeled(&w).unit_pose(1).rot;
    // The trailer's rightmost wheel sphere and the car's left rear one, side by side.
    let t = w.shapes()[0].spheres.iter().filter(|s| s.gear && s.body == 1).copied();
    let t = t.min_by(|a, b| (rot.inverse() * a.center).y.total_cmp(&(rot.inverse() * b.center).y)).unwrap();
    let car = w.agent(1).vehicle.pose();
    let c = w.shapes()[1].spheres.iter().filter(|s| s.gear).copied();
    let c = c.max_by(|a, b| {
        let (pa, pb) = (car.inverse_transform_point(a.center), car.inverse_transform_point(b.center));
        (pa.y - pa.x).total_cmp(&(pb.y - pb.x))
    });
    let c = c.unwrap();
    let local = car.inverse_transform_point(c.center);
    let dz = c.center.z - t.center.z;
    let lateral = ((t.radius + c.radius - 0.02).powi(2) - dz * dz).sqrt();
    let target = t.center + rot * DVec3::new(0.0, -lateral, dz);
    let pos = target - rot * local;
    w.place_agent(1, Pose::new(DVec3::new(pos.x, pos.y, car.pos.z), rot), DVec3::ZERO, DVec3::ZERO);
    let mut seen = false;
    w.step_with(&mut |w: &WorldInstance| {
        let c = w.agent_contacts();
        let sum = |k: usize| c[k].forces.iter().fold(DVec3::ZERO, |f, (fk, _, _)| f + *fk);
        assert_eq!(sum(0), -sum(1));
        seen |= !c[0].forces.is_empty();
        for &(_, _, link) in &c[0].forces {
            assert_eq!(link, trailer_link);
        }
        assert!(!c[0].crashed);
    });
    assert!(seen, "no contact");
}

#[test]
fn recordings_carry_the_joints() {
    let sc = rig("farm_trailer", "");
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("trailer.mcap");
    let mut b = BatchSim::from_compiled(sc.clone(), 1, 3, 1).unwrap();
    b.attach_recorder(0, Recorder::create(&path, RecorderConfig { state_hz: 50, lidar: false }).unwrap());
    for _ in 0..100 {
        b.step(&[&[0.3, 0.6]]);
    }
    b.detach_recorder(0).unwrap().finish().unwrap();
    let rec = Recording::read(&path).unwrap();
    let last = rec.episodes[0].states[0].last().unwrap();
    let live = b.world(0).agent(0).vehicle.as_wheeled().unwrap();
    assert_eq!(last.joints, live.joints());
    assert!(live.articulation(1).0.abs() > 0.01, "{:?}", live.articulation(1));
    // Shown from the record, every unit sits where it was.
    let mut shown = live.clone();
    let init = autonomousim_vehicles::ground::WheeledInit {
        pose: Pose::new(last.position, last.orientation),
        lin_vel_world: last.velocity,
        ang_vel_body: last.rates,
    };
    shown.show(&init, &last.joints, last.steering, &[], live.powertrain());
    for u in 1..live.num_units() {
        assert_eq!(shown.articulation(u).0, live.articulation(u).0);
    }
}

/// Bay goals: the rig spawns in a farm yard in line, its tail the sampled distance ahead of
/// the bay; it settles without events; reversing straight back first brings the tail toward the bay.
#[test]
fn bay_goals_put_the_rig_in_a_yard() {
    let sc = compile(
        r#"
        name = "bay"
        map = { type = "rural", seed = 0, count = 3, preset = "training" }
        [[groups]]
        name = "rig"
        vehicle = "truck_6x4"
        trailers = ["semitrailer_3axle"]
        action_mode = "vk"
        goals = { kind = "bay", distance = [12.0, 24.0] }
        obs = [ { term = "trailer_goal" } ]
        "#,
    );
    let tail = field("tail");
    for seed in 0..6 {
        let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(seed));
        let yards = autonomousim_sim::bay::yards(w.map());
        let goal = w.agent(0).goal();
        // The bay lies in a yard, behind its centre.
        let yard = yards.iter().find(|y| (y.centre.truncate() - goal.position.truncate()).length() < 20.0).unwrap();
        let local = glam::DVec2::from_angle(-yard.heading).rotate(goal.position.truncate() - yard.centre.truncate());
        assert!((local.x + 16.0).abs() < 1e-9 && local.y.abs() <= 4.0, "{local}");
        assert!((goal.yaw - yard.heading).abs() < 1e-12);
        drive(&mut w, 0.0, 0.0, 2.0);
        let s = state(&w);
        let rel = glam::DVec2::from_angle(-goal.yaw)
            .rotate(glam::DVec2::new(s[tail], s[tail + 1]) - goal.position.truncate());
        assert!(rel.x > 11.5 && rel.x < 24.5 && rel.y.abs() < 2.2, "seed {seed}: tail at {rel} from the bay");
        let heading = autonomousim_core::math::quat::wrap_angle(s[tail + 2] - goal.yaw);
        assert!(heading.abs() < 10.5f64.to_radians(), "{heading}");
        let before = rel.length();
        // (Straight back only for a few metres: reversing a trailer is unstable.)
        drive(&mut w, -1.5, 0.0, 4.0);
        let s = state(&w);
        let after = (glam::DVec2::new(s[tail], s[tail + 1]) - goal.position.truncate()).length();
        assert!(after < before - 3.0, "seed {seed}: {before} → {after}");
    }
    // Bay goals need ground vehicles and yards.
    let bad = |vehicle: &str, map: &str| {
        Scenario::from_toml(&format!(
            "name = \"x\"\nmap = {map}\n[[groups]]\nname = \"g\"\nvehicle = \"{vehicle}\"\ngoals = {{ kind = \"bay\" }}"
        ))
        .unwrap()
        .compile()
        .unwrap_err()
        .to_string()
    };
    assert!(bad("cf2x", "{ type = \"testworld\", kind = \"flat\", size = 100.0 }").contains("ground vehicle"));
    assert!(bad("sedan_like", "{ type = \"testworld\", kind = \"flat\", size = 100.0 }").contains("farm yards"));
}

/// Episodes with bay goals are drawn from the maps with farm yards only.
#[test]
fn bay_episodes_skip_maps_without_yards() {
    let sc = compile(
        r#"
        name = "bay"
        map = { type = "rural", seed = 1000, count = 9, preset = "training" }
        [[groups]]
        name = "rig"
        vehicle = "truck_6x4"
        trailers = ["semitrailer_3axle"]
        action_mode = "vk"
        goals = { kind = "bay" }
        "#,
    );
    // Map 8 of this pool has no farm.
    assert!(autonomousim_sim::bay::yards(&sc.maps[8]).is_empty());
    assert_eq!(sc.episode_maps, (0..8).collect::<Vec<_>>());
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(0));
    let mut seen = [false; 9];
    for seed in 0..40 {
        w.reset(Some(seed));
        seen[w.map_index()] = true;
    }
    assert!(!seen[8] && seen.iter().filter(|s| **s).count() >= 6, "{seen:?}");
}

/// A LiDAR on the trailer's rear scans from the trailer, whatever the articulation; inertial
/// sensors and missing units are rejected.
#[test]
fn sensors_ride_on_trailers() {
    let mut spec = Scenario::from_toml(
        r#"
        name = "rig"
        map = { type = "testworld", kind = "flat", size = 400.0 }
        [[groups]]
        name = "rig"
        vehicle = "truck_6x4"
        trailers = ["semitrailer_3axle"]
        action_mode = "vk"
        [[groups.sensors]]
        name = "rear"
        type = "lidar"
        unit = 1
        max_range = 30.0
        mount = { position = [-11.8, 0.0, 1.0], rotation = [0.0, 0.0, 3.141592653589793] }
        pattern = { type = "rings", elevations = [0.0], azimuths = 8, azimuth_fov = 90.0 }
        "#,
    )
    .unwrap();
    let sc = Arc::new(spec.clone().compile().unwrap());
    let mut w = WorldInstance::new(sc, Seed::from_u64(4));
    drive(&mut w, 3.0, 0.08, 10.0);
    let v = wheeled(&w);
    assert!(v.articulation(1).0 < -0.1);
    let scan = w.agent(0).sensors.iter().find_map(|s| match s {
        autonomousim_sensors::Sensor::Lidar(l) => l.latest().cloned(),
        _ => None,
    });
    let scan = scan.unwrap();
    // Scanned at most a LiDAR period ago from the trailer's rear, looking back.
    let trailer = v.unit_pose(1);
    let want = trailer.transform_point(DVec3::new(-11.8, 0.0, 1.0));
    assert!((scan.pose.pos - want).length() < 0.5, "{} vs {want}", scan.pose.pos);
    let back = scan.pose.rot * DVec3::X;
    assert!(back.dot(trailer.rot * -DVec3::X) > 0.99, "{back}");
    // An IMU on the trailer, or a LiDAR on a unit that does not exist, is rejected.
    spec.groups[0].sensors[0].unit = 2;
    assert!(spec.clone().compile().unwrap_err().to_string().contains("units"));
    spec.groups[0].sensors[0].unit = 1;
    spec.groups[0].sensors[0].config =
        autonomousim_sensors::SensorConfig::Imu(autonomousim_sensors::ImuConfig::ideal());
    assert!(spec.compile().unwrap_err().to_string().contains("rangefinders and LiDARs"));
}
