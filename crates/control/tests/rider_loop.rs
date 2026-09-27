//! Closed-loop checks of the rider controller on the two-wheeler presets over flat asphalt:
//! holding a straight line after a sideways push, curvature steps, the initial countersteer,
//! launching from rest and stopping on the feet, and riding at walking pace.

use autonomousim_control::ground::*;
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

struct Rig {
    v: Wheeled,
    ctrl: GroundController,
    terrain: FlatTerrain,
    materials: MaterialTable,
    /// Sideways force at the centre of mass (N, world frame) during the next steps.
    push: DVec3,
    t: f64,
}

/// What a run records every step.
#[derive(Clone, Copy, Debug)]
struct Sample {
    t: f64,
    speed: f64,
    lean: f64,
    steer: f64,
    /// Path curvature (1/m, positive left) from the yaw rate about the vertical.
    curvature: f64,
    heading: f64,
    feet: bool,
}

impl Rig {
    /// On flat asphalt, heading east at `speed`, default controller.
    fn new(name: &str, speed: f64) -> Self {
        let def = Arc::new(presets::wheeled(name).unwrap());
        let mut v = Wheeled::new(def.clone(), DT);
        let init = v.rest(DVec3::ZERO, 0.0, speed);
        v.reset(&init);
        let ctrl = GroundController::new(&def, DT, &GroundConfig::default()).unwrap();
        assert!(ctrl.is_single_track());
        Self {
            v,
            ctrl,
            terrain: FlatTerrain::new(0.0, MaterialId::ASPHALT),
            materials: MaterialTable::standard(),
            push: DVec3::ZERO,
            t: 0.0,
        }
    }

    fn sample(&self) -> Sample {
        let v = &self.v;
        let (yaw, _, roll) = v.orientation().to_euler(glam::EulerRot::ZYX);
        let omega = (v.orientation() * v.ang_vel_body()).z;
        let speed = v.speed();
        Sample {
            t: self.t,
            speed,
            lean: roll,
            steer: v.steering_head().unwrap().0,
            curvature: if speed > 0.5 { omega / speed } else { 0.0 },
            heading: yaw,
            feet: v.feet_down(),
        }
    }

    fn step(&mut self, sp: &GroundSetpoint) {
        let input = self.ctrl.update(sp, &GroundEstimate::of(&self.v));
        let scene = StaticScene { terrain: &self.terrain, obstacles: &NoObstacles, materials: &self.materials };
        let v = &mut self.v;
        v.begin_step();
        v.apply_drive(&input, &AirData::default());
        v.apply_tires(&scene);
        v.apply_contacts(&scene);
        if self.push != DVec3::ZERO {
            let com = v.pose().transform_point(v.def().total_com());
            v.apply_force(self.push, com);
        }
        v.finish_step(DVec3::new(0.0, 0.0, -G)).unwrap();
        self.t += DT;
    }

    fn run(&mut self, sp: &GroundSetpoint, duration: f64) -> Vec<Sample> {
        (0..(duration / DT).round() as usize)
            .map(|_| {
                self.step(sp);
                self.sample()
            })
            .collect()
    }
}

fn vk(speed: f64, curvature: f64) -> GroundSetpoint {
    GroundSetpoint::SpeedCurvature { speed, curvature }
}

fn worst(samples: &[Sample], f: impl Fn(&Sample) -> f64) -> f64 {
    samples.iter().map(|s| f(s).abs()).fold(0.0, f64::max)
}

fn mean(samples: &[Sample], f: impl Fn(&Sample) -> f64) -> f64 {
    samples.iter().map(f).sum::<f64>() / samples.len() as f64
}

/// Riding straight at 3, 10 and 25 m/s (the bicycle at 3 and 8), a sideways shove at the
/// centre of mass (0.5 m/s worth of the whole mass over 0.1 s): back upright and going
/// straight within 6 s, without falling.
#[test]
fn straight_line_hold_under_a_lateral_impulse() {
    for (name, speeds) in [("motorcycle_sport", &[3.0, 10.0, 25.0][..]), ("bicycle_city", &[3.0, 8.0][..])] {
        for &speed in speeds {
            let mut rig = Rig::new(name, speed);
            let sp = vk(speed, 0.0);
            rig.run(&sp, 2.0);
            let mass = rig.v.def().total_mass();
            rig.push = DVec3::new(0.0, -mass * 0.5 / 0.1, 0.0);
            let mut trace = rig.run(&sp, 0.1);
            rig.push = DVec3::ZERO;
            trace.extend(rig.run(&sp, 7.9));
            let tail = &trace[trace.len() - 2000..];
            let (max_lean, lean, curvature) =
                (worst(&trace, |s| s.lean), worst(tail, |s| s.lean), worst(tail, |s| s.curvature));
            let heading = trace.last().unwrap().heading;
            println!(
                "{name} at {speed} m/s: largest lean {max_lean:.3} rad, heading change {heading:.3} rad; \
                 last 2 s: lean {lean:.4} rad, curvature {curvature:.5} 1/m, speed {:.2} m/s",
                trace.last().unwrap().speed
            );
            assert!(max_lean < 0.4, "{name} at {speed} m/s: lean {max_lean}");
            assert!(lean < 0.01, "{name} at {speed} m/s: still leaning {lean}");
            assert!(curvature < 0.002, "{name} at {speed} m/s: still turning {curvature}");
            assert!((trace.last().unwrap().speed - speed).abs() < 0.2, "{name} at {speed} m/s: speed");
            assert!(trace.iter().all(|s| !s.feet), "{name} at {speed} m/s: feet down");
        }
    }
}

/// At 50 m/s the rider holds the bars loosely (the regulator's gains fade out above 25 m/s):
/// after the same shove the weave dies out within 15 s.
#[test]
fn high_speed_weave_dies_out() {
    let speed = 50.0;
    let mut rig = Rig::new("motorcycle_sport", speed);
    let sp = vk(speed, 0.0);
    rig.run(&sp, 2.0);
    rig.push = DVec3::new(0.0, -rig.v.def().total_mass() * 0.5 / 0.1, 0.0);
    let mut trace = rig.run(&sp, 0.1);
    rig.push = DVec3::ZERO;
    trace.extend(rig.run(&sp, 14.9));
    let tail = &trace[trace.len() - 2000..];
    let (max_lean, lean, steer) = (worst(&trace, |s| s.lean), worst(tail, |s| s.lean), worst(tail, |s| s.steer));
    println!("50 m/s: largest lean {max_lean:.3} rad; last 2 s: lean {lean:.4} rad, steer {steer:.4} rad");
    assert!(max_lean < 0.1 && lean < 0.01 && steer < 0.005, "lean {max_lean}, {lean}, steer {steer}");
}

/// Curvature steps into a turn, through to the opposite one and back to straight: each settles
/// within 5 % of the step (plus 0.0005 1/m) within 6 s, the lean near the steady turn's.
#[test]
fn curvature_steps_settle_without_falling() {
    for (name, speed, k) in [("motorcycle_sport", 15.0, 0.02), ("bicycle_city", 5.0, 0.05)] {
        let mut rig = Rig::new(name, speed);
        rig.run(&vk(speed, 0.0), 1.0);
        let mut from = 0.0;
        for target in [k, -k, 0.0] {
            let trace = rig.run(&vk(speed, target), 8.0);
            let tail = &trace[trace.len() - 2000..];
            let curvature = mean(tail, |s| s.curvature);
            let lean = mean(tail, |s| s.lean);
            let want = rig.ctrl.turn_lean(speed, target).unwrap();
            let max_lean = worst(&trace, |s| s.lean);
            println!(
                "{name} at {speed} m/s, {from} → {target} 1/m: curvature {curvature:.5}, lean {lean:.4} \
                 (steady turn {want:.4}), largest {max_lean:.3} rad"
            );
            let step: f64 = target - from;
            assert!(
                (curvature - target).abs() < 0.05 * step.abs() + 5e-4,
                "{name}: curvature {curvature} for {target}"
            );
            assert!((lean - want).abs() < 0.05, "{name}: lean {lean} vs {want}");
            assert!(max_lean < 0.7, "{name}: lean {max_lean}");
            assert!((tail.last().unwrap().speed - speed).abs() < 0.2, "{name}: speed");
            from = target;
        }
    }
}

/// Turning left from straight: the steering first turns right, which leans the vehicle left,
/// then follows into the turn.
#[test]
fn turns_begin_with_a_countersteer() {
    for (name, speed, k) in [("motorcycle_sport", 15.0, 0.02), ("bicycle_city", 5.0, 0.05)] {
        let mut rig = Rig::new(name, speed);
        rig.run(&vk(speed, 0.0), 1.0);
        let trace = rig.run(&vk(speed, k), 3.0);
        let first = trace.iter().find(|s| s.steer.abs() > 1e-3).unwrap();
        let right = trace.iter().map(|s| s.steer).fold(0.0, f64::min);
        let lean_left = trace.iter().position(|s| s.lean < -0.01).unwrap();
        let left = trace.iter().position(|s| s.steer > 1e-3).unwrap();
        println!(
            "{name}: steers right first ({:.4} rad at {:.3} s, down to {right:.4}), leans left from \
             {:.3} s, steers left from {:.3} s",
            first.steer,
            first.t - 1.0,
            trace[lean_left].t - 1.0,
            trace[left].t - 1.0
        );
        assert!(first.steer < 0.0, "{name}: first steers {}", first.steer);
        assert!(lean_left < left, "{name}: steered left before leaning left");
        assert!(trace.last().unwrap().lean < -0.01 && trace.last().unwrap().steer > 0.0, "{name}: not turning left");
    }
}

/// From standstill on the feet to riding: the feet come up and the rider balances; then a stop
/// back onto the feet, standing still.
#[test]
fn launch_from_rest_and_stop_on_the_feet() {
    for (name, speed) in [("motorcycle_sport", 10.0), ("bicycle_city", 5.0)] {
        let mut rig = Rig::new(name, 0.0);
        let rest = rig.run(&vk(0.0, 0.0), 1.0);
        assert!(rest.last().unwrap().feet, "{name}: feet up at rest");
        let launch = rig.run(&vk(speed, 0.0), 10.0);
        let lifted = launch.iter().position(|s| !s.feet).unwrap();
        let end = launch.last().unwrap();
        println!(
            "{name}: feet up at {:.2} s ({:.2} m/s), largest lean {:.3} rad; after 10 s {:.2} m/s, lean {:.4} rad, \
             heading {:.3} rad",
            launch[lifted].t - 1.0,
            launch[lifted].speed,
            worst(&launch, |s| s.lean),
            end.speed,
            end.lean,
            end.heading
        );
        assert!((end.speed - speed).abs() < 0.2, "{name}: speed {}", end.speed);
        assert!(end.lean.abs() < 0.01 && !end.feet, "{name}: lean {}", end.lean);
        assert!(worst(&launch, |s| s.lean) < 0.25, "{name}: lean in the launch");
        let stop = rig.run(&vk(0.0, 0.0), 8.0);
        let down = stop.iter().position(|s| s.feet).unwrap();
        let end = stop.last().unwrap();
        println!(
            "{name}: feet down at {:.2} m/s, stopped at lean {:.3} rad, largest lean {:.3} rad",
            stop[down].speed,
            end.lean,
            worst(&stop, |s| s.lean)
        );
        assert!(end.speed.abs() < 0.02 && end.feet, "{name}: speed {}", end.speed);
        assert!(worst(&stop, |s| s.lean) < 0.25, "{name}: lean in the stop");
    }
}

/// The bicycle at walking pace: balanced at 2 m/s (well below its self-stable range) after a
/// nudge, and crawling straight on its feet at 1.2 m/s.
#[test]
fn bicycle_at_walking_speed() {
    let mut rig = Rig::new("bicycle_city", 2.0);
    rig.push = DVec3::new(0.0, 100.0, 0.0);
    let mut trace = rig.run(&vk(2.0, 0.0), 0.1);
    rig.push = DVec3::ZERO;
    trace.extend(rig.run(&vk(2.0, 0.0), 14.9));
    let end = trace.last().unwrap();
    println!(
        "2 m/s: largest lean {:.3} rad, final lean {:.4} rad, speed {:.2} m/s, feet {}",
        worst(&trace, |s| s.lean),
        end.lean,
        end.speed,
        end.feet
    );
    assert!(trace.iter().all(|s| !s.feet), "feet down");
    assert!(end.lean.abs() < 0.01 && (end.speed - 2.0).abs() < 0.1, "lean {}, speed {}", end.lean, end.speed);
    let mut rig = Rig::new("bicycle_city", 0.0);
    let trace = rig.run(&vk(1.2, 0.0), 12.0);
    let end = trace.last().unwrap();
    println!(
        "1.2 m/s: speed {:.2} m/s, feet {}, lean {:.3} rad, heading {:.3} rad, sideways {:.3} m",
        end.speed,
        end.feet,
        end.lean,
        end.heading,
        rig.v.position().y
    );
    assert!(end.feet && (end.speed - 1.2).abs() < 0.1, "speed {}", end.speed);
    assert!(end.heading.abs() < 0.1 && rig.v.position().y.abs() < 0.5, "heading {}", end.heading);
}

/// `vw`: a yaw rate becomes the path curvature at the commanded speed. `raw`: the pedal only
/// brakes below zero, and the rider's lean passes through.
#[test]
fn yaw_rate_and_raw_inputs() {
    let mut rig = Rig::new("motorcycle_sport", 10.0);
    let trace = rig.run(&GroundSetpoint::SpeedYawRate { speed: 10.0, yaw_rate: 0.2 }, 8.0);
    let tail = &trace[trace.len() - 2000..];
    let curvature = mean(tail, |s| s.curvature);
    println!("vw 0.2 rad/s at 10 m/s: curvature {curvature:.5} 1/m");
    assert!((curvature - 0.02).abs() < 5e-4, "curvature {curvature}");
    let raw = GroundSetpoint::Pedal { drive: -0.5, steering: 0.25, handbrake: false, lean: 0.5 };
    let input = rig.ctrl.update(&raw, &GroundEstimate::of(&rig.v));
    assert_eq!((input.throttle, input.brake, input.steering, input.lean), (0.0, 0.5, 0.25, 0.5));
    assert!(!input.reverse);
}
