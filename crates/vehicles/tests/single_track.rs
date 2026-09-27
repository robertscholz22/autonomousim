//! Single-track vehicles: the benchmark bicycle's layout and statics, standing still (it falls
//! over unless the feet hold it), energy conservation of the steering head, wheels and rider
//! without dissipation, and a sprung fork.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::dynamics::{KinCache, forward_kinematics, kinetic_energy};
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::presets;
use glam::{DQuat, DVec3};
use std::sync::Arc;

const DT: f64 = 1e-3;

fn benchmark() -> WheeledDef {
    presets::wheeled("bicycle_benchmark").unwrap()
}

/// Lean of the chassis (rad, positive to the right).
fn lean(v: &Wheeled) -> f64 {
    let z = v.orientation() * DVec3::Z;
    (-z.y).atan2(z.z)
}

/// Run on flat asphalt at `ground` height under gravity `g`, calling `each` after every step.
fn run(v: &mut Wheeled, input: &DriveInput, seconds: f64, ground: f64, g: f64, mut each: impl FnMut(&Wheeled)) {
    let terrain = FlatTerrain::new(ground, MaterialId::ASPHALT);
    let materials = MaterialTable::standard();
    let env = GroundStepEnv {
        scene: StaticScene { terrain: &terrain, obstacles: &NoObstacles, materials: &materials },
        gravity: DVec3::new(0.0, 0.0, -g),
        air: AirData::default(),
    };
    for _ in 0..(seconds / DT).round() as usize {
        v.step(input, &env).unwrap();
        each(v);
    }
}

/// Whether a collider of `part` touches the ground.
fn touching(v: &Wheeled, part: GroundPart) -> bool {
    v.contacts().iter().any(|c| v.colliders()[c.collider as usize].group == part as u8)
}

/// At rest, leaning `phi` to the right about the ground line under the wheels.
fn leaning(v: &mut Wheeled, phi: f64) {
    let mut init = v.rest(DVec3::ZERO, 0.0, 0.0);
    let rot = DQuat::from_rotation_x(phi);
    init.pose = Pose::new(rot * init.pose.pos, rot * init.pose.rot);
    v.reset(&init);
}

#[test]
fn benchmark_layout_and_static_loads() {
    let d = benchmark();
    assert_eq!(d.num_wheels(), 2);
    assert!(d.is_single_track());
    assert_eq!((d.wheel_axle(1), d.wheel_side(1), d.axle_wheels(1)), (1, 0, 1..2));
    assert!((d.total_mass() - 94.0).abs() < 1e-12);
    // Whole-bicycle centre of mass (Meijaard et al. 2007: B, H and both wheels).
    let com = d.total_com();
    let x = (85.0 * 0.3 + 4.0 * 0.9 + 3.0 * 1.02) / 94.0;
    let z = (85.0 * 0.9 + 4.0 * 0.7 + 2.0 * 0.3 + 3.0 * 0.35) / 94.0;
    assert!((com.x - x).abs() < 1e-12 && (com.z - z).abs() < 1e-12, "{com}");
    // The trail follows from the head angle and the fork offset.
    let (head, w) = d.steering_head().unwrap();
    assert_eq!(w, 1);
    let r = d.tire(1).radius();
    let trail = (r * head.angle.sin() - head.offset) / head.angle.cos();
    assert!((trail - 0.08).abs() < 1e-9, "trail {trail}");
    // Static loads by the lever rule, upright.
    let st = d.rest_state().expect("a static equilibrium");
    let weight = 94.0 * STANDARD_GRAVITY;
    let rear = weight * (1.02 - x) / 1.02;
    assert!(st.roll == 0.0);
    for (got, want) in st.loads.iter().zip([rear, weight - rear]) {
        assert!((got - want).abs() < 1e-3 * weight, "loads {:?} vs {rear}", st.loads);
    }
}

#[test]
fn benchmark_settles_upright_and_capsizes_standing_still() {
    let d = Arc::new(benchmark());
    let st = d.rest_state().unwrap().clone();
    // Exactly upright it stays so, on its static loads.
    let mut v = Wheeled::new(d.clone(), DT);
    run(&mut v, &DriveInput::default(), 2.0, 0.0, STANDARD_GRAVITY, |_| {});
    assert!(lean(&v).abs() < 1e-9, "lean {}", lean(&v));
    assert!(v.lin_vel_world().length() < 1e-4, "still moving {}", v.lin_vel_world());
    for (w, s) in v.wheels().enumerate() {
        assert!((s.tire.fz - st.loads[w]).abs() < 1e-3 * st.loads[w], "wheel {w}: {} vs {}", s.tire.fz, st.loads[w]);
    }
    // Leaning a little, it falls over (with the front wheel flopping into the fall) onto the
    // rider.
    let mut v = Wheeled::new(d, DT);
    leaning(&mut v, 0.02);
    let mut steer = 0.0f64;
    run(&mut v, &DriveInput::default(), 3.0, 0.0, STANDARD_GRAVITY, |v| steer = steer.min(v.steering_angle()));
    assert!(lean(&v) > 1.0, "lean {}", lean(&v));
    assert!(steer < -0.1, "the handlebar turns right, into the fall: {steer}");
    assert!(touching(&v, GroundPart::Body), "the rider hits the ground");
}

#[test]
fn feet_hold_it_up_and_lift_when_riding() {
    // On ordinary tyres: the preset's knife-edge approximations are undamped springs at
    // standstill, and it would bounce on the foot.
    let mut d = benchmark();
    for a in &mut d.axles {
        let Some(TireSpec::Fiala(p)) = &mut a.tire else { panic!("Fiala tyres") };
        p.rigid_rolling = false;
        [p.slip_stiffness, p.cornering_stiffness, p.relaxation_x, p.relaxation_y, p.vxlow] =
            [8000.0, 6000.0, 0.03, 0.05, 1.0];
    }
    d.feet =
        Some(FeetDef { down: DVec3::new(0.45, 0.3, 0.07), up: DVec3::new(0.45, 0.12, 0.3), radius: 0.05, speed: 1.5 });
    d.finish().unwrap();
    let mut v = Wheeled::new(Arc::new(d), DT);
    assert!(v.feet_down());
    leaning(&mut v, 0.02);
    run(&mut v, &DriveInput::default(), 3.0, 0.0, STANDARD_GRAVITY, |_| {});
    // It tips onto the right foot and stays there (the foot touches at about 4°).
    let phi = lean(&v);
    assert!(phi > 0.03 && phi < 0.12, "lean {phi}");
    assert!(touching(&v, GroundPart::Skid));
    assert!(v.lin_vel_world().length() < 1e-2);
    // Upright at speed, the feet go up.
    let mut init = v.rest(DVec3::ZERO, 0.0, 5.0);
    init.ang_vel_body = DVec3::ZERO;
    v.reset(&init);
    assert!(!v.feet_down());
}

#[test]
fn freewheeling_without_dissipation_conserves_energy() {
    let mut d = benchmark();
    d.rider = Some(RiderDef {
        mass: 30.0,
        com: DVec3::new(0.25, 0.0, 1.3),
        inertia: DVec3::new(1.5, 1.4, 0.4),
        products: DVec3::ZERO,
        hip: DVec3::new(0.2, 0.0, 0.95),
        max_lean: 0.5,
        stiffness: 400.0,
        damping: 0.0,
        max_torque: 1e6,
    });
    d.chassis.mass -= 30.0;
    // Free of gravity and the ground, the handlebar wanders (taking up the angular momentum
    // of the rider's sway and the wheels' gyroscopic moments): no (damped) stops.
    d.axles[1].steering_head.as_mut().unwrap().lock_stiffness = 1e-12;
    d.finish().unwrap();
    let rider = d.rider.clone().unwrap();
    let mut v = Wheeled::new(Arc::new(d), DT);
    // Floating without gravity, far above the ground: rolling, pitching, wobbling the
    // handlebar and swaying the rider.
    let mut init = v.rest(DVec3::new(0.0, 0.0, 50.0), 0.0, 1.5);
    init.ang_vel_body = DVec3::new(0.05, -0.03, 0.03);
    v.reset(&init);
    let links = |v: &Wheeled| {
        let names: Vec<&str> = v.model().links().iter().map(|l| l.name.as_str()).collect();
        let dof = |name: &str| {
            let l = names.iter().position(|n| *n == name).unwrap();
            v.model().v_offset(l)
        };
        (dof("head_1"), dof("rider"))
    };
    let (h, r) = links(&v);
    v.state.v[h] = 0.2;
    v.state.v[r] = 0.3;
    let energy = |v: &Wheeled| {
        let mut kin = KinCache::new(v.model());
        forward_kinematics(v.model(), &v.state.q, &v.state.v, &mut kin);
        let phi = v.rider_lean().0;
        kinetic_energy(v.model(), &kin) + 0.5 * rider.stiffness * phi * phi
    };
    let e0 = energy(&v);
    let (mut worst, mut max_steer) = (0.0f64, 0.0f64);
    run(&mut v, &DriveInput::default(), 10.0, 0.0, 0.0, |v| {
        worst = worst.max((energy(v) - e0).abs() / e0);
        max_steer = max_steer.max(v.steering_angle().abs());
    });
    assert!(max_steer > 0.02, "steering {max_steer}");
    assert!(v.rider_lean().1 != 0.0);
    assert!(v.wheels().all(|w| w.tire.fz == 0.0 && w.drive_torque == 0.0 && w.brake_torque == 0.0));
    assert!(worst < 1e-3, "energy drift {worst:.2e}");
}

#[test]
fn a_sprung_fork_and_swing_arm_carry_the_static_load() {
    let mut d = benchmark();
    let susp = |carrier: f64, rate: f64| -> SuspensionDef {
        serde_json::from_value(serde_json::json!({
            "carrier_mass": carrier,
            "carrier_inertia": [0.01, 0.01, 0.01],
            "spring": { "rate": rate },
            "damper": { "bump": 800.0, "rebound": 1200.0 },
        }))
        .unwrap()
    };
    let mut rear = susp(1.0, 2.0e4);
    rear.trailing_arm = Some(TrailingArmDef { length: 0.45, angle: 0.1 });
    d.axles[0].suspension = Some(rear);
    d.axles[1].suspension = Some(susp(0.5, 1.2e4));
    d.chassis.mass -= 1.5;
    d.finish().unwrap();
    let st = d.rest_state().unwrap().clone();
    assert!(st.travel.iter().all(|t| t.abs() < 1e-9), "automatic preloads: zero travel {:?}", st.travel);
    // The fork slides along the steering axis.
    let v = Wheeled::new(Arc::new(d.clone()), DT);
    let names: Vec<&str> = v.model().links().iter().map(|l| l.name.as_str()).collect();
    let fork = names.iter().position(|n| *n == "carrier_1").unwrap();
    assert_eq!(v.model().link(fork).parent, names.iter().position(|n| *n == "head_1"));
    let mut v = Wheeled::new(Arc::new(d), DT);
    run(&mut v, &DriveInput::default(), 2.0, 0.0, STANDARD_GRAVITY, |_| {});
    assert!(lean(&v).abs() < 1e-9);
    for (w, s) in v.wheels().enumerate() {
        assert!(s.travel.abs() < 1e-3, "wheel {w} travel {}", s.travel);
        assert!((s.tire.fz - st.loads[w]).abs() < 2e-3 * st.loads[w], "wheel {w}: {} vs {}", s.tire.fz, st.loads[w]);
    }
}
