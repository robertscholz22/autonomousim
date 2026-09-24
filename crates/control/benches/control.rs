//! Cost of one controller update per action mode, and of controller + physics step, for the
//! multirotor cascade and the ground controller.

use autonomousim_control::ground::{
    GroundActionLimits, GroundActionMap, GroundActionMode, GroundConfig, GroundController, GroundEstimate,
};
use autonomousim_control::multirotor::{
    ActionLimits, ActionMap, ActionMode, ControllerConfig, MultirotorController, StateEstimate,
};
use autonomousim_core::math::Pose;
use autonomousim_core::math::frames::GRAVITY_ENU;
use autonomousim_vehicles::ground::Wheeled;
use autonomousim_vehicles::multirotor::{AirData, InitialState, MotorInit, Multirotor, StepEnv};
use autonomousim_vehicles::presets;
use criterion::{Criterion, criterion_group, criterion_main};
use glam::DVec3;
use std::hint::black_box;
use std::sync::Arc;

fn bench(c: &mut Criterion) {
    let def = Arc::new(presets::multirotor("cf2x").unwrap());
    let dt = 0.002;
    let mut quad = Multirotor::new(def.clone(), dt);
    let init = InitialState::at(Pose::from_translation(DVec3::new(0.0, 0.0, 10.0)), MotorInit::Speed(1500.0));
    quad.reset(&init);
    let est = StateEstimate::of(&quad);
    let env = StepEnv { scene: None, gravity: GRAVITY_ENU, air: AirData::default(), ground: None };
    let mut ctrl = MultirotorController::new(&def, dt, &ControllerConfig::default()).unwrap();
    let mut cmd = [0.0; 4];
    for mode in ActionMode::ALL {
        let map = ActionMap::new(mode, ActionLimits::default(), &def, ctrl.max_thrust());
        let sp = map.setpoint(&[0.1, -0.2, 0.05, 0.0], &est);
        c.bench_function(&format!("controller/{mode}"), |b| {
            b.iter(|| ctrl.update(black_box(&sp), black_box(&est), &mut cmd))
        });
    }
    let map = ActionMap::new(ActionMode::Velocity, ActionLimits::default(), &def, ctrl.max_thrust());
    let sp = map.setpoint(&[0.3, 0.0, 0.1, 0.2], &est);
    let mut k = 0;
    c.bench_function("controller/velocity_plus_physics_step", |b| {
        b.iter(|| {
            k += 1;
            if k % 1000 == 0 {
                quad.reset(&init);
                ctrl.reset();
            }
            ctrl.update(&sp, &StateEstimate::of(&quad), &mut cmd);
            quad.step(black_box(&cmd), &env).unwrap();
        })
    });
}

fn bench_ground(c: &mut Criterion) {
    for (name, mode) in [("sedan_like", GroundActionMode::Vk), ("rover_skid", GroundActionMode::Vw)] {
        let def = Arc::new(presets::wheeled(name).unwrap());
        let mut v = Wheeled::new(def.clone(), 1e-3);
        v.reset(&v.rest(DVec3::ZERO, 0.0, 1.0));
        let est = GroundEstimate::of(&v);
        let mut ctrl = GroundController::new(&def, 1e-3, &GroundConfig::default()).unwrap();
        let map = GroundActionMap::new(mode, &GroundActionLimits::default(), &def).unwrap();
        let sp = map.setpoint(&[0.3, 0.2]);
        c.bench_function(&format!("ground_controller/{name}/{mode}"), |b| {
            b.iter(|| ctrl.update(black_box(&sp), black_box(&est)))
        });
    }
}

criterion_group!(benches, bench, bench_ground);
criterion_main!(benches);
