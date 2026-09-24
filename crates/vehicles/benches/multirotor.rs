//! Cost of one multirotor physics step (rotors, drag, contacts, ABA, integration).

use autonomousim_core::contact::StaticScene;
use autonomousim_core::math::Pose;
use autonomousim_core::math::frames::GRAVITY_ENU;
use autonomousim_core::terrain::Terrain;
use autonomousim_vehicles::multirotor::{AirData, GroundPlane, InitialState, MotorInit, Multirotor, StepEnv};
use autonomousim_vehicles::presets;
use autonomousim_world::testworlds;
use criterion::{Criterion, criterion_group, criterion_main};
use glam::DVec3;
use std::hint::black_box;
use std::sync::Arc;

fn bench(c: &mut Criterion) {
    let world = testworlds::forest_patch(512.0, 150.0, 1);
    let scene = StaticScene { terrain: world.terrain(), obstacles: world.obstacles(), materials: world.materials() };
    for name in ["cf2x", "iris_like"] {
        let def = Arc::new(presets::multirotor(name).unwrap());
        let hover = def.hover_omega(9.80665, 1.2);
        let mut quad = Multirotor::new(def, 0.002);
        let start = DVec3::new(0.0, 0.0, world.terrain().height(0.0, 0.0) + 60.0);
        let init = InitialState {
            lin_vel_world: DVec3::new(3.0, 1.0, 0.0),
            ang_vel_body: DVec3::new(0.2, -0.1, 0.3),
            ..InitialState::at(Pose::from_translation(start), MotorInit::Speed(hover))
        };
        quad.reset(&init);
        let air = AirData { density: 1.2, wind: DVec3::new(2.0, 0.0, 0.0) };
        // Small yaw differential; reset every 2 s so the vehicle stays airborne.
        let cmd: Vec<f64> = quad.def().rotors.iter().map(|m| hover * (1.0 + 0.01 * m.spin.sign())).collect();
        let mut k = 0;
        c.bench_function(&format!("multirotor/{name}_step_airborne"), |b| {
            b.iter(|| {
                k += 1;
                if k % 1000 == 0 {
                    quad.reset(&init);
                }
                let ground = GroundPlane::below(world.terrain(), quad.position(), 2.0);
                let env = StepEnv { scene: Some(scene), gravity: GRAVITY_ENU, air, ground };
                quad.step(black_box(&cmd), &env).unwrap();
            })
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
