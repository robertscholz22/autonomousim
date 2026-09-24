//! Cost of a Magic Formula evaluation (combined slip, all outputs) and of a full tyre step
//! (road plane from the height grid, vertical load, transient slip, model, wrench).

use autonomousim_core::terrain::Terrain;
use autonomousim_vehicles::ground::tire::{MfInput, MfParams, Surface, Tire, WheelMotion};
use autonomousim_world::testworlds;
use criterion::{Criterion, criterion_group, criterion_main};
use glam::DVec3;
use std::hint::black_box;
use std::path::PathBuf;

fn bench(c: &mut Criterion) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let world = testworlds::forest_patch(512.0, 150.0, 1);
    for name in ["Sedan_Pac02Tire", "HMMWV_Pac02Tire"] {
        let params = MfParams::read(root.join(format!("assets/tires/{name}.tir"))).unwrap();
        let fz = params.fnomin * params.lfzo;
        let mut input = MfInput::new(fz, 0.03, 0.04, 0.02, 15.0);
        c.bench_function(&format!("tire/{name}_mf_eval"), |b| {
            b.iter(|| {
                input.kappa = -input.kappa;
                black_box(params.eval(black_box(&input)))
            })
        });

        let tire = Tire::magic_formula(params.clone(), None).unwrap();
        let mut state = tire.initial_state();
        let (x, y) = (10.0, 20.0);
        let center = DVec3::new(x, y, world.terrain().height(x, y) + 0.97 * tire.radius());
        let mut motion = WheelMotion {
            center,
            axis: DVec3::Y,
            velocity: DVec3::new(15.0, -0.5, 0.0),
            carrier_angvel: DVec3::ZERO,
            spin: 15.2 / tire.radius(),
        };
        c.bench_function(&format!("tire/{name}_step_on_heightgrid"), |b| {
            b.iter(|| {
                motion.spin = -motion.spin;
                let contact = tire.contact(world.terrain(), &motion);
                black_box(tire.step(&mut state, contact.as_ref(), black_box(&motion), Surface::REFERENCE, 1e-3))
            })
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
