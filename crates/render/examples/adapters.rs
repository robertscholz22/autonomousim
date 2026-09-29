//! Lists the Vulkan adapters and renders a test frame on each: `cargo run -p autonomousim-render --example adapters`.

use autonomousim_render::{
    AdapterChoice, CameraPose, Draw, GpuContext, GpuMesh, Intrinsics, Renderer, SemanticClass, Shading, View,
};
use autonomousim_scene::mesh::cuboid;
use glam::{DQuat, DVec3, Vec3};

fn main() {
    for choice in [AdapterChoice::Auto, AdapterChoice::Software] {
        let ctx = match GpuContext::new(&choice) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("{choice:?}: {e}");
                continue;
            }
        };
        let mesh = GpuMesh::new(&ctx, &cuboid(Vec3::splat(1.0), [0.5, 0.5, 0.5, 1.0]), SemanticClass::Building);
        let mut r = Renderer::new(&ctx);
        let view = View {
            pose: CameraPose::new(DVec3::new(-5.0, 0.0, 0.0), DQuat::IDENTITY),
            intrinsics: Intrinsics::new(64, 64, 1.2),
            shading: Shading::default(),
        };
        let draws = [Draw::world(&mesh)];
        let frame = r.render(&ctx, &view, &draws).unwrap();
        let n = 200;
        let start = std::time::Instant::now();
        for _ in 0..n {
            r.render(&ctx, &view, &draws).unwrap();
        }
        let per = start.elapsed().as_secs_f64() / n as f64;
        println!(
            "{choice:?}: {}; centre depth {:.4} m (4 expected); {:.0} µs per 64² frame with readback",
            ctx.describe(),
            frame.depth_at(32, 32),
            per * 1e6
        );
    }
}
