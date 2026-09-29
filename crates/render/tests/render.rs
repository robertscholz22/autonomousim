//! Rendered images against analytic scenes, and reproducibility on the software rasterizer.
//!
//! All tests render on lavapipe. Bless the golden image hash with
//! `AUTONOMOUSIM_BLESS=1 cargo test -p autonomousim-render --test render`.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::path::PathBuf;
use std::sync::OnceLock;

use autonomousim_render::{
    AdapterChoice, CameraPose, Draw, Frame, GpuContext, GpuMesh, Intrinsics, Job, Renderer, SemanticClass, Shading,
    View,
};
use autonomousim_scene::MeshData;
use autonomousim_scene::mesh::{cuboid, icosphere};
use glam::{DQuat, DVec3, Vec3};

fn ctx() -> &'static GpuContext {
    static CTX: OnceLock<GpuContext> = OnceLock::new();
    CTX.get_or_init(|| {
        let ctx = GpuContext::new(&AdapterChoice::Software).expect("lavapipe (mesa-vulkan-drivers)");
        assert!(ctx.is_software());
        ctx
    })
}

/// A square of half size `h` at z = 0, facing up.
fn ground(h: f32, color: [f32; 4]) -> MeshData {
    let mut m = MeshData::new();
    let c = |x: f32, y: f32| Vec3::new(x * h, y * h, 0.0);
    m.push_flat_triangle(c(-1., -1.), c(1., -1.), c(1., 1.), color);
    m.push_flat_triangle(c(-1., -1.), c(1., 1.), c(-1., 1.), color);
    m
}

const GREY: [f32; 4] = [0.3, 0.3, 0.3, 1.0];

/// Camera at `position` looking along the heading `yaw`, pitched down by `pitch`.
fn pose(position: DVec3, yaw: f64, pitch: f64) -> CameraPose {
    CameraPose::new(position, DQuat::from_rotation_z(yaw) * DQuat::from_rotation_y(pitch))
}

/// World-frame ray (unit depth along the optical axis) through the centre of pixel `(i, j)`,
/// offset by `(du, dv)` px.
fn world_ray(view: &View, i: u32, j: u32, du: f64, dv: f64) -> DVec3 {
    view.pose.orientation * view.intrinsics.ray(i as f64 + 0.5 + du, j as f64 + 0.5 + dv)
}

/// Axis depth `t` of a ray from `o` along `d` (unit axis depth) hitting the ground z = 0.
fn hit_ground(o: DVec3, d: DVec3, half: f64) -> Option<f64> {
    if d.z >= 0.0 {
        return None;
    }
    let t = -o.z / d.z;
    let p = o + t * d;
    (p.x.abs() < half && p.y.abs() < half).then_some(t)
}

/// Slab test against a box of half extents `h` at `centre`, rotated by `rot`: the entry depth
/// and the entry face (0..6).
fn hit_box(o: DVec3, d: DVec3, centre: DVec3, rot: DQuat, h: DVec3) -> Option<(f64, usize)> {
    let (o, d) = (rot.inverse() * (o - centre), rot.inverse() * d);
    let (mut t0, mut t1, mut face) = (f64::NEG_INFINITY, f64::INFINITY, 0);
    for k in 0..3 {
        if d[k].abs() < 1e-15 {
            if o[k].abs() > h[k] {
                return None;
            }
            continue;
        }
        let (a, b) = ((-h[k] - o[k]) / d[k], (h[k] - o[k]) / d[k]);
        let (near, far) = if a < b { (a, b) } else { (b, a) };
        if near > t0 {
            t0 = near;
            face = 2 * k + usize::from(d[k] < 0.0);
        }
        t1 = t1.min(far);
    }
    (t0 <= t1 && t0 > 0.0).then_some((t0, face))
}

fn check_depth(frame: &Frame, i: u32, j: u32, t: f64) {
    let got = frame.depth_at(i, j) as f64;
    assert!((got - t).abs() <= 1e-4 * t, "pixel ({i}, {j}): depth {got} vs {t}");
}

#[test]
fn plane_depth_matches_the_pinhole_model() {
    let ctx = ctx();
    let mesh = GpuMesh::new(ctx, &ground(500.0, GREY), SemanticClass::Grass);
    let mut r = Renderer::new(ctx);
    // Tilted and yawed, so the horizon crosses the image at an angle.
    let view = View {
        pose: CameraPose::new(
            DVec3::new(3.0, -2.0, 7.0),
            DQuat::from_rotation_z(0.7) * DQuat::from_rotation_y(0.35) * DQuat::from_rotation_x(0.2),
        ),
        intrinsics: Intrinsics::new(96, 64, 80f64.to_radians()),
        shading: Shading::default(),
    };
    let frame = r.render(ctx, &view, &[Draw::world(&mesh)]).unwrap();
    assert_eq!((frame.width, frame.height), (96, 64));
    let (mut ground_px, mut sky_px) = (0, 0);
    for j in 0..frame.height {
        for i in 0..frame.width {
            let o = view.pose.position;
            let hits: Vec<Option<f64>> = [(0.0, 0.0), (-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
                .iter()
                .map(|&(du, dv)| hit_ground(o, world_ray(&view, i, j, du, dv), 500.0).filter(|t| *t < 900.0))
                .collect();
            // Only pixels well inside the ground or the sky: the edge is up to the rasterizer.
            match hits[0] {
                Some(t) if hits.iter().all(Option::is_some) => {
                    check_depth(&frame, i, j, t);
                    assert_eq!(frame.class_at(i, j), SemanticClass::Grass.id());
                    ground_px += 1;
                }
                None if hits.iter().all(Option::is_none) => {
                    assert_eq!(frame.depth_at(i, j), 0.0, "sky at ({i}, {j})");
                    assert_eq!(frame.class_at(i, j), SemanticClass::Sky.id());
                    sky_px += 1;
                }
                _ => {}
            }
        }
    }
    assert!(ground_px > 2500 && sky_px > 1000, "{ground_px} ground, {sky_px} sky pixels");
}

#[test]
fn a_box_occludes_the_ground() {
    let ctx = ctx();
    let h = DVec3::new(1.5, 1.0, 1.2);
    let (centre, rot) = (DVec3::new(9.0, 0.5, 1.2), DQuat::from_rotation_z(0.4));
    let floor = GpuMesh::new(ctx, &ground(200.0, GREY), SemanticClass::Grass);
    let block = GpuMesh::new(ctx, &cuboid(h.as_vec3(), GREY), SemanticClass::Building);
    let mut r = Renderer::new(ctx);
    let view = View {
        pose: pose(DVec3::new(0.0, 0.0, 3.0), 0.0, 0.25),
        intrinsics: Intrinsics::new(64, 64, 70f64.to_radians()),
        shading: Shading::default(),
    };
    let frame = r.render(ctx, &view, &[Draw::world(&floor), Draw::new(&block, centre, rot)]).unwrap();
    // What the ray sees: (depth, what) with what = face index, 6 for the ground, 7 for the sky.
    let see = |d: DVec3| -> (f64, usize) {
        let o = view.pose.position;
        let g = hit_ground(o, d, 200.0).map(|t| (t, 6));
        match (hit_box(o, d, centre, rot, h), g) {
            (Some(b), Some(g)) if g.0 < b.0 => g,
            (Some(b), _) => b,
            (None, Some(g)) => g,
            (None, None) => (0.0, 7),
        }
    };
    let mut box_px = 0;
    for j in 0..frame.height {
        for i in 0..frame.width {
            let (t, what) = see(world_ray(&view, i, j, 0.0, 0.0));
            let clear = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
                .iter()
                .all(|&(du, dv)| see(world_ray(&view, i, j, du, dv)).1 == what);
            if !clear {
                continue;
            }
            let class = match what {
                0..6 => SemanticClass::Building,
                6 => SemanticClass::Grass,
                _ => SemanticClass::Sky,
            };
            assert_eq!(frame.class_at(i, j), class.id(), "class at ({i}, {j})");
            if what < 7 {
                check_depth(&frame, i, j, t);
            }
            box_px += usize::from(what < 6);
        }
    }
    assert!(box_px > 100, "{box_px} box pixels");
}

/// Linear → sRGB-encoded byte.
fn srgb(x: f64) -> u8 {
    let s = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn assert_rgb(got: [u8; 3], linear: [f64; 3], what: &str) {
    for k in 0..3 {
        let want = srgb(linear[k]);
        assert!(got[k].abs_diff(want) <= 1, "{what}: {got:?} vs {:?}", linear.map(srgb));
    }
}

#[test]
fn shading_follows_the_sun() {
    let ctx = ctx();
    let color = [0.2f32, 0.5, 0.8, 1.0];
    let lin = |k: f64| [0, 1, 2].map(|i| color[i] as f64 * k);
    let floor = GpuMesh::new(ctx, &ground(100.0, color), SemanticClass::Grass);
    // A wall facing +x, seen from its back by a camera looking along +x.
    let mut wall = MeshData::new();
    let c = |y: f32, z: f32| Vec3::new(0.0, y, z);
    wall.push_flat_triangle(c(-5., -5.), c(5., -5.), c(5., 5.), color);
    wall.push_flat_triangle(c(-5., -5.), c(5., 5.), c(-5., 5.), color);
    let wall = GpuMesh::new(ctx, &wall, SemanticClass::Building);
    let mut r = Renderer::new(ctx);
    let k = Intrinsics::new(32, 32, 60f64.to_radians());
    let ambient = 0.35;
    let shade = |sun: DVec3| Shading { sun, ambient, ..Shading::default() };
    let down = pose(DVec3::new(0.0, 0.0, 10.0), 0.0, 0.5 * PI);
    let facing = pose(DVec3::new(-10.0, 0.0, 0.0), 0.0, 0.0);
    let pixel = |r: &mut Renderer, pose: CameraPose, mesh: &GpuMesh, sun: DVec3| {
        let f = r.render(ctx, &View { pose, intrinsics: k, shading: shade(sun) }, &[Draw::world(mesh)]).unwrap();
        f.rgb_at(16, 16)
    };
    // Ground: overhead sun, sun at 60° from the zenith, sun below the horizon.
    assert_rgb(pixel(&mut r, down, &floor, DVec3::Z), lin(1.0), "overhead sun");
    let slant = DVec3::new((PI / 3.0).sin(), 0.0, (PI / 3.0).cos());
    assert_rgb(pixel(&mut r, down, &floor, slant), lin(ambient + (1.0 - ambient) * 0.5), "slanted sun");
    assert_rgb(pixel(&mut r, down, &floor, -DVec3::Z), lin(ambient), "sun below");
    // The wall's back face is lit from the camera's side only.
    assert_rgb(pixel(&mut r, facing, &wall, -DVec3::X), lin(1.0), "back face, sun behind the camera");
    assert_rgb(pixel(&mut r, facing, &wall, DVec3::X), lin(ambient), "back face, sun behind the wall");
    // A mesh's rotation turns its normals: the floor turned upside down still faces the camera.
    let flipped = r
        .render(
            ctx,
            &View { pose: down, intrinsics: k, shading: shade(DVec3::Z) },
            &[Draw::new(&floor, DVec3::ZERO, DQuat::from_rotation_x(PI))],
        )
        .unwrap();
    assert_rgb(flipped.rgb_at(16, 16), lin(1.0), "flipped floor");
    // The sky is the clear colour.
    let up = r
        .render(
            ctx,
            &View { pose: pose(DVec3::Z, 0.0, -0.5 * PI), intrinsics: k, shading: shade(DVec3::Z) },
            &[Draw::world(&floor)],
        )
        .unwrap();
    assert_rgb(up.rgb_at(16, 16), Shading::default().sky, "sky");
}

/// A scene with every kind of surface: ground, boxes, a smooth sphere, far and near geometry.
fn scene_hash() -> String {
    let ctx = ctx();
    let floor = GpuMesh::new(ctx, &ground(300.0, [0.25, 0.4, 0.15, 1.0]), SemanticClass::Grass);
    let block = GpuMesh::new(ctx, &cuboid(Vec3::new(2.0, 1.0, 1.5), [0.6, 0.5, 0.4, 1.0]), SemanticClass::Building);
    let ball = GpuMesh::new(ctx, &icosphere(1.0, 3, true, [0.8, 0.1, 0.1, 1.0]), SemanticClass::Vehicle);
    let draws = [
        Draw::world(&floor),
        Draw::new(&block, DVec3::new(12.0, -3.0, 1.5), DQuat::from_rotation_z(0.3)),
        Draw::new(&block, DVec3::new(40.0, 10.0, 1.5), DQuat::from_rotation_z(-1.1)),
        Draw::new(&ball, DVec3::new(6.0, 1.5, 1.0), DQuat::from_rotation_x(0.2)),
    ];
    let view = View {
        pose: pose(DVec3::new(0.0, 0.0, 2.5), 0.1, 0.12),
        intrinsics: Intrinsics::new(128, 96, 90f64.to_radians()),
        shading: Shading::default(),
    };
    let mut r = Renderer::new(ctx);
    let a = r.render(ctx, &view, &draws).unwrap();
    // Same renderer again, and a new renderer: the same frame.
    assert_eq!(a, r.render(ctx, &view, &draws).unwrap());
    assert_eq!(a, Renderer::new(ctx).render(ctx, &view, &draws).unwrap());
    let classes: std::collections::BTreeSet<u8> = a.class.iter().copied().collect();
    assert_eq!(classes, [0, 1, 11, 13].into_iter().collect(), "every surface in view");
    let mut h = blake3::Hasher::new();
    h.update(&a.rgb);
    for d in &a.depth {
        h.update(&d.to_le_bytes());
    }
    h.update(&a.class);
    h.finalize().to_string()
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden_images.toml")
}

#[test]
fn golden_image_on_lavapipe() {
    let hash = scene_hash();
    let hashes = BTreeMap::from([("scene".to_string(), hash)]);
    if std::env::var_os("AUTONOMOUSIM_BLESS").is_some() {
        let text = format!(
            "# Golden image hashes on lavapipe ({}),\n# crates/render/tests/render.rs; bless with AUTONOMOUSIM_BLESS=1.\n{}",
            ctx().describe(),
            toml::to_string(&hashes).unwrap()
        );
        std::fs::write(golden_path(), text).unwrap();
        return;
    }
    let text = std::fs::read_to_string(golden_path()).expect("fixtures/golden_images.toml");
    let golden: BTreeMap<String, String> = toml::from_str(&text).unwrap();
    assert_eq!(hashes, golden, "rendered images changed (bless if intended)");
}

/// Helper for [`the_image_does_not_depend_on_the_thread_count`].
#[test]
#[ignore = "run by the_image_does_not_depend_on_the_thread_count"]
fn print_scene_hash() {
    println!("\nHASH {}", scene_hash());
}

#[test]
fn the_image_does_not_depend_on_the_thread_count() {
    let here = scene_hash();
    for threads in ["1", "4"] {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "print_scene_hash", "--nocapture", "--test-threads", "1"])
            .env("LP_NUM_THREADS", threads)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let hash = stdout.lines().find_map(|l| l.strip_prefix("HASH ")).unwrap_or_else(|| panic!("{stdout}"));
        assert_eq!(hash, here, "LP_NUM_THREADS={threads}");
    }
}

/// A batch renders each image exactly as it renders alone, whatever its size, its place in
/// the batch or the number of images (more than one set of layered targets holds).
#[test]
fn a_batch_renders_each_image_as_alone() {
    let ctx = ctx();
    let floor = GpuMesh::new(ctx, &ground(300.0, [0.25, 0.4, 0.15, 1.0]), SemanticClass::Grass);
    let block = GpuMesh::new(ctx, &cuboid(Vec3::new(2.0, 1.0, 1.5), [0.6, 0.5, 0.4, 1.0]), SemanticClass::Building);
    let ball = GpuMesh::new(ctx, &icosphere(1.0, 2, true, [0.8, 0.1, 0.1, 1.0]), SemanticClass::Vehicle);
    let mut draws = Vec::new();
    let mut jobs = Vec::new();
    for i in 0..300 {
        let a = i as f64 * 0.37;
        // Most images are 8×8 (more than the 256 layers of one set of targets), some 24×18.
        let (w, h) = if i % 10 == 3 { (24, 18) } else { (8, 8) };
        let view = View {
            pose: pose(DVec3::new(a.sin() * 3.0, a.cos() * 3.0, 2.5), a, 0.2),
            intrinsics: Intrinsics::new(w, h, 80f64.to_radians()),
            shading: Shading::default(),
        };
        let first = draws.len();
        draws.push(Draw::world(&floor));
        draws.push(Draw::new(&block, DVec3::new(8.0, -2.0, 1.5), DQuat::from_rotation_z(a)));
        if i % 2 == 0 {
            draws.push(Draw::new(&ball, DVec3::new(4.0 * a.cos(), 4.0 * a.sin(), 1.0), DQuat::IDENTITY));
        }
        jobs.push(Job { view, draws: first..draws.len() });
    }
    let mut r = Renderer::new(ctx);
    // A small batch first, so that the targets grow.
    let few = r.render_batch(ctx, &jobs[..5], &draws).unwrap();
    let all = r.render_batch(ctx, &jobs, &draws).unwrap();
    assert_eq!(&all[..5], &few[..]);
    assert_eq!(all, r.render_batch(ctx, &jobs, &draws).unwrap());
    let mut alone = Renderer::new(ctx);
    for (i, (job, frame)) in jobs.iter().zip(&all).enumerate() {
        assert_eq!(*frame, alone.render(ctx, &job.view, &draws[job.draws.clone()]).unwrap(), "image {i}");
    }
    assert!(all.iter().any(|f| f.class.contains(&SemanticClass::Vehicle.id())));
    assert!(r.render_batch(ctx, &[], &draws).unwrap().is_empty());
}
