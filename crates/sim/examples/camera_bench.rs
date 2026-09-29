//! Camera throughput of a batch: drones with one down camera over generated maps.
//!
//! ```text
//! cargo run -p autonomousim-sim --release --example camera_bench -- \
//!     [--adapter auto|software|<name>] [--envs 64] [--size 64] [--threads 10] [--steps 100]
//!     [--repeat 5] [--map rural|wild|forest] [--maps 4] [--save]
//! ```
//!
//! Every policy step renders one frame per world (RGB + depth in the observation). Reports
//! the median over `--repeat` runs of the step time with and without the camera and the
//! camera frames per second of the whole step; `--save` appends the results to
//! `benchmarks/results/<date>-<host>.json` under `"camera"`.

use autonomousim_render::AdapterChoice;
use autonomousim_sim::camera;
use autonomousim_sim::{BatchSim, Scenario};
use std::time::Instant;

struct Options {
    adapter: String,
    envs: usize,
    size: u32,
    threads: usize,
    steps: usize,
    repeat: usize,
    map: String,
    maps: u32,
    save: bool,
}

fn options() -> Options {
    let mut o = Options {
        adapter: "auto".into(),
        envs: 64,
        size: 64,
        threads: 10,
        steps: 100,
        repeat: 5,
        map: "rural".into(),
        maps: 4,
        save: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--adapter" => o.adapter = value(),
            "--envs" => o.envs = value().parse().expect("--envs"),
            "--size" => o.size = value().parse().expect("--size"),
            "--threads" => o.threads = value().parse().expect("--threads"),
            "--steps" => o.steps = value().parse().expect("--steps"),
            "--repeat" => o.repeat = value().parse().expect("--repeat"),
            "--map" => o.map = value(),
            "--maps" => o.maps = value().parse().expect("--maps"),
            "--save" => o.save = true,
            _ => panic!("unknown argument {a}"),
        }
    }
    o
}

fn scenario(o: &Options, camera: bool) -> Scenario {
    let map = match o.map.as_str() {
        "rural" => format!(r#"{{ type = "rural", seed = 0, count = {}, preset = "training" }}"#, o.maps),
        "wild" => format!(r#"{{ type = "wild", seed = 0, count = {}, preset = "training" }}"#, o.maps),
        "forest" => r#"{ type = "testworld", kind = "forest_patch", size = 200.0, density = 150.0, seed = 5 }"#.into(),
        m => panic!("unknown map {m}"),
    };
    let (sensors, obs) = if camera {
        (
            format!(
                r#"sensors = [ {{ name = "down", type = "camera", width = {0}, height = {0}, fov_deg = 90.0, rate_hz = 50, mount = {{ rotation = [0.0, 1.5707963267948966, 0.0] }} }} ]"#,
                o.size
            ),
            r#"{ term = "camera", sensor = "down", output = "rgb" }, { term = "camera", sensor = "down", output = "depth", range = 50.0 },"#,
        )
    } else {
        (String::new(), "")
    };
    Scenario::from_toml(&format!(
        r#"
        name = "camera_bench"
        map = {map}
        [[groups]]
        name = "drones"
        vehicle = "iris_like"
        action_mode = "velocity"
        spawn = {{ agl = [10.0, 30.0] }}
        {sensors}
        obs = [ {obs} {{ term = "lin_vel_body" }}, {{ term = "rot6d" }} ]
        "#
    ))
    .expect("benchmark scenario")
}

/// Median time per step (s) over the runs.
fn time_steps(b: &mut BatchSim, o: &Options) -> f64 {
    let dim = b.scenario().groups[0].act_dim() * b.num_envs();
    let actions: Vec<Vec<f32>> =
        (0..16).map(|k| (0..dim).map(|i| 0.3 * ((k * 5 + i * 3) as f32 * 0.7).sin()).collect()).collect();
    for k in 0..10 {
        b.step(&[&actions[k % 16]]);
    }
    let mut runs: Vec<f64> = (0..o.repeat)
        .map(|_| {
            let t = Instant::now();
            for k in 0..o.steps {
                b.step(&[&actions[k % 16]]);
            }
            t.elapsed().as_secs_f64() / o.steps as f64
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    runs[runs.len() / 2]
}

fn main() {
    let o = options();
    camera::use_adapter(&AdapterChoice::parse(&o.adapter));
    let adapter = camera::gpu().expect("GPU context").describe();
    let with = scenario(&o, true).compile().expect("scenario");
    let without = scenario(&o, false).compile().expect("scenario");
    let physics = time_steps(&mut BatchSim::from_compiled(without.into(), o.envs, 0, o.threads).unwrap(), &o);
    let t = Instant::now();
    let mut b = BatchSim::from_compiled(with.into(), o.envs, 0, o.threads).unwrap();
    let setup = t.elapsed().as_secs_f64();
    let step = time_steps(&mut b, &o);
    let frames_per_s = o.envs as f64 / step;
    println!("adapter: {adapter}");
    println!(
        "{} worlds, {}×{} RGB + depth, {} map(s) {}, {} threads: step {:.2} ms (physics only {:.2} ms), cameras {:.2} ms/step, {:.0} frames/s, {:.0} env steps/s; setup {:.2} s",
        o.envs,
        o.size,
        o.size,
        o.maps,
        o.map,
        b.num_threads(),
        1e3 * step,
        1e3 * physics,
        1e3 * (step - physics),
        frames_per_s,
        frames_per_s,
        setup
    );
    if o.save {
        save(&o, &adapter, step, physics);
    }
}

fn save(o: &Options, adapter: &str, step: f64, physics: f64) {
    let out = |cmd: &str, args: &[&str]| {
        String::from_utf8(std::process::Command::new(cmd).args(args).output().expect(cmd).stdout)
            .expect("utf-8")
            .trim()
            .to_string()
    };
    let (date, host) = (out("date", &["+%F"]), out("uname", &["-n"]));
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join(format!("benchmarks/results/{date}-{host}.json"));
    let mut data: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({ "date": date, "host": host }));
    let row = serde_json::json!({
        "adapter": adapter,
        "num_envs": o.envs,
        "size": o.size,
        "map": o.map,
        "maps": o.maps,
        "threads": o.threads,
        "step_ms": 1e3 * step,
        "physics_ms": 1e3 * physics,
        "camera_ms": 1e3 * (step - physics),
        "frames_per_s": o.envs as f64 / step,
    });
    data.as_object_mut().expect("a JSON object").entry("camera").or_insert_with(|| serde_json::json!([]));
    data["camera"].as_array_mut().expect("an array").push(row);
    std::fs::write(&path, serde_json::to_string_pretty(&data).expect("JSON") + "\n").expect("writing the results");
    println!("appended to {}", path.display());
}
