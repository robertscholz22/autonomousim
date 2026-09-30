//! Urban generator: golden hashes (independent of the thread count), street network
//! invariants (planar, junction angles, segment lengths, connectivity, grades), sections,
//! materials, timing, cache.
//!
//! Regenerate the golden hashes after an intended change of the generator output (and bump
//! `URBAN_VERSION`) with
//! `AUTONOMOUSIM_BLESS=1 cargo test -p autonomousim-procgen --test urban`.

use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::Terrain;
use autonomousim_procgen::urban::{self, URBAN_VERSION, UrbanConfig, UrbanPreset};
use autonomousim_procgen::{MapCache, ProcgenError};
use autonomousim_world::{NodeKind, RoadClass, StaticWorld};
use glam::DVec2;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
struct Golden {
    urban: Vec<GoldenMap>,
}

#[derive(Clone, Serialize, Deserialize)]
struct GoldenMap {
    preset: UrbanPreset,
    seed: u64,
    hash: String,
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden_hashes_urban.toml")
}

fn generate_with_threads(c: &UrbanConfig, seed: u64, threads: usize) -> StaticWorld {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| urban::generate(c, seed)).unwrap().0
}

#[test]
fn golden_hashes_do_not_depend_on_the_thread_count() {
    let bless = std::env::var_os("AUTONOMOUSIM_BLESS").is_some();
    let golden: Golden = match std::fs::read_to_string(golden_path()) {
        Ok(text) => toml::from_str(&text).unwrap(),
        Err(_) if bless => Golden {
            urban: [(UrbanPreset::Training, 1), (UrbanPreset::Training, 2), (UrbanPreset::Showcase, 3)]
                .map(|(preset, seed)| GoldenMap { preset, seed, hash: String::new() })
                .to_vec(),
        },
        Err(e) => panic!("fixtures/golden_hashes_urban.toml: {e}"),
    };
    let mut blessed = Vec::new();
    for g in &golden.urban {
        let c = g.preset.config();
        let one = generate_with_threads(&c, g.seed, 1).content_hash().to_string();
        let many = generate_with_threads(&c, g.seed, 12).content_hash().to_string();
        assert_eq!(one, many, "{:?} seed {}: 1 vs 12 threads", g.preset, g.seed);
        if bless {
            blessed.push(GoldenMap { hash: one, ..g.clone() });
        } else {
            assert_eq!(one, g.hash, "{:?} seed {} changed (bump URBAN_VERSION and bless)", g.preset, g.seed);
        }
    }
    if bless {
        let header = "# Content hashes of generated urban maps (see crates/procgen/tests/urban.rs).\n\
                      # Regenerate: AUTONOMOUSIM_BLESS=1 cargo test -p autonomousim-procgen --test urban\n\n";
        let body = toml::to_string(&Golden { urban: blessed }).unwrap();
        std::fs::write(golden_path(), format!("{header}{body}")).unwrap();
    }
}

/// Proper crossing of segments `p–q` and `a–b` (away from their ends).
fn crosses(p: DVec2, q: DVec2, a: DVec2, b: DVec2) -> bool {
    let (r, s) = (q - p, b - a);
    let den = r.perp_dot(s);
    if den.abs() < 1e-12 {
        return false;
    }
    let t = (a - p).perp_dot(s) / den;
    let u = (a - p).perp_dot(r) / den;
    (1e-6..=1.0 - 1e-6).contains(&t) && (1e-6..=1.0 - 1e-6).contains(&u)
}

/// Everything the generator promises about the street network and the terrain under it.
fn check_invariants(c: &UrbanConfig, seed: u64) {
    let (w, stats) = urban::generate(c, seed).unwrap();
    assert_eq!((w.meta.generator.as_str(), w.meta.generator_version, w.meta.seed), ("urban", URBAN_VERSION, seed));
    let net = w.roads();
    let t = w.grid();
    assert!(net.has_sections());
    assert!(stats.roads[0] >= 4 && stats.roads[1] >= 4 && stats.roads[2] >= 10, "seed {seed}: {stats:?}");
    assert!(stats.districts[0] >= 1, "seed {seed}: a downtown");
    let nodes = net.nodes();
    let roads = net.roads();

    // Planar: road centre lines meet only at their nodes.
    const CELL: f64 = 16.0;
    let half = 0.5 * c.size + CELL;
    let k = (2.0 * half / CELL).ceil() as usize;
    let mut cells: Vec<Vec<(usize, usize)>> = vec![Vec::new(); k * k];
    let cell = |p: DVec2| (((p.x + half) / CELL) as usize).min(k - 1) + k * (((p.y + half) / CELL) as usize).min(k - 1);
    for (r, road) in roads.iter().enumerate() {
        let pts = road.line.points();
        for i in 0..pts.len() - 1 {
            let (a, b) = (pts[i].truncate(), pts[i + 1].truncate());
            let (lo, hi) = (a.min(b), a.max(b));
            let (c0, c1) = (cell(lo), cell(hi));
            for y in c0 / k..=c1 / k {
                for x in c0 % k..=c1 % k {
                    for &(r2, j) in &cells[x + k * y] {
                        let p2 = roads[r2].line.points();
                        assert!(
                            !crosses(a, b, p2[j].truncate(), p2[j + 1].truncate()),
                            "seed {seed}: roads {r} and {r2} cross near {a}"
                        );
                    }
                }
            }
            for y in c0 / k..=c1 / k {
                for x in c0 % k..=c1 % k {
                    cells[x + k * y].push((r, i));
                }
            }
        }
    }

    // Junction angles and road lengths.
    let min_angle = c.streets.min_angle_deg.to_radians();
    let mut leaving: Vec<Vec<DVec2>> = vec![Vec::new(); nodes.len()];
    for road in roads {
        let (len, line) = (road.line.length(), &road.line);
        let d = 3.0_f64.min(0.5 * len);
        let (a, b) = (line.point_at(0.0).truncate(), line.point_at(len).truncate());
        leaving[road.start as usize].push((line.point_at(d).truncate() - a).normalize());
        leaving[road.end as usize].push((line.point_at(len - d).truncate() - b).normalize());
    }
    for (n, dirs) in leaving.iter().enumerate() {
        for i in 0..dirs.len() {
            for j in i + 1..dirs.len() {
                let angle = dirs[i].angle_to(dirs[j]).abs();
                assert!(
                    angle >= min_angle - 0.05,
                    "seed {seed}: node {n} at {} has streets {:.1}° apart",
                    nodes[n].position,
                    angle.to_degrees()
                );
            }
        }
        assert!(dirs.len() <= 5, "seed {seed}: node {n} has {} streets", dirs.len());
    }

    // Connected: every node reachable from the first arterial.
    let first = roads.iter().position(|r| r.class == RoadClass::Arterial).expect("an arterial");
    let mut seen = vec![false; nodes.len()];
    let mut stack = vec![roads[first].start];
    seen[roads[first].start as usize] = true;
    while let Some(n) = stack.pop() {
        for r in roads {
            for (a, b) in [(r.start, r.end), (r.end, r.start)] {
                if a == n && !seen[b as usize] {
                    seen[b as usize] = true;
                    stack.push(b);
                }
            }
        }
    }
    assert!(seen.iter().all(|&s| s), "seed {seed}: {} nodes cut off", seen.iter().filter(|s| !**s).count());

    for (k, road) in roads.iter().enumerate() {
        let section = net.section(k);
        let ring = nodes[road.start as usize].kind == NodeKind::Roundabout
            && nodes[road.end as usize].kind == NodeKind::Roundabout
            && section.one_way();
        let class = c.classes.class(road.class);
        let line = &road.line;
        let len = line.length();
        let pts = line.points();
        assert!((section.width() - road.width).abs() < 1e-9);
        if ring {
            // Counter-clockwise.
            assert!(line.curvature_at(0.5 * len) > 0.0, "seed {seed} road {k}: ring turns right");
        } else {
            assert!(len >= c.streets.min_length - 1e-6, "seed {seed} road {k}: {len:.1} m long");
        }
        // Continuous: the road meets its nodes.
        let (z0, z1) = (nodes[road.start as usize].position.z, nodes[road.end as usize].position.z);
        assert!((pts[0].z - z0).abs() < 1e-9 && (pts[pts.len() - 1].z - z1).abs() < 1e-9, "seed {seed} road {k}");
        // Grade.
        for w2 in pts.windows(2) {
            let ds = w2[0].truncate().distance(w2[1].truncate());
            let grade = (w2[1].z - w2[0].z).abs() / ds;
            assert!(grade <= class.max_grade + 1e-6, "seed {seed} road {k}: grade {grade:.3} at {}", w2[0]);
        }
        // The terrain follows the carriageway, which is asphalt.
        for s in [0.3 * len, 0.5 * len, 0.7 * len] {
            let p = line.point_at(s);
            assert!((t.height(p.x, p.y) - p.z).abs() < 0.1, "seed {seed} road {k}: surface off at {p}");
            if net.on_road(p.truncate()).is_some_and(|rp| rp.road as usize == k) {
                assert_eq!(t.material(p.x, p.y), MaterialId::ASPHALT, "seed {seed} road {k} at {p}");
            }
        }
    }
}

#[test]
fn urban_map_invariants() {
    for seed in 1..=6 {
        check_invariants(&UrbanPreset::Training.config(), seed);
    }
}

#[test]
#[ignore = "slow in debug builds; run with --release --ignored"]
fn showcase_map_invariants_and_timing() {
    let c = UrbanPreset::Showcase.config();
    let t = std::time::Instant::now();
    let (_, stats) = urban::generate(&c, 7).unwrap();
    let elapsed = t.elapsed().as_secs_f64();
    eprintln!("showcase: {elapsed:.2} s {stats:#?}");
    let t = std::time::Instant::now();
    urban::generate(&UrbanPreset::Training.config(), 7).unwrap();
    let training = t.elapsed().as_secs_f64();
    eprintln!("training: {training:.2} s");
    if !cfg!(debug_assertions) {
        assert!(elapsed < 15.0 && training < 2.0, "showcase {elapsed:.1} s, training {training:.2} s");
    }
    check_invariants(&c, 7);
}

#[test]
fn config_round_trips_and_is_validated() {
    for c in [UrbanConfig::training(), UrbanConfig::showcase()] {
        let text = toml::to_string(&c).unwrap();
        assert_eq!(toml::from_str::<UrbanConfig>(&text).unwrap(), c);
    }
    let c: UrbanConfig = toml::from_str("size = 512.0\n[city]\nradius = 200.0").unwrap();
    assert_eq!((c.size, c.city.radius), (512.0, 200.0));
    assert!(toml::from_str::<UrbanConfig>("sise = 256.0").is_err());
    let mut bad = UrbanConfig::training();
    bad.streets.min_length = 1.0;
    assert!(matches!(urban::generate(&bad, 0), Err(ProcgenError::Config(_))));
    let o = serde_json::json!({"classes": {"local": {"lane_width": 3.2}}});
    let c = UrbanConfig::from_preset(UrbanPreset::Training, Some(&o)).unwrap();
    assert_eq!((c.classes.local.lane_width, c.classes.local.max_grade), (3.2, 0.12));
}

#[test]
fn cache_keeps_the_sections() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("procgen-cache-urban");
    let _ = std::fs::remove_dir_all(&dir);
    let cache = MapCache::new(&dir);
    let c = UrbanConfig {
        size: 512.0,
        city: urban::CityConfig { radius: 200.0, ..Default::default() },
        ..UrbanConfig::training()
    };
    let first = cache.urban(&c, 5).unwrap();
    assert!(first.generated.is_some());
    let second = cache.urban(&c, 5).unwrap();
    assert!(second.generated.is_none(), "second request must hit the cache");
    assert_eq!(second.hash, first.hash);
    let (a, b) = (first.world.roads(), second.world.roads());
    assert!(b.has_sections() && !b.is_empty());
    assert!((0..a.roads().len()).all(|i| a.section(i) == b.section(i)));
    std::fs::remove_dir_all(&dir).unwrap();
}
