//! Rural generator: golden hashes (independent of the thread count), road network invariants,
//! terrain blending, cache.
//!
//! Regenerate the golden hashes after an intended change of the generator output (and bump
//! `RURAL_VERSION`) with
//! `AUTONOMOUSIM_BLESS=1 cargo test -p autonomousim-procgen --test rural`.

use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::Terrain;
use autonomousim_procgen::rural::{self, RURAL_VERSION, RuralConfig, RuralPreset};
use autonomousim_procgen::{MapCache, ProcgenError};
use autonomousim_world::{NodeKind, RoadClass, StaticWorld};
use glam::DVec2;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
struct Golden {
    rural: Vec<GoldenMap>,
}

#[derive(Clone, Serialize, Deserialize)]
struct GoldenMap {
    preset: RuralPreset,
    size: f64,
    seed: u64,
    hash: String,
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden_hashes_rural.toml")
}

fn config(preset: RuralPreset, size: f64) -> RuralConfig {
    RuralConfig { size, ..preset.config() }
}

fn generate_with_threads(c: &RuralConfig, seed: u64, threads: usize) -> StaticWorld {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| rural::generate(c, seed)).unwrap().0
}

#[test]
fn golden_hashes_do_not_depend_on_the_thread_count() {
    let bless = std::env::var_os("AUTONOMOUSIM_BLESS").is_some();
    let golden: Golden = match std::fs::read_to_string(golden_path()) {
        Ok(text) => toml::from_str(&text).unwrap(),
        Err(_) if bless => Golden {
            rural: [
                (RuralPreset::Training, 512.0, 1),
                (RuralPreset::Training, 512.0, 2),
                (RuralPreset::Showcase, 1024.0, 3),
            ]
            .map(|(preset, size, seed)| GoldenMap { preset, size, seed, hash: String::new() })
            .to_vec(),
        },
        Err(e) => panic!("fixtures/golden_hashes_rural.toml: {e}"),
    };
    let mut blessed = Vec::new();
    for g in &golden.rural {
        let c = config(g.preset, g.size);
        let one = generate_with_threads(&c, g.seed, 1).content_hash().to_string();
        let many = generate_with_threads(&c, g.seed, 12).content_hash().to_string();
        assert_eq!(one, many, "{:?} {} m seed {}: 1 vs 12 threads", g.preset, g.size, g.seed);
        if bless {
            blessed.push(GoldenMap { hash: one, ..g.clone() });
        } else {
            assert_eq!(
                one, g.hash,
                "{:?} {} m seed {} changed (bump RURAL_VERSION and bless)",
                g.preset, g.size, g.seed
            );
        }
    }
    if bless {
        let header = "# Content hashes of generated rural maps (see crates/procgen/tests/rural.rs).\n\
                      # Regenerate: AUTONOMOUSIM_BLESS=1 cargo test -p autonomousim-procgen --test rural\n\n";
        let body = toml::to_string(&Golden { rural: blessed }).unwrap();
        std::fs::write(golden_path(), format!("{header}{body}")).unwrap();
    }
}

/// Everything the generator promises about roads, farms and the terrain under them.
fn check_invariants(c: &RuralConfig, seed: u64) {
    let (w, stats) = rural::generate(c, seed).unwrap();
    assert_eq!((w.meta.generator.as_str(), w.meta.generator_version, w.meta.seed), ("rural", RURAL_VERSION, seed));
    let net = w.roads();
    let t = w.terrain();
    assert!(stats.roads[0] >= 1 && stats.farms >= 1, "seed {seed}: {stats:?}");
    assert!(stats.farms * 2 >= stats.farm_sites, "seed {seed}: most farms connect: {stats:?}");

    // Every farm yard is reachable from both ends of the main road.
    let ends: Vec<DVec2> =
        net.nodes().iter().filter(|n| n.kind == NodeKind::End).map(|n| n.position.truncate()).collect();
    assert_eq!(ends.len(), 2);
    let yards: Vec<_> = net.nodes().iter().filter(|n| n.kind == NodeKind::Yard).collect();
    assert_eq!(yards.len(), stats.farms);
    for y in &yards {
        for &e in &ends {
            assert!(net.route(y.position.truncate(), e, 1.0).is_some(), "seed {seed}: farm at {} cut off", y.position);
        }
    }

    for (k, road) in net.roads().iter().enumerate() {
        let class = c.roads.class(road.class);
        let line = &road.line;
        let pts = line.points();
        let len = line.length();
        // Grade.
        for w2 in pts.windows(2) {
            let ds = w2[0].truncate().distance(w2[1].truncate());
            let grade = (w2[1].z - w2[0].z).abs() / ds;
            assert!(grade <= class.max_grade + 1e-6, "seed {seed} road {k}: grade {grade:.3} at {}", w2[0]);
        }
        // Curvature away from the nodes.
        let mut s = 12.0;
        while s <= len - 12.0 {
            let kappa = line.curvature_at(s).abs();
            assert!(kappa <= 1.15 / class.min_radius, "seed {seed} road {k}: radius {:.1} m at s = {s}", 1.0 / kappa);
            s += 1.0;
        }
        // Dry, blended into the terrain, and surfaced to its class.
        let expected = match road.class {
            RoadClass::Paved => MaterialId::ASPHALT,
            RoadClass::Gravel => MaterialId::GRAVEL,
            RoadClass::Track => MaterialId::DIRT,
        };
        let yard = |p: DVec2| t.material(p.x, p.y) == MaterialId::CONCRETE;
        let mut s = 0.0;
        while s <= len {
            let p = line.point_at(s);
            let h = line.heading_at(s);
            for off in [-0.5, 0.0, 0.5] {
                let q = p.truncate() + off * road.width * DVec2::new(-h.sin(), h.cos());
                assert!(t.water_level(q.x, q.y).is_none(), "seed {seed} road {k}: water at {q}");
            }
            let q = p.truncate();
            let dz = t.height(q.x, q.y) - p.z;
            // Junctions and yards mix neighbouring surfaces; elsewhere the fit is tight.
            let near_node = s < 8.0 || s > len - 8.0;
            if !near_node {
                assert!(dz.abs() < 0.08, "seed {seed} road {k}: terrain {dz:+.3} m off the road at {q}");
                let m = t.material(q.x, q.y);
                assert!(m == expected || yard(q) || m == MaterialId::ASPHALT, "seed {seed} road {k}: {m:?} at {q}");
            }
            s += 2.0;
        }
    }

    // Yards are flat concrete.
    for y in &yards {
        let p = y.position;
        assert_eq!(t.material(p.x, p.y), MaterialId::CONCRETE);
        assert!((t.height(p.x, p.y) - p.z).abs() < 0.05, "seed {seed}: yard at {p} not flat");
    }
}

#[test]
fn rural_map_invariants() {
    for seed in 1..=8 {
        check_invariants(&config(RuralPreset::Training, 512.0), seed);
    }
}

#[test]
#[ignore = "slow in debug builds; run with --release --ignored"]
fn showcase_map_invariants_and_timing() {
    let c = RuralPreset::Showcase.config();
    let t = std::time::Instant::now();
    let (_, stats) = rural::generate(&c, 7).unwrap();
    let elapsed = t.elapsed().as_secs_f64();
    eprintln!("showcase: {elapsed:.2} s {stats:#?}");
    let t = std::time::Instant::now();
    rural::generate(&RuralPreset::Training.config(), 7).unwrap();
    let training = t.elapsed().as_secs_f64();
    eprintln!("training: {training:.2} s");
    if !cfg!(debug_assertions) {
        assert!(elapsed < 20.0 && training < 1.5, "showcase {elapsed:.1} s, training {training:.2} s");
    }
    check_invariants(&c, 7);
}

#[test]
fn config_round_trips_and_is_validated() {
    for c in [RuralConfig::training(), RuralConfig::showcase()] {
        let text = toml::to_string(&c).unwrap();
        assert_eq!(toml::from_str::<RuralConfig>(&text).unwrap(), c);
    }
    let c: RuralConfig = toml::from_str("size = 256.0\n[farms]\nspacing = 100.0").unwrap();
    assert_eq!((c.size, c.farms.spacing), (256.0, 100.0));
    assert!(toml::from_str::<RuralConfig>("sise = 256.0").is_err());
    for bad in
        [RuralConfig { size: 255.0, ..RuralConfig::training() }, RuralConfig { cell: 3.0, ..RuralConfig::training() }]
    {
        assert!(matches!(rural::generate(&bad, 0), Err(ProcgenError::Config(_))));
    }
    let o = serde_json::json!({"roads": {"paved": {"width": 7.0}}});
    let c = RuralConfig::from_preset(RuralPreset::Training, Some(&o)).unwrap();
    assert_eq!((c.roads.paved.width, c.roads.paved.max_grade), (7.0, 0.08));
}

#[test]
fn cache_keeps_the_road_network() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("procgen-cache-rural");
    let _ = std::fs::remove_dir_all(&dir);
    let cache = MapCache::new(&dir);
    let c = config(RuralPreset::Training, 256.0);
    let first = cache.rural(&c, 5).unwrap();
    assert!(first.generated.is_some());
    let second = cache.rural(&c, 5).unwrap();
    assert!(second.generated.is_none(), "second request must hit the cache");
    assert_eq!(second.hash, first.hash);
    assert_eq!(second.world.roads().roads().len(), first.world.roads().roads().len());
    assert!(!second.world.roads().is_empty());
    std::fs::remove_dir_all(&dir).unwrap();
}
