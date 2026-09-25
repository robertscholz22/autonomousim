//! `autonomousim` command-line tool.

use anyhow::{Context, bail};
use autonomousim_procgen::rural::{self, RuralConfig, RuralPreset};
use autonomousim_procgen::wild::{self, WildConfig, WildPreset};
use autonomousim_procgen::{MapCache, RuralStats, WildStats};
use autonomousim_world::{RoadClass, StaticWorld, mapfile, obstacles::tags};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;

mod preview;

#[derive(Parser)]
#[command(name = "autonomousim", version, about = "autonomousim command-line tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print version information.
    Version,
    /// Generate a wild or rural map (or load it from the map cache) and print statistics.
    Mapgen(MapgenArgs),
    /// Print the content hash of map files (verifying each).
    MapHash {
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },
    /// Describe a map file.
    MapInfo { file: PathBuf },
}

#[derive(Args)]
struct MapgenArgs {
    #[arg(long, value_enum, default_value_t = Generator::Wild)]
    generator: Generator,
    /// Starting configuration: training (512 m) or showcase (2 km); wild maps also have
    /// offroad (512 m, gentle).
    #[arg(long, default_value = "training")]
    preset: String,
    /// TOML file whose values override the preset (same layout as `--print-config`).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Map edge length in metres (overrides preset and config).
    #[arg(long)]
    size: Option<f64>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Worker threads (0 = one per logical CPU). The map does not depend on this.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Also write the map to this file.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Always generate; do not read or write the map cache.
    #[arg(long)]
    no_cache: bool,
    /// Write a top-down preview image (binary PPM).
    #[arg(long)]
    preview: Option<PathBuf>,
    /// Map cells per preview pixel.
    #[arg(long, default_value_t = 1)]
    preview_stride: usize,
    /// Print the effective configuration as TOML and exit.
    #[arg(long)]
    print_config: bool,
    /// Print statistics as JSON.
    #[arg(long)]
    json: bool,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Version => println!("autonomousim {}", env!("CARGO_PKG_VERSION")),
        Command::Mapgen(args) => mapgen(args)?,
        Command::MapHash { files } => {
            for f in files {
                let (_, hash) = mapfile::load(&f).with_context(|| format!("reading {}", f.display()))?;
                println!("{hash}  {}", f.display());
            }
        }
        Command::MapInfo { file } => {
            let t = Instant::now();
            let (world, hash) = mapfile::load(&file).with_context(|| format!("reading {}", file.display()))?;
            println!("{}", info(&world, &hash.to_string()));
            println!("loaded in {:.3} s", t.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

#[derive(Clone, Copy, ValueEnum)]
enum Generator {
    Wild,
    Rural,
}

/// The effective configuration of either generator.
enum Config {
    Wild(Box<WildConfig>),
    Rural(Box<RuralConfig>),
}

/// Statistics of either generator.
#[derive(serde::Serialize)]
#[serde(untagged)]
enum Stats {
    Wild(WildStats),
    Rural(RuralStats),
}

fn config(args: &MapgenArgs) -> anyhow::Result<Config> {
    let overrides = match &args.config {
        Some(path) => {
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            Some(toml::from_str::<serde_json::Value>(&text).with_context(|| format!("parsing {}", path.display()))?)
        }
        None => None,
    };
    let preset = |e: String| anyhow::anyhow!(e);
    Ok(match args.generator {
        Generator::Wild => {
            let mut c =
                WildConfig::from_preset(args.preset.parse::<WildPreset>().map_err(preset)?, overrides.as_ref())?;
            if let Some(size) = args.size {
                c.size = size;
            }
            c.validate()?;
            Config::Wild(Box::new(c))
        }
        Generator::Rural => {
            let mut c =
                RuralConfig::from_preset(args.preset.parse::<RuralPreset>().map_err(preset)?, overrides.as_ref())?;
            if let Some(size) = args.size {
                c.size = size;
            }
            c.validate()?;
            Config::Rural(Box::new(c))
        }
    })
}

fn mapgen(args: MapgenArgs) -> anyhow::Result<()> {
    let config = config(&args)?;
    if args.print_config {
        match &config {
            Config::Wild(c) => print!("{}", toml::to_string(c)?),
            Config::Rural(c) => print!("{}", toml::to_string(c)?),
        }
        return Ok(());
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build()?;
    let t = Instant::now();
    let (world, hash, stats, source) = pool.install(|| -> anyhow::Result<_> {
        if args.no_cache {
            let (world, stats) = match &config {
                Config::Wild(c) => wild::generate(c, args.seed).map(|(w, s)| (w, Stats::Wild(s)))?,
                Config::Rural(c) => rural::generate(c, args.seed).map(|(w, s)| (w, Stats::Rural(s)))?,
            };
            let hash = world.content_hash();
            Ok((world, hash, Some(stats), "generated (cache disabled)".to_owned()))
        } else {
            let Some(cache) = MapCache::user() else { bail!("no cache directory (set AUTONOMOUSIM_MAP_CACHE)") };
            let (world, hash, path, stats) = match &config {
                Config::Wild(c) => {
                    let c = cache.wild(c, args.seed)?;
                    (c.world, c.hash, c.path, c.generated.map(Stats::Wild))
                }
                Config::Rural(c) => {
                    let c = cache.rural(c, args.seed)?;
                    (c.world, c.hash, c.path, c.generated.map(Stats::Rural))
                }
            };
            let source = match stats {
                Some(_) => format!("generated, cached at {}", path.display()),
                None => format!("loaded from {}", path.display()),
            };
            Ok((world, hash, stats, source))
        }
    })?;
    let total = t.elapsed().as_secs_f64();
    if let Some(out) = &args.out {
        mapfile::save(&world, out).with_context(|| format!("writing {}", out.display()))?;
    }
    if let Some(p) = &args.preview {
        preview::write_ppm(&world, args.preview_stride.max(1), p)
            .with_context(|| format!("writing {}", p.display()))?;
    }
    if args.json {
        let v = serde_json::json!({
            "hash": hash.to_string(),
            "seed": args.seed,
            "source": source,
            "seconds": total,
            "threads": pool.current_num_threads(),
            "stats": stats,
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    match &stats {
        Some(Stats::Wild(s)) => print_stats(s),
        Some(Stats::Rural(s)) => print_rural_stats(s),
        None => {}
    }
    println!("{}", info(&world, &hash.to_string()));
    println!("{source} in {total:.2} s ({} threads)", pool.current_num_threads());
    if let Some(out) = &args.out {
        println!("written to {}", out.display());
    }
    Ok(())
}

fn print_stats(s: &WildStats) {
    println!("stages:");
    for (name, secs) in &s.stages {
        println!("  {name:<18} {secs:>7.3} s");
    }
    println!("  {:<18} {:>7.3} s", "total", s.total_seconds());
    println!(
        "erosion droplets {}, lakes {} ({} cells), trees {} ({} conifers), rocks {}",
        s.droplets, s.lakes, s.lake_cells, s.trees, s.conifers, s.rocks
    );
    let [p50, p90, p99] = s.slope_percentiles_deg;
    println!(
        "heights {:.1} … {:.1} m, slope p50 {p50:.1}°, p90 {p90:.1}°, p99 {p99:.1}°",
        s.height_range.0, s.height_range.1
    );
    let cells: usize = s.materials.iter().map(|m| m.1).sum();
    let shares: Vec<String> =
        s.materials.iter().map(|(n, c)| format!("{n} {:.1}%", 100.0 * *c as f64 / cells as f64)).collect();
    println!("materials: {}", shares.join(", "));
}

fn print_rural_stats(s: &RuralStats) {
    println!("stages:");
    for (name, secs) in &s.stages {
        println!("  {name:<18} {secs:>7.3} s");
    }
    println!("  {:<18} {:>7.3} s", "total", s.total_seconds());
    println!(
        "lakes {} ({} cells), farms {} of {} sites, {} junctions, {} parcels, {} field tracks",
        s.lakes, s.lake_cells, s.farms, s.farm_sites, s.junctions, s.parcels, s.tracks
    );
    println!("buildings {}, hedge pieces {}, fence pieces {}, trees {}", s.buildings, s.hedges, s.fences, s.trees);
    println!("heights {:.1} … {:.1} m", s.height_range.0, s.height_range.1);
    let cells: usize = s.materials.iter().map(|m| m.1).sum();
    let shares: Vec<String> =
        s.materials.iter().map(|(n, c)| format!("{n} {:.1}%", 100.0 * *c as f64 / cells as f64)).collect();
    println!("materials: {}", shares.join(", "));
}

fn info(w: &StaticWorld, hash: &str) -> String {
    let t = w.terrain();
    let (nx, ny) = t.dims();
    let (lo, hi) = t.height_range();
    let (min, max) = w.extent();
    let mut by_tag = std::collections::BTreeMap::<u16, usize>::new();
    for o in w.obstacles().obstacles() {
        *by_tag.entry(o.tag).or_default() += 1;
    }
    let tag = |t: u16| match t {
        tags::TRUNK => "trunks".to_owned(),
        tags::CANOPY => "conifer crowns".to_owned(),
        tags::CANOPY_BROADLEAF => "broadleaf crowns".to_owned(),
        tags::ROCK => "rocks".to_owned(),
        tags::PILLAR => "pillars".to_owned(),
        tags::WALL => "walls".to_owned(),
        tags::HEDGE => "hedge pieces".to_owned(),
        tags::FENCE => "fence pieces".to_owned(),
        tags::BUILDING => "buildings".to_owned(),
        tags::SILO => "silos".to_owned(),
        t => format!("tag {t}"),
    };
    let obstacles: Vec<String> = by_tag.iter().map(|(t, n)| format!("{n} {}", tag(*t))).collect();
    let wet = t.water().map_or(0, |w| w.iter().filter(|v| !v.is_nan()).count());
    let net = w.roads();
    let roads = if net.is_empty() {
        "none".to_owned()
    } else {
        let classes = [(RoadClass::Paved, "paved"), (RoadClass::Gravel, "gravel"), (RoadClass::Track, "track")];
        let parts: Vec<String> = classes
            .iter()
            .filter_map(|&(class, name)| {
                let (n, len) = net
                    .roads()
                    .iter()
                    .filter(|r| r.class == class)
                    .fold((0, 0.0), |(n, l), r| (n + 1, l + r.line.length()));
                (n > 0).then(|| format!("{n} {name} ({:.2} km)", len / 1000.0))
            })
            .collect();
        format!("{}, {} nodes", parts.join(", "), net.nodes().len())
    };
    format!(
        "map {:?} (generator {} v{}, seed {})\n  extent [{:.0}, {:.0}] × [{:.0}, {:.0}] m, {nx}×{ny} vertices at {} m\n  \
         heights {lo:.1} … {hi:.1} m, {wet} water cells\n  obstacles: {}\n  roads: {roads}\n  hash {hash}",
        w.meta.name,
        w.meta.generator,
        w.meta.generator_version,
        w.meta.seed,
        min.x,
        max.x,
        min.y,
        max.y,
        t.cell_size(),
        if obstacles.is_empty() { "none".to_owned() } else { obstacles.join(", ") },
    )
}
