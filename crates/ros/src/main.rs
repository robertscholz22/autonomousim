//! `autonomousim-ros`: runs a simulation as a ROS 2 node.
//!
//!   autonomousim-ros run --scenario assets/scenarios/hover.toml
//!   autonomousim-ros run --policy runs/<run>/policy.json --map-seed 1000 --fast

use anyhow::Context;
use autonomousim_ros::bridge::{Bridge, BridgeConfig, Pacing};
use autonomousim_sim::Scenario;
use autonomousim_sim::policy::PolicyFile;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(version, about = "ROS 2 bridge for autonomousim")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a scenario (or a trained policy's scenario, flown by the policy) as a ROS 2 node.
    Run(RunArgs),
}

#[derive(clap::Args)]
struct RunArgs {
    /// Scenario file (TOML).
    #[arg(long, conflicts_with = "policy", required_unless_present = "policy")]
    scenario: Option<PathBuf>,
    /// An exported policy (policy.json): its scenario, its group flown by the policy.
    #[arg(long)]
    policy: Option<PathBuf>,
    /// With --policy: the map pool seed (one map; training maps are 0.., evaluation 1000..).
    #[arg(long, default_value_t = 1000)]
    map_seed: u64,
    /// Episode seed of the first episode.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// DDS domain (default: $ROS_DOMAIN_ID or 0).
    #[arg(long)]
    domain: Option<u16>,
    /// Run as fast as possible instead of in real time.
    #[arg(long)]
    fast: bool,
    /// Real-time factor (simulated seconds per wall-clock second).
    #[arg(long, default_value_t = 1.0)]
    rate: f64,
    /// Odometry and /tf rate (Hz; 0: the policy rate).
    #[arg(long, default_value_t = 0)]
    odom_hz: u32,
    /// Episode length (s; default: the policy's episode time, else until every agent is disabled).
    #[arg(long)]
    episode_time: Option<f64>,
    /// Simulated time (s) without a command after which an agent holds still.
    #[arg(long, default_value_t = 0.5)]
    command_timeout: f64,
    /// Wait before each policy step until every commanded agent has sent a command since the
    /// last /clock (at most this many wall-clock seconds; 0: no lockstep).
    #[arg(long, default_value_t = 0.0, value_name = "TIMEOUT")]
    lockstep: f64,
    /// Stop after this much simulated time (s; default: run until interrupted).
    #[arg(long)]
    duration: Option<f64>,
}

fn run(args: RunArgs) -> anyhow::Result<()> {
    let (scenario, policy) = match (&args.scenario, &args.policy) {
        (Some(path), _) => (Scenario::load(path).with_context(|| format!("loading {}", path.display()))?, None),
        (None, Some(path)) => {
            let file = PolicyFile::read(path).with_context(|| format!("reading {}", path.display()))?;
            let mut sc = file.scenario.clone();
            sc.map.set_pool(args.map_seed, 1, true);
            (sc, Some(file))
        }
        (None, None) => unreachable!("clap requires one"),
    };
    let domain_id = match args.domain {
        Some(d) => d,
        None => std::env::var("ROS_DOMAIN_ID").ok().and_then(|s| s.parse().ok()).unwrap_or(0),
    };
    if !args.rate.is_finite() || args.rate <= 0.0 {
        anyhow::bail!("--rate must be positive");
    }
    let config = BridgeConfig {
        domain_id,
        pacing: if args.fast { Pacing::Fast } else { Pacing::Realtime(args.rate) },
        odom_hz: args.odom_hz,
        episode_time: args.episode_time.or(policy.as_ref().map(|p| p.episode_time)),
        seed: args.seed,
        command_timeout: args.command_timeout,
        lockstep: (args.lockstep > 0.0).then_some(args.lockstep),
    };
    eprintln!("compiling the scenario (maps are generated or loaded from the cache)...");
    let compiled = Arc::new(scenario.compile()?);
    let mut bridge = Bridge::new(compiled.clone(), config)?;
    if let Some(file) = &policy {
        let group = compiled
            .group_index(&file.group)
            .with_context(|| format!("the scenario has no agent group {:?}", file.group))?;
        bridge.fly(group, file.policy()?)?;
    }
    eprintln!(
        "bridging on DDS domain {domain_id}: {} agent(s), physics {} Hz, policy {} Hz{}",
        bridge.world().agents().iter().filter(|a| !compiled.groups[a.group].scripted()).count(),
        compiled.spec.physics_hz,
        compiled.spec.policy_hz,
        if policy.is_some() { ", flown by the policy" } else { "" }
    );
    if args.lockstep > 0.0 {
        eprintln!("lockstep: each step waits for the commands of every agent not flown by the policy");
    }
    let until = args.duration.unwrap_or(f64::INFINITY);
    let mut report = 10.0;
    while bridge.time() < until - 1e-9 {
        bridge.step();
        if bridge.time() >= report {
            eprintln!("t = {:.0} s, episode {}", bridge.time(), bridge.episodes() + 1);
            report += 10.0;
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Run(args) => run(args),
    }
}
