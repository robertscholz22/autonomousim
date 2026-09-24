//! A trained policy (exported by `examples/export_policy.py`) flying the agents of its group in
//! the live simulation, at the policy rate it was trained at. The keyboard can take over the
//! followed agent (T); the others stay with the policy.

use autonomousim_sim::policy::{Policy, PolicyFile};
use autonomousim_sim::{Events, WorldInstance};

/// Events after which an agent's episode is over (as in the Python tasks).
pub const EPISODE_END: Events = Events(Events::TERMINAL.0 | Events::DISABLED.0 | Events::FINISHED.0);
/// Simulated seconds shown after the episode ended, before the next one starts.
pub const RESET_DELAY: f64 = 1.5;

#[derive(Debug)]
pub struct Autopilot {
    pub name: String,
    policy: Policy,
    pub group: usize,
    /// Seconds after which the task truncates an episode.
    pub episode_time: f64,
    /// The policy flies the followed agent too; otherwise the keyboard does.
    pub flies_pilot: bool,
    obs: Vec<f32>,
    action: Vec<f32>,
    /// Simulated time at which the current episode ended.
    pub ended: Option<f64>,
    /// Episodes flown by the policy (per agent) and how many of them reached the last goal.
    pub flown: u64,
    pub finished: u64,
}

impl Autopilot {
    /// The policy of `file` for the agents of `world`'s group of the same name.
    pub fn new(file: &PolicyFile, world: &WorldInstance) -> anyhow::Result<Self> {
        let policy = file.policy()?;
        let sc = world.scenario();
        let group = sc
            .group_index(&file.group)
            .ok_or_else(|| anyhow::anyhow!("the scenario has no agent group {:?}", file.group))?;
        let g = &sc.groups[group];
        if (g.obs_dim(), g.act_dim()) != (policy.obs_dim(), policy.act_dim()) {
            anyhow::bail!(
                "group {:?} observes {} values and takes {} actions, the policy {} and {}",
                file.group,
                g.obs_dim(),
                g.act_dim(),
                policy.obs_dim(),
                policy.act_dim()
            );
        }
        Ok(Self {
            name: file.name.clone(),
            obs: vec![0.0; g.spec.count * g.obs_dim()],
            action: vec![0.0; g.act_dim()],
            policy,
            group,
            episode_time: file.episode_time,
            flies_pilot: true,
            ended: None,
            flown: 0,
            finished: 0,
        })
    }

    /// Before a physics tick: at policy-step boundaries, set the actions of the group's agents
    /// from their observations, except for `manual` (flown from the keyboard).
    pub fn before_tick(&mut self, world: &mut WorldInstance, manual: Option<usize>) {
        if !world.clock().tick.is_multiple_of(u64::from(world.scenario().decimation)) {
            return;
        }
        world.observe(self.group, &mut self.obs);
        let first = world.scenario().groups[self.group].first_agent;
        for (k, obs) in self.obs.chunks_exact(self.policy.obs_dim()).enumerate() {
            if Some(first + k) == manual {
                continue;
            }
            self.policy.act(obs, &mut self.action);
            world.set_action(first + k, &self.action);
        }
    }

    /// Whether the episode of every agent of the group is over (`latched`: events per agent
    /// since the reset) or the task would have truncated it.
    pub fn episode_over(&self, world: &WorldInstance, latched: &[Events]) -> bool {
        let g = &world.scenario().groups[self.group];
        world.time() >= self.episode_time
            || latched[g.first_agent..g.first_agent + g.spec.count].iter().all(|e| e.intersects(EPISODE_END))
    }

    /// Count the ending episode of every agent the policy flew.
    pub fn count_episode(&mut self, world: &WorldInstance, latched: &[Events], manual: Option<usize>) {
        let g = &world.scenario().groups[self.group];
        for i in g.first_agent..g.first_agent + g.spec.count {
            if Some(i) != manual {
                self.flown += 1;
                self.finished += u64::from(latched[i].contains(Events::FINISHED));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sim::Sim;
    use autonomousim_core::rng::Seed;
    use autonomousim_sim::policy::PolicyFile;
    use autonomousim_sim::{Scenario, WorldInstance};
    use serde_json::json;
    use std::sync::Arc;

    /// Two drones on a flat test world and a policy that always flies forward at half speed.
    fn forward_policy(episode_time: f64) -> (PolicyFile, WorldInstance) {
        let scenario = json!({
            "name": "t",
            "map": { "type": "testworld", "kind": "flat", "size": 400.0 },
            "groups": [{ "name": "agent", "count": 2, "vehicle": "iris_like", "action_mode": "velocity" }],
        });
        let compiled = serde_json::from_value::<Scenario>(scenario.clone()).unwrap().compile().unwrap();
        let (obs, act) = (compiled.groups[0].obs_dim(), compiled.groups[0].act_dim());
        let mut bias = vec![0.0; act];
        bias[0] = 0.5;
        let file = json!({
            "format": "autonomousim-policy", "version": 1, "name": "forward", "env_id": "test",
            "algo": "ppo", "group": "agent", "episode_time": episode_time, "scenario": scenario,
            "obs_norm": { "mean": vec![0.0; obs], "var": vec![1.0; obs], "clip": 10.0, "eps": 1e-8 },
            "layers": [{ "shape": [act, obs], "weight": vec![0.0; act * obs], "bias": bias, "activation": "identity" }],
            "output": "clip",
        });
        let file = PolicyFile::from_json(&file.to_string()).unwrap();
        (file, WorldInstance::new(Arc::new(compiled), Seed::from_u64(3)))
    }

    #[test]
    fn the_policy_flies_all_agents_until_the_pilot_is_taken_over() {
        let (file, world) = forward_policy(60.0);
        let start: Vec<_> = world.agents().iter().map(|a| a.vehicle.position()).collect();
        let mut s = Sim::new(world);
        s.autopilot = Some(super::Autopilot::new(&file, &s.world).unwrap());
        s.autopilot.as_mut().unwrap().flies_pilot = false;
        for _ in 0..30 {
            s.advance(0.1);
        }
        // Agent 1 flies forward (the action held is the policy's); the pilot hovers on the keys.
        assert_eq!(s.world.agent(1).action.as_slice(), &[0.5, 0.0, 0.0, 0.0]);
        assert!(s.world.agent(0).action.iter().all(|&a| a == 0.0));
        let moved = |s: &Sim, i: usize| (s.world.agent(i).vehicle.position() - start[i]).truncate().length();
        assert!(moved(&s, 1) > 5.0 && moved(&s, 0) < 0.5, "{} {}", moved(&s, 1), moved(&s, 0));
        // Handed back, the pilot follows the policy too.
        s.autopilot.as_mut().unwrap().flies_pilot = true;
        s.advance(0.1);
        assert_eq!(s.world.agent(0).action.as_slice(), &[0.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn episodes_restart_after_the_task_time_only_while_the_policy_flies() {
        let (file, world) = forward_policy(1.0);
        let mut s = Sim::new(world);
        s.autopilot = Some(super::Autopilot::new(&file, &s.world).unwrap());
        for _ in 0..20 {
            s.advance(0.1);
        }
        // Over after 1 s, shown for RESET_DELAY more, then the next episode starts.
        assert_eq!(s.episodes, 1);
        for _ in 0..6 {
            s.advance(0.1);
        }
        let a = s.autopilot.as_ref().unwrap();
        assert_eq!((s.episodes, a.flown, a.finished, a.ended), (2, 2, 0, None));
        assert!(s.world.time() < 0.2);
        // Flying the pilot by hand, the episode goes on.
        s.autopilot.as_mut().unwrap().flies_pilot = false;
        for _ in 0..40 {
            s.advance(0.1);
        }
        assert_eq!(s.episodes, 2);
    }

    #[test]
    fn a_policy_for_other_dimensions_is_rejected() {
        let (mut file, world) = forward_policy(1.0);
        file.group = "other".into();
        assert!(super::Autopilot::new(&file, &world).is_err());
        let (mut file, world) = forward_policy(1.0);
        let layer = &mut file.layers[0];
        layer.shape[0] -= 1;
        layer.bias.pop();
        layer.weight.truncate(layer.shape[0] * layer.shape[1]);
        assert!(super::Autopilot::new(&file, &world).unwrap_err().to_string().contains("actions"));
    }

    /// Success rate of an exported policy flying in Rust, for comparison with the Python
    /// evaluation: `AUTONOMOUSIM_POLICY=$PWD/runs/<run>/policy.json cargo test -p
    /// autonomousim-viewer --release -- --ignored --nocapture exported_policy`.
    #[test]
    #[ignore = "needs an exported policy (AUTONOMOUSIM_POLICY)"]
    fn exported_policy_success_rate() {
        let path = std::env::var("AUTONOMOUSIM_POLICY").expect("AUTONOMOUSIM_POLICY");
        let file = PolicyFile::read(path).unwrap();
        // Agents per world (AUTONOMOUSIM_AGENTS, default 1 as in training): they can collide.
        let agents = std::env::var("AUTONOMOUSIM_AGENTS").map_or(1, |n| n.parse().unwrap());
        for map_seed in [1000, 1001, 1002, 1003] {
            let sc = crate::policy_scenario(&file, map_seed, Some(agents), true).unwrap();
            let world = WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(0));
            let mut s = Sim::new(world);
            s.autopilot = Some(super::Autopilot::new(&file, &s.world).unwrap());
            while s.autopilot.as_ref().unwrap().flown < 64 {
                s.advance(0.1);
            }
            let a = s.autopilot.as_ref().unwrap();
            println!("map {map_seed}: {} of {} episodes reached the last goal", a.finished, a.flown);
        }
    }
}
