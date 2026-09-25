//! Many worlds of one scenario, stepped in parallel on a dedicated thread pool.
//!
//! Outputs are flat arrays per agent group, `[num_envs, count, dim]` in row-major order, so
//! Python can view them without copying. Each world first writes its outputs into its own
//! buffers (in parallel); they are then copied into the batch arrays in world order, so the
//! results do not depend on the thread count.
//!
//! World `i` has the base seed `seed/env/i`: its episodes do not depend on the number of
//! worlds in the batch.

use crate::SimError;
use crate::record::Recorder;
use crate::scenario::{CompiledScenario, Scenario};
use crate::world::{STATE_DIM, WorldInstance};
use autonomousim_core::rng::Seed;
use rayon::prelude::*;
use std::sync::Arc;

/// One world with its output buffers (one per group).
struct Slot {
    world: WorldInstance,
    obs: Vec<Vec<f32>>,
    state: Vec<Vec<f64>>,
    events: Vec<Vec<u32>>,
    recorder: Option<Recorder>,
}

impl Slot {
    fn write_outputs(&mut self) {
        for g in 0..self.obs.len() {
            self.world.observe(g, &mut self.obs[g]);
            self.world.write_state(g, &mut self.state[g]);
            self.world.write_events(g, &mut self.events[g]);
        }
    }

    fn step(&mut self) {
        match &mut self.recorder {
            Some(r) => {
                r.on_actions(&self.world);
                self.world.step_with(&mut |w| r.on_tick(w));
            }
            None => self.world.step(),
        }
        self.write_outputs();
    }

    fn reset(&mut self, seed: Option<u64>) {
        self.world.reset(seed);
        if let Some(r) = &mut self.recorder {
            r.on_reset(&self.world);
        }
        self.write_outputs();
    }
}

pub struct BatchSim {
    scenario: Arc<CompiledScenario>,
    slots: Vec<Slot>,
    pool: rayon::ThreadPool,
    obs: Vec<Vec<f32>>,
    state: Vec<Vec<f64>>,
    events: Vec<Vec<u32>>,
}

impl BatchSim {
    /// `num_envs` worlds of `scenario`, reset to their first episode. `num_threads = 0` uses
    /// one thread per logical CPU.
    pub fn new(scenario: Scenario, num_envs: usize, seed: u64, num_threads: usize) -> Result<Self, SimError> {
        Self::from_compiled(Arc::new(scenario.compile()?), num_envs, seed, num_threads)
    }

    pub fn from_compiled(
        scenario: Arc<CompiledScenario>,
        num_envs: usize,
        seed: u64,
        num_threads: usize,
    ) -> Result<Self, SimError> {
        if num_envs == 0 {
            return Err(SimError::Scenario("a batch needs at least one world".into()));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("autonomousim-{i}"))
            .build()
            .map_err(|e| SimError::Scenario(format!("thread pool: {e}")))?;
        let groups = &scenario.groups;
        let obs_len: Vec<usize> = groups.iter().map(|g| g.spec.count * g.obs_dim()).collect();
        let state_len: Vec<usize> = groups.iter().map(|g| g.spec.count * STATE_DIM).collect();
        let events_len: Vec<usize> = groups.iter().map(|g| g.spec.count).collect();
        let base = Seed::from_u64(seed).child("env");
        let slots: Vec<Slot> = pool.install(|| {
            (0..num_envs)
                .into_par_iter()
                .map(|i| {
                    let mut s = Slot {
                        world: WorldInstance::new(scenario.clone(), base.child_index(i as u64)),
                        obs: obs_len.iter().map(|&n| vec![0.0; n]).collect(),
                        state: state_len.iter().map(|&n| vec![0.0; n]).collect(),
                        events: events_len.iter().map(|&n| vec![0; n]).collect(),
                        recorder: None,
                    };
                    s.write_outputs();
                    s
                })
                .collect()
        });
        let mut b = Self {
            obs: obs_len.iter().map(|&n| vec![0.0; n * num_envs]).collect(),
            state: state_len.iter().map(|&n| vec![0.0; n * num_envs]).collect(),
            events: events_len.iter().map(|&n| vec![0; n * num_envs]).collect(),
            scenario,
            slots,
            pool,
        };
        b.gather(None);
        Ok(b)
    }

    /// Copy the outputs of the worlds in `mask` (all if `None`) into the batch arrays.
    fn gather(&mut self, mask: Option<&[bool]>) {
        for (i, s) in self.slots.iter().enumerate() {
            if mask.is_some_and(|m| !m[i]) {
                continue;
            }
            for g in 0..s.obs.len() {
                copy_row(&mut self.obs[g], &s.obs[g], i);
                copy_row(&mut self.state[g], &s.state[g], i);
                copy_row(&mut self.events[g], &s.events[g], i);
            }
        }
    }

    /// One policy step of every world. `actions[g]` holds the normalised actions of group `g`
    /// (`[num_envs, count, act_dim]`).
    pub fn step(&mut self, actions: &[&[f32]]) {
        let n = self.slots.len();
        assert_eq!(actions.len(), self.scenario.groups.len(), "one action array per group");
        let dims: Vec<usize> = self.scenario.groups.iter().map(|g| g.spec.count * g.act_dim()).collect();
        for (g, (a, d)) in actions.iter().zip(&dims).enumerate() {
            assert_eq!(a.len(), n * d, "action array of group {:?}", self.scenario.groups[g].spec.name);
        }
        let slots = &mut self.slots;
        self.pool.install(|| {
            slots.par_iter_mut().enumerate().for_each(|(i, s)| {
                for (g, (a, d)) in actions.iter().zip(&dims).enumerate() {
                    s.world.set_actions(g, &a[i * d..(i + 1) * d]);
                }
                s.step();
            });
        });
        self.gather(None);
    }

    /// Reset the worlds in `mask` (all if `None`). With `seeds`, world `i` starts the first
    /// episode of `seeds[i]`; otherwise its next episode.
    pub fn reset(&mut self, mask: Option<&[bool]>, seeds: Option<&[u64]>) {
        let n = self.slots.len();
        if let Some(m) = mask {
            assert_eq!(m.len(), n, "reset mask length");
        }
        if let Some(s) = seeds {
            assert_eq!(s.len(), n, "reset seeds length");
        }
        let slots = &mut self.slots;
        self.pool.install(|| {
            slots.par_iter_mut().enumerate().for_each(|(i, s)| {
                if mask.is_none_or(|m| m[i]) {
                    s.reset(seeds.map(|s| s[i]));
                }
            });
        });
        self.gather(mask);
    }

    /// Stop the agents of group `g` where `mask` (`[num_envs, count]`) is true, for the rest
    /// of their episodes ([`WorldInstance::disable_agent`]).
    pub fn disable(&mut self, g: usize, mask: &[bool]) {
        let group = &self.scenario.groups[g];
        let (first, count) = (group.first_agent, group.spec.count);
        assert_eq!(mask.len(), self.slots.len() * count, "disable mask of group {:?}", group.spec.name);
        for (s, m) in self.slots.iter_mut().zip(mask.chunks_exact(count)) {
            for (k, _) in m.iter().enumerate().filter(|(_, d)| **d) {
                s.world.disable_agent(first + k);
            }
        }
    }

    // ------------------------------------------------------------------------ outputs

    /// Observations of group `g`, `[num_envs, count, obs_dim]`.
    pub fn obs(&self, g: usize) -> &[f32] {
        &self.obs[g]
    }

    /// State rows ([`STATE_FIELDS`](crate::STATE_FIELDS)) of group `g`,
    /// `[num_envs, count, STATE_DIM]`.
    pub fn state(&self, g: usize) -> &[f64] {
        &self.state[g]
    }

    /// Event bits of group `g` during the last policy step, `[num_envs, count]`.
    pub fn events(&self, g: usize) -> &[u32] {
        &self.events[g]
    }

    // ------------------------------------------------------------------------ access

    pub fn num_envs(&self) -> usize {
        self.slots.len()
    }

    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    pub fn scenario(&self) -> &Arc<CompiledScenario> {
        &self.scenario
    }

    pub fn world(&self, i: usize) -> &WorldInstance {
        &self.slots[i].world
    }

    /// Mutable access to a world (e.g. to set goals). Call [`refresh`](Self::refresh)
    /// afterwards if the outputs should reflect the change before the next step.
    pub fn world_mut(&mut self, i: usize) -> &mut WorldInstance {
        &mut self.slots[i].world
    }

    /// Rewrite the outputs of world `i` from its current state.
    pub fn refresh(&mut self, i: usize) {
        self.slots[i].write_outputs();
        let s = &self.slots[i];
        for g in 0..s.obs.len() {
            copy_row(&mut self.obs[g], &s.obs[g], i);
            copy_row(&mut self.state[g], &s.state[g], i);
            copy_row(&mut self.events[g], &s.events[g], i);
        }
    }

    /// Record world `i` from now on (replacing and returning a previous recorder). The
    /// recorder sees the current state as if the world had just been reset.
    pub fn attach_recorder(&mut self, i: usize, mut recorder: Recorder) -> Option<Recorder> {
        recorder.on_reset(&self.slots[i].world);
        self.slots[i].recorder.replace(recorder)
    }

    /// Stop recording world `i`; call [`Recorder::finish`] on the result.
    pub fn detach_recorder(&mut self, i: usize) -> Option<Recorder> {
        self.slots[i].recorder.take()
    }
}

fn copy_row<T: Copy>(dst: &mut [T], src: &[T], i: usize) {
    let n = src.len();
    dst[i * n..(i + 1) * n].copy_from_slice(src);
}
