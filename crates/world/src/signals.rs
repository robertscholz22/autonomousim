//! Fixed-time traffic signal controllers, derived from the lane graph (M8a step 4).
//!
//! Every [`Signal`](crate::lanes::JunctionKind::Signal) junction gets one [`Controller`]: its
//! approaches are paired into groups of opposing approaches, and each group gets a main phase of
//! the movements that cross nothing else in it (straight and right first, then left), followed
//! by protected phases for what remains (the left turns across oncoming traffic). No two
//! connectors that conflict (cross or merge; see
//! [`ConflictKind::exclusive`](crate::lanes::ConflictKind::exclusive)) are ever green in
//! the same phase.
//!
//! Phases run in turn: green, amber, then all red. The state is a pure function of the time
//! into the cycle; the simulation adds a per-episode offset per controller.

use crate::lanes::{Connector, Control, Junction, Lane, Turn};
use crate::roads::wrap_angle;
use std::f64::consts::PI;

/// Amber time (s) at the end of each phase.
pub const AMBER: f64 = 3.0;

/// All-red time (s) after each amber.
pub const ALL_RED: f64 = 2.0;

/// Green time (s) of a protected phase.
pub const PROTECTED_GREEN: f64 = 8.0;

/// Shortest green time (s) of a main phase.
pub const MIN_GREEN: f64 = 12.0;

/// Largest difference (rad) from opposite arrival headings for two approaches to share a phase.
const OPPOSING: f64 = 0.7;

/// What a signal shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Light {
    Green,
    Amber,
    Red,
}

/// A set of connectors that are green together.
#[derive(Clone, Debug, PartialEq)]
pub struct Phase {
    pub connectors: Vec<u32>,
    /// Green time (s).
    pub green: f64,
    /// Whether it is a protected phase (for movements left over from a main one).
    pub protected: bool,
}

/// The fixed-time controller of a signalized junction.
#[derive(Clone, Debug, PartialEq)]
pub struct Controller {
    /// The junction (and node) it controls.
    pub junction: u32,
    pub phases: Vec<Phase>,
    /// Amber and all-red time (s) after each phase's green.
    pub amber: f64,
    pub all_red: f64,
}

impl Controller {
    /// Length (s) of a phase including its amber and all red.
    fn span(&self, k: usize) -> f64 {
        self.phases[k].green + self.amber + self.all_red
    }

    /// Cycle length (s).
    pub fn cycle(&self) -> f64 {
        (0..self.phases.len()).map(|k| self.span(k)).sum()
    }

    /// Start (s) of phase `k` in the cycle.
    pub fn start(&self, k: usize) -> f64 {
        (0..k).map(|j| self.span(j)).sum()
    }

    /// The phase running at time `t` (s; wrapped into the cycle) and the time into it.
    pub fn active(&self, t: f64) -> (usize, f64) {
        let mut u = t.rem_euclid(self.cycle());
        for k in 0..self.phases.len() {
            let s = self.span(k);
            if u < s {
                return (k, u);
            }
            u -= s;
        }
        let k = self.phases.len() - 1;
        (k, self.span(k))
    }

    /// The light of phase `k` at time `t` (s).
    pub fn light(&self, k: usize, t: f64) -> Light {
        self.light_left(k, t).0
    }

    /// The light of phase `k` at time `t` (s) and how long it stays so (s).
    pub fn light_left(&self, k: usize, t: f64) -> (Light, f64) {
        let (a, u) = self.active(t);
        let green = self.phases[k].green;
        if a == k && u < green {
            (Light::Green, green - u)
        } else if a == k && u < green + self.amber {
            (Light::Amber, green + self.amber - u)
        } else {
            (Light::Red, (self.start(k) - t).rem_euclid(self.cycle()))
        }
    }
}

/// Target cycle length (s) for `n` phases.
fn target_cycle(n: usize) -> f64 {
    match n {
        0..=2 => 60.0,
        3 => 75.0,
        _ => 90.0,
    }
}

/// Controllers of the signalized junctions, and for each connector its (controller, phase).
pub(crate) fn build(
    lanes: &[Lane],
    connectors: &[Connector],
    junctions: &[Junction],
) -> (Vec<Controller>, Vec<Option<(u32, u8)>>) {
    let mut out = Vec::new();
    let mut of = vec![None; connectors.len()];
    for (j, junction) in junctions.iter().enumerate() {
        let signalled: Vec<usize> =
            (0..junction.approaches.len()).filter(|&a| junction.approaches[a].control == Control::Signal).collect();
        if signalled.is_empty() {
            continue;
        }
        // Arrival heading of each approach.
        let heading = |a: usize| {
            let l = &lanes[junction.approaches[a].lanes[0] as usize];
            l.line.heading_at(l.line.length())
        };
        // Pair opposing approaches (the closest to opposite first).
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (x, &a) in signalled.iter().enumerate() {
            for &b in &signalled[x + 1..] {
                let off = PI - wrap_angle(heading(a) - heading(b)).abs();
                if off < OPPOSING {
                    pairs.push((off, a, b));
                }
            }
        }
        pairs.sort_by(|p, q| p.0.total_cmp(&q.0).then((p.1, p.2).cmp(&(q.1, q.2))));
        let mut group_of = vec![usize::MAX; junction.approaches.len()];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (_, a, b) in pairs {
            if group_of[a] == usize::MAX && group_of[b] == usize::MAX {
                group_of[a] = groups.len();
                group_of[b] = groups.len();
                groups.push(vec![a, b]);
            }
        }
        for &a in &signalled {
            if group_of[a] == usize::MAX {
                group_of[a] = groups.len();
                groups.push(vec![a]);
            }
        }
        // Groups of the major roads first (widest approach, then order).
        let width = |g: &Vec<usize>| g.iter().map(|&a| junction.approaches[a].lanes.len()).max().unwrap_or(0);
        groups.sort_by(|g, h| width(h).cmp(&width(g)).then(g.cmp(h)));

        let conflicts =
            |a: u32, b: u32| connectors[a as usize].conflicts.iter().any(|c| c.other == b && c.kind.exclusive());
        let order = |t: Turn| match t {
            Turn::Straight | Turn::Right => 0,
            Turn::Left => 1,
            Turn::UTurn => 2,
        };
        let mut phases: Vec<Phase> = Vec::new();
        for g in &groups {
            let mut left: Vec<u32> = junction
                .connectors
                .iter()
                .copied()
                .filter(|&c| {
                    let from = connectors[c as usize].from;
                    g.iter().any(|&a| junction.approaches[a].lanes.contains(&from))
                })
                .collect();
            left.sort_by_key(|&c| (order(connectors[c as usize].turn), c));
            let mut group_phases: Vec<Phase> = Vec::new();
            while !left.is_empty() {
                let mut phase: Vec<u32> = Vec::new();
                let mut rest = Vec::new();
                for c in left {
                    if phase.iter().all(|&p| !conflicts(c, p)) {
                        phase.push(c);
                    } else {
                        rest.push(c);
                    }
                }
                phase.sort_unstable();
                let protected = !group_phases.is_empty();
                group_phases.push(Phase { connectors: phase, green: 0.0, protected });
                left = rest;
            }
            // Protected phases lead their main phase.
            if !group_phases.is_empty() {
                group_phases.rotate_left(1);
            }
            phases.extend(group_phases);
        }
        if phases.is_empty() {
            continue;
        }
        let n = phases.len();
        let n_main = phases.iter().filter(|p| !p.protected).count().max(1);
        let spare = target_cycle(n)
            - n as f64 * (AMBER + ALL_RED)
            - phases.iter().filter(|p| p.protected).count() as f64 * PROTECTED_GREEN;
        let main = (spare / n_main as f64).max(MIN_GREEN);
        for p in &mut phases {
            p.green = if p.protected { PROTECTED_GREEN } else { main };
        }
        let id = out.len() as u32;
        for (k, p) in phases.iter().enumerate() {
            for &c in &p.connectors {
                of[c as usize] = Some((id, k as u8));
            }
        }
        out.push(Controller { junction: j as u32, phases, amber: AMBER, all_red: ALL_RED });
    }
    (out, of)
}
