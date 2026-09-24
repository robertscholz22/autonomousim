//! Trained policies exported from Python (`examples/export_policy.py`), so that Rust programs
//! (the viewer; later the ROS 2 bridge) can run them without Python.
//!
//! A policy file (JSON) holds the scenario the policy was trained in, the observation
//! normalisation and the actor network, a multilayer perceptron:
//!
//! ```text
//! x₀ = clip((o − mean) / √(var + eps), ±clip)      (as float32, like autonomousim.rl.ObsNormalizer)
//! xᵢ = actᵢ(Wᵢ·xᵢ₋₁ + bᵢ)
//! a  = clip(x_n, ±1)  (PPO: the action mean)   or   tanh(x_n)  (SAC)
//! ```
//!
//! The file also carries a few observations with the actions PyTorch computed for them.
//! [`PolicyFile::policy`] checks the network against them, so a file whose layout this code
//! misreads fails when it is loaded instead of flying badly.

use crate::{Scenario, SimError};
use serde::Deserialize;
use std::path::Path;

/// Value of the `format` field.
pub const FORMAT: &str = "autonomousim-policy";
/// Largest action difference to PyTorch accepted by the load check.
pub const CHECK_TOLERANCE: f32 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Tanh,
    Relu,
    Identity,
}

/// How the last layer's output becomes an action in [−1, 1].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Output {
    Clip,
    Tanh,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    /// `[outputs, inputs]`.
    pub shape: [usize; 2],
    /// Row-major, one row per output.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub activation: Activation,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObsNorm {
    pub mean: Vec<f64>,
    pub var: Vec<f64>,
    pub clip: f64,
    pub eps: f64,
}

/// Observations and the actions PyTorch computed for them.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub obs: Vec<Vec<f32>>,
    pub action: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PolicyFile {
    pub format: String,
    pub version: u32,
    /// Name of the training run.
    pub name: String,
    pub env_id: String,
    /// Training algorithm (`ppo`, `sac`).
    pub algo: String,
    /// The task's scenario, as the training environments were built from it.
    pub scenario: Scenario,
    /// The agent group the policy acts for.
    pub group: String,
    /// Seconds after which the task truncates an episode.
    pub episode_time: f64,
    pub obs_norm: ObsNorm,
    pub layers: Vec<Layer>,
    pub output: Output,
    #[serde(default)]
    pub check: Check,
}

fn invalid(msg: impl Into<String>) -> SimError {
    SimError::Policy(msg.into())
}

impl PolicyFile {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, SimError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self, SimError> {
        let file: Self = serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
        if file.format != FORMAT || file.version != 1 {
            return Err(invalid(format!("unknown format {:?} version {}", file.format, file.version)));
        }
        Ok(file)
    }

    /// The network, checked against the actions PyTorch computed.
    pub fn policy(&self) -> Result<Policy, SimError> {
        let mut policy = Policy::new(self.obs_norm.clone(), self.layers.clone(), self.output)?;
        if self.check.obs.len() != self.check.action.len() {
            return Err(invalid("check: as many observations as actions expected"));
        }
        let mut action = vec![0.0; policy.act_dim()];
        for (k, (obs, expected)) in self.check.obs.iter().zip(&self.check.action).enumerate() {
            if obs.len() != policy.obs_dim() || expected.len() != policy.act_dim() {
                return Err(invalid(format!("check {k}: wrong dimensions")));
            }
            policy.act(obs, &mut action);
            let error = action.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            if error.is_nan() || error > CHECK_TOLERANCE {
                return Err(invalid(format!("check {k}: actions {action:?}, PyTorch computed {expected:?}")));
            }
        }
        Ok(policy)
    }
}

/// A normalised MLP policy. [`Policy::act`] allocates nothing.
#[derive(Clone, Debug)]
pub struct Policy {
    norm: ObsNorm,
    layers: Vec<Layer>,
    output: Output,
    x: Vec<f32>,
    y: Vec<f32>,
}

impl Policy {
    pub fn new(norm: ObsNorm, layers: Vec<Layer>, output: Output) -> Result<Self, SimError> {
        let Some(first) = layers.first() else { return Err(invalid("no layers")) };
        let mut inputs = first.shape[1];
        if norm.mean.len() != inputs || norm.var.len() != inputs {
            return Err(invalid(format!("normalisation for {} values, network takes {inputs}", norm.mean.len())));
        }
        if !(norm.clip > 0.0 && norm.eps >= 0.0 && norm.var.iter().all(|&v| v + norm.eps > 0.0)) {
            return Err(invalid("normalisation: clip must be positive and var + eps too"));
        }
        for (i, l) in layers.iter().enumerate() {
            let [out, inp] = l.shape;
            if inp != inputs || l.weight.len() != out * inp || l.bias.len() != out {
                return Err(invalid(format!("layer {i}: shape {:?} does not fit its inputs or values", l.shape)));
            }
            inputs = out;
        }
        let width = layers.iter().map(|l| l.shape[0].max(l.shape[1])).max().unwrap_or(0);
        Ok(Self { norm, layers, output, x: Vec::with_capacity(width), y: Vec::with_capacity(width) })
    }

    pub fn obs_dim(&self) -> usize {
        self.layers[0].shape[1]
    }

    pub fn act_dim(&self) -> usize {
        self.layers[self.layers.len() - 1].shape[0]
    }

    /// The deterministic action for one raw observation.
    pub fn act(&mut self, obs: &[f32], action: &mut [f32]) {
        assert_eq!(obs.len(), self.obs_dim(), "observation length");
        assert_eq!(action.len(), self.act_dim(), "action length");
        let n = &self.norm;
        self.x.clear();
        self.x.extend(
            obs.iter().zip(n.mean.iter().zip(&n.var)).map(|(&o, (&mean, &var))| {
                ((f64::from(o) - mean) / (var + n.eps).sqrt()).clamp(-n.clip, n.clip) as f32
            }),
        );
        for layer in &self.layers {
            self.y.clear();
            for (row, &b) in layer.weight.chunks_exact(layer.shape[1]).zip(&layer.bias) {
                let z = row.iter().zip(&self.x).map(|(w, x)| w * x).sum::<f32>() + b;
                self.y.push(match layer.activation {
                    Activation::Tanh => z.tanh(),
                    Activation::Relu => z.max(0.0),
                    Activation::Identity => z,
                });
            }
            std::mem::swap(&mut self.x, &mut self.y);
        }
        for (a, &z) in action.iter_mut().zip(&self.x) {
            *a = match self.output {
                Output::Clip => z.clamp(-1.0, 1.0),
                Output::Tanh => z.tanh(),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Two inputs, a tanh layer of two units, a linear output: small enough to work out by hand.
    fn file(output: &str, check_action: f32) -> String {
        json!({
            "format": FORMAT, "version": 1, "name": "test", "env_id": "autonomousim/QuadHover-v0",
            "algo": "ppo", "group": "agent", "episode_time": 10.0,
            "scenario": { "name": "t", "map": { "type": "testworld", "kind": "flat", "size": 50.0 } },
            "obs_norm": { "mean": [1.0, -1.0], "var": [4.0, 0.25], "clip": 5.0, "eps": 0.0 },
            "layers": [
                { "shape": [2, 2], "weight": [1.0, 0.0, 0.0, 2.0], "bias": [0.0, 0.5], "activation": "tanh" },
                { "shape": [1, 2], "weight": [3.0, -1.0], "bias": [0.25], "activation": "identity" },
            ],
            "output": output,
            "check": { "obs": [[3.0, -1.0]], "action": [[check_action]] },
        })
        .to_string()
    }

    #[test]
    fn forward_pass_matches_the_hand_computation() {
        // Normalised (1, 0); hidden (tanh 1, tanh 0.5); output 3·tanh 1 − tanh 0.5 + 0.25.
        let z = 3.0 * 1f32.tanh() - 0.5f32.tanh() + 0.25;
        let f = PolicyFile::from_json(&file("clip", z.clamp(-1.0, 1.0))).unwrap();
        let mut p = f.policy().unwrap();
        assert_eq!((p.obs_dim(), p.act_dim()), (2, 1));
        let mut a = [0.0];
        p.act(&[3.0, -1.0], &mut a);
        assert_eq!(a[0], 1.0, "clipped from {z}");
        // Normalised values are clipped to ±5: (100 − 1)/2 → 5.
        p.act(&[100.0, -1.0], &mut a);
        assert_eq!(a[0], 1.0);

        let f = PolicyFile::from_json(&file("tanh", z.tanh())).unwrap();
        let mut p = f.policy().unwrap();
        p.act(&[3.0, -1.0], &mut a);
        assert!((a[0] - z.tanh()).abs() < 1e-6);
    }

    #[test]
    fn a_failed_check_or_bad_shapes_are_rejected() {
        let f = PolicyFile::from_json(&file("clip", 0.3)).unwrap();
        assert!(f.policy().unwrap_err().to_string().contains("PyTorch computed"));
        let mut f = PolicyFile::from_json(&file("clip", 1.0)).unwrap();
        f.layers[1].shape = [1, 3];
        assert!(f.policy().is_err());
        let bad = file("clip", 1.0).replace(FORMAT, "other");
        assert!(PolicyFile::from_json(&bad).is_err());
    }
}
