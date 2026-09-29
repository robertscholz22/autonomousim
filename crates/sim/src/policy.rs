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
//! Pixel policies (`ppo_pixels.py`) add an image encoder whose features come before the
//! normalised state in the first layer's input:
//!
//! ```text
//! c₀ = image / 255   (u8 [H, W, C] as [C, H, W], float32)
//! cᵢ = actᵢ(conv(cᵢ₋₁; Kᵢ, stride, padding) + bᵢ)
//! f  = act(W_f·flatten(c_n) + b_f)                   (flattened in [C, H, W] order)
//! x₀ = [f, clip((o − mean) / √(var + eps), ±clip)]
//! ```
//!
//! Sums are accumulated in f64, so the result differs from PyTorch's by PyTorch's own
//! rounding only.
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

/// A 2-D convolution over `[C, H, W]` inputs.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conv {
    /// `[outputs, inputs, kernel height, kernel width]`.
    pub shape: [usize; 4],
    pub stride: usize,
    pub padding: usize,
    /// Row-major in `shape` order (PyTorch's layout).
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub activation: Activation,
}

impl Conv {
    /// Output size along an axis of `n` inputs.
    fn out_len(&self, n: usize, k: usize) -> usize {
        (n + 2 * self.padding).saturating_sub(k) / self.stride + 1
    }
}

/// The image encoder of a pixel policy: convolutions, then a dense layer over their flattened
/// output.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Encoder {
    /// `[height, width, channels]` of the `u8` image.
    pub image_shape: [usize; 3],
    pub convs: Vec<Conv>,
    pub fc: Layer,
}

impl Encoder {
    /// Shape `[C, H, W]` after each convolution, after checking the layers fit together.
    fn shapes(&self) -> Result<Vec<[usize; 3]>, SimError> {
        let [h, w, c] = self.image_shape;
        let mut shape = [c, h, w];
        let mut out = Vec::with_capacity(self.convs.len());
        for (i, conv) in self.convs.iter().enumerate() {
            let [o, inp, kh, kw] = conv.shape;
            let ok = inp == shape[0]
                && conv.stride > 0
                && kh > 0
                && kw > 0
                && shape[1] + 2 * conv.padding >= kh
                && shape[2] + 2 * conv.padding >= kw
                && conv.weight.len() == o * inp * kh * kw
                && conv.bias.len() == o;
            if !ok {
                return Err(invalid(format!(
                    "convolution {i}: shape {:?} does not fit its inputs or values",
                    conv.shape
                )));
            }
            shape = [o, conv.out_len(shape[1], kh), conv.out_len(shape[2], kw)];
            out.push(shape);
        }
        let [f, n] = self.fc.shape;
        if n != shape.iter().product::<usize>() || self.fc.weight.len() != f * n || self.fc.bias.len() != f {
            return Err(invalid(format!("encoder layer: shape {:?} does not fit the convolutions", self.fc.shape)));
        }
        Ok(out)
    }
}

/// Observations and the actions PyTorch computed for them.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub obs: Vec<Vec<f32>>,
    pub action: Vec<Vec<f32>>,
    /// Pixel policies: the images with the observations (base64 of the `u8` image).
    #[serde(default)]
    pub image: Vec<String>,
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
    /// Pixel policies: the image encoder.
    #[serde(default)]
    pub encoder: Option<Encoder>,
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
        let mut policy =
            Policy::with_encoder(self.obs_norm.clone(), self.encoder.clone(), self.layers.clone(), self.output)?;
        let images = if policy.image_len() > 0 { self.check.obs.len() } else { 0 };
        if self.check.obs.len() != self.check.action.len() || self.check.image.len() != images {
            return Err(invalid("check: as many observations as actions (and images for pixel policies) expected"));
        }
        let mut action = vec![0.0; policy.act_dim()];
        for (k, (obs, expected)) in self.check.obs.iter().zip(&self.check.action).enumerate() {
            if obs.len() != policy.obs_dim() || expected.len() != policy.act_dim() {
                return Err(invalid(format!("check {k}: wrong dimensions")));
            }
            let image = match self.check.image.get(k) {
                Some(text) => decode_base64(text).ok_or_else(|| invalid(format!("check {k}: bad image data")))?,
                None => Vec::new(),
            };
            if image.len() != policy.image_len() {
                return Err(invalid(format!(
                    "check {k}: image of {} bytes, the encoder takes {}",
                    image.len(),
                    policy.image_len()
                )));
            }
            policy.act_with_image(obs, &image, &mut action);
            let error = action.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            if error.is_nan() || error > CHECK_TOLERANCE {
                return Err(invalid(format!("check {k}: actions {action:?}, PyTorch computed {expected:?}")));
            }
        }
        Ok(policy)
    }
}

/// Standard base64 (with or without padding); `None` if malformed.
fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    };
    let digits = text.trim_end_matches('=').as_bytes();
    if digits.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::with_capacity(digits.len() * 3 / 4);
    for chunk in digits.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= u32::from(value(c)?) << (18 - 6 * i);
        }
        out.extend_from_slice(&n.to_be_bytes()[1..chunk.len()]);
    }
    Some(out)
}

/// A normalised MLP policy, with an image encoder for pixel policies. [`Policy::act`] and
/// [`Policy::act_with_image`] allocate nothing.
#[derive(Clone, Debug)]
pub struct Policy {
    norm: ObsNorm,
    encoder: Option<Encoder>,
    /// Shape `[C, H, W]` after each convolution.
    conv_shapes: Vec<[usize; 3]>,
    layers: Vec<Layer>,
    output: Output,
    x: Vec<f32>,
    y: Vec<f32>,
    /// Encoder activations (`[C, H, W]`).
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Policy {
    pub fn new(norm: ObsNorm, layers: Vec<Layer>, output: Output) -> Result<Self, SimError> {
        Self::with_encoder(norm, None, layers, output)
    }

    pub fn with_encoder(
        norm: ObsNorm,
        encoder: Option<Encoder>,
        layers: Vec<Layer>,
        output: Output,
    ) -> Result<Self, SimError> {
        let Some(first) = layers.first() else { return Err(invalid("no layers")) };
        let conv_shapes = match &encoder {
            Some(e) => e.shapes()?,
            None => Vec::new(),
        };
        let features = encoder.as_ref().map_or(0, |e| e.fc.shape[0]);
        let mut inputs = first.shape[1];
        if norm.mean.len() + features != inputs || norm.var.len() != norm.mean.len() {
            return Err(invalid(format!(
                "normalisation for {} values and {features} image features, network takes {inputs}",
                norm.mean.len()
            )));
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
        let [h, w, c] = encoder.as_ref().map_or([0; 3], |e| e.image_shape);
        let largest = conv_shapes.iter().map(|s| s.iter().product()).fold(h * w * c, usize::max);
        Ok(Self {
            norm,
            encoder,
            conv_shapes,
            layers,
            output,
            x: Vec::with_capacity(width),
            y: Vec::with_capacity(width),
            a: Vec::with_capacity(largest),
            b: Vec::with_capacity(largest),
        })
    }

    /// Length of the state observation (the image comes separately).
    pub fn obs_dim(&self) -> usize {
        self.norm.mean.len()
    }

    /// `[height, width, channels]` of the image a pixel policy takes.
    pub fn image_shape(&self) -> Option<[usize; 3]> {
        self.encoder.as_ref().map(|e| e.image_shape)
    }

    /// Bytes of the image (0 without an encoder).
    pub fn image_len(&self) -> usize {
        self.image_shape().map_or(0, |[h, w, c]| h * w * c)
    }

    pub fn act_dim(&self) -> usize {
        self.layers[self.layers.len() - 1].shape[0]
    }

    /// The deterministic action for one raw observation (policies without an image).
    pub fn act(&mut self, obs: &[f32], action: &mut [f32]) {
        self.act_with_image(obs, &[], action);
    }

    /// The deterministic action for one raw observation and, for pixel policies, its image
    /// (`u8 [H, W, C]`; empty otherwise).
    pub fn act_with_image(&mut self, obs: &[f32], image: &[u8], action: &mut [f32]) {
        assert_eq!(obs.len(), self.obs_dim(), "observation length");
        assert_eq!(image.len(), self.image_len(), "image length");
        assert_eq!(action.len(), self.act_dim(), "action length");
        self.x.clear();
        self.encode(image);
        let n = &self.norm;
        self.x.extend(
            obs.iter().zip(n.mean.iter().zip(&n.var)).map(|(&o, (&mean, &var))| {
                ((f64::from(o) - mean) / (var + n.eps).sqrt()).clamp(-n.clip, n.clip) as f32
            }),
        );
        for layer in &self.layers {
            self.y.clear();
            dense(layer, &self.x, &mut self.y);
            std::mem::swap(&mut self.x, &mut self.y);
        }
        for (a, &z) in action.iter_mut().zip(&self.x) {
            *a = match self.output {
                Output::Clip => z.clamp(-1.0, 1.0),
                Output::Tanh => z.tanh(),
            };
        }
    }

    /// Image features into `x` (nothing without an encoder).
    fn encode(&mut self, image: &[u8]) {
        let Some(e) = &self.encoder else { return };
        let [h, w, c] = e.image_shape;
        // [H, W, C] bytes → [C, H, W] in [0, 1], as float32 like PyTorch's `.float() / 255`.
        self.a.clear();
        self.a.extend((0..c).flat_map(|ch| (0..h * w).map(move |p| f32::from(image[p * c + ch]) / 255.0)));
        let mut shape = [c, h, w];
        for (conv, &out) in e.convs.iter().zip(&self.conv_shapes) {
            conv_forward(conv, &self.a, shape, out, &mut self.b);
            std::mem::swap(&mut self.a, &mut self.b);
            shape = out;
        }
        dense(&e.fc, &self.a, &mut self.x);
    }
}

fn activate(activation: Activation, z: f32) -> f32 {
    match activation {
        Activation::Tanh => z.tanh(),
        Activation::Relu => z.max(0.0),
        Activation::Identity => z,
    }
}

/// `out.extend(act(W·x + b))`, summed in f64.
fn dense(layer: &Layer, x: &[f32], out: &mut Vec<f32>) {
    for (row, &b) in layer.weight.chunks_exact(layer.shape[1]).zip(&layer.bias) {
        let z = row.iter().zip(x).map(|(&w, &x)| f64::from(w) * f64::from(x)).sum::<f64>() + f64::from(b);
        out.push(activate(layer.activation, z as f32));
    }
}

/// Convolution of `x` (`[C, H, W]` = `input`) into `out` (`[O, H', W']` = `output`), summed in
/// f64; zero padding.
fn conv_forward(conv: &Conv, x: &[f32], input: [usize; 3], output: [usize; 3], out: &mut Vec<f32>) {
    let [_, inp, kh, kw] = conv.shape;
    let [_, h, w] = input;
    let [o, oh, ow] = output;
    let (s, p) = (conv.stride as isize, conv.padding as isize);
    out.clear();
    for k in 0..o {
        let kernel = &conv.weight[k * inp * kh * kw..(k + 1) * inp * kh * kw];
        for oy in 0..oh as isize {
            for ox in 0..ow as isize {
                let mut z = f64::from(conv.bias[k]);
                for ci in 0..inp {
                    for ky in 0..kh as isize {
                        let y = oy * s - p + ky;
                        if y < 0 || y >= h as isize {
                            continue;
                        }
                        let row = &x[(ci * h + y as usize) * w..][..w];
                        let wrow = &kernel[(ci * kh + ky as usize) * kw..][..kw];
                        for (kx, &wk) in wrow.iter().enumerate() {
                            let xx = ox * s - p + kx as isize;
                            if xx >= 0 && xx < w as isize {
                                z += f64::from(wk) * f64::from(row[xx as usize]);
                            }
                        }
                    }
                }
                out.push(activate(conv.activation, z as f32));
            }
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

    /// A 2×2 two-channel image, one 3×3 convolution with stride 2 and padding 1 that sums
    /// channel 0, a dense feature layer and a linear output over (feature, state).
    fn pixel_file(check_action: f32) -> String {
        let mut kernel = vec![1.0; 9];
        kernel.extend([0.0; 9]);
        json!({
            "format": FORMAT, "version": 1, "name": "test", "env_id": "autonomousim/QuadHoverPad-v0",
            "algo": "ppo_pixels", "group": "agent", "episode_time": 10.0,
            "scenario": { "name": "t", "map": { "type": "testworld", "kind": "flat", "size": 50.0 } },
            "obs_norm": { "mean": [0.0], "var": [1.0], "clip": 5.0, "eps": 0.0 },
            "encoder": {
                "image_shape": [2, 2, 2],
                "convs": [{ "shape": [1, 2, 3, 3], "stride": 2, "padding": 1, "weight": kernel, "bias": [0.5], "activation": "relu" }],
                "fc": { "shape": [1, 1], "weight": [2.0], "bias": [0.0], "activation": "relu" },
            },
            "layers": [{ "shape": [1, 2], "weight": [0.1, 1.0], "bias": [0.0], "activation": "identity" }],
            "output": "clip",
            // Channel 0 is 255 at the top left only, channel 1 everywhere: [H, W, C] bytes.
            "check": { "obs": [[0.25]], "action": [[check_action]], "image": ["//8A/wD/AP8="] },
        })
        .to_string()
    }

    #[test]
    fn pixel_policies_encode_the_image() {
        // Convolution 1 + 0.5 (channel 0 only, one bright pixel), feature 2·1.5, output
        // 0.1·3 + 0.25.
        let f = PolicyFile::from_json(&pixel_file(0.55)).unwrap();
        let mut p = f.policy().unwrap();
        assert_eq!((p.obs_dim(), p.act_dim(), p.image_shape(), p.image_len()), (1, 1, Some([2, 2, 2]), 8));
        let mut a = [0.0];
        // Channel 1 only: the convolution is its bias, the feature 1.
        p.act_with_image(&[0.0], &[0, 255, 0, 255, 0, 255, 0, 255], &mut a);
        assert!((a[0] - 0.1).abs() < 1e-7, "{}", a[0]);
        // A channel-first reading of the check image gives another action and fails the check.
        assert!(PolicyFile::from_json(&pixel_file(0.75)).unwrap().policy().is_err());
        let mut f = PolicyFile::from_json(&pixel_file(0.55)).unwrap();
        f.check.image[0] = "AAAA".into();
        assert!(f.policy().unwrap_err().to_string().contains("image of 3 bytes"));
        f.encoder.as_mut().unwrap().fc.shape = [1, 2];
        assert!(f.policy().is_err());
        assert_eq!(decode_base64("AP//AA==").unwrap(), [0, 255, 255, 0]);
        assert!(decode_base64("A").is_none() && decode_base64("A*==").is_none());
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
