//! Low-poly pedestrian figures with a walk cycle: per outfit, [`FRAMES`] meshes at evenly
//! spaced phases of the cycle, the first one standing.
//!
//! Figures are in the pedestrian's FLU frame (x facing, z up), feet at the origin, and
//! [`HEIGHT`] tall; renderers scale them to each pedestrian's height.

use crate::mesh::{MeshData, cuboid, icosphere, srgb};
use glam::{DQuat, DVec3, Vec3};
use std::f64::consts::TAU;

/// Frames of the walk cycle.
pub const FRAMES: usize = 8;

/// Outfits (colour sets).
pub const OUTFITS: usize = 8;

/// Height of a figure (m).
pub const HEIGHT: f64 = 1.75;

/// Distance walked over one cycle (two steps) at the figure's height (m).
pub const CYCLE_LENGTH: f64 = 1.4;

/// Swing of the legs and of the arms at full stride (rad).
const LEG_SWING: f64 = 0.45;
const ARM_SWING: f64 = 0.35;

/// Height of the hips and of the shoulders (m).
const HIP: f64 = 0.92;
const SHOULDER: f64 = 1.45;

/// Shirt, trousers and skin colours of outfit `k` (sRGB).
fn colors(k: usize) -> [[u8; 3]; 3] {
    const SHIRTS: [[u8; 3]; OUTFITS] = [
        [196, 62, 52],
        [52, 92, 160],
        [232, 226, 214],
        [62, 128, 82],
        [224, 170, 60],
        [64, 64, 70],
        [150, 84, 150],
        [90, 160, 190],
    ];
    const TROUSERS: [[u8; 3]; 4] = [[40, 46, 70], [60, 58, 54], [120, 104, 82], [30, 30, 34]];
    const SKIN: [[u8; 3]; 4] = [[236, 196, 164], [198, 148, 110], [140, 96, 66], [86, 60, 44]];
    [SHIRTS[k % OUTFITS], TROUSERS[(k * 3 + 1) % 4], SKIN[(k * 5 + 2) % 4]]
}

/// A limb `length` long and `half_width` thick hanging from `pivot`, swung forward by `swing`.
fn limb(m: &mut MeshData, pivot: DVec3, length: f64, half_width: f32, swing: f64, color: [f32; 4]) {
    let part = cuboid(Vec3::new(half_width, half_width, 0.5 * length as f32), color);
    // Swung about the lateral axis: forward (+x) for positive `swing`.
    let rot = DQuat::from_rotation_y(-swing);
    m.append_transformed(&part, rot, pivot + rot * DVec3::new(0.0, 0.0, -0.5 * length));
}

/// The figure of outfit `outfit` at `frame` of the walk cycle (0 standing, with the legs
/// together as they pass each other).
pub fn pedestrian(outfit: usize, frame: usize) -> MeshData {
    let [shirt, trousers, skin] = colors(outfit).map(srgb);
    let s = (TAU * (frame % FRAMES) as f64 / FRAMES as f64).sin();
    let leg = HIP;
    // The body bobs: down by what the swung legs lose in height, the feet on the ground.
    let z = -leg * (1.0 - (LEG_SWING * s).cos());
    let mut m = MeshData::new();
    for side in [1.0, -1.0] {
        limb(&mut m, DVec3::new(0.0, 0.1 * side, HIP + z), leg, 0.07, side * LEG_SWING * s, trousers);
        limb(&mut m, DVec3::new(0.0, 0.25 * side, SHOULDER + z), 0.62, 0.05, -side * ARM_SWING * s, shirt);
    }
    let torso = cuboid(Vec3::new(0.12, 0.2, 0.5 * (SHOULDER + 0.05 - HIP) as f32), shirt);
    m.append_transformed(&torso, DQuat::IDENTITY, DVec3::new(0.0, 0.0, 0.5 * (SHOULDER + 0.05 + HIP) + z));
    let head = icosphere(0.11, 1, false, skin);
    m.append_transformed(&head, DQuat::IDENTITY, DVec3::new(0.0, 0.0, HEIGHT - 0.11 + z));
    m
}

/// Frame of the walk cycle at `phase` (cycles walked; any real).
pub fn frame(phase: f64) -> usize {
    ((phase.rem_euclid(1.0) * FRAMES as f64) as usize).min(FRAMES - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn figures_stand_on_their_feet_and_swing_their_legs() {
        for outfit in 0..OUTFITS {
            for k in 0..FRAMES {
                let m = pedestrian(outfit, k);
                let (lo, hi) = m.bounds().unwrap();
                assert!(
                    (-0.05..1e-4).contains(&lo.z) && (f64::from(hi.z) - HEIGHT).abs() < 0.1,
                    "{outfit}/{k}: {lo} {hi}"
                );
                assert!(m.triangle_count() < 200);
            }
        }
        // Standing: no swing; a quarter cycle on: the stride.
        let reach = |k: usize| pedestrian(0, k).bounds().map(|(lo, hi)| hi.x - lo.x).unwrap();
        assert!(reach(0) < 0.3, "{}", reach(0));
        assert!((f64::from(pedestrian(0, 0).bounds().unwrap().1.z) - HEIGHT).abs() < 1e-4);
        assert!(reach(FRAMES / 4) > 0.7, "{}", reach(FRAMES / 4));
        assert_eq!(frame(0.0), 0);
        assert_eq!(frame(1.25), FRAMES / 4);
        assert_eq!(frame(-0.01), FRAMES - 1);
    }
}
