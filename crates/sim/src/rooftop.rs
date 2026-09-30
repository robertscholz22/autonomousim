//! Rooftop goals (`GoalKind::Rooftop`): an aircraft flies to a landing pad on a flat roof of
//! an urban map. It starts on another pad (a share `roof_start` of the trips) or at street
//! level on a sidewalk, the goal pad lying a distance from the goal spec's `distance` range
//! away (horizontally).

use crate::scenario::GoalSpec;
use autonomousim_core::geometry::{HitMask, StaticGeometry};
use autonomousim_core::rng::SimRng;
use autonomousim_core::terrain::Terrain;
use autonomousim_world::{Area, Pad, StaticWorld};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

/// Settings of `GoalKind::Rooftop`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RooftopGoals {
    /// Share of the trips starting on another pad (the rest start on a sidewalk).
    pub roof_start: f64,
    /// Street starts keep this distance from solid obstacles and foliage up to 10 m above the
    /// sidewalk (m).
    pub clearance: f64,
}

impl Default for RooftopGoals {
    fn default() -> Self {
        Self { roof_start: 0.3, clearance: 2.0 }
    }
}

impl RooftopGoals {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let ok = (0.0..=1.0).contains(&self.roof_start) && self.clearance.is_finite() && self.clearance >= 0.0;
        if ok {
            Ok(())
        } else {
            Err(format!("rooftop settings need roof_start in [0, 1] and clearance ≥ 0: {self:?}"))
        }
    }
}

/// A rooftop trip: the start (on the surface: a pad or the sidewalk) and the goal pad.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RooftopTrip {
    pub from: DVec3,
    pub to: Pad,
}

/// Heights above the sidewalk (m) at which a street start's clearance is measured.
const STREET_COLUMN: [f64; 4] = [0.5, 2.0, 5.0, 10.0];
/// Steepest ground (rise per metre) of a street start: sidewalks on steep streets tilt a
/// multirotor on its gear until its frame touches.
const STREET_SLOPE: f64 = 0.08;
/// Tries for a street start per distance range.
const STREET_TRIES: usize = 400;

/// Whether `world` has rooftop pads to fly between (at least two).
pub fn usable(world: &StaticWorld) -> bool {
    world.sites().pads.len() >= 2
}

/// Sample a trip: the goal pad (preferring one not in `used`, which it is added to), then the
/// start: with chance `roof_start` another pad whose horizontal distance lies in
/// `spec.distance` (the closest to the range when none does), else a sidewalk point in that
/// range, clear of obstacles (the range widened to half its lower and 1.5 its upper bound,
/// then to the whole map, when none is found).
pub fn trip(world: &StaticWorld, spec: &GoalSpec, used: &mut Vec<usize>, rng: &mut SimRng) -> Option<RooftopTrip> {
    let pads = &world.sites().pads;
    if pads.len() < 2 {
        return None;
    }
    let free: Vec<usize> = (0..pads.len()).filter(|k| !used.contains(k)).collect();
    let j = if free.is_empty() {
        rng.below(pads.len() as u64) as usize
    } else {
        free[rng.below(free.len() as u64) as usize]
    };
    used.push(j);
    let to = pads[j];
    let goal = to.centre.truncate();
    let [lo, hi] = spec.distance;
    let settings = &spec.rooftop;
    if rng.uniform() < settings.roof_start {
        let miss = |k: usize| {
            let d = pads[k].centre.truncate().distance(goal);
            (lo - d).max(d - hi).max(0.0)
        };
        let others: Vec<usize> = (0..pads.len()).filter(|&k| k != j).collect();
        let inside: Vec<usize> = others.iter().copied().filter(|&k| miss(k) == 0.0).collect();
        let k = if inside.is_empty() {
            *others.iter().min_by(|a, b| miss(**a).total_cmp(&miss(**b)))?
        } else {
            inside[rng.below(inside.len() as u64) as usize]
        };
        return Some(RooftopTrip { from: pads[k].centre, to });
    }
    let (m0, m1) = world.extent();
    let ranges = [[lo, hi], [0.5 * lo, 1.5 * hi], [0.0, (m1 - m0).length()]];
    for [a, b] in ranges {
        for _ in 0..STREET_TRIES {
            let r = a + (b - a) * rng.uniform();
            let p = goal + r * DVec2::from_angle(std::f64::consts::TAU * rng.uniform());
            if let Some(from) = street_start(world, p, settings.clearance) {
                return Some(RooftopTrip { from, to });
            }
        }
    }
    None
}

/// `p` on the surface if it lies on a level enough sidewalk inside the map, clear of
/// obstacles.
fn street_start(world: &StaticWorld, p: DVec2, clearance: f64) -> Option<DVec3> {
    let (lo, hi) = world.extent();
    let inside = p.x > lo.x + 5.0 && p.y > lo.y + 5.0 && p.x < hi.x - 5.0 && p.y < hi.y - 5.0;
    if !inside || world.roads().area(p) != Area::Sidewalk {
        return None;
    }
    let (ground, normal) = world.grid().height_normal(p.x, p.y);
    if normal.truncate().length() > STREET_SLOPE * normal.z {
        return None;
    }
    let clear = STREET_COLUMN.iter().all(|h| {
        let q = p.extend(ground + h);
        world.obstacles().nearest_distance(q, clearance, HitMask::SOLID | HitMask::FOLIAGE).is_none()
    });
    clear.then(|| p.extend(ground))
}
