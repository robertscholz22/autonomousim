//! Bay goals (`GoalKind::Bay`): a ground vehicle starts in a farm yard facing the yard's road,
//! and its last unit (the trailer of a rig) is to be reversed into a bay at the far side of the
//! yard, in front of the buildings. Yard goals (`GoalKind::Yard`): an aircraft starts on the
//! pad of one farm yard and flies to the pad of another.
//!
//! Farm yards are the `Yard` nodes of the road network: rectangles centred on the node with
//! their long side along the road leaving it. The yard's frame has x along that road (toward
//! the exit) and y to its left.

use crate::scenario::{Goal, GoalSpec};
use autonomousim_core::geometry::{HitMask, StaticGeometry};
use autonomousim_core::rng::SimRng;
use autonomousim_core::terrain::Terrain;
use autonomousim_vehicles::ground::WheeledDef;
use autonomousim_world::{NodeKind, StaticWorld};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

/// Settings of `GoalKind::Bay`. The goal is the pose of the last unit's tail
/// ([`WheeledDef::tail`]) at the bay, heading toward the exit; the spawn puts the tail a
/// distance from the goal spec's `distance` range ahead of the bay along the yard, with the
/// vehicle in line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BayGoals {
    /// Distance of the bay behind the yard's centre, away from its road (m).
    pub depth: f64,
    /// Lateral position of the bay in the yard: uniform in ±`offset` (m).
    pub offset: f64,
    /// Lateral position of the tail at the spawn relative to the bay: uniform in ±`lateral` (m).
    pub lateral: f64,
    /// Spawn heading: the yard's plus a uniform ±`yaw_deg` (degrees).
    pub yaw_deg: f64,
}

impl Default for BayGoals {
    fn default() -> Self {
        Self { depth: 16.0, offset: 4.0, lateral: 2.0, yaw_deg: 10.0 }
    }
}

impl BayGoals {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let ok = [self.depth, self.offset, self.lateral, self.yaw_deg].iter().all(|x| x.is_finite() && *x >= 0.0);
        if ok { Ok(()) } else { Err(format!("bay settings must be finite and ≥ 0: {self:?}")) }
    }
}

/// A farm yard: its centre (on the surface) and heading (along its road, toward the exit).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yard {
    pub centre: DVec3,
    pub heading: f64,
}

impl Yard {
    /// The point at `local` (x along the road, y to the left) in the yard's frame.
    pub fn point(&self, local: DVec2) -> DVec2 {
        self.centre.truncate() + DVec2::from_angle(self.heading).rotate(local)
    }
}

/// The farm yards of `world`'s road network, in node order.
pub fn yards(world: &StaticWorld) -> Vec<Yard> {
    let net = world.roads();
    let mut out = Vec::new();
    for (k, node) in net.nodes().iter().enumerate() {
        if node.kind != NodeKind::Yard {
            continue;
        }
        // The road leaving the yard (which starts there): the direction to its eighth point,
        // as the generator lays the yard out.
        let Some(road) = net.roads().iter().find(|r| r.start as usize == k) else {
            continue;
        };
        let points = road.line.points();
        let d = points[points.len().min(8) - 1].truncate() - points[0].truncate();
        out.push(Yard { centre: node.position, heading: d.y.atan2(d.x) });
    }
    out
}

/// A spawn and bay: the towing unit's position (horizontal) and heading, and the goal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaySpawn {
    pub xy: DVec2,
    pub yaw: f64,
    pub goal: Goal,
}

/// Sample a bay in one of `yards` (preferring one not in `used`, which it is added to) and a
/// spawn ahead of it for vehicle `def`; the goal lies `lift` above the surface.
pub fn sample(
    world: &StaticWorld,
    yards: &[Yard],
    spec: &GoalSpec,
    def: &WheeledDef,
    lift: f64,
    used: &mut Vec<usize>,
    rng: &mut SimRng,
) -> Option<BaySpawn> {
    if yards.is_empty() {
        return None;
    }
    let free: Vec<usize> = (0..yards.len()).filter(|k| !used.contains(k)).collect();
    let k = if free.is_empty() {
        rng.below(yards.len() as u64) as usize
    } else {
        free[rng.below(free.len() as u64) as usize]
    };
    used.push(k);
    let yard = yards[k];
    let b = &spec.bay;
    let across = rng.range(-b.offset, b.offset);
    let bay = yard.point(DVec2::new(-b.depth, across));
    let d = rng.range(spec.distance[0], spec.distance[1]);
    let tail = yard.point(DVec2::new(-b.depth + d, across + rng.range(-b.lateral, b.lateral)));
    let yaw = yard.heading + rng.range(-b.yaw_deg, b.yaw_deg).to_radians();
    // The tail with all units in line, in the towing unit's frame.
    let behind = (def.unit_origin(def.num_units() - 1) + def.tail()).truncate();
    let xy = tail - DVec2::from_angle(yaw).rotate(behind);
    let z = world.surface_height(bay.x, bay.y) + lift;
    Some(BaySpawn { xy, yaw, goal: Goal { position: bay.extend(z), yaw: yard.heading } })
}

/// A yard-to-yard trip: the spawn pad and the goal pad (horizontal positions).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YardTrip {
    pub from: DVec2,
    pub to: DVec2,
}

/// Heights above the ground (m) at which a pad's clearance is measured.
const PAD_COLUMN: [f64; 5] = [1.0, 4.0, 10.0, 20.0, 40.0];

/// The pad of a yard: the most open point of a 2 m grid over the yard (±16 m along it,
/// ±12 m across), by the least distance to solid obstacles and foliage over [`PAD_COLUMN`]
/// (up to 15 m),
/// less 0.1 per metre from the centre, so that open yards land near the middle.
pub fn pad(world: &StaticWorld, yard: &Yard) -> DVec2 {
    let mut best = (f64::NEG_INFINITY, yard.centre.truncate());
    for i in -8..=8 {
        for j in -6..=6 {
            let local = DVec2::new(2.0 * i as f64, 2.0 * j as f64);
            let q = yard.point(local);
            let ground = world.terrain().height(q.x, q.y);
            let open = PAD_COLUMN
                .iter()
                .map(|h| {
                    let p = q.extend(ground + h);
                    world.obstacles().nearest_distance(p, 15.0, HitMask::SOLID | HitMask::FOLIAGE).unwrap_or(15.0)
                })
                .fold(f64::INFINITY, f64::min);
            let score = open - 0.1 * local.length();
            if score > best.0 {
                best = (score, q);
            }
        }
    }
    best.1
}

/// Sample a trip between two of `yards` (at least two): a start (preferring one not in
/// `used`, which it is added to) and a destination whose horizontal distance from it lies in
/// `spec.distance`, or, when no yard does, the one closest to that range.
pub fn trip(
    world: &StaticWorld,
    yards: &[Yard],
    spec: &GoalSpec,
    used: &mut Vec<usize>,
    rng: &mut SimRng,
) -> Option<YardTrip> {
    if yards.len() < 2 {
        return None;
    }
    let free: Vec<usize> = (0..yards.len()).filter(|k| !used.contains(k)).collect();
    let k = if free.is_empty() {
        rng.below(yards.len() as u64) as usize
    } else {
        free[rng.below(free.len() as u64) as usize]
    };
    used.push(k);
    let from = pad(world, &yards[k]);
    let [lo, hi] = spec.distance;
    // Distance outside the range (0 inside) of every other yard.
    let miss = |j: usize| {
        let d = (yards[j].centre.truncate() - from).length();
        (lo - d).max(d - hi).max(0.0)
    };
    let others: Vec<usize> = (0..yards.len()).filter(|&j| j != k).collect();
    let inside: Vec<usize> = others.iter().copied().filter(|&j| miss(j) == 0.0).collect();
    let j = if inside.is_empty() {
        *others.iter().min_by(|a, b| miss(**a).total_cmp(&miss(**b)))?
    } else {
        inside[rng.below(inside.len() as u64) as usize]
    };
    Some(YardTrip { from, to: pad(world, &yards[j]) })
}
