//! Bay goals (`GoalKind::Bay`): a ground vehicle's last unit (the trailer of a rig) is to be
//! reversed into a bay. In a farm yard the vehicle starts facing the yard's road and the bay
//! lies at the far side of the yard, in front of the buildings; on an urban map the bays are
//! the marked ones of the parking lots (the vehicle starts in the aisle, past the bay) and
//! of the streets' parking lanes (it starts in the lane beside, ahead of the bay, for
//! parallel parking). Yard goals (`GoalKind::Yard`): an aircraft starts on the
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
use autonomousim_world::{BayKind, NodeKind, ParkingBay, StaticWorld};
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
    /// Spawn heading: the yard's (the aisle's, the street's) plus a uniform ±`yaw_deg`
    /// (degrees).
    pub yaw_deg: f64,
    /// Which bays of urban maps to use (all by default).
    #[serde(skip_serializing_if = "is_all_kinds")]
    pub kinds: Vec<BayKind>,
}

fn is_all_kinds(k: &[BayKind]) -> bool {
    k == [BayKind::Lot, BayKind::Street]
}

impl Default for BayGoals {
    fn default() -> Self {
        Self { depth: 16.0, offset: 4.0, lateral: 2.0, yaw_deg: 10.0, kinds: vec![BayKind::Lot, BayKind::Street] }
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

/// Where a vehicle can be parked: a farm yard or a marked bay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Slot {
    Yard(Yard),
    /// A bay and its index in the map's `sites().bays`.
    Bay(usize, ParkingBay),
}

/// The slots of `world` for bay goals: its farm yards, then the bays of the kinds in
/// `settings.kinds`.
pub fn slots(world: &StaticWorld, settings: &BayGoals) -> Vec<Slot> {
    let mut out: Vec<Slot> = yards(world).into_iter().map(Slot::Yard).collect();
    let bays = world.sites().bays.iter().enumerate().filter(|(_, b)| settings.kinds.contains(&b.kind));
    out.extend(bays.map(|(k, b)| Slot::Bay(k, *b)));
    out
}

/// Clearance (m) of the tail from the inner end of a marked bay.
const BAY_END: f64 = 0.3;

/// A spawn and bay: the towing unit's position (horizontal) and heading, and the goal; the
/// index of a marked bay in the map's `sites().bays`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaySpawn {
    pub xy: DVec2,
    pub yaw: f64,
    pub goal: Goal,
    pub bay: Option<usize>,
}

/// Sample a bay in one of `slots` (preferring one not in `used`, which it is added to) and a
/// spawn for vehicle `def`; the goal lies `lift` above the surface.
///
/// In a yard the bay lies `depth` behind the centre, ±`offset` across, and the tail starts a
/// distance from `spec.distance` ahead of it (±`lateral`). In a lot the goal has the tail at
/// the bay's inner end, heading out, and the tail starts in the middle of the aisle, that
/// distance past the bay on either side, heading along the aisle. In a parking lane the goal
/// has the tail at the bay's rear end, heading with the traffic, and the tail starts that
/// distance ahead in the lane beside it.
pub fn sample(
    world: &StaticWorld,
    slots: &[Slot],
    spec: &GoalSpec,
    def: &WheeledDef,
    lift: f64,
    used: &mut Vec<usize>,
    rng: &mut SimRng,
) -> Option<BaySpawn> {
    if slots.is_empty() {
        return None;
    }
    let free: Vec<usize> = (0..slots.len()).filter(|k| !used.contains(k)).collect();
    let k = if free.is_empty() {
        rng.below(slots.len() as u64) as usize
    } else {
        free[rng.below(free.len() as u64) as usize]
    };
    used.push(k);
    let b = &spec.bay;
    // The tail with all units in line, in the towing unit's frame.
    let behind = (def.unit_origin(def.num_units() - 1) + def.tail()).truncate();
    let index = match slots[k] {
        Slot::Bay(i, _) => Some(i),
        Slot::Yard(_) => None,
    };
    let (bay, heading, tail, yaw) = match slots[k] {
        Slot::Yard(yard) => {
            let across = rng.range(-b.offset, b.offset);
            let bay = yard.point(DVec2::new(-b.depth, across));
            let d = rng.range(spec.distance[0], spec.distance[1]);
            let tail = yard.point(DVec2::new(-b.depth + d, across + rng.range(-b.lateral, b.lateral)));
            let yaw = yard.heading + rng.range(-b.yaw_deg, b.yaw_deg).to_radians();
            (bay, yard.heading, tail, yaw)
        }
        Slot::Bay(_, bay) => {
            let half = 0.5 * bay.size.x;
            let d = rng.range(spec.distance[0], spec.distance[1]);
            let jitter = rng.range(-b.yaw_deg, b.yaw_deg).to_radians();
            match bay.kind {
                BayKind::Lot => {
                    // Nose in: the aisle lies beyond the bay's outer end (x < 0). The vehicle
                    // (its length and 2 m more) must fit in the aisle clear of solids: failing
                    // that on the drawn side, the other, then at half and a quarter of the
                    // distance, and last right beside the bay.
                    let goal = bay.point(DVec2::new(half - BAY_END, 0.0));
                    let side = if rng.chance(0.5) { 1.0 } else { -1.0 };
                    let length = behind.length() + 2.0;
                    // Clear from the tail at `along` to the front, toward `dir` (±1).
                    let fits = |along: f64, dir: f64| {
                        let n = length.ceil() as usize;
                        (0..=n).all(|i| {
                            let q = bay.point(DVec2::new(-half - 3.0, along + dir * length * i as f64 / n as f64));
                            let z = world.terrain().height(q.x, q.y) + 1.0;
                            world.obstacle_clearance(q.extend(z), 1.5) >= 1.5
                        })
                    };
                    let mut tries =
                        [d, 0.5 * d, 0.25 * d].into_iter().flat_map(|x| [(side * x, side), (-side * x, -side)]);
                    let (along, s) = tries.find(|&(x, s)| fits(x, s)).unwrap_or((0.0, side));
                    let aisle = bay.point(DVec2::new(-half - 3.0, along));
                    (goal, bay.yaw + std::f64::consts::PI, aisle, bay.yaw + s * std::f64::consts::FRAC_PI_2 + jitter)
                }
                BayKind::Street => {
                    // The lane beside lies to the left of the bay.
                    let goal = bay.point(DVec2::new(-half + BAY_END, 0.0));
                    let lane = bay.point(DVec2::new(-half + BAY_END + d, 0.5 * bay.size.y + 1.75));
                    (goal, bay.yaw, lane, bay.yaw + jitter)
                }
            }
        }
    };
    let xy = tail - DVec2::from_angle(yaw).rotate(behind);
    let z = world.surface_height(bay.x, bay.y) + lift;
    Some(BaySpawn { xy, yaw, goal: Goal { position: bay.extend(z), yaw: heading }, bay: index })
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
