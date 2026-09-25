//! Per-agent event bits, accumulated over the physics ticks of a policy step.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{BitOr, BitOrAssign};

/// Set of events; see the associated constants.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Events(pub u32);

impl Events {
    pub const NONE: Self = Self(0);
    /// Hit terrain with the airframe or rotors, or with the gear faster than the crash speed.
    pub const CRASH_TERRAIN: Self = Self(1 << 0);
    /// The same with a solid obstacle (trunk, rock, wall).
    pub const CRASH_OBSTACLE: Self = Self(1 << 1);
    /// Touched another agent.
    pub const CRASH_AGENT: Self = Self(1 << 2);
    /// A collider touched a water surface.
    pub const WATER: Self = Self(1 << 3);
    /// Left the map (or flew above the configured height limits).
    pub const OUT_OF_BOUNDS: Self = Self(1 << 4);
    /// The state became non-finite or the dynamics failed.
    pub const NAN: Self = Self(1 << 5);
    /// Inside foliage (tree canopy, bush); not a crash by itself.
    pub const FOLIAGE: Self = Self(1 << 6);
    /// Resting on or pushing against terrain or an obstacle with the landing gear.
    pub const GROUND_CONTACT: Self = Self(1 << 7);
    /// On the gear and (almost) at rest.
    pub const LANDED: Self = Self(1 << 8);
    /// The agent is frozen after a terminal event (reported on every step until the reset).
    pub const DISABLED: Self = Self(1 << 9);
    /// Came within the goal radius of the current goal and moved on to the next one (see
    /// `GoalSpec::radius`).
    pub const GOAL_REACHED: Self = Self(1 << 10);
    /// Reached the last goal (set together with `GOAL_REACHED`).
    pub const FINISHED: Self = Self(1 << 11);
    /// A ground vehicle tilted past the rollover angle (`EventConfig::rollover_deg`).
    pub const ROLLOVER: Self = Self(1 << 12);
    /// A ground vehicle has moved less than `EventConfig::stuck_distance` for
    /// `EventConfig::stuck_time` seconds (set on every step until it moves; not terminal).
    pub const STUCK: Self = Self(1 << 13);
    /// An articulation angle of a ground vehicle with trailers exceeded
    /// `EventConfig::jackknife_deg`.
    pub const JACKKNIFE: Self = Self(1 << 14);

    /// Events after which the vehicle cannot continue.
    pub const TERMINAL: Self = Self(
        Self::CRASH_TERRAIN.0
            | Self::CRASH_OBSTACLE.0
            | Self::CRASH_AGENT.0
            | Self::WATER.0
            | Self::OUT_OF_BOUNDS.0
            | Self::NAN.0
            | Self::ROLLOVER.0
            | Self::JACKKNIFE.0,
    );

    pub const NAMES: [(&'static str, Events); 15] = [
        ("crash_terrain", Self::CRASH_TERRAIN),
        ("crash_obstacle", Self::CRASH_OBSTACLE),
        ("crash_agent", Self::CRASH_AGENT),
        ("water", Self::WATER),
        ("out_of_bounds", Self::OUT_OF_BOUNDS),
        ("nan", Self::NAN),
        ("foliage", Self::FOLIAGE),
        ("ground_contact", Self::GROUND_CONTACT),
        ("landed", Self::LANDED),
        ("disabled", Self::DISABLED),
        ("goal_reached", Self::GOAL_REACHED),
        ("finished", Self::FINISHED),
        ("rollover", Self::ROLLOVER),
        ("stuck", Self::STUCK),
        ("jackknife", Self::JACKKNIFE),
    ];

    #[inline]
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn is_terminal(self) -> bool {
        self.intersects(Self::TERMINAL)
    }

    /// Names of the set bits.
    pub fn names(self) -> impl Iterator<Item = &'static str> {
        Self::NAMES.into_iter().filter(move |(_, e)| self.contains(*e)).map(|(n, _)| n)
    }
}

impl BitOr for Events {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Events {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl fmt::Debug for Events {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Events(")?;
        for (i, n) in self.names().enumerate() {
            if i > 0 {
                f.write_str(" | ")?;
            }
            f.write_str(n)?;
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_are_distinct_and_named() {
        let all = Events::NAMES.iter().fold(0u32, |acc, (_, e)| {
            assert_eq!(e.0.count_ones(), 1);
            assert_eq!(acc & e.0, 0);
            acc | e.0
        });
        assert_eq!(all, (1 << Events::NAMES.len()) - 1);
        let e = Events::WATER | Events::FOLIAGE;
        assert!(e.is_terminal() && !Events::FOLIAGE.is_terminal());
        assert_eq!(format!("{e:?}"), "Events(water | foliage)");
    }
}
