//! Integer simulation time.
//!
//! Time is an integer tick count; `t = tick · dt`. There is never a floating-point time
//! accumulator, so rates that divide the physics rate stay exactly aligned forever.

use serde::{Deserialize, Serialize};

/// Physics clock: fixed step and current tick.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clock {
    /// Physics rate in Hz (integer so that all dividers are exact).
    pub physics_hz: u32,
    pub tick: u64,
}

/// Error for a rate that does not evenly divide the physics rate.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("rate {rate_hz} Hz for `{what}` must evenly divide the physics rate {physics_hz} Hz")]
pub struct RateError {
    pub what: String,
    pub rate_hz: u32,
    pub physics_hz: u32,
}

impl Clock {
    pub fn new(physics_hz: u32) -> Self {
        assert!(physics_hz > 0, "physics rate must be positive");
        Self { physics_hz, tick: 0 }
    }

    #[inline]
    pub fn dt(&self) -> f64 {
        1.0 / self.physics_hz as f64
    }

    #[inline]
    pub fn time(&self) -> f64 {
        self.tick as f64 / self.physics_hz as f64
    }

    #[inline]
    pub fn advance(&mut self) {
        self.tick += 1;
    }

    /// Divider for a sub-rate (e.g. 50 Hz at 500 Hz physics → 10). Errors if not exact.
    pub fn divider(&self, what: &str, rate_hz: u32) -> Result<u32, RateError> {
        if rate_hz == 0 || rate_hz > self.physics_hz || !self.physics_hz.is_multiple_of(rate_hz) {
            return Err(RateError { what: what.to_string(), rate_hz, physics_hz: self.physics_hz });
        }
        Ok(self.physics_hz / rate_hz)
    }

    /// True on ticks where a component with the given divider should run.
    #[inline]
    pub fn is_due(&self, divider: u32) -> bool {
        self.tick.is_multiple_of(divider as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dividers() {
        let c = Clock::new(500);
        assert_eq!(c.divider("policy", 50).unwrap(), 10);
        assert_eq!(c.divider("imu", 500).unwrap(), 1);
        assert!(c.divider("bad", 30).is_err());
        assert!(c.divider("zero", 0).is_err());
        assert!(c.divider("too fast", 1000).is_err());
    }

    #[test]
    fn time_is_exact() {
        let mut c = Clock::new(500);
        for _ in 0..1_000_000 {
            c.advance();
        }
        assert_eq!(c.time(), 2000.0);
    }
}
