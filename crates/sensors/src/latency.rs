//! Exact transport delay in physics ticks.

use std::collections::VecDeque;

/// A reading with the tick (and time) at which it was measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stamped<T> {
    /// Physics tick of the measurement.
    pub tick: u64,
    /// Measurement time (s).
    pub time: f64,
    pub value: T,
}

/// Holds readings back for a fixed number of ticks; [`latest`](Self::latest) is the newest
/// reading whose delay has passed.
#[derive(Clone, Debug)]
pub struct DelayLine<T> {
    latency: u64,
    queue: VecDeque<Stamped<T>>,
    current: Option<Stamped<T>>,
}

impl<T> DelayLine<T> {
    pub fn new(latency: u64) -> Self {
        Self { latency, queue: VecDeque::new(), current: None }
    }

    pub fn latency(&self) -> u64 {
        self.latency
    }

    pub fn clear(&mut self) {
        self.queue.clear();
        self.current = None;
    }

    /// Queue a reading measured at `reading.tick`.
    pub fn push(&mut self, reading: Stamped<T>) {
        self.queue.push_back(reading);
    }

    /// Release every reading due by tick `now`; true if a new one became visible.
    pub fn poll(&mut self, now: u64) -> bool {
        let mut changed = false;
        while self.queue.front().is_some_and(|r| r.tick + self.latency <= now) {
            self.current = self.queue.pop_front();
            changed = true;
        }
        changed
    }

    pub fn latest(&self) -> Option<&Stamped<T>> {
        self.current.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn releases_exactly_after_latency() {
        let mut d = DelayLine::new(3);
        let mut seen = Vec::new();
        for tick in 0..12u64 {
            if tick % 2 == 0 {
                d.push(Stamped { tick, time: tick as f64, value: tick });
            }
            if d.poll(tick) {
                seen.push((tick, d.latest().unwrap().value));
            }
        }
        assert_eq!(seen, [(3, 0), (5, 2), (7, 4), (9, 6), (11, 8)]);
        // Zero latency: visible in the same tick.
        let mut d = DelayLine::new(0);
        d.push(Stamped { tick: 5, time: 0.0, value: 1 });
        assert!(d.poll(5) && d.latest().unwrap().value == 1);
    }
}
