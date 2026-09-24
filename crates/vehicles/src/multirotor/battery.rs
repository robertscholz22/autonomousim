//! Battery state of charge and voltage sag.

use super::def::BatteryDef;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatteryState {
    /// State of charge in `[0, 1]`.
    pub soc: f64,
    /// Terminal voltage under the last load (V).
    pub voltage: f64,
    /// Current drawn (A).
    pub current: f64,
}

impl BatteryDef {
    pub fn full_voltage(&self) -> f64 {
        self.cells as f64 * self.cell_full
    }

    pub fn open_circuit_voltage(&self, soc: f64) -> f64 {
        self.cells as f64 * (self.cell_empty + soc.clamp(0.0, 1.0) * (self.cell_full - self.cell_empty))
    }

    /// Unloaded state at the given charge.
    pub fn state(&self, soc: f64) -> BatteryState {
        BatteryState { soc: soc.clamp(0.0, 1.0), voltage: self.open_circuit_voltage(soc), current: 0.0 }
    }

    /// Terminal voltage and current when drawing electrical power `power` (W): the larger root
    /// of `V² − V_oc·V + P·R = 0`. Beyond the maximum transferable power `V_oc²/4R` the pack
    /// browns out at `V_oc/2`.
    pub fn load(&self, soc: f64, power: f64) -> (f64, f64) {
        let voc = self.open_circuit_voltage(soc);
        let r = self.internal_resistance;
        if r <= 0.0 {
            return (voc, power / voc);
        }
        let disc = voc * voc - 4.0 * power * r;
        let v = 0.5 * (voc + disc.max(0.0).sqrt());
        (v, (voc - v) / r)
    }

    /// Advance by `dt` with the given total shaft power (W).
    pub fn step(&self, state: &mut BatteryState, shaft_power: f64, dt: f64) {
        let power = shaft_power.max(0.0) / self.efficiency + self.idle_power;
        let (v, i) = self.load(state.soc, power);
        state.voltage = v;
        state.current = i;
        state.soc = (state.soc - i * dt / (3600.0 * self.capacity)).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack() -> BatteryDef {
        BatteryDef {
            cells: 3,
            capacity: 5.0,
            internal_resistance: 0.05,
            efficiency: 0.8,
            idle_power: 2.0,
            cell_full: 4.2,
            cell_empty: 3.3,
        }
    }

    #[test]
    fn sag_and_drain() {
        let b = pack();
        let mut s = b.state(1.0);
        assert!((s.voltage - 12.6).abs() < 1e-12);
        b.step(&mut s, 158.4, 1.0);
        // P = 158.4/0.8 + 2 = 200 W → V = (12.6 + √(12.6² − 40))/2, I = P/V.
        let v = 0.5 * (12.6 + (12.6f64 * 12.6 - 40.0).sqrt());
        assert!((s.voltage - v).abs() < 1e-12);
        assert!((s.current * s.voltage - 200.0).abs() < 1e-9);
        assert!((1.0 - s.soc - s.current / 18_000.0).abs() < 1e-15);
        // Brown-out caps the voltage at V_oc/2.
        let (v, i) = b.load(0.5, 1e6);
        assert!((v - 0.5 * b.open_circuit_voltage(0.5)).abs() < 1e-12 && i > 0.0);
    }
}
