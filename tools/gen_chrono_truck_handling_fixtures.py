"""Handling reference runs of the Chrono::Vehicle trucks (PyChrono) with the truck tyre on
every wheel, for the M4b step-3 validation in crates/vehicles/tests/truck_handling.rs, on flat
rigid ground with μ equal to the tyre's reference friction (as
tools/gen_chrono_truck_fixtures.py, whose `Rig` builds the vehicles):

    make fixtures-chrono

Writes fixtures/chrono/truck_handling.json with
* `man_constant_steer`: the MAN 10t at a fixed steering input while the speed rises slowly
  from 3 m/s (ISO 4138 constant steering-wheel angle), sampled every 0.25 s;
* `kraz_step_steer`: the Kraz tractor and semitrailer at 60 km/h; the steering steps by a
  fixed input in 0.2 s (ISO 7401 style), sampled every 10 ms;
* `kraz_sine_steer`: the same rig with a single period of sinusoidal steering at 0.4 Hz on top
  of the straight-running input (ISO 14791 open-loop single sine, for rearward amplification),
  sampled every 10 ms.

The straight approach is held by a pure-pursuit driver on the line y = 0 (the rigs drift
slightly otherwise), and the speed by a PI on throttle and brake, active throughout. The
manoeuvres add their steering to the driver's input at the start of the manoeuvre. Both laws
are simple enough to re-implement exactly in the tests.

Every sample holds, per unit, the chassis reference frame's position and yaw, the centre of
mass's velocity in the world, the yaw rate and roll; and the tractor's speed, the steering
input, the mean road-wheel angle of each steered axle relative to the chassis, and the
articulation (trailer yaw − tractor yaw).
"""

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_chrono_truck_fixtures import DT, MU0, Rig, ch, veh  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "chrono"
SPEED_KP, SPEED_KI = 0.5, 0.1
# Pure pursuit on y = 0: lookahead (m) = max(LOOKAHEAD_MIN, LOOKAHEAD_TIME · v).
LOOKAHEAD_MIN, LOOKAHEAD_TIME = 10.0, 1.0
# Steering input per unit bicycle angle (rad) for the drivers (the full-lock lead-axle angle).
STEER_GAIN = {"kraz_rig": 0.421, "man_10t": 0.4677}
# Distance from the front axle to the unsteered axles' mean (the drivers' wheelbase).
WHEELBASE = {"kraz_rig": 4.78, "man_10t": 4.28}


def sig(x: float) -> float:
    return float(f"{x:.6g}")


class Speed:
    def __init__(self):
        self.i = 0.0

    def __call__(self, v_ref: float, v: float, dt: float) -> tuple[float, float]:
        e = v_ref - v
        self.i = min(max(self.i + e * dt, -5.0), 5.0)
        u = SPEED_KP * e + SPEED_KI * self.i
        return min(max(u, 0.0), 1.0), min(max(-u, 0.0), 1.0)


def pursuit(y: float, yaw: float, v: float, wheelbase: float) -> float:
    """Bicycle angle (rad) toward the point of y = 0 one lookahead ahead."""
    ld = max(LOOKAHEAD_MIN, LOOKAHEAD_TIME * v)
    alpha = math.atan2(-y, ld) - yaw
    return math.atan2(2.0 * wheelbase * math.sin(alpha), math.hypot(ld, y))


class Truck:
    def __init__(self, name: str):
        self.name = name
        self.rig = Rig(name, patch=(4000.0, 4000.0), centre=(1500.0, 0.0), powertrain=True)
        self.speed = Speed()
        self.k = 0

    def yaw(self, k: int) -> float:
        fwd = self.rig.frame(k).TransformDirectionLocalToParent(ch.ChVector3d(1, 0, 0))
        return math.atan2(fwd.y, fwd.x)

    def drive(self, v_ref: float, steering=None):
        """One step at speed `v_ref`; `steering` is the input, or None for the straight driver.
        The driver and the speed PI run at 1 kHz (every other step)."""
        rig = self.rig
        if self.k % 2 == 0:
            if steering is None:
                p = rig.frame(0).GetPos()
                delta = pursuit(p.y, self.yaw(0), rig.v.GetSpeed(), WHEELBASE[self.name])
                rig.inputs.m_steering = min(max(delta / STEER_GAIN[self.name], -1.0), 1.0)
            rig.inputs.m_throttle, rig.inputs.m_braking = self.speed(v_ref, rig.v.GetSpeed(), 2 * DT)
        if steering is not None:
            rig.inputs.m_steering = steering
        rig.step()
        self.k += 1

    def state(self) -> dict:
        rig = self.rig
        units = []
        for k, u in enumerate(rig.units):
            f = rig.frame(k)
            body = u.GetChassis().GetBody()
            y = f.GetRot().Rotate(ch.ChVector3d(0, 1, 0))
            units.append({
                "x": sig(f.GetPos().x), "y": sig(f.GetPos().y), "yaw": sig(self.yaw(k)),
                "com_vel": [sig(body.GetPosDt().x), sig(body.GetPosDt().y)],
                "yaw_rate": sig(body.GetAngVelParent().z), "roll": sig(math.asin(y.z)),
            })
        f = rig.frame(0)
        delta = []
        for ax in rig.v.GetAxles():
            s = ax.m_suspension
            angles = []
            for side in (veh.LEFT, veh.RIGHT):
                axis = f.TransformDirectionParentToLocal(s.GetSpindleRot(side).Rotate(ch.ChVector3d(0, 1, 0)))
                angles.append(math.atan2(-axis.x, axis.y))
            delta.append(sig(0.5 * sum(angles)))
        out = {"t": round(rig.t, 6), "speed": sig(rig.v.GetSpeed()), "steering": sig(rig.inputs.m_steering),
               "throttle": sig(rig.inputs.m_throttle), "brake": sig(rig.inputs.m_braking),
               "delta": delta, "units": units}
        if len(units) > 1:
            out["articulation"] = sig(math.remainder(units[1]["yaw"] - units[0]["yaw"], math.tau))
        return out


def man_constant_steer() -> dict:
    steering, v0, v1, ramp = 0.7, 3.0, 9.0, 0.1
    truck = Truck("man_10t")
    out, n = [], round(0.25 / DT)
    while True:
        t = truck.rig.t
        v_ref = min(v0 + ramp * max(t - 5.0, 0.0), v1)
        truck.drive(v_ref, steering * min(t / 1.0, 1.0))
        if truck.k % n == 0:
            out.append(truck.state())
        if v_ref >= v1 and t > 5.0 + (v1 - v0) / ramp + 3.0:
            break
    last = out[-1]
    print(f"MAN constant steer: {len(out)} samples, a_y {last['speed'] * last['units'][0]['yaw_rate']:.2f}, "
          f"delta {last['delta'][:2]}", file=sys.stderr)
    return {"steering": steering, "speed": [v0, v1], "ramp": ramp, "samples": out}


def kraz(profile, seconds: float, label: str) -> dict:
    """Approach at 60 km/h on the straight driver for 40 s, then the manoeuvre: the input at
    its start plus `profile(t)`."""
    v = 60 / 3.6
    truck = Truck("kraz_rig")
    while truck.rig.t < 40.0 - 1e-9:
        truck.drive(v)
    s0 = truck.rig.inputs.m_steering
    t0 = truck.rig.t
    out, n = [truck.state()], round(0.01 / DT)
    while truck.rig.t - t0 < seconds - 1e-9:
        truck.drive(v, s0 + profile(truck.rig.t - t0))
        if truck.k % n == 0:
            out.append(truck.state())
    for s in out:
        s["t"] = round(s["t"] - t0, 6)
    peak = max(abs(s["units"][0]["yaw_rate"]) for s in out)
    art = max(abs(s["articulation"]) for s in out)
    print(f"Kraz {label}: speed {out[0]['speed']:.2f}, input {s0:.4f}, peak yaw rate {peak:.4f}, "
          f"articulation {art:.4f}", file=sys.stderr)
    return {"speed": v, "approach_steering": s0, "samples": out}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    step, ramp = 0.08, 0.2
    amplitude, freq = 0.08, 0.4

    def sine(t):
        return amplitude * math.sin(2 * math.pi * freq * t) if t < 1.0 / freq else 0.0

    data = {"generator": "tools/gen_chrono_truck_handling_fixtures.py", "mu": MU0, "dt": DT,
            "speed_pi": [SPEED_KP, SPEED_KI], "lookahead": [LOOKAHEAD_MIN, LOOKAHEAD_TIME],
            "steer_gain": STEER_GAIN, "wheelbase": WHEELBASE,
            "man_constant_steer": man_constant_steer(),
            "kraz_step_steer": {"step": step, "ramp": ramp,
                                **kraz(lambda t: step * min(t / ramp, 1.0), 8.0, "step steer")},
            "kraz_sine_steer": {"amplitude": amplitude, "frequency": freq, **kraz(sine, 8.0, "sine steer")}}
    path = OUT / "truck_handling.json"
    path.write_text(json.dumps(data, indent=0))
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
