"""Generate handling reference runs of the Chrono::Vehicle Sedan (PyChrono), for the M2 step-7
validation suite in crates/vehicles/tests/handling.rs, on flat rigid ground with μ equal to
the tyres' reference friction (as tools/gen_chrono_vehicle_fixtures.py):

    make fixtures-chrono

Writes fixtures/chrono/handling_sedan.json with
* `constant_steer` (ISO 4138, constant steering-wheel angle): a fixed steering input while the
  speed rises slowly from 5 m/s, sampled every 0.25 s;
* `step_steer` (ISO 7401): from 80 km/h on a straight, the throttle is frozen and the
  steering ramps to a fixed input in 0.1 s; two amplitudes (about 1 and 4 m/s² steady
  lateral acceleration), sampled every 10 ms;
* `lane_change` (ISO 3888-1 double lane change) at 80 km/h, steered by the pure-pursuit
  driver below along the course centreline, sampled every 20 ms;
* `brake`: straight braking from 100 km/h at pedal 0.3 to 1, sampled every 50 ms.

Every sample holds the chassis reference frame's position, yaw, velocity in chassis axes,
yaw rate and roll, and the mean road-wheel angle of the front axle relative to the chassis
(the mean cancels the static toe); constant-steer samples add per-wheel tyre loads, lateral
forces, aligning moments and slip angles. The Sedan's steering maps its input nonlinearly to the
road-wheel angle, so the Rust side matches that angle rather than the input.

The speed controller and the driver are simple enough to be re-implemented exactly in the
test, so both vehicles are driven by the same laws.
"""

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_chrono_vehicle_fixtures import DT, MU0, VEHICLES, Run, ch, veh  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "chrono"
FILES = VEHICLES["sedan"]
WHEELBASE = 2.776
# Chrono's Sedan: mean front road-wheel angle per unit steering input at small inputs.
STEER_GAIN = 0.573
# Speed PI (throttle positive, brake negative).
SPEED_KP, SPEED_KI = 0.5, 0.1
# ISO 3888-1 course for a 1.8 m wide car: sections (start, end) along x, lane (low, high) in y.
WIDTH = 1.8
LANE1 = 1.1 * WIDTH + 0.25
LANE3 = 1.2 * WIDTH + 0.25
LANE5 = 1.3 * WIDTH + 0.25
COURSE = [
    (0.0, 15.0, -LANE1 / 2, LANE1 / 2),
    (45.0, 70.0, LANE1 / 2 + 1.0, LANE1 / 2 + 1.0 + LANE3),
    (95.0, 125.0, -LANE1 / 2, -LANE1 / 2 + LANE5),
]
OFFSET = LANE1 / 2 + 1.0 + LANE3 / 2
# Pure pursuit: lookahead distance (m) = max(LOOKAHEAD_MIN, LOOKAHEAD_TIME · v).
LOOKAHEAD_MIN, LOOKAHEAD_TIME = 8.0, 0.6


def path_y(x: float) -> float:
    """Centreline: straight, a cosine blend to the offset lane over section 2, the offset lane,
    and a blend back over section 4 (then straight)."""

    def blend(a, b):
        s = min(max((x - a) / (b - a), 0.0), 1.0)
        return 0.5 - 0.5 * math.cos(math.pi * s)

    return OFFSET * (blend(15.0, 45.0) - blend(70.0, 95.0))


def pursuit(x: float, y: float, yaw: float, v: float) -> float:
    """Bicycle steering angle (rad) toward the centreline point one lookahead ahead in x."""
    ld = max(LOOKAHEAD_MIN, LOOKAHEAD_TIME * v)
    tx = x + ld
    dx, dy = tx - x, path_y(tx) - y
    alpha = math.atan2(dy, dx) - yaw
    return math.atan2(2.0 * WHEELBASE * math.sin(alpha), math.hypot(dx, dy))


def sig(x: float) -> float:
    """Six significant digits."""
    return float(f"{x:.6g}")


class Speed:
    def __init__(self):
        self.i = 0.0

    def __call__(self, v_ref: float, v: float, dt: float) -> tuple[float, float]:
        e = v_ref - v
        self.i = min(max(self.i + e * dt, -5.0), 5.0)
        u = SPEED_KP * e + SPEED_KI * self.i
        return min(max(u, 0.0), 1.0), min(max(-u, 0.0), 1.0)


class Car(Run):
    def __init__(self, speed: float, z0: float):
        super().__init__(FILES, z0, speed, patch=(1000.0, 1000.0), start=-400.0)

    def state(self, wheels: bool = False) -> dict:
        ref = self.v.GetChassis().GetBody().GetFrameRefToAbs()
        body = self.v.GetChassisBody()
        fwd = ref.TransformDirectionLocalToParent(ch.ChVector3d(1, 0, 0))
        vel = ref.TransformDirectionParentToLocal(body.GetPosDt())
        omega = ref.TransformDirectionParentToLocal(body.GetAngVelParent())
        steer = []
        for axle in (0, 1):
            for side in (veh.LEFT, veh.RIGHT):
                axis = ref.TransformDirectionParentToLocal(
                    self.v.GetSpindleRot(axle, side).Rotate(ch.ChVector3d(0, 1, 0)))
                steer.append(math.atan2(-axis.x, axis.y))
        p = ref.GetPos()
        s = {"t": round(self.t, 6), "x": sig(p.x - self.start), "y": sig(p.y), "yaw": sig(math.atan2(fwd.y, fwd.x)),
                "vx": sig(vel.x), "vy": sig(vel.y), "yaw_rate": sig(omega.z), "roll": sig(self.v.GetRoll()),
                "delta": sig(0.5 * (steer[0] + steer[1])), "delta_rear": sig(0.5 * (steer[2] + steer[3])),
                "steering": sig(self.inputs.m_steering), "throttle": sig(self.inputs.m_throttle),
                "brake": sig(self.inputs.m_braking)}
        if wheels:
            # Tyre forces and moments (reported in the world frame, about the wheel centres): the
            # vertical load, the lateral force in the heading frame and the moment about the
            # vertical, and the slip angle; front left, front right, rear left, rear right.
            yaw = s["yaw"]
            reports = [t.ReportTireForce(self.terrain) for t in self.tires]
            s["fz"] = [sig(r.force.z) for r in reports]
            s["fy"] = [sig(-math.sin(yaw) * r.force.x + math.cos(yaw) * r.force.y) for r in reports]
            s["mz"] = [sig(r.moment.z) for r in reports]
            s["alpha"] = [sig(t.GetSlipAngle()) for t in self.tires]
        return s


def constant_steer(z0: float) -> dict:
    steering, v0, v1, ramp = 0.12, 5.0, 16.0, 0.15
    car, speed = Car(v0, z0), Speed()
    out, n = [], round(0.25 / DT)
    k = 0
    while True:
        t = car.t
        car.inputs.m_steering = steering * min(t / 1.0, 1.0)
        v_ref = min(v0 + ramp * max(t - 4.0, 0.0), v1)
        car.inputs.m_throttle, car.inputs.m_braking = speed(v_ref, car.v.GetSpeed(), DT)
        car.step()
        k += 1
        if k % n == 0:
            out.append(car.state(wheels=True))
        if v_ref >= v1 and t > 4.0 + (v1 - v0) / ramp + 3.0:
            break
    print(f"constant steer: {len(out)} samples, final speed {out[-1]['vx']:.2f}, "
          f"a_y {out[-1]['vx'] * out[-1]['yaw_rate']:.2f}", file=sys.stderr)
    return {"steering": steering, "ramp": ramp, "samples": out}


def approach(v: float, z0: float, seconds: float):
    car, speed = Car(v, z0), Speed()
    while car.t < seconds - 1e-9:
        car.inputs.m_throttle, car.inputs.m_braking = speed(v, car.v.GetSpeed(), DT)
        car.step()
    return car, speed


def step_steer(z0: float) -> list[dict]:
    runs = []
    for steering in (0.01, 0.045):
        car, _ = approach(80 / 3.6, z0, 5.0)
        throttle = car.inputs.m_throttle
        car.inputs.m_braking = 0.0
        t0 = car.t
        out, n, k = [car.state()], round(0.01 / DT), 0
        while car.t - t0 < 4.0 - 1e-9:
            car.inputs.m_steering = steering * min((car.t - t0) / 0.1, 1.0)
            car.inputs.m_throttle = throttle
            car.step()
            k += 1
            if k % n == 0:
                out.append(car.state())
        for s in out:
            s["t"] = round(s["t"] - t0, 6)
        print(f"step steer {steering}: yaw rate {out[-1]['yaw_rate']:.4f}, "
              f"peak {max(s['yaw_rate'] for s in out):.4f}", file=sys.stderr)
        runs.append({"steering": steering, "throttle": throttle, "samples": out})
    return runs


def lane_change(z0: float) -> dict:
    v = 80 / 3.6
    car, speed = approach(v, z0, 3.0)
    # Restart the course at the current position.
    x0 = car.state()["x"] + 10.0
    out, n, k = [], round(0.02 / DT), 0
    t0 = car.t
    while True:
        s = car.state()
        x = s["x"] - x0
        if k % 2 == 0:  # the driver runs at 1 kHz
            delta = pursuit(x, s["y"], s["yaw"], s["vx"])
            car.inputs.m_steering = min(max(delta / STEER_GAIN, -1.0), 1.0)
            car.inputs.m_throttle, car.inputs.m_braking = speed(v, car.v.GetSpeed(), 2 * DT)
        car.step()
        k += 1
        if k % n == 0:
            s = car.state()
            s["x"] -= x0
            s["t"] = round(s["t"] - t0, 6)
            out.append(s)
        if x > 140.0:
            break
    print(f"lane change: peak yaw rate {max(abs(s['yaw_rate']) for s in out):.3f}", file=sys.stderr)
    return {"speed": v, "x0": x0, "samples": out}


def brake(z0: float) -> list[dict]:
    """Straight braking from 100 km/h at several pedal levels. Without ABS, the Sedan's rear
    wheels lock at full pedal and it spins; the samples show it (yaw rate, v_y)."""
    runs = []
    for pedal in (0.3, 0.5, 0.7, 1.0):
        car, _ = approach(100 / 3.6, z0, 3.0)
        car.inputs.m_throttle, car.inputs.m_braking = 0.0, pedal
        t0, x0 = car.t, car.state()["x"]
        out, n, k = [], round(0.05 / DT), 0
        while car.t - t0 < 8.0 - 1e-9:
            car.step()
            k += 1
            if k % n == 0:
                s = car.state()
                s["x"] -= x0
                s["t"] = round(s["t"] - t0, 6)
                out.append(s)
        stop = next((s for s in out if s["vx"] < 0.1), out[-1])
        spin = max(abs(s["yaw_rate"]) for s in out if s["t"] <= stop["t"])
        print(f"brake {pedal}: stop after {stop['x']:.2f} m in {stop['t']:.2f} s, peak yaw rate {spin:.3f}",
              file=sys.stderr)
        runs.append({"pedal": pedal, "samples": out})
    return runs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z0 = json.loads((OUT / "vehicle_sedan.json").read_text())["static"]["ref_z"]
    data = {"generator": "tools/gen_chrono_handling_fixtures.py", "vehicle": "sedan", "mu": MU0, "dt": DT,
            "wheelbase": WHEELBASE, "steer_gain": STEER_GAIN, "speed_pi": [SPEED_KP, SPEED_KI],
            "course": {"sections": COURSE, "offset": OFFSET, "lookahead": [LOOKAHEAD_MIN, LOOKAHEAD_TIME]},
            "constant_steer": constant_steer(z0), "step_steer": step_steer(z0),
            "lane_change": lane_change(z0), "brake": brake(z0)}
    path = OUT / "handling_sedan.json"
    path.write_text(json.dumps(data, indent=0))
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
