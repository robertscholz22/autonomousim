"""Reference data for the truck presets from Chrono::Vehicle (PyChrono): the Kraz 64431 tractor
alone and with its Krone semitrailer, and the MAN 10t 8x8, all fitted with the 315/80 R22.5
PAC2002 truck tyre (Chrono's CityBus_Pac02Tire, `assets/tires/Truck_Pac02Tire.tir`) so that
our runs compare like for like. Flat rigid terrain with μ equal to the tyre's reference
friction.

    make fixtures-chrono

Writes fixtures/chrono/truck_<name>.json, consumed by crates/vehicles/tests/trucks.rs and
used to build the presets (assets/vehicles/truck_6x4.toml, semitrailer_3axle.toml,
truck_8x8.toml):
* `design`: masses (per unit) and the spindle positions in each unit's frame as initialised;
* `static`: after settling, each unit's reference height and attitude, the spindle positions
  in the unit's frame, the spindle heights and the vertical load per wheel (both tyres of a
  dual wheel together), and the spring lengths;
* `lock`: the road wheels' steering angles (rad, left and right per steered axle) at full
  steering input, standing.
"""

import json
import math
import pathlib
import sys
import tempfile

import pychrono as ch
import pychrono.vehicle as veh

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "chrono"
MU0 = 0.8
DT = 5e-4
TIRE = "citybus/tire/CityBus_Pac02Tire.json"


def vec(v) -> list[float]:
    return [v.x, v.y, v.z]


class Rig:
    """A Chrono truck (with its trailer for `kraz_rig`) on flat ground, fitted with the truck
    tyre; `powertrain` adds the vehicle's engine and gearbox, `patch` sizes the ground around
    `centre`."""

    def __init__(self, name: str, z0: float = 0.6, patch=(200.0, 40.0), centre=(0.0, 0.0),
                 powertrain: bool = False):
        self.sys = ch.ChSystemNSC()
        self.sys.SetGravitationalAcceleration(ch.ChVector3d(0, 0, -9.81))
        self.sys.SetCollisionSystemType(ch.ChCollisionSystem.Type_BULLET)
        self.terrain = veh.RigidTerrain(self.sys)
        mat = ch.ChContactMaterialNSC()
        mat.SetFriction(MU0)
        self.terrain.AddPatch(mat, ch.ChCoordsysd(ch.ChVector3d(*centre, 0), ch.QUNIT), *patch)
        self.terrain.Initialize()
        pos = ch.ChCoordsysd(ch.ChVector3d(0, 0, z0), ch.QUNIT)
        self.units = []  # (vehicle or trailer, chassis body)
        if name.startswith("kraz"):
            v = veh.Kraz_tractor(self.sys, False)
            v.Initialize(pos, 0.0)
            self.units.append(v)
            if name == "kraz_rig":
                tr = veh.Kraz_trailer(self.sys)
                tr.Initialize(v.GetChassis())
                self.units.append(tr)
        else:
            v = veh.MAN_10t_Vehicle(self.sys, False, veh.BrakeType_SIMPLE, veh.CollisionType_NONE, True)
            v.Initialize(pos, 0.0)
            self.units.append(v)
        self.v = v
        if powertrain:
            v.InitializePowertrain(veh.ChPowertrainAssembly(*powertrain_of(name)))
        # Tyres per wheel (a dual wheel has two).
        self.tires = []
        for u in self.units:
            per_unit = []
            for ax in u.GetAxles():
                wheels = list(ax.GetWheels())
                tyres = []
                for w in wheels:
                    t = veh.Pac02Tire(veh.GetVehicleDataFile(TIRE))
                    u.InitializeTire(t, w, ch.VisualizationType_NONE)
                    tyres.append(t)
                # Chrono orders a dual axle's wheels inner L, inner R, outer L, outer R.
                if len(tyres) == 4:
                    per_unit += [[tyres[0], tyres[2]], [tyres[1], tyres[3]]]
                else:
                    per_unit += [[tyres[0]], [tyres[1]]]
            self.tires.append(per_unit)
        self.inputs = veh.DriverInputs()
        self.t = 0.0

    def step(self):
        for u in self.units:
            u.Synchronize(self.t, self.inputs, self.terrain)
        self.terrain.Synchronize(self.t)
        for u in self.units:
            u.Advance(DT)
        self.terrain.Advance(DT)
        self.sys.DoStepDynamics(DT)
        self.t += DT

    def frame(self, k: int):
        return self.units[k].GetChassis().GetBody().GetFrameRefToAbs()

    def spindles(self, k: int) -> list:
        u = self.units[k]
        out = []
        for ax in u.GetAxles():
            s = ax.m_suspension
            out += [s.GetSpindlePos(veh.LEFT), s.GetSpindlePos(veh.RIGHT)]
        return out

    def unit_state(self, k: int) -> dict:
        f = self.frame(k)
        rot = f.GetRot()
        # Pitch and roll of R_y(pitch)·R_x(roll) (no yaw at rest).
        x = rot.Rotate(ch.ChVector3d(1, 0, 0))
        y = rot.Rotate(ch.ChVector3d(0, 1, 0))
        return {
            "ref_z": f.GetPos().z,
            "ref_x": f.GetPos().x,
            "pitch": math.asin(-x.z),
            "roll": math.asin(y.z / math.cos(math.asin(-x.z))),
            "spindles": [vec(f.TransformPointParentToLocal(p)) for p in self.spindles(k)],
            "spindle_heights": [p.z for p in self.spindles(k)],
            "loads": [sum(t.ReportTireForce(self.terrain).force.z for t in pair) for pair in self.tires[k]],
            "spring_lengths": self.spring_lengths(k),
        }

    def spring_lengths(self, k: int) -> list[float]:
        out = []
        for ax in self.units[k].GetAxles():
            s = ax.m_suspension
            for cast in ("ChToeBarLeafspringAxle", "ChLeafspringAxle", "ChSolidBellcrankThreeLinkAxle",
                         "ChSolidThreeLinkAxle"):
                c = getattr(veh, "CastTo" + cast)(s)
                if c is not None:
                    out += [c.GetSpringLength(side) for side in (veh.LEFT, veh.RIGHT)]
                    break
            else:
                out += [float("nan")] * 2
        return out

    def mass(self, k: int) -> float:
        u = self.units[k]
        if u is self.v:
            return u.GetMass()
        # Trailer: chassis, suspensions and wheels.
        m = u.GetChassis().GetBody().GetMass()
        for ax in u.GetAxles():
            m += ax.m_suspension.GetMass()
            for w in ax.GetWheels():
                m += w.GetMass() + w.GetTire().GetMass()
        return m


def powertrain_of(name: str):
    """Engine and gearbox: the MAN 7t maps Chrono fits to the MAN 10t, and the Kraz tractor's
    (C++ only in Chrono, so rebuilt here as JSON from Kraz_tractor_EngineSimpleMap.cpp and
    Kraz_tractor_AutomaticTransmissionSimpleMap.cpp)."""
    if name.startswith("man"):
        return veh.MAN_7t_EngineSimpleMap("Engine"), veh.MAN_7t_AutomaticTransmissionSimpleMap("Transmission")
    rpm = 30.0 / math.pi
    tune = 1.587
    engine = {
        "Name": "Kraz tractor engine", "Type": "Engine", "Template": "EngineSimpleMap",
        "Maximal Engine Speed RPM": 2700.0,
        "Map Full Throttle": [[-10.472 * rpm, 406.7 * tune]] + [[r, t * tune] for r, t in [
            (500, 400), (1000, 500), (1200, 572), (1400, 664), (1600, 713), (1800, 733), (2000, 725),
            (2100, 717), (2200, 707), (2300, 682), (2400, -800.0), (2500, -271.2)]],
        "Map Zero Throttle": [[w * rpm, t] for w, t in [
            (-10.472, 0.0), (83.776, -20.0), (104.720, -20.0), (125.664, -30.0), (146.608, -30.0),
            (167.552, -30.0), (188.496, -40.0), (209.440, -50.0), (230.383, -70.0), (251.327, -100.0),
            (282.743, -800.0)]],
    }
    transmission = {
        "Name": "Kraz tractor transmission", "Type": "Transmission", "Template": "AutomaticTransmissionSimpleMap",
        "Gear Box": {
            "Reverse Gear Ratio": -0.162337662,
            "Forward Gear Ratios": [0.162337662, 0.220750552, 0.283286119, 0.414937759, 0.571428571, 0.78125, 1.0],
            "Shift Points Map RPM": [[1000, 2226], [1000, 2226], [1000, 2225], [1000, 2210], [1000, 2226],
                                     [1000, 2225], [1000, 2700]],
        },
    }
    out = []
    for kind, data, read in (("engine", engine, veh.ReadEngineJSON), ("transmission", transmission,
                                                                      veh.ReadTransmissionJSON)):
        path = pathlib.Path(tempfile.gettempdir()) / f"kraz_tractor_{kind}.json"
        path.write_text(json.dumps(data))
        out.append(read(str(path)))
    return out


def generate(name: str) -> dict:
    rig = Rig(name)
    design = {
        "mass": [rig.mass(k) for k in range(len(rig.units))],
        "spindles": [[vec(rig.frame(k).TransformPointParentToLocal(p)) for p in rig.spindles(k)]
                     for k in range(len(rig.units))],
        "tyre_radius": rig.tires[0][0][0].GetRadius(),
    }
    rig.inputs.m_braking = 1.0
    for _ in range(round(5.0 / DT)):
        rig.step()
    static = [rig.unit_state(k) for k in range(len(rig.units))]
    total = sum(sum(s["loads"]) for s in static) / 9.81
    print(f"{name}: mass {design['mass']} kg (loads {total:.1f} kg), ride {[round(s['ref_z'], 4) for s in static]}, "
          f"loads {[[round(l) for l in s['loads']] for s in static]}", file=sys.stderr)
    # Full steering, standing (brakes released so the wheels can scrub round).
    lock = []
    if len(rig.units) == 1 or name == "kraz_rig":
        rig.inputs.m_braking = 0.0
        for i in range(round(3.0 / DT)):
            rig.inputs.m_steering = min(1.0, i * DT)
            rig.step()
        f = rig.frame(0)
        for i, ax in enumerate(rig.v.GetAxles()):
            s = ax.m_suspension
            angles = []
            for side in (veh.LEFT, veh.RIGHT):
                axis = f.TransformDirectionParentToLocal(s.GetSpindleRot(side).Rotate(ch.ChVector3d(0, 1, 0)))
                angles.append(math.atan2(-axis.x, axis.y))
            lock.append(angles)
        print(f"{name}: lock {[[round(math.degrees(a), 2) for a in l] for l in lock]} deg", file=sys.stderr)
    return {"generator": "tools/gen_chrono_truck_fixtures.py", "vehicle": name, "mu": MU0, "dt": DT,
            "design": design, "static": static, "lock": lock}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or ["kraz_tractor", "kraz_rig", "man_10t"]
    for name in names:
        data = generate(name)
        (OUT / f"truck_{name}.json").write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
