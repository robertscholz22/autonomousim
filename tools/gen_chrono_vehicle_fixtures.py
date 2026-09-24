"""Generate full-vehicle reference runs with Chrono::Vehicle (PyChrono): the Sedan and the
4WD HMMWV with their SimpleMap engine and automatic transmission and Pac02 tyres, on flat
rigid terrain with μ equal to the tyres' reference friction (μ scale 1).

    make fixtures-chrono

Writes fixtures/chrono/vehicle_<name>.json, consumed by crates/vehicles/tests/ground_vehicle.rs:
* `design`: total mass, the spindle positions and the chassis COM in the chassis reference
  frame as initialised (design ride height, before settling);
* `static`: after settling, the chassis reference height and attitude, the spindle positions
  (chassis frame) and heights, and the vertical tyre loads;
* `accel`: full throttle from standstill (t, x, speed, gear, engine speed);
* `coast`: released throttle from 25 m/s, in gear (engine drag from the zero-throttle map);
* `brake`: full brake from 25 m/s;
* `suspension`: per axle (left corner), the settled state under gravity scaled by `g`
  (spindle position, camber and toe in the chassis frame, tyre load, spring and shock
  lengths): the kinematics and the effective wheel rate, from which the presets are built.
"""

import json
import math
import pathlib
import sys

import pychrono as ch
import pychrono.vehicle as veh

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "chrono"
MU0 = 0.8
DT = 5e-4
VEHICLES = {
    "sedan": ("sedan/vehicle/Sedan_Vehicle.json", "sedan/powertrain/Sedan_EngineSimpleMap.json",
              "sedan/powertrain/Sedan_AutomaticTransmissionSimpleMap.json", "sedan/tire/Sedan_Pac02Tire.json"),
    "hmmwv": ("hmmwv/vehicle/HMMWV_Vehicle_4WD.json", "hmmwv/powertrain/HMMWV_EngineSimpleMap.json",
              "hmmwv/powertrain/HMMWV_AutomaticTransmissionSimpleMap.json", "hmmwv/tire/HMMWV_Pac02Tire.json"),
}


def vec(v) -> list[float]:
    return [v.x, v.y, v.z]


class Run:
    def __init__(self, files, z0: float, speed: float, g: float = 9.81, patch=(4000.0, 40.0), start=-1900.0):
        f = veh.GetVehicleDataFile
        self.sys = ch.ChSystemNSC()
        self.sys.SetGravitationalAcceleration(ch.ChVector3d(0, 0, -g))
        self.sys.SetCollisionSystemType(ch.ChCollisionSystem.Type_BULLET)
        self.terrain = veh.RigidTerrain(self.sys)
        mat = ch.ChContactMaterialNSC()
        mat.SetFriction(MU0)
        self.terrain.AddPatch(mat, ch.ChCoordsysd(ch.ChVector3d(0, 0, 0), ch.QUNIT), *patch)
        self.terrain.Initialize()
        v = veh.WheeledVehicle(self.sys, f(files[0]))
        v.Initialize(ch.ChCoordsysd(ch.ChVector3d(start, 0, z0), ch.QUNIT), speed)
        self.start = start
        v.InitializePowertrain(veh.ChPowertrainAssembly(veh.ReadEngineJSON(f(files[1])),
                                                        veh.ReadTransmissionJSON(f(files[2]))))
        self.tires = []
        for ax in v.GetAxles():
            for w in ax.GetWheels():
                t = veh.ReadTireJSON(f(files[3]))
                v.InitializeTire(t, w, ch.VisualizationType_NONE)
                self.tires.append(t)
        self.v = v
        self.inputs = veh.DriverInputs()
        self.t = 0.0

    def spindles(self) -> list:
        return [self.v.GetSpindlePos(i, side) for i in range(self.v.GetNumberAxles()) for side in (veh.LEFT, veh.RIGHT)]

    def step(self):
        self.v.Synchronize(self.t, self.inputs, self.terrain)
        self.terrain.Synchronize(self.t)
        self.v.Advance(DT)
        self.terrain.Advance(DT)
        self.sys.DoStepDynamics(DT)
        self.t += DT

    def sample(self) -> dict:
        tr, eng = self.v.GetTransmission(), self.v.GetEngine()
        return {"t": round(self.t, 6), "x": self.v.GetPos().x - self.start, "speed": self.v.GetSpeed(),
                "gear": tr.GetCurrentGear(), "engine_rpm": eng.GetMotorSpeed() * 30 / 3.141592653589793}

    def record(self, duration: float, every: float = 0.05) -> list[dict]:
        out, k, n = [self.sample()], 0, round(every / DT)
        while self.t < duration - 1e-9:
            self.step()
            k += 1
            if k % n == 0:
                out.append(self.sample())
        return out


def suspension_sweep(name: str, files) -> list[list[dict]]:
    """Settle under several gravity scales; record each axle's left corner."""
    f = veh.GetVehicleDataFile
    axles = json.loads(pathlib.Path(f(files[0])).read_text())["Axles"]
    specs = [json.loads(pathlib.Path(f(a["Suspension Input File"])).read_text()) for a in axles]
    out = [[] for _ in axles]
    for scale in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        run = Run(files, 0.5, 0.0, 9.81 * scale)
        ref0 = run.v.GetChassis().GetBody().GetFrameRefToAbs()
        # Lower control arms (left) by axle, and their frames at design.
        arms = sorted((b for b in run.sys.GetBodies() if b.GetName().endswith("LCA_L")), key=lambda b: -b.GetPos().x)
        arm_local = []
        for i, (a, spec) in enumerate(zip(axles, specs)):
            loc = ch.ChVector3d(*a["Suspension Location"])
            pts = {k: ref0.TransformPointLocalToParent(loc + ch.ChVector3d(*spec[k]["Location Arm"])) for k in ("Spring", "Shock")}
            arm_local.append({k: arms[i].GetFrameRefToAbs().TransformPointParentToLocal(v) for k, v in pts.items()})
        run.inputs.m_braking = 1.0
        for _ in range(round(4.0 / DT)):
            run.step()
        ref = run.v.GetChassis().GetBody().GetFrameRefToAbs()
        for i, (a, spec) in enumerate(zip(axles, specs)):
            loc = ch.ChVector3d(*a["Suspension Location"])
            p = ref.TransformPointParentToLocal(run.v.GetSpindlePos(i, veh.LEFT))
            axis = ref.TransformDirectionParentToLocal(run.v.GetSpindleRot(i, veh.LEFT).Rotate(ch.ChVector3d(0, 1, 0)))
            lengths = {}
            for k in ("Spring", "Shock"):
                top = ref.TransformPointLocalToParent(loc + ch.ChVector3d(*spec[k]["Location Chassis"]))
                bottom = arms[i].GetFrameRefToAbs().TransformPointLocalToParent(arm_local[i][k])
                lengths[k.lower() + "_length"] = (top - bottom).Length()
            out[i].append({"g": scale, "spindle": vec(p), "camber": math.asin(axis.z), "toe": math.atan2(-axis.x, axis.y),
                           "load": run.tires[2 * i].ReportTireForce(run.terrain).force.z, **lengths})
        print(f"{name} g x{scale}: travel {[round(o[-1]['spindle'][2], 4) for o in out]}", file=sys.stderr)
    return out


def generate(name: str, files) -> dict:
    run = Run(files, 0.5, 0.0)
    ref = run.v.GetChassis().GetBody().GetFrameRefToAbs()
    design = {
        "mass": run.v.GetMass(),
        "com": vec(run.v.GetCOMFrame().GetPos()),
        "spindles": [vec(ref.TransformPointParentToLocal(p)) for p in run.spindles()],
        "tyre_radius": run.tires[0].GetRadius(),
    }
    # Settle from a small drop.
    run.inputs.m_braking = 1.0
    for _ in range(round(4.0 / DT)):
        run.step()
    ref = run.v.GetChassis().GetBody().GetFrameRefToAbs()
    static = {
        "ref_z": ref.GetPos().z,
        "pitch": run.v.GetPitch(),
        "roll": run.v.GetRoll(),
        "spindles": [vec(ref.TransformPointParentToLocal(p)) for p in run.spindles()],
        "spindle_heights": [p.z for p in run.spindles()],
        "loads": [t.ReportTireForce(run.terrain).force.z for t in run.tires],
    }
    print(f"{name}: mass {design['mass']:.1f} kg, ride {static['ref_z']:.4f} m, loads {[round(l) for l in static['loads']]}", file=sys.stderr)
    # Full throttle from rest (the settled state).
    run.inputs.m_braking = 0.0
    run.inputs.m_throttle = 1.0
    run.t = 0.0
    accel = run.record(30.0)
    runs = {}
    for label, throttle, brake, duration in (("coast", 0.0, 0.0, 30.0), ("brake", 0.0, 1.0, 6.0)):
        r = Run(files, static["ref_z"], 25.0)
        # Let the tyres build up load and slip state at speed before the manoeuvre.
        r.inputs.m_throttle = 0.35 if name == "sedan" else 0.5
        r.record(2.0)
        r.inputs.m_throttle, r.inputs.m_braking = throttle, brake
        r.t = 0.0
        runs[label] = r.record(duration)
    return {"generator": "tools/gen_chrono_vehicle_fixtures.py", "vehicle": name, "mu": MU0, "dt": DT,
            "design": design, "static": static, "accel": accel, **runs, "suspension": suspension_sweep(name, files)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, files in VEHICLES.items():
        data = generate(name, files)
        path = OUT / f"vehicle_{name}.json"
        path.write_text(json.dumps(data, indent=0))
        a = data["accel"]
        t100 = next((s["t"] for s in a if s["speed"] >= 100 / 3.6), None)
        print(f"wrote {path}: 0-100 {t100} s, coast 30 s -> {data['coast'][-1]['speed']:.2f} m/s, "
              f"brake stop at x {data['brake'][-1]['x'] - data['brake'][0]['x']:.1f} m", file=sys.stderr)


if __name__ == "__main__":
    main()
