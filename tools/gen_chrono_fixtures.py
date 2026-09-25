"""Generate Magic Formula cross-check values with Project Chrono's ChPac02Tire (PyChrono).

    make fixtures-chrono

(runs this script in the `chrono` micromamba environment, see docs/PLAN.md, M2 step 0).
Writes fixtures/chrono/<tyre>.json, consumed by crates/vehicles/tests/tire_fixtures.rs.

The tyre is evaluated directly (Synchronize + Advance) on a wheel whose state is set for
each point, on flat rigid terrain with the friction the tyre file was fitted for (μ scale 1).
Chrono computes its slip quantities from that state; the fixture records Chrono's own
(κ, α, F_z) with its outputs in Chrono's internal convention (α = atan(−v_y/(v_x + 0.1)),
F_y and M_z negated with respect to the ISO forces it reports), so the Rust test evaluates
the formula at exactly those inputs.

Chrono deviates from the MF 5.2 reference (MFeval) in ways the test accounts for:
* camber is never passed to the formulas (γ = 0 always);
* B·x is clamped to ±(π/2 − 0.01) in Fx0 and Fy0, flattening the curves past the peak;
* the pneumatic trail lacks MFeval's (TNO's) factor LFZO;
* the stiffness factors have +0.1 in the denominator (negligible);
* in combined mode the equivalent trail angle takes the sign of κ instead of α_t.
The sweeps therefore use USE_MODE 3 (uncombined: F_x0, F_y0 and M_z0) and USE_MODE 4 with
FE_METHOD 'NO' (Pacejka combined-slip functions; F_x and F_y compared).
"""

import json
import os
import pathlib
import sys
import tempfile

import pychrono as ch
import pychrono.vehicle as veh

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "chrono"
TYRES = ["HMMWV_Pac02Tire", "Sedan_Pac02Tire", "Truck_Pac02Tire"]
MU0 = 0.8  # "Coefficient of Friction" of the tyre JSON; road μ equal to it gives μ scale 1


def variant(tir: pathlib.Path, mode: int, tmp: pathlib.Path) -> pathlib.Path:
    """A copy of the .tir file with USE_MODE `mode` and Pacejka combined slip, wrapped in the
    JSON file that Chrono's Pac02Tire reads."""
    lines = []
    for line in tir.read_text().splitlines():
        key = line.split("=")[0].strip().upper()
        if key == "USE_MODE":
            line = f"USE_MODE = {mode}"
        lines.append(line)
        if line.strip().upper() == "[MODEL]":
            lines.append("FE_METHOD = 'NO'")
    path = tmp / f"{tir.stem}_mode{mode}.tir"
    path.write_text("\n".join(lines) + "\n")
    spec = {
        "Name": tir.stem,
        "Type": "Tire",
        "Template": "Pac02Tire",
        "Mass": 30.0,
        "Inertia": [2.0, 4.0, 2.0],
        # Chrono resolves the path against its vehicle data directory.
        "TIR Specification File": os.path.relpath(path, veh.GetVehicleDataPath()),
        "Coefficient of Friction": MU0,
    }
    js = tmp / f"{tir.stem}_mode{mode}.json"
    js.write_text(json.dumps(spec))
    return js


class Rig:
    """A wheel with a prescribed state above flat terrain."""

    def __init__(self, spec: pathlib.Path):
        self.sys = ch.ChSystemNSC()
        self.terrain = veh.RigidTerrain(self.sys)
        mat = ch.ChContactMaterialNSC()
        mat.SetFriction(MU0)
        self.terrain.AddPatch(mat, ch.ChCoordsysd(ch.ChVector3d(0, 0, 0), ch.QUNIT), 500, 500)
        self.terrain.Initialize()
        self.spindle = veh.ChSpindle()
        self.sys.AddBody(self.spindle)
        self.wheel = veh.Wheel(veh.GetVehicleDataFile("hmmwv/wheel/HMMWV_Wheel.json"))
        self.wheel.Initialize(None, self.spindle, veh.LEFT, 0)
        self.tire = veh.Pac02Tire(str(spec))
        self.tire.SetStepsize(1e-3)
        self.tire.Initialize(self.wheel)
        self.wheel.SetTire(self.tire)
        self.radius = self.tire.GetRadius()

    def evaluate(self, depth: float, vx: float, slip: float, heading: float) -> dict:
        s = self.spindle
        s.SetPos(ch.ChVector3d(0, 0, self.radius - depth))
        s.SetRot(ch.QuatFromAngleZ(heading))
        s.SetPosDt(ch.ChVector3d(vx, 0, 0))
        s.SetAngVelLocal(ch.ChVector3d(0, vx * (1 + slip) / self.radius, 0))
        self.tire.Synchronize(0.0, self.terrain)
        self.tire.Advance(1e-3)
        frame = ch.ChCoordsysd()
        f = self.tire.ReportTireForceLocal(self.terrain, frame)
        return {
            "fz": f.force.z,
            "kappa": self.tire.GetLongitudinalSlip_internal(),
            "alpha": self.tire.GetSlipAngle_internal(),
            "Fx": f.force.x,
            "Fy": -f.force.y,
            "Mz": -f.moment.z,
        }


def depths(rig: Rig, loads: list[float]) -> list[float]:
    """Deflections giving roughly the loads (the normal force is linear in depth here)."""
    ref = 0.01
    fz = rig.evaluate(ref, 10.0, 0.0, 0.0)["fz"]
    return [ref * load / fz for load in loads]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vx = 15.0  # above Chrono's low-speed blending (1–3 m/s)
    for name in TYRES:
        tir = ROOT / "assets" / "tires" / f"{name}.tir"
        fz0 = next(float(l.split("=")[1].split("$")[0]) for l in tir.read_text().splitlines() if l.split("=")[0].strip() == "FNOMIN")
        lfzo = next(float(l.split("=")[1].split("$")[0]) for l in tir.read_text().splitlines() if l.split("=")[0].strip() == "LFZO")
        loads = [f * fz0 * lfzo for f in (0.5, 1.0, 1.5)]
        with tempfile.TemporaryDirectory() as tmp:
            pure, combined = Rig(variant(tir, 3, pathlib.Path(tmp))), Rig(variant(tir, 4, pathlib.Path(tmp)))
            pts = {"pure_kappa": [], "pure_alpha": [], "combined": []}
            for d in depths(pure, loads):
                for k in range(-20, 21):
                    pts["pure_kappa"].append(pure.evaluate(d, vx, 0.02 * k, 0.0))
                for a in range(-20, 21):
                    pts["pure_alpha"].append(pure.evaluate(d, vx, 0.0, 0.015 * a))
            for d in depths(combined, loads):
                for k in range(-6, 7):
                    for a in range(-6, 7):
                        pts["combined"].append(combined.evaluate(d, vx, 0.03 * k, 0.03 * a))
        data = {"generator": f"tools/gen_chrono_fixtures.py (PyChrono {ch.GetChronoVersion() if hasattr(ch, 'GetChronoVersion') else '10'})", "tyre": name, **pts}
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=0))
        print(f"wrote {OUT / name}.json", file=sys.stderr)


if __name__ == "__main__":
    main()
