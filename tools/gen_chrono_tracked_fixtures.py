"""Reference data for the `tracked_apc` preset from Chrono::Vehicle's M113 (PyChrono, BSD-3):
single-pin shoes, the braked-differential-steering driveline (BDS), simple brakes, on flat
rigid ground with μ = 0.8 (the reference friction of our materials). Chrono's M113 is run with
NSC contact, as in its demos: with SMC contact at 0.5 ms the braked vehicle creeps at 0.5 m/s
on its chattering shoes. Even with NSC the shoes chatter (the hull rocks by ±0.1 m/s), so
static values are averaged over the last seconds.

    make fixtures-chrono      (slow: Chrono runs the M113 at ~1/60 real time)
    python tools/gen_chrono_tracked_fixtures.py [design] [static]

Writes fixtures/chrono/tracked_m113.json, consumed by crates/vehicles/tests/tracks.rs and used
to build assets/vehicles/tracked_apc.toml:
* `design`: the hull with everything that does not move relative to it in our model lumped in
  (hull, sprockets, idlers and their carriers, track shoes): mass, centre of mass and inertia
  about it in chassis axes; road wheels, suspension arms (masses, inertias, pivot), sprocket
  and idler positions and radii, shoe mass, pitch and height, brake torque;
* `static`: after settling on the brakes, averaged: chassis (reference frame) height and pitch, road-wheel
  positions in the chassis frame and above the ground, arm angles and torsion-spring torques
  (which include the track tension: the springs carry about twice the weight), and the
  vertical ground reaction under each road wheel (see `GroundLoads`);
* `driveline`: the conical gear ratio of the BDS driveline.
"""

import json
import pathlib
import sys

import pychrono as ch
import pychrono.vehicle as veh

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "chrono" / "tracked_m113.json"
MU0 = 0.8
DT = 5e-4
Z0 = 0.66  # chassis reference frame, 3 cm above its rest height


def vec(v) -> list[float]:
    return [v.x, v.y, v.z]


class M113:
    """Chrono's M113 on flat rigid ground."""

    def __init__(self):
        m = veh.M113()
        m.SetContactMethod(ch.ChContactMethod_NSC)
        m.SetTrackShoeType(veh.TrackShoeType_SINGLE_PIN)
        m.SetDrivelineType(veh.DrivelineTypeTV_BDS)
        m.SetEngineType(veh.EngineModelType_SIMPLE_MAP)
        m.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
        m.SetBrakeType(veh.BrakeType_SIMPLE)
        m.SetInitPosition(ch.ChCoordsysd(ch.ChVector3d(0, 0, Z0), ch.QUNIT))
        m.Initialize()
        self.m = m
        self.v = m.GetVehicle()
        sys_ = self.v.GetSystem()
        sys_.SetGravitationalAcceleration(ch.ChVector3d(0, 0, -9.81))
        sys_.SetSolverType(ch.ChSolver.Type_BARZILAIBORWEIN)
        sys_.GetSolver().AsIterative().SetMaxIterations(150)
        self.terrain = veh.RigidTerrain(sys_)
        mat = ch.ChContactMaterialNSC()
        mat.SetFriction(MU0)
        mat.SetRestitution(0.01)
        self.terrain.AddPatch(mat, ch.ChCoordsysd(ch.ChVector3d(0, 0, 0), ch.QUNIT), 200, 40)
        self.terrain.Initialize()
        self.inputs = veh.DriverInputs()
        self.t = 0.0

    def step(self):
        self.m.Synchronize(self.t, self.inputs)
        self.terrain.Synchronize(self.t)
        self.m.Advance(DT)
        self.terrain.Advance(DT)
        self.t += DT

    def frame(self):
        b = self.v.GetChassisBody()
        return b.GetFrameRefToAbs() if hasattr(b, "GetFrameRefToAbs") else b.GetFrame_REF_to_abs()

    def track(self, side):
        return self.v.GetTrackAssembly(veh.LEFT if side == 0 else veh.RIGHT)


def lumped(bodies) -> dict:
    """Mass, centre of mass and inertia about it (chassis axes, full matrix) of rigid bodies."""
    mass = sum(b.GetMass() for b in bodies)
    com = [sum(b.GetMass() * vec(b.GetPos())[i] for b in bodies) / mass for i in range(3)]
    inertia = [[0.0] * 3 for _ in range(3)]
    for b in bodies:
        rot = b.GetRot()
        axes = [rot.Rotate(ch.ChVector3d(*e)) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        local = [vec(b.GetInertiaXX())[i] for i in range(3)]
        # Rotated principal inertia (the bodies' axes are their principal axes here) plus the
        # parallel-axis term.
        d = [vec(b.GetPos())[i] - com[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                inertia[i][j] += sum(local[k] * vec(axes[k])[i] * vec(axes[k])[j] for k in range(3))
                inertia[i][j] += b.GetMass() * ((sum(x * x for x in d) if i == j else 0.0) - d[i] * d[j])
    return {"mass": mass, "com": com, "inertia": inertia}


def design(sim: M113) -> dict:
    v = sim.v
    frame = sim.frame()
    local = lambda p: vec(frame.TransformPointParentToLocal(p))
    rigid = [v.GetChassisBody()]
    sides = []
    for side in (0, 1):
        ta = sim.track(side)
        sp, idl = ta.GetSprocket(), ta.GetIdler()
        rigid += [sp.GetGearBody(), idl.GetIdlerWheel().GetBody(), idl.GetCarrierBody()]
        rigid += [ta.GetTrackShoe(i).GetShoeBody() for i in range(ta.GetNumTrackShoes())]
        wheels = []
        for i in range(ta.GetNumTrackSuspensions()):
            s = ta.GetTrackSuspension(i)
            w, arm = s.GetWheelBody(), s.GetCarrierBody()
            wheels.append({
                "position": local(w.GetPos()),
                "mass": w.GetMass(),
                "inertia": vec(w.GetInertiaXX()),
                "radius": s.GetWheelRadius(),
                "arm_mass": arm.GetMass(),
                "arm_com": local(arm.GetPos()),
                "arm_inertia": vec(arm.GetInertiaXX()),
            })
        shoe = ta.GetTrackShoe(0)
        sides.append({
            "road_wheels": wheels,
            "sprocket": {"position": local(sp.GetGearBody().GetPos()), "radius": sp.GetAssemblyRadius(),
                         "mass": sp.GetGearBody().GetMass(), "inertia": vec(sp.GetGearBody().GetInertiaXX())},
            "idler": {"position": local(idl.GetIdlerWheel().GetBody().GetPos()),
                      "radius": idl.GetIdlerWheel().GetRadius(), "mass": idl.GetIdlerWheel().GetBody().GetMass(),
                      "inertia": vec(idl.GetIdlerWheel().GetBody().GetInertiaXX())},
            "shoes": ta.GetNumTrackShoes(),
            "shoe": {"mass": shoe.GetShoeBody().GetMass(), "pitch": shoe.GetPitch(), "height": shoe.GetHeight()},
        })
    hull = v.GetChassisBody()
    return {
        "total_mass": v.GetMass(),
        "hull": {"mass": hull.GetMass(), "com": local(hull.GetPos()), "inertia": vec(hull.GetInertiaXX())},
        "sprung": lumped(rigid) | {"com": local(ch.ChVector3d(*lumped(rigid)["com"]))},
        "sides": sides,
        "brake_torque": 10000.0,
    }


class GroundLoads(ch.ReportContactCallback):
    """Vertical ground reaction summed per road wheel: over the band between the midpoints to
    the neighbouring road wheels (chassis x), the outermost bins open-ended, per side."""

    def __init__(self, frame, edges):
        super().__init__()
        self.frame, self.edges = frame, edges
        self.loads = [0.0] * (2 * (len(edges) + 1))

    def OnReportContact(self, pA, pB, plane, distance, radius, force, torque, a, b, offset=0):
        n = plane.GetAxisX()
        if abs(pA.z) > 0.02 or abs(n.z) < 0.9:
            return True  # not on the ground (wheels and rollers on the shoes)
        p = self.frame.TransformPointParentToLocal(pA)
        side = 0 if p.y > 0 else 1
        k = sum(p.x < e for e in self.edges)
        self.loads[side * (len(self.edges) + 1) + k] += abs(force.x * n.z)
        return True


def settle(sim: M113, seconds: float = 5.0, average: float = 2.0, every: int = 20) -> dict:
    sim.inputs.m_braking = 1.0
    n = 0
    acc = {"height": 0.0, "pitch": 0.0, "wheels": None, "ground": None}
    ta = sim.track(0)
    xs = [ta.GetTrackSuspension(i).GetWheelBody().GetPos().x - sim.frame().GetPos().x
          for i in range(ta.GetNumTrackSuspensions())]
    edges = [0.5 * (a + b) for a, b in zip(xs, xs[1:])]
    k = 0
    while sim.t < seconds:
        sim.step()
        k += 1
        if sim.t < seconds - average or k % every:
            continue
        frame = sim.frame()
        report = GroundLoads(frame, edges)
        sim.v.GetSystem().GetContactContainer().ReportAllContacts(report)
        acc["ground"] = report.loads if acc["ground"] is None else [
            a + b for a, b in zip(acc["ground"], report.loads)]
        rows = []
        for side in (0, 1):
            ta = sim.track(side)
            for i in range(ta.GetNumTrackSuspensions()):
                s = ta.GetTrackSuspension(i)
                w = s.GetWheelBody().GetPos()
                f = s.ReportSuspensionForce()
                rows.append(vec(frame.TransformPointParentToLocal(w)) + [w.z, s.GetCarrierAngle(), f.spring_ft])
        acc["height"] += frame.GetPos().z
        acc["pitch"] += sim.v.GetPitch()
        acc["wheels"] = rows if acc["wheels"] is None else [
            [a + b for a, b in zip(r0, r1)] for r0, r1 in zip(acc["wheels"], rows)]
        n += 1
    wheels = [[x / n for x in r] for r in acc["wheels"]]
    return {
        "height": acc["height"] / n,
        "pitch": acc["pitch"] / n,
        "wheels": [{"position": r[:3], "center_height": r[3], "arm_angle": r[4], "spring_torque": r[5],
                    "ground_load": g / n} for r, g in zip(wheels, acc["ground"])],
    }


def driveline(sim: M113, seconds: float = 3.0, every: int = 100) -> dict:
    """The conical gear ratio (the M113 class's constants are not exposed): mean sprocket speed
    over driveshaft speed while driving off at full throttle, which the open differential fixes
    kinematically. The iterative solver leaves the shaft constraints loose (single samples
    scatter by ±0.1), so the ratio is averaged over the samples after 0.5 s; also the speed
    reached, for reference."""
    sim.inputs.m_throttle = 1.0
    d = sim.v.GetDriveline()
    ratios = []
    k = 0
    while sim.t < seconds:
        sim.step()
        k += 1
        if sim.t > 0.5 and k % every == 0:
            sprocket = 0.5 * (d.GetSprocketSpeed(veh.LEFT) + d.GetSprocketSpeed(veh.RIGHT))
            ratios.append(sprocket / d.GetOutputDriveshaftSpeed())
    mean = sum(ratios) / len(ratios)
    spread = (sum((r - mean) ** 2 for r in ratios) / len(ratios)) ** 0.5
    return {"conical_ratio": mean, "conical_ratio_std": spread, "samples": len(ratios), "time": seconds,
            "speed": sim.v.GetSpeed()}


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    which = sys.argv[1:] or ["design", "static", "driveline"]
    if "design" in which:
        out["design"] = design(M113())
    if "static" in which:
        out["static"] = settle(M113())
    if "driveline" in which:
        out["driveline"] = driveline(M113())
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
