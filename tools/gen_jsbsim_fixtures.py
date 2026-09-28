"""JSBSim reference runs of the c172p for the fixed-wing validation (fixtures/jsbsim/).

Run with the oracle venv: ``make fixtures-jsbsim`` (JSBSim from PyPI, see the Makefile).
The loading matches assets/vehicles/c172_like.toml: two 90 lb occupants, 100 lb fuel per tank.

Writes ``fixtures/jsbsim/c172.json`` with, in SI units and radians:

- ``trim``: straight and level trims at 1000 ft over an airspeed sweep (true airspeed, α,
  pitch, elevator deflection, throttle, engine speed, thrust, air density);
- ``modes``: eigenvalues of the longitudinal and lateral linear models at 90 kt, classified into
  short period, phugoid, Dutch roll, roll subsidence and spiral;
- ``doublet``: the response to an elevator doublet from the 90 kt trim (time, q, θ, α, V);
- ``takeoff``: full-throttle ground roll from rest (distance and time to 55 kt).
"""

import json
import math
import os

import jsbsim
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "fixtures", "jsbsim", "c172.json")

KT = 0.514444
FT = 0.3048
LBF = 4.4482216
SLUG_FT3 = 515.378818


def aircraft(**ic):
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.load_model("c172p")
    fdm["inertia/pointmass-weight-lbs[0]"] = 90.0
    fdm["inertia/pointmass-weight-lbs[1]"] = 90.0
    fdm["propulsion/tank[0]/contents-lbs"] = 100.0
    fdm["propulsion/tank[1]/contents-lbs"] = 100.0
    for k, v in ic.items():
        fdm[f"ic/{k}"] = v
    fdm.run_ic()
    fdm["propulsion/set-running"] = -1
    fdm["fcs/mixture-cmd-norm"] = 1.0
    return fdm


def trim(kts, h_ft=1000.0):
    fdm = aircraft(**{"h-sl-ft": h_ft, "vc-kts": kts, "gamma-deg": 0.0, "psi-true-deg": 0.0})
    fdm["fcs/throttle-cmd-norm"] = 0.6
    fdm.do_trim(1)
    return fdm


def trim_row(fdm):
    return {
        "airspeed": fdm["velocities/vt-fps"] * FT,
        "density": fdm["atmosphere/rho-slugs_ft3"] * SLUG_FT3,
        "alpha": fdm["aero/alpha-rad"],
        "pitch": fdm["attitude/theta-rad"],
        "elevator": fdm["fcs/elevator-pos-rad"],
        "throttle": fdm["fcs/throttle-cmd-norm"],
        "rpm": fdm["propulsion/engine/engine-rpm"],
        "thrust": fdm["propulsion/engine/thrust-lbs"] * LBF,
    }


def classify(eig, lateral):
    """Mode name → (natural frequency rad/s, damping ratio) or time constant for real roots."""
    osc = sorted([e for e in eig if e.imag > 1e-6], key=lambda e: abs(e))
    real = sorted([e.real for e in eig if abs(e.imag) <= 1e-6 and abs(e) > 1e-6], key=abs)
    fz = lambda e: {"wn": abs(e), "zeta": -e.real / abs(e)}  # noqa: E731
    if not lateral:
        return {"phugoid": fz(osc[0]), "short_period": fz(osc[-1])}
    return {
        "dutch_roll": fz(osc[-1]),
        "spiral": {"lambda": real[0]},
        "roll": {"lambda": real[-1]},
    }


def modes(kts):
    fdm = trim(kts)
    lin = jsbsim.FGLinearization(fdm)
    a = np.array(lin.system_matrix)
    names = list(lin.x_names)
    idx = lambda ns: [names.index(n) for n in ns]  # noqa: E731
    # The engine speed couples into the longitudinal modes (its own real root is dropped).
    lon = idx(["Vt", "Alpha", "Theta", "Q", "Rpm0"])
    lat = idx(["Beta", "Phi", "P", "R"])
    out = {}
    for sel, lateral in [(lon, False), (lat, True)]:
        sub = a[np.ix_(sel, sel)]
        out.update(classify(np.linalg.eigvals(sub), lateral))
    return {"airspeed": fdm["velocities/vt-fps"] * FT, "names": names, **out}


def doublet(kts, amplitude=0.05, width=1.0, duration=10.0):
    fdm = trim(kts)
    e0 = fdm["fcs/elevator-cmd-norm"]
    dt = fdm.get_delta_t()
    rows = []
    t = 0.0
    while t < duration:
        # Elevator command (normalised), as a JSBSim pilot would move it.
        cmd = e0 + (amplitude if 1.0 <= t < 1.0 + width else -amplitude if 1.0 + width <= t < 1.0 + 2 * width else 0.0)
        fdm["fcs/elevator-cmd-norm"] = cmd
        fdm.run()
        t += dt
        if round(t / dt) % 6 == 0:
            rows.append(
                [
                    t,
                    fdm["velocities/q-rad_sec"],
                    fdm["attitude/theta-rad"],
                    fdm["aero/alpha-rad"],
                    fdm["velocities/vt-fps"] * FT,
                    fdm["fcs/elevator-pos-rad"],
                ]
            )
    return {"columns": ["time", "q", "theta", "alpha", "airspeed", "elevator"], "rows": rows}


def takeoff():
    fdm = aircraft(**{"h-agl-ft": 0.0, "vc-kts": 0.0, "psi-true-deg": 90.0})
    fdm["fcs/throttle-cmd-norm"] = 1.0
    for cmd in ["left-brake-cmd-norm", "right-brake-cmd-norm"]:
        fdm[f"fcs/{cmd}"] = 1.0
    for _ in range(int(10.0 / fdm.get_delta_t())):
        fdm.run()
    static = {
        "rpm": fdm["propulsion/engine/engine-rpm"],
        "thrust": fdm["propulsion/engine/thrust-lbs"] * LBF,
    }
    for cmd in ["left-brake-cmd-norm", "right-brake-cmd-norm"]:
        fdm[f"fcs/{cmd}"] = 0.0
    x0, t0 = fdm["position/distance-from-start-mag-mt"], fdm.get_sim_time()
    while fdm["velocities/vc-kts"] < 55.0:
        fdm.run()
    return {
        "static": static,
        "distance_to_55kt": fdm["position/distance-from-start-mag-mt"] - x0,
        "time_to_55kt": fdm.get_sim_time() - t0,
    }


def main():
    speeds = [60, 70, 80, 90, 100, 110, 120]
    out = {
        "source": f"JSBSim {jsbsim.__version__} c172p, 1880 lb (2 x 90 lb occupants, 2 x 100 lb fuel)",
        "trim": [trim_row(trim(k)) for k in speeds],
        "modes": modes(90),
        "doublet": doublet(90),
        "takeoff": takeoff(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", OUT)
    for r in out["trim"]:
        print({k: round(v, 4) for k, v in r.items()})
    print(json.dumps({k: v for k, v in out["modes"].items() if k != "names"}, indent=1))
    print(out["takeoff"])


if __name__ == "__main__":
    main()
