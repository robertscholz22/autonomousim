"""Generate assets/vehicles/c172_like.toml from JSBSim's c172p model (offline; JSBSim is an
oracle, not a dependency).

Mass properties come from JSBSim itself at the loading used throughout (1500 lb empty, two
90 lb occupants side by side so the centre of mass stays on the centreline, 2 × 100 lb of
fuel); aerodynamics, gear, propeller and engine from the XML files. Everything is converted to
SI units and to the body frame (FLU, origin at the centre of mass).

    ~/.local/share/autonomousim-oracles/jsbsim-venv/bin/python tools/gen_c172_like.py
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

import jsbsim

IN = 0.0254
FT = 0.3048
LBF = 4.4482216152605
SLUG = 14.593902937
SLUGFT2 = SLUG * FT * FT
DEG = math.pi / 180.0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "vehicles", "c172_like.toml")


def mass_properties():
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.load_model("c172p")
    fdm["inertia/pointmass-weight-lbs[0]"] = 90.0
    fdm["inertia/pointmass-weight-lbs[1]"] = 90.0
    fdm["propulsion/tank[0]/contents-lbs"] = 100.0
    fdm["propulsion/tank[1]/contents-lbs"] = 100.0
    fdm["ic/h-sl-ft"] = 3000.0
    fdm["ic/vc-kts"] = 100.0
    fdm.run_ic()
    cg = [fdm[f"inertia/cg-{a}-in"] for a in "xyz"]
    return {
        "mass": fdm["inertia/mass-slugs"] * SLUG,
        "cg": cg,
        "inertia": [fdm[f"inertia/i{a}{a}-slugs_ft2"] * SLUGFT2 for a in "xyz"],
        # JSBSim reports −∫xz dm (FRD); Beard & McLain's J_xz is +∫xz dm.
        "jxz": -fdm["inertia/ixz-slugs_ft2"] * SLUGFT2,
        "root": fdm.get_root_dir(),
    }


def static_torque():
    """Shaft torque (N m) and rpm at full throttle, standing on the brakes at sea level."""
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.load_model("c172p")
    fdm["ic/h-agl-ft"] = 0.0
    fdm["ic/vc-kts"] = 0.0
    fdm.run_ic()
    fdm["propulsion/set-running"] = -1
    for cmd in ["mixture-cmd-norm", "throttle-cmd-norm", "left-brake-cmd-norm", "right-brake-cmd-norm"]:
        fdm[f"fcs/{cmd}"] = 1.0
    for _ in range(int(10.0 / fdm.get_delta_t())):
        fdm.run()
    rpm = fdm["propulsion/engine/engine-rpm"]
    return fdm["propulsion/engine/power-hp"] * 745.699872 / (rpm * math.pi / 30.0), rpm


def body(loc, cg):
    """JSBSim structural inches (x aft, y right, z up) → FLU metres about the CG."""
    return [(cg[0] - loc[0]) * IN, (cg[1] - loc[1]) * IN, (loc[2] - cg[2]) * IN]


def location(e):
    return [float(e.find(a).text) for a in "xyz"]


def fmt(x):
    return f"{x:.6g}" if abs(x) >= 1e-4 or x == 0 else f"{x:.4e}"


def vec(v):
    return "[" + ", ".join(fmt(x) for x in v) + "]"


VARS = {
    "aero/alpha-rad": "alpha",
    "aero/beta-rad": "beta",
    "aero/mag-beta-rad": "abs_beta",
    "fcs/elevator-pos-rad": "elevator",
    "fcs/mag-elevator-pos-rad": "abs_elevator",
    "fcs/left-aileron-pos-rad": "aileron",
    "fcs/rudder-pos-rad": "rudder",
    "fcs/flap-pos-deg": "flap",
    "aero/stall-hyst-norm": "stall",
    "aero/h_b-mac-ft": "height_over_span",
}
DROP = {"aero/qbar-psf", "metrics/Sw-sqft", "metrics/bw-ft", "metrics/cbarw-ft"}
RATES = {
    ("aero/bi2vel", "velocities/p-aero-rad_sec"): "p_hat",
    ("aero/bi2vel", "velocities/r-aero-rad_sec"): "r_hat",
    ("aero/ci2vel", "velocities/q-aero-rad_sec"): "q_hat",
    ("aero/ci2vel", "aero/alphadot-rad_sec"): "alpha_dot_hat",
}


def breakpoints(var, xs):
    return [x * DEG for x in xs] if var == "flap" else xs


def table(e):
    ivs = [VARS[i.text.strip()] for i in e.findall("independentVar")]
    rows = [[float(x) for x in line.split()] for line in e.find("tableData").text.strip().splitlines()]
    if len(ivs) == 1:
        return {"row": ivs[0], "rows": breakpoints(ivs[0], [r[0] for r in rows]), "data": [r[1] for r in rows]}
    cols = rows[0]
    return {
        "row": ivs[0],
        "rows": breakpoints(ivs[0], [r[0] for r in rows[1:]]),
        "col": ivs[1],
        "cols": breakpoints(ivs[1], cols),
        "data": [v for r in rows[1:] for v in r[1:]],
    }


def term(fn, functions):
    prod = [c for c in fn if c.tag not in ("description",)][0]
    assert prod.tag == "product", prod.tag
    t = {"name": fn.get("name").split("/")[-1], "scale": 1.0, "vars": [], "tables": [], "pressure": "free"}
    props = []
    for c in prod:
        if c.tag == "property":
            props.append(c.text.strip())
        elif c.tag == "value":
            t["scale"] *= float(c.text)
        elif c.tag == "table":
            t["tables"].append(table(c))
        else:
            raise ValueError(c.tag)
    for pair, var in RATES.items():
        if all(p in props for p in pair):
            for p in pair:
                props.remove(p)
            t["vars"].append(var)
    for p in props:
        if p in DROP:
            continue
        if p == "aero/function/qbar-induced-psf":
            t["pressure"] = "slipstream"
        elif p in functions:
            t["tables"].append(functions[p])
        else:
            t["vars"].append(VARS[p])
    return t


def aero_section(root):
    aero = root.find("aerodynamics")
    functions = {}
    for f in aero.findall("function"):
        tb = f.find("table")
        if tb is not None:
            functions[f.get("name")] = table(tb)
    hyst = aero.find("hysteresis_limits")
    lines = [
        "[aero]",
        'model = "tables"',
        f"stall_hysteresis = [{fmt(float(hyst.find('min').text))}, {fmt(float(hyst.find('max').text))}]",
    ]
    names = {"DRAG": "drag", "SIDE": "side", "LIFT": "lift", "ROLL": "roll", "PITCH": "pitch", "YAW": "yaw"}
    for ax in aero.findall("axis"):
        axis = names[ax.get("name")]
        for f in ax.findall("function"):
            t = term(f, functions)
            if t["scale"] == 0.0:
                continue
            lines += ["", f"[[aero.{axis}]]", f'name = "{t["name"]}"']
            if t["scale"] != 1.0:
                lines.append(f"scale = {fmt(t['scale'])}")
            if t["vars"]:
                lines.append("vars = [" + ", ".join(f'"{v}"' for v in t["vars"]) + "]")
            if t["pressure"] != "free":
                lines.append(f'pressure = "{t["pressure"]}"')
            for tb in t["tables"]:
                lines += [f"[[aero.{axis}.tables]]", f'row = "{tb["row"]}"', f"rows = {vec(tb['rows'])}"]
                if "col" in tb:
                    lines += [f'col = "{tb["col"]}"', f"cols = {vec(tb['cols'])}"]
                lines.append(f"data = {vec(tb['data'])}")
    return lines


def curve(e):
    rows = [[float(x) for x in line.split()] for line in e.find("tableData").text.strip().splitlines()]
    return "{ x = " + vec([r[0] for r in rows]) + ", y = " + vec([r[1] for r in rows]) + " }"


def main():
    mp = mass_properties()
    cg = mp["cg"]
    ac = os.path.join(mp["root"], "aircraft", "c172p", "c172p.xml")
    root = ET.parse(ac).getroot()
    m = root.find("metrics")
    area = float(m.find("wingarea").text) * FT * FT
    span = float(m.find("wingspan").text) * FT
    chord = float(m.find("chord").text) * FT
    arp = body(location(m.find("location[@name='AERORP']")), cg)

    eng = root.find("propulsion/engine")
    thr = eng.find("thruster")
    prop_pos = body(location(thr.find("location")), cg)
    sense = float(thr.find("sense").text)
    engine = ET.parse(os.path.join(mp["root"], "engine", eng.get("file") + ".xml")).getroot()
    prop = ET.parse(os.path.join(mp["root"], "engine", thr.get("file") + ".xml")).getroot()
    tables = {t.get("name"): t for t in prop.findall("table")}
    # JSBSim's piston model makes about 575 N m at full throttle at sea level, well above the
    # 160 hp rating; calibrate the constant-torque model (friction fraction f = 0.2, the
    # default) to that torque at JSBSim's static rpm so the takeoff performance matches.
    rated_rpm = float(engine.find("maxrpm").text)
    torque, rpm = static_torque()
    friction = 0.2
    max_power = torque / (1.0 + friction - friction * rpm / rated_rpm) * rated_rpm * math.pi / 30.0

    out = [
        "# Cessna 172 generated by tools/gen_c172_like.py from JSBSim's c172p model; do not edit.",
        'type = "fixed_wing"',
        'name = "c172_like"',
        'source = """',
        "JSBSim c172p (aircraft/c172p/c172p.xml, engine/eng_io320.xml, engine/prop_75in2f.xml): \\",
        "aerodynamic coefficient build-up, stall hysteresis, gear and propeller tables; engine torque \\",
        "calibrated to JSBSim's full-throttle static run (it exceeds the 160 hp rating). \\",
        "Mass properties from JSBSim at 1880 lb (1500 lb empty, two 90 lb occupants, 2 x 100 lb fuel). \\",
        "Airframe colliders, servo rates and the simplified engine model (constant torque, \\",
        'Gagg-Ferrar altitude factor) are estimates; P-factor and fuel burn are not modelled."""',
        "",
        "[body]",
        f"mass = {fmt(mp['mass'])}",
        f"inertia = {vec(mp['inertia'])}",
        f"product_xz = {fmt(mp['jxz'])}",
        "",
        "[geometry]",
        f"area = {fmt(area)}",
        f"span = {fmt(span)}",
        f"chord = {fmt(chord)}",
        f"aero_reference = {vec(arp)}",
        "",
        "# JSBSim's surface ranges; slow flaps (0-30 deg in about 4 s).",
        "[controls]",
        f"aileron = {{ max = {fmt(15 * DEG)}, min = {fmt(-20 * DEG)}, rate = 2.0, tau = 0.05 }}",
        f"elevator = {{ max = {fmt(23 * DEG)}, min = {fmt(-28 * DEG)}, rate = 2.0, tau = 0.05 }}",
        f"rudder = {{ max = {fmt(16 * DEG)}, rate = 2.0, tau = 0.05 }}",
        f"flap = {{ max = {fmt(30 * DEG)}, rate = {fmt(7.5 * DEG)} }}",
        "",
        "[propulsion]",
        f"position = {vec(prop_pos)}",
        f"sense = {fmt(sense)}",
        "",
        "[propulsion.propeller]",
        f"diameter = {fmt(float(prop.find('diameter').text) * IN)}",
        f"inertia = {fmt(float(prop.find('ixx').text) * SLUGFT2)}",
        f"ct = {curve(tables['C_THRUST'])}",
        f"cp = {curve(tables['C_POWER'])}",
        "",
        "[propulsion.engine]",
        'type = "piston"',
        f"max_power = {fmt(max_power)}",
        f"rated_rpm = {fmt(rated_rpm)}",
        f"idle_rpm = {fmt(float(engine.find('idlerpm').text))}",
    ]
    for c in root.findall("ground_reactions/contact"):
        if c.get("type") != "BOGEY":
            continue
        k = float(c.find("spring_coeff").text) * LBF / FT
        d = float(c.find("damping_coeff").text) * LBF / FT
        brake = c.find("brake_group").text.strip() != "NONE"
        steer = float(c.find("max_steer").text) * DEG
        out += [
            "",
            "[[gear]]",
            f'name = "{c.get("name").lower()}"',
            f"position = {vec(body(location(c.find('location')), cg))}",
            f"spring = {fmt(k)}",
            f"damping = {fmt(d)}",
            f"rolling_friction = {fmt(float(c.find('rolling_friction').text))}",
            f"side_friction = {fmt(float(c.find('static_friction').text))}",
        ]
        if brake:
            out.append("brake_friction = 0.6")
        if steer > 0:
            out.append(f"max_steer = {fmt(steer)}")
        out.append("wheel_radius = 0.2")
    structure = {c.get("name"): body(location(c.find("location")), cg) for c in root.findall("ground_reactions/contact")
                 if c.get("type") == "STRUCTURE"}
    colliders = [
        ("propeller disc", prop_pos, 0.95, "rotor"),
        ("tail skid", structure["TAIL_SKID"], 0.1, "frame"),
        ("left wing tip", structure["LEFT_TIP"], 0.15, "frame"),
        ("right wing tip", structure["RIGHT_TIP"], 0.15, "frame"),
        ("cabin", [0.5, 0.0, 0.0], 0.6, "frame"),
        ("tail cone", [-2.0, 0.0, -0.1], 0.35, "frame"),
    ]
    for name, c, r, part in colliders:
        out += ["", f"# {name}", "[[colliders]]", f"center = {vec(c)}", f"radius = {fmt(r)}", f'part = "{part}"']
    out += [""] + aero_section(root) + [""]
    with open(OUT, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
