"""Generate golden multibody-dynamics fixtures with Pinocchio.

    uv run --group oracle python tools/gen_pinocchio_fixtures.py

Writes fixtures/pinocchio/<model>.json in autonomousim conventions, consumed by
crates/core/tests/pinocchio_fixtures.rs:

* links in topological order: parent index (or null), joint {type, axis}, joint frame
  {pos, rot=[x,y,z,w]} in the parent link frame, inertia {mass, com, i_com (row-major)};
* free-joint velocities are [ω, v] (Pinocchio: [v, ω]); spatial forces are [n, f];
* external forces are per link, in link coordinates about the link origin.

Pinocchio has no fixed joint, so fixed links are merged into the nearest moving ancestor.
"""

import json
import pathlib

import numpy as np
import pinocchio as pin

G = 9.80665
OUT = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "pinocchio"


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def random_quat(rng):
    return unit(rng.normal(size=4))  # x, y, z, w


def quat_to_rot(q):
    return pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()


def random_inertia(rng):
    mass = rng.uniform(0.2, 3.0)
    # Principal moments satisfying the triangle inequality, rotated randomly.
    p = rng.uniform(0.01, 0.2, size=3)
    p[2] = min(p[2], p[0] + p[1] - 1e-3)
    r = quat_to_rot(random_quat(rng))
    return {"mass": mass, "com": rng.uniform(-0.2, 0.2, size=3).tolist(), "i_com": (r @ np.diag(p) @ r.T).tolist()}


def random_frame(rng, scale=0.4):
    return {"pos": rng.uniform(-scale, scale, size=3).tolist(), "rot": random_quat(rng).tolist()}


def link(parent, jtype, rng, axis=None, frame=None):
    joint = {"type": jtype}
    if jtype in ("revolute", "prismatic"):
        joint["axis"] = unit(axis if axis is not None else rng.normal(size=3)).tolist()
    return {
        "parent": parent,
        "joint": joint,
        "frame": frame if frame is not None else random_frame(rng),
        "inertia": random_inertia(rng),
    }


def build_pin(links):
    """Pinocchio model plus, per link, (joint id, placement of the link frame in that joint)."""
    model = pin.Model()
    model.gravity = pin.Motion(np.array([0.0, 0.0, -G]), np.zeros(3))
    attach = []
    for i, l in enumerate(links):
        f = l["frame"]
        placement = pin.SE3(quat_to_rot(f["rot"]), np.array(f["pos"]))
        if l["parent"] is None:
            parent_jid, parent_m = 0, pin.SE3.Identity()
        else:
            parent_jid, parent_m = attach[l["parent"]]
        jt = l["joint"]["type"]
        inert = l["inertia"]
        inertia = pin.Inertia(inert["mass"], np.array(inert["com"]), np.array(inert["i_com"]))
        if jt == "fixed":
            m = parent_m * placement
            model.appendBodyToJoint(parent_jid, inertia, m)
            attach.append((parent_jid, m))
            continue
        jm = {
            "free": lambda: pin.JointModelFreeFlyer(),
            "spherical": lambda: pin.JointModelSpherical(),
            "revolute": lambda: pin.JointModelRevoluteUnaligned(np.array(l["joint"]["axis"])),
            "prismatic": lambda: pin.JointModelPrismaticUnaligned(np.array(l["joint"]["axis"])),
        }[jt]()
        jid = model.addJoint(parent_jid, jm, parent_m * placement, f"j{i}")
        model.appendBodyToJoint(jid, inertia, pin.SE3.Identity())
        attach.append((jid, pin.SE3.Identity()))
    return model, attach


def nq_nv(jt):
    return {"free": (7, 6), "spherical": (4, 3), "revolute": (1, 1), "prismatic": (1, 1), "fixed": (0, 0)}[jt]


def v_perm(links):
    """Index map ours -> Pinocchio for velocity-sized vectors (free joints swap ω and v)."""
    perm, off = [], 0
    for l in links:
        _, nv = nq_nv(l["joint"]["type"])
        if l["joint"]["type"] == "free":
            perm += [off + 3, off + 4, off + 5, off + 0, off + 1, off + 2]
        else:
            perm += list(range(off, off + nv))
        off += nv
    return np.array(perm, dtype=int)


def random_q(links, rng):
    q = []
    for l in links:
        jt = l["joint"]["type"]
        if jt == "free":
            q += rng.uniform(-2, 2, size=3).tolist() + random_quat(rng).tolist()
        elif jt == "spherical":
            q += random_quat(rng).tolist()
        elif jt == "revolute":
            q.append(rng.uniform(-np.pi, np.pi))
        elif jt == "prismatic":
            q.append(rng.uniform(-0.5, 0.5))
    return np.array(q)


def mass_matrix(model, q):
    """M(q) column by column from RNEA (pin.crba assumes depth-first joint order, which random
    topological orders violate)."""
    model = model.copy()
    model.gravity = pin.Motion.Zero()
    data = model.createData()
    zero = np.zeros(model.nv)
    return np.column_stack([pin.rnea(model, data, q, zero, e) for e in np.eye(model.nv)])


def is_depth_first(links):
    """True if every subtree occupies a contiguous index range."""
    stack = []
    for i, l in enumerate(links):
        while stack and stack[-1] != l["parent"]:
            stack.pop()
        if l["parent"] is not None and not stack:
            return False
        stack.append(i)
    return True


def sample(links, model, attach, rng):
    data = model.createData()
    nv = model.nv
    perm = v_perm(links)  # ours[k] = pin[perm[k]]
    q = random_q(links, rng)
    v, tau, qdd = rng.normal(size=nv), rng.normal(size=nv) * 2.0, rng.normal(size=nv)
    f_ext = rng.normal(size=(len(links), 6))  # ours: [n, f] per link

    fext = pin.StdVec_Force()
    for _ in range(model.njoints):
        fext.append(pin.Force.Zero())
    for i, (jid, m) in enumerate(attach):
        f_link = pin.Force(f_ext[i, 3:], f_ext[i, :3])  # (linear, angular)
        fext[jid] = fext[jid] + m.act(f_link)

    def to_pin(x):
        y = np.empty(nv)
        y[perm] = x
        return y

    def from_pin(y):
        return y[perm]

    v_p, tau_p, qdd_p = to_pin(v), to_pin(tau), to_pin(qdd)
    aba_qdd = from_pin(pin.aba(model, data, q, v_p, tau_p, fext).copy())
    aba_qdd_nofext = from_pin(pin.aba(model, data, q, v_p, tau_p).copy())
    rnea_tau = from_pin(pin.rnea(model, data, q, v_p, qdd_p, fext).copy())
    m = mass_matrix(model, q)
    if is_depth_first(links):
        m_crba = pin.crba(model, data, q).copy()
        assert np.allclose(np.triu(m_crba) + np.triu(m_crba, 1).T, m, atol=1e-12)
    m = m[np.ix_(perm, perm)]
    ke = pin.computeKineticEnergy(model, data, q, v_p)
    com = pin.centerOfMass(model, data, q).copy()
    return {
        "q": q.tolist(),
        "v": v.tolist(),
        "tau": tau.tolist(),
        "qdd": qdd.tolist(),
        "f_ext": f_ext.tolist(),
        "aba_qdd": aba_qdd.tolist(),
        "aba_qdd_no_fext": aba_qdd_nofext.tolist(),
        "rnea_tau": rnea_tau.tolist(),
        "mass_matrix": m.tolist(),
        "kinetic_energy": float(ke),
        "com": com.tolist(),
    }


def random_tree(rng, n, root="free"):
    kinds = ["revolute", "revolute", "prismatic", "spherical", "fixed"]
    links = [link(None, root, rng)]
    for i in range(1, n):
        links.append(link(int(rng.integers(0, i)), kinds[int(rng.integers(0, len(kinds)))], rng))
    return links


def models(rng):
    yield "free_body", [link(None, "free", rng)]
    arm = [link(None, "revolute", rng, axis=[0, 0, 1], frame={"pos": [0, 0, 0.1], "rot": [0, 0, 0, 1]})]
    for i in range(1, 7):
        arm.append(link(i - 1, "revolute", rng))
    yield "arm7", arm
    # Floating chassis with four suspended, steered wheels (car-like topology) and a fixed payload.
    car = [link(None, "free", rng)]
    for corner in range(4):
        x, y = (1.3 if corner < 2 else -1.3), (0.8 if corner % 2 == 0 else -0.8)
        car.append(link(0, "prismatic", rng, axis=[0, 0, 1], frame={"pos": [x, y, -0.2], "rot": [0, 0, 0, 1]}))
        car.append(link(len(car) - 1, "revolute", rng, axis=[0, 0, 1], frame={"pos": [0, 0, 0], "rot": [0, 0, 0, 1]}))
        car.append(link(len(car) - 1, "revolute", rng, axis=[0, 1, 0], frame={"pos": [0, 0.1, 0], "rot": [0, 0, 0, 1]}))
    car.append(link(0, "fixed", rng))
    yield "car_like", car
    yield "spherical_chain", [link(None, "spherical", rng)] + [link(i, "spherical", rng) for i in range(3)]
    for k in range(3):
        yield f"random_tree_{k}", random_tree(rng, 8 + 4 * k, root="free" if k != 1 else "revolute")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260923)
    for name, links in models(rng):
        model, attach = build_pin(links)
        doc = {
            "generator": f"tools/gen_pinocchio_fixtures.py (pinocchio {pin.__version__})",
            "gravity": [0.0, 0.0, -G],
            "links": links,
            "samples": [sample(links, model, attach, rng) for _ in range(4)],
        }
        (OUT / f"{name}.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{name}: nq={model.nq} nv={model.nv} links={len(links)}")


if __name__ == "__main__":
    main()
