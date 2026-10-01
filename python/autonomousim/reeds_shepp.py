"""Reeds–Shepp paths: the shortest paths of a car that drives forward and backward with a
bounded turning radius (Reeds & Shepp 1990), in the 48 words of the five families CSC, CCC,
CCCC, CCSC and CCSCC (formulas after OMPL's ``ReedsSheppStateSpace``).

A path is a list of segments ``(kind, length)``: ``kind`` is ``"L"``, ``"R"`` (turning left
or right at the radius) or ``"S"`` (straight), ``length`` is the signed length in units of
the radius (an angle for turns), positive forward. ``paths`` gives every word that connects
the poses, ``sample`` the poses along one.
"""

import math

import numpy as np

HALF_PI = 0.5 * math.pi
_EPS = 1e-9


def _mod2pi(x: float) -> float:
    """Angle in [−π, π)."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _polar(x: float, y: float) -> tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def _tau_omega(u: float, v: float, xi: float, eta: float, phi: float) -> tuple[float, float]:
    delta = _mod2pi(u - v)
    a = math.sin(u) - math.sin(delta)
    b = math.cos(u) - math.cos(delta) - 1.0
    t1 = math.atan2(eta * a - xi * b, xi * a + eta * b)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0
    tau = _mod2pi(t1 + math.pi) if t2 < 0 else _mod2pi(t1)
    return tau, _mod2pi(tau - u + v - phi)


def _lpsplp(x, y, phi):  # (8.1)
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= -_EPS:
        v = _mod2pi(phi - t)
        if v >= -_EPS:
            return t, u, v
    return None


def _lpsprp(x, y, phi):  # (8.2)
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 *= u1
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        t = _mod2pi(t1 + math.atan2(2.0, u))
        v = _mod2pi(t - phi)
        if t >= -_EPS and v >= -_EPS:
            return t, u, v
    return None


def _lprml(x, y, phi):  # (8.3)
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = _mod2pi(theta + 0.5 * u + math.pi)
        v = _mod2pi(phi - t + u)
        if t >= -_EPS and u <= _EPS:
            return t, u, v
    return None


def _lprupl_umrm(x, y, phi):  # (8.7)
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.hypot(xi, eta))
    if rho <= 1.0:
        u = math.acos(rho)
        t, v = _tau_omega(u, -u, xi, eta, phi)
        if t >= -_EPS and v <= _EPS:
            return t, u, v
    return None


def _lprumlump(x, y, phi):  # (8.8)
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -HALF_PI:
            t, v = _tau_omega(u, u, xi, eta, phi)
            if t >= -_EPS and v >= -_EPS:
                return t, u, v
    return None


def _lprmsmlm(x, y, phi):  # (8.9)
    rho, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = _mod2pi(theta + math.atan2(r, -2.0))
        v = _mod2pi(phi - HALF_PI - t)
        if t >= -_EPS and u <= _EPS and v <= _EPS:
            return t, u, v
    return None


def _lprmsmrm(x, y, phi):  # (8.10)
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho, theta = _polar(-eta, xi)
    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = _mod2pi(t + HALF_PI - phi)
        if t >= -_EPS and u <= _EPS and v <= _EPS:
            return t, u, v
    return None


def _lprmslmrp(x, y, phi):  # (8.11)
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho, _ = _polar(xi, eta)
    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= _EPS:
            t = _mod2pi(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = _mod2pi(t - phi)
            if t >= -_EPS and v >= -_EPS:
                return t, u, v
    return None


def _swap(word: str) -> str:
    return word.translate(str.maketrans("LR", "RL"))


def _words(x: float, y: float, phi: float) -> list[tuple[str, list[float]]]:
    """Every word to the pose ``(x, y, phi)`` (units of the radius) from the origin."""
    out: list[tuple[str, list[float]]] = []

    # The four symmetric variants (OMPL): (x, y, φ), (−x, y, −φ), (x, −y, −φ), (−x, −y, φ).
    def variants(f, word, lengths, xs, ys, ps):
        for sx, sy, sp, flip, reflect in (
            (1, 1, 1, 1.0, False),
            (-1, 1, -1, -1.0, False),
            (1, -1, -1, 1.0, True),
            (-1, -1, 1, -1.0, True),
        ):
            r = f(sx * xs, sy * ys, sp * ps)
            if r is not None:
                out.append((_swap(word) if reflect else word, [flip * s for s in lengths(*r)]))

    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
    # CSC
    variants(_lpsplp, "LSL", lambda t, u, v: (t, u, v), x, y, phi)
    variants(_lpsprp, "LSR", lambda t, u, v: (t, u, v), x, y, phi)
    # CCC, forward and backward
    variants(_lprml, "LRL", lambda t, u, v: (t, u, v), x, y, phi)
    variants(_lprml, "LRL", lambda t, u, v: (v, u, t), xb, yb, phi)
    # CCCC
    variants(_lprupl_umrm, "LRLR", lambda t, u, v: (t, u, -u, v), x, y, phi)
    variants(_lprumlump, "LRLR", lambda t, u, v: (t, u, u, v), x, y, phi)
    # CCSC, forward and backward
    variants(_lprmsmlm, "LRSL", lambda t, u, v: (t, -HALF_PI, u, v), x, y, phi)
    variants(_lprmsmrm, "LRSR", lambda t, u, v: (t, -HALF_PI, u, v), x, y, phi)
    variants(_lprmsmlm, "LSRL", lambda t, u, v: (v, u, -HALF_PI, t), xb, yb, phi)
    variants(_lprmsmrm, "RSRL", lambda t, u, v: (v, u, -HALF_PI, t), xb, yb, phi)
    # CCSCC
    variants(_lprmslmrp, "LRSLR", lambda t, u, v: (t, -HALF_PI, u, -HALF_PI, v), x, y, phi)
    return out


Path = list[tuple[str, float]]


def end_pose(start: tuple[float, float, float], path: Path, radius: float) -> tuple[float, float, float]:
    """The pose reached from ``start`` (x, y, heading) along ``path`` at ``radius``."""
    x, y, th = start
    for kind, s in path:
        if kind == "S":
            x, y = x + s * radius * math.cos(th), y + s * radius * math.sin(th)
        else:
            k = 1.0 if kind == "L" else -1.0
            dth = k * s
            x += radius * k * (math.sin(th + dth) - math.sin(th))
            y += radius * k * (math.cos(th) - math.cos(th + dth))
            th += dth
    return x, y, th


def paths(
    start: tuple[float, float, float], goal: tuple[float, float, float], radius: float, tol: float = 1e-6
) -> list[Path]:
    """Every Reeds–Shepp path from ``start`` to ``goal`` (x, y, heading) at the turning
    ``radius``, shortest first, without zero-length segments. Each is checked to reach the
    goal (within ``tol`` of the radius)."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    x, y = (c * dx + s * dy) / radius, (-s * dx + c * dy) / radius
    phi = _mod2pi(goal[2] - start[2])
    out = []
    for word, lengths in _words(x, y, phi):
        p = [(k, ln) for k, ln in zip(word, lengths, strict=True) if abs(ln) > 1e-10]
        ex, ey, eth = end_pose(start, p, radius)
        miss = math.hypot(ex - goal[0], ey - goal[1]) / radius + abs(_mod2pi(eth - goal[2]))
        if miss < tol:
            out.append(p)
    out.sort(key=length)
    return out


def length(path: Path) -> float:
    """Length in units of the radius."""
    return sum(abs(s) for _, s in path)


def cusps(path: Path) -> int:
    """Number of changes of direction."""
    signs = [math.copysign(1.0, s) for _, s in path]
    return sum(a != b for a, b in zip(signs, signs[1:], strict=False))


def sample(start: tuple[float, float, float], path: Path, radius: float, step: float = 0.1) -> np.ndarray:
    """Poses along ``path`` from ``start`` every ``step`` m (and at each segment's end):
    rows of x, y, heading, direction (±1) and curvature (1/m, positive turning left)."""
    rows = [(*start, math.copysign(1.0, path[0][1]) if path else 1.0, 0.0)]
    pose = start
    for kind, s in path:
        d = math.copysign(1.0, s)
        k = 0.0 if kind == "S" else (1.0 if kind == "L" else -1.0) / radius
        n = max(1, math.ceil(abs(s) * radius / step))
        for i in range(1, n + 1):
            x, y, th = end_pose(pose, [(kind, s * i / n)], radius)
            rows.append((x, y, th, d, k))
        pose = end_pose(pose, [(kind, s)], radius)
    return np.array(rows)
