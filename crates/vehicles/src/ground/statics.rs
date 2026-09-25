//! Static equilibrium on flat, level ground for any wheel layout and chain of units, by
//! minimising the potential energy: gravity, tyre deflection, suspension springs and stops,
//! anti-roll bars and the couplings' roll springs and stops.
//!
//! Unknowns: the towing unit's height, pitch and roll, each further unit's pitch and roll on
//! its joint (couplings; a hinge only pitches, a turntable stays straight) and the suspension
//! travels. The gradient is the virtual work of the forces over finite-difference motions of
//! the tree (forward kinematics), the Hessian that of the tyre, spring and coupling
//! stiffnesses over the same motions (Gauss–Newton), so the minimum is exact while the
//! Newton steps are approximate. Tyres are vertical springs under the wheel centres.

use super::def::{StaticState, WheeledDef, deflection_at};
use super::tree::build;
use super::units::UnitJoint;
use autonomousim_core::dynamics::{KinCache, forward_kinematics};
use autonomousim_core::math::DenseMatrix;
use glam::{DQuat, DVec3};

#[derive(Clone, Copy, PartialEq)]
enum Param {
    Height,
    Pitch,
    Roll,
    UnitPitch(usize),
    UnitRoll(usize),
    Travel(usize),
}

/// Solve with gravity `g`; with `auto = Some(u)`, the springs without a given preload on units
/// `u` and later stay at zero travel and their preloads are read from the equilibrium (the
/// others keep the definition's). Returns the state and every wheel's preload.
pub(super) fn solve(def: &WheeledDef, g: f64, auto: Option<usize>) -> Result<(StaticState, Vec<f64>), String> {
    let tree = build(def, &def.spin_inertia());
    let model = &tree.model;
    let n = def.num_wheels();
    let mut kin = KinCache::new(model);
    let v = vec![0.0; model.nv()];
    let automatic: Vec<bool> = def
        .wheels()
        .map(|(a, _)| {
            auto.is_some_and(|u| a.unit >= u)
                && a.suspension.as_ref().is_some_and(|s| s.spring.travel.is_empty() && s.spring.preload.is_none())
        })
        .collect();
    let mut params = vec![Param::Height, Param::Pitch, Param::Roll];
    for (k, u) in def.units.iter().enumerate() {
        match u.joint {
            UnitJoint::Coupling(_) => params.extend([Param::UnitPitch(k + 1), Param::UnitRoll(k + 1)]),
            UnitJoint::Hinge => params.push(Param::UnitPitch(k + 1)),
            UnitJoint::Turntable => {}
        }
    }
    params.extend((0..n).filter(|&w| tree.corners[w].travel.is_some()).map(Param::Travel));
    let np = params.len();
    let free: Vec<usize> = (0..np).filter(|&i| !matches!(params[i], Param::Travel(w) if automatic[w])).collect();
    let preload: Vec<f64> = (0..n).map(|w| if automatic[w] { 0.0 } else { def.preload(w) }).collect();

    let configure = |p: &[f64]| -> Vec<f64> {
        let mut q = model.neutral_state().q.to_vec();
        let (mut h, mut pitch, mut roll) = (0.0, 0.0, 0.0);
        let mut unit = vec![[0.0; 2]; def.units.len() + 1];
        for (&x, par) in p.iter().zip(&params) {
            match *par {
                Param::Height => h = x,
                Param::Pitch => pitch = x,
                Param::Roll => roll = x,
                Param::UnitPitch(u) => unit[u][0] = x,
                Param::UnitRoll(u) => unit[u][1] = x,
                Param::Travel(w) => q[tree.corners[w].travel.expect("sprung").0] = x,
            }
        }
        q[2] = h;
        q[3..7].copy_from_slice(&(DQuat::from_rotation_y(pitch) * DQuat::from_rotation_x(roll)).to_array());
        for (k, u) in def.units.iter().enumerate() {
            let [up, ur] = unit[k + 1];
            let off = tree.units[k + 1].q;
            match u.joint {
                UnitJoint::Coupling(_) => {
                    let rot = DQuat::from_rotation_y(up) * DQuat::from_rotation_x(ur);
                    q[off..off + 4].copy_from_slice(&rot.to_array());
                }
                UnitJoint::Hinge => q[off] = up,
                UnitJoint::Turntable => {}
            }
        }
        q
    };
    let masses: Vec<usize> = (0..model.num_links()).filter(|&l| model.link(l).inertia.mass > 0.0).collect();
    let couplings: Vec<usize> =
        (0..def.units.len()).filter(|&k| matches!(def.units[k].joint, UnitJoint::Coupling(_))).collect();
    // Heights of the masses' centres and of the wheel centres, and the coupling rotations.
    let mut measure = |q: &[f64]| -> (Vec<f64>, Vec<f64>, Vec<DQuat>) {
        forward_kinematics(model, q, &v, &mut kin);
        let com = masses.iter().map(|&l| kin.pose[l].transform_point(model.link(l).inertia.com).z).collect();
        let wheels = tree.corners.iter().map(|c| kin.pose[c.wheel].pos.z).collect();
        let rots = couplings
            .iter()
            .map(|&k| {
                let off = tree.units[k + 1].q;
                DQuat::from_slice(&q[off..off + 4])
            })
            .collect();
        (com, wheels, rots)
    };
    let tire_force = |w: usize, d: f64| -> (f64, f64) {
        let t = def.tire(w / 2);
        let nominal = t.nominal_load();
        let soft = 1e-3 * nominal / deflection_at(t, nominal);
        if d <= 0.0 {
            return (soft * d, soft);
        }
        let h = 1e-6;
        let k = (t.vertical_force(d + h) - t.vertical_force((d - h).max(0.0))) / (d + h - (d - h).max(0.0));
        (t.vertical_force(d), k.max(soft))
    };

    // Start with the towing unit's tyres at their nominal deflection, everything level.
    let mut p = vec![0.0; np];
    let front: Vec<usize> = (0..n).filter(|&w| def.axles[w / 2].unit == 0).collect();
    p[0] = front
        .iter()
        .map(|&w| {
            let t = def.tire(w / 2);
            t.radius() - deflection_at(t, t.nominal_load()) - def.wheel_position(w).z
        })
        .sum::<f64>()
        / front.len() as f64;
    let h = 1e-6;
    let mut last = f64::INFINITY;
    let mut grad = vec![0.0; np];
    for _ in 0..100 {
        let q = configure(&p);
        let (_, wz, rots) = measure(&q);
        // Jacobians of the measures by central differences.
        let mut j_com = vec![vec![0.0; np]; masses.len()];
        let mut j_wheel = vec![vec![0.0; np]; n];
        let mut j_rot = vec![vec![DVec3::ZERO; np]; couplings.len()];
        for i in 0..np {
            let (mut pp, mut pm) = (p.clone(), p.clone());
            pp[i] += h;
            pm[i] -= h;
            let (cp, wp, rp) = measure(&configure(&pp));
            let (cm, wm, rm) = measure(&configure(&pm));
            for k in 0..masses.len() {
                j_com[k][i] = (cp[k] - cm[k]) / (2.0 * h);
            }
            for w in 0..n {
                j_wheel[w][i] = (wp[w] - wm[w]) / (2.0 * h);
            }
            for k in 0..couplings.len() {
                j_rot[k][i] = (rm[k].inverse() * rp[k]).to_scaled_axis() / (2.0 * h);
            }
        }
        grad.fill(0.0);
        let mut hess = DenseMatrix::zeros(np, np);
        let add = |grad: &mut [f64], hess: &mut DenseMatrix, force: f64, stiffness: f64, row: &[f64]| {
            for i in 0..np {
                grad[i] += force * row[i];
                for j in 0..np {
                    hess[(i, j)] += stiffness * row[i] * row[j];
                }
            }
        };
        for (k, &l) in masses.iter().enumerate() {
            add(&mut grad, &mut hess, model.link(l).inertia.mass * g, 0.0, &j_com[k]);
        }
        let mut loads = vec![0.0; n];
        let mut deflection = vec![0.0; n];
        for w in 0..n {
            deflection[w] = def.tire(w / 2).radius() - wz[w];
            let (f, k) = tire_force(w, deflection[w]);
            loads[w] = f;
            // The tyre pushes the wheel centre up: energy falls as the centre rises.
            add(&mut grad, &mut hess, -f, k, &j_wheel[w]);
        }
        for (k, &u) in couplings.iter().enumerate() {
            let UnitJoint::Coupling(c) = def.units[u].joint else { unreachable!() };
            let tau = c.torque(rots[k], DVec3::ZERO);
            let stiff = c.stiffness(rots[k]);
            for (axis, (t, s)) in [(tau.x, stiff.x), (tau.y, stiff.y), (tau.z, stiff.z)].into_iter().enumerate() {
                let row: Vec<f64> = j_rot[k].iter().map(|d| d[axis]).collect();
                add(&mut grad, &mut hess, -t, s, &row);
            }
        }
        // Springs, stops and anti-roll bars act on the travels directly.
        let travel_param = |w: usize| params.iter().position(|&x| x == Param::Travel(w));
        for w in 0..n {
            let (Some(i), Some(s)) = (travel_param(w), &def.axles[w / 2].suspension) else { continue };
            let x = p[i];
            let f = s.spring_force(x, preload[w]);
            let k = (s.spring_force(x + h, preload[w]) - s.spring_force(x - h, preload[w])) / (2.0 * h);
            grad[i] += f;
            hess[(i, i)] += k;
            if w % 2 == 0
                && s.anti_roll != 0.0
                && let Some(j) = travel_param(w + 1)
            {
                let f = s.anti_roll * (x - p[j]);
                grad[i] += f;
                grad[j] -= f;
                hess[(i, i)] += s.anti_roll;
                hess[(j, j)] += s.anti_roll;
                hess[(i, j)] -= s.anti_roll;
                hess[(j, i)] -= s.anti_roll;
            }
        }
        if last < 1e-9 {
            // Converged: the automatic preloads balance the remaining force on their travel.
            let mut out = preload.clone();
            for w in (0..n).filter(|&w| automatic[w]) {
                out[w] = -grad[travel_param(w).expect("sprung")];
            }
            if let Some(w) = (0..n).find(|&w| deflection[w] <= 0.0) {
                return Err(format!("wheel {w} does not touch the ground at rest"));
            }
            let mut joints = vec![[0.0; 2]; def.units.len()];
            for (i, par) in params.iter().enumerate() {
                match *par {
                    Param::UnitPitch(u) => joints[u - 1][0] = p[i],
                    Param::UnitRoll(u) => joints[u - 1][1] = p[i],
                    _ => {}
                }
            }
            let travel = (0..n).map(|w| travel_param(w).map_or(0.0, |i| p[i])).collect();
            let state = StaticState { height: p[0], pitch: p[1], roll: p[2], loads, travel, deflection, joints };
            return Ok((state, out));
        }
        // Newton step on the free unknowns, at most 5 cm or 0.05 rad.
        let m = free.len();
        let mut a = DenseMatrix::zeros(m, m);
        let scale = free.iter().map(|&i| hess[(i, i)]).fold(0.0, f64::max);
        for (r, &i) in free.iter().enumerate() {
            for (c, &j) in free.iter().enumerate() {
                a[(r, c)] = hess[(i, j)];
            }
            a[(r, r)] += 1e-12 * scale;
        }
        let b: Vec<f64> = free.iter().map(|&i| -grad[i]).collect();
        let step = a.solve_spd(&b).map_err(|_| "no static equilibrium: the layout is not supported".to_string())?;
        let size = step.iter().fold(0.0f64, |s, x| s.max(x.abs()));
        let limit = if size > 0.05 { 0.05 / size } else { 1.0 };
        for (r, &i) in free.iter().enumerate() {
            p[i] += limit * step[r];
        }
        last = size;
    }
    Err(format!("static equilibrium did not converge (last step {last:.1e})"))
}
