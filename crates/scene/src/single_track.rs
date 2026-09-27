//! Visual of a single-track vehicle and its rider: a motorcycle (frame spars, engine, tank,
//! seat and tail, fairing with a headlight, exhaust) or a bicycle (tube frame, saddle, rack),
//! a fork and handlebar turned about the steering axis, toroidal tyres on spoked rims, and a
//! rider whose upper body leans about the hip, with arms reaching for the grips and legs
//! reaching for the pegs, pedals or the ground.
//!
//! The steered parts, the upper body and the limbs are posed by the renderer from the
//! simulated steering angle, lean and feet ([`SingleTrackVisual`]); [`bend`] places elbows
//! and knees.

use crate::mesh::{self, MeshData, srgb};
use crate::props::WheeledVisual;
use autonomousim_vehicles::ground::{PowertrainDef, SteeringHeadDef, WheeledDef};
use glam::{DQuat, DVec3, Vec3};

/// Posed parts of a single-track vehicle, in the chassis frame (FLU).
#[derive(Clone, Debug)]
pub struct SingleTrackVisual {
    /// Fork tubes, triple clamps and handlebar about the steering axis' pivot `pivot`
    /// (straight ahead): turned about `axis` by the steering angle.
    pub steered: MeshData,
    pub pivot: DVec3,
    pub axis: DVec3,
    /// The front wheel; its fork sliders stand `fork_offset` to each side of its centre
    /// (square to the axis, turning with the steering) and reach `fork_leg` up the axis, so
    /// that they telescope over the tubes as the fork travels.
    pub front_wheel: usize,
    pub fork_offset: f64,
    pub fork_leg: f64,
    /// The rider's upper body about the hip, upright: leaned about x by the rider's lean.
    pub torso: MeshData,
    pub hip: DVec3,
    /// Left shoulder relative to the hip (upright) and left grip relative to the pivot
    /// (straight ahead); the right ones mirror them. Upper arm and forearm lengths (m).
    pub shoulder: DVec3,
    pub grip: DVec3,
    pub arm: [f64; 2],
    /// Left leg: hip joint, foot on the pegs or pedals and foot down (the feet colliders'
    /// centres), thigh and shin lengths (m).
    pub leg_hip: DVec3,
    pub foot_up: DVec3,
    pub foot_down: DVec3,
    pub leg: [f64; 2],
    /// Limbs as unit cylinders along z (from −0.5 to 0.5), in this order: fork slider,
    /// upper arm, forearm, thigh, shin; and a boot about the foot's centre.
    pub limbs: [MeshData; 5],
    pub boot: MeshData,
}

/// Limbs of [`SingleTrackVisual::limbs`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limb {
    ForkSlider = 0,
    UpperArm = 1,
    Forearm = 2,
    Thigh = 3,
    Shin = 4,
}

impl SingleTrackVisual {
    /// Rotation of the steered parts at steering angle `steer` (rad, positive left).
    pub fn steer_rotation(&self, steer: f64) -> DQuat {
        DQuat::from_axis_angle(self.axis, steer)
    }

    /// Rotation of the upper body at lean `lean` (rad, positive right).
    pub fn lean_rotation(lean: f64) -> DQuat {
        DQuat::from_rotation_x(lean)
    }

    /// Ends of the fork slider on side `side` (+1 left, −1 right) below the front wheel's
    /// centre `wheel` (chassis frame) at steering angle `steer`: bottom, top.
    pub fn fork_slider(&self, side: f64, wheel: DVec3, steer: f64) -> [DVec3; 2] {
        let bottom = wheel + self.steer_rotation(steer) * DVec3::new(0.0, side * self.fork_offset, 0.0);
        [bottom, bottom + self.axis * self.fork_leg]
    }

    /// Shoulder, elbow and hand on side `side` at the given steering angle and lean.
    pub fn arm_joints(&self, side: f64, steer: f64, lean: f64) -> [DVec3; 3] {
        let mirror = DVec3::new(1.0, side, 1.0);
        let shoulder = self.hip + Self::lean_rotation(lean) * (self.shoulder * mirror);
        let hand = self.pivot + self.steer_rotation(steer) * (self.grip * mirror);
        // Elbows down and out.
        let elbow = bend(shoulder, hand, self.arm[0], self.arm[1], DVec3::new(0.0, 0.6 * side, -1.0));
        [shoulder, elbow, hand]
    }

    /// Hip, knee and foot on side `side`, the feet down or on the pegs.
    pub fn leg_joints(&self, side: f64, feet_down: bool) -> [DVec3; 3] {
        let mirror = DVec3::new(1.0, side, 1.0);
        let hip = self.leg_hip * mirror;
        let foot = if feet_down { self.foot_down } else { self.foot_up } * mirror;
        // Knees forward and a little out.
        let knee = bend(hip, foot, self.leg[0], self.leg[1], DVec3::new(1.0, 0.35 * side, 0.0));
        [hip, knee, foot]
    }
}

/// The middle joint of a two-segment limb from `a` to `b` with segment lengths `l1` and
/// `l2`, bent towards `hint`; on the line between them (in proportion) when out of reach.
pub fn bend(a: DVec3, b: DVec3, l1: f64, l2: f64, hint: DVec3) -> DVec3 {
    let d = b - a;
    let len = d.length();
    if len < 1e-9 {
        return a + hint.normalize_or(DVec3::Z) * l1;
    }
    let u = d / len;
    if len >= l1 + l2 || len <= (l1 - l2).abs() {
        return a + d * (l1 / (l1 + l2));
    }
    let along = (l1 * l1 - l2 * l2 + len * len) / (2.0 * len);
    let out = (l1 * l1 - along * along).max(0.0).sqrt();
    let perp = (hint - u * hint.dot(u)).normalize_or(u.any_orthonormal_vector());
    a + u * along + perp * out
}

/// Bar of radius `r` from `a` to `b`.
fn bar(m: &mut MeshData, a: DVec3, b: DVec3, r: f64, color: [f32; 4]) {
    let d = b - a;
    let rot = DQuat::from_rotation_arc(DVec3::Z, d.normalize_or(DVec3::Z));
    m.append_transformed(&mesh::cylinder(r as f32, (0.5 * d.length()) as f32, 10, color), rot, 0.5 * (a + b));
}

/// Box of half extents `h` whose z axis runs along `dir`, centred at `at`.
fn slab(m: &mut MeshData, at: DVec3, dir: DVec3, h: DVec3, color: [f32; 4]) {
    let rot = DQuat::from_rotation_arc(DVec3::Z, dir.normalize_or(DVec3::Z));
    m.append_transformed(&mesh::cuboid(h.as_vec3(), color), rot, at);
}

fn blob(m: &mut MeshData, at: DVec3, h: DVec3, color: [f32; 4]) {
    m.append_transformed(&mesh::ellipsoid(h.as_vec3(), 2, color), DQuat::IDENTITY, at);
}

/// Unit limb (a cylinder along z from −0.5 to 0.5) of radius `r`.
fn limb(r: f64, color: [f32; 4]) -> MeshData {
    mesh::cylinder(r as f32, 0.5, 10, color)
}

/// Build the visual of a single-track vehicle with steering head `head` on wheel `front`.
pub fn visual(def: &WheeledDef, head: &SteeringHeadDef, front: usize) -> WheeledVisual {
    let n = def.num_wheels();
    let rear = (0..n).find(|&w| w != front).unwrap_or(front);
    let (pf, pr) = (def.wheel_position(front), def.wheel_position(rear));
    let (rf, rr) = (def.wheel_tire(front).radius(), def.wheel_tire(rear).radius());
    let section = |w: usize| def.wheel_tire(w).section_width();
    let motorcycle = matches!(def.powertrain, PowertrainDef::Combustion(_));
    let wheelbase = (pf.x - pr.x).abs().max(0.5);

    let axis = head.axis();
    let pivot = head.pivot(pf);
    let lambda = head.angle;
    // Square to the axis, forward: the wheel centre lies `offset` along it from the pivot.
    let normal = DVec3::new(lambda.cos(), 0.0, lambda.sin());

    // The rider: hip on the lean axis, head from the highest collider on the centre plane
    // above the hip (else above the upper body's centre of mass).
    let (hip, rider_com) =
        def.rider.as_ref().map_or((pr + DVec3::new(0.35 * wheelbase, 0.0, 2.2 * rr), None), |r| (r.hip, Some(r.com)));
    let head_collider = def
        .colliders
        .iter()
        .filter(|c| c.center.y.abs() < 0.05 && c.center.z > hip.z + 0.3)
        .max_by(|a, b| a.center.z.total_cmp(&b.center.z));
    let (head_at, head_r) = match (head_collider, rider_com) {
        (Some(c), _) => (c.center, 0.9 * c.radius),
        (None, Some(com)) => (hip + 2.4 * (com - hip), 0.11),
        (None, None) => (hip + DVec3::new(0.15, 0.0, 0.6), 0.11),
    };
    let shoulders = hip + 0.8 * (head_at - hip);
    // Grips: the handlebar-end colliders, else a bar above the tyre.
    let grip = def
        .colliders
        .iter()
        .filter(|c| c.center.y > 0.15 && c.center.x > hip.x + 0.2 && c.radius < 0.08)
        .map(|c| c.center)
        .next()
        .unwrap_or(pivot + axis * (rf + 0.35) + DVec3::new(0.0, 0.3, 0.0));

    // Along the axis from the pivot: the lower triple clamp above the tyre, the upper one at
    // the grips.
    let crown = rf + if motorcycle { 0.08 } else { 0.05 };
    let top = (grip - pivot).dot(axis).max(crown + 0.08);
    let at = |t: f64| pivot + axis * t;
    let fork_offset = 0.5 * section(front) + if motorcycle { 0.035 } else { 0.015 };
    let tube_r = if motorcycle { 0.022 } else { 0.012 };

    // Colours.
    let frame = if motorcycle { srgb([175, 175, 180]) } else { srgb([40, 110, 80]) };
    let paint = srgb([30, 80, 170]);
    let accent = srgb([200, 40, 36]);
    let dark = srgb([35, 35, 38]);
    let metal = srgb([190, 190, 195]);
    let (jacket, trousers) =
        if motorcycle { (srgb([40, 40, 45]), srgb([30, 30, 34])) } else { (srgb([205, 120, 40]), srgb([40, 50, 90])) };
    let skin = srgb([225, 185, 150]);
    let helmet = srgb([235, 235, 235]);

    // ---- steered parts, about the pivot
    let mut steered = MeshData::new();
    let rel = |p: DVec3| p - pivot;
    for side in [-1.0, 1.0] {
        let y = DVec3::new(0.0, side * fork_offset, 0.0);
        let (lo, hi) = (rel(pf) + y + axis * (0.45 * crown), rel(pf) + y + axis * top);
        bar(&mut steered, lo, hi, tube_r, metal);
    }
    let clamp = DVec3::new(0.035, fork_offset + tube_r + 0.015, 0.018);
    for t in [crown, top] {
        slab(&mut steered, rel(at(t)) + normal * (0.5 * head.offset), axis, clamp, dark);
    }
    let grip_l = rel(grip);
    let grip_r = grip_l * DVec3::new(1.0, -1.0, 1.0);
    let bar_mid = 0.5 * (grip_l + grip_r);
    if motorcycle {
        // Clip-ons from the fork tubes out to the grips.
        for g in [grip_l, grip_r] {
            let tube = rel(pf) + axis * (top - 0.04) + DVec3::new(0.0, g.y.signum() * fork_offset, 0.0);
            bar(&mut steered, tube, g, 0.013, dark);
        }
    } else {
        bar(&mut steered, rel(at(top)), bar_mid, 0.014, dark);
        bar(&mut steered, grip_l, grip_r, 0.012, dark);
        // A lamp on the crown.
        let lamp = rel(at(crown + 0.02)) + normal * (head.offset + 0.06);
        blob(&mut steered, lamp, DVec3::new(0.03, 0.035, 0.03), srgb([250, 245, 200]));
    }
    for g in [grip_l, grip_r] {
        let dir = DVec3::new(0.0, g.y.signum(), 0.0);
        bar(&mut steered, g - dir * 0.02, g + dir * 0.06, 0.018, dark);
    }

    // ---- body (chassis frame)
    let mut body = MeshData::new();
    let (tube_lo, tube_hi) = (at(crown + 0.03), at(top - 0.03));
    bar(&mut body, tube_lo, tube_hi, if motorcycle { 0.04 } else { 0.025 }, frame);
    if motorcycle {
        // Swing arm pivot: ahead of the rear wheel along its trailing arm.
        let arm = def.axles[def.wheel_axle(rear)].suspension.as_ref().and_then(|s| s.trailing_arm);
        let swing = arm
            .map_or(pr + DVec3::new(0.55, 0.0, 0.1), |a| pr + a.length * DVec3::new(a.angle.cos(), 0.0, a.angle.sin()));
        // Twin spars from the head tube down to the swing arm pivot.
        for side in [-1.0, 1.0] {
            let y = DVec3::new(0.0, side * 0.12, 0.0);
            let (a, b) = (tube_hi - axis * 0.05 + y, swing + DVec3::new(0.0, 0.0, 0.12) + y);
            slab(&mut body, 0.5 * (a + b), b - a, DVec3::new(0.018, 0.035, 0.5 * (b - a).length()), frame);
            bar(&mut body, swing + y - DVec3::new(0.0, 0.0, 0.02), swing + y + DVec3::new(0.0, 0.0, 0.14), 0.03, frame);
        }
        // Engine below the spars, its cylinders leaning forward.
        let scale = wheelbase / 1.4;
        let engine = DVec3::new(0.5 * (swing.x + tube_lo.x) - 0.02, 0.0, rr + 0.06 * scale);
        slab(&mut body, engine, DVec3::Z, DVec3::new(0.2, 0.15, 0.16) * scale, dark);
        let cylinders = engine + DVec3::new(0.12, 0.0, 0.2) * scale;
        slab(&mut body, cylinders, DVec3::new(0.5, 0.0, 1.0), DVec3::new(0.09, 0.14, 0.12) * scale, dark);
        // Lower fairing over the engine's sides.
        blob(&mut body, engine + DVec3::new(0.14, 0.0, 0.02) * scale, DVec3::new(0.22, 0.2, 0.15) * scale, paint);
        // Tank over the spars, seat and tail behind it.
        let tank = DVec3::new(0.5 * (hip.x + tube_hi.x) + 0.03, 0.0, hip.z + 0.02);
        blob(&mut body, tank, DVec3::new(0.3 * (tube_hi.x - hip.x).max(0.3), 0.16, 0.11), paint);
        let seat_front = hip.x + 0.1;
        let seat_back = hip.x - 0.3;
        let seat = DVec3::new(0.5 * (seat_front + seat_back), 0.0, hip.z - 0.04);
        slab(&mut body, seat, DVec3::Z, DVec3::new(0.5 * (seat_front - seat_back), 0.13, 0.035), dark);
        let tail_end = pr.x - 0.12;
        let tail = DVec3::new(0.5 * (seat_back + tail_end), 0.0, hip.z - 0.02);
        slab(&mut body, tail, DVec3::Z, DVec3::new(0.5 * (seat_back - tail_end), 0.1, 0.07), paint);
        slab(
            &mut body,
            tail - DVec3::new(0.5 * (seat_back - tail_end), 0.0, 0.0),
            DVec3::Z,
            DVec3::new(0.01, 0.06, 0.025),
            accent,
        );
        // Front fairing around the top of the head tube, a screen and a headlight.
        let nose = tube_hi + DVec3::new(0.12, 0.0, -0.1);
        blob(&mut body, nose, DVec3::new(0.2, 0.19, 0.17), paint);
        slab(
            &mut body,
            nose + DVec3::new(-0.04, 0.0, 0.17),
            DVec3::new(0.6, 0.0, 1.0),
            DVec3::new(0.005, 0.15, 0.09),
            srgb([60, 70, 85]),
        );
        blob(&mut body, nose + DVec3::new(0.18, 0.0, -0.02), DVec3::new(0.03, 0.09, 0.05), srgb([250, 245, 200]));
        // Exhaust along the right side to beside the rear wheel.
        let exhaust_end = pr + DVec3::new(0.1, -(0.5 * section(rear) + 0.1), 0.12);
        bar(
            &mut body,
            engine + DVec3::new(0.1, -0.1, -0.12) * scale,
            exhaust_end + DVec3::new(0.25, 0.0, -0.08),
            0.03,
            metal,
        );
        bar(&mut body, exhaust_end + DVec3::new(0.25, 0.0, -0.08), exhaust_end, 0.055, metal);
    } else {
        // Tube frame from the bottom bracket (at the pedals).
        let bb = def.feet.map_or(pr + DVec3::new(0.4 * wheelbase, 0.0, -0.07), |f| DVec3::new(f.up.x, 0.0, f.up.z));
        let seat_top = bb + 0.8 * (hip - DVec3::new(0.0, 0.0, 0.05) - bb);
        bar(&mut body, tube_hi - axis * 0.02, seat_top, 0.018, frame);
        bar(&mut body, tube_lo, bb, 0.022, frame);
        bar(&mut body, bb, seat_top, 0.018, frame);
        for side in [-1.0, 1.0] {
            let axle = pr + DVec3::new(0.0, side * (0.5 * section(rear) + 0.02), 0.0);
            bar(&mut body, bb, axle, 0.011, frame);
            bar(&mut body, seat_top, axle, 0.01, frame);
        }
        bar(&mut body, seat_top, hip - DVec3::new(0.0, 0.0, 0.05), 0.013, metal);
        slab(&mut body, hip - DVec3::new(0.0, 0.0, 0.03), DVec3::Z, DVec3::new(0.12, 0.07, 0.03), dark);
        // Crank and rack.
        cylinder_y(&mut body, bb, 0.09, 0.04, dark);
        slab(&mut body, pr + DVec3::new(0.02, 0.0, rr + 0.1), DVec3::Z, DVec3::new(0.2, 0.08, 0.008), dark);
        for side in [-1.0, 1.0] {
            let y = DVec3::new(0.0, side * 0.08, 0.0);
            bar(&mut body, pr + y + DVec3::new(0.2, 0.0, rr + 0.1), pr + y, 0.006, dark);
        }
    }
    // The rider's lower body at the hip.
    if def.rider.is_some() {
        blob(&mut body, hip + DVec3::new(-0.02, 0.0, 0.0), DVec3::new(0.13, 0.17, 0.1), trousers);
    }

    // ---- upper body, about the hip
    let mut torso = MeshData::new();
    let up = shoulders - hip;
    slab(&mut torso, 0.5 * up, up, DVec3::new(0.11, 0.17, 0.5 * up.length() + 0.03), jacket);
    let head_rel = head_at - hip;
    if motorcycle {
        blob(&mut torso, head_rel, DVec3::splat(head_r), helmet);
        let visor = head_rel + DVec3::new(0.8 * head_r, 0.0, 0.05 * head_r);
        blob(&mut torso, visor, DVec3::new(0.3 * head_r, 0.7 * head_r, 0.35 * head_r), srgb([40, 40, 50]));
        blob(
            &mut torso,
            head_rel + DVec3::new(0.0, 0.0, 0.9 * head_r),
            DVec3::new(0.5 * head_r, 0.2 * head_r, 0.2 * head_r),
            accent,
        );
    } else {
        blob(&mut torso, head_rel, DVec3::splat(head_r), skin);
        blob(
            &mut torso,
            head_rel + DVec3::new(-0.01, 0.0, 0.45 * head_r),
            DVec3::new(1.1 * head_r, 1.0 * head_r, 0.7 * head_r),
            helmet,
        );
    }

    // ---- wheels: toroidal tyres on spoked rims (about their centre, spin axis y)
    let spin_axis = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
    let wheels = (0..n)
        .map(|w| {
            let r = def.wheel_tire(w).radius();
            let s = section(w).min(0.8 * r);
            let mut m = MeshData::new();
            let minor = 0.5 * s;
            let tyre = mesh::torus((r - minor) as f32, minor as f32, 40, 10, dark);
            m.append_transformed(&tyre, spin_axis, DVec3::ZERO);
            let rim_r = r - 1.8 * minor;
            let rim = mesh::torus(rim_r as f32, (0.25 * minor).max(0.008) as f32, 40, 6, metal);
            m.append_transformed(&rim, spin_axis, DVec3::ZERO);
            m.append_transformed(
                &mesh::cylinder(0.12 * r as f32, 0.6 * minor as f32, 12, metal),
                spin_axis,
                DVec3::ZERO,
            );
            let (spokes, thick, color) =
                if motorcycle { (5, 0.05 * r, srgb([200, 170, 60])) } else { (16, 0.006, metal) };
            for k in 0..spokes {
                let a = std::f64::consts::TAU * k as f64 / spokes as f64;
                let dir = DVec3::new(a.cos(), 0.0, a.sin());
                slab(&mut m, dir * (0.5 * rim_r), dir, DVec3::new(thick, 0.3 * thick.max(0.01), 0.5 * rim_r), color);
            }
            if motorcycle {
                // Brake discs.
                let discs: &[f64] = if w == front { &[-1.0, 1.0] } else { &[-1.0] };
                for &side in discs {
                    let disc = mesh::cylinder(0.55 * r as f32, 0.004, 24, metal);
                    m.append_transformed(&disc, spin_axis, DVec3::new(0.0, side * 0.7 * minor, 0.0));
                }
            }
            m
        })
        .collect();

    // Swing arm and shock from the frame to the rear wheel's centre.
    let links = (0..n)
        .map(|w| {
            let arm = def.axles[def.wheel_axle(w)].suspension.as_ref()?.trailing_arm?;
            let p = def.wheel_position(w);
            let swing = p + arm.length * DVec3::new(arm.angle.cos(), 0.0, arm.angle.sin());
            let shock = DVec3::new(p.x + 0.45 * (swing.x - p.x), 0.0, hip.z - 0.1);
            Some([swing, shock])
        })
        .collect();

    let feet = def.feet;
    let foot_up = feet.map_or(DVec3::new(hip.x - 0.1, 0.2, pr.z), |f| f.up);
    let foot_down = feet.map_or(foot_up, |f| f.down);
    let limbs = [
        limb(if motorcycle { 0.034 } else { 0.014 }, if motorcycle { srgb([200, 170, 60]) } else { frame }),
        limb(0.05, jacket),
        limb(0.043, jacket),
        limb(0.075, trousers),
        limb(0.055, trousers),
    ];
    let boot = mesh::cuboid(Vec3::new(0.12, 0.05, 0.06), dark);
    let reach = (0..n).map(|w| def.wheel_position(w).length() + def.wheel_tire(w).radius()).fold(0.0, f64::max);
    let span = reach.max(head_at.length() + head_r) as f32;
    let eye = head_at + DVec3::new(head_r + 0.02, 0.0, 0.0);
    let rear_eye = DVec3::new(pr.x - rr - 0.2, 0.0, hip.z + 0.3);
    let visual = SingleTrackVisual {
        steered,
        pivot,
        axis,
        front_wheel: front,
        fork_offset,
        fork_leg: 0.6 * crown,
        torso,
        hip,
        shoulder: shoulders - hip + DVec3::new(0.0, 0.17, 0.0),
        grip: grip - pivot,
        arm: [0.3, 0.3],
        leg_hip: hip + DVec3::new(0.0, 0.1, -0.03),
        foot_up,
        foot_down,
        leg: [0.44, 0.46],
        limbs,
        boot,
    };
    WheeledVisual {
        body,
        units: Vec::new(),
        wheels,
        span,
        eye,
        rear_eye,
        links,
        link: mesh::cylinder((0.08 * rr) as f32, 0.5, 8, frame),
        single_track: Some(visual),
    }
}

/// Cylinder of radius `r` and width `w` across y, centred at `at`.
fn cylinder_y(m: &mut MeshData, at: DVec3, r: f64, w: f64, color: [f32; 4]) {
    let across = DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2);
    m.append_transformed(&mesh::cylinder(r as f32, (0.5 * w) as f32, 16, color), across, at);
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_vehicles::presets;

    fn def(name: &str) -> WheeledDef {
        match presets::get(name).unwrap() {
            autonomousim_vehicles::VehicleDef::Wheeled(w) => w,
            _ => unreachable!(),
        }
    }

    #[test]
    fn bend_keeps_the_segment_lengths() {
        let (a, b) = (DVec3::ZERO, DVec3::new(0.5, 0.0, 0.0));
        let k = bend(a, b, 0.4, 0.3, DVec3::Z);
        assert!(((k - a).length() - 0.4).abs() < 1e-12 && ((b - k).length() - 0.3).abs() < 1e-12);
        assert!(k.z > 0.0 && k.y.abs() < 1e-12);
        // Out of reach: on the line, in proportion.
        let k = bend(a, DVec3::new(2.0, 0.0, 0.0), 0.4, 0.6, DVec3::Z);
        assert!((k - DVec3::new(0.8, 0.0, 0.0)).length() < 1e-12);
    }

    /// Both presets: the fork tubes run through the front wheel's centre along the steering
    /// axis, the grips are where the handlebar colliders are, the hands reach them, the
    /// head is inside its collider and every part lies within the span.
    #[test]
    fn presets_are_assembled_around_their_geometry() {
        for name in ["motorcycle_sport", "bicycle_city"] {
            let d = def(name);
            let (head, front) = d.steering_head().unwrap();
            let v = visual(&d, head, front);
            let s = v.single_track.as_ref().unwrap();
            assert_eq!(v.wheels.len(), 2);
            assert!(
                (s.axis - head.axis()).length() < 1e-12
                    && (s.pivot - head.pivot(d.wheel_position(front))).length() < 1e-12
            );
            // Straight ahead, the slider stands beside the wheel centre, along the axis.
            let [bottom, top] = s.fork_slider(1.0, d.wheel_position(front), 0.0);
            assert!((bottom - d.wheel_position(front)).y > 0.5 * d.wheel_tire(front).section_width());
            assert!(((top - bottom).normalize() - s.axis).length() < 1e-12);
            // Turned left, the left slider moves back.
            let [turned, _] = s.fork_slider(1.0, d.wheel_position(front), 0.3);
            assert!(turned.x < bottom.x);
            for side in [-1.0, 1.0] {
                let [shoulder, elbow, hand] = s.arm_joints(side, 0.0, 0.0);
                assert!(hand.y * side > 0.2 && shoulder.z > hand.z, "{name}: {shoulder} {hand}");
                assert!(elbow.z < shoulder.z.max(hand.z));
                for down in [false, true] {
                    let [hip, knee, foot] = s.leg_joints(side, down);
                    assert!(knee.x > hip.x.min(foot.x) - 1e-9, "{name}: knee {knee}");
                    assert!(foot.z < hip.z);
                }
            }
            // Leaning right swings the shoulders right.
            let [upright, ..] = s.arm_joints(1.0, 0.0, 0.0);
            let [leaned, ..] = s.arm_joints(1.0, 0.0, 0.2);
            assert!(leaned.y < upright.y);
            let (lo, hi) = v.body.bounds().unwrap();
            assert!(lo.length().max(hi.length()) <= v.span + 0.05, "{name}: body {lo} {hi}, span {}", v.span);
            assert!(v.eye.x > s.hip.x && v.eye.z > s.hip.z);
        }
    }
}
