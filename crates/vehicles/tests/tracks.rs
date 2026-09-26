//! Tracked running gear on rigid ground: the static solution, straight running, holding and
//! sliding on slopes around `atan μ`, skid steering, and the track patches' steady shear
//! against Janosi–Hanamoto's integral.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::Pose;
use autonomousim_core::terrain::{FlatTerrain, PlaneTerrain, Terrain};
use autonomousim_vehicles::ground::tire::{TRACK_CELLS, TireModel};
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::presets;
use glam::{DQuat, DVec3};
use std::sync::Arc;

const DT: f64 = 1e-3;
const G: f64 = 9.81;

struct World {
    terrain: Box<dyn Terrain>,
    materials: MaterialTable,
}

impl World {
    fn new(terrain: impl Terrain + 'static) -> Self {
        Self { terrain: Box::new(terrain), materials: MaterialTable::standard() }
    }

    fn flat() -> Self {
        Self::new(FlatTerrain::new(0.0, MaterialId::ASPHALT))
    }

    fn run(&self, v: &mut Wheeled, input: &DriveInput, seconds: f64) {
        let env = GroundStepEnv {
            scene: StaticScene { terrain: self.terrain.as_ref(), obstacles: &NoObstacles, materials: &self.materials },
            gravity: DVec3::new(0.0, 0.0, -G),
            air: AirData::default(),
        };
        for _ in 0..(seconds / DT).round() as usize {
            v.step(input, &env).unwrap();
        }
    }
}

fn rover(drag: bool) -> Wheeled {
    let mut d = presets::wheeled("rover_tracked").unwrap();
    if !drag {
        d.chassis.drag_area = DVec3::ZERO;
    }
    let mut v = Wheeled::new(Arc::new(d), DT);
    let init = v.rest(DVec3::ZERO, 0.0, 0.0);
    v.reset(&init);
    v
}

fn patch(v: &Wheeled) -> tire::TrackPatch {
    let TireModel::Track(p) = &v.def().tire(0).model else { panic!("track expected") };
    p.clone()
}

/// Band speed of a side (0 left, 1 right) from its road wheels' spin (m/s).
fn band(v: &Wheeled, side: usize) -> f64 {
    let ws: Vec<_> = v.wheels().enumerate().filter(|(w, _)| w % 2 == side).map(|(_, s)| s).collect();
    ws.iter().map(|s| s.spin * (patch(v).radius - s.tire.deflection)).sum::<f64>() / ws.len() as f64
}

#[test]
fn rover_loads_and_rests_at_the_static_solution() {
    let d = presets::wheeled("rover_tracked").unwrap();
    assert!((d.total_mass() - 56.0).abs() < 1e-9);
    assert_eq!(d.num_wheels(), 8);
    assert!(d.tire(0).is_track() && d.track.is_some());
    // Road wheels on each side are neighbours along the track, front to rear.
    let nb = d.track_neighbours();
    assert_eq!(nb[0], [None, Some(2)]);
    assert_eq!(nb[3], [Some(1), Some(5)]);
    assert_eq!(nb[7], [Some(5), None]);
    // Sprocket and idler on both sides are running-gear colliders.
    let cs = d.sphere_colliders();
    assert_eq!(cs.len(), 2 + 4);
    // The trailing arm swings the wheel back as it rises.
    let table = d.axles[0].suspension.as_ref().unwrap().table(0).unwrap();
    assert!(table.eval(0.03).position.x < table.eval(0.0).position.x);

    let st = d.static_state(G).unwrap();
    let mut v = rover(true);
    World::flat().run(&mut v, &DriveInput { parking: true, ..Default::default() }, 3.0);
    assert!(v.lin_vel_world().length() < 1e-3 && v.contacts().is_empty());
    assert!((v.position().z - st.height).abs() < 5e-4, "height {} vs {}", v.position().z, st.height);
    for (k, wh) in v.wheels().enumerate() {
        assert!((wh.tire.fz / st.loads[k] - 1.0).abs() < 5e-3, "wheel {k}: {} vs {}", wh.tire.fz, st.loads[k]);
        assert!(wh.travel.abs() < 5e-4, "wheel {k}: travel {}", wh.travel);
    }
    let total: f64 = st.loads.iter().sum();
    assert!((total / (v.mass() * G) - 1.0).abs() < 1e-9);
}

/// On the level at constant speed the motors carry the internal resistance and the tracks run
/// without slip, straight. Climbing a 15° grade at constant speed, the tracks' traction
/// carries the grade's pull, at the slip where the patches' steady shear (Janosi–Hanamoto
/// over each patch under its load, shear rising from the front of the track) gives it; the
/// cells' shear grows from front to rear. Braked, the tracks stop without creep.
#[test]
fn drives_straight_and_climbs_with_the_slip_of_its_traction() {
    let input = DriveInput { throttle: 0.7, ..Default::default() };
    let w = World::flat();
    let mut v = rover(false);
    w.run(&mut v, &input, 4.0);
    let u = v.lin_vel_body().x;
    assert!(u > 0.8, "speed {u}");
    assert!(v.position().y.abs() < 1e-6 && v.ang_vel_body().z.abs() < 1e-9, "not straight");
    for side in 0..2 {
        let slip = 1.0 - u / band(&v, side);
        assert!(slip.abs() < 1e-3, "side {side}: slip {slip}");
    }

    let angle = f64::to_radians(15.0);
    let w = World::new(PlaneTerrain::incline(angle, MaterialId::ASPHALT));
    let tilt = DQuat::from_rotation_y(-angle);
    let mut v = rover(false);
    let rest = v.rest(DVec3::ZERO, 0.0, 0.0);
    v.reset(&WheeledInit { pose: Pose::new(tilt * rest.pose.pos, tilt * rest.pose.rot), ..rest });
    w.run(&mut v, &input, 4.0);
    let u0 = v.lin_vel_body().x;
    w.run(&mut v, &input, 0.2);
    let u = v.lin_vel_body().x;
    assert!(u > 0.5 && ((u - u0) / 0.2).abs() < 0.01, "speed {u0} → {u}");
    let fx: f64 = v.wheels().map(|s| s.tire.fx).sum();
    let pull = v.mass() * G * angle.sin();
    assert!((fx / pull - 1.0).abs() < 0.01, "traction {fx} vs grade {pull}");
    let p = patch(&v);
    let mu = p.mu * MaterialTable::standard().get(MaterialId::ASPHALT).friction / 0.8;
    for side in 0..2 {
        let slip = 1.0 - u / band(&v, side);
        // Steady traction at slip i of the side's patches (front to rear), under their loads.
        let loads: Vec<f64> = (0..4).map(|k| v.wheel(2 * k + side).tire.fz).collect();
        let traction = |i: f64| -> f64 {
            let (k, l) = (p.shear_modulus, p.length);
            let at = |x: f64| if i * x < 1e-12 { 0.0 } else { 1.0 - k / (i * x) * (1.0 - (-i * x / k).exp()) };
            // Patch n spans [n·l, (n+1)·l] from the front: the mean stress over it.
            loads
                .iter()
                .enumerate()
                .map(|(n, fz)| mu * fz * ((n + 1) as f64 * at((n + 1) as f64 * l) - n as f64 * at(n as f64 * l)))
                .sum()
        };
        let side_fx: f64 = (0..4).map(|k| v.wheel(2 * k + side).tire.fx).sum();
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if traction(mid) < side_fx { lo = mid } else { hi = mid }
        }
        assert!(slip > 0.0 && (slip / lo - 1.0).abs() < 0.05, "side {side}: slip {slip} vs {lo}");
    }
    let shear: Vec<f64> = (0..4).flat_map(|k| v.tire_state(2 * k).shear.map(|c| c[0])).collect();
    assert_eq!(shear.len(), 4 * TRACK_CELLS);
    assert!(shear.windows(2).all(|p| p[1] > p[0]), "{shear:?}");

    let braked = DriveInput { brake: 1.0, ..Default::default() };
    w.run(&mut v, &braked, 5.0);
    let x = v.position();
    w.run(&mut v, &braked, 3.0);
    assert!((v.position() - x).length() < 1e-4, "crept {}", (v.position() - x).length());
}

/// Braked, the tracks hold on a slope below `atan μ` without creep and slide down one above
/// it at `g(sin θ − μ cos θ)`.
#[test]
fn holds_below_atan_mu_and_slides_above() {
    let mu = 0.9 * MaterialTable::standard().get(MaterialId::ASPHALT).friction / 0.8;
    let parked = DriveInput { parking: true, brake: 1.0, ..Default::default() };
    for (deg, holds) in [(38.0, true), (50.0, false)] {
        let angle = f64::to_radians(deg);
        assert_eq!(angle.tan() < mu, holds);
        let w = World::new(PlaneTerrain::incline(angle, MaterialId::ASPHALT));
        let tilt = DQuat::from_rotation_y(-angle);
        for yaw in [0.0, std::f64::consts::PI] {
            let mut v = rover(false);
            let rest = v.rest(DVec3::ZERO, yaw, 0.0);
            v.reset(&WheeledInit { pose: Pose::new(tilt * rest.pose.pos, tilt * rest.pose.rot), ..rest });
            w.run(&mut v, &parked, if holds { 5.0 } else { 2.0 });
            let p = v.position();
            let vel = v.lin_vel_world();
            w.run(&mut v, &parked, 2.0);
            assert!(v.contacts().is_empty(), "{deg}° yaw {yaw}: hull on the ground");
            if holds {
                let drift = (v.position() - p).length();
                assert!(drift < 1e-4, "{deg}° yaw {yaw}: crept {drift} m in 2 s");
            } else {
                let a = (v.lin_vel_world() - vel).length() / 2.0;
                let expected = G * (angle.sin() - mu * angle.cos());
                assert!((a / expected - 1.0).abs() < 0.05, "{deg}° yaw {yaw}: slides at {a} vs {expected}");
                assert!(v.lin_vel_world().x < 0.0, "slides downhill");
            }
        }
    }
}

/// Skid steering: with the right track faster the rover circles left at a steady rate, on a
/// radius wider than the tracks' speeds alone give (the tracks slip); the outer track drives
/// and the inner one brakes. With opposite tracks it turns on the spot.
#[test]
fn skid_steers_in_a_circle_and_on_the_spot() {
    let w = World::flat();
    let mut v = rover(false);
    let input = DriveInput { throttle: 0.6, yaw: 0.3, ..Default::default() };
    w.run(&mut v, &input, 5.0);
    let r0 = v.ang_vel_body().z;
    w.run(&mut v, &input, 2.0);
    let (r, u) = (v.ang_vel_body().z, v.lin_vel_body().x);
    assert!(r > 0.3 && (r / r0 - 1.0).abs() < 0.01, "yaw rate {r0} → {r}");
    let (left, right) = (band(&v, 0), band(&v, 1));
    let tread = 2.0 * v.def().axles[0].position.y;
    let kinematic = 0.5 * tread * (right + left) / (right - left);
    let radius = u / r;
    assert!(radius > kinematic && radius < 3.0 * kinematic, "radius {radius} vs kinematic {kinematic}");
    let fx = |side: usize| v.wheels().enumerate().filter(|(w, _)| w % 2 == side).map(|(_, s)| s.tire.fx).sum::<f64>();
    assert!(fx(1) > 0.0 && fx(0) < 0.0, "outer {} and inner {} track forces", fx(1), fx(0));

    let spot = DriveInput { yaw: 1.0, ..Default::default() };
    w.run(&mut v, &spot, 3.0);
    let p = v.position();
    w.run(&mut v, &spot, 2.0);
    assert!(v.ang_vel_body().z > 0.5, "spot turn at {}", v.ang_vel_body().z);
    assert!((v.position() - p).length() < 0.05, "drifted {} m while turning", (v.position() - p).length());
}

/// A side's patches in steady slip `i` (band speed against ground speed) sum to
/// Janosi–Hanamoto's integral over the track's contact length `ℓ`,
/// `μF_z·(1 − K/(iℓ)·(1 − e^(−iℓ/K)))`, driving and braking.
#[test]
fn steady_shear_matches_janosi_hanamoto() {
    let d = presets::wheeled("rover_tracked").unwrap();
    let tire = d.tire(0).clone();
    let p = patch(&rover(false));
    let terrain = FlatTerrain::new(0.0, MaterialId::ASPHALT);
    let (fz, n, vx) = (100.0, 4, 2.0);
    let rho = fz / p.vertical_stiffness;
    let l = n as f64 * p.length;
    for slip in [0.002, 0.01, 0.05, 0.2, -0.01, -0.1] {
        let band = vx / (1.0 - slip);
        let mut states = vec![tire.initial_state(); n];
        let mut total = 0.0;
        for _ in 0..(3.0 / DT) as usize {
            for k in 0..n {
                states[k].inflow =
                    [k.checked_sub(1).map(|f| states[f].shear[TRACK_CELLS - 1]), states.get(k + 1).map(|r| r.shear[0])];
            }
            total = 0.0;
            for (k, state) in states.iter_mut().enumerate() {
                let motion = tire::WheelMotion {
                    center: DVec3::new(-(k as f64) * p.length, 0.0, p.radius - rho),
                    axis: DVec3::Y,
                    velocity: DVec3::new(vx, 0.0, 0.0),
                    carrier_angvel: DVec3::ZERO,
                    spin: band / (p.radius - rho),
                };
                let contact = tire.contact(&terrain, &motion);
                let f = tire.step(state, contact.as_ref(), &motion, tire::Surface::REFERENCE, DT);
                assert!((f.fz - fz).abs() < 1e-6);
                total += f.fx;
            }
        }
        let k = p.shear_modulus;
        let i = slip.abs();
        let integral = slip.signum() * p.mu * n as f64 * fz * (1.0 - k / (i * l) * (1.0 - (-i * l / k).exp()));
        assert!((total / integral - 1.0).abs() < 0.01, "slip {slip}: {total} vs {integral}");
    }
}
