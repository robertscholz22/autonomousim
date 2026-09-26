//! Tracks on soft soil (Bekker–Wong, Wong's *Theory of Ground Vehicles*, ch. 2 and 4): the
//! patches' sinkage against Bekker's pressure–sinkage law, the motion resistance against the
//! compaction integral (and Rankine's bulldozing), and the drawbar pull against slip against
//! Janosi–Hanamoto's integral with the soil's shear strength `c + p·tan φ`.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable, Soil};
use autonomousim_core::terrain::FlatTerrain;
use autonomousim_vehicles::ground::tire::{TRACK_CELLS, TireModel};
use autonomousim_vehicles::ground::*;
use autonomousim_vehicles::multirotor::AirData;
use autonomousim_vehicles::presets;
use glam::DVec3;
use std::sync::Arc;

const DT: f64 = 1e-3;
const G: f64 = 9.81;

fn soil(m: MaterialId) -> Soil {
    MaterialTable::rural().get(m).soil.unwrap()
}

struct World {
    terrain: FlatTerrain,
    materials: MaterialTable,
}

impl World {
    fn new(material: MaterialId) -> Self {
        Self { terrain: FlatTerrain::new(0.0, material), materials: MaterialTable::rural() }
    }

    /// Run for `seconds`, pulling the chassis origin back with `drawbar` (N).
    fn run(&self, v: &mut Wheeled, seconds: f64, drawbar: f64, input: impl FnMut(&Wheeled) -> DriveInput) {
        self.run_ramped(v, seconds, 0.0, drawbar, input);
    }

    /// As [`run`](Self::run), with gravity ramped up from 0 over the first `ramp` s.
    fn run_ramped(
        &self,
        v: &mut Wheeled,
        seconds: f64,
        ramp: f64,
        drawbar: f64,
        mut input: impl FnMut(&Wheeled) -> DriveInput,
    ) {
        let scene = StaticScene { terrain: &self.terrain, obstacles: &NoObstacles, materials: &self.materials };
        let air = AirData::default();
        for step in 0..(seconds / DT).round() as usize {
            let g = if ramp > 0.0 { G * (step as f64 * DT / ramp).min(1.0) } else { G };
            let i = input(v);
            v.begin_step();
            v.apply_drive(&i, &air);
            v.apply_tires(&scene);
            v.apply_contacts(&scene);
            if drawbar != 0.0 {
                let heading = v.orientation() * DVec3::X;
                v.apply_force(-drawbar * DVec3::new(heading.x, heading.y, 0.0).normalize(), v.position());
            }
            v.finish_step(DVec3::new(0.0, 0.0, -g)).unwrap();
        }
    }
}

fn vehicle(name: &str) -> Wheeled {
    let mut d = presets::wheeled(name).unwrap();
    d.chassis.drag_area = DVec3::ZERO;
    let mut v = Wheeled::new(Arc::new(d), DT);
    let init = v.rest(DVec3::ZERO, 0.0, 0.0);
    v.reset(&init);
    v
}

fn patch(v: &Wheeled) -> tire::TrackPatch {
    let TireModel::Track(p) = &v.def().tire(0).model else { panic!("track expected") };
    p.clone()
}

/// Wheels of a side (0 left, 1 right), front to rear.
fn side(v: &Wheeled, s: usize) -> Vec<usize> {
    let mut ws: Vec<usize> = (0..v.def().num_wheels()).filter(|w| w % 2 == s).collect();
    ws.sort_by(|a, b| v.def().wheel_position(*b).x.total_cmp(&v.def().wheel_position(*a).x));
    ws
}

/// A side's patches in a chain under equal loads on soil, in steady slip `i`: each sinks to
/// Bekker's `z = (p/(k_c/b + k_φ))^(1/n)`, the rear ones riding in the front one's rut, and
/// together they pull Janosi–Hanamoto's integral with `τ_max = c + p·tan φ` less the
/// compaction work `b·(k_c/b + k_φ)·z^(n+1)/(n+1)` and Rankine's bulldozing, once per track.
#[test]
fn patch_chain_sinks_and_pulls_like_bekker_and_janosi_hanamoto() {
    let d = presets::wheeled("tracked_apc").unwrap();
    let tire = d.tire(0).clone();
    let TireModel::Track(p) = &tire.model else { panic!() };
    let terrain = FlatTerrain::new(0.0, MaterialId::PLOWED);
    let s = soil(MaterialId::PLOWED);
    let surface = tire::Surface::of(MaterialTable::rural().get(MaterialId::PLOWED));
    let (fz, n, vx) = (15000.0, 5, 2.0);
    let (b, l) = (p.width, n as f64 * p.length);
    let area = b * p.length;
    let z = s.sinkage(fz / area, b);
    assert!(z > 0.03 && z < 0.06, "sinkage {z}");
    let rho = fz / p.vertical_stiffness + z;
    let resistance = s.compaction(b, 0.0, z) + s.bulldozing(b, z, 9.80665);
    for slip in [0.0, 0.02, 0.1, 0.3, -0.05] {
        let band = vx / (1.0 - slip);
        let mut states = vec![tire.initial_state(); n];
        let mut out = vec![tire::TireForces::default(); n];
        for _ in 0..(3.0 / DT) as usize {
            for k in 0..n {
                states[k].inflow =
                    [k.checked_sub(1).map(|f| states[f].shear[TRACK_CELLS - 1]), states.get(k + 1).map(|r| r.shear[0])];
                states[k].rut_in = [k.checked_sub(1).map(|f| states[f].sinkage), states.get(k + 1).map(|r| r.sinkage)];
            }
            for (k, state) in states.iter_mut().enumerate() {
                let motion = tire::WheelMotion {
                    center: DVec3::new(-(k as f64) * p.length, 0.0, p.radius - rho),
                    axis: DVec3::Y,
                    velocity: DVec3::new(vx, 0.0, 0.0),
                    carrier_angvel: DVec3::ZERO,
                    spin: band / (p.radius - rho + z),
                };
                let contact = tire.contact(&terrain, &motion);
                out[k] = tire.step(state, contact.as_ref(), &motion, surface, DT);
            }
        }
        for f in &out {
            assert!(
                (f.sinkage / z - 1.0).abs() < 1e-6 && (f.fz / fz - 1.0).abs() < 1e-6,
                "{} m, {} N",
                f.sinkage,
                f.fz
            );
        }
        let total: f64 = out.iter().map(|f| f.fx).sum();
        let (k, i) = (s.shear_modulus, slip.abs());
        let strength = n as f64 * (s.cohesion * area + fz * s.friction_angle.tan());
        let thrust =
            if i == 0.0 { 0.0 } else { slip.signum() * strength * (1.0 - k / (i * l) * (1.0 - (-i * l / k).exp())) };
        let expected = thrust - resistance;
        assert!((total - expected).abs() < 0.01 * expected.abs().max(resistance), "slip {slip}: {total} vs {expected}");
    }
}

/// Parked on plowed soil and on sand, loaded gently (gravity ramped up over 2 s), the APC's
/// loaded patches sink by Bekker's law under their loads, and the hull sits lower than on
/// rigid ground by about as much. The soil keeps the rut of the peak load: as the hull sinks,
/// the end road wheels take up load from the middle ones, which stay up to 8 % deeper than
/// their final loads give. (Dropped onto the soil from its rigid-ground pose, the hull
/// overshoots, up to 30 % deeper.) The driving test checks Bekker's law closely.
#[test]
fn apc_sinks_by_bekkers_law() {
    let parked = |_: &Wheeled| DriveInput { parking: true, ..Default::default() };
    let rigid_height = {
        let mut v = vehicle("tracked_apc");
        World::new(MaterialId::ASPHALT).run(&mut v, 3.0, 0.0, parked);
        v.position().z
    };
    for m in [MaterialId::PLOWED, MaterialId::SAND] {
        let w = World::new(m);
        let mut v = vehicle("tracked_apc");
        let top = v.position() + DVec3::Z * 0.01;
        let rest = v.rest(DVec3::ZERO, 0.0, 0.0);
        v.reset(&WheeledInit { pose: autonomousim_core::math::Pose::new(top, rest.pose.rot), ..rest });
        w.run_ramped(&mut v, 10.0, 2.0, 0.0, parked);
        assert!(v.lin_vel_world().length() < 1e-3, "{m:?}: still moving at {}", v.lin_vel_world());
        let (s, p) = (soil(m), patch(&v));
        let mean = v.def().total_mass() * G / 6.0;
        let mut sunk = Vec::new();
        for (k, wh) in v.wheels().enumerate() {
            if wh.tire.fz < 0.5 * mean {
                continue;
            }
            let bekker = s.sinkage(wh.tire.fz / (p.width * p.length), p.width);
            let ratio = wh.tire.sinkage / bekker;
            assert!(ratio > 0.999 && ratio < 1.08, "{m:?} wheel {k}: {} vs {bekker}", wh.tire.sinkage);
            sunk.push(wh.tire.sinkage);
        }
        assert_eq!(sunk.len(), 6);
        let drop = rigid_height - v.position().z;
        let lo = sunk.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = sunk.iter().copied().fold(0.0, f64::max);
        assert!(drop > 0.8 * lo && drop < 1.2 * hi, "{m:?}: hull {drop} m lower, patches {lo}–{hi}");
        assert!(v.sinkage() > 0.5 * lo && v.sinkage() < hi, "{m:?}: sinkage {}", v.sinkage());
    }
}

/// Driving at a steady speed on soft soil, the drive torque carries the internal resistance
/// and the soil's motion resistance: per track, the compaction work of a rut as deep as the
/// most loaded patch sinks and the bulldozing ahead of it. Each patch sinks by Bekker's law
/// under its load, or rides in the rut of a more loaded one ahead. (The rover is left out: in
/// soil soft enough for it to sink, its sprocket and idler colliders reach the rigid surface.)
#[test]
fn motion_resistance_matches_the_compaction_integral() {
    for (name, m, speed) in [("tracked_apc", MaterialId::PLOWED, 2.0), ("tracked_apc", MaterialId::SAND, 2.0)] {
        let w = World::new(m);
        let mut v = vehicle(name);
        let mut integral = 0.0;
        let mut control = |v: &Wheeled| {
            let e = speed - v.lin_vel_body().x;
            integral = (integral + 0.5 * e * DT).clamp(0.0, 1.0);
            DriveInput { throttle: (e + integral).clamp(0.0, 1.0), ..Default::default() }
        };
        w.run(&mut v, 20.0, 0.0, &mut control);
        let (s, p) = (soil(m), patch(&v));
        let (mut drive, mut internal, mut expected, mut speed_sum) = (0.0, 0.0, 0.0, 0.0);
        let samples = 2000;
        for _ in 0..samples {
            w.run(&mut v, DT, 0.0, &mut control);
            speed_sum += v.lin_vel_body().x;
            for sd in 0..2 {
                let ws = side(&v, sd);
                let mut rut = 0.0;
                for &wh in &ws {
                    let t = &v.wheel(wh).tire;
                    drive += v.wheel(wh).drive_torque / (p.radius - t.deflection);
                    internal += p.rolling_resistance * t.fz;
                    if t.fz > 0.0 {
                        rut = f64::max(rut, s.sinkage(t.fz / (p.width * p.length), p.width));
                    }
                }
                expected += s.compaction(p.width, 0.0, rut) + s.bulldozing(p.width, rut, 9.80665);
            }
        }
        let n = samples as f64;
        let u = speed_sum / n;
        assert!((u / speed - 1.0).abs() < 0.02, "{name} on {m:?}: speed {u}");
        let (resistance, expected) = ((drive - internal) / n, expected / n);
        assert!((resistance / expected - 1.0).abs() < 0.05, "{name} on {m:?}: {resistance} N vs {expected} N");
        // Sinkage: Bekker under the patch's load, or the rut ahead.
        for sd in 0..2 {
            let mut rut: f64 = 0.0;
            for wh in side(&v, sd) {
                let t = &v.wheel(wh).tire;
                if t.fz <= 0.0 {
                    continue;
                }
                rut = rut.max(s.sinkage(t.fz / (p.width * p.length), p.width));
                assert!((t.sinkage / rut - 1.0).abs() < 0.05, "{name} on {m:?}, wheel {wh}: {} vs {rut}", t.sinkage);
            }
        }
    }
}

/// Pulling a drawbar load on plowed soil at a steady speed, the rover's tracks slip until
/// Janosi–Hanamoto's integral over each side's patches (under their loads, shear growing from
/// the front, `τ_max = c + p·tan φ`) less the soil's motion resistance carries the load.
#[test]
fn rover_drawbar_pull_follows_janosi_hanamoto() {
    let m = MaterialId::PLOWED;
    let (s, w) = (soil(m), World::new(m));
    let weight = presets::wheeled("rover_tracked").unwrap().total_mass() * G;
    let mut last_slip = 0.0;
    for share in [0.1, 0.25, 0.4] {
        let mut v = vehicle("rover_tracked");
        let drawbar = share * weight;
        let input = |_: &Wheeled| DriveInput { throttle: 0.8, ..Default::default() };
        w.run(&mut v, 6.0, drawbar, input);
        let u0 = v.lin_vel_body().x;
        w.run(&mut v, 0.5, drawbar, input);
        let u = v.lin_vel_body().x;
        assert!(u > 0.2 && ((u - u0) / 0.5).abs() < 0.01, "speed {u0} → {u}");
        let p = patch(&v);
        let (k, l) = (s.shear_modulus, p.length);
        let at = |i: f64, x: f64| if i * x < 1e-12 { 0.0 } else { 1.0 - k / (i * x) * (1.0 - (-i * x / k).exp()) };
        let mut pull = 0.0;
        let mut slips = [0.0; 2];
        for sd in 0..2 {
            let ws = side(&v, sd);
            let band: f64 = ws.iter().map(|&w| v.wheel(w).spin * (p.radius - v.wheel(w).tire.deflection)).sum::<f64>()
                / ws.len() as f64;
            let i = 1.0 - u / band;
            slips[sd] = i;
            for (n, &wh) in ws.iter().enumerate() {
                let fz = v.wheel(wh).tire.fz;
                let strength = s.cohesion * p.width * l + fz * s.friction_angle.tan();
                pull += strength * ((n + 1) as f64 * at(i, (n + 1) as f64 * l) - n as f64 * at(i, n as f64 * l));
            }
            let max_load = ws.iter().map(|&w| v.wheel(w).tire.fz).fold(0.0, f64::max);
            let z = s.sinkage(max_load / (p.width * l), p.width);
            pull -= s.compaction(p.width, 0.0, z) + s.bulldozing(p.width, z, 9.80665);
        }
        assert!((pull / drawbar - 1.0).abs() < 0.05, "drawbar {drawbar} N at slips {slips:?}: integral {pull} N");
        assert!(slips[0] > last_slip, "slip {slips:?} not above {last_slip}");
        last_slip = slips[0];
    }
}
