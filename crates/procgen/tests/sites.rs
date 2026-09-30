//! Lots, buildings, pads and parking bays of generated urban maps: clear of the streets and of
//! each other, fronting their streets, and matched by the obstacles and ray casts.

use autonomousim_core::geometry::{HitMask, Ray};
use autonomousim_procgen::urban::{self, UrbanPreset};
use autonomousim_world::lanes::Area;
use autonomousim_world::obstacles::tags;
use autonomousim_world::{BayKind, JunctionKind, Light, Roof, StaticWorld, Zone};
use glam::{DVec2, DVec3};

/// An oriented rectangle: centre, heading of its x axis, half sizes.
#[derive(Clone, Copy)]
struct Rect {
    c: DVec2,
    yaw: f64,
    half: DVec2,
}

impl Rect {
    fn axes(&self) -> [DVec2; 2] {
        let x = DVec2::from_angle(self.yaw);
        [x, x.perp()]
    }

    /// Half the extent along unit `u`.
    fn radius(&self, u: DVec2) -> f64 {
        let [x, y] = self.axes();
        self.half.x * x.dot(u).abs() + self.half.y * y.dot(u).abs()
    }

    /// Whether the interiors overlap by more than `tol` (separating axis test).
    fn overlaps(&self, o: &Rect, tol: f64) -> bool {
        let d = o.c - self.c;
        self.axes().into_iter().chain(o.axes()).all(|u| d.dot(u).abs() < self.radius(u) + o.radius(u) - tol)
    }

    /// Points on a grid of about `step` over the rectangle, edges included.
    fn samples(&self, step: f64) -> Vec<DVec2> {
        let [x, y] = self.axes();
        let (nx, ny) = ((2.0 * self.half.x / step).ceil() as i32, (2.0 * self.half.y / step).ceil() as i32);
        let mut out = Vec::new();
        for i in 0..=nx {
            for j in 0..=ny {
                let (u, v) = (i as f64 / nx as f64 - 0.5, j as f64 / ny as f64 - 0.5);
                out.push(self.c + x * (2.0 * u * self.half.x) + y * (2.0 * v * self.half.y));
            }
        }
        out
    }
}

fn check(w: &StaticWorld, label: &str) {
    let sites = w.sites();
    let net = w.roads();
    assert!(sites.buildings.len() > 100, "{label}: {} buildings", sites.buildings.len());
    for zone in [Zone::Downtown, Zone::Commercial, Zone::Residential, Zone::Parking] {
        assert!(sites.lots.iter().any(|l| l.zone == zone), "{label}: no {zone:?} lots");
    }
    assert!(!sites.pads.is_empty(), "{label}: no rooftop pads");
    for kind in [BayKind::Lot, BayKind::Street] {
        assert!(sites.bays.iter().filter(|b| b.kind == kind).count() > 20, "{label}: few {kind:?} bays");
    }
    let obstacles = w.obstacles().set().expect("obstacles").obstacles();
    signals(w, label);

    // Every lot fronts its street: the middle of its frontage lies within the raster's margin
    // (1.5 m and a cell's diagonal) of the road's sidewalk.
    for (k, lot) in sites.lots.iter().enumerate() {
        let road = &net.roads()[lot.road as usize];
        let pr = road.line.project(lot.front);
        let sidewalk = net.section(lot.road as usize).sidewalk[usize::from(pr.offset > 0.0)];
        let gap = pr.distance - 0.5 * road.width - sidewalk;
        assert!((-0.5..4.5).contains(&gap), "{label}: lot {k} front {gap} m from its street");
    }

    let footprints: Vec<Rect> =
        sites.buildings.iter().map(|b| Rect { c: b.centre, yaw: b.yaw, half: 0.5 * b.size }).collect();
    let reach = 40.0;
    for (k, (b, r)) in sites.buildings.iter().zip(&footprints).enumerate() {
        // Clear of every road and sidewalk.
        for p in r.samples(1.0) {
            if let Some((d, road)) = net.edge_distance(p, reach, true, None) {
                assert!(d > 0.0, "{label}: building {k} {d} m into road {road} (sidewalk included)");
            }
        }
        // Of no other building.
        for (j, o) in footprints.iter().enumerate().skip(k + 1) {
            assert!(!r.overlaps(o, 1e-6), "{label}: buildings {k} and {j} overlap");
        }
        // Its obstacles: the pieces within the footprint, from the base to the top.
        let [a, e] = b.obstacles;
        assert!(a < e && (e as usize) <= obstacles.len(), "{label}: building {k} obstacles {a}..{e}");
        let pieces = &obstacles[a as usize..e as usize];
        assert!(
            pieces
                .iter()
                .all(|o| [tags::BLOCK, tags::ROOF, tags::PARAPET, tags::PAD, tags::ROOF_UNIT].contains(&o.tag))
        );
        assert_eq!(pieces.iter().any(|o| o.tag == tags::ROOF), b.roof == Roof::Gable, "{label}: building {k}");
        // Ray casts: horizontally at the front face (every shape spans it) and down onto the
        // roof of rectangular flat buildings.
        let [x, y] = r.axes();
        let z = b.base + 0.5 * b.height;
        let from = b.centre - y * (0.5 * b.size.y + 0.5);
        let hit = w.raycast(&Ray::new(from.extend(z), y.extend(0.0)), 5.0, HitMask::SOLID).expect("front face");
        assert!((hit.toi - 0.5).abs() < 1e-6, "{label}: building {k} front face at {}", hit.toi);
        let blocks = pieces.iter().filter(|o| o.tag == tags::BLOCK).count();
        if b.roof == Roof::Flat && blocks == 1 {
            let p = b.centre + x * (0.25 * b.size.x) + y * (0.25 * b.size.y);
            let hit =
                w.raycast(&Ray::new(p.extend(b.top() + 50.0), DVec3::NEG_Z), 100.0, HitMask::SOLID).expect("roof");
            let top = hit.point.z;
            assert!(top >= b.top() - 1e-6 && top <= b.top() + 2.0 + 1e-6, "{label}: building {k} roof at {top}");
        }
    }

    // Pads on flat roofs, their surface where a ray cast down meets it.
    for (k, pad) in sites.pads.iter().enumerate() {
        let b = &sites.buildings[pad.building as usize];
        assert_eq!(b.roof, Roof::Flat);
        assert!((pad.centre.z - b.top() - 0.1).abs() < 1e-9);
        for q in [DVec2::ZERO, DVec2::new(0.8, -0.6) * pad.half] {
            let p = pad.centre.truncate() + DVec2::from_angle(pad.yaw).rotate(q);
            let hit = w.raycast(&Ray::new(p.extend(pad.centre.z + 30.0), DVec3::NEG_Z), 60.0, HitMask::SOLID).unwrap();
            assert!((hit.point.z - pad.centre.z).abs() < 1e-6, "{label}: pad {k} surface at {}", hit.point.z);
        }
    }

    // Bays: off the buildings; lot bays off the streets, street bays in parking lanes.
    for (k, bay) in sites.bays.iter().enumerate() {
        let r = Rect { c: bay.centre.truncate(), yaw: bay.yaw, half: 0.5 * bay.size };
        for (j, f) in footprints.iter().enumerate() {
            assert!(!r.overlaps(f, 1e-6), "{label}: bay {k} overlaps building {j}");
        }
        match bay.kind {
            BayKind::Lot => {
                for p in r.samples(0.5) {
                    if let Some((d, road)) = net.edge_distance(p, reach, true, None) {
                        assert!(d > 0.0, "{label}: lot bay {k} {d} m into road {road}");
                    }
                }
            }
            BayKind::Street => {
                let shrunk = Rect { half: r.half - DVec2::splat(0.05), ..r };
                for p in shrunk.samples(0.5) {
                    assert_eq!(net.area(p), Area::Parking, "{label}: street bay {k} at {p}");
                }
            }
        }
    }
}

/// Every signalized junction has a controller over all its connectors, with a 60–120 s cycle,
/// and no two conflicting connectors are ever green (or amber) together.
fn signals(w: &StaticWorld, label: &str) {
    let g = w.roads().lanes();
    let signalled = g.junctions().iter().filter(|j| j.kind == JunctionKind::Signal).count();
    assert!(signalled > 0, "{label}: no signals");
    assert_eq!(g.controllers().len(), signalled, "{label}");
    for ctl in g.controllers() {
        let j = &g.junctions()[ctl.junction as usize];
        assert!((2..=6).contains(&ctl.phases.len()), "{label}: junction {} {} phases", j.node, ctl.phases.len());
        let cycle = ctl.cycle();
        assert!((60.0 - 1e-9..=120.0).contains(&cycle), "{label}: cycle {cycle}");
        let phase = |c: u32| g.connector_signal(c).expect("signalled").1 as usize;
        for s in 0..(cycle / 0.25) as usize {
            let t = s as f64 * 0.25;
            let go: Vec<u32> = j.connectors.iter().copied().filter(|&c| ctl.light(phase(c), t) != Light::Red).collect();
            for &a in &go {
                for e in &g.connectors()[a as usize].conflicts {
                    assert!(!go.contains(&e.other), "{label}: junction {}: {a} and {} at {t}", j.node, e.other);
                }
            }
        }
    }
}

#[test]
fn urban_sites() {
    let c = UrbanPreset::Training.config();
    for seed in [1, 2, 5] {
        let (w, _) = urban::generate(&c, seed).unwrap();
        check(&w, &format!("urban {seed}"));
    }
}

/// More seeds and a showcase map.
#[test]
#[ignore]
fn many_urban_sites() {
    let (w, _) = urban::generate(&UrbanPreset::Showcase.config(), 3).unwrap();
    check(&w, "showcase 3");
    for seed in 1..30 {
        let (w, _) = urban::generate(&UrbanPreset::Training.config(), seed).unwrap();
        check(&w, &format!("urban {seed}"));
    }
}
