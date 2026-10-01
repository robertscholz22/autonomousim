//! The pedestrian network of urban maps (`Walkways`, the lane graph's crossings and their
//! pedestrian signals): connected, off the carriageways and clear of obstacles, crossings
//! meeting sidewalks at both ends, walk phases never alongside green through or protected
//! movements over the crossing, places reachable from each other.

use autonomousim_core::rng::Seed;
use autonomousim_procgen::urban::{self, UrbanPreset};
use autonomousim_world::lanes::Area;
use autonomousim_world::signals::MIN_WALK;
use autonomousim_world::{Light, PlaceKind, StaticWorld, Turn, WalkKind};
use glam::DVec3;

fn map(seed: u64) -> StaticWorld {
    urban::generate(&UrbanPreset::Training.config(), seed).unwrap().0
}

#[test]
fn walkways_connect_the_city_off_the_carriageways() {
    for seed in [1u64, 2, 3] {
        let world = map(seed);
        let t = std::time::Instant::now();
        let w = world.walkways();
        let built = t.elapsed();
        let net = world.roads();
        let g = net.lanes();
        assert!(!w.is_empty(), "seed {seed}");

        // One connected network.
        let comp = w.components();
        assert!(comp.iter().all(|&c| c == comp[0]), "seed {seed}: more than one part");

        // Nearly every urban street with sidewalks is walked along (links between junctions
        // so close that their areas leave less than 10 m of lane are crossed in them).
        let streets: Vec<usize> = (0..net.roads().len())
            .filter(|&i| {
                let r = &net.roads()[i];
                let [sa, sb] = g.setbacks(i);
                r.class.is_urban()
                    && net.section(i).sidewalk[0] > 0.0
                    && r.line.length() - sa - sb >= 10.0
                    && !g.is_ring(i)
            })
            .collect();
        let walked = streets
            .iter()
            .filter(|&&i| w.edges().iter().any(|e| e.kind == WalkKind::Sidewalk && e.road == Some(i as u32)))
            .count();
        assert!(walked as f64 >= 0.95 * streets.len() as f64, "seed {seed}: {walked} of {} streets", streets.len());

        // Sidewalks, corners and paths keep off carriageways and junction areas; all walkways
        // keep 0.3 m from solid obstacles (paths from a metre beyond their place).
        for (k, e) in w.edges().iter().enumerate() {
            let mut along = 0.0;
            let pts = e.line.points();
            for (j, p) in pts.iter().enumerate() {
                if j > 0 {
                    along += p.truncate().distance(pts[j - 1].truncate());
                }
                if e.kind != WalkKind::Crossing {
                    let a = net.area(p.truncate());
                    assert!(matches!(a, Area::Sidewalk | Area::Off), "seed {seed}: {:?} {k} at {p}: {a:?}", e.kind);
                }
                if e.kind != WalkKind::Path || along > 1.0 {
                    let clear = world.obstacle_clearance(*p + DVec3::Z, 1.0);
                    assert!(clear >= 0.3, "seed {seed}: {:?} {k} at {p}: {clear:.2} m from an obstacle", e.kind);
                }
            }
        }

        // Crossings meet sidewalks at both ends, and most of the lane graph's have a walkway.
        let mut crossed = 0;
        for e in w.edges().iter().filter(|e| e.kind == WalkKind::Crossing) {
            crossed += 1;
            let c = &g.crossings()[e.crossing.expect("its crossing") as usize];
            assert_eq!(e.road, Some(c.road));
            for n in [e.a, e.b] {
                let node = &w.nodes()[n as usize];
                assert!(
                    node.edges
                        .iter()
                        .any(|&f| matches!(w.edges()[f as usize].kind, WalkKind::Sidewalk | WalkKind::Corner)),
                    "seed {seed}: crossing {:?} ends at {} off the sidewalks",
                    e.crossing,
                    node.position
                );
                let a = net.area(node.position.truncate());
                assert!(
                    matches!(a, Area::Sidewalk | Area::Off),
                    "seed {seed}: crossing {:?} at {}: {a:?}",
                    e.crossing,
                    node.position
                );
            }
            // Across the band.
            let mid = e.line.point_at(0.5 * e.line.length()).truncate();
            assert_eq!(net.area(mid), Area::Crosswalk, "seed {seed}: crossing {:?} at {mid}", e.crossing);
        }
        assert!(
            crossed as f64 >= 0.9 * g.crossings().len() as f64,
            "seed {seed}: {crossed} of {}",
            g.crossings().len()
        );

        // Places: most buildings have an entrance, there are bus stops, and places reach
        // each other.
        let entrances = w.places().iter().filter(|p| matches!(p.kind, PlaceKind::Entrance { .. })).count();
        let bus = w.places().iter().filter(|p| matches!(p.kind, PlaceKind::BusStop { .. })).count();
        assert!(entrances as f64 >= 0.8 * world.sites().buildings.len() as f64, "seed {seed}: {entrances} entrances");
        assert!(bus > 50, "seed {seed}: {bus} bus stops");
        let mut rng = Seed::from_u64(seed).child("routes").rng();
        for _ in 0..50 {
            let a = w.places()[rng.below(w.places().len() as u64) as usize].node;
            let b = w.places()[rng.below(w.places().len() as u64) as usize].node;
            let r = w.route(a, b).expect("connected");
            let (pa, pb) = (w.nodes()[a as usize].position, w.nodes()[b as usize].position);
            assert!(r.length + 1e-9 >= pa.truncate().distance(pb.truncate()));
            assert_eq!((r.nodes[0], *r.nodes.last().unwrap()), (a, b));
            let sum: f64 = r.edges.iter().map(|&(e, _)| w.edges()[e as usize].line.length()).sum();
            assert!((sum - r.length).abs() < 1e-6);
        }
        assert!(built.as_secs_f64() < 0.5, "seed {seed}: built in {built:?}");
        println!(
            "seed {seed}: {} nodes, {} edges, {crossed} crossings ({} mid-block), {entrances} entrances, {bus} bus stops, built in {built:?}",
            w.nodes().len(),
            w.edges().len(),
            g.crossings().iter().filter(|c| c.node.is_none()).count()
        );
    }
}

#[test]
fn pedestrians_never_walk_alongside_green_through_traffic() {
    for seed in [1u64, 2] {
        let world = map(seed);
        let g = world.roads().lanes();
        let mut signalled = 0;
        for (k, c) in g.crossings().iter().enumerate() {
            let Some((ctl, _)) = c.signal else { continue };
            signalled += 1;
            let controller = &g.controllers()[ctl as usize];
            let lanes: Vec<u32> = c.lanes.iter().map(|l| l.0).collect();
            let touching: Vec<u32> = g.junctions()[controller.junction as usize]
                .connectors
                .iter()
                .copied()
                .filter(|&x| {
                    let x = &g.connectors()[x as usize];
                    lanes.contains(&x.from) || lanes.contains(&x.to)
                })
                .collect();
            let cycle = controller.cycle();
            let (mut walk, mut t) = (0.0, 0.0);
            while t < cycle {
                let (go, left) = g.crossing_walk(k as u32, t).expect("signalled");
                assert!(left > 0.0);
                if go {
                    walk += 0.1;
                    for &x in &touching {
                        let Some((xc, phase)) = g.connector_signal(x) else { continue };
                        assert_eq!(xc, ctl);
                        let light = controller.light(phase as usize, t);
                        let conn = &g.connectors()[x as usize];
                        let protected = controller.phases[phase as usize].protected;
                        assert!(
                            light == Light::Red || (conn.turn != Turn::Straight && !protected),
                            "seed {seed}: crossing {k} walks at {t:.1} s with connector {x} ({:?}) {light:?}",
                            conn.turn
                        );
                    }
                }
                t += 0.1;
            }
            assert!(walk >= MIN_WALK - 0.2, "seed {seed}: crossing {k} walks {walk:.1} s a cycle");
        }
        assert!(signalled > 20, "seed {seed}: {signalled} signalled crossings");
    }
}
