//! The lane graphs of generated urban and rural maps: strongly connected, drivable connector
//! curvature, a conflict for every crossing, legal lane routes.

use autonomousim_core::rng::Seed;
use autonomousim_procgen::rural::{self, RuralPreset};
use autonomousim_procgen::urban::{self, UrbanPreset};
use autonomousim_world::lanes::{ConflictKind, LaneGraph, MIN_TURN_RADIUS, RouteStep, Turn, max_curvature};
use autonomousim_world::{JunctionKind, RoadNetwork};

/// Lanes reachable from lane 0 forwards and backwards, over connectors and lane changes.
fn reach(g: &LaneGraph, forward: bool) -> Vec<bool> {
    let n = g.lanes().len();
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, l) in g.lanes().iter().enumerate() {
        let mut add = |a: usize, b: usize| if forward { edges[a].push(b) } else { edges[b].push(a) };
        for &c in &l.successors {
            add(i, g.connectors()[c as usize].to as usize);
        }
        for nb in [l.left, l.right].into_iter().flatten() {
            add(i, nb as usize);
        }
    }
    let mut seen = vec![false; n];
    let mut stack = vec![0];
    seen[0] = true;
    while let Some(u) = stack.pop() {
        for &v in &edges[u] {
            if !seen[v] {
                seen[v] = true;
                stack.push(v);
            }
        }
    }
    seen
}

fn crosses(a: &[glam::DVec3], b: &[glam::DVec3]) -> bool {
    for w in a.windows(2) {
        for v in b.windows(2) {
            let (p, q, c, d) = (w[0].truncate(), w[1].truncate(), v[0].truncate(), v[1].truncate());
            let (r, s) = (q - p, d - c);
            let den = r.perp_dot(s);
            if den.abs() < 1e-12 {
                continue;
            }
            let t = (c - p).perp_dot(s) / den;
            let u = (c - p).perp_dot(r) / den;
            if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                return true;
            }
        }
    }
    false
}

fn check(roads: &RoadNetwork, seed: u64, label: &str) {
    let g = roads.lanes();
    assert!(!g.is_empty(), "{label}: no lanes");
    // Strongly connected (dead ends turn round).
    for forward in [true, false] {
        let seen = reach(g, forward);
        let missing: Vec<usize> = (0..seen.len()).filter(|&i| !seen[i]).collect();
        assert!(missing.is_empty(), "{label}: lanes {missing:?} not reachable (forward {forward})");
    }
    // Every lane leads somewhere (no lane change needed at its end).
    let ends: Vec<usize> = (0..g.lanes().len()).filter(|&i| g.lanes()[i].successors.is_empty()).collect();
    assert!(ends.is_empty(), "{label}: lanes {ends:?} lead nowhere");
    // U-turns keep their turning space off other roads' lanes.
    for (c, k) in g.connectors().iter().enumerate().filter(|(_, k)| k.turn == Turn::UTurn) {
        let road = g.lanes()[k.from as usize].road;
        for (i, l) in g.lanes().iter().enumerate().filter(|(_, l)| l.road != road) {
            let mut s = 0.0;
            while s <= k.line.length() {
                let d = l.line.project(k.line.point_at(s).truncate()).distance;
                assert!(d >= 5.0, "{label}: U-turn {c} comes {d:.2} m from lane {i} of another road");
                s += 1.0;
            }
        }
    }
    // Connector curvature within a car's turning circle, wherever the roads leave room for
    // it (on roads too short for both junctions the lanes shrink to a metre and the turns
    // tighten; those are rare).
    let cramped = |road: u32| {
        let [a, b] = g.setbacks(road as usize);
        a + b >= roads.roads()[road as usize].line.length() - 1.0 - 1e-6
    };
    let mut tight = 0;
    for (i, c) in g.connectors().iter().enumerate() {
        let k = max_curvature(&c.line);
        assert!(c.speed > 0.0 && c.speed.is_finite() && c.speed <= (2.5 / k).sqrt() + 1e-9);
        if k <= 1.0 / MIN_TURN_RADIUS + 0.01 {
            continue;
        }
        tight += 1;
        let (a, b) = (g.lanes()[c.from as usize].road, g.lanes()[c.to as usize].road);
        assert!(cramped(a) || cramped(b), "{label}: connector {i} ({:?}) curvature {k}", c.turn);
    }
    let _ = tight;
    // Every pair of crossing connectors from different lanes has a conflict, one of the two
    // yielding.
    for j in g.junctions() {
        for (x, &a) in j.connectors.iter().enumerate() {
            for &b in &j.connectors[x + 1..] {
                let (ca, cb) = (&g.connectors()[a as usize], &g.connectors()[b as usize]);
                if ca.from == cb.from {
                    continue;
                }
                let merge = ca.to == cb.to;
                if !merge && !crosses(ca.line.points(), cb.line.points()) {
                    continue;
                }
                let e = ca.conflicts.iter().find(|e| e.other == b);
                let f = cb.conflicts.iter().find(|e| e.other == a);
                let (Some(e), Some(f)) = (e, f) else {
                    panic!("{label}: connectors {a} and {b} cross without a conflict")
                };
                assert_eq!(e.kind == ConflictKind::Merge, merge);
                assert!(e.yields != f.yields, "{label}: {a} and {b}");
            }
        }
        match j.kind {
            JunctionKind::DeadEnd => assert!(j.approaches.len() <= 1),
            JunctionKind::Through => assert!(j.approaches.len() <= 2),
            _ => {}
        }
    }
    // Lane routes between random pairs exist and follow legal moves.
    let mut rng = Seed::from_u64(seed).child("lane-routes").rng();
    let n = g.lanes().len() as u64;
    for _ in 0..50 {
        let (a, b) = (rng.below(n) as u32, rng.below(n) as u32);
        let (la, lb) = (&g.lanes()[a as usize], &g.lanes()[b as usize]);
        let (s0, s1) = (0.5 * la.line.length(), 0.5 * lb.line.length());
        let r = g.route(a, s0, b, s1).unwrap_or_else(|| panic!("{label}: no route from lane {a} to {b}"));
        assert!(r.length >= 0.0 && r.length.is_finite());
        assert_eq!(r.steps.first(), Some(&RouteStep::Lane(a)));
        assert_eq!(r.steps.last(), Some(&RouteStep::Lane(b)));
        for w in r.steps.windows(2) {
            match (w[0], w[1]) {
                (RouteStep::Lane(l), RouteStep::Connector(c)) => assert_eq!(g.connectors()[c as usize].from, l),
                (RouteStep::Connector(c), RouteStep::Lane(l)) => assert_eq!(g.connectors()[c as usize].to, l),
                (RouteStep::Lane(l), RouteStep::Change(x, y)) => {
                    assert_eq!(l, x);
                    let lane = &g.lanes()[x as usize];
                    assert!(lane.left == Some(y) || lane.right == Some(y));
                }
                (RouteStep::Change(_, y), RouteStep::Lane(l)) => assert_eq!(l, y),
                other => panic!("{label}: illegal steps {other:?}"),
            }
        }
    }
    // Every lane can be found from its own midpoint.
    for (i, l) in g.lanes().iter().enumerate().step_by(7) {
        let s = 0.5 * l.line.length();
        let p = l.line.point_at(s).truncate();
        let (lane, _, off) = g.nearest_lane(p, 5.0, Some(l.line.heading_at(s))).expect("nearest lane");
        assert!(off.abs() < 1e-6, "{label}: lane {i} found at offset {off}");
        let found = &g.lanes()[lane as usize];
        assert!(found.line.point_at(0.5 * found.line.length()).truncate().distance(p) < 1.0 || lane == i as u32);
    }
}

#[test]
fn urban_lane_graphs() {
    let c = UrbanPreset::Training.config();
    for seed in [1, 2, 5, 9] {
        let (w, _) = urban::generate(&c, seed).unwrap();
        check(w.roads(), seed, &format!("urban {seed}"));
    }
}

#[test]
fn rural_lane_graphs() {
    let c = RuralPreset::Training.config();
    for seed in [1, 2] {
        let (w, _) = rural::generate(&c, seed).unwrap();
        check(w.roads(), seed, &format!("rural {seed}"));
    }
}

/// More seeds and a showcase map (about a minute).
#[test]
#[ignore]
fn many_lane_graphs() {
    let (w, _) = urban::generate(&UrbanPreset::Showcase.config(), 3).unwrap();
    check(w.roads(), 3, "showcase 3");
    for seed in 1..60 {
        let (w, _) = urban::generate(&UrbanPreset::Training.config(), seed).unwrap();
        check(w.roads(), seed, &format!("urban {seed}"));
    }
    for seed in 1..15 {
        let (w, _) = rural::generate(&RuralPreset::Training.config(), seed).unwrap();
        check(w.roads(), seed, &format!("rural {seed}"));
    }
}
