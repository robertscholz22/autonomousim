//! Depression filling, drainage and lakes on the vertex grid of the final height field.
//!
//! Priority-Flood (Barnes, Lehman & Mulla 2014, the variant with a plain queue for cells
//! inside depressions) floods the grid inward from the map border. Every vertex is discovered
//! by exactly one already-flooded neighbour, its *parent*: the parent links form a drainage
//! tree that also crosses flats and filled lakes, and accumulating cell counts along it in
//! reverse flooding order gives the upstream area. Heap keys order heights totally and break
//! ties by index, so everything is sequential and deterministic.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

/// No parent (a border vertex drains off the map).
pub(crate) const OUTLET: u32 = u32::MAX;

pub(crate) struct Drainage {
    /// Height after filling depressions up to their spill level.
    pub filled: Vec<f32>,
    /// Vertices in flooding order (non-decreasing `filled`).
    pub order: Vec<u32>,
    pub parent: Vec<u32>,
}

/// `f32` bits mapped to an unsigned integer with the same total order.
#[inline]
fn ordered(h: f32) -> u32 {
    let b = h.to_bits();
    if b >> 31 == 1 { !b } else { b | 0x8000_0000 }
}

const NEIGHBOURS: [(isize, isize); 8] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

pub(crate) fn priority_flood(h: &[f32], w: usize, ht: usize) -> Drainage {
    let n = w * ht;
    assert_eq!(h.len(), n);
    let mut filled = h.to_vec();
    let mut parent = vec![OUTLET; n];
    let mut closed = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut open: BinaryHeap<Reverse<u64>> = BinaryHeap::with_capacity(4 * (w + ht));
    let mut pit: VecDeque<u32> = VecDeque::new();
    let key = |z: f32, i: usize| Reverse(((ordered(z) as u64) << 32) | i as u64);
    for iy in 0..ht {
        for ix in 0..w {
            if ix == 0 || iy == 0 || ix == w - 1 || iy == ht - 1 {
                let i = iy * w + ix;
                closed[i] = true;
                open.push(key(h[i], i));
            }
        }
    }
    loop {
        let c = if let Some(c) = pit.pop_front() {
            c as usize
        } else if let Some(Reverse(k)) = open.pop() {
            (k & 0xFFFF_FFFF) as usize
        } else {
            break;
        };
        order.push(c as u32);
        let (cx, cy) = ((c % w) as isize, (c / w) as isize);
        let level = filled[c];
        for (dx, dy) in NEIGHBOURS {
            let (x, y) = (cx + dx, cy + dy);
            if x < 0 || y < 0 || x >= w as isize || y >= ht as isize {
                continue;
            }
            let j = y as usize * w + x as usize;
            if closed[j] {
                continue;
            }
            closed[j] = true;
            parent[j] = c as u32;
            if h[j] <= level {
                filled[j] = level;
                pit.push_back(j as u32);
            } else {
                open.push(key(h[j], j));
            }
        }
    }
    Drainage { filled, order, parent }
}

/// Number of vertices draining through each vertex (itself included).
pub(crate) fn accumulation(d: &Drainage) -> Vec<u32> {
    let mut acc = vec![1u32; d.parent.len()];
    for &c in d.order.iter().rev() {
        let p = d.parent[c as usize];
        if p != OUTLET {
            acc[p as usize] += acc[c as usize];
        }
    }
    acc
}

pub(crate) struct Lakes {
    /// Water surface per cell (`NaN` = dry).
    pub water: Vec<f32>,
    pub count: usize,
    pub cells: usize,
}

/// Lakes form in depressions: 8-connected groups of vertices that Priority-Flood raised. A
/// depression holds water up to its spill level unless the lake would then cover more than
/// `max_share` of its catchment (the vertices draining into it); then the level drops until
/// it does not (a lake without outflow, whose level evaporation sets), possibly leaving several
/// pools at the same level. Lakes shallower than `min_depth` or smaller than `min_vertices`
/// stay dry. A cell is wet if one of its corners is under water, with that lake's level.
pub(crate) fn lakes(
    h: &[f32],
    filled: &[f32],
    acc: &[u32],
    w: usize,
    ht: usize,
    min_depth: f32,
    min_vertices: usize,
    max_share: f64,
) -> Lakes {
    let n = w * ht;
    let mut seen = vec![false; n];
    // Level of the lake each vertex is under (NaN = dry).
    let mut level = vec![f32::NAN; n];
    let mut count = 0;
    let mut stack = Vec::new();
    let mut members: Vec<usize> = Vec::new();
    let mut ground: Vec<f32> = Vec::new();
    for s in 0..n {
        if seen[s] || filled[s] <= h[s] {
            continue;
        }
        seen[s] = true;
        stack.push(s);
        members.clear();
        let mut catchment = 0;
        while let Some(c) = stack.pop() {
            members.push(c);
            catchment = catchment.max(acc[c]);
            let (cx, cy) = ((c % w) as isize, (c / w) as isize);
            for (dx, dy) in NEIGHBOURS {
                let (x, y) = (cx + dx, cy + dy);
                if x < 0 || y < 0 || x >= w as isize || y >= ht as isize {
                    continue;
                }
                let j = y as usize * w + x as usize;
                if !seen[j] && filled[j] > h[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        let spill = filled[members[0]];
        let limit = ((max_share * catchment as f64) as usize).min(members.len());
        ground.clear();
        ground.extend(members.iter().map(|&m| h[m]));
        ground.sort_unstable_by(f32::total_cmp);
        // Vertices strictly below the level are submerged.
        let lvl = if limit == members.len() { spill } else { ground[limit].min(spill) };
        let wet = ground.partition_point(|&g| g < lvl);
        if wet < min_vertices.max(1) || lvl - ground[0] < min_depth {
            continue;
        }
        count += 1;
        for &m in &members {
            if h[m] < lvl {
                level[m] = lvl;
            }
        }
    }
    let (cw, ch) = (w - 1, ht - 1);
    let mut water = vec![f32::NAN; cw * ch];
    let mut cells = 0;
    for cy in 0..ch {
        for cx in 0..cw {
            let corners = [cy * w + cx, cy * w + cx + 1, (cy + 1) * w + cx, (cy + 1) * w + cx + 1];
            if let Some(&v) = corners.iter().find(|&&v| !level[v].is_nan()) {
                water[cy * cw + cx] = level[v];
                cells += 1;
            }
        }
    }
    Lakes { water, count, cells }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 9×9 bowl (depth 2 at the centre) with a 1-deep notch in its east rim.
    fn bowl() -> (Vec<f32>, usize) {
        let w = 9;
        let mut h = vec![3.0f32; w * w];
        for y in 1..8 {
            for x in 1..8 {
                let r = ((x as f32 - 4.0).abs()).max((y as f32 - 4.0).abs());
                h[y * w + x] = 1.0 + 0.5 * r;
            }
        }
        h[4 * w + 8] = 2.5; // spill point
        (h, w)
    }

    #[test]
    fn flood_fills_to_the_spill_level_and_drains_everything() {
        let (h, w) = bowl();
        let d = priority_flood(&h, w, w);
        assert_eq!(d.order.len(), w * w);
        // Inside the rim everything is raised to the spill level 2.5 (the rim ring at r = 3
        // is at 2.5 too).
        for y in 1..8 {
            for x in 1..8 {
                assert_eq!(d.filled[y * w + x], 2.5, "{x} {y}");
            }
        }
        let acc = accumulation(&d);
        // All 7×7 inner vertices drain through the spill vertex.
        assert!(acc[4 * w + 8] >= 49, "{}", acc[4 * w + 8]);
        let total: u32 = (0..w * w).filter(|&i| d.parent[i] == OUTLET).map(|i| acc[i]).sum();
        assert_eq!(total as usize, w * w);

        let l = lakes(&h, &d.filled, &acc, w, w, 0.3, 4, 1.0);
        assert_eq!(l.count, 1);
        assert!(l.water.iter().filter(|v| !v.is_nan()).all(|&v| v == 2.5));
        assert_eq!(l.cells, 36);
        let none = lakes(&h, &d.filled, &acc, w, w, 5.0, 4, 1.0);
        assert_eq!((none.count, none.cells), (0, 0));
        // A small catchment share lowers the level below the spill point; all pools share it.
        let low = lakes(&h, &d.filled, &acc, w, w, 0.3, 1, 9.0 / acc[4 * w + 8] as f64);
        assert!(low.cells > 0 && low.cells < 36, "{}", low.cells);
        let lvl = low.water.iter().find(|v| !v.is_nan()).copied().unwrap();
        assert!(lvl < 2.5 && low.water.iter().filter(|v| !v.is_nan()).all(|&v| v == lvl));
    }
}
