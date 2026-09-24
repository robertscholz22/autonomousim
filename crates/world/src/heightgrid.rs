//! Regular-grid height field with per-cell materials, optional water and a min/max pyramid.
//!
//! Vertex `(ix, iy)` sits at `origin + cell · (ix, iy)`. Each cell is split into two triangles
//! along the `(x0, y0)–(x1, y1)` diagonal, the same triangulation as a parry `HeightField` placed
//! with [`heightfield_to_enu_rotation`](autonomousim_core::math::frames::heightfield_to_enu_rotation)
//! and reversed rows. Heights are stored as `f32` (sub-millimetre at mountain scale, half the
//! memory); all evaluation is in `f64` from the stored values, so every query is consistent.
//!
//! The grid is a solid block over its extent: a ray that enters the extent through a side
//! below the surface hits at the entry point (a vertical "skirt"), and a ray starting below the
//! surface hits at `t = 0`. Point queries outside the extent clamp to the border.

use autonomousim_core::geometry::{HitKind, HitMask, Ray, RayHit, SurfacePoint, closest_point_on_triangle};
use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::Terrain;
use glam::{DVec2, DVec3};

/// Largest search radius (in cells) of [`HeightGrid::closest_point`].
const MAX_SEARCH_CELLS: f64 = 16.0;

#[derive(Clone, Debug)]
struct Level {
    w: usize,
    h: usize,
    min: Vec<f32>,
    /// Maximum of ground and water surface.
    max: Vec<f32>,
}

/// Height field on a regular grid (see the module docs for the layout).
#[derive(Clone, Debug)]
pub struct HeightGrid {
    origin: DVec2,
    cell: f64,
    inv_cell: f64,
    nx: usize,
    ny: usize,
    heights: Vec<f32>,
    materials: Vec<MaterialId>,
    water: Option<Vec<f32>>,
    pyramid: Vec<Level>,
}

impl HeightGrid {
    /// Grid with `nx × ny` vertices (row-major, `x` fastest) and `(nx-1) × (ny-1)` cell materials.
    pub fn new(origin: DVec2, cell: f64, nx: usize, ny: usize, heights: Vec<f32>, materials: Vec<MaterialId>) -> Self {
        assert!(nx >= 2 && ny >= 2, "height grid needs at least 2×2 vertices");
        assert!(cell > 0.0 && cell.is_finite());
        assert_eq!(heights.len(), nx * ny, "heights must have nx·ny entries");
        assert_eq!(materials.len(), (nx - 1) * (ny - 1), "materials must have one entry per cell");
        assert!(heights.iter().all(|h| h.is_finite()), "heights must be finite");
        let mut g =
            Self { origin, cell, inv_cell: 1.0 / cell, nx, ny, heights, materials, water: None, pyramid: Vec::new() };
        g.build_pyramid();
        g
    }

    /// Grid sampled from `height(x, y)` at the vertices and `material(x, y)` at cell centres.
    pub fn from_fn(
        origin: DVec2,
        cell: f64,
        nx: usize,
        ny: usize,
        height: impl Fn(f64, f64) -> f64,
        material: impl Fn(f64, f64) -> MaterialId,
    ) -> Self {
        let mut heights = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                heights.push(height(origin.x + ix as f64 * cell, origin.y + iy as f64 * cell) as f32);
            }
        }
        let mut materials = Vec::with_capacity((nx - 1) * (ny - 1));
        for cy in 0..ny - 1 {
            for cx in 0..nx - 1 {
                materials.push(material(origin.x + (cx as f64 + 0.5) * cell, origin.y + (cy as f64 + 0.5) * cell));
            }
        }
        Self::new(origin, cell, nx, ny, heights, materials)
    }

    /// Attach per-cell water surface heights (`NaN` = dry).
    pub fn with_water(mut self, water: Vec<f32>) -> Self {
        assert_eq!(water.len(), self.num_cells(), "water must have one entry per cell");
        self.water = water.iter().any(|w| !w.is_nan()).then_some(water);
        self.build_pyramid();
        self
    }

    pub fn origin(&self) -> DVec2 {
        self.origin
    }

    pub fn cell_size(&self) -> f64 {
        self.cell
    }

    /// Vertex counts `(nx, ny)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Cell counts `(nx-1, ny-1)`.
    pub fn cells(&self) -> (usize, usize) {
        (self.nx - 1, self.ny - 1)
    }

    pub fn num_cells(&self) -> usize {
        (self.nx - 1) * (self.ny - 1)
    }

    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    pub fn materials(&self) -> &[MaterialId] {
        &self.materials
    }

    pub fn water(&self) -> Option<&[f32]> {
        self.water.as_deref()
    }

    #[inline]
    pub fn vertex_height(&self, ix: usize, iy: usize) -> f64 {
        self.heights[iy * self.nx + ix] as f64
    }

    #[inline]
    pub fn cell_material(&self, cx: usize, cy: usize) -> MaterialId {
        self.materials[cy * (self.nx - 1) + cx]
    }

    #[inline]
    pub fn cell_water(&self, cx: usize, cy: usize) -> Option<f64> {
        let w = self.water.as_ref()?[cy * (self.nx - 1) + cx];
        (!w.is_nan()).then_some(w as f64)
    }

    /// Lowest ground and highest surface (ground or water) of the whole grid.
    pub fn height_range(&self) -> (f64, f64) {
        let top = self.pyramid.last().unwrap();
        (top.min[0] as f64, top.max[0] as f64)
    }

    fn build_pyramid(&mut self) {
        let (w, h) = self.cells();
        let mut min = Vec::with_capacity(w * h);
        let mut max = Vec::with_capacity(w * h);
        for cy in 0..h {
            for cx in 0..w {
                let c = self.corners(cx, cy).map(|v| v as f32);
                let lo = c[0].min(c[1]).min(c[2]).min(c[3]);
                let mut hi = c[0].max(c[1]).max(c[2]).max(c[3]);
                if let Some(wl) = self.cell_water(cx, cy) {
                    hi = hi.max(wl as f32);
                }
                min.push(lo);
                max.push(hi);
            }
        }
        let mut levels = vec![Level { w, h, min, max }];
        while levels.last().is_some_and(|l| l.w > 1 || l.h > 1) {
            let prev = levels.last().unwrap();
            let (w, h) = (prev.w.div_ceil(2), prev.h.div_ceil(2));
            let mut min = vec![f32::INFINITY; w * h];
            let mut max = vec![f32::NEG_INFINITY; w * h];
            for y in 0..prev.h {
                for x in 0..prev.w {
                    let (i, j) = ((y / 2) * w + x / 2, y * prev.w + x);
                    min[i] = min[i].min(prev.min[j]);
                    max[i] = max[i].max(prev.max[j]);
                }
            }
            levels.push(Level { w, h, min, max });
        }
        self.pyramid = levels;
    }

    /// Corner heights `[h00, h10, h01, h11]` of cell `(cx, cy)`.
    #[inline]
    fn corners(&self, cx: usize, cy: usize) -> [f64; 4] {
        let i = cy * self.nx + cx;
        [
            self.heights[i] as f64,
            self.heights[i + 1] as f64,
            self.heights[i + self.nx] as f64,
            self.heights[i + self.nx + 1] as f64,
        ]
    }

    /// Grid coordinates (in cells) of a world point.
    #[inline]
    fn local(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - self.origin.x) * self.inv_cell, (y - self.origin.y) * self.inv_cell)
    }

    /// Cell containing grid coordinates `(u, v)` (clamped to the grid) and the fractional
    /// position inside it (clamped to `[0, 1]`).
    #[inline]
    fn locate(&self, u: f64, v: f64) -> (usize, usize, f64, f64) {
        let (w, h) = self.cells();
        // Truncation equals floor for the non-negative (clamped) values; avoids libm calls.
        let cx = (u.max(0.0) as usize).min(w - 1);
        let cy = (v.max(0.0) as usize).min(h - 1);
        (cx, cy, (u - cx as f64).clamp(0.0, 1.0), (v - cy as f64).clamp(0.0, 1.0))
    }

    /// Gradient (per cell unit) of the lower `(00, 10, 11)` or upper `(00, 11, 01)` triangle;
    /// both planes pass through `h00` at `(0, 0)`.
    #[inline]
    fn plane_gradient(c: [f64; 4], lower: bool) -> (f64, f64) {
        let [h00, h10, h01, h11] = c;
        if lower { (h10 - h00, h11 - h10) } else { (h11 - h01, h01 - h00) }
    }

    /// Height and gradient (per cell unit) inside a cell; lower triangle when `fx ≥ fy`.
    #[inline]
    fn eval(c: [f64; 4], fx: f64, fy: f64) -> (f64, f64, f64) {
        let (gx, gy) = Self::plane_gradient(c, fx >= fy);
        (c[0] + fx * gx + fy * gy, gx, gy)
    }

    #[inline]
    fn normal_from_gradient(&self, gx: f64, gy: f64) -> DVec3 {
        DVec3::new(-gx * self.inv_cell, -gy * self.inv_cell, 1.0).normalize()
    }

    /// The two triangles of a cell in world coordinates: lower `(00, 10, 11)`, upper `(00, 11, 01)`.
    fn cell_triangles(&self, cx: usize, cy: usize) -> [[DVec3; 3]; 2] {
        let [h00, h10, h01, h11] = self.corners(cx, cy);
        let x0 = self.origin.x + cx as f64 * self.cell;
        let y0 = self.origin.y + cy as f64 * self.cell;
        let (x1, y1) = (x0 + self.cell, y0 + self.cell);
        let p00 = DVec3::new(x0, y0, h00);
        let p10 = DVec3::new(x1, y0, h10);
        let p01 = DVec3::new(x0, y1, h01);
        let p11 = DVec3::new(x1, y1, h11);
        [[p00, p10, p11], [p00, p11, p01]]
    }

    /// Inclusive cell index range covering world rectangle `[min, max]` (clamped).
    fn cell_range(&self, min: DVec2, max: DVec2) -> (usize, usize, usize, usize) {
        let (u0, v0) = self.local(min.x, min.y);
        let (u1, v1) = self.local(max.x, max.y);
        let (cx0, cy0, _, _) = self.locate(u0, v0);
        let (cx1, cy1, _, _) = self.locate(u1, v1);
        (cx0, cy0, cx1, cy1)
    }

    /// Ground/water hit inside cell `(cx, cy)` for ray parameters in `[ta, tb]`.
    #[allow(clippy::too_many_arguments)]
    fn cell_hit(
        &self,
        cx: usize,
        cy: usize,
        lr: &LocalRay,
        ta: f64,
        tb: f64,
        ground: bool,
        water: bool,
    ) -> Option<(f64, DVec3, HitKind)> {
        let mut best: Option<(f64, DVec3, HitKind)> = None;
        if ground {
            let c = self.corners(cx, cy);
            let (fx0, fy0) = (lr.u0 - cx as f64, lr.v0 - cy as f64);
            // Split the parameter interval where the ray crosses the diagonal fx = fy.
            let (g0, gd) = (fx0 - fy0, lr.du - lr.dv);
            let mut cuts = [ta, tb, tb];
            if gd != 0.0 {
                let td = -g0 / gd;
                if td > ta && td < tb {
                    cuts = [ta, td, tb];
                }
            }
            for k in 0..2 {
                let (s0, s1) = (cuts[k], cuts[k + 1]);
                if k == 1 && s0 >= s1 {
                    break;
                }
                let tm = 0.5 * (s0 + s1);
                let lower = fx0 + lr.du * tm >= fy0 + lr.dv * tm;
                let (gx, gy) = Self::plane_gradient(c, lower);
                let f = |t: f64| lr.z0 + lr.dz * t - (c[0] + (fx0 + lr.du * t) * gx + (fy0 + lr.dv * t) * gy);
                let (f0, f1) = (f(s0), f(s1));
                let t_hit = if f0 <= 0.0 {
                    Some(s0)
                } else if f1 <= 0.0 {
                    Some(s0 + (s1 - s0) * f0 / (f0 - f1))
                } else {
                    None
                };
                if let Some(t) = t_hit {
                    best = Some((t, self.normal_from_gradient(gx, gy), HitKind::Terrain));
                    break;
                }
            }
        }
        if water && let Some(wl) = self.cell_water(cx, cy) {
            let (za, zb) = (lr.z0 + lr.dz * ta, lr.z0 + lr.dz * tb);
            let t_w = if za <= wl {
                Some(ta)
            } else if zb <= wl {
                Some((wl - lr.z0) / lr.dz)
            } else {
                None
            };
            if let Some(t) = t_w
                && best.is_none_or(|b| t < b.0)
            {
                best = Some((t, DVec3::Z, HitKind::Water));
            }
        }
        best
    }
}

/// A ray in grid coordinates: `u = u0 + du·t` (cells), `z = z0 + dz·t` (metres), `t` in metres.
struct LocalRay {
    u0: f64,
    v0: f64,
    z0: f64,
    du: f64,
    dv: f64,
    dz: f64,
}

/// Parameter interval of `p0 + d·t` inside `[lo, hi]` (slab test), or `None`.
#[inline]
fn slab(p0: f64, d: f64, lo: f64, hi: f64, t: (f64, f64)) -> Option<(f64, f64)> {
    if d == 0.0 {
        return (p0 >= lo && p0 <= hi).then_some(t);
    }
    let (a, b) = ((lo - p0) / d, (hi - p0) / d);
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    let r = (t.0.max(a), t.1.min(b));
    (r.0 <= r.1).then_some(r)
}

impl Terrain for HeightGrid {
    fn extent(&self) -> (DVec2, DVec2) {
        let (w, h) = self.cells();
        (self.origin, self.origin + DVec2::new(w as f64, h as f64) * self.cell)
    }

    fn height(&self, x: f64, y: f64) -> f64 {
        let (u, v) = self.local(x, y);
        let (cx, cy, fx, fy) = self.locate(u, v);
        Self::eval(self.corners(cx, cy), fx, fy).0
    }

    fn height_normal(&self, x: f64, y: f64) -> (f64, DVec3) {
        let (u, v) = self.local(x, y);
        let (cx, cy, fx, fy) = self.locate(u, v);
        let (h, gx, gy) = Self::eval(self.corners(cx, cy), fx, fy);
        (h, self.normal_from_gradient(gx, gy))
    }

    fn material(&self, x: f64, y: f64) -> MaterialId {
        let (u, v) = self.local(x, y);
        let (cx, cy, _, _) = self.locate(u, v);
        self.cell_material(cx, cy)
    }

    fn water_level(&self, x: f64, y: f64) -> Option<f64> {
        self.water.as_ref()?;
        let (u, v) = self.local(x, y);
        let (cx, cy, _, _) = self.locate(u, v);
        self.cell_water(cx, cy)
    }

    fn height_bounds(&self, min: DVec2, max: DVec2) -> (f64, f64) {
        let (cx0, cy0, cx1, cy1) = self.cell_range(min, max);
        // Coarsest useful level: the rectangle spans at most 2×2 blocks.
        let mut level = 0;
        while level + 1 < self.pyramid.len()
            && ((cx1 >> level) - (cx0 >> level) > 1 || (cy1 >> level) - (cy0 >> level) > 1)
        {
            level += 1;
        }
        let l = &self.pyramid[level];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for by in (cy0 >> level)..=(cy1 >> level) {
            for bx in (cx0 >> level)..=(cx1 >> level) {
                lo = lo.min(l.min[by * l.w + bx]);
                hi = hi.max(l.max[by * l.w + bx]);
            }
        }
        (lo as f64, hi as f64)
    }

    fn closest_point(&self, p: DVec3, max_dist: f64) -> Option<SurfacePoint> {
        let (u, v) = self.local(p.x, p.y);
        let (hx, hy, fx, fy) = self.locate(u, v);
        let below = p.z < Self::eval(self.corners(hx, hy), fx, fy).0;
        let reach = max_dist.min(MAX_SEARCH_CELLS * self.cell);
        if !below && p.z - reach > self.height_bounds(p.truncate() - reach, p.truncate() + reach).1 {
            return None;
        }
        // Start from the cell below `p` (which contains its vertical projection inside the grid),
        // then visit neighbours whose bounding boxes could hold a closer point.
        let mut best = (f64::INFINITY, DVec3::ZERO, hx, hy, 0usize);
        let visit = |cx: usize, cy: usize, best: &mut (f64, DVec3, usize, usize, usize)| {
            let c = self.corners(cx, cy);
            let lx = p.x - (self.origin.x + cx as f64 * self.cell);
            let ly = p.y - (self.origin.y + cy as f64 * self.cell);
            for k in 0..2 {
                // Lower bound: squared distance to the triangle's plane plus the squared xy distance
                // from the foot of the perpendicular to the triangle's footprint (bounded below by
                // the largest violated edge half-plane). The foot is the answer when inside.
                let (gu, gv) = Self::plane_gradient(c, k == 0);
                let (gx, gy) = (gu * self.inv_cell, gv * self.inv_cell);
                let d = p.z - (c[0] + gx * lx + gy * ly);
                let t = d / (1.0 + gx * gx + gy * gy);
                let fx = (lx + gx * t) * self.inv_cell;
                let fy = (ly + gy * t) * self.inv_cell;
                let out = if k == 0 {
                    (-fy).max(fx - 1.0).max((fy - fx) * std::f64::consts::FRAC_1_SQRT_2)
                } else {
                    (-fx).max(fy - 1.0).max((fx - fy) * std::f64::consts::FRAC_1_SQRT_2)
                };
                let lateral = out.max(0.0) * self.cell;
                if d * t + lateral * lateral >= best.0 {
                    continue;
                }
                let q = if out <= 0.0 {
                    DVec3::new(p.x + gx * t, p.y + gy * t, p.z - t)
                } else {
                    let tri = self.cell_triangles(cx, cy)[k];
                    closest_point_on_triangle(p, tri[0], tri[1], tri[2])
                };
                let d2 = (p - q).length_squared();
                if d2 < best.0 {
                    *best = (d2, q, cx, cy, k);
                }
            }
        };
        visit(hx, hy, &mut best);
        let r = best.0.sqrt().min(if below { f64::INFINITY } else { max_dist }).min(MAX_SEARCH_CELLS * self.cell);
        let (cx0, cy0, cx1, cy1) = self.cell_range(p.truncate() - DVec2::splat(r), p.truncate() + DVec2::splat(r));
        for cy in cy0..=cy1 {
            let y0 = self.origin.y + cy as f64 * self.cell;
            let dy = (y0 - p.y).max(p.y - y0 - self.cell).max(0.0);
            for cx in cx0..=cx1 {
                if (cx, cy) == (hx, hy) {
                    continue;
                }
                let x0 = self.origin.x + cx as f64 * self.cell;
                let dx = (x0 - p.x).max(p.x - x0 - self.cell).max(0.0);
                let c = self.corners(cx, cy);
                let (lo, hi) = (c[0].min(c[1]).min(c[2].min(c[3])), c[0].max(c[1]).max(c[2].max(c[3])));
                let dz = (lo - p.z).max(p.z - hi).max(0.0);
                if dx * dx + dy * dy + dz * dz < best.0 {
                    visit(cx, cy, &mut best);
                }
            }
        }
        let (d2, q, cx, cy, k) = best;
        let dist = d2.sqrt();
        if !below && dist > max_dist {
            return None;
        }
        let normal = if dist > 1e-9 {
            if below { (q - p) / dist } else { (p - q) / dist }
        } else {
            let t = self.cell_triangles(cx, cy)[k];
            (t[1] - t[0]).cross(t[2] - t[0]).normalize()
        };
        Some(SurfacePoint {
            point: q,
            normal,
            distance: if below { -dist } else { dist },
            material: self.cell_material(cx, cy),
            kind: HitKind::Terrain,
        })
    }

    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        let ground = mask.contains(HitMask::TERRAIN);
        let water = mask.contains(HitMask::WATER) && self.water.is_some();
        if !ground && !water {
            return None;
        }
        let (w, h) = self.cells();
        let (u0, v0) = self.local(ray.origin.x, ray.origin.y);
        let lr = LocalRay {
            u0,
            v0,
            z0: ray.origin.z,
            du: ray.dir.x * self.inv_cell,
            dv: ray.dir.y * self.inv_cell,
            dz: ray.dir.z,
        };
        let (_, zmax) = self.height_range();
        let range = slab(lr.u0, lr.du, 0.0, w as f64, (0.0, max_toi))
            .and_then(|r| slab(lr.v0, lr.dv, 0.0, h as f64, r))
            .and_then(|r| slab(lr.z0, lr.dz, f64::NEG_INFINITY, zmax, r));
        let (t_lo, t_hi) = range?;

        let top = self.pyramid.len() - 1;
        let mut level = top;
        let (mut cx, mut cy, _, _) = self.locate(lr.u0 + lr.du * t_lo, lr.v0 + lr.dv * t_lo);
        let mut t = t_lo;
        loop {
            let l = &self.pyramid[level];
            let (bx, by) = (cx >> level, cy >> level);
            let (x0, x1) = (bx << level, ((bx + 1) << level).min(w));
            let (y0, y1) = (by << level, ((by + 1) << level).min(h));
            let tx = if lr.du > 0.0 {
                (x1 as f64 - lr.u0) / lr.du
            } else if lr.du < 0.0 {
                (x0 as f64 - lr.u0) / lr.du
            } else {
                f64::INFINITY
            };
            let ty = if lr.dv > 0.0 {
                (y1 as f64 - lr.v0) / lr.dv
            } else if lr.dv < 0.0 {
                (y0 as f64 - lr.v0) / lr.dv
            } else {
                f64::INFINITY
            };
            let t_exit = tx.min(ty).min(t_hi).max(t);
            let z_lo = (lr.z0 + lr.dz * t).min(lr.z0 + lr.dz * t_exit);
            let above = z_lo > l.max[by * l.w + bx] as f64;
            if !above {
                if level > 0 {
                    level -= 1;
                    continue;
                }
                if let Some((toi, normal, kind)) = self.cell_hit(cx, cy, &lr, t, t_exit, ground, water) {
                    let material = if kind == HitKind::Water { MaterialId::WATER } else { self.cell_material(cx, cy) };
                    return Some(RayHit { toi, point: ray.at(toi), normal, material, kind });
                }
            }
            if t_exit >= t_hi {
                return None;
            }
            // Step into the neighbouring block across the exit face (at cell granularity).
            if tx <= ty {
                cx = if lr.du > 0.0 { x1 } else { x0.checked_sub(1)? };
                if cx >= w {
                    return None;
                }
                let v = lr.v0 + lr.dv * t_exit;
                cy = (v.floor() as i64).clamp(y0 as i64, y1 as i64 - 1) as usize;
            } else {
                cy = if lr.dv > 0.0 { y1 } else { y0.checked_sub(1)? };
                if cy >= h {
                    return None;
                }
                let u = lr.u0 + lr.du * t_exit;
                cx = (u.floor() as i64).clamp(x0 as i64, x1 as i64 - 1) as usize;
            }
            t = t_exit;
            level = (level + 1).min(top);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::math::frames::heightfield_to_enu_rotation;
    use autonomousim_core::parry::query::{Ray as PRay, RayCast};
    use autonomousim_core::parry::shape::HeightField;
    use autonomousim_core::parry::utils::Array2;
    use autonomousim_core::rng::Seed;

    fn hills(origin: DVec2, cell: f64, nx: usize, ny: usize) -> HeightGrid {
        HeightGrid::from_fn(
            origin,
            cell,
            nx,
            ny,
            |x, y| 8.0 * (0.11 * x).sin() * (0.07 * y).cos() + 0.03 * x + 2.0 * (0.5 * x + 0.3 * y).sin(),
            |x, _| if x > 0.0 { MaterialId::GRASS } else { MaterialId::ROCK },
        )
    }

    fn parry_field(g: &HeightGrid) -> (HeightField, autonomousim_core::parry::math::Pose) {
        let (nx, ny) = g.dims();
        let mut a = Array2::zeros(ny, nx);
        for i in 0..ny {
            for j in 0..nx {
                a[(i, j)] = g.vertex_height(j, ny - 1 - i);
            }
        }
        let size = DVec2::new((nx - 1) as f64, (ny - 1) as f64) * g.cell_size();
        let hf = HeightField::new(a, DVec3::new(size.x, 1.0, size.y));
        let center = g.origin() + size * 0.5;
        let pose = autonomousim_core::parry::math::Pose::from_parts(center.extend(0.0), heightfield_to_enu_rotation());
        (hf, pose)
    }

    /// Brute-force ray cast over every triangle (Möller–Trumbore).
    fn brute_raycast(g: &HeightGrid, ray: &Ray, max_toi: f64) -> Option<f64> {
        let (w, h) = g.cells();
        let mut best: Option<f64> = None;
        for cy in 0..h {
            for cx in 0..w {
                for [a, b, c] in g.cell_triangles(cx, cy) {
                    let (e1, e2) = (b - a, c - a);
                    let pv = ray.dir.cross(e2);
                    let det = e1.dot(pv);
                    if det.abs() < 1e-14 {
                        continue;
                    }
                    let tv = ray.origin - a;
                    let uu = tv.dot(pv) / det;
                    let qv = tv.cross(e1);
                    let vv = ray.dir.dot(qv) / det;
                    let t = e2.dot(qv) / det;
                    if (-1e-12..=1.0 + 1e-12).contains(&uu)
                        && vv >= -1e-12
                        && uu + vv <= 1.0 + 1e-12
                        && (0.0..=max_toi).contains(&t)
                        && best.is_none_or(|b| t < b)
                    {
                        best = Some(t);
                    }
                }
            }
        }
        best
    }

    #[test]
    fn planar_grid_is_exact() {
        let (a, b, c) = (3.0, 0.2, -0.1);
        let g = HeightGrid::from_fn(
            DVec2::new(-10.0, -20.0),
            0.5,
            41,
            81,
            |x, y| a + b * x + c * y,
            |_, _| MaterialId::DIRT,
        );
        let n = DVec3::new(-b, -c, 1.0).normalize();
        for (x, y) in [(0.0, 0.0), (-9.9, 19.9), (3.37, -7.21), (9.99, 0.001)] {
            let (hh, nn) = g.height_normal(x, y);
            assert!((hh - (a + b * x + c * y)).abs() < 1e-5, "height at ({x},{y})");
            assert!((nn - n).length() < 1e-5);
        }
        let hit =
            g.raycast(&Ray::new(DVec3::new(1.0, 2.0, 20.0), DVec3::new(0.3, -0.2, -1.0)), 200.0, HitMask::ALL).unwrap();
        assert!((hit.point.z - g.height(hit.point.x, hit.point.y)).abs() < 1e-9);
        assert!((hit.normal - n).length() < 1e-5);
        assert_eq!(hit.material, MaterialId::DIRT);
    }

    /// The grid matches a parry heightfield placed with the documented ENU conversion.
    #[test]
    fn matches_parry_heightfield() {
        let g = hills(DVec2::new(-23.0, 11.0), 1.5, 33, 29);
        let (hf, pose) = parry_field(&g);
        let mut rng = Seed::from_u64(1).child("parry-hf").rng();
        let (lo, hi) = g.extent();
        for _ in 0..500 {
            let x = rng.range(lo.x + 0.01, hi.x - 0.01);
            let y = rng.range(lo.y + 0.01, hi.y - 0.01);
            // Vertical ray: parry height must equal ours.
            let pr = PRay::new(DVec3::new(x, y, 100.0), -DVec3::Z);
            let toi = hf.cast_ray(&pose, &pr, 1000.0, true).expect("vertical ray must hit");
            assert!((100.0 - toi - g.height(x, y)).abs() < 1e-9, "height mismatch at ({x}, {y})");
            // Random oblique ray.
            let origin = DVec3::new(x, y, rng.range(10.0, 30.0));
            let dir = DVec3::new(rng.normal(), rng.normal(), -rng.range(0.05, 1.0)).normalize();
            let ours = g.raycast(&Ray::new(origin, dir), 500.0, HitMask::TERRAIN);
            let theirs = hf.cast_ray_and_get_normal(&pose, &PRay::new(origin, dir), 500.0, true);
            match (ours, theirs) {
                (Some(a), Some(b)) => {
                    assert!((a.toi - b.time_of_impact).abs() < 1e-8, "toi {} vs parry {}", a.toi, b.time_of_impact);
                    assert!((a.normal - b.normal).length() < 1e-6 || (a.normal + b.normal).length() < 1e-6);
                }
                (None, None) => {}
                (a, b) => panic!("hit disagreement: ours {a:?}, parry {b:?}"),
            }
        }
    }

    #[test]
    fn pyramid_raycast_matches_brute_force() {
        let g = hills(DVec2::new(0.0, 0.0), 1.0, 70, 45);
        let mut rng = Seed::from_u64(2).child("rays").rng();
        let mut hits = 0;
        for k in 0..2000 {
            // Origins above the surface inside the grid, or outside the grid above all terrain
            // (a ray entering through the side below the surface hits the "skirt" at entry,
            // which the brute force does not model).
            let origin = if k % 3 == 0 {
                DVec3::new(rng.range(-10.0, 80.0), rng.range(-10.0, 55.0), rng.range(11.0, 25.0))
            } else {
                let (x, y) = (rng.range(0.0, 69.0), rng.range(0.0, 44.0));
                DVec3::new(x, y, g.height(x, y) + rng.range(0.01, 15.0))
            };
            // Include horizontal, grazing and axis-aligned rays.
            let dir = match k % 5 {
                0 => DVec3::new(rng.normal(), rng.normal(), 0.0),
                1 => DVec3::new(1.0, 0.0, -0.05),
                2 => DVec3::new(0.0, -1.0, rng.range(-0.3, 0.1)),
                _ => rng.unit_vector(),
            };
            let ray = Ray::new(origin, dir);
            let entry = slab(ray.origin.x, ray.dir.x, 0.0, 69.0, (0.0, 150.0))
                .and_then(|r| slab(ray.origin.y, ray.dir.y, 0.0, 44.0, r));
            if entry.is_some_and(|(t0, _)| ray.at(t0).z < g.height_range().1) && k % 3 == 0 {
                continue; // enters through the side below the surface
            }
            let ours = g.raycast(&ray, 150.0, HitMask::TERRAIN).map(|h| h.toi);
            let want = brute_raycast(&g, &ray, 150.0);
            match (ours, want) {
                (Some(a), Some(b)) => {
                    hits += 1;
                    assert!((a - b).abs() < 1e-7, "ray {k}: {a} vs brute force {b}")
                }
                (None, None) => {}
                (a, b) => panic!("ray {k} {ray:?}: ours {a:?}, brute force {b:?}"),
            }
        }
        assert!(hits > 500, "too few hits ({hits}) to be a meaningful test");
    }

    #[test]
    fn height_bounds_are_conservative() {
        let g = hills(DVec2::new(5.0, -5.0), 2.0, 60, 50);
        let mut rng = Seed::from_u64(3).child("bounds").rng();
        for _ in 0..300 {
            let a = DVec2::new(rng.range(0.0, 130.0), rng.range(-10.0, 100.0));
            let b = a + DVec2::new(rng.range(0.0, 30.0), rng.range(0.0, 30.0));
            let (lo, hi) = g.height_bounds(a, b);
            for _ in 0..20 {
                let (x, y) = (rng.range(a.x, b.x), rng.range(a.y, b.y));
                let hh = g.height(x, y);
                assert!(hh >= lo - 1e-6 && hh <= hi + 1e-6, "{hh} not in [{lo}, {hi}]");
            }
        }
    }

    #[test]
    fn closest_point_matches_brute_force() {
        let g = hills(DVec2::new(0.0, 0.0), 1.0, 40, 40);
        let mut rng = Seed::from_u64(4).child("closest").rng();
        for _ in 0..300 {
            let (x, y) = (rng.range(5.0, 35.0), rng.range(5.0, 35.0));
            let p = DVec3::new(x, y, g.height(x, y) + rng.range(-0.3, 1.0));
            let s = g.closest_point(p, 2.0).expect("within range");
            let mut best = f64::INFINITY;
            for cy in 0..39 {
                for cx in 0..39 {
                    for [a, b, c] in g.cell_triangles(cx, cy) {
                        best = best.min((p - closest_point_on_triangle(p, a, b, c)).length());
                    }
                }
            }
            assert!((s.distance.abs() - best).abs() < 1e-9, "distance {} vs {best}", s.distance);
            let below = p.z < g.height(x, y);
            assert_eq!(s.distance < 0.0, below);
            assert!(s.normal.z > 0.0, "normal must point out of the ground");
            assert!((s.point + s.normal * s.distance - p).length() < 1e-9 || s.distance.abs() < 1e-9);
        }
        assert!(g.closest_point(DVec3::new(20.0, 20.0, 50.0), 1.0).is_none());
        // Points beside the grid (heights clamp to the border) and near its edges.
        for _ in 0..200 {
            let (x, y) = (rng.range(-2.0, 42.0), rng.range(-2.0, 42.0));
            let p = DVec3::new(x, y, g.height(x, y) + rng.range(0.05, 1.0));
            let mut best = f64::INFINITY;
            for cy in 0..39 {
                for cx in 0..39 {
                    for [a, b, c] in g.cell_triangles(cx, cy) {
                        best = best.min((p - closest_point_on_triangle(p, a, b, c)).length());
                    }
                }
            }
            match g.closest_point(p, 3.0) {
                Some(s) => assert!((s.distance - best).abs() < 1e-9, "{p}: {} vs {best}", s.distance),
                None => assert!(best > 3.0, "{p}: missed surface at {best}"),
            }
        }
    }

    #[test]
    fn water_hits_and_levels() {
        let g = HeightGrid::from_fn(DVec2::ZERO, 1.0, 21, 21, |x, _| 0.2 * (x - 10.0).abs(), |_, _| MaterialId::SAND);
        let (w, h) = g.cells();
        let water: Vec<f32> = (0..w * h).map(|i| if (8..12).contains(&(i % w)) { 1.0 } else { f32::NAN }).collect();
        let g = g.with_water(water);
        assert_eq!(g.water_level(10.2, 5.0), Some(1.0));
        assert_eq!(g.water_level(2.0, 5.0), None);
        let ray = Ray::new(DVec3::new(10.5, 5.0, 10.0), -DVec3::Z);
        let hit = g.raycast(&ray, 100.0, HitMask::TERRAIN | HitMask::WATER).unwrap();
        assert_eq!(hit.kind, HitKind::Water);
        assert!((hit.toi - 9.0).abs() < 1e-9);
        let hit = g.raycast(&ray, 100.0, HitMask::TERRAIN).unwrap();
        assert_eq!(hit.kind, HitKind::Terrain);
        assert!((hit.point.z - 0.1).abs() < 1e-6);
    }
}
