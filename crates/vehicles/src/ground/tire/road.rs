//! Tyre contact with the terrain: a local road plane fitted through four terrain samples around
//! the patch (front, rear, left, right: an envelope over features shorter than the patch, as in
//! Chrono's and MF-Tyre's single-contact-point models), and the contact point where the wheel
//! plane meets it.

use autonomousim_core::material::MaterialId;
use autonomousim_core::terrain::Terrain;
use glam::DVec3;

/// Where a wheel meets the local road plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoadContact {
    /// Contact point: on the road plane, in the wheel plane, below the wheel centre.
    pub point: DVec3,
    /// Road-plane normal (up) and the contact frame's forward and left axes in the plane
    /// (`x = axis × n`, `y = n × x`).
    pub normal: DVec3,
    pub x: DVec3,
    pub y: DVec3,
    /// Loaded radius: wheel centre to contact point (m). Negative if the centre is below the road.
    pub loaded_radius: f64,
    /// `sin γ = axis · n`: positive when the wheel top leans towards −y (a right-hand rotation
    /// about the forward axis).
    pub sin_gamma: f64,
    pub material: MaterialId,
}

/// The road plane below a wheel with centre `center` and spin axis `axis` (unit, pointing to
/// the wheel's left), sampled `half_length` ahead and behind and `half_width` to each side.
/// `None` if the wheel is further than `reach` above the ground or lies on its side.
pub fn road_contact<T: Terrain + ?Sized>(
    terrain: &T,
    center: DVec3,
    axis: DVec3,
    half_length: f64,
    half_width: f64,
    reach: f64,
) -> Option<RoadContact> {
    let (h0, n0) = terrain.height_normal(center.x, center.y);
    if center.z - h0 > reach {
        return None;
    }
    let x0 = axis.cross(n0);
    if x0.length_squared() < 1e-6 {
        return None;
    }
    let x0 = x0.normalize();
    let y0 = n0.cross(x0);
    // Centre projected onto the terrain's tangent plane.
    let base = center - n0 * (center.z - h0) * n0.z;
    let sample = |p: DVec3| DVec3::new(p.x, p.y, terrain.height(p.x, p.y));
    let front = sample(base + x0 * half_length);
    let rear = sample(base - x0 * half_length);
    let left = sample(base + y0 * half_width);
    let right = sample(base - y0 * half_width);
    let normal = (front - rear).cross(left - right).normalize();
    let origin = (front + rear + left + right) * 0.25;

    let x = axis.cross(normal);
    if x.length_squared() < 1e-6 {
        return None;
    }
    let x = x.normalize();
    let y = normal.cross(x);
    let sin_gamma = axis.dot(normal);
    let cos_gamma = (1.0 - sin_gamma * sin_gamma).sqrt();
    // Up direction in the wheel plane, and the distance to the plane along it.
    let up = (normal - axis * sin_gamma) / cos_gamma;
    let loaded_radius = (center - origin).dot(normal) / cos_gamma;
    let point = center - up * loaded_radius;
    Some(RoadContact { point, normal, x, y, loaded_radius, sin_gamma, material: terrain.material(point.x, point.y) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_core::terrain::{FlatTerrain, PlaneTerrain};
    use glam::DQuat;

    #[test]
    fn flat_ground_upright_and_cambered_wheel() {
        let flat = FlatTerrain::new(1.0, MaterialId::GRASS);
        let c = road_contact(&flat, DVec3::new(3.0, 4.0, 1.3), DVec3::Y, 0.1, 0.1, 1.0).unwrap();
        assert!((c.point - DVec3::new(3.0, 4.0, 1.0)).length() < 1e-12);
        assert!((c.x - DVec3::X).length() < 1e-12 && (c.y - DVec3::Y).length() < 1e-12);
        assert!((c.loaded_radius - 0.3).abs() < 1e-12 && c.sin_gamma.abs() < 1e-12);
        assert_eq!(c.material, MaterialId::GRASS);
        assert!(road_contact(&flat, DVec3::new(0.0, 0.0, 3.0), DVec3::Y, 0.1, 0.1, 1.0).is_none());

        // Heading along +y, leaning by a right-hand rotation of 0.2 rad about the heading.
        let q = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2) * DQuat::from_rotation_x(0.2);
        let axis = q * DVec3::Y;
        let c = road_contact(&flat, DVec3::new(0.0, 0.0, 1.3), axis, 0.1, 0.1, 1.0).unwrap();
        assert!((c.sin_gamma - 0.2f64.sin()).abs() < 1e-12);
        assert!((c.x - DVec3::Y).length() < 1e-12);
        assert!((c.loaded_radius - 0.3 / 0.2f64.cos()).abs() < 1e-12);
        assert!((c.point.z - 1.0).abs() < 1e-12 && (c.point - DVec3::new(0.0, 0.0, 1.3)).dot(axis).abs() < 1e-12);
    }

    #[test]
    fn slope_normal_and_depth() {
        let t = 0.3f64;
        let slope = PlaneTerrain::incline(t, MaterialId::ASPHALT);
        let n = DVec3::new(-t.sin(), 0.0, t.cos());
        let center = DVec3::new(2.0, 0.0, 2.0 * t.tan()) + n * 0.4;
        let c = road_contact(&slope, center, DVec3::Y, 0.15, 0.1, 1.0).unwrap();
        assert!((c.normal - n).length() < 1e-12);
        assert!((c.loaded_radius - 0.4).abs() < 1e-12);
        assert!((c.point - (center - n * 0.4)).length() < 1e-12);
        assert!((c.x - DVec3::new(t.cos(), 0.0, t.sin())).length() < 1e-12);
    }
}
