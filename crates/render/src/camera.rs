//! Pinhole intrinsics and camera poses.

use glam::{DQuat, DVec3};

/// An ideal pinhole camera with square pixels and the principal point at the image centre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    pub width: u32,
    pub height: u32,
    /// Horizontal field of view (rad).
    pub fov_x: f64,
    /// Near and far clip distances along the optical axis (m).
    pub near: f64,
    pub far: f64,
}

impl Intrinsics {
    pub fn new(width: u32, height: u32, fov_x: f64) -> Self {
        Self { width, height, fov_x, near: 0.05, far: 1000.0 }
    }

    /// Focal length in pixels (the same both ways).
    pub fn focal(&self) -> f64 {
        0.5 * self.width as f64 / (0.5 * self.fov_x).tan()
    }

    /// Principal point (px).
    pub fn centre(&self) -> (f64, f64) {
        (0.5 * self.width as f64, 0.5 * self.height as f64)
    }

    /// Vertical field of view (rad).
    pub fn fov_y(&self) -> f64 {
        2.0 * (0.5 * self.height as f64 / self.focal()).atan()
    }

    /// Camera-frame direction through image point `(u, v)` (px), scaled to unit depth
    /// (x = 1). Pixel `(i, j)`'s centre is `(i + 0.5, j + 0.5)`.
    pub fn ray(&self, u: f64, v: f64) -> DVec3 {
        let (cx, cy) = self.centre();
        let f = self.focal();
        DVec3::new(1.0, -(u - cx) / f, -(v - cy) / f)
    }

    /// Image point (px) of a camera-frame point, if it lies in front of the camera.
    pub fn project(&self, p: DVec3) -> Option<(f64, f64)> {
        if p.x <= 0.0 {
            return None;
        }
        let (cx, cy) = self.centre();
        let f = self.focal();
        Some((cx - f * p.y / p.x, cy - f * p.z / p.x))
    }
}

/// Where a camera is and where it looks: `orientation` rotates the camera frame (FLU,
/// optical axis +x) into the world (ENU).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    pub position: DVec3,
    pub orientation: DQuat,
}

impl CameraPose {
    pub fn new(position: DVec3, orientation: DQuat) -> Self {
        Self { position, orientation }
    }

    /// Camera-frame coordinates of world point `p`.
    pub fn to_camera(&self, p: DVec3) -> DVec3 {
        self.orientation.inverse() * (p - self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_and_projection_round_trip() {
        let k = Intrinsics::new(64, 48, 90f64.to_radians());
        assert!((k.focal() - 32.0).abs() < 1e-12);
        assert!((k.fov_y() - 2.0 * (24.0f64 / 32.0).atan()).abs() < 1e-12);
        for (u, v) in [(0.5, 0.5), (32.0, 24.0), (63.5, 10.25), (7.0, 47.5)] {
            let (pu, pv) = k.project(k.ray(u, v) * 7.5).unwrap();
            assert!((pu - u).abs() < 1e-12 && (pv - v).abs() < 1e-12);
        }
        // Right of the image is the camera's −y, the top its +z; the corner rays span the fov.
        assert!(k.ray(64.0, 24.0).y < 0.0 && k.ray(32.0, 0.0).z > 0.0);
        assert!((k.ray(64.0, 24.0).y.atan2(1.0) + 0.25 * std::f64::consts::PI).abs() < 1e-12);
        assert!(k.project(DVec3::new(-1.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn poses_map_the_world_into_the_camera() {
        // Looking straight down from 10 m: the ground below is 10 m ahead on the axis.
        let down = CameraPose::new(DVec3::new(3.0, 4.0, 10.0), DQuat::from_rotation_y(0.5 * std::f64::consts::PI));
        let p = down.to_camera(DVec3::new(3.0, 4.0, 0.0));
        assert!((p - DVec3::new(10.0, 0.0, 0.0)).length() < 1e-12, "{p}");
    }
}
