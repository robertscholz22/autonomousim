//! Cameras following the piloted vehicle: chase (behind it, turning with its heading), orbit
//! (mouse-controlled around it), first-person (fixed to the airframe, or from the driver's
//! seat) and, for ground vehicles, a reversing camera at the tail of the last unit (the
//! trailer's, for a rig) looking back; and a free-flying camera (W/A/S/D, Space/Shift, mouse drag to look, wheel for speed,
//! Ctrl for 4×). Ground vehicles are chased from lower and closer, relative to their size.
//!
//! Angles and positions are computed in ENU and converted once.

use crate::convert;
use crate::sim::Sim;
use crate::world_view::MapView;
use autonomousim_core::geometry::{HitMask, Ray};
use autonomousim_core::math::quat::yaw;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy_egui::EguiContexts;
use glam::{DQuat, DVec3};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CameraMode {
    #[default]
    Chase,
    Orbit,
    Fpv,
    Rear,
    Free,
}

impl CameraMode {
    /// The next mode; `ground`: with the reversing camera.
    pub fn next(self, ground: bool) -> Self {
        match self {
            CameraMode::Chase => CameraMode::Orbit,
            CameraMode::Orbit => CameraMode::Fpv,
            CameraMode::Fpv if ground => CameraMode::Rear,
            CameraMode::Fpv | CameraMode::Rear => CameraMode::Free,
            CameraMode::Free => CameraMode::Chase,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CameraMode::Chase => "chase",
            CameraMode::Orbit => "orbit",
            CameraMode::Fpv => "first person",
            CameraMode::Rear => "reversing",
            CameraMode::Free => "free",
        }
    }
}

/// The camera and its state.
#[derive(Component)]
pub struct CameraRig {
    pub mode: CameraMode,
    /// Heading of the camera (ENU, rad) and its downward pitch (rad).
    pub heading: f64,
    pub pitch: f64,
    /// Distance from the vehicle (m).
    pub distance: f64,
    /// Size of the vehicle (m), which sets distance limits.
    pub span: f64,
    /// Closest chase distance (m), and the height of the chased point above the vehicle
    /// origin (m).
    pub min_distance: f64,
    pub focus_height: f64,
    /// First-person eye point in the vehicle frame (m); `None`: ahead of the centre.
    pub eye_offset: Option<DVec3>,
    /// Reversing camera point in the last unit's frame (m); `None`: not a ground vehicle.
    pub rear_eye: Option<DVec3>,
    /// Where the camera is (ENU), and the speed of the free camera (m/s).
    pub eye: DVec3,
    pub speed: f64,
}

impl CameraRig {
    pub fn new(span: f64, heading: f64) -> Self {
        Self {
            mode: CameraMode::Chase,
            heading,
            pitch: 0.3,
            distance: (6.0 * span).max(1.2),
            span,
            min_distance: 2.0 * span,
            focus_height: 0.5 * span,
            eye_offset: None,
            rear_eye: None,
            eye: DVec3::ZERO,
            speed: 10.0,
        }
    }

    /// For a ground vehicle of size `span` with the driver's eye at `eye` (vehicle frame) and
    /// the reversing camera at `rear_eye` (the last unit's frame).
    pub fn ground(span: f64, heading: f64, eye: DVec3, rear_eye: DVec3) -> Self {
        Self {
            pitch: 0.2,
            distance: (2.6 * span).max(1.5),
            min_distance: (1.2 * span).max(0.6),
            focus_height: eye.z.max(0.3 * span),
            eye_offset: Some(eye),
            rear_eye: Some(rear_eye),
            ..Self::new(span, heading)
        }
    }

    /// Unit viewing direction of the chase, orbit and free cameras (ENU).
    fn look(&self) -> DVec3 {
        DVec3::new(self.heading.cos() * self.pitch.cos(), self.heading.sin() * self.pitch.cos(), -self.pitch.sin())
    }
}

pub fn update_camera(
    time: Res<Time>,
    sim: Res<Sim>,
    view: Res<MapView>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut egui: EguiContexts,
    mut camera: Query<(&mut CameraRig, &mut Transform)>,
) {
    let Ok((mut rig, mut transform)) = camera.single_mut() else { return };
    let dt = f64::from(time.delta_secs());
    let egui_busy = egui.ctx_mut().is_ok_and(|c| c.egui_wants_pointer_input() || c.is_pointer_over_egui());
    let typing = egui.ctx_mut().is_ok_and(|c| c.egui_wants_keyboard_input());
    if keys.just_pressed(KeyCode::KeyC) && !typing {
        rig.mode = rig.mode.next(rig.rear_eye.is_some());
        if rig.mode == CameraMode::Free {
            // Start where the first-person camera was, looking the same way.
            let forward = convert::enu(transform.forward().as_vec3());
            rig.heading = forward.y.atan2(forward.x);
            rig.pitch = (-forward.z).clamp(-1.0, 1.0).asin();
        }
    }
    if !egui_busy {
        if scroll.delta.y != 0.0 {
            let factor = 0.9f64.powf(f64::from(scroll.delta.y));
            if rig.mode == CameraMode::Free {
                rig.speed = (rig.speed / factor).clamp(1.0, 200.0);
            } else {
                rig.distance = (rig.distance * factor).clamp(rig.min_distance, 300.0);
            }
        }
        if buttons.pressed(MouseButton::Left) || buttons.pressed(MouseButton::Right) {
            rig.heading -= 0.005 * f64::from(motion.delta.x);
            rig.pitch = (rig.pitch + 0.005 * f64::from(motion.delta.y)).clamp(-1.2, 1.45);
        }
    }

    let pose = sim.render_pose(sim.pilot);
    let target = pose.pos;
    let (eye, look, up) = match rig.mode {
        CameraMode::Chase | CameraMode::Orbit => {
            if rig.mode == CameraMode::Chase && !buttons.pressed(MouseButton::Left) {
                // Swing behind the vehicle with a time constant of about half a second.
                let wanted = yaw(pose.rot);
                let err = (wanted - rig.heading + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI;
                rig.heading += err * (1.0 - (-dt / 0.5).exp());
            }
            let dir = rig.look();
            let focus = target + DVec3::Z * rig.focus_height;
            // Move in front of terrain, trunks and crowns between the vehicle and the camera.
            let mask = HitMask(HitMask::TERRAIN.0 | HitMask::SOLID.0 | HitMask::FOLIAGE.0);
            let distance = view
                .world
                .raycast(&Ray::new(focus, -dir), rig.distance, mask)
                .map_or(rig.distance, |hit| (hit.toi - 0.15).max(0.75 * rig.min_distance));
            let mut eye = focus - dir * distance;
            // Stay above the ground and water.
            let floor = view.world.surface_height(eye.x, eye.y) + (0.3 * rig.span).max(0.15);
            eye.z = eye.z.max(floor);
            (eye, focus, DVec3::Z)
        }
        CameraMode::Fpv => {
            // From the driver's seat, looking 5° down; for aircraft slightly ahead of and above
            // the centre, tilted up by 10°.
            let (offset, tilt) = match rig.eye_offset {
                Some(eye) => (eye, 5f64),
                None => (DVec3::new(0.4 * rig.span, 0.0, 0.15 * rig.span), -10f64),
            };
            let camera_rot = pose.rot * DQuat::from_rotation_y(tilt.to_radians());
            let eye = target + pose.rot * offset;
            (eye, eye + camera_rot * DVec3::X, camera_rot * DVec3::Z)
        }
        CameraMode::Rear => {
            // From the tail, looking back and 20° down; the last unit is posed relative to the
            // interpolated chassis.
            let unit = match sim.world.agent(sim.pilot).vehicle.as_wheeled() {
                Some(w) => pose * (w.unit_pose(0).inverse() * w.unit_pose(w.num_units() - 1)),
                None => pose,
            };
            let camera_rot = unit.rot * DQuat::from_rotation_z(std::f64::consts::PI) * DQuat::from_rotation_y(0.35);
            let eye = unit.transform_point(rig.rear_eye.unwrap_or_default());
            (eye, eye + camera_rot * DVec3::X, camera_rot * DVec3::Z)
        }
        CameraMode::Free => {
            let dir = rig.look();
            let right = dir.cross(DVec3::Z).normalize_or(DVec3::X);
            let axis =
                |pos: KeyCode, neg: KeyCode| f64::from(keys.pressed(pos) as u8) - f64::from(keys.pressed(neg) as u8);
            let fast = if keys.pressed(KeyCode::ControlLeft) { 4.0 } else { 1.0 };
            let mut eye = rig.eye
                + (dir * axis(KeyCode::KeyW, KeyCode::KeyS)
                    + right * axis(KeyCode::KeyD, KeyCode::KeyA)
                    + DVec3::Z * axis(KeyCode::Space, KeyCode::ShiftLeft))
                    * (rig.speed * fast * dt);
            eye.z = eye.z.max(view.world.surface_height(eye.x, eye.y) + 0.3);
            (eye, eye + dir, DVec3::Z)
        }
    };
    rig.eye = eye;
    *transform = Transform::from_translation(convert::vec(eye)).looking_at(convert::vec(look), convert::vec(up));
}
