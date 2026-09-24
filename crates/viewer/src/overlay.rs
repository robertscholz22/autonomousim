//! Markers drawn over the scene with gizmos: the goals of every agent and the line to the
//! current one, the path flown over the last seconds, and the hits of the latest LiDAR scan.
//! O toggles the goals and trails, L the LiDAR points.

use crate::convert;
use crate::sim::Sim;
use autonomousim_core::math::Pose;
use autonomousim_sensors::{LidarConfig, Sensor};
use autonomousim_world::StaticWorld;
use bevy::prelude::*;
use glam::DVec3;
use std::collections::VecDeque;
use std::sync::Arc;

/// Seconds of flight shown as a trail.
const TRAIL: f64 = 10.0;
/// LiDAR hits drawn per scan at most.
const MAX_POINTS: usize = 4096;

#[derive(Resource)]
pub struct Overlay {
    pub markers: bool,
    pub lidar: bool,
    /// Live trails: (time, position) of every agent.
    trails: Vec<VecDeque<(f64, DVec3)>>,
    map: Option<Arc<StaticWorld>>,
}

impl Default for Overlay {
    fn default() -> Self {
        Self { markers: true, lidar: true, trails: Vec::new(), map: None }
    }
}

const GOAL: Color = Color::srgb(1.0, 0.8, 0.1);
const NEXT_GOALS: Color = Color::srgba(1.0, 0.8, 0.1, 0.4);
const TO_GOAL: Color = Color::srgba(1.0, 0.8, 0.1, 0.6);
const TRAIL_COLOR: Color = Color::srgb(0.2, 0.9, 1.0);
const PILOT_TRAIL: Color = Color::srgb(1.0, 0.35, 0.8);

/// Hit colour by range: red near, through yellow, to green far.
pub fn range_color(r: f64, max: f64) -> Color {
    let x = (r / max.max(1.0)).clamp(0.0, 1.0) as f32;
    Color::hsl(120.0 * x, 0.9, 0.55)
}

/// A LiDAR scan as the viewer shows it, live or replayed.
pub struct Scan<'a> {
    pub config: &'a LidarConfig,
    /// Beam directions in the sensor frame.
    pub directions: &'a [DVec3],
    /// Sensor pose in the world.
    pub pose: Pose,
    /// Range per beam (m); `None` without a return.
    pub ranges: Vec<Option<f64>>,
}

/// The latest LiDAR scan of `agent`: live from its sensor, in a replay the recorded scan at or
/// before the playback time. `None` without a LiDAR or before its first scan.
pub fn latest_scan(sim: &Sim, agent: usize) -> Option<Scan<'_>> {
    let lidar = sim.world.agent(agent).sensors.iter().find_map(|s| match s {
        Sensor::Lidar(l) => Some(l),
        _ => None,
    })?;
    let (pose, ranges) = if let Some(r) = &sim.replay {
        let scans = r.current().scans.get(agent)?;
        let k = scans.partition_point(|s| s.time <= r.time).checked_sub(1)?;
        let scan = &scans[k];
        (Pose { pos: scan.position, rot: scan.orientation }, scan.ranges.clone())
    } else {
        let scan = lidar.latest()?;
        let ranges = scan.ranges.iter().map(|&r| r.is_finite().then_some(f64::from(r))).collect();
        (scan.pose, ranges)
    };
    Some(Scan { config: lidar.config(), directions: lidar.directions(), pose, ranges })
}

pub fn draw(keys: Res<ButtonInput<KeyCode>>, sim: Res<Sim>, mut overlay: ResMut<Overlay>, mut gizmos: Gizmos) {
    if keys.just_pressed(KeyCode::KeyO) {
        overlay.markers = !overlay.markers;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        overlay.lidar = !overlay.lidar;
    }
    let n = sim.world.agents().len();
    let now = sim.time();

    // Live trails start over with a new episode or map.
    let map = sim.world.map();
    let restart = !overlay.map.as_ref().is_some_and(|m| Arc::ptr_eq(m, map))
        || overlay.trails.len() != n
        || overlay.trails.iter().any(|t| t.back().is_some_and(|&(t, _)| t > now));
    if restart {
        overlay.trails = vec![VecDeque::new(); n];
        overlay.map = Some(map.clone());
    }
    if sim.replay.is_none() && !sim.paused {
        for (i, trail) in overlay.trails.iter_mut().enumerate() {
            if trail.back().is_none_or(|&(t, _)| t < now) {
                trail.push_back((now, sim.render_pose(i).pos));
            }
            while trail.front().is_some_and(|&(t, _)| t < now - TRAIL) {
                trail.pop_front();
            }
        }
    }

    if overlay.markers {
        for i in 0..n {
            let agent = sim.world.agent(i);
            let pos = sim.render_pose(i).pos;
            // Goals: the current one solid, the ones after it faint and joined up; as large as
            // the vehicle.
            let arm = agent.vehicle.def().rotors.iter().map(|r| r.position.length()).fold(0.0, f64::max);
            let radius = (1.5 * arm).max(0.05) as f32;
            let current = agent.goal_index.min(agent.goals.len().saturating_sub(1));
            for (k, g) in agent.goals.iter().enumerate().skip(current) {
                let p = convert::vec(g.position);
                let color = if k == current { GOAL } else { NEXT_GOALS };
                gizmos.sphere(Isometry3d::from_translation(p), radius, color);
                if let Some(next) = agent.goals.get(k + 1) {
                    gizmos.line(p, convert::vec(next.position), NEXT_GOALS);
                }
            }
            if let Some(g) = agent.goals.get(current) {
                gizmos.line(convert::vec(pos), convert::vec(g.position), TO_GOAL);
            }
            let color = if i == sim.pilot { PILOT_TRAIL } else { TRAIL_COLOR };
            let points: Vec<Vec3> = match &sim.replay {
                Some(r) => r.trail(i, TRAIL).into_iter().map(convert::vec).collect(),
                None => overlay.trails[i].iter().map(|&(_, p)| convert::vec(p)).chain([convert::vec(pos)]).collect(),
            };
            gizmos.linestrip(points, color);
        }
    }

    if overlay.lidar
        && let Some(scan) = latest_scan(&sim, sim.pilot)
    {
        let max = scan.config.max_range;
        let hits: Vec<(DVec3, f64)> =
            scan.ranges.iter().zip(scan.directions).filter_map(|(r, d)| r.map(|r| (*d * r, r))).collect();
        let stride = hits.len().div_ceil(MAX_POINTS).max(1);
        for &(p, r) in hits.iter().step_by(stride) {
            let size = (0.012 * r).clamp(0.05, 0.6) as f32;
            let p = convert::vec(scan.pose.transform_point(p));
            gizmos.cross(Isometry3d::from_translation(p), size, range_color(r, max));
        }
    }
}
