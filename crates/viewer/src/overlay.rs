//! Markers drawn over the scene with gizmos: the goals of every agent and the line to the
//! current one, the route of agents with `route` goals, the path flown over the last seconds
//! (scripted NPCs get a ring instead: grey while kinematic, orange in full physics), the hits of the latest LiDAR scan, and on urban maps the lane graph near the camera (lanes,
//! connectors in their signals' colours, yield points and stop lines).
//! O cycles through goals and trails, those with the lane graph, and nothing; L toggles the
//! LiDAR points.

use crate::CameraRig;
use crate::convert::RenderOrigin;
use crate::sim::Sim;
use autonomousim_core::math::Pose;
use autonomousim_sensors::{LidarConfig, Sensor};
use autonomousim_sim::scenario::GoalKind;
use autonomousim_world::lanes::Control;
use autonomousim_world::{Light, Polyline, StaticWorld, Turn};
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
    /// The lane graph (with the markers).
    pub lanes: bool,
    pub lidar: bool,
    /// Live trails: (time, position) of every agent.
    trails: Vec<VecDeque<(f64, DVec3)>>,
    map: Option<Arc<StaticWorld>>,
}

impl Default for Overlay {
    fn default() -> Self {
        Self { markers: true, lanes: false, lidar: true, trails: Vec::new(), map: None }
    }
}

const GOAL: Color = Color::srgb(1.0, 0.8, 0.1);
const NEXT_GOALS: Color = Color::srgba(1.0, 0.8, 0.1, 0.4);
const TO_GOAL: Color = Color::srgba(1.0, 0.8, 0.1, 0.6);
const ROUTE: Color = Color::srgba(1.0, 0.45, 0.1, 0.8);
const TRAIL_COLOR: Color = Color::srgb(0.2, 0.9, 1.0);
const PILOT_TRAIL: Color = Color::srgb(1.0, 0.35, 0.8);

/// Scripted NPCs: a ring over those driven kinematically, a brighter one over those promoted
/// to full physics.
const NPC_KINEMATIC: Color = Color::srgba(0.6, 0.65, 0.7, 0.7);
const NPC_PROMOTED: Color = Color::srgb(1.0, 0.3, 0.1);

/// Whether scripted agent `agent` is driven kinematically (live, or as recorded); `None` for
/// the agents not scripted.
pub fn npc_kinematic(sim: &Sim, agent: usize) -> Option<bool> {
    let a = sim.world.agent(agent);
    a.driver.as_ref()?;
    Some(match &sim.replay {
        Some(r) => r.sample(agent).is_some_and(|s| s.last.kinematic),
        None => a.is_kinematic(),
    })
}

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

pub fn draw(
    keys: Res<ButtonInput<KeyCode>>,
    sim: Res<Sim>,
    origin: Res<RenderOrigin>,
    camera: Query<&CameraRig>,
    mut overlay: ResMut<Overlay>,
    mut gizmos: Gizmos,
) {
    if keys.just_pressed(KeyCode::KeyO) {
        (overlay.markers, overlay.lanes) = match (overlay.markers, overlay.lanes) {
            (true, false) => (true, true),
            (true, true) => (false, false),
            _ => (true, false),
        };
    }
    if overlay.lanes
        && let Ok(rig) = camera.single()
    {
        draw_lanes(&sim, &origin, rig.eye, &mut gizmos);
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
            if sim.world.agent(i).driver.is_some() {
                continue;
            }
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
            let group = &sim.world.scenario().groups[agent.group];
            // Scripted NPCs get a ring only (no goals, route or trail; streets are full of them).
            if let Some(kinematic) = npc_kinematic(&sim, i) {
                let lift = DVec3::Z * (2.0 * group.radius / 1.5).max(1.0);
                let ring = Isometry3d::new(origin.pos(pos + lift), Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
                let color = if kinematic { NPC_KINEMATIC } else { NPC_PROMOTED };
                gizmos.circle(ring, 0.5, color);
                continue;
            }
            // Goals: the current one solid, the ones after it faint and joined up; as large as
            // the vehicle.
            let arm = match agent.vehicle.as_multirotor() {
                Some(m) => m.def().rotors.iter().map(|r| r.position.length()).fold(0.0, f64::max),
                None => group.radius / 1.5,
            };
            // A bay is marked by the tail's success area (1 m) and its heading.
            let bay = group.spec.goals.kind == GoalKind::Bay;
            let radius = if bay { 1.0 } else { (1.5 * arm).max(0.05) as f32 };
            let current = agent.goal_index.min(agent.goals.len().saturating_sub(1));
            for (k, g) in agent.goals.iter().enumerate().skip(current) {
                let p = origin.pos(g.position);
                let color = if k == current { GOAL } else { NEXT_GOALS };
                gizmos.sphere(Isometry3d::from_translation(p), radius, color);
                if bay {
                    let ahead = g.position + 4.0 * DVec3::new(g.yaw.cos(), g.yaw.sin(), 0.0);
                    gizmos.arrow(p, origin.pos(ahead), color);
                }
                if let Some(next) = agent.goals.get(k + 1) {
                    gizmos.line(p, origin.pos(next.position), NEXT_GOALS);
                }
            }
            if let Some(g) = agent.goals.get(current) {
                gizmos.line(origin.pos(pos), origin.pos(g.position), TO_GOAL);
            }
            // The route's lane, half a metre above the road.
            if let Some(route) = &agent.route {
                let pts = route.points();
                let last = pts.len().saturating_sub(1);
                let lane = (0..pts.len()).step_by(2).chain([last]).map(|k| origin.pos(pts[k] + DVec3::Z * 0.5));
                gizmos.linestrip(lane, ROUTE);
            }
            let color = if i == sim.pilot { PILOT_TRAIL } else { TRAIL_COLOR };
            let points: Vec<Vec3> = match &sim.replay {
                Some(r) => r.trail(i, TRAIL).into_iter().map(|p| origin.pos(p)).collect(),
                None => overlay.trails[i].iter().map(|&(_, p)| origin.pos(p)).chain([origin.pos(pos)]).collect(),
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
            let p = origin.pos(scan.pose.transform_point(p));
            gizmos.cross(Isometry3d::from_translation(p), size, range_color(r, max));
        }
    }
}

/// Lanes and connectors are drawn within this distance of the camera (m).
const LANE_RANGE: f64 = 150.0;
const LANE: Color = Color::srgba(0.35, 0.7, 1.0, 0.8);
const CONNECTOR: Color = Color::srgba(0.85, 0.85, 0.85, 0.6);
const CONFLICT: Color = Color::srgb(1.0, 0.45, 0.1);

fn light_color(l: Light) -> Color {
    match l {
        Light::Green => Color::srgb(0.1, 1.0, 0.3),
        Light::Amber => Color::srgb(1.0, 0.7, 0.05),
        Light::Red => Color::srgb(1.0, 0.1, 0.1),
    }
}

/// The lane graph near `eye`: lanes (blue, with their direction), connectors (grey, or in
/// their signal's colour), the points where a connector yields (orange crosses), and stop lines
/// in the light of the movement straight on.
fn draw_lanes(sim: &Sim, origin: &RenderOrigin, eye: DVec3, gizmos: &mut Gizmos) {
    let net = sim.world.map().roads();
    if !net.has_sections() {
        return;
    }
    let g = net.lanes();
    let t = sim.time();
    let signals = sim.world.signals();
    let lift = DVec3::Z * 0.3;
    let near = |line: &Polyline| {
        let p = line.points();
        [p[0], p[p.len() / 2], p[p.len() - 1]].iter().any(|q| q.truncate().distance(eye.truncate()) < LANE_RANGE)
    };
    let strip = |line: &Polyline| {
        let p = line.points();
        let last = p.len() - 1;
        (0..p.len()).step_by(2).chain([last]).map(|k| origin.pos(p[k] + lift)).collect::<Vec<_>>()
    };
    for l in g.lanes() {
        if !near(&l.line) {
            continue;
        }
        gizmos.linestrip(strip(&l.line), LANE);
        let len = l.line.length();
        let tip = l.line.point_at(len);
        gizmos.arrow(origin.pos(l.line.point_at((len - 3.0).max(0.0)) + lift), origin.pos(tip + lift), LANE);
    }
    for (i, c) in g.connectors().iter().enumerate() {
        if !near(&c.line) {
            continue;
        }
        let color = match g.connector_signal(i as u32) {
            Some(_) => light_color(signals.light(g, i as u32, t)),
            None => CONNECTOR,
        };
        gizmos.linestrip(strip(&c.line), color);
        for k in c.conflicts.iter().filter(|k| k.yields) {
            let p = origin.pos(c.line.point_at(k.station + 0.5 * k.length) + lift);
            gizmos.cross(Isometry3d::from_translation(p), 0.4, CONFLICT);
        }
    }
    for j in g.junctions() {
        for a in &j.approaches {
            let mid = 0.5 * (a.stop_line[0] + a.stop_line[1]);
            if mid.truncate().distance(eye.truncate()) > LANE_RANGE || a.control != Control::Signal {
                continue;
            }
            let straight = a
                .lanes
                .iter()
                .flat_map(|&l| &g.lanes()[l as usize].successors)
                .copied()
                .find(|&c| g.connectors()[c as usize].turn == Turn::Straight)
                .or_else(|| a.lanes.first().and_then(|&l| g.lanes()[l as usize].successors.first().copied()));
            let Some(c) = straight else { continue };
            let color = light_color(signals.light(g, c, t));
            gizmos.line(origin.pos(a.stop_line[0] + lift), origin.pos(a.stop_line[1] + lift), color);
        }
    }
}
