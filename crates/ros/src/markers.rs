//! What the bridge shows as `visualization_msgs/MarkerArray`s (frame `map`): scripted agents
//! (NPCs) as boxes, pedestrians as cylinders, traffic signal heads as spheres in the colour
//! of their light; and agents' routes as `nav_msgs/Path`.

use crate::bridge::{header, quaternion, vector};
use crate::msgs::geometry_msgs::{Point, Pose, PoseStamped};
use crate::msgs::nav_msgs::Path;
use crate::msgs::std_msgs::ColorRGBA;
use crate::msgs::visualization_msgs::{Marker, MarkerArray};
use autonomousim_scene::streets::{SignalHead, signal_heads};
use autonomousim_sim::WorldInstance;
use autonomousim_sim::pedestrians::PedState;
use autonomousim_world::Light;
use glam::{DQuat, DVec3};

fn color(r: f32, g: f32, b: f32) -> ColorRGBA {
    ColorRGBA { r, g, b, a: 1.0 }
}

fn point(p: DVec3) -> Point {
    Point { x: p.x, y: p.y, z: p.z }
}

fn marker(
    t: f64,
    ns: &str,
    id: i32,
    kind: i32,
    position: DVec3,
    rotation: DQuat,
    scale: DVec3,
    c: ColorRGBA,
) -> Marker {
    Marker {
        header: header(t, "map"),
        ns: ns.to_string(),
        id,
        kind,
        action: Marker::ADD,
        pose: Pose { position: point(position), orientation: quaternion(rotation) },
        scale: vector(scale),
        color: c,
        frame_locked: false,
        ..Default::default()
    }
}

/// The box of an agent's collision spheres in its body frame: centre and size (m).
pub fn body_box(world: &WorldInstance, agent: usize) -> (DVec3, DVec3) {
    let pose = world.agent(agent).vehicle.pose();
    let inv = pose.rot.inverse();
    let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    for s in &world.shapes()[agent].spheres {
        let c = inv * (s.center - pose.pos);
        lo = lo.min(c - s.radius);
        hi = hi.max(c + s.radius);
    }
    if lo.x > hi.x {
        return (DVec3::ZERO, DVec3::splat(1.0));
    }
    (0.5 * (lo + hi), hi - lo)
}

/// A scripted agent (NPC) at `pose` as its body box (`centre`, `size` in the body frame);
/// a disabled one is deleted.
pub fn npc(
    t: f64,
    id: u32,
    position: DVec3,
    rotation: DQuat,
    (centre, size): (DVec3, DVec3),
    disabled: bool,
) -> Marker {
    let mut m = marker(
        t,
        "npcs",
        id as i32,
        Marker::CUBE,
        position + rotation * centre,
        rotation,
        size,
        color(0.35, 0.45, 0.6),
    );
    if disabled {
        m.action = Marker::DELETE;
    }
    m
}

/// The scripted agents (`npcs`: index in the world and body box) as boxes; disabled ones are
/// deleted.
pub fn npcs(world: &WorldInstance, t: f64, npcs: &[(usize, (DVec3, DVec3))]) -> MarkerArray {
    let markers = npcs
        .iter()
        .map(|&(i, b)| {
            let a = world.agent(i);
            let pose = a.vehicle.pose();
            npc(t, a.id, pose.pos, pose.rot, b, a.disabled)
        })
        .collect();
    MarkerArray { markers }
}

/// Pedestrian `k` (feet at `position`) as a cylinder, coloured by what it does.
pub fn pedestrian(
    t: f64,
    k: usize,
    position: DVec3,
    heading: f64,
    radius: f64,
    height: f64,
    state: PedState,
) -> Marker {
    let c = match state {
        PedState::Hit => color(0.9, 0.1, 0.1),
        PedState::Crossing => color(0.95, 0.75, 0.2),
        _ => color(0.55, 0.32, 0.25),
    };
    let size = DVec3::new(2.0 * radius, 2.0 * radius, height);
    let centre = position + DVec3::Z * (0.5 * height);
    marker(t, "pedestrians", k as i32, Marker::CYLINDER, centre, DQuat::from_rotation_z(heading), size, c)
}

/// The pedestrians as cylinders, coloured by what they do.
pub fn pedestrians(world: &WorldInstance, t: f64) -> MarkerArray {
    let markers = world
        .crowd()
        .peds
        .iter()
        .enumerate()
        .map(|(k, p)| pedestrian(t, k, DVec3::new(p.pos.x, p.pos.y, p.z), p.heading, p.radius, p.height, p.state))
        .collect();
    MarkerArray { markers }
}

/// The signal heads of the world's current map.
pub fn heads(world: &WorldInstance) -> Vec<SignalHead> {
    if world.signals().is_empty() { Vec::new() } else { signal_heads(world.map()) }
}

/// Signal head `k` as a sphere on its face towards the traffic, in the colour of `light`.
pub fn signal(t: f64, k: usize, head: &SignalHead, light: Light) -> Marker {
    let c = match light {
        Light::Green => color(0.1, 0.85, 0.2),
        Light::Amber => color(1.0, 0.65, 0.0),
        Light::Red => color(0.95, 0.1, 0.1),
    };
    let face = head.position + head.rotation * DVec3::new(head.half_extents.x + 0.1, 0.0, 0.0);
    marker(t, "signals", k as i32, Marker::SPHERE, face, DQuat::IDENTITY, DVec3::splat(0.4), c)
}

/// The signal heads as spheres on the face towards the traffic, in the colour of their light.
pub fn signals(world: &WorldInstance, t: f64, heads: &[SignalHead]) -> MarkerArray {
    let lanes = world.map().roads().lanes();
    let markers = heads
        .iter()
        .enumerate()
        .map(|(k, h)| signal(t, k, h, world.signals().light(lanes, h.connector, world.time())))
        .collect();
    MarkerArray { markers }
}

/// A path (frame `map`) through `points`.
pub fn path(t: f64, points: &[DVec3]) -> Path {
    let poses = points
        .iter()
        .map(|&p| PoseStamped {
            header: header(t, "map"),
            pose: Pose { position: point(p), orientation: Default::default() },
        })
        .collect();
    Path { header: header(t, "map"), poses }
}

/// Agent `agent`'s route (the lane points of a `route` goal, or the planned path of `path`
/// goals), if it has one.
pub fn route(world: &WorldInstance, agent: usize, t: f64) -> Option<Path> {
    let a = world.agent(agent);
    let points: Vec<DVec3> = if !a.legs.is_empty() {
        // Each leg starts where the previous one ends.
        a.legs.iter().enumerate().flat_map(|(k, l)| l.points()[usize::from(k > 0)..].iter().copied()).collect()
    } else {
        a.route.as_ref()?.points().to_vec()
    };
    Some(path(t, &points))
}
