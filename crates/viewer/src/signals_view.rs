//! Signal heads: the lamps of every head of the map, lit by the simulation's signal state
//! (live, or recomputed from the recorded offsets in replays).

use crate::convert::{self, RenderOrigin};
use crate::sim::Sim;
use crate::world_view::{Anchor, MapEntity};
use autonomousim_core::math::Pose;
use autonomousim_scene::streets::{self, Lamp, SignalHead};
use autonomousim_world::{Light, StaticWorld};
use bevy::prelude::*;
use std::sync::Arc;

/// The lamps of one head.
#[derive(Component)]
pub struct HeadLamps {
    connector: u32,
    /// Index into [`SignalLamps::meshes`].
    shape: usize,
    lit: Lamp,
}

/// Lamp meshes (dark, red, amber, green) per head size, and the map they were made for.
#[derive(Resource, Default)]
pub struct SignalLamps {
    map: Option<Arc<StaticWorld>>,
    meshes: Vec<(glam::DVec3, [Handle<Mesh>; 4])>,
    material: Option<Handle<StandardMaterial>>,
}

#[cfg(test)]
impl HeadLamps {
    pub fn connector(&self) -> u32 {
        self.connector
    }

    pub fn lit(&self) -> Lamp {
        self.lit
    }
}

fn index(l: Lamp) -> usize {
    match l {
        Lamp::Dark => 0,
        Lamp::Red => 1,
        Lamp::Amber => 2,
        Lamp::Green => 3,
    }
}

/// The lamp lit for `light`.
pub fn lamp(light: Light) -> Lamp {
    match light {
        Light::Green => Lamp::Green,
        Light::Amber => Lamp::Amber,
        Light::Red => Lamp::Red,
    }
}

/// Spawn the lamps of a new map, and light them by the signal state at the simulation time.
pub fn sync_signals(
    mut commands: Commands,
    sim: Res<Sim>,
    origin: Res<RenderOrigin>,
    mut lamps: ResMut<SignalLamps>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut heads: Query<(&mut HeadLamps, &mut Mesh3d)>,
) {
    let map = sim.world.map();
    if !lamps.map.as_ref().is_some_and(|m| Arc::ptr_eq(m, map)) {
        // The old lamps are map entities: `sync_map` despawns them with the map.
        lamps.map = Some(map.clone());
        lamps.meshes.clear();
        let material = lamps
            .material
            .get_or_insert_with(|| {
                materials.add(StandardMaterial { base_color: Color::WHITE, unlit: true, ..default() })
            })
            .clone();
        for head in streets::signal_heads(map) {
            let SignalHead { position, rotation, half_extents, connector } = head;
            let shape = match lamps.meshes.iter().position(|(h, _)| *h == half_extents) {
                Some(k) => k,
                None => {
                    let set = [Lamp::Dark, Lamp::Red, Lamp::Amber, Lamp::Green]
                        .map(|l| meshes.add(convert::mesh(&streets::signal_lamps(half_extents, l))));
                    lamps.meshes.push((half_extents, set));
                    lamps.meshes.len() - 1
                }
            };
            commands.spawn((
                HeadLamps { connector, shape, lit: Lamp::Dark },
                Mesh3d(lamps.meshes[shape].1[0].clone()),
                MeshMaterial3d(material.clone()),
                bevy::light::NotShadowCaster,
                MapEntity,
                Anchor(position),
                origin.transform(&Pose::new(position, rotation)),
            ));
        }
        return;
    }
    let lanes = map.roads().lanes();
    let t = sim.time();
    for (mut head, mut mesh) in &mut heads {
        let lit = lamp(sim.world.signals().light(lanes, head.connector, t));
        if lit != head.lit {
            head.lit = lit;
            mesh.0 = lamps.meshes[head.shape].1[index(lit)].clone();
        }
    }
}
