//! Surface materials shared by contacts, tyres (M2), sensors and the viewer.

use serde::{Deserialize, Serialize};

/// Index into a [`MaterialTable`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MaterialId(pub u8);

impl MaterialId {
    pub const ROCK: Self = Self(0);
    pub const SCREE: Self = Self(1);
    pub const SNOW: Self = Self(2);
    pub const SAND: Self = Self(3);
    pub const MUD: Self = Self(4);
    pub const GRASS: Self = Self(5);
    pub const FOREST_FLOOR: Self = Self(6);
    pub const WATER: Self = Self(7);
    pub const ASPHALT: Self = Self(8);
    pub const CONCRETE: Self = Self(9);
    pub const GRAVEL: Self = Self(10);
    pub const DIRT: Self = Self(11);
    pub const WOOD: Self = Self(12);
    pub const FOLIAGE: Self = Self(13);
    pub const METAL: Self = Self(14);
}

/// Physical and visual properties of a surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    /// Coulomb friction coefficient for generic (non-tyre) contacts.
    pub friction: f64,
    /// Rolling-resistance coefficient for pneumatic tyres (M2).
    pub rolling_resistance: f64,
    /// Scale on the contact natural frequency (< 1 for soft ground such as mud or snow).
    pub stiffness_scale: f64,
    /// Diffuse reflectivity in [0, 1] seen by LiDAR.
    pub reflectivity: f64,
    /// Linear sRGB albedo used by the viewer.
    pub color: [u8; 3],
}

impl Material {
    fn new(
        name: &str,
        friction: f64,
        rolling_resistance: f64,
        stiffness_scale: f64,
        reflectivity: f64,
        color: [u8; 3],
    ) -> Self {
        Self { name: name.to_owned(), friction, rolling_resistance, stiffness_scale, reflectivity, color }
    }
}

/// Material properties indexed by [`MaterialId`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialTable {
    materials: Vec<Material>,
}

impl MaterialTable {
    /// The built-in table; indices match the `MaterialId` constants.
    pub fn standard() -> Self {
        let m = Material::new;
        Self {
            materials: vec![
                m("rock", 0.7, 0.015, 1.0, 0.35, [120, 116, 110]),
                m("scree", 0.55, 0.04, 0.8, 0.3, [145, 138, 128]),
                m("snow", 0.25, 0.06, 0.4, 0.8, [235, 238, 242]),
                m("sand", 0.5, 0.15, 0.5, 0.45, [214, 196, 150]),
                m("mud", 0.35, 0.12, 0.3, 0.15, [96, 78, 58]),
                m("grass", 0.45, 0.05, 0.7, 0.4, [92, 128, 60]),
                m("forest_floor", 0.55, 0.06, 0.6, 0.25, [78, 70, 48]),
                m("water", 0.05, 0.3, 0.1, 0.05, [52, 92, 130]),
                m("asphalt", 0.8, 0.013, 1.0, 0.1, [60, 60, 64]),
                m("concrete", 0.75, 0.012, 1.0, 0.4, [170, 168, 160]),
                m("gravel", 0.6, 0.02, 0.8, 0.35, [150, 142, 130]),
                m("dirt", 0.6, 0.03, 0.7, 0.25, [126, 100, 72]),
                m("wood", 0.5, 0.02, 1.0, 0.3, [104, 78, 52]),
                // Softness comes from the foliage contact class, not from the material.
                m("foliage", 0.3, 0.1, 1.0, 0.5, [60, 110, 50]),
                m("metal", 0.4, 0.01, 1.0, 0.6, [180, 182, 188]),
            ],
        }
    }

    #[inline]
    pub fn get(&self, id: MaterialId) -> &Material {
        &self.materials[id.0 as usize]
    }

    pub fn find(&self, name: &str) -> Option<MaterialId> {
        self.materials.iter().position(|m| m.name == name).map(|i| MaterialId(i as u8))
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Add a material and return its id.
    pub fn push(&mut self, material: Material) -> MaterialId {
        assert!(self.materials.len() < 256, "at most 256 materials");
        self.materials.push(material);
        MaterialId((self.materials.len() - 1) as u8)
    }
}

impl Default for MaterialTable {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_ids_match_names() {
        let t = MaterialTable::standard();
        for (id, name) in [
            (MaterialId::ROCK, "rock"),
            (MaterialId::WATER, "water"),
            (MaterialId::FOREST_FLOOR, "forest_floor"),
            (MaterialId::FOLIAGE, "foliage"),
            (MaterialId::METAL, "metal"),
        ] {
            assert_eq!(t.get(id).name, name);
            assert_eq!(t.find(name), Some(id));
        }
    }
}
