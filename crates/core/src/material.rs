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
    /// Farmland, in [`MaterialTable::rural`] only.
    pub const MEADOW: Self = Self(15);
    pub const CROP: Self = Self(16);
    /// Freshly plowed soil: loose and soft.
    pub const PLOWED: Self = Self(17);
}

/// Bekker–Wong parameters of a deformable soil (Wong, *Theory of Ground Vehicles*, 4th ed.,
/// §2.2–2.4): the pressure–sinkage law `p = (k_c/b + k_φ)·z^n` of a plate of width `b`, and the
/// shear strength `τ_max = c + p·tan φ` reached over the shear displacement `K`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Soil {
    /// Exponent `n`, cohesive modulus `k_c` (N/m^(n+1)) and frictional modulus `k_φ`
    /// (N/m^(n+2)) of the pressure–sinkage law.
    pub n: f64,
    pub k_c: f64,
    pub k_phi: f64,
    /// Cohesion `c` (Pa) and angle of internal friction `φ` (rad).
    pub cohesion: f64,
    pub friction_angle: f64,
    /// Shear deformation modulus `K` (m).
    pub shear_modulus: f64,
    /// Bulk density (kg/m³), for the bulldozing resistance.
    pub density: f64,
}

impl Soil {
    /// Wong's Table 2.3 values (the moduli in kN) and a shear modulus and density per soil.
    fn table(n: f64, k_c: f64, k_phi: f64, cohesion_kpa: f64, phi_deg: f64, shear_modulus: f64, density: f64) -> Self {
        Self {
            n,
            k_c: k_c * 1e3,
            k_phi: k_phi * 1e3,
            cohesion: cohesion_kpa * 1e3,
            friction_angle: phi_deg.to_radians(),
            shear_modulus,
            density,
        }
    }

    /// The soil of a built-in material, by name.
    pub fn of_material(name: &str) -> Option<Self> {
        let t = Self::table;
        match name {
            // Dry sand (Land Locomotion Laboratory).
            "sand" => Some(t(1.1, 0.99, 1528.43, 1.04, 28.0, 0.025, 1600.0)),
            // Clayey soil (Thailand), remoulded: weak in friction.
            "mud" => Some(t(0.5, 13.19, 692.15, 4.14, 13.0, 0.025, 1800.0)),
            // Snow (U.S.).
            "snow" => Some(t(1.6, 4.37, 196.72, 1.03, 19.7, 0.04, 300.0)),
            // Grenville loam.
            "meadow" => Some(t(1.01, 0.06, 5880.0, 3.1, 29.8, 0.01, 1500.0)),
            // Rubicon sandy loam.
            "crop" => Some(t(0.66, 6.9, 752.0, 3.7, 30.2, 0.015, 1500.0)),
            // Upland sandy loam, loosened.
            "plowed" => Some(t(1.1, 74.6, 2080.0, 3.3, 33.7, 0.025, 1300.0)),
            _ => None,
        }
    }

    /// The sinkage modulus `k_c/b + k_φ` of a plate of width `b` (m).
    #[inline]
    pub fn modulus(&self, b: f64) -> f64 {
        self.k_c / b + self.k_phi
    }

    /// Sinkage (m) of a plate of width `b` under pressure `p` (Pa).
    pub fn sinkage(&self, p: f64, b: f64) -> f64 {
        (p.max(0.0) / self.modulus(b)).powf(1.0 / self.n)
    }

    /// Work (J/m, a force) of compacting a rut of width `b` from depth `z0` to `z`:
    /// `b·(k_c/b + k_φ)·(z^(n+1) − z0^(n+1))/(n+1)`, the compaction resistance.
    pub fn compaction(&self, b: f64, z0: f64, z: f64) -> f64 {
        let e = self.n + 1.0;
        b * self.modulus(b) * (z.max(0.0).powf(e) - z0.max(0.0).powf(e)).max(0.0) / e
    }

    /// Bulldozing resistance (N) of a blade of width `b` pushing soil to depth `z` under
    /// gravity `g`: Rankine's passive earth pressure `b·(2c·z·√K_p + ½ρg·z²·K_p)`,
    /// `K_p = tan²(45° + φ/2)`.
    pub fn bulldozing(&self, b: f64, z: f64, g: f64) -> f64 {
        let z = z.max(0.0);
        let kp = (std::f64::consts::FRAC_PI_4 + 0.5 * self.friction_angle).tan().powi(2);
        b * (2.0 * self.cohesion * z * kp.sqrt() + 0.5 * self.density * g * z * z * kp)
    }
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
    /// Deformable soil under tracks (`None`: rigid). A built-in property of the material's
    /// name ([`Soil::of_material`]), left out of map files and their content hashes.
    #[serde(skip)]
    pub soil: Option<Soil>,
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
        Self {
            name: name.to_owned(),
            friction,
            rolling_resistance,
            stiffness_scale,
            reflectivity,
            color,
            soil: Soil::of_material(name),
        }
    }
}

/// Material properties indexed by [`MaterialId`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "StoredTable")]
pub struct MaterialTable {
    materials: Vec<Material>,
}

/// A table as stored, without the soils; they are restored by name.
#[derive(Deserialize)]
struct StoredTable {
    materials: Vec<Material>,
}

impl From<StoredTable> for MaterialTable {
    fn from(t: StoredTable) -> Self {
        let materials = t.materials.into_iter().map(|m| Material { soil: Soil::of_material(&m.name), ..m }).collect();
        Self { materials }
    }
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

    /// The standard table plus farmland (meadow, crop, plowed soil), for rural maps. The
    /// standard table stays as it is so that the content hashes of other maps do not change.
    pub fn rural() -> Self {
        let mut t = Self::standard();
        let m = Material::new;
        t.push(m("meadow", 0.45, 0.06, 0.6, 0.4, [118, 150, 72]));
        t.push(m("crop", 0.45, 0.08, 0.5, 0.45, [196, 176, 92]));
        t.push(m("plowed", 0.5, 0.14, 0.3, 0.2, [112, 86, 62]));
        t
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
        let r = MaterialTable::rural();
        assert_eq!(r.len(), t.len() + 3);
        for (id, name) in [(MaterialId::MEADOW, "meadow"), (MaterialId::CROP, "crop"), (MaterialId::PLOWED, "plowed")] {
            assert_eq!(r.find(name), Some(id));
        }
    }

    #[test]
    fn soft_materials_have_soils_that_survive_serialization() {
        let r = MaterialTable::rural();
        let soft = ["snow", "sand", "mud", "meadow", "crop", "plowed"];
        for i in 0..r.len() {
            let m = r.get(MaterialId(i as u8));
            assert_eq!(m.soil.is_some(), soft.contains(&m.name.as_str()), "{}", m.name);
        }
        let back: MaterialTable = postcard::from_bytes(&postcard::to_allocvec(&r).unwrap()).unwrap();
        assert_eq!(back, r);
        // Soils are not stored.
        let mut rigid = r.clone();
        rigid.materials.iter_mut().for_each(|m| m.soil = None);
        assert_eq!(postcard::to_allocvec(&rigid).unwrap(), postcard::to_allocvec(&r).unwrap());
    }

    #[test]
    fn soil_laws() {
        let s = Soil::of_material("plowed").unwrap();
        let (b, p) = (0.38, 70e3);
        let z = s.sinkage(p, b);
        assert!((s.modulus(b) * z.powf(s.n) / p - 1.0).abs() < 1e-12);
        // The compaction work is the integral of the pressure over the sinkage.
        let steps = 100_000;
        let dz = z / steps as f64;
        let work: f64 = (0..steps).map(|k| b * s.modulus(b) * ((k as f64 + 0.5) * dz).powf(s.n) * dz).sum();
        assert!((s.compaction(b, 0.0, z) / work - 1.0).abs() < 1e-6);
        assert!((s.compaction(b, 0.5 * z, z) + s.compaction(b, 0.0, 0.5 * z) - work).abs() < 1e-6 * work);
        assert!(s.bulldozing(b, 0.0, 9.81) == 0.0 && s.bulldozing(b, 0.1, 9.81) > 0.0);
    }
}
