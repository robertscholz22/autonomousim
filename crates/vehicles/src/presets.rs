//! Built-in vehicle and trailer definitions (the TOML files in `assets/vehicles`, embedded at
//! compile time).

use crate::{VehicleDef, VehicleError};

const PRESETS: &[(&str, &str)] = &[
    ("cf2x", include_str!("../../../assets/vehicles/cf2x.toml")),
    ("iris_like", include_str!("../../../assets/vehicles/iris_like.toml")),
    ("sedan_like", include_str!("../../../assets/vehicles/sedan_like.toml")),
    ("offroad_4x4", include_str!("../../../assets/vehicles/offroad_4x4.toml")),
    ("rover_diff", include_str!("../../../assets/vehicles/rover_diff.toml")),
    ("rover_skid", include_str!("../../../assets/vehicles/rover_skid.toml")),
    ("truck_6x4", include_str!("../../../assets/vehicles/truck_6x4.toml")),
    ("truck_8x8", include_str!("../../../assets/vehicles/truck_8x8.toml")),
    ("farm_tractor", include_str!("../../../assets/vehicles/farm_tractor.toml")),
];

const TRAILERS: &[(&str, &str)] = &[
    ("semitrailer_3axle", include_str!("../../../assets/vehicles/semitrailer_3axle.toml")),
    ("farm_trailer", include_str!("../../../assets/vehicles/farm_trailer.toml")),
];

/// Names of the built-in presets.
pub fn names() -> impl Iterator<Item = &'static str> {
    PRESETS.iter().map(|(n, _)| *n)
}

/// Names of the built-in trailers.
pub fn trailer_names() -> impl Iterator<Item = &'static str> {
    TRAILERS.iter().map(|(n, _)| *n)
}

/// A built-in trailer by name.
pub fn trailer(name: &str) -> Result<crate::ground::TrailerDef, VehicleError> {
    let (_, src) = TRAILERS.iter().find(|(n, _)| *n == name).ok_or_else(|| VehicleError::UnknownPreset(name.into()))?;
    crate::ground::TrailerDef::from_toml(src)
}

/// Load a built-in preset by name.
pub fn get(name: &str) -> Result<VehicleDef, VehicleError> {
    let (_, src) = PRESETS.iter().find(|(n, _)| *n == name).ok_or_else(|| VehicleError::UnknownPreset(name.into()))?;
    VehicleDef::from_toml(src)
}

/// A built-in multirotor preset.
pub fn multirotor(name: &str) -> Result<crate::multirotor::MultirotorDef, VehicleError> {
    match get(name)? {
        VehicleDef::Multirotor(m) => Ok(m),
        _ => Err(VehicleError::Invalid(format!("{name} is not a multirotor"))),
    }
}

/// A built-in wheeled-vehicle preset.
pub fn wheeled(name: &str) -> Result<crate::ground::WheeledDef, VehicleError> {
    match get(name)? {
        VehicleDef::Wheeled(w) => Ok(w),
        _ => Err(VehicleError::Invalid(format!("{name} is not a wheeled vehicle"))),
    }
}
