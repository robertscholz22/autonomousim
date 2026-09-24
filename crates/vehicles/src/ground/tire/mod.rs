//! Tyre models: the Magic Formula (MF 5.2 / 6.1 from `.tir` files) and Fiala.

mod mf;
mod tir;

pub use mf::{MfInput, MfOutput, MfParams, MfVersion, REQUIRED};
pub use tir::{TirError, TirFile, TirValue};
