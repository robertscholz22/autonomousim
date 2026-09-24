//! Write a `.tir` file with every Magic Formula coefficient as this crate reads it, so that
//! oracles with other defaults for missing keys evaluate the same tyre:
//! `cargo run -p autonomousim-vehicles --example tir_canonical -- <in.tir> <out.tir>`.

use autonomousim_vehicles::ground::tire::MfParams;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output] = args.as_slice() else {
        return Err("usage: tir_canonical <in.tir> <out.tir>".into());
    };
    std::fs::write(output, MfParams::read(input)?.to_tir())?;
    Ok(())
}
