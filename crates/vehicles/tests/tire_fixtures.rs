//! The Magic Formula against reference values from MFeval.jl (`make fixtures-mfeval`, see
//! `tools/gen_mfeval_fixtures.jl`): MF 5.2 and 6.1 sample tyres (camber, pressure) and the
//! PAC2002 files of the vehicle presets. The presets are also checked against Project Chrono's
//! ChPac02Tire (`make fixtures-chrono`).

use autonomousim_vehicles::ground::tire::{MfInput, MfParams};
use serde_json::Value;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Largest deviation of each output relative to its largest magnitude over all points.
fn compare(tir: &str, name: &str) {
    let params = MfParams::read(root().join(tir)).unwrap();
    let path = root().join(format!("fixtures/mfeval/{name}.json"));
    let data: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("run `make fixtures-mfeval`")).unwrap();
    let points = data["points"].as_array().unwrap();
    let fields =
        ["Fx", "Fy", "Mx", "My", "Mz", "Kxk", "Kya", "mux", "muy", "t", "Mzr", "sigmax", "sigmay", "Re", "two_a"];
    let mut worst = vec![(0.0f64, 0.0f64, 0usize); fields.len()];
    for (k, pt) in points.iter().enumerate() {
        let g = |f: &str| pt[f].as_f64().unwrap();
        let mut input = MfInput::new(g("fz"), g("kappa"), g("alpha"), g("gamma"), g("vx"));
        input.pressure = Some(g("pressure"));
        let out = params.eval(&input);
        let ours = [
            out.fx,
            out.fy,
            out.mx,
            out.my,
            out.mz,
            out.kxk,
            out.kya,
            out.mux,
            out.muy,
            out.trail,
            out.mzr,
            out.sigma_x,
            out.sigma_y,
            params.effective_radius(g("fz"), g("omega"), Some(g("pressure"))),
            2.0 * params.contact_half_length(g("fz"), Some(g("pressure"))),
        ];
        for (i, f) in fields.iter().enumerate() {
            let reference = g(f);
            let w = &mut worst[i];
            w.0 = w.0.max(reference.abs());
            if (ours[i] - reference).abs() > w.1 {
                *w = (w.0, (ours[i] - reference).abs(), k);
            }
        }
    }
    let mut failed = vec![];
    for (i, f) in fields.iter().enumerate() {
        let (scale, err, k) = worst[i];
        let rel = err / scale.max(1e-12);
        // MFeval takes cos α' as V_cx/(V_c + 1e-6), which moves the trail by ~1e-7.
        let tolerance = if ["t", "Mz"].contains(f) { 1e-6 } else { 1e-9 };
        // MFeval ignores LCZ (the vertical stiffness scale of PAC2002 files).
        let skip = params.lcz != 1.0 && ["Re", "two_a"].contains(f);
        println!(
            "{name} {f}: max |error| {err:.3e} ({rel:.1e} of peak), point {k}{}",
            if skip { " (skipped)" } else { "" }
        );
        if rel > tolerance && !skip {
            failed.push(format!("{f} (point {k}: {:?})", points[k]));
        }
    }
    assert!(failed.is_empty(), "{name}: {failed:#?}");
}

#[test]
fn mf52_sample_tyre_matches_mfeval() {
    compare("fixtures/tir/MagicFormula52_Parameters.tir", "MagicFormula52_Parameters");
}

#[test]
fn mf61_sample_tyre_matches_mfeval() {
    compare("fixtures/tir/MagicFormula61_Parameters.tir", "MagicFormula61_Parameters");
}

#[test]
fn preset_tyres_match_mfeval() {
    compare("assets/tires/HMMWV_Pac02Tire.tir", "HMMWV_Pac02Tire");
    compare("assets/tires/Sedan_Pac02Tire.tir", "Sedan_Pac02Tire");
}

/// The PAC2002 preset tyres against Project Chrono's ChPac02Tire (`make fixtures-chrono`, see
/// `tools/gen_chrono_fixtures.py` for Chrono's deviations): pure-slip F_x, F_y, M_z and
/// combined-slip F_x, F_y within 1 % of their peak, where Chrono does not clamp the curves.
fn compare_chrono(name: &str) {
    let params = MfParams::read(root().join(format!("assets/tires/{name}.tir"))).unwrap();
    let path = root().join(format!("fixtures/chrono/{name}.json"));
    let data: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("run `make fixtures-chrono`")).unwrap();
    let p = &params;
    let eval = |pt: &Value, kappa: Option<f64>| {
        let g = |f: &str| pt[f].as_f64().unwrap();
        let alpha = g("alpha");
        let input = MfInput {
            fz: g("fz"),
            kappa: kappa.unwrap_or(g("kappa")),
            alpha,
            cos_alpha: alpha.cos(),
            gamma: 0.0,
            gamma_star: 0.0,
            vx: 15.0,
            pressure: None,
            mu_scale: 1.0,
        };
        let out = p.eval(&input);
        // Chrono clamps B·x to ±(π/2 − 0.01) in both pure-slip curves.
        let fz0p = p.fnomin * p.lfzo;
        let dfz = (input.fz - fz0p) / fz0p;
        let bx = out.kxk / (p.pcx1 * p.lcx * out.mux * input.fz) * (input.kappa + (p.phx1 + p.phx2 * dfz) * p.lhx);
        let by = out.kya / (p.pcy1 * p.lcy * out.muy * input.fz) * (alpha + (p.phy1 + p.phy2 * dfz) * p.lhy);
        let clamped = bx.abs() > std::f64::consts::FRAC_PI_2 - 0.02 || by.abs() > std::f64::consts::FRAC_PI_2 - 0.02;
        (out, clamped)
    };
    type Quantity<'a> = (&'a str, &'a str, Box<dyn Fn(&Value) -> Option<f64> + 'a>);
    let checks: Vec<Quantity> = vec![
        ("pure_kappa", "Fx", Box::new(|pt| Some(eval(pt, None)).filter(|e| !e.1).map(|e| e.0.fx))),
        ("pure_alpha", "Fy", Box::new(|pt| Some(eval(pt, Some(0.0))).filter(|e| !e.1).map(|e| e.0.fy))),
        (
            "pure_alpha",
            "Mz",
            // Chrono's M_z0 = −t F_y0 + M_zr without the factor LFZO on the trail.
            Box::new(|pt| Some(eval(pt, Some(0.0))).filter(|e| !e.1).map(|(o, _)| -o.trail / p.lfzo * o.fy + o.mzr)),
        ),
        ("combined", "Fx", Box::new(|pt| Some(eval(pt, None)).filter(|e| !e.1).map(|e| e.0.fx))),
        ("combined", "Fy", Box::new(|pt| Some(eval(pt, None)).filter(|e| !e.1).map(|e| e.0.fy))),
    ];
    for (set, field, ours) in checks {
        let points = data[set].as_array().unwrap();
        let peak = points.iter().map(|pt| pt[field].as_f64().unwrap().abs()).fold(0.0, f64::max);
        let (mut worst, mut used) = (0.0f64, 0);
        for pt in points {
            if let Some(v) = ours(pt) {
                used += 1;
                worst = worst.max((v - pt[field].as_f64().unwrap()).abs() / peak);
            }
        }
        println!("{name} {set} {field}: {used} of {} points, max error {:.2e} of peak", points.len(), worst);
        // The κ sweep reaches the clamp at a few percent slip (about a dozen points per load).
        assert!(used >= 30, "{name} {set} {field}: too few points below the clamp");
        // The M2 acceptance bound is 1 % of peak; the remaining differences (~1e-5, Chrono's +0.1
        // in the stiffness denominators) are far below it, so a tighter bound guards regressions.
        assert!(worst < 1e-4, "{name} {set} {field}: {worst:.3e} of peak");
    }
}

#[test]
fn preset_tyres_match_chrono_pac02() {
    compare_chrono("HMMWV_Pac02Tire");
    compare_chrono("Sedan_Pac02Tire");
    compare_chrono("Truck_Pac02Tire");
}
