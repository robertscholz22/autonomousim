//! Reader for tyre property files (`.tir`, the ADAMS/TNO "TeimOrbit" format): `[SECTION]`
//! headers, `KEY = value $comment` lines, `$`/`!` comment lines and quoted strings. Tables
//! such as `[SHAPE]` (lines without `=`) are skipped. Only SI units are accepted.

use std::path::Path;

/// A value of a `.tir` entry.
#[derive(Clone, Debug, PartialEq)]
pub enum TirValue {
    Number(f64),
    Text(String),
}

/// The entries of a `.tir` file in file order, with their section.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TirFile {
    entries: Vec<(String, String, TirValue)>,
}

#[derive(Debug, thiserror::Error)]
pub enum TirError {
    #[error("cannot read tyre property file: {0}")]
    Io(#[from] std::io::Error),
    #[error("line {line}: {msg}")]
    Syntax { line: usize, msg: String },
    #[error("unsupported tyre property file: {0}")]
    Unsupported(String),
    #[error("missing tyre parameter {0}")]
    Missing(&'static str),
    #[error("invalid tyre parameter {key}: {msg}")]
    Invalid { key: String, msg: String },
}

/// Units that the parameters are read in (anything else is rejected rather than converted).
const SI_UNITS: [(&str, &[&str]); 6] = [
    ("LENGTH", &["meter", "metre", "m"]),
    ("FORCE", &["newton", "n"]),
    ("ANGLE", &["radian", "radians", "rad"]),
    ("MASS", &["kg", "kilogram"]),
    ("TIME", &["second", "sec", "s"]),
    ("PRESSURE", &["pascal", "pa"]),
];

impl TirFile {
    pub fn parse(text: &str) -> Result<Self, TirError> {
        let mut section = String::new();
        let mut entries = Vec::new();
        for (k, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('$') || line.starts_with('!') {
                continue;
            }
            if let Some(rest) = line.strip_prefix('[') {
                let end = rest
                    .find(']')
                    .ok_or_else(|| TirError::Syntax { line: k + 1, msg: "unterminated section header".into() })?;
                section = rest[..end].trim().to_ascii_uppercase();
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue; // table rows
            };
            let key = key.trim().to_ascii_uppercase();
            let value = value.split('$').next().unwrap_or("").trim();
            if key.is_empty() || value.is_empty() {
                continue;
            }
            let value = if let Some(q) = value.strip_prefix('\'') {
                TirValue::Text(q.trim_end_matches('\'').trim().to_string())
            } else {
                // Fortran-style exponents (1.0D+03) occur in older files.
                let v: f64 = value.replace(['D', 'd'], "e").parse().map_err(|_| TirError::Syntax {
                    line: k + 1,
                    msg: format!("{key}: cannot read {value:?} as a number"),
                })?;
                TirValue::Number(v)
            };
            entries.push((section.clone(), key, value));
        }
        let file = Self { entries };
        file.check_units()?;
        Ok(file)
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, TirError> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    fn check_units(&self) -> Result<(), TirError> {
        for (section, key, value) in &self.entries {
            if section != "UNITS" {
                continue;
            }
            let TirValue::Text(unit) = value else { continue };
            if let Some((_, allowed)) = SI_UNITS.iter().find(|(k, _)| k == key)
                && !allowed.contains(&unit.to_ascii_lowercase().as_str())
            {
                return Err(TirError::Unsupported(format!("{key} unit {unit:?} (only SI units are supported)")));
            }
        }
        Ok(())
    }

    /// The last value of `key` outside `[UNITS]` (where e.g. `MASS` names a unit).
    pub fn get(&self, key: &str) -> Option<&TirValue> {
        self.entries.iter().rev().find(|(s, k, _)| s != "UNITS" && k == key).map(|(_, _, v)| v)
    }

    pub fn number(&self, key: &str) -> Result<Option<f64>, TirError> {
        match self.get(key) {
            None => Ok(None),
            Some(TirValue::Number(v)) if v.is_finite() => Ok(Some(*v)),
            Some(v) => Err(TirError::Invalid { key: key.into(), msg: format!("expected a number, found {v:?}") }),
        }
    }

    pub fn text(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(TirValue::Text(s)) => Some(s),
            _ => None,
        }
    }

    /// All `(section, key, value)` entries in file order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &str, &TirValue)> {
        self.entries.iter().map(|(s, k, v)| (s.as_str(), k.as_str(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_sections_numbers_strings_and_comments() {
        let text = "$ header comment\n[UNITS]\nLENGTH = 'meter'\nMASS = 'kg'\n[MODEL]\nFITTYP = 61 $version\n\
                    PROPERTY_FILE_FORMAT ='PAC2002'\n[SHAPE]\n{radial width}\n 1.0 0.0\n[INERTIA]\nmass = 9.3\n\
                    ! other comment\nQ = 1.5D-03\n";
        let f = TirFile::parse(text).unwrap();
        assert_eq!(f.number("FITTYP").unwrap(), Some(61.0));
        assert_eq!(f.text("PROPERTY_FILE_FORMAT"), Some("PAC2002"));
        assert_eq!(f.number("MASS").unwrap(), Some(9.3));
        assert_eq!(f.number("Q").unwrap(), Some(1.5e-3));
        assert_eq!(f.number("MISSING").unwrap(), None);
        assert!(f.number("PROPERTY_FILE_FORMAT").is_err());
        assert!(TirFile::parse("[UNITS]\nLENGTH = 'mm'\n").is_err());
        assert!(TirFile::parse("[MODEL]\nX = abc\n").is_err());
    }
}
