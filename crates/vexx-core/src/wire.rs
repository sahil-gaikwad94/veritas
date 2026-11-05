//! JSON wire format shared by the Python extension and the wasm inspector.
//!
//! The Python engine owns the schema semantics; the native core receives only the numeric
//! fragments it can decide, in a compact, explicitly versioned shape:
//!
//! ```text
//! domains     {"0": [0.0, 1.0], "1": [0.0, 1.0]}
//! constraints [{"kind": "linear", "coeffs": {"0": 1.0}, "rhs": 1.0, "op": "le"},
//!              {"kind": "sum", "vars": [0, 1], "rhs": 1.0, "op": "eq"},
//!              {"kind": "square", "var": 0, "rhs": 0.25, "op": "eq"},
//!              {"kind": "product", "a": 0, "b": 1, "rhs": 4.0, "op": "eq"},
//!              {"kind": "unsupported"}]
//! linear      [{"coeffs": {"0": 1.0}, "rhs": 1.0, "op": "le"}]
//! ```
//!
//! Both directions are total: malformed input is an `Err(String)` with a readable reason,
//! never a panic.

use crate::contractor::{Constraint, Contractor};
use crate::expr::CmpOp;
use crate::interval::Interval;
use crate::json::Json;
use crate::solver::{Linear, Outcome};
use std::collections::BTreeMap;

pub const WIRE_VERSION: u32 = 1;

pub fn number(value: f64) -> Json {
    Json::Number(value)
}

/// Print a bound: ±∞ travels as `null`, everything else as a round-tripping number.
fn endpoint(value: f64) -> Json {
    if value.is_finite() {
        number(value)
    } else {
        Json::Null
    }
}

/// Read a bound: `null` is -∞ for a lower endpoint and +∞ for an upper one.
fn unbounded(node: &Json, infinite: f64) -> Option<f64> {
    match node {
        Json::Null => Some(infinite),
        other => other.as_f64(),
    }
}

fn field<'a>(node: &'a Json, key: &str) -> Result<&'a Json, String> {
    node.get(key)
        .ok_or_else(|| format!("missing field '{}'", key))
}

fn as_f64(node: &Json, key: &str) -> Result<f64, String> {
    field(node, key)?
        .as_f64()
        .ok_or_else(|| format!("field '{}' must be a number", key))
}

fn as_u32(node: &Json, key: &str) -> Result<u32, String> {
    let value = as_f64(node, key)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > u32::MAX as f64 {
        return Err(format!("field '{}' must be a variable id", key));
    }
    Ok(value as u32)
}

fn as_op(node: &Json) -> Result<CmpOp, String> {
    let text = field(node, "op")?
        .as_str()
        .ok_or_else(|| "field 'op' must be a string".to_string())?;
    CmpOp::parse(text).ok_or_else(|| format!("unknown comparison '{}'", text))
}

fn coeffs(node: &Json) -> Result<Vec<(u32, f64)>, String> {
    let mut out: Vec<(u32, f64)> = Vec::new();
    let entries = field(node, "coeffs")?
        .as_object()
        .ok_or_else(|| "field 'coeffs' must be an object".to_string())?;
    for (key, value) in entries {
        let var: u32 = key
            .parse()
            .map_err(|_| format!("coefficient key '{}' is not a variable id", key))?;
        let coeff = value
            .as_f64()
            .ok_or_else(|| format!("coefficient '{}' must be a number", key))?;
        if !coeff.is_finite() {
            return Err(format!("coefficient '{}' is not finite", key));
        }
        out.push((var, coeff));
    }
    out.sort_by_key(|(var, _)| *var);
    Ok(out)
}

/// Parse a `{"0": [lo, hi]}` domain map.
pub fn parse_domains(node: &Json) -> Result<BTreeMap<u32, Interval>, String> {
    let mut domains = BTreeMap::new();
    let entries = node
        .as_object()
        .ok_or_else(|| "domains must be an object".to_string())?;
    for (key, value) in entries {
        let var: u32 = key
            .parse()
            .map_err(|_| format!("domain key '{}' is not a variable id", key))?;
        let pair = value
            .as_array()
            .ok_or_else(|| format!("domain '{}' must be [lo, hi]", key))?;
        if pair.is_empty() {
            domains.insert(var, Interval::EMPTY);
            continue;
        }
        if pair.len() != 2 {
            return Err(format!("domain '{}' must have two endpoints", key));
        }
        let lo = unbounded(&pair[0], f64::NEG_INFINITY)
            .ok_or_else(|| format!("domain '{}' has a non-numeric lower bound", key))?;
        let hi = unbounded(&pair[1], f64::INFINITY)
            .ok_or_else(|| format!("domain '{}' has a non-numeric upper bound", key))?;
        if lo.is_nan() || hi.is_nan() {
            return Err(format!("domain '{}' is NaN", key));
        }
        domains.insert(var, Interval::new(lo, hi));
    }
    Ok(domains)
}

/// Parse a constraint array; the second value counts entries the contractor cannot decide.
pub fn parse_constraints(node: &Json) -> Result<(Vec<Constraint>, usize), String> {
    let mut constraints = Vec::new();
    let mut unsupported = 0usize;
    let entries = node
        .as_array()
        .ok_or_else(|| "constraints must be an array".to_string())?;
    for entry in entries {
        let kind = field(entry, "kind")?
            .as_str()
            .ok_or_else(|| "constraint 'kind' must be a string".to_string())?;
        match kind {
            "sum" => {
                let vars = field(entry, "vars")?
                    .as_array()
                    .ok_or_else(|| "'sum' needs a 'vars' array".to_string())?
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .filter(|v| v.is_finite() && *v >= 0.0)
                            .map(|v| v as u32)
                            .ok_or_else(|| "'vars' must hold variable ids".to_string())
                    })
                    .collect::<Result<Vec<u32>, String>>()?;
                constraints.push(Constraint::Sum {
                    vars,
                    rhs: as_f64(entry, "rhs")?,
                    op: as_op(entry)?,
                });
            }
            "linear" => constraints.push(Constraint::Linear {
                coeffs: coeffs(entry)?,
                rhs: as_f64(entry, "rhs")?,
                op: as_op(entry)?,
            }),
            "square" => constraints.push(Constraint::Square {
                var: as_u32(entry, "var")?,
                rhs: as_f64(entry, "rhs")?,
                op: as_op(entry)?,
            }),
            "product" => constraints.push(Constraint::Product {
                a: as_u32(entry, "a")?,
                b: as_u32(entry, "b")?,
                rhs: as_f64(entry, "rhs")?,
                op: as_op(entry)?,
            }),
            "quadratic_form" => {
                let vars = field(entry, "vars")?
                    .as_array()
                    .ok_or_else(|| "'quadratic_form' needs a 'vars' array".to_string())?
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .filter(|v| v.is_finite() && *v >= 0.0)
                            .map(|v| v as u32)
                            .ok_or_else(|| "'vars' must hold variable ids".to_string())
                    })
                    .collect::<Result<Vec<u32>, String>>()?;
                let cholesky_l = field(entry, "cholesky_l")?
                    .as_array()
                    .ok_or_else(|| "'quadratic_form' needs a 'cholesky_l' array".to_string())?
                    .iter()
                    .map(|row| {
                        row.as_array()
                            .ok_or_else(|| "cholesky_l rows must be arrays".to_string())?
                            .iter()
                            .map(|value| {
                                value
                                    .as_f64()
                                    .filter(|v| v.is_finite())
                                    .ok_or_else(|| "cholesky_l entries must be finite numbers".to_string())
                            })
                            .collect::<Result<Vec<f64>, String>>()
                    })
                    .collect::<Result<Vec<Vec<f64>>, String>>()?;
                // Validate lower triangular structure
                if cholesky_l.len() != vars.len() {
                    return Err("cholesky_l dimension must match vars length".to_string());
                }
                for (i, row) in cholesky_l.iter().enumerate() {
                    if row.len() != i + 1 {
                        return Err(format!("cholesky_l row {} must have length {}", i, i + 1));
                    }
                }
                constraints.push(Constraint::QuadraticForm {
                    vars,
                    cholesky_l,
                    rhs: as_f64(entry, "rhs")?,
                    op: as_op(entry)?,
                });
            }
            _ => unsupported += 1,
        }
    }
    Ok((constraints, unsupported))
}

/// Build the contractor for a wire program and run it. The flag is `true` when the box was
/// proven empty.
pub fn run_contractor(
    domains: &Json,
    constraints: &Json,
    rounds: usize,
) -> Result<(Contractor, bool), String> {
    let mut contractor = Contractor::new(parse_domains(domains)?);
    let (parsed, unsupported) = parse_constraints(constraints)?;
    for constraint in parsed {
        contractor.push(constraint);
    }
    for _ in 0..unsupported {
        contractor.note_unsupported();
    }
    let feasible = contractor.propagate(rounds.max(1));
    Ok((contractor, !feasible))
}

pub fn parse_linear(node: &Json) -> Result<Vec<Linear>, String> {
    let entries = node
        .as_array()
        .ok_or_else(|| "linear constraints must be an array".to_string())?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        out.push(Linear {
            coeffs: coeffs(entry)?,
            rhs: as_f64(entry, "rhs")?,
            op: as_op(entry)?,
        });
    }
    Ok(out)
}

pub fn domains_json(contractor: &Contractor) -> Json {
    let mut entries = Vec::new();
    for (var, interval) in contractor.domains() {
        // `null` is the wire spelling of an infinite bound: JSON cannot print ±∞, and an empty
        // interval is `[]` so it cannot be confused with the unbounded `[null, null]`.
        let pair = if interval.is_empty() {
            Json::Array(Vec::new())
        } else {
            Json::Array(vec![endpoint(interval.lo), endpoint(interval.hi)])
        };
        entries.push((var.to_string(), pair));
    }
    Json::Object(entries)
}

/// `{"status": "empty"|"feasible", "domains": {...}, "unsupported": n, "constraints": n}`
pub fn contractor_json(contractor: &Contractor, refuted: bool) -> Json {
    Json::Object(vec![
        (
            "status".to_string(),
            Json::Str(if refuted { "empty" } else { "feasible" }.to_string()),
        ),
        ("domains".to_string(), domains_json(contractor)),
        (
            "unsupported".to_string(),
            number(contractor.unsupported() as f64),
        ),
        (
            "constraints".to_string(),
            number(contractor.constraint_count() as f64),
        ),
        ("version".to_string(), number(WIRE_VERSION as f64)),
    ])
}

pub fn outcome_json(outcome: &Outcome) -> Json {
    match outcome {
        Outcome::Sat { model, exact } => Json::Object(vec![
            ("status".to_string(), Json::Str("sat".to_string())),
            (
                "model".to_string(),
                Json::Object(
                    model
                        .iter()
                        .map(|(var, value)| (var.to_string(), number(*value)))
                        .collect(),
                ),
            ),
            (
                "exact".to_string(),
                Json::Object(
                    exact
                        .iter()
                        .map(|(var, value)| (var.to_string(), Json::Str(value.to_string())))
                        .collect(),
                ),
            ),
        ]),
        Outcome::Unsat { core } => Json::Object(vec![
            ("status".to_string(), Json::Str("unsat".to_string())),
            (
                "core".to_string(),
                Json::Array(core.iter().map(|index| number(*index as f64)).collect()),
            ),
        ]),
        Outcome::Unknown { reason } => Json::Object(vec![
            ("status".to_string(), Json::Str("unknown".to_string())),
            ("reason".to_string(), Json::Str(reason.clone())),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    #[test]
    fn round_trips_domains_and_constraints() {
        let domains = parse("{\"0\": [0.0, 1.0], \"1\": [-1.0, 2.0]}").unwrap();
        let constraints = parse(
            "[{\"kind\": \"sum\", \"vars\": [0, 1], \"rhs\": 1.0, \"op\": \"eq\"}, {\"kind\": \"mystery\"}]",
        )
        .unwrap();
        let (contractor, refuted) = run_contractor(&domains, &constraints, 8).expect("program");
        assert!(!refuted);
        assert_eq!(contractor.unsupported(), 1);
        assert_eq!(contractor.constraint_count(), 1);
        let report = contractor_json(&contractor, refuted).render();
        assert!(report.contains("\"status\": \"feasible\""));
        assert!(report.contains("\"unsupported\": 1"));

        let impossible = parse(
            "[{\"kind\": \"sum\", \"vars\": [0, 1], \"rhs\": 5.0, \"op\": \"eq\"}]",
        )
        .unwrap();
        let (contractor, refuted) = run_contractor(&domains, &impossible, 8).expect("program");
        assert!(refuted);
        assert!(contractor_json(&contractor, refuted)
            .render()
            .contains("\"status\": \"empty\""));
    }

    #[test]
    fn unbounded_and_empty_domains_survive_the_wire() {
        let node = parse(
            "{\"0\": [null, 1.0], \"1\": [0.5, null], \"2\": [], \"3\": [null, null]}",
        )
        .unwrap();
        let domains = parse_domains(&node).expect("unbounded domains parse");
        assert_eq!(domains[&0], Interval::new(f64::NEG_INFINITY, 1.0));
        assert_eq!(domains[&1], Interval::new(0.5, f64::INFINITY));
        assert!(domains[&2].is_empty());
        // `[null, null]` is the unbounded domain, not the empty one: the spellings differ.
        assert_eq!(domains[&3], Interval::ENTIRE);

        // The printer reproduces what the parser accepts, losslessly.
        let contractor = Contractor::new(domains.clone());
        let report = domains_json(&contractor);
        assert_eq!(parse_domains(&report).expect("round trip"), domains);
        assert!(report.render().contains("\"2\": []"));
    }

    #[test]
    fn malformed_input_is_an_error_not_a_panic() {
        assert!(parse_domains(&parse("{\"x\": [0, 1]}").unwrap()).is_err());
        assert!(parse_domains(&parse("{\"0\": [0]}").unwrap()).is_err());
        assert!(parse_constraints(&parse("{}").unwrap()).is_err());
        assert!(parse_linear(&parse("[{\"rhs\": 1.0, \"op\": \"nope\"}]").unwrap()).is_err());
    }

    #[test]
    fn reports_solver_outcomes() {
        let outcome = Outcome::Unsat { core: vec![0, 2] };
        let rendered = outcome_json(&outcome).render();
        assert!(rendered.contains("\"unsat\""));
        let sat = Outcome::Sat {
            model: vec![(0, 0.5)],
            exact: vec![(0, crate::num::Rational::new(1, 2).unwrap())],
        };
        let rendered = outcome_json(&sat).render();
        assert!(rendered.contains("\"status\": \"sat\""));
        assert!(rendered.contains("\"0\": 0.5"));
    }
}
