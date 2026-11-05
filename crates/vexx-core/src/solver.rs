//! Exact-rational linear feasibility (Fourier–Motzkin) with conflict cores.
//!
//! The interval contractor refutes what it can see; this module decides the linear fragment
//! *exactly*, so `UNSAT` is a proof and `SAT` carries a witness. Strictness is tracked per
//! row, because `x < 0 ∧ x >= 0` is infeasible while `x <= 0 ∧ x >= 0` is not — the classic
//! float-solver bug this module must not have.
//!
//! Everything that leaves the representable range returns [`Outcome::Unknown`] with a reason.
//! Unknown is a first-class answer: the Python engine then makes the call.

use crate::expr::CmpOp;
use crate::num::Rational;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Variables beyond this are not attempted (elimination cost is exponential).
pub const MAX_VARS: usize = 10;
const MAX_ROWS: usize = 4096;
const MAX_CONSTRAINTS: usize = 64;
const CORE_LIMIT: usize = 24;

type Res<T> = Result<T, String>;

/// One linear atom: `Σ coeffs[i].1 * x[coeffs[i].0] <op> rhs`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub coeffs: Vec<(u32, f64)>,
    pub rhs: f64,
    pub op: CmpOp,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Sat {
        model: Vec<(u32, f64)>,
        exact: Vec<(u32, Rational)>,
    },
    Unsat {
        /// Indices into the input slice that are jointly infeasible (minimized by deletion).
        core: Vec<usize>,
    },
    Unknown {
        reason: String,
    },
}

#[derive(Clone, Debug)]
struct Row {
    coeffs: BTreeMap<u32, Rational>,
    bound: Rational,
    strict: bool,
    /// Ancestor bitmask of the original constraints this row was combined from.
    /// Tracked for unsat-core provenance; no longer used for pruning.
    sources: u64,
}

impl Row {
    fn new() -> Row {
        Row {
            coeffs: BTreeMap::new(),
            bound: Rational::ZERO,
            strict: false,
            sources: 0,
        }
    }

    fn add_scaled(&mut self, other: &Row, factor: Rational) -> Res<()> {
        let bound = Rational::mul(other.bound, factor)
            .ok_or_else(|| "exact arithmetic overflow (bound)".to_string())?;
        for (var, coeff) in &other.coeffs {
            let scaled = coeff
                .mul(factor)
                .ok_or_else(|| "exact arithmetic overflow (coefficient)".to_string())?;
            let merged = match self.coeffs.get(var) {
                Some(existing) => existing
                    .add(scaled)
                    .ok_or_else(|| "exact arithmetic overflow (merge)".to_string())?,
                None => scaled,
            };
            if merged.is_zero() {
                self.coeffs.remove(var);
            } else {
                self.coeffs.insert(*var, merged);
            }
        }
        self.bound = self
            .bound
            .add(bound)
            .ok_or_else(|| "exact arithmetic overflow (accumulate)".to_string())?;
        self.strict = self.strict || other.strict;
        Ok(())
    }
}

fn rat(value: f64) -> Res<Rational> {
    if !value.is_finite() {
        return Err(format!("non-finite literal {}", value));
    }
    Rational::from_f64(value).ok_or_else(|| format!("literal {} is not exactly representable", value))
}

fn rows_from(constraints: &[Linear]) -> Res<Vec<Row>> {
    let mut rows: Vec<Row> = Vec::with_capacity(constraints.len());
    for (index, constraint) in constraints.iter().enumerate() {
        let (sign, strict) = match constraint.op {
            CmpOp::Le => (1.0, false),
            CmpOp::Lt => (1.0, true),
            CmpOp::Ge => (-1.0, false),
            CmpOp::Gt => (-1.0, true),
            CmpOp::Eq => (1.0, false),
            other => return Err(format!("'{}' is not a linear comparison", other.symbol())),
        };
        let mut row = Row::new();
        // Up to 64 constraints (enforced by callers); bit `index` is this constraint's lineage.
        row.sources = 1u64 << index;
        for (var, coeff) in &constraint.coeffs {
            let scaled = rat(*coeff * sign)?;
            let merged = match row.coeffs.get(var) {
                Some(existing) => existing
                    .add(scaled)
                    .ok_or_else(|| "exact arithmetic overflow (coefficient)".to_string())?,
                None => scaled,
            };
            if merged.is_zero() {
                row.coeffs.remove(var);
            } else {
                row.coeffs.insert(*var, merged);
            }
        }
        row.bound = rat(constraint.rhs * sign)?;
        row.strict = strict;
        rows.push(row.clone());
        if constraint.op == CmpOp::Eq {
            let mut mirror = Row::new();
            for (var, coeff) in &row.coeffs {
                mirror.coeffs.insert(*var, coeff.neg());
            }
            mirror.bound = row.bound.neg();
            mirror.strict = false;
            mirror.sources = row.sources;
            rows.push(mirror);
        }
    }
    Ok(rows)
}

fn eliminate(rows: &[Row], var: u32) -> Res<Vec<Row>> {
    let mut positive: Vec<&Row> = Vec::new();
    let mut negative: Vec<&Row> = Vec::new();
    let mut rest: Vec<Row> = Vec::new();
    for row in rows {
        match row.coeffs.get(&var) {
            None => rest.push(row.clone()),
            Some(coeff) if coeff.is_positive() => positive.push(row),
            Some(_) => negative.push(row),
        }
    }
    if positive.is_empty() || negative.is_empty() {
        // One direction is unbounded: the projection drops those rows entirely.
        return Ok(rest);
    }
    if positive.len().saturating_mul(negative.len()) > MAX_ROWS {
        return Err("elimination blowup".to_string());
    }
    for left in &positive {
        let left_coeff = *left
            .coeffs
            .get(&var)
            .ok_or_else(|| "internal: missing coefficient".to_string())?;
        for right in &negative {
            let right_coeff = *right
                .coeffs
                .get(&var)
                .ok_or_else(|| "internal: missing coefficient".to_string())?;
            let mut combined = Row::new();
            combined.add_scaled(left, right_coeff.neg())?;
            combined.add_scaled(right, left_coeff)?;
            combined.coeffs.remove(&var);
            // Lineage carries forward for the next elimination step.
            combined.sources = left.sources | right.sources;
            rest.push(combined);
        }
    }
    // Redundant-inequality elimination by *subsumption*: rows are bucketed by
    // direction-normalized coefficient sign/magnitude and only the strongest bound in
    // each bucket survives.  Subsumption is sound (the dropped row is implied by the
    // kept one).  The previous lineage-size rule could drop the only contradictory
    // combination and flip UNSAT to a wrong verdict.
    Ok(subsume(rest))
}

/// Direction-normalized key: multiply by -1 so the largest-|coefficient| is positive,
/// divide by it, and encode exactly as (var, numer, denom) triples so buckets compare
/// exactly.  Constant rows share the empty key.
fn normalized_key(row: &Row) -> Vec<(u32, i128, i128)> {
    let items: Vec<(u32, Rational)> = row.coeffs.iter().map(|(&var, &coeff)| (var, coeff)).collect();
    if items.is_empty() {
        return Vec::new();
    }
    // Pivot: the largest-|coefficient|.  Rows are scaled by a POSITIVE constant (the pivot
    // magnitude) so the largest coefficient becomes +/-1 *with its sense intact*; scaling by
    // a negative constant would silently swap the inequality direction.
    let mut pivot = items[0].1;
    for &(_, coeff) in items.iter().skip(1) {
        let abs_c = if coeff.is_negative() { coeff.neg() } else { coeff };
        let abs_p = if pivot.is_negative() { pivot.neg() } else { pivot };
        if abs_c.compare(abs_p) == Some(Ordering::Greater) {
            pivot = coeff;
        }
    }
    let abs_pivot = if pivot.is_negative() { pivot.neg() } else { pivot };
    let mut key = Vec::with_capacity(items.len());
    for (var, coeff) in items {
        match coeff.div(abs_pivot) {
            Some(c) => key.push((var, c.numer(), c.denom())),
            // Overflow: fall back to exact raw terms (buckets stay tiny — nothing is dropped).
            None => key.push((var, coeff.numer(), coeff.denom())),
        }
    }
    key
}

fn subsume(rows: Vec<Row>) -> Vec<Row> {
    let mut buckets: BTreeMap<Vec<(u32, i128, i128)>, Row> = BTreeMap::new();
    for row in rows {
        let key = normalized_key(&row);
        match buckets.get_mut(&key) {
            Some(current) => {
                let tighter = match row.bound.compare(current.bound) {
                    Some(Ordering::Less) => true,
                    Some(Ordering::Equal) => row.strict && !current.strict,
                    _ => false,
                };
                if tighter {
                    *current = Row {
                        coeffs: row.coeffs,
                        bound: row.bound,
                        strict: row.strict,
                        sources: current.sources | row.sources,
                    };
                } else {
                    current.sources |= row.sources;
                }
            }
            None => {
                buckets.insert(key, row);
            }
        }
    }
    buckets.into_values().collect()
}

fn eliminate_all(rows: &[Row], order: &[u32]) -> Res<Vec<Row>> {
    let mut current = rows.to_vec();
    for var in order {
        if current.len() > MAX_ROWS {
            return Err("system blowup".to_string());
        }
        current = eliminate(&current, *var)?;
    }
    Ok(current)
}

/// `true` = constant rows are all satisfiable.
fn constantly_feasible(rows: &[Row]) -> Res<bool> {
    for row in rows {
        if !row.coeffs.is_empty() {
            return Err("internal: row still has variables".to_string());
        }
        match row.bound.compare(Rational::ZERO) {
            None => return Err("exact comparison overflow".to_string()),
            Some(Ordering::Less) => return Ok(false),
            Some(Ordering::Equal) if row.strict => return Ok(false),
            Some(_) => {}
        }
    }
    Ok(true)
}

fn decide(rows: &[Row], order: &[u32]) -> Res<bool> {
    let reduced = eliminate_all(rows, order)?;
    constantly_feasible(&reduced)
}

fn substitute(rows: &mut [Row], values: &[(u32, Rational)]) -> Res<()> {
    for row in rows.iter_mut() {
        for (var, value) in values {
            if let Some(coeff) = row.coeffs.remove(var) {
                let product = coeff
                    .mul(*value)
                    .ok_or_else(|| "exact arithmetic overflow (substitution)".to_string())?;
                row.bound = row
                    .bound
                    .sub(product)
                    .ok_or_else(|| "exact arithmetic overflow (substitution)".to_string())?;
            }
        }
    }
    Ok(())
}

fn tighten_lower(
    current: Option<Rational>,
    current_strict: bool,
    value: Rational,
    strict: bool,
) -> Res<(Option<Rational>, bool)> {
    match current {
        None => Ok((Some(value), strict)),
        Some(existing) => match value.compare(existing) {
            None => Err("exact comparison overflow".to_string()),
            Some(Ordering::Greater) => Ok((Some(value), strict)),
            Some(Ordering::Less) => Ok((Some(existing), current_strict)),
            Some(Ordering::Equal) => Ok((Some(existing), current_strict || strict)),
        },
    }
}

fn tighten_upper(
    current: Option<Rational>,
    current_strict: bool,
    value: Rational,
    strict: bool,
) -> Res<(Option<Rational>, bool)> {
    match current {
        None => Ok((Some(value), strict)),
        Some(existing) => match value.compare(existing) {
            None => Err("exact comparison overflow".to_string()),
            Some(Ordering::Less) => Ok((Some(value), strict)),
            Some(Ordering::Greater) => Ok((Some(existing), current_strict)),
            Some(Ordering::Equal) => Ok((Some(existing), current_strict || strict)),
        },
    }
}

struct Bounds {
    lo: Option<Rational>,
    lo_strict: bool,
    hi: Option<Rational>,
    hi_strict: bool,
}

fn bounds_on(rows: &[Row], var: u32) -> Res<Bounds> {
    let mut bounds = Bounds {
        lo: None,
        lo_strict: false,
        hi: None,
        hi_strict: false,
    };
    for row in rows {
        match row.coeffs.len() {
            0 => {
                if !constantly_feasible(std::slice::from_ref(row))? {
                    return Err("projected system is refuted".to_string());
                }
            }
            1 => {
                let (_, coeff) = row
                    .coeffs
                    .iter()
                    .next()
                    .ok_or_else(|| "internal: empty row".to_string())?;
                if *row.coeffs.keys().next().unwrap() != var {
                    return Err("projection retained another variable".to_string());
                }
                let limit = row
                    .bound
                    .div(*coeff)
                    .ok_or_else(|| "exact arithmetic overflow (bound)".to_string())?;
                if coeff.is_positive() {
                    let (hi, hi_strict) =
                        tighten_upper(bounds.hi, bounds.hi_strict, limit, row.strict)?;
                    bounds.hi = hi;
                    bounds.hi_strict = hi_strict;
                } else {
                    let (lo, lo_strict) =
                        tighten_lower(bounds.lo, bounds.lo_strict, limit, row.strict)?;
                    bounds.lo = lo;
                    bounds.lo_strict = lo_strict;
                }
            }
            _ => return Err("projection retained several variables".to_string()),
        }
    }
    Ok(bounds)
}

fn pick(bounds: Bounds) -> Res<Rational> {
    match (bounds.lo, bounds.hi) {
        (None, None) => Ok(Rational::ZERO),
        (Some(lo), None) => lo
            .add(Rational::ONE)
            .ok_or_else(|| "exact arithmetic overflow (pick)".to_string()),
        (None, Some(hi)) => hi
            .sub(Rational::ONE)
            .ok_or_else(|| "exact arithmetic overflow (pick)".to_string()),
        (Some(lo), Some(hi)) => match lo.compare(hi) {
            None => Err("exact comparison overflow (pick)".to_string()),
            Some(Ordering::Greater) => Err("empty interval while picking".to_string()),
            Some(Ordering::Equal) => {
                if bounds.lo_strict || bounds.hi_strict {
                    Err("empty interval while picking".to_string())
                } else {
                    Ok(lo)
                }
            }
            Some(Ordering::Less) => {
                let sum = lo
                    .add(hi)
                    .ok_or_else(|| "exact arithmetic overflow (midpoint)".to_string())?;
                sum.div(Rational::from_int(2))
                    .ok_or_else(|| "exact arithmetic overflow (midpoint)".to_string())
            }
        },
    }
}

fn witness(rows: &[Row], order: &[u32]) -> Res<Vec<(u32, Rational)>> {
    let mut chosen: Vec<(u32, Rational)> = Vec::with_capacity(order.len());
    for (position, var) in order.iter().enumerate() {
        let mut pinned = rows.to_vec();
        substitute(&mut pinned, &chosen)?;
        let remaining: Vec<u32> = order[position + 1..].to_vec();
        let projected = eliminate_all(&pinned, &remaining)?;
        let bounds = bounds_on(&projected, *var)?;
        let value = pick(bounds)?;
        chosen.push((*var, value));
    }
    Ok(chosen)
}

fn verify(model: &[(u32, f64)], constraints: &[Linear]) -> bool {
    let lookup: BTreeMap<u32, f64> = model.iter().copied().collect();
    for constraint in constraints {
        let mut total = 0.0;
        for (var, coeff) in &constraint.coeffs {
            match lookup.get(var) {
                Some(value) => total += value * coeff,
                None => return false,
            }
        }
        let tolerance = 1e-9 * (1.0 + total.abs() + constraint.rhs.abs());
        let satisfied = match constraint.op {
            CmpOp::Le => total <= constraint.rhs + tolerance,
            CmpOp::Lt => total < constraint.rhs + tolerance,
            CmpOp::Ge => total >= constraint.rhs - tolerance,
            CmpOp::Gt => total > constraint.rhs - tolerance,
            CmpOp::Eq => (total - constraint.rhs).abs() <= tolerance,
            _ => return false,
        };
        if !satisfied {
            return false;
        }
    }
    true
}

fn variables(constraints: &[Linear]) -> Vec<u32> {
    let mut vars: BTreeSet<u32> = BTreeSet::new();
    for constraint in constraints {
        for (var, coeff) in &constraint.coeffs {
            if *coeff != 0.0 {
                vars.insert(*var);
            }
        }
    }
    vars.into_iter().collect()
}

fn unsat_core(constraints: &[Linear]) -> Res<Vec<usize>> {
    let mut keep: Vec<usize> = (0..constraints.len()).collect();
    if constraints.len() > CORE_LIMIT {
        return Ok(keep);
    }
    let mut index = 0usize;
    while index < keep.len() {
        let mut trial = keep.clone();
        trial.remove(index);
        let subset: Vec<Linear> = trial.iter().map(|slot| constraints[*slot].clone()).collect();
        let order = variables(&subset);
        if order.len() > MAX_VARS {
            index += 1;
            continue;
        }
        let rows = rows_from(&subset)?;
        if decide(&rows, &order)? {
            index += 1;
        } else {
            keep = trial;
        }
    }
    Ok(keep)
}

/// Decide a conjunction of linear atoms over the reals.
pub fn solve(constraints: &[Linear]) -> Outcome {
    match solve_inner(constraints) {
        Ok(outcome) => outcome,
        Err(reason) => Outcome::Unknown { reason },
    }
}

fn solve_inner(constraints: &[Linear]) -> Res<Outcome> {
    if constraints.len() > MAX_CONSTRAINTS {
        return Ok(Outcome::Unknown {
            reason: format!("{} constraints is more than the solver attempts", constraints.len()),
        });
    }
    let order = variables(constraints);
    if order.len() > MAX_VARS {
        return Ok(Outcome::Unknown {
            reason: format!("{} variables is more than the solver attempts", order.len()),
        });
    }
    let rows = rows_from(constraints)?;
    if !decide(&rows, &order)? {
        let core = unsat_core(constraints)?;
        return Ok(Outcome::Unsat { core });
    }
    let exact = witness(&rows, &order)?;
    let model: Vec<(u32, f64)> = exact.iter().map(|(var, value)| (*var, value.to_f64())).collect();
    if !verify(&model, constraints) {
        // The witness is the only thing we cannot prove on paper; refuse to hand out a
        // model that fails its own constraints.
        return Ok(Outcome::Unknown {
            reason: "exact witness failed numeric verification".to_string(),
        });
    }
    Ok(Outcome::Sat { model, exact })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(coeffs: &[(u32, f64)], rhs: f64, op: CmpOp) -> Linear {
        Linear {
            coeffs: coeffs.to_vec(),
            rhs,
            op,
        }
    }

    #[test]
    fn refutes_disjoint_bounds() {
        let constraints = [
            atom(&[(0, 1.0)], 0.6, CmpOp::Ge),
            atom(&[(0, 1.0)], 0.5, CmpOp::Le),
        ];
        match solve(&constraints) {
            Outcome::Unsat { core } => assert_eq!(core, vec![0, 1]),
            other => panic!("expected UNSAT, got {:?}", other),
        }
    }

    #[test]
    fn sat_witness_satisfies_every_constraint() {
        let constraints = [
            atom(&[(0, 1.0), (1, 1.0)], 1.0, CmpOp::Le),
            atom(&[(0, 1.0)], 0.3, CmpOp::Ge),
            atom(&[(1, 1.0)], 0.3, CmpOp::Ge),
        ];
        match solve(&constraints) {
            Outcome::Sat { model, .. } => {
                let lookup: BTreeMap<u32, f64> = model.into_iter().collect();
                assert!(lookup[&0] + lookup[&1] <= 1.0 + 1e-9);
                assert!(lookup[&0] >= 0.3 - 1e-9);
                assert!(lookup[&1] >= 0.3 - 1e-9);
            }
            other => panic!("expected SAT, got {:?}", other),
        }
    }

    #[test]
    fn strictness_is_decided_exactly() {
        let strict_conflict = [
            atom(&[(0, 1.0)], 0.0, CmpOp::Lt),
            atom(&[(0, 1.0)], 0.0, CmpOp::Ge),
        ];
        assert!(matches!(solve(&strict_conflict), Outcome::Unsat { .. }));

        let touching = [
            atom(&[(0, 1.0)], 0.0, CmpOp::Le),
            atom(&[(0, 1.0)], 0.0, CmpOp::Ge),
        ];
        match solve(&touching) {
            Outcome::Sat { exact, .. } => assert_eq!(exact[0].1, Rational::ZERO),
            other => panic!("expected SAT at the touching point, got {:?}", other),
        }
    }

    #[test]
    fn exact_fractions_survive_elimination() {
        let constraints = [atom(&[(0, 3.0)], 1.0, CmpOp::Eq)];
        match solve(&constraints) {
            Outcome::Sat { exact, .. } => {
                assert_eq!(exact[0].1, Rational::new(1, 3).expect("1/3"));
            }
            other => panic!("expected SAT, got {:?}", other),
        }
    }

    #[test]
    fn eliminated_variables_are_reconstructed() {
        // x + y + z <= 1, x >= 0.4, y >= 0.4, z >= 0.2 -> infeasible
        let infeasible = [
            atom(&[(0, 1.0), (1, 1.0), (2, 1.0)], 1.0, CmpOp::Le),
            atom(&[(0, 1.0)], 0.4, CmpOp::Ge),
            atom(&[(1, 1.0)], 0.4, CmpOp::Ge),
            atom(&[(2, 1.0)], 0.2, CmpOp::Ge),
        ];
        assert!(matches!(solve(&infeasible), Outcome::Unsat { .. }));

        // x + y == 2, x - y == 0 -> x = y = 1
        let determined = [
            atom(&[(0, 1.0), (1, 1.0)], 2.0, CmpOp::Eq),
            atom(&[(0, 1.0), (1, -1.0)], 0.0, CmpOp::Eq),
        ];
        match solve(&determined) {
            Outcome::Sat { exact, .. } => {
                let lookup: BTreeMap<u32, Rational> = exact.into_iter().collect();
                assert_eq!(lookup[&0], Rational::ONE);
                assert_eq!(lookup[&1], Rational::ONE);
            }
            other => panic!("expected SAT at (1, 1), got {:?}", other),
        }
    }

    #[test]
    fn irrelevant_constraints_are_dropped_from_the_core() {
        let constraints = [
            atom(&[(0, 1.0)], 0.6, CmpOp::Ge),
            atom(&[(0, 1.0)], 0.5, CmpOp::Le),
            atom(&[(1, 1.0)], 100.0, CmpOp::Ge),
        ];
        match solve(&constraints) {
            Outcome::Unsat { core } => assert_eq!(core, vec![0, 1]),
            other => panic!("expected UNSAT, got {:?}", other),
        }
    }

    #[test]
    fn pruned_combinations_do_not_flip_unsat() {
        // a + b == 1 (split into two rows), a >= 0.7, b >= 0.7, plus upper rows that used to
        // hide the contradiction from lineage-size pruning.
        let constraints = [
            atom(&[(0, 1.0), (1, 1.0)], 1.0, CmpOp::Eq),
            atom(&[(0, 1.0)], 0.7, CmpOp::Ge),
            atom(&[(1, 1.0)], 0.7, CmpOp::Ge),
            atom(&[(0, 1.0)], 1.0, CmpOp::Le),
            atom(&[(1, 1.0)], 1.0, CmpOp::Le),
        ];
        assert!(matches!(solve(&constraints), Outcome::Unsat { .. }));
    }

    #[test]
    fn unsupported_comparisons_are_unknown() {
        let constraints = [atom(&[(0, 1.0)], 1.0, CmpOp::Ne)];
        assert!(matches!(solve(&constraints), Outcome::Unknown { .. }));
    }
}
