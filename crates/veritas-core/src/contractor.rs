//! Windowed HC4 contractor: the decoding engine that keeps `UNSAT` honest.
//!
//! Mirrors the `bound` / `sum` / `linear` / `compare` revisions in
//! `python/veritas/interval.py` and adds the quadratic forms (`sqr`, `mul` of the same
//! variable) that the Python engine handles through `simplify_clause`'s `square`/`product`
//! shapes. Every revision intersects with outward-rounded arithmetic, so a fixpoint can only
//! remove values that no solution can use. Returning `false` is a *proof* of infeasibility
//! for the constraints known to the contractor; constraints it does not understand are
//! counted, never assumed satisfied.

use crate::expr::CmpOp;
use crate::interval::Interval;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub enum Constraint {
    /// `Σ coeffs[i].1 * x[coeffs[i].0] op rhs`
    Linear {
        coeffs: Vec<(u32, f64)>,
        rhs: f64,
        op: CmpOp,
    },
    /// `Σ x[vars] op rhs`
    Sum { vars: Vec<u32>, rhs: f64, op: CmpOp },
    /// `x[var]^2 op rhs`
    Square { var: u32, rhs: f64, op: CmpOp },
    /// `x[a] * x[b] op rhs`
    Product { a: u32, b: u32, rhs: f64, op: CmpOp },
    /// `xᵀ A x op rhs` where `A = Lᵀ L` is given by its Cholesky factor `L` (lower triangular).
    /// `vars` are the variable ids in order matching the rows/columns of L.
    QuadraticForm {
        vars: Vec<u32>,
        /// Cholesky factor L (lower triangular), row-major: L[i][j] for j <= i
        cholesky_l: Vec<Vec<f64>>,
        rhs: f64,
        op: CmpOp,
    },
}

#[derive(Clone, Debug, Default)]
pub struct Contractor {
    domains: BTreeMap<u32, Interval>,
    constraints: Vec<Constraint>,
    unsupported: usize,
}

impl Contractor {
    pub fn new(domains: BTreeMap<u32, Interval>) -> Contractor {
        Contractor {
            domains,
            constraints: Vec::new(),
            unsupported: 0,
        }
    }

    pub fn push(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    pub fn note_unsupported(&mut self) {
        self.unsupported += 1;
    }

    pub fn unsupported(&self) -> usize {
        self.unsupported
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    pub fn domains(&self) -> &BTreeMap<u32, Interval> {
        &self.domains
    }

    pub fn domain(&self, var: u32) -> Interval {
        *self.domains.get(&var).unwrap_or(&Interval::ENTIRE)
    }

    pub fn set_domain(&mut self, var: u32, interval: Interval) -> bool {
        let current = self.domain(var);
        let narrowed = current.intersect(interval);
        if narrowed.is_empty() {
            return false;
        }
        self.domains.insert(var, narrowed);
        true
    }

    pub fn shrink(&mut self, var: u32, interval: Interval) -> bool {
        self.set_domain(var, interval)
    }

    /// Log volume of the box; `-inf` means a dimension is pinned, `None` means unbounded.
    pub fn log_volume(&self) -> Option<f64> {
        let mut total = 0.0;
        for interval in self.domains.values() {
            total += interval.log_volume()?;
        }
        Some(total)
    }

    /// Run to a fixpoint. `false` means the box is proven empty.
    pub fn propagate(&mut self, max_rounds: usize) -> bool {
        let constraints = self.constraints.clone();
        for _ in 0..max_rounds.max(1) {
            let mut changed = false;
            for constraint in &constraints {
                match revise(constraint, &mut self.domains) {
                    Revise::Empty => return false,
                    Revise::Changed => changed = true,
                    Revise::Stable => {}
                }
            }
            if !changed {
                return true;
            }
        }
        true
    }

    pub fn is_empty(&self) -> bool {
        self.domains.values().any(|interval| interval.is_empty())
    }
}

/// Revision result. `Empty` is a refutation, `Changed` asks for another round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Revise {
    Empty,
    Changed,
    Stable,
}

fn revise(constraint: &Constraint, domains: &mut BTreeMap<u32, Interval>) -> Revise {
    match constraint {
        Constraint::Linear { coeffs, rhs, op } => revise_linear(coeffs, *rhs, *op, domains),
        Constraint::Sum { vars, rhs, op } => {
            let coeffs: Vec<(u32, f64)> = vars.iter().map(|var| (*var, 1.0)).collect();
            revise_linear(&coeffs, *rhs, *op, domains)
        }
        Constraint::Square { var, rhs, op } => {
            if !matches!(
                op,
                CmpOp::Eq | CmpOp::Le | CmpOp::Lt | CmpOp::Ge | CmpOp::Gt
            ) {
                return Revise::Stable;
            }
            let target = Interval::point(*rhs);
            let actual = get(domains, *var).sqr();
            if !compatible(actual, *op, target) {
                return Revise::Empty;
            }
            // Only an upper bound on x^2 narrows the domain to the two-sided root hull;
            // a lower bound (x^2 >= r) admits every point outside it and must not prune.
            if !matches!(op, CmpOp::Eq | CmpOp::Le | CmpOp::Lt) {
                return Revise::Stable;
            }
            let magnitude = Interval::point(rhs.max(0.0)).sqrt();
            let pruned = get(domains, *var).intersect(magnitude.hull(magnitude.neg()));
            if pruned.is_empty() {
                return Revise::Empty;
            }
            if update(domains, *var, pruned) {
                Revise::Changed
            } else {
                Revise::Stable
            }
        }
        Constraint::Product { a, b, rhs, op } => {
            let target = Interval::point(*rhs);
            let actual = get(domains, *a).mul(get(domains, *b));
            if !compatible(actual, *op, target) {
                return Revise::Empty;
            }
            if *op != CmpOp::Eq {
                return Revise::Stable;
            }
            let mut changed = false;
            let narrowed_a = get(domains, *a).intersect(target.div(get(domains, *b)));
            if narrowed_a.is_empty() {
                return Revise::Empty;
            }
            changed |= update(domains, *a, narrowed_a);
            let narrowed_b = get(domains, *b).intersect(target.div(get(domains, *a)));
            if narrowed_b.is_empty() {
                return Revise::Empty;
            }
            changed |= update(domains, *b, narrowed_b);
            if changed {
                Revise::Changed
            } else {
                Revise::Stable
            }
        }
        Constraint::QuadraticForm { vars, cholesky_l, rhs, op } => {
            revise_quadratic_form(vars, cholesky_l, *rhs, *op, domains)
        }
    }
}

fn revise_linear(
    coeffs: &[(u32, f64)],
    rhs: f64,
    op: CmpOp,
    domains: &mut BTreeMap<u32, Interval>,
) -> Revise {
    if coeffs.is_empty() {
        let only = Interval::point(0.0);
        return if compatible(only, op, Interval::point(rhs)) {
            Revise::Stable
        } else {
            Revise::Empty
        };
    }
    let target = Interval::point(rhs);
    // Forward check first: it refutes whole boxes cheaply.
    let mut total = Interval::point(0.0);
    for (var, coeff) in coeffs {
        total = total.add(get(domains, *var).mul(Interval::point(*coeff)));
    }
    if !compatible(total, op, target) {
        return Revise::Empty;
    }
    if op == CmpOp::Eq {
        // Narrow only on equality: inequalities need the directional inverse to stay sound
        // and are already covered by the forward test above.
        let mut changed = false;
        for (index, (var, coeff)) in coeffs.iter().enumerate() {
            let mut rest = Interval::point(0.0);
            for (other_index, (other_var, other_coeff)) in coeffs.iter().enumerate() {
                if other_index == index {
                    continue;
                }
                rest = rest.add(get(domains, *other_var).mul(Interval::point(*other_coeff)));
            }
            let narrowed =
                get(domains, *var).intersect(target.sub(rest).div(Interval::point(*coeff)));
            if narrowed.is_empty() {
                return Revise::Empty;
            }
            changed |= update(domains, *var, narrowed);
        }
        return if changed {
            Revise::Changed
        } else {
            Revise::Stable
        };
    }
    Revise::Stable
}

/// Helper for quadratic revision: y = Lᵀ x (L is lower-triangular), so xᵀ A x = ||y||²
/// where A = L Lᵀ.  Interval arithmetic with outward rounding keeps the check sound.
fn apply_lt(domains: &BTreeMap<u32, Interval>, vars: &[u32], cholesky_l: &[Vec<f64>]) -> Option<Vec<Interval>> {
    // y = Lᵀ x (L is lower-triangular): y_i = Σ_{j≥i} L[j][i] * x[j].
    let n = vars.len();
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc = Interval::point(0.0);
        for j in i..n {
            if i >= cholesky_l[j].len() {
                return None;
            }
            let coeff = cholesky_l[j][i];
            if coeff == 0.0 {
                continue;
            }
            acc = acc.add(get(domains, vars[j]).mul(Interval::point(coeff)));
        }
        y.push(acc);
    }
    Some(y)
}

fn sum_square(terms: &[Interval]) -> Interval {
    let mut acc = Interval::point(0.0);
    for term in terms {
        acc = acc.add(term.sqr());
    }
    acc
}

fn revise_quadratic_form(
    vars: &[u32],
    cholesky_l: &[Vec<f64>],
    rhs: f64,
    op: CmpOp,
    domains: &mut BTreeMap<u32, Interval>,
) -> Revise {
    let n = vars.len();
    if n == 0 || cholesky_l.len() != n {
        return Revise::Stable;
    }
    // Forward check: xᵀ A x = ||Lᵀ x||² must be compatible with the comparison.  This is a
    // *sound* necessary condition — it never removes a feasible value, it only proves
    // emptiness.  (The previous sampled-point equality narrowing could exclude feasible
    // assignments and has been removed.)
    let Some(y) = apply_lt(domains, vars, cholesky_l) else {
        return Revise::Stable;
    };
    let total = sum_square(&y);
    let target = Interval::point(rhs);
    if !compatible(total, op, target) {
        return Revise::Empty;
    }
    let _ = domains; // no narrowing: forward check only (sound, outward-rounded)
    Revise::Stable
}

fn compatible(actual: Interval, op: CmpOp, target: Interval) -> bool {
    if actual.is_empty() {
        return false;
    }
    match op {
        CmpOp::Eq | CmpOp::In => !target.intersect(actual).is_empty(),
        CmpOp::Ne | CmpOp::NotIn => !(actual.width() == 0.0 && target.contains(actual.lo)),
        CmpOp::Le => actual.lo <= target.hi,
        CmpOp::Lt => actual.lo < target.hi,
        CmpOp::Ge => actual.hi >= target.lo,
        CmpOp::Gt => actual.hi > target.lo,
    }
}

fn get(domains: &BTreeMap<u32, Interval>, var: u32) -> Interval {
    *domains.get(&var).unwrap_or(&Interval::ENTIRE)
}

/// Returns whether the domain actually changed.
fn update(domains: &mut BTreeMap<u32, Interval>, var: u32, narrowed: Interval) -> bool {
    let current = get(domains, var);
    if narrowed.lo > current.lo || narrowed.hi < current.hi {
        domains.insert(var, narrowed);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(values: &[(u32, f64, f64)]) -> BTreeMap<u32, Interval> {
        let mut domains = BTreeMap::new();
        for (var, lo, hi) in values {
            domains.insert(*var, Interval::new(*lo, *hi));
        }
        domains
    }

    #[test]
    fn equality_contracts_both_directions() {
        // Blueprint example: w1 + w2 + w3 == 1 with w_i in [0.05, 0.50]; pinning w1 = 0.10
        // must narrow w2 and w3 to [0.40, 0.50].
        let mut contractor = Contractor::new(boxed(&[
            (0, 0.05, 0.5),
            (1, 0.05, 0.5),
            (2, 0.05, 0.5),
        ]));
        contractor.push(Constraint::Sum {
            vars: vec![0, 1, 2],
            rhs: 1.0,
            op: CmpOp::Eq,
        });
        assert!(contractor.set_domain(0, Interval::point(0.1)));
        assert!(contractor.propagate(16));
        let second = contractor.domain(1);
        assert!(second.lo >= 0.39 && second.lo <= 0.400_000_000_1, "lo = {}", second.lo);
        assert!(second.hi <= 0.5);
        let third = contractor.domain(2);
        assert!(third.lo >= 0.39 && third.hi <= 0.5);

        // The same box refutes an impossible pin before any search starts.
        let mut impossible = Contractor::new(boxed(&[(0, 0.05, 0.5), (1, 0.05, 0.5)]));
        impossible.push(Constraint::Sum {
            vars: vec![0, 1],
            rhs: 1.0,
            op: CmpOp::Eq,
        });
        assert!(impossible.set_domain(0, Interval::point(0.1)));
        assert!(!impossible.propagate(16));
    }

    #[test]
    fn refutes_infeasible_systems() {
        let mut contractor = Contractor::new(boxed(&[(0, 0.8, 1.0)]));
        contractor.push(Constraint::Linear {
            coeffs: vec![(0, 3.0)],
            rhs: 2.0,
            op: CmpOp::Eq,
        });
        // 3x == 2 needs x = 2/3, which is outside [0.8, 1.0].
        assert!(!contractor.propagate(8));

        let mut quadratic = Contractor::new(boxed(&[(0, 2.0, 3.0)]));
        quadratic.push(Constraint::Square {
            var: 0,
            rhs: 1.0,
            op: CmpOp::Eq,
        });
        assert!(!quadratic.propagate(8));
    }

    #[test]
    fn quadratic_contracts_the_negative_branch() {
        let mut contractor = Contractor::new(boxed(&[(0, -3.0, 3.0)]));
        contractor.push(Constraint::Square {
            var: 0,
            rhs: 0.25,
            op: CmpOp::Eq,
        });
        assert!(contractor.propagate(8));
        let domain = contractor.domain(0);
        assert!(domain.contains(0.5) && domain.contains(-0.5));
        assert!(!domain.contains(1.0));
    }

    #[test]
    fn product_contracts_and_division_by_zero_widens() {
        let mut contractor = Contractor::new(boxed(&[(0, 1.0, 2.0), (1, 2.0, 4.0)]));
        contractor.push(Constraint::Product {
            a: 0,
            b: 1,
            rhs: 4.0,
            op: CmpOp::Eq,
        });
        assert!(contractor.propagate(8));
        assert!(contractor.domain(0).contains(1.0));
        assert!(contractor.domain(0).hi <= 2.0);

        // While b still spans zero, dividing by it must widen rather than narrow; once b has
        // been narrowed to [0.75, 1] the a-domain follows soundly (a = 1.5 / b).
        let mut boxed_product = Contractor::new(boxed(&[(0, 1.0, 2.0), (1, -1.0, 1.0)]));
        boxed_product.push(Constraint::Product {
            a: 0,
            b: 1,
            rhs: 1.5,
            op: CmpOp::Eq,
        });
        assert!(boxed_product.propagate(4));
        let a = boxed_product.domain(0);
        let b = boxed_product.domain(1);
        assert!(a.contains(1.5) && a.contains(2.0), "a = {:?}", a);
        assert!(a.lo >= 1.5 - 1e-9, "a must stay inside 1.5/b for b in [0.75, 1]");
        assert!(b.contains(0.75) && b.hi <= 1.0 + 1e-9, "b = {:?}", b);
    }

    #[test]
    fn unknown_constraints_are_counted_not_assumed() {
        let mut contractor = Contractor::new(boxed(&[(0, 0.0, 1.0)]));
        contractor.note_unsupported();
        contractor.note_unsupported();
        assert_eq!(contractor.unsupported(), 2);
        assert!(contractor.propagate(4));
    }
}
