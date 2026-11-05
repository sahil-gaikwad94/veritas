//! Incremental solver front end: push/pop scopes, tracked assertions, cores.
//!
//! Port of `python/vexx/nano_solver.py`, sharing the exact Fourier–Motzkin engine in
//! [`crate::solver`] so `UNSAT` is a proof and `SAT` carries a witness. Two behaviours matter
//! for decoding and are covered by tests:
//!
//! * **scoped assertions** — a rejected branch pops its constraints and the solver returns to
//!   exactly the previous state (no leaking bounds);
//! * **tracked cores** — every assertion can carry a tracker id, and the core is reported in
//!   tracker terms, because the decoder reasons about clauses, not about interned rows.
//!
//! The pure difference fragment (`x - y <= c`) also gets an incremental Bellman-Ford check,
//! which reports the *cycle* that caused the conflict: a much smaller explanation than a
//! general linear core.

use crate::expr::CmpOp;
use crate::solver::{self, Linear, Outcome};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverStatus {
    Sat,
    Unsat,
    Unknown,
}

impl SolverStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            SolverStatus::Sat => "sat",
            SolverStatus::Unsat => "unsat",
            SolverStatus::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug)]
struct Tracked {
    constraint: Linear,
    tracker: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct VexxNanoSolver {
    frames: Vec<Vec<Tracked>>,
    status: SolverStatus,
    core: Vec<u64>,
    model: Vec<(u32, f64)>,
    reason: String,
    max_variables: usize,
}

impl Default for VexxNanoSolver {
    fn default() -> VexxNanoSolver {
        VexxNanoSolver::new()
    }
}

impl VexxNanoSolver {
    pub fn new() -> VexxNanoSolver {
        VexxNanoSolver {
            frames: vec![Vec::new()],
            status: SolverStatus::Unknown,
            core: Vec::new(),
            model: Vec::new(),
            reason: "not checked".to_string(),
            max_variables: solver::MAX_VARS,
        }
    }

    /// Open a scope; every assertion until the matching [`VexxNanoSolver::pop`] is dropped
    /// with it.
    pub fn push(&mut self) {
        self.frames.push(Vec::new());
    }

    /// Close `levels` scopes. The base scope cannot be popped.
    pub fn pop(&mut self, levels: usize) -> bool {
        let mut closed = 0;
        while closed < levels && self.frames.len() > 1 {
            self.frames.pop();
            closed += 1;
        }
        let closed_any = closed > 0;
        if closed_any {
            self.status = SolverStatus::Unknown;
            self.reason = "scopes changed".to_string();
        }
        closed_any
    }

    pub fn depth(&self) -> usize {
        self.frames.len() - 1
    }

    pub fn assert_and_track(&mut self, constraint: Linear, tracker: Option<u64>) {
        if let Some(frame) = self.frames.last_mut() {
            frame.push(Tracked {
                constraint,
                tracker,
            });
        }
    }

    /// `var >= lo` and/or `var <= hi`.
    pub fn add_bound(&mut self, var: u32, lo: Option<f64>, hi: Option<f64>) {
        if let Some(bound) = lo {
            self.assert_and_track(
                Linear {
                    coeffs: vec![(var, 1.0)],
                    rhs: bound,
                    op: CmpOp::Ge,
                },
                None,
            );
        }
        if let Some(bound) = hi {
            self.assert_and_track(
                Linear {
                    coeffs: vec![(var, 1.0)],
                    rhs: bound,
                    op: CmpOp::Le,
                },
                None,
            );
        }
    }

    /// `x - y <= offset`.
    pub fn add_difference(&mut self, x: u32, y: u32, offset: f64) {
        self.assert_and_track(
            Linear {
                coeffs: vec![(x, 1.0), (y, -1.0)],
                rhs: offset,
                op: CmpOp::Le,
            },
            None,
        );
    }

    fn flatten(&self) -> Vec<&Tracked> {
        self.frames
            .iter()
            .flat_map(|frame| frame.iter())
            .collect()
    }

    pub fn check(&mut self) -> SolverStatus {
        // Everything borrowed from `self` is resolved inside this block, so the borrow ends
        // before the state is updated.
        let computed = {
            let tracked = self.flatten();
            let constraints: Vec<Linear> = tracked
                .iter()
                .map(|item| item.constraint.clone())
                .collect();
            let variables: BTreeSet<u32> = constraints
                .iter()
                .flat_map(|constraint| constraint.coeffs.iter().map(|(var, _)| *var))
                .collect();
            if variables.len() > self.max_variables {
                Err(format!(
                    "{} variables is more than the solver attempts",
                    variables.len()
                ))
            } else {
                match solver::solve(&constraints) {
                    Outcome::Sat { model, .. } => Ok((SolverStatus::Sat, model, Vec::new(), String::new())),
                    Outcome::Unsat { core } => Ok((
                        SolverStatus::Unsat,
                        Vec::new(),
                        core.into_iter()
                            .filter_map(|index| tracked.get(index).and_then(|item| item.tracker))
                            .collect(),
                        String::new(),
                    )),
                    Outcome::Unknown { reason } => {
                        Ok((SolverStatus::Unknown, Vec::new(), Vec::new(), reason))
                    }
                }
            }
        };
        match computed {
            Ok((status, model, core, reason)) => {
                self.status = status;
                self.model = model;
                self.core = core;
                self.reason = reason;
            }
            Err(reason) => {
                self.status = SolverStatus::Unknown;
                self.reason = reason;
                self.core.clear();
                self.model.clear();
            }
        }
        self.status
    }

    pub fn status(&self) -> SolverStatus {
        self.status
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }

    pub fn get_unsat_core(&self) -> &[u64] {
        &self.core
    }

    pub fn get_model(&self) -> &[(u32, f64)] {
        &self.model
    }

    pub fn get_float_model(&self) -> BTreeMap<u32, f64> {
        self.model.iter().copied().collect()
    }
}

/// `y - x <= weight` as an edge of the difference-constraint graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiffEdge {
    pub x: u32,
    pub y: u32,
    pub weight: i64,
    pub tracker: Option<u64>,
}

/// Bellman-Ford: return the edges of a negative cycle, i.e. the reason the difference system
/// `y - x <= weight` has no solution.
pub fn negative_cycle(edges: &[DiffEdge]) -> Option<Vec<usize>> {
    let mut nodes: BTreeSet<u32> = BTreeSet::new();
    for edge in edges {
        nodes.insert(edge.x);
        nodes.insert(edge.y);
    }
    if nodes.is_empty() {
        return None;
    }
    let mut distance: BTreeMap<u32, i64> = nodes.iter().map(|node| (*node, 0)).collect();
    let mut previous: BTreeMap<u32, (u32, usize)> = BTreeMap::new();
    let mut changed: Option<u32> = None;
    for _ in 0..nodes.len() {
        changed = None;
        for (index, edge) in edges.iter().enumerate() {
            let from = distance[&edge.x];
            let candidate = from.checked_add(edge.weight)?;
            if candidate < distance[&edge.y] {
                distance.insert(edge.y, candidate);
                previous.insert(edge.y, (edge.x, index));
                changed = Some(edge.y);
            }
        }
        changed?;
    }
    let start = changed?;
    // Walk back far enough to be inside the cycle, then collect it.
    let mut node = start;
    for _ in 0..nodes.len() {
        node = previous.get(&node).map(|(from, _)| *from)?;
    }
    let mut cycle: Vec<usize> = Vec::new();
    let mut current = node;
    loop {
        let (from, edge_index) = previous.get(&current)?;
        cycle.push(*edge_index);
        current = *from;
        if current == node {
            break;
        }
        if cycle.len() > edges.len() + 1 {
            return None;
        }
    }
    cycle.reverse();
    Some(cycle)
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
    fn pops_restore_the_previous_state_exactly() {
        let mut engine = VexxNanoSolver::new();
        engine.add_bound(0, Some(0.0), Some(1.0));
        assert_eq!(engine.check(), SolverStatus::Sat);

        engine.push();
        engine.assert_and_track(atom(&[(0, 1.0)], 5.0, CmpOp::Ge), Some(7));
        assert_eq!(engine.check(), SolverStatus::Unsat);
        assert_eq!(engine.get_unsat_core(), &[7]);
        assert!(engine.pop(1));

        // Back to the outer state: the impossible bound is gone.
        assert_eq!(engine.check(), SolverStatus::Sat);
        let model = engine.get_float_model();
        assert!(model[&0] >= -1e-9 && model[&0] <= 1.0 + 1e-9);
    }

    #[test]
    fn tracked_cores_name_the_clauses_not_the_rows() {
        let mut engine = VexxNanoSolver::new();
        engine.assert_and_track(atom(&[(0, 1.0)], 0.6, CmpOp::Ge), Some(11));
        engine.assert_and_track(atom(&[(0, 1.0)], 0.5, CmpOp::Le), Some(12));
        engine.assert_and_track(atom(&[(1, 1.0)], 3.0, CmpOp::Ge), Some(13));
        assert_eq!(engine.check(), SolverStatus::Unsat);
        assert_eq!(engine.get_unsat_core(), &[11, 12]);
    }

    #[test]
    fn the_base_scope_cannot_be_popped() {
        let mut engine = VexxNanoSolver::new();
        assert!(!engine.pop(1));
        assert_eq!(engine.depth(), 0);
        engine.push();
        assert!(engine.pop(4));
        assert_eq!(engine.depth(), 0);
    }

    #[test]
    fn difference_logic_reports_the_cycle() {
        let edges = [
            DiffEdge { x: 0, y: 1, weight: 1, tracker: Some(1) },  // y - x <= 1
            DiffEdge { x: 1, y: 0, weight: -2, tracker: Some(2) }, // x - y <= -2
        ];
        let cycle = negative_cycle(&edges).expect("negative cycle");
        assert_eq!(cycle.len(), 2);
        assert!(cycle.contains(&0) && cycle.contains(&1));

        let feasible = [
            DiffEdge { x: 0, y: 1, weight: 3, tracker: None },
            DiffEdge { x: 1, y: 0, weight: -2, tracker: None },
        ];
        assert!(negative_cycle(&feasible).is_none());
        assert!(negative_cycle(&[]).is_none());
    }

    #[test]
    fn difference_systems_are_also_solved_by_the_linear_engine() {
        let mut engine = VexxNanoSolver::new();
        engine.add_difference(1, 0, 1.0); // y - x <= 1
        engine.add_difference(0, 1, -2.0); // x - y <= -2
        assert_eq!(engine.check(), SolverStatus::Unsat);
    }

    #[test]
    fn too_many_variables_is_unknown_not_unsat() {
        let mut engine = VexxNanoSolver::new();
        for var in 0..(crate::solver::MAX_VARS as u32 + 2) {
            engine.add_bound(var, Some(0.0), None);
        }
        assert_eq!(engine.check(), SolverStatus::Unknown);
        assert!(!engine.reason().is_empty());
    }
}
