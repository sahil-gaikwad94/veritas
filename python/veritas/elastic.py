"""Minimal relaxation suggestions for infeasible schemas (``veritas explain-unsat``).

Given a compiled spec whose linear relaxation is UNSAT, locate the conflicting clauses and
compute, for each one, the smallest additive slack ``delta >= 0`` that would make the system
feasible when only that clause is relaxed.  ``ge``-type clauses are relaxed by lowering the
bound, ``le``-type clauses by raising it, and ``eq`` clauses in both directions.

These are **conservative** suggestions: tightening two clauses jointly may need less total
slack than any single-clause relaxation, and the deltas are computed with the outward-closed
normalisation of :func:`veritas.linear.simplify_clause` (strict inequalities are treated as
closed, so a suggested delta can only underestimate the true requirement by epsilon).  They are
a repair wizard, not a theorem prover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .compiler.vir import VirSpec
from .linear import compile_clauses, simplify_clause
from .nano_solver import SolverStatus, VeritasNanoSolver

__all__ = ["Relaxation", "suggest_relaxations"]


@dataclass
class Relaxation:
    clause_index: int
    op: str
    rhs: float
    direction: str  # "raise" | "lower"
    delta: float
    new_rhs: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause": self.clause_index,
            "op": self.op,
            "rhs": self.rhs,
            "direction": self.direction,
            "delta": round(self.delta, 12),
            "new_rhs": self.new_rhs,
        }


def _check(spec: VirSpec, relax_index: int | None, delta: float) -> SolverStatus:
    solver = VeritasNanoSolver()
    for clause in compile_clauses(spec):
        simple = simplify_clause(clause)
        if simple is None:
            continue
        op, coeffs, rhs = simple
        if clause.index == relax_index:
            if op == "le":
                rhs, delta_sign = rhs + delta, +1.0
            elif op == "ge":
                rhs, delta_sign = rhs - delta, -1.0
            elif op == "eq":
                # eq assumed to block in its binding direction; relax toward both sides
                # by treating it as two inequalities and relaxing each independently.
                solver.assert_and_track(("le", coeffs, rhs + delta), (clause.index, "eq+"))
                solver.assert_and_track(("ge", coeffs, rhs - delta), (clause.index, "eq-"))
                continue
            else:
                continue
            _ = delta_sign
        if op == "eq":
            solver.assert_and_track(("le", coeffs, rhs), (clause.index, "eq+"))
            solver.assert_and_track(("ge", coeffs, rhs), (clause.index, "eq-"))
        else:
            solver.assert_and_track((op, coeffs, rhs), clause.index)
    # Type bounds: the *declared* bounds of the model, never the constraint-contracted box
    # (``initial_domains`` folds clauses back in, which would re-impose the infeasibility).
    for vid, info in spec.variables.items():
        vtype = info["type"]
        kind = getattr(vtype, "kind", None)
        if kind not in ("real", "int", "bool"):
            continue
        if kind == "bool":
            lo, hi = 0.0, 1.0
            lo_ex, hi_ex = False, False
        else:
            lo = getattr(vtype, "min", None)
            hi = getattr(vtype, "max", None)
            lo_ex = bool(getattr(vtype, "min_exclusive", False))
            hi_ex = bool(getattr(vtype, "max_exclusive", False))
        if lo is not None and math.isfinite(float(lo)):
            solver.assert_and_track(("gt" if lo_ex else "ge", {vid: 1}, float(lo)), ("type", vid))
        if hi is not None and math.isfinite(float(hi)):
            solver.assert_and_track(("lt" if hi_ex else "le", {vid: 1}, float(hi)), ("type", vid))
    return solver.check()


def _min_delta(spec: VirSpec, relax_index: int, op: str, upper: float) -> float | None:
    """Bisect the smallest delta in ``[0, upper]`` making the system SAT; None if none."""

    lo, hi = 0.0, upper
    if _check(spec, relax_index, hi) != SolverStatus.SAT:
        return float("inf")  # no finite single-clause relaxation repairs this core
    if _check(spec, relax_index, lo) == SolverStatus.SAT:
        return 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _check(spec, relax_index, mid) == SolverStatus.SAT:
            hi = mid
        else:
            lo = mid
    return hi


def suggest_relaxations(spec: VirSpec, core: list[int]) -> list[Relaxation]:
    suggestions: list[Relaxation] = []
    clauses = compile_clauses(spec)
    if len(spec.variables) > 12:
        # Too large for repeated Fourier-Motzkin probes: report the core only.
        return [
            Relaxation(int(i), "?", float("nan"), "?", float("inf"), float("nan"))
            for i in sorted({int(i) for i in core if isinstance(i, int)})
        ]
    for index in sorted({int(i) for i in core if isinstance(i, int)}):
        if index < 0 or index >= len(clauses):
            continue
        clause = clauses[index]
        simple = simplify_clause(clause)
        if simple is None:
            continue
        op, _coeffs, rhs = simple
        if op == "eq":
            direction = "widen"  # eq relaxation widens both sides of the equality
        else:
            direction = "raise" if op == "le" else "lower"
        delta = _min_delta(spec, index, op, 1e9)
        new_rhs = rhs if direction == "widen" else (rhs + delta if direction == "raise" else rhs - delta)
        if not math.isfinite(delta):
            suggestions.append(Relaxation(index, op, rhs, direction, float("inf"), float("nan")))
            continue
        suggestions.append(Relaxation(index, op, rhs, direction, delta, new_rhs))
    return suggestions


def explain_unsat(spec: VirSpec) -> dict[str, Any]:
    """Full explain-unsat report: preflight status, core, per-clause minimal slack.

    Also surfaces exact 1-D bound conflicts from the transpiler-tightened field bounds (an
    empty ``[lo, hi]`` window cannot be repaired by relaxing any clause).
    """

    from .compiler.preflight import _bound_conflicts, run_preflight

    result = run_preflight(spec)
    report: dict[str, Any] = {
        "status": result.status,
        "conflicts": list(result.conflicts),
        "details": list(result.details),
        "suggestions": [],
        "bound_conflicts": [],
    }
    if result.status != "UNSAT":
        return report
    report["bound_conflicts"] = [
        {"clause": index, "detail": reason} for index, reason in _bound_conflicts(spec)
    ]
    core = [int(i) for i in result.conflicts if isinstance(i, int) and i >= 0]
    suggestions = suggest_relaxations(spec, core)
    report["suggestions"] = [item.to_dict() for item in suggestions]
    report["any_single_clause_repair"] = any(math.isfinite(item.delta) for item in suggestions)
    report["all_repairs_need_joint_or_structural_fix"] = bool(suggestions) and not report["any_single_clause_repair"]
    return report
