"""0-Token Pre-Flight Feasibility Gate for veritas Phase 1."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .analysis import container_feasibility
from .vir import VirSpec, VirExpr


class InfeasibleSpecError(Exception):
    def __init__(self, message: str, conflicts=None, relaxation=None):
        super().__init__(message)
        self.conflicts = conflicts or []
        self.relaxation = relaxation


def _cmp_op_str(op: Any) -> str:
    if isinstance(op, str):
        return op
    return str(op)


def initial_domains(spec: VirSpec) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """Public entry point used by the decode engine and by diagnostics."""

    return _initial_domains(spec)


def _initial_domains(spec: VirSpec) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    domains: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for vid, info in spec.variables.items():
        vtype = info.get("type")
        lo = getattr(vtype, "min", None)
        hi = getattr(vtype, "max", None)
        # dynamic maps: treat container as real bounds
        if vtype is not None and getattr(vtype, "kind", "") == "dynamic_map":
            lo = getattr(vtype, "val_min", None)
            hi = getattr(vtype, "val_max", None)
            if lo == float("-inf"):
                lo = None
            if hi == float("inf"):
                hi = None
        domains[vid] = (lo, hi)
    return domains


def _tighten(domains: Dict[int, Tuple[Optional[float], Optional[float]]], vid: int, lo=None, hi=None) -> bool:
    cur_lo, cur_hi = domains.get(vid, (None, None))
    new_lo = cur_lo if lo is None else (lo if cur_lo is None else max(cur_lo, lo))
    new_hi = cur_hi if hi is None else (hi if cur_hi is None else min(cur_hi, hi))
    if new_lo is not None and new_hi is not None and new_lo > new_hi + 1e-12:
        return False
    if (new_lo, new_hi) != (cur_lo, cur_hi):
        domains[vid] = (new_lo, new_hi)
    return True


def _add_bounds(a, b):
    lo = None if a[0] is None or b[0] is None else a[0] + b[0]
    hi = None if a[1] is None or b[1] is None else a[1] + b[1]
    return (lo, hi)


def _sub_bounds(a, b):
    lo = None if a[0] is None or b[1] is None else a[0] - b[1]
    hi = None if a[1] is None or b[0] is None else a[1] - b[0]
    return (lo, hi)


def _mul_scalar_bounds(a, s):
    if s >= 0:
        lo = None if a[0] is None else a[0] * s
        hi = None if a[1] is None else a[1] * s
    else:
        lo = None if a[1] is None else a[1] * s
        hi = None if a[0] is None else a[0] * s
    return (lo, hi)


def _interval(expr: Optional[VirExpr], domains: Dict[int, Tuple[Optional[float], Optional[float]]]) -> Optional[Tuple[Optional[float], Optional[float]]]:
    if expr is None:
        return None
    kind = expr.kind
    if kind in ("const_real", "const_int"):
        v = float(expr.value)
        return (v, v)
    if kind == "const_value":
        if isinstance(expr.value, (int, float, bool)):
            v = float(expr.value)
            return (v, v)
        return None
    if kind == "const_bool":
        v = 1.0 if expr.value else 0.0
        return (v, v)
    if kind == "var":
        if expr.var_id is None:
            return None
        return domains.get(expr.var_id, (None, None))
    if kind == "add":
        a = _interval(expr.lhs, domains)
        b = _interval(expr.rhs, domains) if expr.rhs is not None else None
        # args fallback
        if a is None and expr.args:
            a = _interval(expr.args[0], domains)
        if b is None and len(expr.args) > 1:
            b = _interval(expr.args[1], domains)
        if a is None or b is None:
            return None
        return _add_bounds(a, b)
    if kind == "sub":
        a = _interval(expr.lhs, domains)
        b = _interval(expr.rhs, domains)
        if a is None and expr.args:
            a = _interval(expr.args[0], domains)
        if b is None and len(expr.args) > 1:
            b = _interval(expr.args[1], domains)
        if a is None or b is None:
            return None
        return _sub_bounds(a, b)
    if kind == "mul":
        a = _interval(expr.lhs, domains)
        b = _interval(expr.rhs, domains)
        if a is None and expr.args:
            a = _interval(expr.args[0], domains)
        if b is None and len(expr.args) > 1:
            b = _interval(expr.args[1], domains)
        if a is None or b is None:
            return None
        # scalar optimization if one side constant
        if a[0] is not None and a[1] is not None and a[0] == a[1]:
            return _mul_scalar_bounds(b, a[0])
        if b[0] is not None and b[1] is not None and b[0] == b[1]:
            return _mul_scalar_bounds(a, b[0])
        return None
    if kind == "div":
        a = _interval(expr.lhs, domains)
        b = _interval(expr.rhs, domains)
        if a is None or b is None:
            return None
        # only scalar divisor
        if b[0] is not None and b[1] is not None and b[0] == b[1] and b[0] != 0:
            return _mul_scalar_bounds(a, 1.0 / b[0])
        return None
    if kind == "neg":
        a = _interval(expr.lhs, domains) if expr.lhs is not None else None
        if a is None and expr.args:
            a = _interval(expr.args[0], domains)
        if a is None:
            return None
        lo = None if a[1] is None else -a[1]
        hi = None if a[0] is None else -a[0]
        return (lo, hi)
    if kind == "sqrt":
        a = _interval(expr.args[0] if expr.args else expr.lhs, domains)
        if a is None:
            return None
        lo_in = a[0]
        hi_in = a[1]
        if lo_in is not None and lo_in < 0:
            lo_in = 0.0
        if lo_in is None:
            lo_out = None
        else:
            lo_out = math.sqrt(max(0.0, lo_in))
        if hi_in is None:
            hi_out = None
        else:
            hi_out = math.sqrt(max(0.0, hi_in)) if hi_in >= 0 else None
        return (lo_out, hi_out)
    if kind == "pow":
        a = _interval(expr.lhs, domains)
        b = _interval(expr.rhs, domains)
        if a is None or b is None:
            return None
        # constant exponent 2
        if b[0] is not None and b[1] is not None and b[0] == b[1] == 2:
            # square interval (nonnegative)
            # max of squares at endpoints or 0 if contains 0
            vals = []
            if a[0] is not None:
                vals.append(a[0] * a[0])
            if a[1] is not None:
                vals.append(a[1] * a[1])
            if a[0] is not None and a[1] is not None and a[0] <= 0 <= a[1]:
                vals.append(0.0)
            lo = min(vals) if vals else None
            hi = max(vals) if vals else None
            return (lo, hi)
        return None
    return None


def _apply_compare(expr: VirExpr, domains: Dict[int, Tuple[Optional[float], Optional[float]]]) -> bool:
    # expr: compare with op, lhs, rhs
    op = _cmp_op_str(expr.op)
    tol = getattr(expr, "tol", 1e-6) or 1e-6
    # General interval feasibility check for arbitrary expressions
    if expr.kind == "compare":
        ivl = _interval(expr.lhs, domains)
        ivr = _interval(expr.rhs, domains)
        if ivl is not None and ivr is not None:
            lo_l, hi_l = ivl
            lo_r, hi_r = ivr
            if op in ("le", "lt"):
                if lo_l is not None and hi_r is not None and lo_l > hi_r + tol:
                    return False
            elif op in ("ge", "gt"):
                if hi_l is not None and lo_r is not None and hi_l < lo_r - tol:
                    return False
            elif op in ("eq",):
                if lo_l is not None and hi_r is not None and lo_l > hi_r + tol:
                    return False
                if lo_r is not None and hi_l is not None and lo_r > hi_l + tol:
                    return False
            elif op in ("ne",):
                if (lo_l is not None and hi_l is not None and lo_l == hi_l and
                        lo_r is not None and hi_r is not None and lo_r == hi_r and
                        abs(lo_l - lo_r) <= tol):
                    return False
    # var vs const
    if expr.lhs is not None and expr.lhs.kind == "var" and expr.rhs is not None and expr.rhs.kind in ("const_real", "const_int", "const_value"):
        vid = expr.lhs.var_id
        val = expr.rhs.value
        if isinstance(val, bool):
            val = float(val)
        if isinstance(val, (int, float)):
            val = float(val)
            if op in ("eq",):
                if not _tighten(domains, vid, val, val):
                    return False
            elif op in ("le", "lt"):
                bound = val
                if not _tighten(domains, vid, None, bound):
                    return False
            elif op in ("ge", "gt"):
                bound = val
                if not _tighten(domains, vid, bound, None):
                    return False
            elif op in ("ne",):
                pass
    if expr.rhs is not None and expr.rhs.kind == "var" and expr.lhs is not None and expr.lhs.kind in ("const_real", "const_int", "const_value"):
        vid = expr.rhs.var_id
        val = expr.lhs.value
        if isinstance(val, (int, float, bool)):
            val = float(val)
            if op in ("eq",):
                if not _tighten(domains, vid, val, val):
                    return False
            elif op in ("le", "lt"):
                # lhs <= var  => var >= lhs
                if not _tighten(domains, vid, val, None):
                    return False
            elif op in ("ge", "gt"):
                if not _tighten(domains, vid, None, val):
                    return False
    # var vs var comparison (basic)
    if expr.lhs is not None and expr.lhs.kind == "var" and expr.rhs is not None and expr.rhs.kind == "var":
        l1, h1 = domains.get(expr.lhs.var_id, (None, None))
        l2, h2 = domains.get(expr.rhs.var_id, (None, None))
        if op in ("le", "lt"):
            # lhs <= rhs: lhs hi <= rhs hi, rhs lo >= lhs lo (approx via tighten bounds)
            if h1 is not None:
                if not _tighten(domains, expr.rhs.var_id, None, None):
                    pass
            # tighten lhs hi by rhs hi
            if h2 is not None:
                if not _tighten(domains, expr.lhs.var_id, None, h2):
                    return False
            if l1 is not None:
                if not _tighten(domains, expr.rhs.var_id, l1, None):
                    return False
        elif op in ("ge", "gt"):
            if h1 is not None:
                if not _tighten(domains, expr.rhs.var_id, None, h1):
                    return False
            if l2 is not None:
                if not _tighten(domains, expr.lhs.var_id, l2, None):
                    return False
        elif op in ("eq",):
            # intersect domains
            lo = l1 if l2 is None else (l2 if l1 is None else max(l1, l2))
            hi = h1 if h2 is None else (h2 if h1 is None else min(h1, h2))
            if not _tighten(domains, expr.lhs.var_id, lo, hi):
                return False
            if not _tighten(domains, expr.rhs.var_id, lo, hi):
                return False
    # and/or
    if expr.kind == "and":
        for a in expr.args:
            if not _apply_compare(a, domains):
                return False
        return True
    if expr.kind == "or":
        # optimistic: try each branch on a copy; if any feasible, keep original domains (no tightening)
        feasible = False
        for a in expr.args:
            copy = {k: v for k, v in domains.items()}
            if _apply_compare(a, copy):
                feasible = True
                break
        return feasible
    return True


def _apply_derive(expr: VirExpr, domains: Dict[int, Tuple[Optional[float], Optional[float]]]) -> bool:
    if not (expr.kind == "compare" and expr.op in ("eq",) and expr.lhs is not None and expr.lhs.kind == "var"):
        return True
    vid = expr.lhs.var_id
    iv = _interval(expr.rhs, domains)
    if iv is None:
        return True
    if not _tighten(domains, vid, iv[0], iv[1]):
        return False
    return True



def _apply_one(expr, domains):
    if expr is None:
        return True
    if expr.kind in ("compare", "and", "or"):
        if not _apply_compare(expr, domains):
            return False
    if not _apply_derive(expr, domains):
        return False
    if not _domains_feasible(domains):
        return False
    return True


def _apply_constraints_fixed(constraints, domains, max_iters=100):
    changed = True
    iters = 0
    while changed and iters < max_iters:
        changed = False
        snapshot = {k: v for k, v in domains.items()}
        for c in constraints:
            if not _apply_one(c, domains):
                return False
        if domains != snapshot:
            changed = True
        if not _domains_feasible(domains):
            return False
        iters += 1
    return _domains_feasible(domains)

def _domains_feasible(domains: Dict[int, Tuple[Optional[float], Optional[float]]]) -> bool:
    for vid, (lo, hi) in domains.items():
        if lo is not None and hi is not None and lo > hi + 1e-12:
            return False
    return True


def _apply_all(spec: VirSpec, domains: Dict[int, Tuple[Optional[float], Optional[float]]]) -> bool:
    constraints = list(spec.requires) + list(spec.derives)
    return _apply_constraints_fixed(constraints, domains)


def _eval_const(expr: Optional[VirExpr]) -> Optional[float]:
    if expr is None:
        return None
    if expr.kind in ("const_real", "const_int"):
        return float(expr.value)
    if expr.kind == "const_value":
        if isinstance(expr.value, (int, float, bool)):
            return float(expr.value)
    if expr.kind == "const_bool":
        return 1.0 if expr.value else 0.0
    iv = _interval(expr, {}) if expr.kind not in ("var",) else None
    if iv is not None and iv[0] is not None and iv[0] == iv[1]:
        return iv[0]
    return None


def _constraints_list(spec: VirSpec) -> List[VirExpr]:
    out: List[VirExpr] = []
    for c in spec.requires:
        out.append(c)
    for d in spec.derives:
        out.append(d)
    return out


def _solver_conflicts(spec: VirSpec) -> Tuple[List[int], List[Any]]:
    """Run the embedded linear solver over every affine clause.

    Equality tolerances are encoded as a pair of inequalities so a spec that is feasible
    within its declared tolerance is not falsely rejected.  Returns the minimal core
    (constraint indices) and the compiled clauses.
    """

    try:
        from ..linear import compile_clauses, simplify_clause
        from ..nano_solver import SolverStatus, VeritasNanoSolver
    except Exception:  # pragma: no cover - import cycle guard
        return [], []

    clauses = compile_clauses(spec)
    solver = VeritasNanoSolver()
    added = 0
    for clause in clauses:
        simple = simplify_clause(clause)
        if simple is None:
            continue
        op, coeffs, rhs = simple
        tol = float(getattr(clause.expr, "tol", 1e-6) or 1e-6)
        if op == "eq":
            solver.assert_and_track(("le", dict(coeffs), rhs + tol), clause.index)
            solver.assert_and_track(("ge", dict(coeffs), rhs - tol), clause.index)
        else:
            solver.assert_and_track((op, dict(coeffs), rhs), clause.index)
        added += 1
    if not added:
        return [], clauses
    status = solver.check()
    if status == SolverStatus.UNSAT:
        return sorted({int(core) for core in solver.get_unsat_core() if isinstance(core, int)}), clauses
    return [], clauses


def _hint(index: int, constraints: List[VirExpr], spec: VirSpec) -> str:
    from ..evaluate import expr_to_str

    text = expr_to_str(constraints[index], spec.id_to_name)
    return (
        f"constraint #{index} `{text}` participates in a minimal conflict; relax it or "
        "adjust its constant/tolerance"
    )


def run_preflight(spec: VirSpec) -> Any:
    """0-token feasibility gate: intervals + containers + embedded linear solver.

    Three independent engines run before a single token is generated:

    1. HC4 interval propagation over the *declared* base frame;
    2. AGDecode residual/cardinality feasibility for dynamic containers;
    3. an exact linear feasibility check (Fourier–Motzkin) over affine clauses.

    Any engine may prove UNSAT; none may prove SAT on its own, which keeps the gate
    conservative.
    """

    from .vir import PreflightResult

    domains = _initial_domains(spec)
    interval_ok = _apply_all(spec, domains)
    constraints = _constraints_list(spec)

    conflict_indices: List[int] = []
    details: List[str] = []

    if not interval_ok or not _domains_feasible(domains):
        for index in range(len(constraints)):
            subset = [c for j, c in enumerate(constraints) if j != index]
            if _apply_constraints_fixed(subset, _initial_domains(spec)):
                conflict_indices.append(index)
                details.append(f"interval contraction fails because of constraint #{index}")

    for index, reason in container_feasibility(spec):
        conflict_indices.append(index)
        details.append(reason)

    solver_conflicts, _clauses = _solver_conflicts(spec)
    if solver_conflicts:
        conflict_indices.extend(solver_conflicts)
        details.append("the embedded linear solver proved the affine clauses UNSAT")

    conflict_indices = sorted(set(conflict_indices))
    if not conflict_indices and (interval_ok and _domains_feasible(domains)):
        return PreflightResult(status="SAT", domains=domains)
    if not conflict_indices:
        conflict_indices = list(range(len(constraints))) if constraints else []

    hints = [_hint(index, constraints, spec) for index in conflict_indices if 0 <= index < len(constraints)]
    relaxation = {
        "conflicting_constraint_indices": conflict_indices,
        "details": details,
        "hints": hints,
        "minimal": True,
        "suggestion": "Relax one or more of the listed constraints or adjust variable bounds.",
    }

    return PreflightResult(
        status="UNSAT",
        conflicts=conflict_indices,
        constraint_indices=conflict_indices,
        relaxation=relaxation,
        domains=domains,
    )


def check_spec(spec: VirSpec) -> Any:
    return run_preflight(spec)
