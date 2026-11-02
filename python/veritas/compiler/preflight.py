"""Conservative three-state preflight: SAT requires a checked witness."""
from __future__ import annotations

import math
from typing import Any

from ..evaluate import expr_to_str
from .analysis import container_feasibility
from .vir import PreflightResult, VirSpec


class InfeasibleSpecError(ValueError):
    def __init__(self, message, conflicts=None, relaxation=None):
        super().__init__(message)
        self.conflicts = list(conflicts or [])
        self.relaxation = relaxation


def initial_domains(spec) -> dict[int, tuple]:
    domains = {}
    for _vid, info in spec.variables.items():
        vtype = info.get("type")
        if getattr(vtype, "kind", None) == "dynamic_map":
            lo, hi = vtype.val_min, vtype.val_max
        else:
            lo, hi = getattr(vtype, "min", None), getattr(vtype, "max", None)
        domains[_vid] = (None if lo is not None and math.isinf(lo) else lo,
                        None if hi is not None and math.isinf(hi) else hi)
    return domains


def _string_witness(vtype: Any) -> str:
    """A concrete string that satisfies the declared ``pattern`` (and length window).

    Without this, pre-flight could only propose ``"xxxx"`` and any pattern-constrained
    schema would be reported ``UNKNOWN`` even when a witness plainly exists.
    """

    pattern = getattr(vtype, "pattern", None)
    min_len = max(0, int(getattr(vtype, "min_len", 0) or 0))
    max_len = getattr(vtype, "max_len", None)
    ceiling = int(max_len) if max_len else max(min_len, 64)
    if pattern:
        from ..pattern import PatternConstraint, UnsupportedPattern

        try:
            constraint = PatternConstraint.compile(pattern)
        except UnsupportedPattern:
            constraint = None
        if constraint is not None:
            found = constraint.witness(min_length=min_len, max_length=ceiling)
            if found is not None:
                return found
    return "x" * max(1, min_len)


def _bound_conflicts(spec: VirSpec) -> list[tuple[int, str]]:
    """Exact 1-D bound contradictions, including *strict* bounds.

    The rational Fourier--Motzkin relaxation treats ``<``/``>`` as ``<=``/``>=`` and can
    only return ``UNKNOWN`` when a witness lands exactly on a strict boundary.  Two
    single-variable bounds can still be refuted exactly, which is cheap and sound:
    ``a >= 1 and a < 1`` is infeasible for every type, and integers promote open bounds
    to closed ones (``a < 1`` means ``a <= 0``).
    """

    from ..linear import compile_clauses, simplify_clause

    lower: dict[int, tuple[float, bool, Any]] = {}
    upper: dict[int, tuple[float, bool, Any]] = {}
    kinds: dict[int, str] = {}

    def tighten(vid: int, value: float, is_lower: bool, strict: bool, tag: Any) -> None:
        table = lower if is_lower else upper
        current = table.get(vid)
        if current is None:
            table[vid] = (value, strict, tag)
            return
        best, best_strict, best_tag = current
        if (is_lower and value > best) or (not is_lower and value < best):
            table[vid] = (value, strict, tag)
        elif value == best and strict and not best_strict:
            table[vid] = (best, True, best_tag)

    for vid, info in spec.variables.items():
        vtype = info["type"]
        kind = getattr(vtype, "kind", None)
        if kind not in ("real", "int", "bool"):
            continue
        kinds[vid] = kind
        if kind == "bool":
            lo, hi, lo_ex, hi_ex = 0.0, 1.0, False, False
        else:
            raw_lo = getattr(vtype, "min", None)
            raw_hi = getattr(vtype, "max", None)
            lo = None if raw_lo is None else float(raw_lo)
            hi = None if raw_hi is None else float(raw_hi)
            lo_ex = bool(getattr(vtype, "min_exclusive", False))
            hi_ex = bool(getattr(vtype, "max_exclusive", False))
            if kind == "int":
                if lo is not None and lo_ex:
                    lo, lo_ex = math.floor(lo) + 1.0, False
                if hi is not None and hi_ex:
                    hi, hi_ex = math.ceil(hi) - 1.0, False
        if lo is not None and math.isfinite(lo):
            tighten(vid, lo, True, lo_ex, ("type", vid))
        if hi is not None and math.isfinite(hi):
            tighten(vid, hi, False, hi_ex, ("type", vid))

    for clause in compile_clauses(spec):
        simple = simplify_clause(clause)
        if simple is None:
            continue
        op, coeffs, rhs = simple
        if len(coeffs) != 1:
            continue
        vid, coef = next(iter(coeffs.items()))
        if not isinstance(vid, int) or coef == 0 or vid not in kinds:
            continue
        # ``simplify_clause`` closes strict inequalities (``lt`` -> ``le``) because the
        # rational solver works on closed half-spaces; the strictness itself is still
        # declared on the compiled expression and matters for exact refutation.
        strict = op in ("lt", "gt") or getattr(clause.expr, "op", None) in ("lt", "gt")
        value = rhs / coef
        if op in ("eq",):
            tighten(vid, value, True, False, clause.index)
            tighten(vid, value, False, False, clause.index)
            continue
        is_lower = (op in ("ge", "gt")) == (coef > 0)
        if kinds[vid] == "int":
            # a*x OP b with x integral: round the bound inward and close it.  A strict
            # bound skips the boundary integer entirely (``x < 1`` means ``x <= 0``).
            if is_lower:
                value = math.floor(value) + 1 if strict else math.ceil(value)
            else:
                value = math.ceil(value) - 1 if strict else math.floor(value)
            strict = False
        tighten(vid, value, is_lower, strict, clause.index)

    problems: list[tuple[int, str]] = []
    for vid in sorted(set(lower) & set(upper)):
        lo, lo_strict, lo_tag = lower[vid]
        hi, hi_strict, hi_tag = upper[vid]
        name = spec.id_to_name.get(vid, str(vid))
        constraint_tags = [t for t in (lo_tag, hi_tag) if isinstance(t, int)]
        target = constraint_tags[0] if constraint_tags else -1
        if lo > hi:
            problems.append((target, f"'{name}' has lower bound {lo:g} above upper bound {hi:g}"))
        elif lo == hi and (lo_strict or hi_strict):
            problems.append(
                (target, f"'{name}' is pinned to the excluded value {lo:g} by open bounds")
            )
    return problems


def _solver(spec):
    from ..linear import compile_clauses, simplify_clause
    from ..nano_solver import VeritasNanoSolver
    solver = VeritasNanoSolver()
    clauses = compile_clauses(spec)
    covered = []
    for clause in clauses:
        simple = simplify_clause(clause)
        if simple is None:
            continue
        op, coeffs, rhs = simple
        if any(key not in spec.name_to_id.values() for key in coeffs):
            continue
        tol = clause.expr.tol if clause.expr.op == "eq" else 0.0
        if op == "eq":
            solver.assert_and_track(("le", coeffs, rhs + tol), clause.index)
            solver.assert_and_track(("ge", coeffs, rhs - tol), clause.index)
        else:
            solver.assert_and_track((op, coeffs, rhs), clause.index)
        covered.append(clause.index)
    for vid, (lo, hi) in initial_domains(spec).items():
        vtype = spec.variables[vid]["type"]
        if vtype.kind in ("real", "int", "bool"):
            if vtype.kind == "bool":
                lo, hi = 0.0, 1.0
            if lo is not None:
                solver.assert_and_track(("gt" if vtype.min_exclusive else "ge", {vid: 1}, lo), ("type", vid))
            if hi is not None:
                solver.assert_and_track(("lt" if vtype.max_exclusive else "le", {vid: 1}, hi), ("type", vid))
    return solver, clauses, covered


def run_preflight(spec: VirSpec) -> PreflightResult:
    from ..instance import verify_values
    from ..nano_solver import SolverStatus
    constraints = spec.all_constraints()
    container_problems = container_feasibility(spec)
    solver, clauses, covered = _solver(spec)
    status = solver.check()
    conflicts, details = [], []
    if status == SolverStatus.UNSAT:
        conflicts = [i for i in solver.get_unsat_core() if isinstance(i, int)]
        details.append("exact rational linear relaxation is infeasible")
    bound_problems = _bound_conflicts(spec)
    if bound_problems:
        conflicts.extend(i for i, _ in bound_problems if i >= 0)
        details.extend(reason for _, reason in bound_problems)
        if status != SolverStatus.UNSAT:
            status = SolverStatus.UNSAT
    if container_problems:
        conflicts.extend(i for i, _ in container_problems)
        details.extend(reason for _, reason in container_problems)
    if conflicts:
        conflicts = sorted(set(conflicts))
        hints = [f"constraint #{i} `{expr_to_str(constraints[i], spec.id_to_name)}` participates in a conflict"
                 for i in conflicts if i < len(constraints)]
        return PreflightResult("UNSAT", conflicts, conflicts,
                               {"conflicting_constraint_indices": conflicts, "details": details, "hints": hints,
                                "minimal": solver.core_is_minimal and not container_problems and not bound_problems,
                                "suggestion": "Relax a listed constraint or a field bound."}, initial_domains(spec), details=details)
    if status != SolverStatus.SAT:
        return PreflightResult("UNKNOWN", details=[solver.unknown_reason or "unsupported constraints"], domains=initial_domains(spec))
    witness = solver.get_float_model()
    values = {spec.id_to_name[vid]: value for vid, value in witness.items() if vid in spec.id_to_name}
    # Integer declarations are not decided by the real relaxation. Try the rounded
    # witness, but never claim SAT merely because its real relaxation was feasible.
    for vid, info in spec.variables.items():
        if info["type"].kind == "int" and vid in witness:
            values[info["name"]] = round(values[info["name"]])
        if info["type"].kind == "bool" and vid in witness:
            values[info["name"]] = values[info["name"]] >= 0.5
    # A checked concrete witness proves existence even for constraints outside the
    # affine fragment. Fill unconstrained non-numeric fields conservatively.
    for _vid, info in spec.variables.items():
        name, vtype = info["name"], info["type"]
        if name in values:
            continue
        field = getattr(spec.model, "model_fields", {}).get(name)
        from pydantic_core import PydanticUndefined
        if field is not None and field.default is not PydanticUndefined:
            values[name] = field.default
        elif field is not None and field.default_factory is not None:
            values[name] = field.default_factory()
        elif vtype.kind == "string":
            values[name] = _string_witness(vtype)
        elif vtype.kind == "bool":
            values[name] = False
        elif vtype.kind == "finite_set" and vtype.universe:
            values[name] = vtype.universe[0]
        elif vtype.kind == "dynamic_map" and not vtype.min_len:
            values[name] = [] if vtype.container_kind == "list" else {}
        elif vtype.nullable:
            values[name] = None
    try:
        report = verify_values(spec, values, derive_missing=True, check_preferences=False)
    except Exception as exc:
        return PreflightResult("UNKNOWN", details=[f"witness check failed: {exc}"], domains=initial_domains(spec))
    if report.ok:
        return PreflightResult("SAT", domains=initial_domains(spec), witness=report.values,
                               details=["complete affine decision and concrete witness verification"])
    return PreflightResult("UNKNOWN", details=["real witness did not satisfy discrete/model semantics: " + report.summary_line()],
                           domains=initial_domains(spec), witness=values)


def check_spec(spec):
    return run_preflight(spec)