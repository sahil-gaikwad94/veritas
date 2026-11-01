"""Conservative three-state preflight: SAT requires a checked witness."""
from __future__ import annotations

import math

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
    if container_problems:
        conflicts.extend(i for i, _ in container_problems)
        details.extend(reason for _, reason in container_problems)
    if conflicts:
        conflicts = sorted(set(conflicts))
        hints = [f"constraint #{i} `{expr_to_str(constraints[i], spec.id_to_name)}` participates in a conflict"
                 for i in conflicts if i < len(constraints)]
        return PreflightResult("UNSAT", conflicts, conflicts,
                               {"conflicting_constraint_indices": conflicts, "details": details, "hints": hints,
                                "minimal": solver.core_is_minimal and not container_problems,
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
            values[name] = "x" * max(1, vtype.min_len or 0)
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