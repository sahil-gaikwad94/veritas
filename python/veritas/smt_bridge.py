"""Optional Z3 oracle with explicit three/four-state results.

The bridge never reports SAT merely because Z3 is importable. It asserts a supported
fragment (affine constraints plus declared scalar bounds), checks the solver, and
returns UNKNOWN for unsupported formulas or timeouts. Raw Z3 assertions are also
accepted for callers that already have a BoolRef.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import z3
    HAS_Z3 = True
except Exception:  # pragma: no cover - optional dependency
    z3 = None
    HAS_Z3 = False


@dataclass
class SMTResult:
    status: str
    model: dict[str, Any] = field(default_factory=dict)
    core: list[Any] = field(default_factory=list)
    reason: str = ""


class SmtBridge:
    def __init__(self, spec: Any = None, timeout_ms: int | None = 5000):
        self.available = HAS_Z3
        self.timeout_ms = timeout_ms
        self._solver = z3.Solver() if HAS_Z3 else None
        self._tracks: dict[str, Any] = {}
        self._names: dict[Any, str] = {}
        self._unsupported: list[str] = []
        self._last = SMTResult("UNAVAILABLE" if not HAS_Z3 else "UNKNOWN")
        if self._solver is not None and timeout_ms is not None:
            self._solver.set("timeout", int(timeout_ms))
        if spec is not None:
            self.add_spec(spec)

    def add_spec(self, spec: Any) -> SmtBridge:
        if not HAS_Z3:
            return self
        from .linear import compile_clauses, simplify_clause
        variables = {}
        for vid, info in spec.variables.items():
            vtype = info.get("type")
            name = info.get("name") or spec.id_to_name.get(vid, str(vid))
            kind = getattr(vtype, "kind", "real")
            if kind == "bool":
                var = z3.Bool(name)
            elif kind == "int":
                var = z3.Int(name)
            elif kind in ("real", "float", "number"):
                var = z3.Real(name)
            else:
                self._unsupported.append(f"variable {name!r} has non-scalar type {kind!r}")
                continue
            variables[vid] = var
            self._names[var] = name
            lo, hi = getattr(vtype, "min", None), getattr(vtype, "max", None)
            if kind == "bool":
                continue
            if lo is not None:
                self._solver.add(var > lo if getattr(vtype, "min_exclusive", False) else var >= lo)
            if hi is not None:
                self._solver.add(var < hi if getattr(vtype, "max_exclusive", False) else var <= hi)
        for clause in compile_clauses(spec):
            simple = simplify_clause(clause)
            if simple is None:
                self._unsupported.append(f"clause #{clause.index} is not affine")
                continue
            op, coeffs, rhs = simple
            if any(vid not in variables for vid in coeffs):
                self._unsupported.append(f"clause #{clause.index} references unsupported variables")
                continue
            expr = sum((variables[vid] * coeff for vid, coeff in coeffs.items()), z3.RealVal(0))
            tol = clause.expr.tol if clause.expr.op == "eq" else 0
            if op == "eq":
                formula = z3.And(expr >= rhs - tol, expr <= rhs + tol)
            elif op == "le":
                formula = expr <= rhs
            elif op == "lt":
                formula = expr < rhs
            elif op == "ge":
                formula = expr >= rhs
            elif op == "gt":
                formula = expr > rhs
            else:
                self._unsupported.append(f"clause #{clause.index} has unknown operator {op!r}")
                continue
            self.assert_and_track(formula, f"c{clause.index}")
        return self

    def assert_and_track(self, formula: Any, name: Any) -> None:
        if not HAS_Z3:
            return
        track = z3.Bool(str(name)) if not isinstance(formula, z3.BoolRef) else z3.Bool(str(name))
        self._tracks[str(name)] = track
        self._solver.assert_and_track(formula, track)

    def check(self, *args, **kwargs) -> str:
        return self.check_with_timeout(kwargs.get("timeout_ms", self.timeout_ms)).status

    def check_with_timeout(self, timeout_ms: int | None = None) -> SMTResult:
        if not HAS_Z3:
            self._last = SMTResult("UNAVAILABLE", reason="z3-solver is not installed")
            return self._last
        if self._unsupported:
            self._last = SMTResult("UNKNOWN", reason="; ".join(self._unsupported))
            return self._last
        if not self._solver.assertions() and not self._tracks:
            self._last = SMTResult("UNKNOWN", reason="no assertions were exported")
            return self._last
        if timeout_ms is not None:
            self._solver.set("timeout", int(timeout_ms))
        result = self._solver.check()
        if result == z3.sat:
            model = self._solver.model()
            values = {}
            for decl in model.decls():
                value = model[decl]
                name = str(decl.name())
                if z3.is_int_value(value):
                    values[name] = value.as_long()
                elif z3.is_rational_value(value):
                    values[name] = float(value.numerator_as_long()) / float(value.denominator_as_long())
                elif z3.is_true(value):
                    values[name] = True
                elif z3.is_false(value):
                    values[name] = False
            self._last = SMTResult("SAT", model=values)
        elif result == z3.unsat:
            core = [str(c) for c in self._solver.unsat_core()]
            self._last = SMTResult("UNSAT", core=core)
        else:
            self._last = SMTResult("UNKNOWN", reason=self._solver.reason_unknown())
        return self._last

    def get_unsat_core(self) -> list[Any]:
        return list(self._last.core)

    def get_model(self) -> dict[str, Any]:
        return dict(self._last.model)


class Z3SmtOracle(SmtBridge):
    pass


IncrementalSmtPool = SmtBridge
SmtOracle = SmtBridge
MaxSmtOracle = SmtBridge

__all__ = ["SMTResult", "SmtBridge", "Z3SmtOracle", "IncrementalSmtPool", "SmtOracle", "MaxSmtOracle", "HAS_Z3"]