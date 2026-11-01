"""Affine extraction, interval evaluation and clause compilation for VIR.

Three consumers depend on this module:

* the **pre-flight gate**, which wants fast, conservative infeasibility proofs before any
  token is generated;
* the **decode engine**, which masks numeric prefixes using contracted domains and pushes
  completed clauses into a solver;
* the **proof layer**, which names and orders constraints consistently.

Everything is conservative by construction: when a form is not affine, extraction returns
``None`` and the caller must fall back to an oracle instead of guessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .compiler.vir import VirExpr, VirSpec
from .evaluate import as_number, expr_to_str
from .interval import Interval

_AFFINE_OPS = ("add", "sub", "mul", "div", "neg")


@dataclass
class LinearForm:
    """``sum(coeffs[v] * v) + const`` over scalar variables."""

    coeffs: dict[Any, float] = field(default_factory=dict)
    const: float = 0.0

    def scaled(self, factor: float) -> LinearForm:
        return LinearForm(
            coeffs={k: v * factor for k, v in self.coeffs.items()},
            const=self.const * factor,
        )

    def plus(self, other: LinearForm) -> LinearForm:
        coeffs = dict(self.coeffs)
        for key, value in other.coeffs.items():
            coeffs[key] = coeffs.get(key, 0.0) + value
        coeffs = {k: v for k, v in coeffs.items() if abs(v) > 1e-15}
        return LinearForm(coeffs=coeffs, const=self.const + other.const)

    def is_constant(self, eps: float = 1e-15) -> bool:
        return all(abs(v) <= eps for v in self.coeffs.values())

    def pretty(self, names: Mapping[int, str] | None = None) -> str:
        names = names or {}
        parts: list[str] = []
        for var, coeff in sorted(self.coeffs.items(), key=lambda kv: repr(kv[0])):
            label = names.get(var, str(var)) if isinstance(var, int) else str(var)
            parts.append(f"{coeff:+.6g}*{label}")
        parts.append(f"{self.const:+.6g}")
        return " ".join(parts)


def linearize(expr: VirExpr | None) -> LinearForm | None:
    """Extract an affine form, or ``None`` when the expression is not affine."""

    if expr is None:
        return None
    kind = getattr(expr, "kind", None)
    if kind in ("const_real", "const_int"):
        return LinearForm({}, float(expr.value))
    if kind == "const_value":
        value = as_number(expr.value)
        return LinearForm({}, value) if value is not None else None
    if kind == "var":
        key = expr.var_id if expr.var_id is not None else expr.name
        if key is None:
            return None
        return LinearForm({key: 1.0}, 0.0)
    if kind == "neg":
        inner = linearize(expr.lhs if expr.lhs is not None else (expr.args[0] if expr.args else None))
        return inner.scaled(-1.0) if inner is not None else None
    if kind in ("add", "sub"):
        left = linearize(expr.lhs if expr.lhs is not None else (expr.args[0] if expr.args else None))
        right = linearize(expr.rhs if expr.rhs is not None else (expr.args[1] if len(expr.args) > 1 else None))
        if left is None or right is None:
            return None
        return left.plus(right) if kind == "add" else left.plus(right.scaled(-1.0))
    if kind == "mul":
        left = linearize(expr.lhs if expr.lhs is not None else (expr.args[0] if expr.args else None))
        right = linearize(expr.rhs if expr.rhs is not None else (expr.args[1] if len(expr.args) > 1 else None))
        if left is None or right is None:
            return None
        if left.is_constant():
            return right.scaled(left.const)
        if right.is_constant():
            return left.scaled(right.const)
        return None
    if kind == "div":
        left = linearize(expr.lhs if expr.lhs is not None else (expr.args[0] if expr.args else None))
        right = linearize(expr.rhs if expr.rhs is not None else (expr.args[1] if len(expr.args) > 1 else None))
        if left is None or right is None or not right.is_constant() or right.const == 0:
            return None
        return left.scaled(1.0 / right.const)
    if kind == "sum":
        inner = expr.args[0] if expr.args else None
        # A sum over a scalar-only list constant is affine; container sums are not.
        if inner is None or getattr(inner, "kind", None) != "const_list":
            return None
        total = LinearForm({}, 0.0)
        for item in inner.value or []:
            num = as_number(item)
            if num is None:
                return None
            total = total.plus(LinearForm({}, num))
        return total
    return None


@dataclass
class Clause:
    """One compiled clause with a stable tracker id."""

    tracker: int
    index: int
    source: str  # "require" | "derive"
    expr: VirExpr
    text: str
    linear: LinearForm | None = None
    comparison: tuple[str, float, float] | None = None  # (op, rhs_const, tol)

    def as_solver_constraint(self) -> tuple[str, dict[Any, float], float] | None:
        """Return ``(op, coeffs, rhs)`` for the nano solver, or ``None``."""

        if self.comparison is None or self.linear is None:
            return None
        op, rhs, _tol = self.comparison
        # linear (lhs - rhs) OP 0  ->  sum(coeff*x) OP -const
        return op, dict(self.linear.coeffs), -self.linear.const + rhs


def compile_clauses(spec: VirSpec) -> list[Clause]:
    clauses: list[Clause] = []
    tracker = 0
    for index, expr in enumerate(spec.requires):
        clause = _compile_one(tracker, index, "require", expr, spec)
        clauses.append(clause)
        tracker += 1
    offset = len(spec.requires)
    for i, expr in enumerate(spec.derives):
        clause = _compile_one(tracker, offset + i, "derive", expr, spec)
        clauses.append(clause)
        tracker += 1
    return clauses


def _compile_one(tracker: int, index: int, source: str, expr: VirExpr, spec: VirSpec) -> Clause:
    text = expr_to_str(expr, spec.id_to_name)
    clause = Clause(tracker=tracker, index=index, source=source, expr=expr, text=text)
    if getattr(expr, "kind", None) != "compare":
        return clause
    lhs = linearize(expr.lhs)
    rhs_expr = expr.rhs
    op = getattr(expr, "op", "eq")
    if lhs is not None:
        right = linearize(rhs_expr)
        if right is not None:
            difference = lhs.plus(right.scaled(-1.0))
            # Difference OP -0  <=>  sum(coeff) OP -const
            clause.linear = LinearForm(difference.coeffs, difference.const)
            clause.comparison = (op, 0.0, float(getattr(expr, "tol", 0.0) or 0.0))
    return clause


def simplify_clause(clause: Clause) -> tuple[str, dict[Any, float], float] | None:
    """Normalize a clause into ``op, coeffs, rhs`` with a real tolerance on rhs."""

    if clause.comparison is None or clause.linear is None:
        return None
    op, _zero, _tol = clause.comparison
    coeffs = dict(clause.linear.coeffs)
    rhs = -clause.linear.const
    # Contractor domains must stay *exact*.  Padding bounds by the equality tolerance
    # (e.g. a + b == 1 +/- 1e-6) makes the contracted box slightly larger than the true
    # feasible set; near an active boundary the next variable's domain can then round to
    # empty.  Tolerance is applied at concrete verification time, not here.
    if op in ("le", "lt"):
        op = "le"
    elif op in ("ge", "gt"):
        op = "ge"
    elif op != "eq":
        return None
    return op, coeffs, rhs


# ---------------------------------------------------------------------------
# interval evaluation
# ---------------------------------------------------------------------------

def interval_eval(
    expr: VirExpr | None,
    domains: Mapping[Any, Interval],
    spec: VirSpec | None = None,
) -> Interval | None:
    """Evaluate an expression with outward-rounded interval arithmetic."""

    if expr is None:
        return None
    kind = getattr(expr, "kind", None)
    if kind in ("const_real", "const_int"):
        value = float(expr.value)
        return Interval(value, value)
    if kind == "const_value":
        value = as_number(expr.value)
        return Interval(value, value) if value is not None else None
    if kind == "var":
        key = expr.var_id if expr.var_id is not None else expr.name
        if key is None:
            return None
        return domains.get(key, Interval.entire())
    if kind == "neg":
        inner = interval_eval(expr.lhs if expr.lhs is not None else (expr.args[0] if expr.args else None), domains, spec)
        return Interval(-inner.hi, -inner.lo) if inner is not None else None
    if kind in ("add", "sub", "mul", "div"):
        left = interval_eval(expr.lhs if expr.lhs is not None else (expr.args[0] if expr.args else None), domains, spec)
        right = interval_eval(expr.rhs if expr.rhs is not None else (expr.args[1] if len(expr.args) > 1 else None), domains, spec)
        if left is None or right is None:
            return None
        if kind == "add":
            return left.add(right)
        if kind == "sub":
            return left.sub(right)
        if kind == "mul":
            return left.mul(right)
        return left.div(right)
    if kind == "pow":
        base = interval_eval(expr.lhs, domains, spec)
        exponent = interval_eval(expr.rhs, domains, spec)
        if base is None or exponent is None or exponent.lo != exponent.hi:
            return None
        power = exponent.lo
        if power == 2:
            return base.sqr()
        if power == 0:
            return Interval(1.0, 1.0)
        if float(power).is_integer() and power > 0:
            result = Interval(1.0, 1.0)
            for _ in range(int(power)):
                result = result.mul(base)
            return result
        return None
    if kind == "sqrt":
        inner = interval_eval(expr.args[0] if expr.args else None, domains, spec)
        return inner.sqrt() if inner is not None else None
    if kind == "abs":
        inner = interval_eval(expr.args[0] if expr.args else None, domains, spec)
        if inner is None:
            return None
        if inner.lo >= 0:
            return inner
        if inner.hi <= 0:
            return Interval(-inner.hi, -inner.lo)
        return Interval(0.0, max(abs(inner.lo), abs(inner.hi)))
    return None


def compare_interval(
    expr: VirExpr, domains: Mapping[Any, Interval], spec: VirSpec | None = None
) -> bool | None:
    """Three-valued interval check: ``True`` (definitely), ``False`` (impossible), ``None``."""

    left = interval_eval(getattr(expr, "lhs", None), domains, spec)
    right = interval_eval(getattr(expr, "rhs", None), domains, spec)
    if left is None or right is None:
        return None
    tol = float(getattr(expr, "tol", 0.0) or 0.0)
    op = getattr(expr, "op", "eq")
    if left.is_empty() or right.is_empty():
        return False if op != "ne" else True
    if op in ("le", "lt"):
        if left.hi <= right.lo + tol:
            return True
        if left.lo > right.hi + tol:
            return False
        return None
    if op in ("ge", "gt"):
        if left.lo >= right.hi - tol:
            return True
        if left.hi < right.lo - tol:
            return False
        return None
    if op == "eq":
        if left.lo > right.hi + tol or right.lo > left.hi + tol:
            return False
        if left.lo == left.hi and right.lo == right.hi:
            return abs(left.lo - right.lo) <= tol
        return None
    if op == "ne":
        if left.lo > right.hi + tol or right.lo > left.hi + tol:
            return True
        if left.lo == left.hi and right.lo == right.hi:
            return abs(left.lo - right.lo) > tol
        return None
    return None


__all__ = [
    "Clause",
    "LinearForm",
    "compile_clauses",
    "compare_interval",
    "interval_eval",
    "linearize",
    "simplify_clause",
]
