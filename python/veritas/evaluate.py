"""Concrete evaluation of Veritas Intermediate Representation (VIR) expressions.

The compiler lowers ``@vt.verified_schema`` bodies into :class:`~veritas.compiler.vir.VirExpr`
trees.  Those trees are symbolic: they carry variable ids, container markers and
lambda-element bindings.  This module gives them *concrete* meaning once values are
known, which is what makes instance verification, 0-DoF derivation and trusted proof
certificates possible.

Design rules
------------
* **Total, never raising.**  Evaluation returns ``None`` when an expression cannot be
  evaluated (missing variable, domain error, unsupported operator).  Callers decide
  whether ``None`` means "incomplete" or "violated"; silently guessing a value is the
  one thing a verification engine must never do.
* **Tri-state booleans.**  ``True`` / ``False`` / ``None`` are propagated through
  ``and``/``or``/``forall`` using Kleene logic, so a partial assignment can never be
  reported as a violation.
* **Tolerance-aware comparisons.**  Every comparison carries an explicit ``tol``.  A
  comparison is satisfied when the residual is within ``tol``; the residual is also
  returned so diagnostics can say *by how much* a constraint failed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

Number = (int, float)
MISSING = object()

_CMP_SYMBOL = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}


# ---------------------------------------------------------------------------
# coercion helpers
# ---------------------------------------------------------------------------

def as_number(value: Any) -> Optional[float]:
    """Return ``value`` as a float, or ``None`` if it is not numeric.

    ``bool`` is deliberately *not* treated as a number: a boolean field compared with
    ``>= 1`` is almost always a user mistake and should surface as unevaluable rather
    than silently coerced.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, Number):
        try:
            out = float(value)
        except (OverflowError, ValueError):
            return None
        if math.isnan(out):
            return None
        return out
    return None


def as_sequence(value: Any) -> Optional[List[Any]]:
    """Iterate a container expression value in a stable order."""

    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if isinstance(value, (set, frozenset)):
            try:
                items = sorted(items, key=repr)
            except Exception:  # pragma: no cover - exotic element types
                pass
        return items
    return None


@dataclass
class Comparison:
    """Outcome of evaluating one comparison."""

    satisfied: Optional[bool]
    lhs: Any = None
    rhs: Any = None
    residual: Optional[float] = None
    tol: float = 1e-6
    detail: str = ""

    @property
    def violated(self) -> bool:
        return self.satisfied is False


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------

def eval_expr(expr: Any, env: Mapping[Any, Any], elem: Optional[Mapping[str, Any]] = None) -> Any:
    """Evaluate ``expr`` against ``env``.

    ``env`` is keyed by variable id (``int``) and/or field name (``str``).  ``elem``
    binds lambda parameters introduced by ``vt.forall``.
    """

    if expr is None:
        return None
    if isinstance(expr, (int, float, bool, str)) and not hasattr(expr, "kind"):
        return expr
    kind = getattr(expr, "kind", None)
    if kind is None:
        return None
    args = getattr(expr, "args", []) or []
    lhs = getattr(expr, "lhs", None)
    rhs = getattr(expr, "rhs", None)
    elem = elem or {}

    if kind in ("const_real", "const_int", "const_bool", "const_str"):
        return getattr(expr, "value", None)
    if kind == "const_none":
        return None
    if kind == "const_value":
        return getattr(expr, "value", None)
    if kind == "const_set":
        value = getattr(expr, "value", None)
        return set(value) if isinstance(value, (list, tuple, set)) else value
    if kind in ("const_list", "const_tuple"):
        value = getattr(expr, "value", None)
        return tuple(value) if kind == "const_tuple" and isinstance(value, list) else value
    if kind == "const_dict":
        return getattr(expr, "value", None)
    if kind == "const":
        return getattr(expr, "value", None)

    if kind == "var":
        vid = getattr(expr, "var_id", None)
        if vid is not None:
            try:
                if vid in env:
                    return env[vid]
            except TypeError:  # pragma: no cover - unhashable env keys
                pass
        name = getattr(expr, "name", None)
        if name is not None and name in env:
            return env[name]
        return None

    if kind == "elem":
        name = getattr(expr, "name", None)
        if name is not None and name in elem:
            return elem[name]
        if len(elem) == 1:
            # Lambdas with a renamed parameter still bind exactly one element.
            return next(iter(elem.values()))
        return None

    if kind == "name":
        name = getattr(expr, "name", None)
        if name is not None:
            if name in env:
                return env[name]
            if name in elem:
                return elem[name]
        return None

    if kind == "model_ref":
        name = getattr(expr, "name", None)
        return env.get(name) if name is not None else None

    if kind == "values":
        base = eval_expr(args[0], env, elem) if args else None
        return as_sequence(base)

    if kind == "sum":
        base = eval_expr(args[0], env, elem) if args else None
        items = as_sequence(base)
        if items is None:
            num = as_number(base)
            return num if num is not None else None
        total = 0.0
        for item in items:
            num = as_number(item)
            if num is None:
                return None
            total += num
        return total

    if kind in ("add", "sub", "mul", "div", "pow", "mod"):
        return _eval_arithmetic(kind, expr, lhs, rhs, args, env, elem)

    if kind == "neg":
        operand = eval_expr(lhs, env, elem) if lhs is not None else (
            eval_expr(args[0], env, elem) if args else None
        )
        num = as_number(operand)
        return -num if num is not None else None

    if kind == "sqrt":
        operand = eval_expr(args[0], env, elem) if args else eval_expr(lhs, env, elem)
        num = as_number(operand)
        if num is None or num < 0:
            return None
        return math.sqrt(num)

    if kind == "abs":
        operand = eval_expr(args[0], env, elem) if args else eval_expr(lhs, env, elem)
        num = as_number(operand)
        return abs(num) if num is not None else None

    if kind == "dot":
        return _eval_dot(expr, args, env, elem)

    if kind == "quad_form_cholesky":
        return _eval_quad_form(expr, args, env, elem)

    if kind == "keys_subset":
        base = eval_expr(args[0], env, elem) if args else None
        allowed = getattr(expr, "value", None)
        if not isinstance(base, Mapping) or allowed is None:
            return None
        try:
            return set(base.keys()).issubset(set(allowed))
        except TypeError:
            return None

    if kind == "forall":
        return _eval_forall(expr, env, elem)

    if kind == "compare":
        outcome = eval_compare(expr, env, elem)
        return outcome.satisfied

    if kind in ("and", "or"):
        values: List[Optional[bool]] = []
        for sub in args:
            values.append(_as_tristate(eval_expr(sub, env, elem)))
        if kind == "and":
            if any(v is False for v in values):
                return False
            if values and all(v is True for v in values):
                return True
            return None
        if any(v is True for v in values):
            return True
        if values and all(v is False for v in values):
            return False
        return None

    if kind == "ifexp":
        if len(args) < 3:
            return None
        test = eval_expr(args[0], env, elem)
        if test is None:
            return None
        return eval_expr(args[1] if bool(test) else args[2], env, elem)

    if kind == "subscript":
        base = eval_expr(args[0], env, elem) if args else None
        key = getattr(expr, "value", None)
        if isinstance(base, Mapping):
            return base.get(key)
        if isinstance(base, (list, tuple)):
            try:
                return base[int(key)]
            except (TypeError, ValueError, IndexError):
                return None
        return None

    if kind == "call":
        return _eval_call(expr, args, env, elem)

    # ``lambda``/``attr``/``unknown`` nodes are not concretely evaluable on their own.
    return None


def _eval_arithmetic(
    kind: str,
    expr: Any,
    lhs: Any,
    rhs: Any,
    args: Sequence[Any],
    env: Mapping[Any, Any],
    elem: Mapping[str, Any],
) -> Any:
    left = eval_expr(lhs, env, elem) if lhs is not None else (
        eval_expr(args[0], env, elem) if len(args) > 0 else None
    )
    right = eval_expr(rhs, env, elem) if rhs is not None else (
        eval_expr(args[1], env, elem) if len(args) > 1 else None
    )
    a = as_number(left)
    b = as_number(right)
    if a is None or b is None:
        if kind == "add" and isinstance(left, str) and isinstance(right, str):
            return left + right
        return None
    try:
        if kind == "add":
            return a + b
        if kind == "sub":
            return a - b
        if kind == "mul":
            return a * b
        if kind == "div":
            return None if b == 0 else a / b
        if kind == "mod":
            return None if b == 0 else math.fmod(a, b)
        if kind == "pow":
            result = a ** b
            if isinstance(result, complex):
                return None
            if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
                return None
            return result
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return None


def _eval_dot(
    expr: Any, args: Sequence[Any], env: Mapping[Any, Any], elem: Mapping[str, Any]
) -> Optional[float]:
    left = eval_expr(args[0], env, elem) if args else None
    right = getattr(expr, "value", None)
    if right is None and len(args) > 1:
        right = eval_expr(args[1], env, elem)
    if left is None or right is None:
        return None
    if isinstance(left, Mapping):
        if isinstance(right, Mapping):
            total = 0.0
            for key in set(left) | set(right):
                a = as_number(left.get(key, 0.0))
                b = as_number(right.get(key, 0.0))
                if a is None or b is None:
                    return None
                total += a * b
            return total
        seq = as_sequence(right)
        if seq is None or len(seq) < len(left):
            return None
        total = 0.0
        for (_, value), other in zip(left.items(), seq):
            a = as_number(value)
            b = as_number(other)
            if a is None or b is None:
                return None
            total += a * b
        return total
    left_seq = as_sequence(left)
    right_seq = as_sequence(right)
    if left_seq is None or right_seq is None:
        return None
    total = 0.0
    for a_raw, b_raw in zip(left_seq, right_seq):
        a = as_number(a_raw)
        b = as_number(b_raw)
        if a is None or b is None:
            return None
        total += a * b
    return total


def _covariance_matrix(
    weights: Sequence[Any], cov: Any, labels: Optional[Sequence[Any]] = None
) -> Optional[List[List[float]]]:
    """Build a symmetric covariance matrix aligned with ``labels``.

    A dict-keyed covariance may cover a *superset* of the emitted weights (the portfolio
    case where only two of three tickers are present).  Only entries whose labels are in
    the selected universe participate; indices are always taken against the selected
    labels, never the full covariance key set.
    """

    n = len(weights)
    if isinstance(cov, Mapping):
        if labels is None:
            derived: List[Any] = []
            for key in cov.keys():
                first = key[0] if isinstance(key, (tuple, list)) and len(key) == 2 else key
                if first not in derived:
                    derived.append(first)
            labels = derived
        labels = list(labels)
        if len(labels) != n:
            return None
        matrix = [[0.0] * n for _ in range(n)]
        for key, value in cov.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                a, b = key
            else:
                a = b = key
            if a not in labels or b not in labels:
                continue
            i, j = labels.index(a), labels.index(b)
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            matrix[i][j] = num
            if i != j:
                matrix[j][i] = num
        return matrix
    if isinstance(cov, (list, tuple)):
        rows = [list(row) for row in cov]
        if len(rows) != n or any(len(row) != n for row in rows):
            return None
        try:
            return [[float(v) for v in row] for row in rows]
        except (TypeError, ValueError):
            return None
    return None


def _eval_quad_form(
    expr: Any, args: Sequence[Any], env: Mapping[Any, Any], elem: Mapping[str, Any]
) -> Optional[float]:
    weights = eval_expr(args[0], env, elem) if args else None
    cov = getattr(expr, "value", None)
    if weights is None or cov is None:
        return None
    if isinstance(weights, Mapping):
        vector = [as_number(v) for v in weights.values()]
        labels = list(weights.keys())
    else:
        seq = as_sequence(weights)
        vector = [as_number(v) for v in seq] if seq is not None else None
        labels = None
    if vector is None or any(v is None for v in vector):
        return None
    matrix = _covariance_matrix(vector, cov, labels if isinstance(weights, Mapping) else None)
    if matrix is None:
        return None
    total = 0.0
    for i in range(len(vector)):
        for j in range(len(vector)):
            total += float(vector[i]) * matrix[i][j] * float(vector[j])
    return total


def _collect_elem_names(expr: Any, out: Optional[Set[str]] = None) -> Set[str]:
    out = out if out is not None else set()
    if expr is None or not hasattr(expr, "kind"):
        return out
    if getattr(expr, "kind", None) == "elem" and getattr(expr, "name", None):
        out.add(expr.name)
    for child in (
        getattr(expr, "lhs", None),
        getattr(expr, "rhs", None),
        getattr(expr, "predicate", None),
        getattr(expr, "iterable", None),
    ):
        if child is not None:
            _collect_elem_names(child, out)
    for arg in getattr(expr, "args", []) or []:
        _collect_elem_names(arg, out)
    return out


def _eval_forall(
    expr: Any, env: Mapping[Any, Any], elem: Mapping[str, Any]
) -> Optional[bool]:
    iterable = eval_expr(getattr(expr, "iterable", None), env, elem)
    items = as_sequence(iterable)
    if items is None:
        return None
    predicate = getattr(expr, "predicate", None)
    if predicate is None:
        return None
    names = _collect_elem_names(predicate)
    bound_name = sorted(names)[0] if names else "elem"
    results: List[Optional[bool]] = []
    for item in items:
        binding = dict(elem)
        binding[bound_name] = item
        results.append(_as_tristate(eval_expr(predicate, env, binding)))
    if any(r is False for r in results):
        return False
    if all(r is True for r in results):
        return True
    return None


_PRIMITIVES = {
    "abs": lambda x: abs(x),
    "min": lambda *xs: min(xs),
    "max": lambda *xs: max(xs),
    "round": lambda x, n=0: round(x, n),
    "len": lambda x: len(x),
    "float": lambda x: float(x),
    "int": lambda x: int(x),
}


def _eval_call(
    expr: Any, args: Sequence[Any], env: Mapping[Any, Any], elem: Mapping[str, Any]
) -> Any:
    name = getattr(expr, "name", "") or ""
    short = str(name).split(".")[-1]
    values = [eval_expr(a, env, elem) for a in args]
    fn = _PRIMITIVES.get(short)
    if fn is None:
        return None
    if any(v is None for v in values):
        return None
    try:
        return fn(*values)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_tristate(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------

def eval_compare(
    expr: Any, env: Mapping[Any, Any], elem: Optional[Mapping[str, Any]] = None
) -> Comparison:
    """Evaluate a comparison, returning the residual for diagnostics."""

    if expr is None or getattr(expr, "kind", None) != "compare":
        return Comparison(satisfied=None, detail="not a comparison")
    tol = float(getattr(expr, "tol", 1e-6) or 1e-6)
    op = getattr(expr, "op", "eq")
    lhs = eval_expr(getattr(expr, "lhs", None), env, elem)
    rhs = eval_expr(getattr(expr, "rhs", None), env, elem)
    if lhs is None or rhs is None:
        return Comparison(satisfied=None, lhs=lhs, rhs=rhs, tol=tol, detail="incomplete")

    a = as_number(lhs)
    b = as_number(rhs)
    if a is not None and b is not None:
        residual = a - b
        if op == "eq":
            ok = abs(residual) <= tol
        elif op == "ne":
            ok = abs(residual) > tol
        elif op == "le":
            ok = a <= b + tol
        elif op == "lt":
            ok = a < b + tol
        elif op == "ge":
            ok = a >= b - tol
        elif op == "gt":
            ok = a > b - tol
        else:
            return Comparison(satisfied=None, lhs=lhs, rhs=rhs, residual=residual, tol=tol,
                              detail=f"unknown operator {op}")
        return Comparison(satisfied=bool(ok), lhs=lhs, rhs=rhs, residual=residual, tol=tol)

    # Non-numeric comparisons are limited to equality of hashable/scalar values.
    try:
        equal = lhs == rhs
    except Exception:  # pragma: no cover - exotic objects
        return Comparison(satisfied=None, lhs=lhs, rhs=rhs, tol=tol, detail="incomparable")
    if op == "eq":
        ok = bool(equal)
    elif op == "ne":
        ok = not bool(equal)
    else:
        return Comparison(satisfied=None, lhs=lhs, rhs=rhs, tol=tol, detail="ordering on non-numeric")
    return Comparison(satisfied=ok, lhs=lhs, rhs=rhs, tol=tol)


# ---------------------------------------------------------------------------
# pretty printing
# ---------------------------------------------------------------------------

def format_number(value: Any) -> str:
    """Render a number compactly without losing information."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e16:
            return str(int(value))
        return repr(value)
    return str(value)


def expr_to_str(expr: Any, names: Optional[Mapping[int, str]] = None, prefix: str = "") -> str:
    """Human-readable rendering of a VIR expression (used in diagnostics and SMT comments)."""

    if expr is None:
        return "∅"
    if not hasattr(expr, "kind"):
        return format_number(expr)
    kind = expr.kind
    names = names or {}

    def var_name() -> str:
        if expr.var_id is not None and expr.var_id in names:
            return f"{prefix}{names[expr.var_id]}"
        return getattr(expr, "name", None) or f"v{getattr(expr, 'var_id', '?')}"

    if kind == "var":
        return var_name()
    if kind == "elem":
        return getattr(expr, "name", None) or "elem"
    if kind == "name":
        return getattr(expr, "name", None) or "?"
    if kind == "model_ref":
        return getattr(expr, "name", None) or "model"
    if kind.startswith("const_"):
        value = getattr(expr, "value", None)
        if kind == "const_bool":
            return "true" if value else "false"
        if kind == "const_str":
            return repr(value)
        if kind == "const_set":
            try:
                return "{" + ", ".join(sorted(format_number(v) for v in value)) + "}"
            except Exception:
                return "{…}"
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(expr_to_str(v, names, prefix) for v in value) + "]"
        if isinstance(value, dict):
            return "{" + ", ".join(f"{k!r}: {format_number(v)}" for k, v in value.items()) + "}"
        return format_number(value) if value is not None else "null"
    if kind == "values":
        return f"{expr_to_str(expr.args[0] if expr.args else None, names, prefix)}.values()"
    if kind in ("add", "sub", "mul", "div", "pow", "mod"):
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^", "mod": "%"}[kind]
        left = expr_to_str(expr.lhs if expr.lhs is not None else (expr.args[0] if expr.args else None), names, prefix)
        right = expr_to_str(expr.rhs if expr.rhs is not None else (expr.args[1] if len(expr.args) > 1 else None), names, prefix)
        return f"({left} {op} {right})"
    if kind == "neg":
        return f"-{expr_to_str(expr.lhs or (expr.args[0] if expr.args else None), names, prefix)}"
    if kind == "sqrt":
        inner = expr.args[0] if expr.args else expr.lhs
        return f"sqrt({expr_to_str(inner, names, prefix)})"
    if kind == "abs":
        inner = expr.args[0] if expr.args else expr.lhs
        return f"abs({expr_to_str(inner, names, prefix)})"
    if kind == "sum":
        inner = expr.args[0] if expr.args else expr.lhs
        return f"sum({expr_to_str(inner, names, prefix)})"
    if kind == "dot":
        inner = ", ".join(expr_to_str(a, names, prefix) for a in expr.args)
        return f"dot({inner})"
    if kind == "quad_form_cholesky":
        inner = ", ".join(expr_to_str(a, names, prefix) for a in expr.args)
        return f"quad_form({inner}, Σ)"
    if kind == "keys_subset":
        inner = expr_to_str(expr.args[0], names, prefix) if expr.args else "?"
        return f"keys_subset({inner}, {sorted(expr.value) if expr.value else '∅'})"
    if kind == "forall":
        return f"forall({expr_to_str(expr.iterable, names, prefix)}, {expr_to_str(expr.predicate, names, prefix)})"
    if kind == "compare":
        symbol = _CMP_SYMBOL.get(expr.op, expr.op)
        left = expr_to_str(expr.lhs, names, prefix)
        right = expr_to_str(expr.rhs, names, prefix)
        return f"{left} {symbol} {right}"
    if kind in ("and", "or"):
        joiner = " ∧ " if kind == "and" else " ∨ "
        return "(" + joiner.join(expr_to_str(a, names, prefix) for a in expr.args) + ")"
    if kind == "ifexp":
        parts = [expr_to_str(a, names, prefix) for a in expr.args]
        return f"({parts[0]} ? {parts[1]} : {parts[2]})" if len(parts) == 3 else "if(…)"
    if kind == "subscript":
        base = expr_to_str(expr.args[0], names, prefix) if expr.args else "?"
        key = expr.value
        if isinstance(key, str):
            return f"{base}[{key!r}]"
        return f"{base}[{format_number(key)}]"
    if kind == "call":
        inner = ", ".join(expr_to_str(a, names, prefix) for a in expr.args)
        return f"{expr.name}({inner})"
    return getattr(expr, "name", None) or kind


def collect_var_ids(expr: Any, out: Optional[List[int]] = None) -> List[int]:
    """Collect variable ids referenced by an expression (order-preserving)."""

    out = out if out is not None else []
    if expr is None or not hasattr(expr, "kind"):
        return out
    vid = getattr(expr, "var_id", None)
    if getattr(expr, "kind", None) == "var" and vid is not None and vid not in out:
        out.append(vid)
    for child in (
        getattr(expr, "lhs", None),
        getattr(expr, "rhs", None),
        getattr(expr, "predicate", None),
        getattr(expr, "iterable", None),
        getattr(expr, "container_id", None),
    ):
        if child is not None and hasattr(child, "kind"):
            collect_var_ids(child, out)
    for arg in getattr(expr, "args", []) or []:
        collect_var_ids(arg, out)
    return out


__all__ = [
    "Comparison",
    "MISSING",
    "as_number",
    "as_sequence",
    "collect_var_ids",
    "eval_compare",
    "eval_expr",
    "expr_to_str",
    "format_number",
]
