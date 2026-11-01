"""Phase 3: 0-DoF evaluator and cascaded expression scheduler."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ZeroDofRule:
    target_var: Any
    target_json_key: Optional[str] = None
    input_vars: List[Any] = field(default_factory=list)
    rhs_eval: Optional[Callable[[Dict[Any, Any]], Any]] = None
    rhs_expr: Any = None
    precision_digits: int = 6


class ZeroDofEngine:
    def __init__(self, rules: Optional[List[ZeroDofRule]] = None):
        self.rules = list(rules or [])
        self.cache: Dict[Any, Any] = {}

    def add_rule(self, rule: ZeroDofRule) -> None:
        self.rules.append(rule)

    def is_ready(self, target_var: Any, values: Dict[Any, Any]) -> bool:
        rule = self._rule_for(target_var)
        if rule is None:
            return False
        return all(v in values for v in rule.input_vars)

    def evaluate(self, target_var: Any, values: Dict[Any, Any]) -> Any:
        rule = self._rule_for(target_var)
        if rule is None:
            return None
        if rule.rhs_eval is not None:
            val = rule.rhs_eval(values)
        elif rule.rhs_expr is not None:
            val = _eval_expr(rule.rhs_expr, values)
            # if rhs_expr is a compare, take rhs
            if val is None and getattr(rule.rhs_expr, "kind", None) == "compare":
                val = _eval_expr(rule.rhs_expr.rhs, values)
        else:
            val = None
        if val is not None:
            self.cache[target_var] = val
        return val

    def evaluate_cascade(self, values: Dict[Any, Any]) -> Dict[Any, Any]:
        result = dict(values)
        changed = True
        passes = 0
        while changed and passes < 100:
            changed = False
            passes += 1
            for rule in self.rules:
                if rule.target_var in result:
                    continue
                if all(v in result for v in rule.input_vars):
                    val = self.evaluate(rule.target_var, result)
                    if val is not None:
                        result[rule.target_var] = val
                        changed = True
        return result

    def format_value(self, value: Any, precision_digits: int = 6) -> str:
        if isinstance(value, float):
            return f"{value:.{precision_digits}f}".rstrip("0").rstrip(".")
        return str(value)

    def _rule_for(self, target_var: Any) -> Optional[ZeroDofRule]:
        for r in self.rules:
            if r.target_var == target_var:
                return r
        return None


# aliases
ZeroDof = ZeroDofEngine
DeterministicCollapse = ZeroDofEngine
ZeroDofScheduler = ZeroDofEngine
CascadedEvaluator = ZeroDofEngine


def _eval_expr(expr, values):
    if expr is None:
        return None
    kind = getattr(expr, "kind", None)
    if kind in ("const_real", "const_int"):
        return expr.value
    if kind == "const_value":
        return expr.value
    if kind == "const_bool":
        return expr.value
    if kind == "const_str":
        return expr.value
    if kind == "const_list":
        return expr.value
    if kind == "const_dict":
        return expr.value
    if kind == "const_set":
        return expr.value
    if kind == "var":
        vid = getattr(expr, "var_id", None)
        if vid is not None and vid in values:
            return values[vid]
        name = getattr(expr, "name", None)
        if name is not None and name in values:
            return values[name]
        return None
    if kind == "name":
        name = getattr(expr, "name", None)
        if name in values:
            return values[name]
        return None
    if kind == "values":
        args = getattr(expr, "args", []) or []
        base = _eval_expr(args[0], values) if args else None
        if isinstance(base, dict):
            return list(base.values())
        if isinstance(base, (list, tuple)):
            return list(base)
        return base
    if kind == "subscript":
        args = getattr(expr, "args", []) or []
        base = _eval_expr(args[0], values) if args else None
        key = getattr(expr, "value", None)
        if isinstance(base, dict):
            return base.get(key)
        if isinstance(base, (list, tuple)):
            try:
                return base[key]
            except Exception:
                return None
        return None
    if kind == "add":
        a = _eval_binary(expr, values, 0)
        b = _eval_binary(expr, values, 1)
        if a is None or b is None:
            # try lhs/rhs
            la = _eval_expr(getattr(expr, "lhs", None), values)
            ra = _eval_expr(getattr(expr, "rhs", None), values)
            if la is None:
                la = a
            if ra is None:
                ra = b
            if la is None or ra is None:
                return None
            return la + ra
        return a + b
    if kind == "sub":
        a = _eval_expr(getattr(expr, "lhs", None), values)
        b = _eval_expr(getattr(expr, "rhs", None), values)
        if a is None:
            args = getattr(expr, "args", []) or []
            a = _eval_expr(args[0], values) if len(args) > 0 else None
            b = _eval_expr(args[1], values) if len(args) > 1 else b
        if a is None or b is None:
            return None
        return a - b
    if kind == "mul":
        a = _eval_expr(getattr(expr, "lhs", None), values)
        b = _eval_expr(getattr(expr, "rhs", None), values)
        if a is None or b is None:
            args = getattr(expr, "args", []) or []
            a = _eval_expr(args[0], values) if len(args) > 0 else a
            b = _eval_expr(args[1], values) if len(args) > 1 else b
        if a is None or b is None:
            return None
        return a * b
    if kind == "div":
        a = _eval_expr(getattr(expr, "lhs", None), values)
        b = _eval_expr(getattr(expr, "rhs", None), values)
        if a is None or b is None:
            args = getattr(expr, "args", []) or []
            a = _eval_expr(args[0], values) if len(args) > 0 else a
            b = _eval_expr(args[1], values) if len(args) > 1 else b
        if a is None or b is None or b == 0:
            return None
        return a / b
    if kind == "neg":
        args = getattr(expr, "args", []) or []
        a = _eval_expr(getattr(expr, "lhs", None), values)
        if a is None and args:
            a = _eval_expr(args[0], values)
        if a is None:
            return None
        return -a
    if kind == "sqrt":
        args = getattr(expr, "args", []) or []
        a = _eval_expr(args[0], values) if args else _eval_expr(getattr(expr, "lhs", None), values)
        if a is None:
            return None
        try:
            import math as _m
            return _m.sqrt(a)
        except Exception:
            return None
    if kind == "pow":
        a = _eval_expr(getattr(expr, "lhs", None), values)
        b = _eval_expr(getattr(expr, "rhs", None), values)
        if a is None or b is None:
            args = getattr(expr, "args", []) or []
            a = _eval_expr(args[0], values) if len(args) > 0 else a
            b = _eval_expr(args[1], values) if len(args) > 1 else b
        if a is None or b is None:
            return None
        return a ** b
    if kind == "dot":
        args = getattr(expr, "args", []) or []
        w = _eval_expr(args[0], values) if args else None
        r = getattr(expr, "value", None)
        if r is None and len(args) > 1:
            r = _eval_expr(args[1], values)
        if isinstance(w, dict):
            if isinstance(r, dict):
                return sum(w.get(k, 0) * r.get(k, 0) for k in (set(w) | set(r)))
            try:
                return sum(a * b for a, b in zip(list(w.values()), r))
            except Exception:
                return None
        try:
            return sum(a * b for a, b in zip(w, r))
        except Exception:
            return None
    if kind == "quad_form_cholesky":
        args = getattr(expr, "args", []) or []
        w = _eval_expr(args[0], values) if args else None
        cov = getattr(expr, "value", None)
        try:
            from veritas.decorators import quad_form as _qf
            return _qf(w, cov)
        except Exception:
            pass
        # manual
        if isinstance(w, dict) and isinstance(cov, dict):
            keys = list(w.keys())
            vec = [w[k] for k in keys]
            n = len(keys)
            mat = [[0.0] * n for _ in range(n)]
            for (a, b), v in cov.items():
                if a in keys and b in keys:
                    i = keys.index(a); j = keys.index(b)
                    mat[i][j] = float(v)
            total = 0.0
            for i in range(n):
                for j in range(n):
                    total += vec[i] * mat[i][j] * vec[j]
            return total
        return None
    if kind == "sum":
        args = getattr(expr, "args", []) or []
        a = _eval_expr(args[0], values) if args else None
        if isinstance(a, (list, tuple)):
            return sum(a)
        if isinstance(a, dict):
            return sum(a.values())
        return a
    if kind == "compare":
        # for derive, evaluate rhs
        r = _eval_expr(getattr(expr, "rhs", None), values)
        if r is not None:
            return r
        return _eval_expr(getattr(expr, "lhs", None), values)
    if kind == "keys_subset":
        args = getattr(expr, "args", []) or []
        d = _eval_expr(args[0], values) if args else None
        allowed = getattr(expr, "value", None)
        try:
            return set(d.keys()).issubset(set(allowed))
        except Exception:
            return None
    if kind == "forall":
        # best effort: evaluate predicate over iterable
        it = getattr(expr, "iterable", None)
        pred = getattr(expr, "predicate", None)
        iterable = _eval_expr(it, values) if it is not None else None
        if iterable is None:
            # maybe it is a values expr already
            iterable = _eval_expr(it, values) if hasattr(it, "kind") else it
        if iterable is None:
            return None
        try:
            elems = list(iterable) if not isinstance(iterable, dict) else list(iterable.values())
        except Exception:
            return None
        # predicate may reference elem via lambda; we can't eval easily; return None
        return None
    if kind == "sqrt":
        args = getattr(expr, "args", []) or []
        a = _eval_expr(args[0], values) if args else _eval_expr(getattr(expr, "lhs", None), values)
        if a is None:
            return None
        try:
            return math.sqrt(a)
        except Exception:
            return None
    if kind in ("and", "or"):
        args = getattr(expr, "args", []) or []
        vals = [_eval_expr(a, values) for a in args]
        if kind == "and":
            return all(bool(v) for v in vals if v is not None) if vals else None
        return any(bool(v) for v in vals if v is not None) if vals else None
    return None


def _eval_binary(expr, values, idx):
    args = getattr(expr, "args", []) or []
    if idx < len(args):
        return _eval_expr(args[idx], values)
    if idx == 0:
        return _eval_expr(getattr(expr, "lhs", None), values)
    if idx == 1:
        return _eval_expr(getattr(expr, "rhs", None), values)
    return None
