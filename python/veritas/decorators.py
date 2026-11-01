"""Public decorators and DSL primitives for veritas Phase 1."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional

from .compiler.ast_transpiler import AstTranspiler
from .compiler.dependency_graph import DependencyGraph, CircularDerivationError
from .compiler.schema_lowering import lower_pydantic_model
from .compiler.preflight import run_preflight, InfeasibleSpecError
from .compiler.vir import VirSpec


class VerifiedSchema:
    def __init__(self, func: Callable, **config):
        self.func = func
        self.config = {
            "solver": "auto",
            "reasoning_mode": "crane_cot",
            "backtrack": "cdcl_domain_cut",
            "max_backtracks": 8,
            "zero_dof_collapse": True,
            "preflight_check": True,
        }
        self.config.update(config)
        self.name = getattr(func, "__name__", "verified_schema")
        self.model = self._resolve_model(func)
        self.transpiler = AstTranspiler(func, self.model, self.config)
        self.spec: VirSpec = self.transpiler.transpile()
        # topological field order already set in spec
        self.field_order = list(self.spec.field_order)
        self.graph: DependencyGraph = self.transpiler.graph
        # preflight
        self.preflight = None
        if self.config.get("preflight_check", True):
            self.preflight = run_preflight(self.spec)
            if not self.preflight.is_sat:
                raise InfeasibleSpecError(
                    f"Spec {self.name} is infeasible at pre-flight",
                    conflicts=self.preflight.conflicts,
                    relaxation=self.preflight.relaxation,
                )
        # convenience alias
        self.preflight_result = self.preflight

    def _resolve_model(self, func):
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params:
            raise TypeError("verified_schema function must accept at least one parameter")
        ann = params[0].annotation
        if ann is inspect._empty:
            ann = func.__annotations__.get(params[0].name) if hasattr(func, "__annotations__") else None
        if isinstance(ann, str):
            s = ann.strip()
            if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                s = s[1:-1].strip()
            try:
                ann = eval(s, {"__builtins__": {}}, getattr(func, "__globals__", {}))
            except Exception:
                ann = getattr(func, "__globals__", {}).get(s, ann)
        return ann

    def __call__(self, *args, **kwargs):
        # A schema object is not directly callable as the original function;
        # return self to allow use as spec.
        return self

    def __repr__(self):
        return f"VerifiedSchema({self.name})"


def verified_schema(
    func=None,
    solver: str = "auto",
    reasoning_mode: str = "crane_cot",
    backtrack: str = "cdcl_domain_cut",
    max_backtracks: int = 8,
    zero_dof_collapse: bool = True,
    preflight_check: bool = True,
    **extra,
):
    config = {
        "solver": solver,
        "reasoning_mode": reasoning_mode,
        "backtrack": backtrack,
        "max_backtracks": max_backtracks,
        "zero_dof_collapse": zero_dof_collapse,
        "preflight_check": preflight_check,
    }
    config.update(extra)

    def decorator(f):
        return VerifiedSchema(f, **config)

    if callable(func) and not isinstance(func, type):
        return decorator(func)
    if func is not None:
        # allow positional call with func passed? fallback
        return decorator(func)
    return decorator


# ---------- DSL primitives ----------
# These are mostly AST markers; runtime versions provide best-effort evaluation.

def require(expr, tol: float = 1e-6):
    return expr


def derive(expr, tol: float = 1e-6):
    return expr


def prefer(expr, weight: float = 1.0, tag: str = ""):
    return expr


def forall(iterable, predicate):
    return ("forall", iterable, predicate)


def sum_(x):  # avoid shadowing builtin? expose as sum
    if hasattr(x, "__iter__"):
        try:
            return sum(x)
        except Exception:
            return x
    return x


# We'll expose `sum` name below by aliasing to a wrapper that doesn't break builtin usage in this module.
def _veritas_sum(*args):
    if len(args) == 1:
        x = args[0]
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes, dict)):
            try:
                return sum(x)
            except Exception:
                return x
        return x
    return sum(args)


def dot(w, r):
    try:
        if isinstance(w, dict) and isinstance(r, dict):
            return sum(w.get(k, 0) * r.get(k, 0) for k in set(w) | set(r))
        # if w/r are vector-like
        return sum(a * b for a, b in zip(w, r))
    except Exception:
        return None


def quad_form(w, cov):
    # runtime helper: compute w^T cov w
    try:
        if isinstance(w, dict):
            keys = list(w.keys())
            vec = [w[k] for k in keys]
        else:
            vec = list(w)
            keys = None
        if isinstance(cov, dict):
            # cov keys may be tuples
            n = len(vec)
            # build matrix
            if keys is None:
                keys = list(range(n))
            mat = [[0.0] * n for _ in range(n)]
            for (a, b), v in cov.items():
                i = keys.index(a) if a in keys else None
                j = keys.index(b) if b in keys else None
                if i is not None and j is not None:
                    mat[i][j] = float(v)
            total = 0.0
            for i in range(n):
                for j in range(n):
                    total += vec[i] * mat[i][j] * vec[j]
            return total
        import numpy as np
        arr = np.array(cov, dtype=float)
        vec_arr = np.array(vec, dtype=float)
        return float(vec_arr @ arr @ vec_arr)
    except Exception:
        return None


def sqrt(x):
    try:
        import math
        return math.sqrt(x)
    except Exception:
        try:
            return x ** 0.5
        except Exception:
            return None


def keys_subset(d, allowed):
    try:
        return set(d.keys()).issubset(set(allowed))
    except Exception:
        return None


# Expose `sum` as veritas.sum (shadowing builtin only in this module namespace)
sum = _veritas_sum
# keep builtin accessible
_builtin_sum = __builtins__["sum"] if isinstance(__builtins__, dict) else __builtins__.sum
# Export aliases
__all__ = [
    "verified_schema",
    "VerifiedSchema",
    "require",
    "derive",
    "prefer",
    "forall",
    "sum",
    "dot",
    "quad_form",
    "sqrt",
    "keys_subset",
    "CircularDerivationError",
    "InfeasibleSpecError",
]
