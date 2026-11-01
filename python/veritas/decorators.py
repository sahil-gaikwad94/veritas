"""Public decorators and DSL primitives for veritas Phase 1."""

from __future__ import annotations

import builtins
import functools
import inspect
from collections.abc import Callable
from typing import Any

from .compiler.ast_transpiler import AstTranspiler
from .compiler.dependency_graph import CircularDerivationError, DependencyGraph
from .compiler.preflight import InfeasibleSpecError, run_preflight
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
        self._validate_config()
        functools.update_wrapper(self, func)
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
            if self.preflight.status == "UNSAT":
                raise InfeasibleSpecError(
                    f"Spec {self.name} is infeasible at pre-flight",
                    conflicts=self.preflight.conflicts,
                    relaxation=self.preflight.relaxation,
                )
        # convenience alias
        self.preflight_result = self.preflight

    def _validate_config(self):
        choices = {
            "solver": {"auto", "nano", "z3"},
            "reasoning_mode": {"crane_cot", "strict"},
            "backtrack": {"cdcl_domain_cut", "none"},
        }
        for key, allowed in choices.items():
            if self.config[key] not in allowed:
                raise ValueError(f"{key} must be one of {sorted(allowed)}")
        count = self.config["max_backtracks"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("max_backtracks must be a non-negative integer")
        for key in ("zero_dof_collapse", "preflight_check"):
            if not isinstance(self.config[key], bool):
                raise ValueError(f"{key} must be boolean")

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

    # ---- concrete verification / derivation ----------------------------
    def verify(self, instance: Any, *, derive_missing: bool = True):
        """Check a concrete instance and return a :class:`VerificationReport`."""

        from .instance import verify_instance

        return verify_instance(self, instance, derive_missing=derive_missing)

    def validate(self, instance: Any, *, derive_missing: bool = True):
        """Check an instance and raise :class:`VerificationError` on failure."""

        report = self.verify(instance, derive_missing=derive_missing)
        report.raise_if_failed()
        return report.model if report.model is not None else report.values

    def derive(self, **values: Any):
        """Fill every currently-ready 0-DoF field and return the resulting dict."""

        from .instance import fill_derived

        filled, _unresolved, _derived = fill_derived(self.spec, values)
        return filled

    def describe(self):
        from .instance import describe_spec

        return describe_spec(self.spec)

    @property
    def digest(self):
        from .serde import spec_digest

        return spec_digest(self.spec)

    @property
    def containers(self):
        return dict(self.spec.containers)


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
    unknown = set(extra) - {"strict"}
    if unknown:
        raise ValueError(f"unknown schema options: {sorted(unknown)}")
    config.update(extra)

    def decorator(f):
        return VerifiedSchema(f, **config)

    if isinstance(func, VerifiedSchema):
        return func
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


def veritas_sum(*args):
    """``vt.sum``: identity for a single iterable, builtin addition for varargs."""

    if len(args) == 1:
        value = args[0]
        if isinstance(value, dict):
            try:
                return builtins.sum(value.values())
            except Exception:
                return value
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            try:
                return builtins.sum(value)
            except Exception:
                return value
        return value
    return builtins.sum(args)


def dot(w, r):
    """``vt.dot``: dict·dict, dict·sequence, or sequence·sequence."""

    try:
        if isinstance(w, dict):
            if isinstance(r, dict):
                return builtins.sum(w.get(k, 0.0) * r.get(k, 0.0) for k in set(w) | set(r))
            return builtins.sum(value * other for value, other in zip(w.values(), r))
        if isinstance(r, dict):
            return builtins.sum(value * r.get(k, 0.0) for k, value in zip(r.keys(), w))
        return builtins.sum(a * b for a, b in zip(w, r))
    except Exception:
        return None


def quad_form(w, cov):
    """``vt.quad_form``: evaluate wᵀΣw, symmetrizing a triangular covariance map."""

    try:
        if isinstance(w, dict):
            labels = list(w.keys())
            vec = [w[k] for k in labels]
        else:
            vec = list(w)
            labels = None
        if isinstance(cov, dict):
            if labels is None:
                labels = sorted({k[0] if isinstance(k, tuple) else k for k in cov})
            n = len(vec)
            matrix = [[0.0] * n for _ in range(n)]
            for key, value in cov.items():
                a, b = key if isinstance(key, tuple) else (key, key)
                if a not in labels or b not in labels:
                    continue
                i, j = labels.index(a), labels.index(b)
                matrix[i][j] = float(value)
                if i != j:
                    matrix[j][i] = float(value)
            total = 0.0
            for i in range(n):
                for j in range(n):
                    total += vec[i] * matrix[i][j] * vec[j]
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


# Identity markers let the compiler distinguish the DSL from similarly named methods.
for _name, _fn in list(globals().items()):
    if _name in {"require", "derive", "prefer", "forall", "dot", "quad_form", "sqrt", "keys_subset"}:
        _fn._veritas_dsl = _name
veritas_sum._veritas_dsl = "sum"

# Expose `sum` under the documented ``vt.sum`` name.
sum = veritas_sum
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
