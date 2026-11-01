"""Static analysis of compiled specs: containers, element bounds and residual sums.

The docs' Trap 6 is real: a ``dict[str, float]`` field has no fixed arity, so a naive
frame either cannot say anything about it or is unsound.  We recover the useful facts
that ``vt.keys_subset``, ``vt.forall`` and ``vt.sum`` actually state, and attach them to
the spec as :class:`~veritas.compiler.vir.ContainerInfo` records.

Only *declared* facts are inferred.  A container with no constraints gets no universe and
no length bounds, which keeps feasibility checks conservative instead of inventing an
arity the user never wrote.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .vir import ContainerInfo, VirExpr, VirSpec

_CONTAINER_KINDS = ("dynamic_map", "list", "dict", "set")


def _is_container_var(expr: Optional[VirExpr], containers: Dict[int, str]) -> Optional[int]:
    if expr is None or getattr(expr, "kind", None) != "var":
        return None
    vid = getattr(expr, "var_id", None)
    if vid is None:
        return None
    return vid if vid in containers else None


def _unwrap_values(expr: Optional[VirExpr]) -> Optional[VirExpr]:
    """Return the underlying container for ``x.values()`` / bare ``x``."""

    if expr is None:
        return None
    if getattr(expr, "kind", None) in ("values", "sum"):
        args = getattr(expr, "args", []) or []
        return args[0] if args else None
    return expr


def _elem_names(expr: Optional[VirExpr], out: Optional[set] = None) -> set:
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
            _elem_names(child, out)
    for arg in getattr(expr, "args", []) or []:
        _elem_names(arg, out)
    return out


def _const_number(expr: Optional[VirExpr]) -> Optional[float]:
    if expr is None:
        return None
    if getattr(expr, "kind", None) in ("const_real", "const_int"):
        try:
            return float(expr.value)
        except (TypeError, ValueError):
            return None
    if getattr(expr, "kind", None) == "const_value" and isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool):
        return float(expr.value)
    if getattr(expr, "kind", None) == "neg":
        inner = _const_number(getattr(expr, "lhs", None))
        return -inner if inner is not None else None
    return None


def _apply_element_comparison(
    info: ContainerInfo, expr: VirExpr, elem_names: set
) -> None:
    """Extract ``elem_min`` / ``elem_max`` from a comparison on the lambda element."""

    if getattr(expr, "kind", None) != "compare":
        return
    lhs = getattr(expr, "lhs", None)
    rhs = getattr(expr, "rhs", None)
    op = getattr(expr, "op", "eq")

    def touches_elem(node: Optional[VirExpr]) -> bool:
        return (
            node is not None
            and getattr(node, "kind", None) == "elem"
            and (getattr(node, "name", None) in elem_names or len(elem_names) <= 1)
        )

    if touches_elem(lhs):
        bound = _const_number(rhs)
        if bound is None:
            return
        if op in ("le", "lt"):
            info.elem_max = min(info.elem_max, bound)
        elif op in ("ge", "gt"):
            info.elem_min = max(info.elem_min, bound)
        elif op == "eq":
            info.elem_min = max(info.elem_min, bound)
            info.elem_max = min(info.elem_max, bound)
    elif touches_elem(rhs):
        bound = _const_number(lhs)
        if bound is None:
            return
        # flip the relation: const OP elem  <=>  elem OP^-1 const
        flipped = {"le": "ge", "lt": "gt", "ge": "le", "gt": "lt", "eq": "eq", "ne": "ne"}.get(op, op)
        if flipped in ("le", "lt"):
            info.elem_max = min(info.elem_max, bound)
        elif flipped in ("ge", "gt"):
            info.elem_min = max(info.elem_min, bound)
        elif flipped == "eq":
            info.elem_min = max(info.elem_min, bound)
            info.elem_max = min(info.elem_max, bound)


def _scan_forall(info: ContainerInfo, expr: VirExpr) -> None:
    iterable = _unwrap_values(getattr(expr, "iterable", None))
    vid = _is_container_var(iterable, {info.var_id: info.name})
    if vid is None:
        return
    predicate = getattr(expr, "predicate", None)
    if predicate is None:
        return
    names = _elem_names(predicate)
    stack = [predicate]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        if getattr(node, "kind", None) == "and":
            stack.extend(node.args or [])
            continue
        _apply_element_comparison(info, node, names)


def _scan_sum(info: ContainerInfo, expr: VirExpr) -> None:
    """Recognize ``sum(container.values()) OP const`` and record the target sum."""

    if getattr(expr, "kind", None) != "compare":
        return
    lhs = getattr(expr, "lhs", None)
    rhs = getattr(expr, "rhs", None)
    op = getattr(expr, "op", "eq")
    target: Optional[float] = None
    total_expr: Optional[VirExpr] = None
    if lhs is not None and getattr(lhs, "kind", None) == "sum":
        total_expr = lhs
        target = _const_number(rhs)
    elif rhs is not None and getattr(rhs, "kind", None) == "sum":
        total_expr = rhs
        target = _const_number(lhs)
        op = {"le": "ge", "lt": "gt", "ge": "le", "gt": "lt", "eq": "eq", "ne": "ne"}.get(op, op)
    if total_expr is None or target is None:
        return
    inner = _unwrap_values(total_expr.args[0] if total_expr.args else None)
    vid = _is_container_var(inner, {info.var_id: info.name})
    if vid is None:
        return
    # Only equality gives a residual accumulator with a meaningful closed form; the
    # one-sided cases still bound feasibility and are recorded via the sum target.
    if op == "eq":
        info.target_sum = target
        info.sum_tol = max(info.sum_tol, float(getattr(expr, "tol", 1e-6) or 1e-6))
    else:
        info.target_sum = target


def analyze_containers(spec: VirSpec) -> Dict[int, ContainerInfo]:
    """Infer container facts from the spec's compiled constraints."""

    containers: Dict[int, ContainerInfo] = {}
    for vid, info in spec.variables.items():
        vtype = info.get("type")
        kind = getattr(vtype, "kind", "any")
        if kind not in _CONTAINER_KINDS:
            continue
        val_min = getattr(vtype, "val_min", float("-inf"))
        val_max = getattr(vtype, "val_max", float("inf"))
        min_len = int(getattr(vtype, "min_len", 0) or 0)
        raw_max = getattr(vtype, "max_len", 0) or 0
        containers[vid] = ContainerInfo(
            var_id=vid,
            name=info.get("name") or spec.id_to_name.get(vid, str(vid)),
            universe=list(vtype.universe) if getattr(vtype, "universe", None) else None,
            min_len=min_len,
            max_len=int(raw_max) if raw_max and raw_max > 0 else None,
            elem_min=float(val_min) if val_min is not None else float("-inf"),
            elem_max=float(val_max) if val_max is not None else float("inf"),
        )

    if not containers:
        return {}

    for expr in spec.requires:
        kind = getattr(expr, "kind", None)
        if kind == "keys_subset":
            args = getattr(expr, "args", []) or []
            vid = _is_container_var(args[0] if args else None, containers)
            allowed = getattr(expr, "value", None)
            if vid is not None and allowed is not None:
                try:
                    universe = sorted(str(k) for k in allowed)
                except TypeError:
                    universe = None
                if universe:
                    info = containers[vid]
                    info.universe = universe
                    if info.max_len is None or info.max_len > len(universe):
                        info.max_len = len(universe)
        elif kind == "forall":
            iterable = _unwrap_values(getattr(expr, "iterable", None))
            vid = _is_container_var(iterable, containers)
            if vid is not None:
                _scan_forall(containers[vid], expr)
        elif kind == "compare":
            lhs = getattr(expr, "lhs", None)
            inner = lhs
            if inner is not None and getattr(inner, "kind", None) == "sum":
                target_container = _unwrap_values(inner.args[0] if inner.args else None)
                vid = _is_container_var(target_container, containers)
                if vid is not None:
                    _scan_sum(containers[vid], expr)
        # Conjunctions are lowered as a single compare only when chained; nested calls
        # are separate top-level require() statements, so no recursion is needed here.

    for vid, info in containers.items():
        if not math.isfinite(info.elem_min):
            info.elem_min = float("-inf")
        if not math.isfinite(info.elem_max):
            info.elem_max = float("inf")
        if info.target_sum is not None and info.target_sum == 0:
            info.min_len = max(info.min_len, 0)
    return containers


def container_for(spec: VirSpec, var_id: Any) -> Optional[ContainerInfo]:
    return spec.containers.get(var_id) if isinstance(var_id, int) else None


def container_for_name(spec: VirSpec, name: str) -> Optional[ContainerInfo]:
    vid = spec.name_to_id.get(name)
    return spec.containers.get(vid) if vid is not None else None


def container_feasibility(spec: VirSpec) -> List[Tuple[int, str]]:
    """Return ``(constraint_index, reason)`` for provably infeasible container specs."""

    problems: List[Tuple[int, str]] = []
    if not spec.containers:
        return problems
    for index, expr in enumerate(spec.requires):
        if getattr(expr, "kind", None) != "compare":
            continue
        lhs = getattr(expr, "lhs", None)
        rhs = getattr(expr, "rhs", None)
        total = None
        target = None
        op = getattr(expr, "op", "eq")
        if lhs is not None and getattr(lhs, "kind", None) == "sum":
            total = lhs
            target = _const_number(rhs)
        elif rhs is not None and getattr(rhs, "kind", None) == "sum":
            total = rhs
            target = _const_number(lhs)
            op = {"le": "ge", "lt": "gt", "ge": "le", "gt": "lt", "eq": "eq"}.get(op, op)
        if total is None or target is None:
            continue
        inner = _unwrap_values(total.args[0] if total.args else None)
        vid = _is_container_var(inner, spec.containers)
        if vid is None:
            continue
        info = spec.containers[vid]
        if info.elem_min > info.elem_max:
            problems.append((index, f"element bounds for {info.name} are contradictory"))
            continue
        lo_len, hi_len = info.cardinality_bounds()
        # A single-sided element bound is still enough to bound the sum.  With at most
        # hi_len elements the minimum is hi_len*elem_min when elements may be negative
        # (choosing fewer elements only raises the sum), and the maximum is
        # hi_len*elem_max when elements may be positive.  If an element bound is
        # infinite, only the opposite side is usable.
        lo_sum = None
        if math.isfinite(info.elem_min):
            if info.elem_min > 0:
                lo_sum = lo_len * info.elem_min
            elif hi_len is not None:
                lo_sum = hi_len * info.elem_min
        hi_sum = None
        if math.isfinite(info.elem_max):
            if info.elem_max < 0:
                hi_sum = lo_len * info.elem_max
            elif hi_len is not None:
                hi_sum = hi_len * info.elem_max
        tol = float(getattr(expr, "tol", 1e-6) or 1e-6)
        if op == "eq":
            if lo_sum is not None and target < lo_sum - tol:
                problems.append(
                    (
                        index,
                        f"sum of {info.name} must equal {target:g} but cannot go below "
                        f"{lo_sum:g} by cardinality/element bounds",
                    )
                )
            elif hi_sum is not None and target > hi_sum + tol:
                problems.append(
                    (
                        index,
                        f"sum of {info.name} must equal {target:g} but cannot exceed "
                        f"{hi_sum:g} by cardinality/element bounds",
                    )
                )
        elif op in ("ge", "gt") and hi_sum is not None and target > hi_sum + tol:
            problems.append((index, f"sum lower bound {target:g} exceeds maximum {hi_sum:g}"))
        elif op in ("le", "lt") and lo_sum is not None and target < lo_sum - tol:
            problems.append((index, f"sum upper bound {target:g} is below minimum {lo_sum:g}"))
    return problems


__all__ = [
    "analyze_containers",
    "container_feasibility",
    "container_for",
    "container_for_name",
]
