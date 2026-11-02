"""Versioned, data-only intermediate representation.

VIR never contains executable Python.  Anything the compiler cannot represent must be
rejected at the authoring boundary rather than lowered to a vacuous expression.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VirType:
    kind: str = "any"
    min: float | None = None
    max: float | None = None
    tol: float = 0.0
    width: int | None = None
    universe: list[Any] | None = None
    min_len: int = 0
    max_len: int = 0
    val_min: float = float("-inf")
    val_max: float = float("inf")
    nullable: bool = False
    pattern: str | None = None
    text_format: str | None = None
    min_exclusive: bool = False
    max_exclusive: bool = False
    item_type: VirType | None = None
    properties: dict[str, VirType] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    additional_properties: bool = True
    model_name: str | None = None
    container_kind: str = "dict"

    @classmethod
    def real(cls, min=None, max=None, tol=1e-6):
        return cls(kind="real", min=min, max=max, tol=tol)

    @classmethod
    def int(cls, min=None, max=None):
        return cls(kind="int", min=min, max=max)

    @classmethod
    def bool(cls):
        return cls(kind="bool")

    @classmethod
    def dynamic_map(cls, allowed_keys=None, min_len=0, max_len=0,
                    val_min=float("-inf"), val_max=float("inf")):
        return cls(kind="dynamic_map", universe=allowed_keys, min_len=min_len,
                   max_len=max_len, val_min=val_min, val_max=val_max)

    def accepts_number(self, value: float) -> bool:
        if self.min is not None and (value <= self.min if self.min_exclusive else value < self.min):
            return False
        if self.max is not None and (value >= self.max if self.max_exclusive else value > self.max):
            return False
        return True


@dataclass
class VirExpr:
    kind: str = "expr"
    var_id: int | None = None
    name: str | None = None
    value: Any = None
    op: str | None = None
    args: list[VirExpr] = field(default_factory=list)
    lhs: VirExpr | None = None
    rhs: VirExpr | None = None
    tol: float = 1e-6
    iterable: Any = None
    predicate: VirExpr | None = None
    container_id: int | None = None
    target_sum: float | None = None
    elem_min: float = 0.0
    elem_max: float = 0.0
    min_remaining: int = 0
    max_remaining: int = 0
    weights_vars: list[int] = field(default_factory=list)
    cholesky_l: list[list[float]] = field(default_factory=list)
    weight: float = 1.0
    tag: str = ""

    def collect_var_ids(self) -> list[int]:
        out: list[int] = []
        seen = set()
        def visit(node):
            if node is None or not hasattr(node, "kind"):
                return
            if node.kind == "var" and node.var_id is not None and node.var_id not in seen:
                seen.add(node.var_id)
                out.append(node.var_id)
            for child in (node.lhs, node.rhs, node.predicate, node.iterable, *node.args):
                visit(child)
        visit(self)
        return out

    def collect_var_names(self, id_to_name: dict[int, str]) -> list[str]:
        return [id_to_name[v] for v in self.collect_var_ids() if v in id_to_name]


@dataclass
class ZeroDofRule:
    target_var: Any = None
    target_json_key: str | None = None
    input_vars: list[Any] = field(default_factory=list)
    rhs_expr: VirExpr | None = None
    rhs_eval: Any | None = None
    precision_digits: int = 6


@dataclass
class SoftPreference:
    id: int
    tag: str = ""
    weight: float = 1.0
    expr: VirExpr | None = None


@dataclass
class ContainerInfo:
    var_id: int
    name: str = ""
    universe: list[str] | None = None
    min_len: int = 0
    max_len: int | None = None
    elem_min: float = float("-inf")
    elem_max: float = float("inf")
    elem_min_exclusive: bool = False
    elem_max_exclusive: bool = False
    target_sum: float | None = None
    sum_tol: float = 1e-6
    sum_op: str = "eq"
    container_kind: str = "dict"

    def cardinality_bounds(self):
        lo = max(0, self.min_len)
        hi = self.max_len
        if self.container_kind == "dict" and self.universe is not None:
            hi = len(self.universe) if hi is None else min(hi, len(self.universe))
        return lo, hi

    def residual_feasible(self, current_sum: float, emitted: int) -> bool:
        if self.target_sum is None:
            return True
        lo_len, hi_len = self.cardinality_bounds()
        remaining = self.target_sum - current_sum
        m_min = max(0, lo_len - emitted)
        if hi_len is None:
            if self.elem_min <= 0 <= self.elem_max:
                return True
            if self.elem_min > 0:
                return remaining >= m_min * self.elem_min - self.sum_tol
            return remaining <= m_min * self.elem_max + self.sum_tol
        m_max = hi_len - emitted
        if m_max < m_min:
            return False
        # The union over integer suffix lengths is an interval here because consecutive
        # lengths have overlapping reachable sums when zero lies in the element domain;
        # otherwise all element signs agree and the extreme cardinalities are sufficient.
        low = min(m_min * self.elem_min, m_max * self.elem_min)
        high = max(m_min * self.elem_max, m_max * self.elem_max)
        if self.elem_min <= 0 <= self.elem_max:
            low = m_max * self.elem_min
            high = m_max * self.elem_max
        return low - self.sum_tol <= remaining <= high + self.sum_tol

    def residual_interval(self, current_sum: float, emitted: int):
        if self.target_sum is None or self.sum_op != "eq":
            return self.elem_min, self.elem_max
        lo_len, hi_len = self.cardinality_bounds()
        if hi_len is None or hi_len <= emitted:
            return (0.0, 0.0) if hi_len == emitted else (self.elem_min, self.elem_max)
        # This is the NEXT element: subtract one before computing suffix cardinalities.
        after = emitted + 1
        m_min = max(0, lo_len - after)
        m_max = hi_len - after
        remaining = self.target_sum - current_sum
        tail_lo = min(m_min * self.elem_min, m_max * self.elem_min)
        tail_hi = max(m_min * self.elem_max, m_max * self.elem_max)
        if self.elem_min <= 0 <= self.elem_max:
            tail_lo, tail_hi = m_max * self.elem_min, m_max * self.elem_max
        return (max(self.elem_min, remaining - tail_hi),
                min(self.elem_max, remaining - tail_lo))


@dataclass
class VirSpec:
    model: Any = None
    variables: dict[int, dict[str, Any]] = field(default_factory=dict)
    id_to_name: dict[int, str] = field(default_factory=dict)
    name_to_id: dict[str, int] = field(default_factory=dict)
    field_order: list[str] = field(default_factory=list)
    requires: list[VirExpr] = field(default_factory=list)
    derives: list[VirExpr] = field(default_factory=list)
    prefers: list[VirExpr] = field(default_factory=list)
    zero_dof_rules: list[ZeroDofRule] = field(default_factory=list)
    soft_preferences: list[SoftPreference] = field(default_factory=list)
    containers: dict[int, ContainerInfo] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    model_schema: dict[str, Any] | None = None
    field_aliases: dict[str, str] = field(default_factory=dict)
    portable_model: bool = True
    compiler_diagnostics: list[str] = field(default_factory=list)

    def all_constraints(self):
        return list(self.requires) + list(self.derives)

    def derived_names(self):
        return list(dict.fromkeys(r.target_json_key for r in self.zero_dof_rules if r.target_json_key))

    def is_derived(self, name):
        return name in self.derived_names()


@dataclass
class PreflightResult:
    status: str = "UNKNOWN"
    conflicts: list[int] = field(default_factory=list)
    constraint_indices: list[int] = field(default_factory=list)
    relaxation: dict[str, Any] | None = None
    domains: dict[int, tuple] = field(default_factory=dict)
    witness: dict[str, Any] | None = None
    details: list[str] = field(default_factory=list)

    @property
    def is_sat(self):
        return self.status == "SAT"