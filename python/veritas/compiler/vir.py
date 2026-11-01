"""Veritas Intermediate Representation (VIR) for Phase 1 (Python mirror of Rust vir.rs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


VarId = int
ClauseId = int


@dataclass
class VirType:
    kind: str = "any"
    min: Optional[float] = None
    max: Optional[float] = None
    tol: float = 1e-6
    width: Optional[int] = None
    universe: Optional[List[str]] = None
    min_len: int = 0
    max_len: int = 0
    val_min: float = float("-inf")
    val_max: float = float("inf")

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
    def dynamic_map(cls, allowed_keys=None, min_len=0, max_len=0, val_min=float("-inf"), val_max=float("inf")):
        return cls(kind="dynamic_map", universe=allowed_keys, min_len=min_len, max_len=max_len, val_min=val_min, val_max=val_max)


@dataclass
class VirExpr:
    kind: str = "expr"
    var_id: Optional[int] = None
    name: Optional[str] = None
    value: Any = None
    op: Optional[str] = None
    args: List["VirExpr"] = field(default_factory=list)
    lhs: Optional["VirExpr"] = None
    rhs: Optional["VirExpr"] = None
    tol: float = 1e-6
    iterable: Any = None
    predicate: Optional["VirExpr"] = None
    # dynamic containers
    container_id: Optional[int] = None
    target_sum: Optional[float] = None
    elem_min: float = 0.0
    elem_max: float = 0.0
    min_remaining: int = 0
    max_remaining: int = 0
    # quad form
    weights_vars: List[int] = field(default_factory=list)
    cholesky_l: List[List[float]] = field(default_factory=list)
    # soft preference
    weight: float = 1.0
    tag: str = ""

    def collect_var_ids(self) -> List[int]:
        out: List[int] = []
        if self.kind == "var" and self.var_id is not None:
            out.append(self.var_id)
        if self.container_id is not None:
            out.append(self.container_id)
        if self.lhs is not None:
            out.extend(self.lhs.collect_var_ids())
        if self.rhs is not None:
            out.extend(self.rhs.collect_var_ids())
        for a in self.args:
            out.extend(a.collect_var_ids())
        if self.predicate is not None:
            out.extend(self.predicate.collect_var_ids())
        if hasattr(self.iterable, "collect_var_ids"):
            out.extend(self.iterable.collect_var_ids())
        # deduplicate while preserving order
        seen = set()
        uniq = []
        for v in out:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    def collect_var_names(self, id_to_name: Dict[int, str]) -> List[str]:
        return [id_to_name[v] for v in self.collect_var_ids() if v in id_to_name]


@dataclass
class ZeroDofRule:
    target_var: Any = None
    target_json_key: Optional[str] = None
    input_vars: List[Any] = field(default_factory=list)
    rhs_expr: Optional[VirExpr] = None
    rhs_eval: Optional[Any] = None
    precision_digits: int = 6


@dataclass
class SoftPreference:
    id: int
    tag: str = ""
    weight: float = 1.0
    expr: Optional[VirExpr] = None


@dataclass
class VirSpec:
    model: Any = None
    variables: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    id_to_name: Dict[int, str] = field(default_factory=dict)
    name_to_id: Dict[str, int] = field(default_factory=dict)
    field_order: List[str] = field(default_factory=list)
    requires: List[VirExpr] = field(default_factory=list)
    derives: List[VirExpr] = field(default_factory=list)
    prefers: List[VirExpr] = field(default_factory=list)
    zero_dof_rules: List[ZeroDofRule] = field(default_factory=list)
    soft_preferences: List[SoftPreference] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def all_constraints(self) -> List[VirExpr]:
        return list(self.requires) + list(self.derives)


@dataclass
class PreflightResult:
    status: str = "SAT"
    conflicts: List[int] = field(default_factory=list)
    constraint_indices: List[int] = field(default_factory=list)
    relaxation: Optional[Dict[str, Any]] = None
    domains: Dict[int, tuple] = field(default_factory=dict)

    @property
    def is_sat(self) -> bool:
        return self.status == "SAT"
