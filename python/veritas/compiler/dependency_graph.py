"""Bipartite dependency graph and topological ordering for veritas Phase 1."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .vir import VirType


class CircularDerivationError(Exception):
    """Raised when vt.derive rules contain a cycle."""


class DependencyGraph:
    def __init__(self):
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        self._types: Dict[int, Any] = {}
        self._next_id: int = 0
        # derive rules: target var id -> list of input var ids
        self._derive_inputs: Dict[int, List[int]] = {}
        self._derive_exprs: Dict[int, Any] = {}

    # ---- variables ----
    def add_variable(self, name: str, vir_type: Optional[VirType] = None) -> int:
        if name in self._name_to_id:
            return self._name_to_id[name]
        vid = self._next_id
        self._next_id += 1
        self._name_to_id[name] = vid
        self._id_to_name[vid] = name
        self._types[vid] = vir_type
        return vid

    def var_id(self, name: str) -> int:
        if name not in self._name_to_id:
            return self.add_variable(name)
        return self._name_to_id[name]

    def name_of(self, vid: int) -> str:
        return self._id_to_name[vid]

    def type_of(self, vid: int) -> Any:
        return self._types.get(vid)

    @property
    def name_to_id(self) -> Dict[str, int]:
        return dict(self._name_to_id)

    @property
    def id_to_name(self) -> Dict[int, str]:
        return dict(self._id_to_name)

    # ---- derives ----
    def add_derive(self, target_name: str, input_names: List[str], expr: Any = None, tol: float = 1e-6) -> None:
        target_id = self.var_id(target_name)
        input_ids = [self.var_id(n) for n in input_names]
        # merge inputs if derive already exists
        existing = self._derive_inputs.get(target_id, [])
        merged = list(dict.fromkeys(existing + input_ids))
        self._derive_inputs[target_id] = merged
        if expr is not None:
            self._derive_exprs[target_id] = expr

    def derive_inputs(self, target_name: str) -> List[str]:
        vid = self._name_to_id.get(target_name)
        if vid is None:
            return []
        return [self._id_to_name[i] for i in self._derive_inputs.get(vid, [])]

    def is_derived(self, name: str) -> bool:
        vid = self._name_to_id.get(name)
        return vid is not None and vid in self._derive_inputs

    # ---- topological order ----
    def topological_order(self) -> List[str]:
        # nodes: all variables
        nodes: Set[int] = set(self._id_to_name.keys())
        # adjacency: input -> target
        in_degree: Dict[int, int] = {n: 0 for n in nodes}
        adj: Dict[int, List[int]] = {n: [] for n in nodes}
        for target_id, inputs in self._derive_inputs.items():
            for inp in inputs:
                if inp in nodes and target_id in nodes:
                    adj[inp].append(target_id)
                    in_degree[target_id] += 1
        # Kahn's algorithm; stable by insertion order of variable ids
        from collections import deque
        q = deque([n for n in sorted(nodes) if in_degree[n] == 0])
        order: List[int] = []
        while q:
            n = q.popleft()
            order.append(n)
            for m in adj[n]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    q.append(m)
        if len(order) != len(nodes):
            raise CircularDerivationError(
                "Cycle detected in vt.derive dependency graph"
            )
        return [self._id_to_name[vid] for vid in order]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name_to_id": dict(self._name_to_id),
            "id_to_name": dict(self._id_to_name),
            "derive_inputs": {self._id_to_name[k]: [self._id_to_name[v] for v in vs] for k, vs in self._derive_inputs.items()},
        }
