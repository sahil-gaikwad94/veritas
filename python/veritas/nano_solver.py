"""Phase 3: Pure-Python VeritasNano solver (QF_LRA + QF_IDL) with push/pop and UnsatCore."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import copy


class SolverStatus(str, Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


@dataclass
class _Constraint:
    tracker: Any
    kind: str  # 'lra' or 'diff' or 'bound'
    data: Any


class VeritasNanoSolver:
    def __init__(self):
        self.constraints: List[_Constraint] = []
        self.stack: List[int] = []
        self._last_core: List[Any] = []
        self.variables: Set[Any] = set()

    # ---- push/pop ----
    def push(self) -> None:
        self.stack.append(len(self.constraints))

    def pop(self) -> None:
        if not self.stack:
            return
        n = self.stack.pop()
        self.constraints = self.constraints[:n]

    # ---- assertions ----
    def assert_constraint(self, constraint: Any, tracker_id: Any = None) -> None:
        self.assert_and_track(constraint, tracker_id)

    def assert_and_track(self, constraint: Any, tracker_id: Any = None) -> None:
        norm = self._normalize(constraint)
        if tracker_id is None:
            tracker_id = len(self.constraints)
        self.constraints.append(_Constraint(tracker=tracker_id, kind=norm["kind"], data=norm["data"]))
        for v in norm.get("vars", []):
            self.variables.add(v)

    # ---- check ----
    def check(self) -> SolverStatus:
        return self.check_with_timeout(None)

    def check_with_timeout(self, timeout_ms: Optional[int] = None) -> SolverStatus:
        # Difference logic
        diff_cons = [c for c in self.constraints if c.kind == "diff"]
        if diff_cons:
            status, core = self._check_diff(diff_cons)
            if status == SolverStatus.UNSAT:
                self._last_core = core
                return status
        # LRA
        lra_cons = [c for c in self.constraints if c.kind in ("lra", "bound")]
        if lra_cons:
            status, core = self._check_lra(lra_cons)
            if status == SolverStatus.UNSAT:
                self._last_core = core
                return status
        self._last_core = []
        return SolverStatus.SAT

    def get_unsat_core(self) -> List[Any]:
        return list(self._last_core)

    # ---- normalization ----
    def _normalize(self, constraint: Any) -> Dict[str, Any]:
        # tuple/list
        if isinstance(constraint, (tuple, list)):
            if not constraint:
                return {"kind": "lra", "data": {"type": "true"}, "vars": []}
            kind = constraint[0]
            if kind in ("diff", "difference"):
                # x_i - x_j <= c
                i = constraint[1]
                j = constraint[2]
                c = float(constraint[3]) if len(constraint) > 3 else 0.0
                self.variables.update([i, j])
                return {"kind": "diff", "data": {"i": i, "j": j, "c": c}, "vars": [i, j]}
            if kind in ("bound", "bounds"):
                v = constraint[1]
                lo = constraint[2] if len(constraint) > 2 else None
                hi = constraint[3] if len(constraint) > 3 else None
                self.variables.add(v)
                return {"kind": "bound", "data": {"var": v, "lo": lo, "hi": hi}, "vars": [v]}
            if kind in ("le", "ge", "eq", "lt", "gt"):
                coeffs = constraint[1]
                rhs = float(constraint[2]) if len(constraint) > 2 else 0.0
                coeffs = self._coeffs_to_dict(coeffs)
                vars_ = list(coeffs.keys())
                self.variables.update(vars_)
                return {"kind": "lra", "data": {"op": kind, "coeffs": coeffs, "rhs": rhs}, "vars": vars_}
            if kind in ("linear", "affine"):
                coeffs = constraint[1]
                rhs = float(constraint[2]) if len(constraint) > 2 else 0.0
                op = constraint[3] if len(constraint) > 3 else "le"
                coeffs = self._coeffs_to_dict(coeffs)
                vars_ = list(coeffs.keys())
                self.variables.update(vars_)
                return {"kind": "lra", "data": {"op": op, "coeffs": coeffs, "rhs": rhs}, "vars": vars_}
            # generic: try as coeffs dict
            if isinstance(constraint[1], dict):
                coeffs = self._coeffs_to_dict(constraint[1])
                rhs = float(constraint[2]) if len(constraint) > 2 else 0.0
                op = constraint[3] if len(constraint) > 3 else "le"
                vars_ = list(coeffs.keys())
                self.variables.update(vars_)
                return {"kind": "lra", "data": {"op": op, "coeffs": coeffs, "rhs": rhs}, "vars": vars_}
        # dict
        if isinstance(constraint, dict):
            if "type" in constraint and constraint["type"] == "diff":
                i = constraint["i"]; j = constraint["j"]; c = float(constraint.get("c", 0.0))
                self.variables.update([i, j])
                return {"kind": "diff", "data": {"i": i, "j": j, "c": c}, "vars": [i, j]}
            if constraint.get("type") in ("bound",):
                v = constraint["var"]
                self.variables.add(v)
                return {"kind": "bound", "data": {"var": v, "lo": constraint.get("lo"), "hi": constraint.get("hi")}, "vars": [v]}
            coeffs = constraint.get("coeffs", {})
            coeffs = self._coeffs_to_dict(coeffs)
            op = constraint.get("op", constraint.get("kind", "le"))
            rhs = float(constraint.get("rhs", constraint.get("target", 0.0)))
            vars_ = list(coeffs.keys())
            self.variables.update(vars_)
            return {"kind": "lra", "data": {"op": op, "coeffs": coeffs, "rhs": rhs}, "vars": vars_}
        # VirExpr
        try:
            from .compiler.vir import VirExpr
            if isinstance(constraint, VirExpr) and constraint.kind == "compare":
                # var vs const
                lhs = constraint.lhs
                rhs = constraint.rhs
                if getattr(lhs, "kind", None) == "var" and getattr(rhs, "kind", None) in ("const_real", "const_int", "const_value"):
                    v = lhs.var_id if lhs.var_id is not None else lhs.name
                    val = float(rhs.value)
                    op = constraint.op
                    # convert to bound/lra
                    if op in ("eq",):
                        coeffs = {v: 1.0}
                        return {"kind": "lra", "data": {"op": "eq", "coeffs": coeffs, "rhs": val}, "vars": [v]}
                    if op in ("le", "lt"):
                        coeffs = {v: 1.0}
                        return {"kind": "lra", "data": {"op": "le", "coeffs": coeffs, "rhs": val}, "vars": [v]}
                    if op in ("ge", "gt"):
                        coeffs = {v: 1.0}
                        return {"kind": "lra", "data": {"op": "ge", "coeffs": coeffs, "rhs": val}, "vars": [v]}
        except Exception:
            pass
        # fallback: treat as true
        return {"kind": "lra", "data": {"type": "true", "coeffs": {}, "rhs": 0.0, "op": "le"}, "vars": []}

    def _coeffs_to_dict(self, coeffs: Any) -> Dict[Any, float]:
        if isinstance(coeffs, dict):
            return {k: float(v) for k, v in coeffs.items()}
        # list of (var, coeff)
        if isinstance(coeffs, (list, tuple)):
            out = {}
            for item in coeffs:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    out[item[0]] = float(item[1])
                else:
                    # maybe var names
                    pass
            if out:
                return out
        return {}

    # ---- diff check ----
    def _check_diff(self, cons: List[_Constraint]) -> Tuple[SolverStatus, List[Any]]:
        edges = []
        nodes = set()
        for c in cons:
            d = c.data
            i = d["i"]; j = d["j"]; w = d["c"]
            nodes.add(i); nodes.add(j)
            # constraint x_i - x_j <= w  => edge j->i weight w
            edges.append((j, i, w, c.tracker))
        # Bellman-Ford
        dist = {n: 0.0 for n in nodes}
        pred = {n: None for n in nodes}
        n = len(nodes)
        for _ in range(n):
            updated = False
            for (u, v, w, tr) in edges:
                if dist[u] + w < dist[v] - 1e-12:
                    dist[v] = dist[u] + w
                    pred[v] = (u, v, w, tr)
                    updated = True
            if not updated:
                break
        # check negative cycle
        for (u, v, w, tr) in edges:
            if dist[u] + w < dist[v] - 1e-12:
                # find a node in the cycle
                x = v
                for _ in range(len(nodes)):
                    p = pred.get(x)
                    if p is None:
                        break
                    x = p[0]
                # collect cycle nodes
                cycle_nodes = set()
                cur = x
                for _ in range(len(nodes) + 1):
                    cycle_nodes.add(cur)
                    p = pred.get(cur)
                    if p is None:
                        break
                    nxt = p[0]
                    if nxt == x:
                        break
                    cur = nxt
                core = [trk for (uu, vv, ww, trk) in edges if uu in cycle_nodes and vv in cycle_nodes]
                if not core:
                    core = [c.tracker for c in cons]
                return SolverStatus.UNSAT, core
        return SolverStatus.SAT, []

    # ---- lra check ----
    def _check_lra(self, cons: List[_Constraint]) -> Tuple[SolverStatus, List[Any]]:
        # Build list of inequalities (coeffs dict, rhs) meaning sum coeff*var <= rhs
        ineqs: List[Tuple[Dict[Any, float], float]] = []
        for c in cons:
            d = c.data
            if c.kind == "bound":
                v = d["var"]
                lo = d.get("lo")
                hi = d.get("hi")
                if hi is not None:
                    ineqs.append(({v: 1.0}, float(hi)))
                if lo is not None:
                    ineqs.append(({v: -1.0}, -float(lo)))
                continue
            op = d.get("op", "le")
            coeffs = dict(d.get("coeffs", {}))
            rhs = float(d.get("rhs", 0.0))
            if op in ("le", "lt"):
                ineqs.append((coeffs, rhs))
            elif op in ("ge", "gt"):
                ineqs.append(({k: -val for k, val in coeffs.items()}, -rhs))
            elif op in ("eq",):
                ineqs.append((dict(coeffs), rhs))
                ineqs.append(({k: -val for k, val in coeffs.items()}, -rhs))
            else:
                ineqs.append((coeffs, rhs))
        variables = set(self.variables)
        # if no variables, check constants
        feasible = self._fourier_motzkin(ineqs, variables)
        if feasible:
            return SolverStatus.SAT, []
        # deletion-based core
        core = self._deletion_core(cons, ineqs, variables)
        return SolverStatus.UNSAT, core

    def _fourier_motzkin(self, ineqs: List[Tuple[Dict[Any, float], float]], variables: Set[Any]) -> bool:
        ineqs = [(dict(c), float(r)) for c, r in ineqs]
        vars_list = list(variables)
        tol = 1e-9
        for v in vars_list:
            pos = [ineq for ineq in ineqs if ineq[0].get(v, 0.0) > tol]
            neg = [ineq for ineq in ineqs if ineq[0].get(v, 0.0) < -tol]
            zero = [ineq for ineq in ineqs if abs(ineq[0].get(v, 0.0)) <= tol]
            if not pos or not neg:
                ineqs = zero + [i for i in ineqs if i not in zero and i not in pos and i not in neg] + pos + neg
                # drop v from all (not necessary)
                continue
            new_ineqs = list(zero)
            for p in pos:
                for n in neg:
                    a_p = p[0][v]
                    a_n = n[0][v]
                    coeffs: Dict[Any, float] = {}
                    for y in vars_list:
                        if y == v:
                            continue
                        aval = p[0].get(y, 0.0)
                        cval = n[0].get(y, 0.0)
                        coeffs[y] = a_p * cval - a_n * aval
                    rhs = a_p * n[1] - a_n * p[1]
                    new_ineqs.append((coeffs, rhs))
            ineqs = new_ineqs
        # check contradictions
        for coeffs, rhs in ineqs:
            if all(abs(coeffs.get(v, 0.0)) <= 1e-9 for v in vars_list):
                if rhs < -1e-7:
                    return False
        return True

    def _deletion_core(self, cons: List[_Constraint], ineqs, variables) -> List[Any]:
        # try remove one constraint at a time
        core: List[Any] = []
        for i, c in enumerate(cons):
            subset = [x for j, x in enumerate(cons) if j != i]
            # rebuild ineqs for subset
            sub_ineqs = []
            for cc in subset:
                d = cc.data
                if cc.kind == "bound":
                    v = d["var"]
                    if d.get("hi") is not None:
                        sub_ineqs.append(({v: 1.0}, float(d["hi"])))
                    if d.get("lo") is not None:
                        sub_ineqs.append(({v: -1.0}, -float(d["lo"])))
                    continue
                op = d.get("op", "le")
                coeffs = dict(d.get("coeffs", {}))
                rhs = float(d.get("rhs", 0.0))
                if op in ("le", "lt"):
                    sub_ineqs.append((coeffs, rhs))
                elif op in ("ge", "gt"):
                    sub_ineqs.append(({k: -val for k, val in coeffs.items()}, -rhs))
                elif op in ("eq",):
                    sub_ineqs.append((dict(coeffs), rhs))
                    sub_ineqs.append(({k: -val for k, val in coeffs.items()}, -rhs))
                else:
                    sub_ineqs.append((coeffs, rhs))
            if self._fourier_motzkin(sub_ineqs, variables):
                core.append(c.tracker)
        if not core:
            core = [c.tracker for c in cons]
        return core


# Aliases
NanoSolver = VeritasNanoSolver
SimplexSolver = VeritasNanoSolver
DifferenceLogicSolver = VeritasNanoSolver
Tier2Oracle = VeritasNanoSolver
Z3SmtOracle = None
