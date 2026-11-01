"""Exact-rational linear/difference feasibility with bounded Fourier–Motzkin.

This is a reference decision procedure, not the dual-simplex engine in the roadmap.
All constraints are checked in one theory. Unsupported assertions raise; resource
exhaustion returns UNKNOWN. SAT includes a rational witness; UNSAT cores use correct
deletion minimization and are labelled non-minimal if the budget expires.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Any


class SolverStatus(StrEnum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


@dataclass
class _Constraint:
    tracker: Any
    coeffs: dict[Any, Fraction]
    rhs: Fraction
    strict: bool = False


@dataclass
class _Row:
    coeffs: dict[Any, Fraction]
    rhs: Fraction
    strict: bool
    sources: frozenset


class VeritasNanoSolver:
    def __init__(self, *, max_rows=10_000, max_variables=24, timeout_ms=1000):
        self.constraints: list[_Constraint] = []
        self.stack: list[int] = []
        self.variables = set()
        self.max_rows = int(max_rows)
        self.max_variables = int(max_variables)
        self.timeout_ms = timeout_ms
        self._last_core = []
        self._last_model = {}
        self._last_status = None
        self.core_is_minimal = False
        self.unknown_reason = ""

    def push(self):
        self.stack.append(len(self.constraints))
        self._invalidate()

    def pop(self, levels=1):
        if levels < 1 or levels > len(self.stack):
            raise IndexError("invalid solver pop depth")
        for _ in range(levels):
            self.constraints = self.constraints[:self.stack.pop()]
        self.variables = {v for c in self.constraints for v in c.coeffs}
        self._invalidate()

    def _invalidate(self):
        self._last_core, self._last_model, self._last_status = [], {}, None
        self.core_is_minimal = False

    @staticmethod
    def _rational(value):
        if isinstance(value, bool):
            raise ValueError("boolean is not a rational coefficient")
        if isinstance(value, Fraction):
            return value
        if isinstance(value, int):
            return Fraction(value)
        if isinstance(value, float) and math.isfinite(value):
            return Fraction(str(value))
        raise ValueError(f"non-finite/non-numeric rational {value!r}")

    def _coeffs(self, values):
        if not isinstance(values, dict):
            values = dict(values)
        out = {}
        for key, value in values.items():
            hash(key)
            number = self._rational(value)
            if number:
                out[key] = out.get(key, Fraction(0)) + number
        return {k: v for k, v in out.items() if v}

    def assert_constraint(self, constraint, tracker_id=None):
        self.assert_and_track(constraint, tracker_id)

    def assert_and_track(self, constraint, tracker_id=None):
        if tracker_id is None:
            tracker_id = len(self.constraints)
        hash(tracker_id)
        if isinstance(constraint, dict):
            kind = constraint.get("type", constraint.get("kind", "linear"))
            if kind in ("bound", "bounds"):
                constraint = ("bound", constraint["var"], constraint.get("lo"), constraint.get("hi"))
            elif kind == "diff":
                constraint = ("diff", constraint["i"], constraint["j"], constraint.get("c", 0))
            else:
                constraint = (constraint.get("op", "le"), constraint.get("coeffs", {}), constraint.get("rhs", 0))
        if not isinstance(constraint, (tuple, list)) or not constraint:
            raise ValueError(f"unsupported solver constraint {constraint!r}")
        op = constraint[0]
        if op in ("diff", "difference") and len(constraint) == 4:
            i, j, rhs = constraint[1], constraint[2], self._rational(constraint[3])
            hash(i)
            hash(j)
            coeffs = {i: Fraction(1), j: Fraction(-1)}
            coeffs = {k: v for k, v in coeffs.items() if v}
            strict = False
        elif op in ("bound", "bounds") and len(constraint) == 4:
            var, lo, hi = constraint[1:]
            hash(var)
            if lo is not None:
                self.assert_and_track(("ge", {var: 1}, lo), (tracker_id, "lo"))
            if hi is not None:
                self.assert_and_track(("le", {var: 1}, hi), (tracker_id, "hi"))
            return
        elif op in ("le", "lt", "ge", "gt", "eq") and len(constraint) == 3:
            coeffs, rhs = self._coeffs(constraint[1]), self._rational(constraint[2])
            if op in ("ge", "gt"):
                coeffs = {k: -v for k, v in coeffs.items()}
                rhs = -rhs
                op = "lt" if op == "gt" else "le"
            strict = op == "lt"
        else:
            raise ValueError(f"unsupported solver constraint {constraint!r}")
        self.constraints.append(_Constraint(tracker_id, coeffs, rhs, strict))
        self.variables.update(coeffs)
        self._invalidate()

    def check(self):
        return self.check_with_timeout(self.timeout_ms)

    def check_with_timeout(self, timeout_ms=None):
        started = time.monotonic()
        deadline = None if timeout_ms is None else started + max(0, timeout_ms) / 1000
        self._invalidate()
        rows = [self._normalize(c.coeffs, c.rhs, c.strict, frozenset({i})) for i, c in enumerate(self.constraints)]
        if len(self.variables) > self.max_variables:
            return self._unknown("variable budget exhausted")
        result = self._eliminate(rows, sorted(self.variables, key=repr), deadline)
        if result is None:
            return self._unknown("time or row budget exhausted")
        feasible, final_rows, stages = result
        if not feasible:
            indices = set().union(*(r.sources for r in final_rows if self._contradiction(r)))
            self._deletion_core(set(indices), deadline)
            self._last_status = SolverStatus.UNSAT
            return SolverStatus.UNSAT
        model = self._witness(stages)
        if model is None:
            return self._unknown("strict-bound witness reconstruction exhausted")
        self._last_model = model
        self._last_status = SolverStatus.SAT
        return SolverStatus.SAT

    def _unknown(self, reason):
        self.unknown_reason = reason
        self._last_status = SolverStatus.UNKNOWN
        return SolverStatus.UNKNOWN

    @staticmethod
    def _normalize(coeffs, rhs, strict, sources):
        coeffs = {k: v for k, v in coeffs.items() if v}
        if coeffs:
            first = sorted(coeffs, key=repr)[0]
            scale = abs(coeffs[first])
            coeffs = {k: v / scale for k, v in coeffs.items()}
            rhs /= scale
        return _Row(coeffs, rhs, strict and bool(coeffs), sources)

    @staticmethod
    def _contradiction(row):
        return not row.coeffs and (row.rhs < 0 or (row.strict and row.rhs == 0))

    def _eliminate(self, rows, variables, deadline):
        rows = list(rows)
        stages = []
        for var in variables:
            if deadline is not None and time.monotonic() >= deadline:
                return None
            pos = [r for r in rows if r.coeffs.get(var, 0) > 0]
            neg = [r for r in rows if r.coeffs.get(var, 0) < 0]
            zero = [r for r in rows if var not in r.coeffs]
            if any(self._contradiction(r) for r in zero):
                return False, zero, stages
            stages.append((var, rows))
            if len(pos) * len(neg) + len(zero) > self.max_rows:
                return None
            combined = []
            for p in pos:
                for n in neg:
                    a, b = p.coeffs[var], -n.coeffs[var]
                    coeffs = {k: a * n.coeffs.get(k, 0) + b * p.coeffs.get(k, 0)
                              for k in p.coeffs.keys() | n.coeffs.keys() if k != var}
                    combined.append(self._normalize(coeffs, a * n.rhs + b * p.rhs,
                                                    p.strict or n.strict, p.sources | n.sources))
            unique = {}
            for row in zero + combined:
                key = (tuple(sorted(((repr(k), k, v) for k, v in row.coeffs.items()))), row.rhs, row.strict)
                if key in unique:
                    old = unique[key]
                    row = _Row(row.coeffs, row.rhs, row.strict, old.sources | row.sources)
                unique[key] = row
            rows = list(unique.values())
        bad = [r for r in rows if self._contradiction(r)]
        return (not bad), rows, stages

    def _witness(self, stages):
        model = {}
        for var, rows in reversed(stages):
            lo = hi = None
            lo_strict = hi_strict = False
            for row in rows:
                coeff = row.coeffs.get(var)
                if not coeff:
                    continue
                tail = sum((c * model.get(k, Fraction(0)) for k, c in row.coeffs.items() if k != var), Fraction(0))
                bound = (row.rhs - tail) / coeff
                if coeff > 0:
                    if hi is None or bound < hi or (bound == hi and row.strict):
                        hi, hi_strict = bound, row.strict
                else:
                    if lo is None or bound > lo or (bound == lo and row.strict):
                        lo, lo_strict = bound, row.strict
            if lo is not None and hi is not None and (lo > hi or (lo == hi and (lo_strict or hi_strict))):
                return None
            if lo is not None and hi is not None:
                value = (lo + hi) / 2
            elif lo is not None:
                value = lo + 1 if lo_strict else lo
            elif hi is not None:
                value = hi - 1 if hi_strict else hi
            else:
                value = Fraction(0)
            model[var] = value
        return model

    def _deletion_core(self, indices, deadline):
        all_indices = set(range(len(self.constraints)))
        def unsat(subset):
            rows = [self._normalize(c.coeffs, c.rhs, c.strict, frozenset({i})) for i, c in enumerate(self.constraints) if i in subset]
            result = self._eliminate(rows, sorted(self.variables, key=repr), deadline)
            return None if result is None else not result[0]
        core = set(indices) or all_indices
        # Verify the seed; provenance can be broader than a conflict, never narrower.
        if unsat(core) is not True:
            core = all_indices
        for index in sorted(core.copy(), key=lambda i: repr(self.constraints[i].tracker)):
            if deadline is not None and time.monotonic() >= deadline:
                self.core_is_minimal = False
                break
            candidate = core - {index}
            result = unsat(candidate)
            if result is True:
                core = candidate
            elif result is None:
                self.core_is_minimal = False
                break
        else:
            self.core_is_minimal = True
        self._last_core = list(dict.fromkeys(self.constraints[i].tracker for i in sorted(core)))
        return self._last_core

    def get_unsat_core(self):
        return list(self._last_core) if self._last_status == SolverStatus.UNSAT else []

    def get_model(self):
        return dict(self._last_model) if self._last_status == SolverStatus.SAT else {}

    def get_float_model(self):
        return {k: float(v) for k, v in self.get_model().items()}


NanoSolver = VeritasNanoSolver
SimplexSolver = VeritasNanoSolver
DifferenceLogicSolver = VeritasNanoSolver
Tier2Oracle = VeritasNanoSolver