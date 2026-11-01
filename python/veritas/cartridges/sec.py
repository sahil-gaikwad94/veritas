"""``veritas.sec`` — agentic IAM, tool safety and finite-trace guardrails.

A :class:`WorldState` is a symbolic record of facts, tool declarations, budgets and an
event trace.  Policy formulas are LTLf expressions from :mod:`veritas.ltlf`, so policies
such as "KYC before any transfer" and "at most three trades" are checked with real
finite-trace semantics rather than ad-hoc ordering checks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..ltlf import (
    Formula,
    Trace,
    always as always_formula,
    at_most as at_most_formula,
    before as before_formula,
    never as never_formula,
    responds as responds_formula,
)


class PolicyViolation(RuntimeError):
    """Raised when a tool call or a trace violates a declared policy."""

    def __init__(self, message: str, *, policy: Optional[str] = None, tool: Optional[str] = None,
                 details: Optional[Mapping[str, Any]] = None):
        self.policy = policy
        self.tool = tool
        self.details = dict(details or {})
        super().__init__(message)

    def as_dict(self) -> Dict[str, Any]:
        return {"message": str(self), "policy": self.policy, "tool": self.tool, "details": self.details}


@dataclass
class ToolSpec:
    name: str
    requires: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    cost: float = 0.0
    budget_key: Optional[str] = None
    description: str = ""
    mutating: bool = False


@dataclass
class Policy:
    name: str
    formula: Formula
    severity: str = "error"


class WorldState:
    """Incremental world state for multi-turn agents and tool guards."""

    def __init__(self):
        self.trace = Trace()
        self.facts: set = set()
        self.tools: Dict[str, ToolSpec] = {}
        self.policies: List[Policy] = []
        self.budgets: Dict[str, float] = {}
        self.spent: Dict[str, float] = {}
        self.history: List[Any] = []  # backwards-compatible ("assert", fact) log
        self.violations: List[PolicyViolation] = []

    # -- declaration -----------------------------------------------------
    def declare_tool(
        self,
        name: str,
        requires: Optional[Sequence[str]] = None,
        effects: Optional[Sequence[str]] = None,
        *,
        cost: float = 0.0,
        budget_key: Optional[str] = None,
        description: str = "",
        mutating: bool = False,
    ) -> ToolSpec:
        tool = ToolSpec(
            name=name,
            requires=list(requires or []),
            effects=list(effects or []),
            cost=float(cost),
            budget_key=budget_key,
            description=description,
            mutating=mutating,
        )
        self.tools[name] = tool
        return tool

    def set_budget(self, key: str, limit: float) -> None:
        self.budgets[key] = float(limit)

    def remaining(self, key: str) -> Optional[float]:
        if key not in self.budgets:
            return None
        return self.budgets[key] - self.spent.get(key, 0.0)

    def spend(self, key: str, amount: float) -> bool:
        if key in self.budgets and self.spent.get(key, 0.0) + amount > self.budgets[key] + 1e-12:
            return False
        self.spent[key] = self.spent.get(key, 0.0) + float(amount)
        return True

    # -- facts / history -------------------------------------------------
    def assert_fact(self, fact: str) -> None:
        self.facts.add(fact)
        self.history.append(("assert", fact))
        self.trace.append(str(fact))

    def revoke_fact(self, fact: str) -> None:
        self.facts.discard(fact)
        self.history.append(("revoke", fact))

    def has(self, fact: str) -> bool:
        return fact in self.facts

    def can_call(self, tool: str, preconditions: Optional[Sequence[str]] = None) -> bool:
        spec = self.tools.get(tool)
        required = list(preconditions) if preconditions is not None else (
            list(spec.requires) if spec is not None else []
        )
        if not all(self.has(fact) for fact in required):
            return False
        if spec is not None and spec.budget_key is not None:
            remaining = self.remaining(spec.budget_key)
            if remaining is not None and spec.cost > remaining + 1e-12:
                return False
        return True

    def record_call(self, tool: str, **data: Any) -> Any:
        spec = self.tools.get(tool)
        event = self.trace.append(tool, dict(data))
        self.history.append(("call", tool))
        if spec is not None:
            for effect in spec.effects:
                self.facts.add(effect)
            if spec.budget_key is not None and spec.cost:
                self.spent[spec.budget_key] = self.spent.get(spec.budget_key, 0.0) + spec.cost
        return event

    def call(self, tool: str, **data: Any) -> Any:
        spec = self.tools.get(tool)
        if spec is not None and not self.can_call(tool):
            missing = [fact for fact in spec.requires if not self.has(fact)]
            raise PolicyViolation(
                f"tool {tool!r} precondition not satisfied; missing facts: {missing}",
                tool=tool,
                details={"missing": missing, "facts": sorted(self.facts)},
            )
        return self.record_call(tool, **data)

    def before_tool_call(self, tool: str, kwargs: Optional[Mapping[str, Any]] = None) -> None:
        spec = self.tools.get(tool)
        if spec is not None and not self.can_call(tool):
            missing = [fact for fact in spec.requires if not self.has(fact)]
            raise PolicyViolation(
                f"tool {tool!r} precondition not satisfied; missing facts: {missing}",
                tool=tool,
                details={"missing": missing},
            )

    # -- temporal policies ----------------------------------------------
    def add_policy(self, formula: Any, name: Optional[str] = None, severity: str = "error") -> Policy:
        if not isinstance(formula, Formula):
            formula = always_formula(formula)
        policy = Policy(name=name or formula.label, formula=formula, severity=severity)
        self.policies.append(policy)
        return policy

    def validate(self) -> List[PolicyViolation]:
        found: List[PolicyViolation] = []
        for policy in self.policies:
            try:
                ok = policy.formula.evaluate(self.trace)
            except Exception as exc:  # pragma: no cover - defensive
                ok = False
                found.append(PolicyViolation(f"policy {policy.name!r} could not be evaluated: {exc}",
                                             policy=policy.name))
                continue
            if not ok:
                found.append(PolicyViolation(
                    f"policy {policy.name!r} violated: {policy.formula.label}",
                    policy=policy.name,
                    details={"trace": self.trace.names()},
                ))
        return found

    def enforce(self) -> None:
        violations = self.validate()
        if violations:
            self.violations.extend(violations)
            raise violations[0]

    # -- compatibility ---------------------------------------------------
    def before(self, earlier: str, later: str) -> bool:
        earlier_positions = self.trace.positions(earlier)
        for index in self.trace.positions(later):
            if not any(j < index for j in earlier_positions):
                return False
        return True

    def always(self, condition: bool) -> bool:
        return bool(condition)

    def trace_names(self) -> List[str]:
        return self.trace.names()

    # -- snapshots -------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "facts": set(self.facts),
            "spent": dict(self.spent),
            "events": list(self.trace.events),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.facts = set(snapshot.get("facts") or [])
        self.spent = dict(snapshot.get("spent") or {})
        self.trace = Trace(list(snapshot.get("events") or []))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "facts": sorted(self.facts),
            "trace": self.trace.as_dict(),
            "budgets": dict(self.budgets),
            "spent": dict(self.spent),
            "policies": [p.name for p in self.policies],
        }


# convenience policy constructors re-exported for the cartridge surface
def kyc_before(tool: str = "transfer", kyc: str = "verify_kyc") -> Formula:
    return before_formula(kyc, tool)


def budget_never_exceeded() -> Formula:
    return never_formula("budget_violation")


def no_double_spend(limit: int = 1, tool: str = "transfer") -> Formula:
    return at_most_formula(tool, limit)


def require_response(trigger: str, response: str) -> Formula:
    return responds_formula(trigger, response)


__all__ = [
    "Policy",
    "PolicyViolation",
    "ToolSpec",
    "WorldState",
    "budget_never_exceeded",
    "kyc_before",
    "no_double_spend",
    "require_response",
]
