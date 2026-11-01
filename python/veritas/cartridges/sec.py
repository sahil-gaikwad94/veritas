"""``veritas.sec`` — agentic IAM, tool safety and finite-trace guardrails.

A :class:`WorldState` is a symbolic record of facts, tool declarations, budgets and an
event trace.  Policy formulas are LTLf expressions from :mod:`veritas.ltlf`, so policies
such as "KYC before any transfer" and "at most three trades" are checked with real
finite-trace semantics rather than ad-hoc ordering checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..ltlf import (
    Formula,
    Trace,
)
from ..ltlf import (
    always as always_formula,
)
from ..ltlf import (
    at_most as at_most_formula,
)
from ..ltlf import (
    before as before_formula,
)
from ..ltlf import (
    never as never_formula,
)
from ..ltlf import (
    responds as responds_formula,
)


class PolicyViolation(RuntimeError):
    """Raised when a tool call or a trace violates a declared policy."""

    def __init__(self, message: str, *, policy: str | None = None, tool: str | None = None,
                 details: Mapping[str, Any] | None = None):
        self.policy = policy
        self.tool = tool
        self.details = dict(details or {})
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        return {"message": str(self), "policy": self.policy, "tool": self.tool, "details": self.details}


@dataclass
class ToolSpec:
    name: str
    requires: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    cost: float = 0.0
    budget_key: str | None = None
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
        self.tools: dict[str, ToolSpec] = {}
        self.policies: list[Policy] = []
        self.budgets: dict[str, float] = {}
        self.spent: dict[str, float] = {}
        self.history: list[Any] = []  # backwards-compatible ("assert", fact) log
        self.violations: list[PolicyViolation] = []

    # -- declaration -----------------------------------------------------
    def declare_tool(
        self,
        name: str,
        requires: Sequence[str] | None = None,
        effects: Sequence[str] | None = None,
        *,
        cost: float = 0.0,
        budget_key: str | None = None,
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

    def remaining(self, key: str) -> float | None:
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

    def can_call(self, tool: str, preconditions: Sequence[str] | None = None) -> bool:
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
        """Atomically check preconditions/budgets/policies, then commit the call."""
        self.before_tool_call(tool, data)
        snapshot = self.snapshot()
        try:
            event = self.trace.append(tool, dict(data))
            self.history.append(("call", tool))
            spec = self.tools.get(tool)
            if spec is not None:
                for effect in spec.effects:
                    self.facts.add(effect)
                if spec.budget_key is not None and spec.cost:
                    if not self.spend(spec.budget_key, spec.cost):
                        raise PolicyViolation(f"budget {spec.budget_key!r} exceeded", tool=tool)
            self.enforce()
            return event
        except Exception:
            self.restore(snapshot)
            raise

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

    def before_tool_call(self, tool: str, kwargs: Mapping[str, Any] | None = None) -> None:
        spec = self.tools.get(tool)
        if spec is not None and not self.can_call(tool):
            missing = [fact for fact in spec.requires if not self.has(fact)]
            remaining = self.remaining(spec.budget_key) if spec.budget_key else None
            raise PolicyViolation(
                f"tool {tool!r} precondition/budget not satisfied; missing facts: {missing}",
                tool=tool,
                details={"missing": missing, "remaining": remaining},
            )
        # Prospective temporal-policy check: would this call violate a policy?
        snapshot = self.snapshot()
        try:
            self.trace.append(tool, dict(kwargs or {}))
            violations = [v for v in self.validate() if v.policy is not None]
            if violations:
                raise violations[0]
        finally:
            self.restore(snapshot)

    @contextmanager
    def transaction(self):
        snapshot = self.snapshot()
        try:
            yield self
        except Exception:
            self.restore(snapshot)
            raise

    # -- temporal policies ----------------------------------------------
    def add_policy(self, formula: Any, name: str | None = None, severity: str = "error") -> Policy:
        if not isinstance(formula, Formula):
            formula = always_formula(formula)
        policy = Policy(name=name or formula.label, formula=formula, severity=severity)
        self.policies.append(policy)
        return policy

    def validate(self, *, errors_only: bool = False) -> list[PolicyViolation]:
        found: list[PolicyViolation] = []
        for policy in self.policies:
            if errors_only and policy.severity != "error":
                continue
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
        violations = self.validate(errors_only=True)
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

    def trace_names(self) -> list[str]:
        return self.trace.names()

    # -- snapshots -------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "facts": set(self.facts),
            "spent": dict(self.spent),
            "events": list(self.trace.events),
            "history": list(self.history),
            "violations": list(self.violations),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.facts = set(snapshot.get("facts") or [])
        self.spent = dict(snapshot.get("spent") or {})
        self.trace = Trace(list(snapshot.get("events") or []))
        self.history = list(snapshot.get("history") or [])
        self.violations = list(snapshot.get("violations") or [])

    def as_dict(self) -> dict[str, Any]:
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
