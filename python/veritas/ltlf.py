"""Linear Temporal Logic over finite traces (LTLf), evaluated on agent traces.

Used by :mod:`veritas.cartridges.sec` to express tool-call safety policies such as
"authentication must precede a transfer" or "a budget is never exceeded".  Formulas are
small first-class objects with explicit finite-trace semantics — no dependency on an
external temporal model checker.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    name: str
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - presentation
        return f"{self.name}@{self.step}"


class Trace:
    """An append-only finite trace of named events."""

    def __init__(self, events: Iterable[Event] | None = None):
        self.events: list[Event] = list(events or [])
        self._step = max((e.step for e in self.events), default=0)

    def append(self, name: str, data: dict[str, Any] | None = None) -> Event:
        self._step += 1
        event = Event(name=name, step=self._step, data=dict(data or {}))
        self.events.append(event)
        return event

    def names(self) -> list[str]:
        return [event.name for event in self.events]

    def positions(self, name: str) -> list[int]:
        return [index for index, event in enumerate(self.events) if event.name == name]

    def occurred(self, name: str) -> bool:
        return any(event.name == name for event in self.events)

    def count(self, name: str) -> int:
        return sum(1 for event in self.events if event.name == name)

    def first(self, name: str) -> Event | None:
        for event in self.events:
            if event.name == name:
                return event
        return None

    def last(self, name: str) -> Event | None:
        for event in reversed(self.events):
            if event.name == name:
                return event
        return None

    def prefix(self, length: int) -> Trace:
        return Trace(self.events[: max(0, length)])

    def suffix(self, start: int) -> Trace:
        return Trace(self.events[max(0, start):])

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def as_dict(self) -> dict[str, Any]:
        return {"events": [{"name": e.name, "step": e.step, "data": e.data} for e in self.events]}


def _trace_of(target: Any) -> Trace:
    if isinstance(target, Trace):
        return target
    trace = getattr(target, "trace", None)
    if isinstance(trace, Trace):
        return trace
    events = getattr(target, "events", None)
    if isinstance(events, list):
        return Trace([e if isinstance(e, Event) else Event(str(e)) for e in events])
    raise TypeError(f"expected a Trace or an object with .trace, got {type(target).__name__}")


class Formula:
    """Base class: a predicate over a finite trace."""

    label: str = "formula"

    def evaluate(self, target: Any) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, target: Any) -> bool:
        return self.evaluate(target)

    def __and__(self, other: Formula) -> Formula:
        return And(self, other)

    def __or__(self, other: Formula) -> Formula:
        return Or(self, other)

    def __invert__(self) -> Formula:
        return Not(self)

    def __repr__(self) -> str:
        return self.label


class Atom(Formula):
    def __init__(self, predicate: Callable[[Trace], bool], label: str = "atom"):
        self.predicate = predicate
        self.label = label

    def evaluate(self, target: Any) -> bool:
        return bool(self.predicate(_trace_of(target)))


class Occurred(Formula):
    def __init__(self, name: str):
        self.name = name
        self.label = f"occurred({name})"

    def evaluate(self, target: Any) -> bool:
        return _trace_of(target).occurred(self.name)


class Never(Formula):
    def __init__(self, name: str):
        self.name = name
        self.label = f"never({name})"

    def evaluate(self, target: Any) -> bool:
        return not _trace_of(target).occurred(self.name)


class Always(Formula):
    def __init__(self, inner: Formula):
        self.inner = inner
        self.label = f"always({inner.label})"

    def evaluate(self, target: Any) -> bool:
        trace = _trace_of(target)
        # LTLf: G φ holds at every position of the (non-empty) trace suffix.
        if not trace.events:
            return True
        return all(self.inner.evaluate(trace.suffix(i)) for i in range(len(trace)))


class Eventually(Formula):
    def __init__(self, inner: Formula):
        self.inner = inner
        self.label = f"eventually({inner.label})"

    def evaluate(self, target: Any) -> bool:
        trace = _trace_of(target)
        if not trace.events:
            return False
        return any(self.inner.evaluate(trace.suffix(i)) for i in range(len(trace)))


class Before(Formula):
    """If ``later`` occurs at step *i*, ``earlier`` must occur at some step < i."""

    def __init__(self, earlier: str, later: str):
        self.earlier = earlier
        self.later = later
        self.label = f"before({earlier}, {later})"

    def evaluate(self, target: Any) -> bool:
        trace = _trace_of(target)
        earlier_positions = trace.positions(self.earlier)
        for index in trace.positions(self.later):
            if not any(j < index for j in earlier_positions):
                return False
        return True


class Responds(Formula):
    """Every ``trigger`` is eventually followed by ``response`` (strict future)."""

    def __init__(self, trigger: str, response: str):
        self.trigger = trigger
        self.response = response
        self.label = f"responds({trigger} -> {response})"

    def evaluate(self, target: Any) -> bool:
        trace = _trace_of(target)
        response_positions = trace.positions(self.response)
        for index in trace.positions(self.trigger):
            if not any(j > index for j in response_positions):
                return False
        return True


class AtMost(Formula):
    def __init__(self, name: str, limit: int):
        self.name = name
        self.limit = int(limit)
        self.label = f"at_most({name}, {limit})"

    def evaluate(self, target: Any) -> bool:
        return _trace_of(target).count(self.name) <= self.limit


class And(Formula):
    def __init__(self, *parts: Formula):
        self.parts = [p for p in parts if p is not None]
        self.label = "and(" + ", ".join(p.label for p in self.parts) + ")"

    def evaluate(self, target: Any) -> bool:
        return all(part.evaluate(target) for part in self.parts)


class Or(Formula):
    def __init__(self, *parts: Formula):
        self.parts = [p for p in parts if p is not None]
        self.label = "or(" + ", ".join(p.label for p in self.parts) + ")"

    def evaluate(self, target: Any) -> bool:
        return any(part.evaluate(target) for part in self.parts)


class Not(Formula):
    def __init__(self, inner: Formula):
        self.inner = inner
        self.label = f"not({inner.label})"

    def evaluate(self, target: Any) -> bool:
        return not self.inner.evaluate(target)


# ---------------------------------------------------------------------------
# constructor functions (the public ``vt.always`` / ``vt.before`` surface)
# ---------------------------------------------------------------------------

def _as_formula(value: Any, label: str = "predicate") -> Formula:
    if isinstance(value, Formula):
        return value
    if callable(value):
        return Atom(value, label=getattr(value, "__name__", label))
    if isinstance(value, str):
        return Occurred(value)
    raise TypeError(f"cannot interpret {value!r} as an LTLf formula")


def always(predicate: Any) -> Formula:
    return Always(_as_formula(predicate, "always"))


def eventually(predicate: Any) -> Formula:
    return Eventually(_as_formula(predicate, "eventually"))


def implies(first: Any, second: Any) -> Formula:
    """``first -> second`` as a finite-trace formula (``false -> q`` is vacuously true)."""

    return Or(Not(_as_formula(first, "implies-lhs")), _as_formula(second, "implies-rhs"))


def negate(formula: Any) -> Formula:
    return Not(_as_formula(formula, "not"))


def conjunction(*formulas: Any) -> Formula:
    parts = [_as_formula(item, "and") for item in formulas]
    return And(*parts)


def never(name: Any) -> Formula:
    if isinstance(name, str):
        return Never(name)
    return Not(_as_formula(name))


def before(earlier: str, later: str) -> Formula:
    return Before(earlier, later)


def responds(trigger: str, response: str) -> Formula:
    return Responds(trigger, response)


def at_most(name: str, limit: int) -> Formula:
    return AtMost(name, limit)


def occurred(name: str) -> Formula:
    return Occurred(name)


once = occurred  # alias: ``_as_formula`` accepts any name string the same way


def evaluate(formula: Any, target: Any) -> bool:
    return _as_formula(formula).evaluate(target)


__all__ = [
    "Always",
    "And",
    "AtMost",
    "Atom",
    "Before",
    "Event",
    "Eventually",
    "Formula",
    "Never",
    "Not",
    "Occurred",
    "Or",
    "Responds",
    "Trace",
    "always",
    "at_most",
    "before",
    "conjunction",
    "evaluate",
    "eventually",
    "implies",
    "negate",
    "never",
    "occurred",
    "once",
    "responds",
]
