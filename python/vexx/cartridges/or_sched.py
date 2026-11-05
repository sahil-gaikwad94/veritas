"""``vexx.or_sched`` — scheduling, precedence and capacity verification.

Provides both declarative checks (``validate``) and a constructive earliest-start solver
with cycle detection.  This is the executable core behind ``vexx.or``-style schemas:
precedence is difference logic, resource usage is an interval-capacity constraint, and a
schedule either satisfies every constraint or reports exactly which one failed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


class SchedulingError(ValueError):
    """Raised for malformed scheduling problems (unknown task, cycle, ...)."""


@dataclass
class Task:
    name: str
    duration: float = 1.0
    predecessors: list[str] = field(default_factory=list)
    resource: str | None = None
    demand: float = 1.0
    release: float = 0.0
    deadline: float | None = None

    @property
    def end(self) -> float:
        return self.duration


@dataclass
class Violation:
    kind: str
    message: str
    tasks: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "message": self.message, "tasks": list(self.tasks)}


class SchedulingProblem:
    def __init__(self, horizon: float | None = None):
        self.tasks: dict[str, Task] = {}
        self.capacities: dict[str, float] = {}
        self.horizon = horizon

    # -- construction ----------------------------------------------------
    def add_task(
        self,
        name: str,
        duration: float = 1.0,
        predecessors: Sequence[str] | None = None,
        *,
        resource: str | None = None,
        demand: float = 1.0,
        release: float = 0.0,
        deadline: float | None = None,
    ) -> Task:
        if duration < 0:
            raise SchedulingError(f"task {name!r} has negative duration")
        task = Task(name, float(duration), list(predecessors or []), resource, float(demand), float(release), deadline)
        self.tasks[name] = task
        return task

    def set_capacity(self, resource: str, capacity: float) -> None:
        if capacity < 0:
            raise SchedulingError("capacity must be non-negative")
        self.capacities[resource] = float(capacity)

    def add_precedence(self, before: str, after: str) -> None:
        for name in (before, after):
            if name not in self.tasks:
                raise SchedulingError(f"unknown task {name!r}")
        self.tasks[after].predecessors.append(before)

    # -- analysis --------------------------------------------------------
    def topological_order(self) -> list[str]:
        state: dict[str, int] = {}
        order: list[str] = []

        def visit(name: str, trail: list[str]) -> None:
            mark = state.get(name, 0)
            if mark == 1:
                raise SchedulingError("precedence cycle: " + " -> ".join(trail + [name]))
            if mark == 2:
                return
            state[name] = 1
            for predecessor in self.tasks[name].predecessors:
                if predecessor not in self.tasks:
                    raise SchedulingError(f"task {name!r} references unknown predecessor {predecessor!r}")
                visit(predecessor, trail + [name])
            state[name] = 2
            order.append(name)

        for name in self.tasks:
            visit(name, [])
        return order

    def earliest_schedule(self) -> dict[str, float]:
        starts: dict[str, float] = {}
        for name in self.topological_order():
            task = self.tasks[name]
            ready = task.release
            for predecessor in task.predecessors:
                ready = max(ready, starts[predecessor] + self.tasks[predecessor].duration)
            starts[name] = ready
        return starts

    def makespan(self) -> float:
        starts = self.earliest_schedule()
        return max((starts[name] + task.duration for name, task in self.tasks.items()), default=0.0)

    def critical_path(self) -> tuple[float, list[str]]:
        order = self.topological_order()
        best: dict[str, tuple[float, list[str]]] = {}
        for name in order:
            task = self.tasks[name]
            candidates = [(task.release, [])]
            for predecessor in task.predecessors:
                candidates.append(best[predecessor])
            ready, path = max(candidates, key=lambda item: item[0])
            best[name] = (ready + task.duration, path + [name])
        if not best:
            return 0.0, []
        return max(best.values(), key=lambda item: item[0])

    # -- verification ----------------------------------------------------
    def validate(self, starts: Mapping[str, float]) -> list[Violation]:
        violations: list[Violation] = []
        missing = set(self.tasks) - set(starts)
        if missing:
            violations.append(Violation("missing", f"no start time for {sorted(missing)}", sorted(missing)))
        for name, task in self.tasks.items():
            if name not in starts:
                continue
            start = float(starts[name])
            if start < -1e-9:
                violations.append(Violation("negative", f"{name} starts before 0", [name]))
            if start < task.release - 1e-9:
                violations.append(Violation("release", f"{name} starts before release {task.release}", [name]))
            if task.deadline is not None and start + task.duration > task.deadline + 1e-9:
                violations.append(Violation("deadline", f"{name} ends after deadline {task.deadline}", [name]))
            for predecessor in task.predecessors:
                if predecessor not in starts:
                    continue
                end = float(starts[predecessor]) + self.tasks[predecessor].duration
                if start + 1e-9 < end:
                    violations.append(
                        Violation("precedence", f"{name} starts before {predecessor} ends", [predecessor, name])
                    )
        if self.horizon is not None:
            for name, task in self.tasks.items():
                if name in starts and float(starts[name]) + task.duration > self.horizon + 1e-9:
                    violations.append(Violation("horizon", f"{name} exceeds horizon {self.horizon}", [name]))
        violations.extend(self._resource_violations(starts))
        return violations

    def _resource_violations(self, starts: Mapping[str, float]) -> list[Violation]:
        violations: list[Violation] = []
        by_resource: dict[str, list[tuple[float, float, str, float]]] = {}
        for name, task in self.tasks.items():
            if task.resource is None or name not in starts:
                continue
            by_resource.setdefault(task.resource, []).append(
                (float(starts[name]), float(starts[name]) + task.duration, name, task.demand)
            )
        for resource, intervals in by_resource.items():
            capacity = self.capacities.get(resource, float("inf"))
            points: list[tuple[float, int, float]] = []
            for start, end, _name, demand in intervals:
                points.append((start, 1, demand))
                points.append((end, -1, demand))
            points.sort(key=lambda item: (item[0], item[1]))
            load = 0.0
            for _time, delta, demand in points:
                load += delta * demand
                if load > capacity + 1e-9:
                    violations.append(
                        Violation(
                            "capacity",
                            f"resource {resource!r} load {load:g} exceeds capacity {capacity:g}",
                            [name for _s, _e, name, _d in intervals],
                        )
                    )
                    break
        return violations

    def feasible(self) -> bool:
        try:
            return not self.validate(self.earliest_schedule())
        except SchedulingError:
            return False

    def solve(self) -> dict[str, float]:
        starts = self.earliest_schedule()
        violations = self.validate(starts)
        if violations:
            raise SchedulingError("; ".join(v.message for v in violations))
        return starts


# ---------------------------------------------------------------------------
# stateless helpers kept for compatibility
# ---------------------------------------------------------------------------

def no_overlap(intervals: Sequence[tuple[float, float]]) -> bool:
    ordered = sorted(intervals)
    for i in range(1, len(ordered)):
        if ordered[i][0] < ordered[i - 1][1] - 1e-12:
            return False
    return True


def precedence_ok(starts: Mapping[str, float], durations: Mapping[str, float], preds: Mapping[str, Sequence[str]]) -> bool:
    for task, predecessors in preds.items():
        for predecessor in predecessors:
            if starts.get(task, 0.0) + 1e-9 < starts.get(predecessor, 0.0) + durations.get(predecessor, 0.0):
                return False
    return True


def capacity_ok(loads: Sequence[float], capacity: float) -> bool:
    return all(load <= capacity + 1e-9 for load in loads)


__all__ = [
    "SchedulingError",
    "SchedulingProblem",
    "Task",
    "Violation",
    "capacity_ok",
    "no_overlap",
    "precedence_ok",
]
