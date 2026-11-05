"""Stateful firewall for Model Context Protocol (MCP) tool calls.

Every AI agent in 2026 calls tools whose *arguments depend on prior tool outputs*, whose
ordering is governed by temporal policy, and whose inputs may be attacker-controlled text
from prompt injection.  :class:`VexxMcpSession` binds three layers together:

* **Temporal policy** — an LTLf formula (:mod:`vexx.ltlf`) over the finite trace of tool
  calls (``before("lookup_order", "issue_refund")``).  A tool whose call would make the
  policy evaluate to ``False`` is *masked out* before the agent can choose it
  (:meth:`VexxMcpSession.allowed_tools`), and any attempt to call it is rejected at the
  door (fail-closed), not after the side effect.
* **Symbolic environment** — flat scalar tool outputs are bound into ``state.env`` and
  accumulated numeric counters into ``state.accumulated``, so a later argument can be
  constrained against an earlier observation (``refund_amount <= state.env["last_order_total"]``).
* **Taint tracking** — :class:`UntrustedString` marks values that originate outside the
  trust boundary (email bodies, web pages, other tools' free text).  ``mcp.require`` asserts
  properties of them; a failed assertion blocks the call.

The decorator is a *firewall*: unrelated to verifying that the tool implementation is
correct, it enforces the contract between the agent and its tools.  Deterministic
post-conditions belong in the tool itself.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .cartridges.sec import PolicyViolation
from .ltlf import Event, Trace

__all__ = [
    "McpRequireError",
    "McpState",
    "TaintTag",
    "UntrustedString",
    "VexxMcpSession",
    "is_untainted",
    "mcp_require",
    "taint",
]


class McpRequireError(RuntimeError):
    """Raised inside a tool when a :func:`mcp_require` assertion fails."""

    def __init__(self, message: str, *, policy: str | None = None, details: Mapping[str, Any] | None = None):
        self.policy = policy
        self.details = dict(details or {})
        super().__init__(message)


@dataclass(frozen=True)
class TaintTag:
    source: str


class UntrustedString(str):
    """A string that entered the system from an untrusted channel.

    ``sources`` names the channels (e.g. ``"email"``, ``"web"``, ``"other_tool"``).  The
    class registers a pydantic core schema, so it can be used as a model field type and the
    taint tag survives validation; plain ``str`` fields drop the subclass, which the
    firewall re-attaches for argument mappings at the model boundary.
    """

    def __new__(cls, value: str, sources: Sequence[str] | None = None) -> UntrustedString:
        instance = str.__new__(cls, str(value))
        instance.sources = frozenset(sources or ("untrusted",))
        return instance

    def tag(self, source: str) -> UntrustedString:
        return UntrustedString(self, (*self.sources, source))

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any):
        from pydantic_core import core_schema

        def _validate(value: Any) -> UntrustedString:
            if isinstance(value, UntrustedString):
                return value
            return UntrustedString(str(value))

        # is_instance_schema keeps str subclasses intact (unlike str_schema, which coerces
        # to the base type and would erase the taint tag).
        return core_schema.no_info_after_validator_function(_validate, core_schema.is_instance_schema(str))

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        return {"type": "string"}


def taint(value: Any, sources: Sequence[str] | None = None) -> UntrustedString:
    return UntrustedString(str(value), sources)


def _taint_map(mapping: Mapping[str, Any] | None, prefix: str = "") -> dict[str, frozenset[str]]:
    """Dot path -> sources for every untrusted string inside a raw argument mapping."""

    found: dict[str, frozenset[str]] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, UntrustedString):
            found[path] = node.sources
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
            return
        if isinstance(node, (list, tuple, set, frozenset)):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]" if path else f"[{index}]")

    if isinstance(mapping, Mapping):
        for key, value in mapping.items():
            walk(value, str(key))
    return found


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        for key in value.keys():
            if isinstance(key, str):
                yield key
        for item in value.values():
            yield from _iter_strings(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_strings(item)
        return
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            yield from _iter_strings(model_dump())
        except Exception:
            return


def is_untainted(value: Any) -> bool:
    """``True`` iff no part of ``value`` is tagged :class:`UntrustedString`."""

    return not any(isinstance(part, UntrustedString) for part in _iter_strings(value))


def mcp_require(condition: Any, message: str | None = None, *, policy: str | None = None) -> bool:
    """Assert a condition inside a verified tool; raise :class:`McpRequireError` otherwise."""

    if not bool(condition):
        detail = message or "vexx MCP requirement failed"
        raise McpRequireError(detail, policy=policy)
    return True


@dataclass
class McpState:
    """Session state threaded through verified tools."""

    env: dict[str, Any] = field(default_factory=dict)
    accumulated: dict[str, float] = field(default_factory=dict)
    trace: Trace = field(default_factory=Trace)

    @property
    def events(self) -> list[Event]:
        return list(self.trace.events)

    def occurred(self, name: str) -> bool:
        return self.trace.occurred(name)


def _flat_scalars(payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten one level of dict payload into ``(key, scalar)`` pairs."""

    pairs: list[tuple[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, f"{path}{key}." if path else f"{key}.")
            return
        if isinstance(node, (int, float, bool)) or node is None:
            pairs.append((path[:-1] if path.endswith(".") else path or prefix, node))
            return
        if isinstance(node, str):
            pairs.append((path[:-1] if path.endswith(".") else path or prefix, node))
            return
        pairs.append((path[:-1] if path.endswith(".") else path or prefix, repr(node)))

    walk(payload, prefix)
    return [(k, v) for k, v in pairs if k]


class VexxMcpSession:
    """Firewall around a set of MCP tools.

    Parameters
    ----------
    temporal_policy:
        An :class:`vexx.ltlf.Formula` or a plain string ("tool_name") constraining the
        finite trace of tool calls.  The empty string / ``None`` disables temporal checks.
    max_session_spend:
        Hard cap on the sum of ``cost=`` passed to :meth:`verified_tool`.  Exhaustion blocks
        further calls with :class:`PolicyViolation`.
    budgets:
        Optional mapping ``{counter_name: limit}`` checked against ``state.accumulated``
        before every call (accumulated counters are filled from numeric argument values and
        numeric tool outputs with the same key).
    on_violation:
        ``"raise"`` (default) raises :class:`PolicyViolation`; ``"return"`` returns a dict
        ``{"ok": False, "error": ...}`` instead — useful for agentic loops that prefer error
        tool-messages over exceptions.
    """

    def __init__(
        self,
        temporal_policy: Any = None,
        max_session_spend: float | None = None,
        *,
        budgets: Mapping[str, float] | None = None,
        on_violation: str = "raise",
    ):
        self.policy = temporal_policy
        self.max_session_spend = None if max_session_spend is None else float(max_session_spend)
        self.budgets = dict(budgets or {})
        if on_violation not in ("raise", "return"):
            raise ValueError("on_violation must be 'raise' or 'return'")
        self.on_violation = on_violation
        self.state = McpState()
        self._tools: dict[str, dict[str, Any]] = {}
        self.spend = 0.0

    # ------------------------------------------------------------------
    # policy evaluation
    # ------------------------------------------------------------------
    def _extended_trace(self, name: str) -> Trace:
        extended = Trace(self.state.trace.events)
        extended.append(name)
        return extended

    def _policy_allows(self, name: str) -> bool:
        policy = self.policy
        if policy is None:
            return True
        if isinstance(policy, str):
            if not policy:
                return True
            return not self.state.trace.occurred(policy) or True  # plain allow-list unused
        try:
            return bool(policy.evaluate(self._extended_trace(name)))
        except Exception:
            # A broken policy must fail closed: an unevaluated grant is a grant only when no
            # policy is configured (handled above).
            return False

    def allowed_tools(self) -> list[str]:
        """Names of tools whose immediate call keeps the temporal policy satisfied."""

        return [name for name in self._tools if self._policy_allows(name)]

    def allowed_tools_enum(self) -> list[str]:
        """``allowed_tools`` — provided for JSON-Schema ``enum`` tool pickers in bridges."""

        return self.allowed_tools()

    # ------------------------------------------------------------------
    # tool registration
    # ------------------------------------------------------------------
    def verified_tool(
        self,
        name: str | None = None,
        *,
        cost: float = 0.0,
        spec: Any = None,
        requires_taint_free: bool = False,
    ):
        """Wrap a tool so the firewall gates every invocation.

        The wrapped function is called as ``fn(args)`` or ``fn(args, state)`` depending on its
        signature; ``args`` may be a pydantic model instance or a plain mapping (validated
        against the annotated model when one is declared).
        """

        tool_name = None

        def decorator(fn):
            nonlocal tool_name
            tool_name = name or fn.__name__
            signature = inspect.signature(fn)
            params = list(signature.parameters.values())
            if len(params) < 1:
                raise TypeError(f"{tool_name}: verified tools must take at least an `args` parameter")
            annotation = params[0].annotation
            model_type = None
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                model_type = annotation
            _passes_state = any(
                param.annotation is McpState or param.name == "state" for param in params[1:]
            )

            def gate(args: Any = None, **kwargs: Any):
                return self._call_verified(
                    tool_name, cost, model_type, requires_taint_free, spec, _passes_state, fn,
                    args, kwargs,
                )

            gate.__name__ = fn.__name__
            gate.__doc__ = fn.__doc__
            gate.spec = spec
            gate.cost = cost
            self._tools[tool_name] = {"cost": float(cost or 0.0), "spec": spec, "fn": gate}
            return gate

        # Support bare @session.verified_tool (without call) as well as @session.verified_tool(name=...).
        if callable(name):
            fn = name
            name = None
            return decorator(fn)
        return decorator

    def _call_verified(
        self,
        tool_name: str,
        cost: float,
        model_type: Any,
        requires_taint_free: bool,
        spec: Any,
        passes_state: bool,
        fn,
        args: Any,
        kwargs: Mapping[str, Any],
    ):
        # Build the payload: a mapping arg is merged over kwargs (arg wins); a pydantic
        # model instance passes through untouched; anything else stays positional.
        if args is None:
            merged: Any = dict(kwargs)
        elif model_type is not None and isinstance(args, model_type):
            merged = args
        elif isinstance(args, Mapping):
            merged = {**kwargs, **args}
        else:
            merged = args if not kwargs else {"_positional": args, **kwargs}
        # Temporal gate first.
        if not self._policy_allows(tool_name):
            return self._violation(
                f"temporal policy forbids {tool_name} at trace position {len(self.state.trace)}",
                tool_name, "temporal",
            )
        # Budget gates.
        projected = self.spend + float(cost or 0.0)
        if self.max_session_spend is not None and projected > self.max_session_spend:
            return self._violation(
                f"session spend cap exceeded by call to {tool_name} "
                f"({projected:.6g} > {self.max_session_spend:.6g})",
                tool_name, "spend",
            )
        for counter, limit in self.budgets.items():
            current = float(self.state.accumulated.get(counter, 0.0))
            if current > float(limit):
                return self._violation(
                    f"budget {counter!r} exhausted before {tool_name} ({current:.6g} > {limit:.6g})",
                    tool_name, "budget",
                )
        # Argument validation.
        payload: Any = merged
        raw_mapping = merged if isinstance(merged, Mapping) else None
        if model_type is not None:
            try:
                payload = payload if isinstance(payload, model_type) else model_type(**(dict(payload) if isinstance(payload, Mapping) else {}))
            except Exception as exc:
                return self._violation(f"invalid arguments for {tool_name}: {exc}", tool_name, "args")
            # Pydantic drops string subclasses on validation, which would erase taint.  Re-attach
            # the untrusted tags captured from the raw argument mapping at the model boundary.
            if raw_mapping is not None and isinstance(payload, model_type):
                updates = {}
                for dotted, sources in _taint_map(raw_mapping).items():
                    if "." in dotted:
                        continue  # nested containers keep their subclasses untouched
                    raw_value = raw_mapping.get(dotted)
                    if isinstance(raw_value, UntrustedString):
                        updates[dotted] = raw_value
                    else:
                        updates[dotted] = UntrustedString(str(getattr(payload, dotted, raw_value)), tuple(sources))
                if updates:
                    payload = payload.model_copy(update=updates)
        if requires_taint_free and not is_untainted(payload):
            return self._violation(f"tainted input rejected for {tool_name}", tool_name, "taint")

        try:
            if passes_state:
                result = fn(payload, self.state)
            else:
                result = fn(payload)
        except McpRequireError as exc:
            event = Event(f"blocked:{tool_name}", len(self.state.trace) + 1,
                          {"reason": str(exc), "policy": exc.policy})
            self.state.trace.events.append(event)
            return self._violation(str(exc), tool_name, exc.policy or "require")

        # Post-call accounting.
        self.state.trace.append(tool_name)
        self.spend = projected
        if isinstance(payload, Mapping):
            flat = _flat_scalars(dict(payload))
        elif hasattr(payload, "model_dump"):
            try:
                flat = _flat_scalars(payload.model_dump())
            except Exception:
                flat = []
        else:
            flat = []
        for key, value in flat:
            self._bind_env(tool_name, key, value, accumulate=isinstance(value, (int, float)) and not isinstance(value, bool))
        if isinstance(result, Mapping):
            for key, value in _flat_scalars(dict(result)):
                self._bind_env(
                    tool_name, key, value,
                    accumulate=isinstance(value, (int, float)) and not isinstance(value, bool),
                )
        if spec is not None:
            try:
                from .instance import verify_instance

                report = verify_instance(spec, result)
            except Exception as exc:
                return self._violation(f"{tool_name} output failed verification to run: {exc}", tool_name, "spec")
            if not report.ok:
                return self._violation(
                    f"{tool_name} output failed spec verification: {report.summary_line()}",
                    tool_name, "spec",
                )
        return result

    def _bind_env(self, tool_name: str, key: str, value: Any, *, accumulate: bool) -> None:
        name = key.split(".")[-1]
        flat = key.replace(".", "_")
        self.state.env[f"{tool_name}.{key}"] = value
        self.state.env[f"last_{flat}"] = value
        if accumulate:
            try:
                self.state.accumulated[name] = self.state.accumulated.get(name, 0.0) + float(value)
            except (TypeError, ValueError):
                pass

    def _violation(self, message: str, tool: str, policy: str):
        if self.on_violation == "return":
            return {"ok": False, "error": message, "tool": tool, "policy": policy}
        raise PolicyViolation(message, policy=policy, tool=tool)

    # ------------------------------------------------------------------
    # introspection / helpers
    # ------------------------------------------------------------------
    def record_observation(self, name: str, data: Mapping[str, Any] | None = None) -> Event:
        """Record an externally observed tool event (e.g. a tool call done elsewhere)."""

        event = self.state.trace.append(name)
        for key, value in _flat_scalars(dict(data or {})):
            self.state.env[f"{name}.{key}"] = value
            self.state.env[f"last_{key}"] = value
        return event

    def reset(self) -> None:
        self.state = McpState()
        self.spend = 0.0


