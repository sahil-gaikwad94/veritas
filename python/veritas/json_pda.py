"""Tier-0 grammar: a real byte/char-level JSON pushdown automaton.

The docs describe a syntactic PDA running at every token, with context-independent
token bitmasks.  This module implements the language-recognition half for JSON — the
grammar used by ``@vt.verified_schema`` output — plus a token masker that turns it into
logit masks, and a canonical renderer used by 0-DoF Jump-Forward splicing.

Scope is honest: this is a deterministic JSON recognizer (not a full CFG stack machine),
and masking is exact over a supplied vocabulary.  Production SIMD bitmask caches belong
in the Rust core; the API here is the reference implementation and is fully testable.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Set, Tuple

_NUMBER_RE = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$")
_HEX = set("0123456789abcdefABCDEF")
_WS = set(" \t\n\r")
_ESCAPES = set('"\\/bfnrt')
_LITERALS = ("true", "false", "null")

# Fast (curated) allowed sets; `allowed_chars_probed` verifies them against the machine.
_VALUE_START = set('{["-tfn0123456789')


class JSONSyntaxError(ValueError):
    """Raised when text is not a well-formed JSON prefix continuation."""


class JSONPushdownAutomaton:
    """Deterministic JSON recognizer with an explicit container stack.

    ``feed`` accepts *prefixes*; call :meth:`accept` to require a complete document.
    """

    __slots__ = (
        "stack",
        "state",
        "error",
        "_number",
        "_string",
        "_string_role",
        "_escape",
        "_unicode_left",
        "_literal",
        "_literal_pos",
        "max_depth",
        "text",
    )

    def __init__(self, max_depth: int = 64):
        self.stack: List[Dict[str, Any]] = []
        self.state = "value"
        self.error: Optional[str] = None
        self._number = ""
        self._string: List[str] = []
        self._string_role = "value"
        self._escape = False
        self._unicode_left = 0
        self._literal = ""
        self._literal_pos = 0
        self.max_depth = max_depth
        self.text = ""

    # -- state -----------------------------------------------------------
    def clone(self) -> "JSONPushdownAutomaton":
        other = JSONPushdownAutomaton(self.max_depth)
        other.stack = [dict(frame) for frame in self.stack]
        other.state = self.state
        other.error = self.error
        other._number = self._number
        other._string = list(self._string)
        other._string_role = self._string_role
        other._escape = self._escape
        other._unicode_left = self._unicode_left
        other._literal = self._literal
        other._literal_pos = self._literal_pos
        other.text = self.text
        return other

    @property
    def depth(self) -> int:
        return len(self.stack)

    def reset(self) -> None:
        self.__init__(self.max_depth)  # type: ignore[misc]

    def is_error(self) -> bool:
        return self.state == "error"

    def accepts(self) -> bool:
        return self.state == "done" and not self.stack

    def in_string(self) -> bool:
        return self.state == "string"

    def current_key_role(self) -> Optional[str]:
        if self.state == "string":
            return self._string_role
        return None

    # -- feeding ---------------------------------------------------------
    def feed(self, text: str) -> bool:
        for ch in text:
            if not self.step(ch):
                return False
        return True

    def step(self, ch: str) -> bool:
        if self.state == "error":
            return False
        if len(ch) != 1:
            return self._fail(f"expected a single character, got {ch!r}")
        self.text += ch
        try:
            self._step(ch)
        except JSONSyntaxError as exc:  # pragma: no cover - defensive
            return self._fail(str(exc))
        return self.state != "error"

    def _fail(self, reason: str) -> bool:
        self.state = "error"
        self.error = f"{reason} at char {len(self.text) - 1}"
        return False

    def _step(self, ch: str) -> None:
        state = self.state

        if state == "done":
            if ch in _WS:
                return
            self._fail("trailing characters after JSON document")
            return

        if state == "string":
            self._step_string(ch)
            return

        if state == "number":
            self._step_number(ch)
            return

        if state == "literal":
            self._step_literal(ch)
            return

        if ch in _WS:
            return

        if state in ("value", "array_value"):
            self._start_value(ch)
            return

        if state == "object_key_or_close":
            if ch == "}":
                self._close_container("obj")
            elif ch == '"':
                self._start_string("key")
            else:
                self._fail("expected object key or '}'")
            return

        if state == "object_key":
            if ch == '"':
                self._start_string("key")
            else:
                self._fail("expected object key after ','")
            return

        if state == "object_colon":
            if ch == ":":
                self.state = "value"
            else:
                self._fail("expected ':' after object key")
            return

        if state == "object_comma_or_close":
            if ch == ",":
                self.state = "object_key"
            elif ch == "}":
                self._close_container("obj")
            else:
                self._fail("expected ',' or '}'")
            return

        if state == "array_value_or_close":
            if ch == "]":
                self._close_container("arr")
            else:
                self._start_value(ch)
            return

        if state == "array_comma_or_close":
            if ch == ",":
                self.state = "array_value"
            elif ch == "]":
                self._close_container("arr")
            else:
                self._fail("expected ',' or ']'")
            return

        self._fail(f"unreachable state {state!r}")

    # -- tokens ----------------------------------------------------------
    def _start_value(self, ch: str) -> None:
        if ch == "{":
            if len(self.stack) >= self.max_depth:
                self._fail("maximum JSON depth exceeded")
                return
            self.stack.append({"type": "obj", "count": 0})
            self.state = "object_key_or_close"
        elif ch == "[":
            if len(self.stack) >= self.max_depth:
                self._fail("maximum JSON depth exceeded")
                return
            self.stack.append({"type": "arr", "count": 0})
            self.state = "array_value_or_close"
        elif ch == '"':
            self._start_string("value")
        elif ch == "-" or ch.isdigit():
            self.state = "number"
            self._number = ""
            self._step_number(ch)
        elif ch in "tfn":
            for literal in _LITERALS:
                if literal.startswith(ch):
                    self._literal = literal
                    self._literal_pos = 1
                    self.state = "literal"
                    if len(literal) == 1:
                        self._finish_literal()
                    return
            self._fail(f"invalid literal starting with {ch!r}")
        else:
            self._fail(f"expected a JSON value, got {ch!r}")

    def _start_string(self, role: str) -> None:
        self.state = "string"
        self._string = []
        self._string_role = role
        self._escape = False
        self._unicode_left = 0

    def _step_string(self, ch: str) -> None:
        if self._escape:
            if self._unicode_left:
                if ch not in _HEX:
                    self._fail("expected hex digit in \\u escape")
                    return
                self._string.append(ch)
                self._unicode_left -= 1
                if self._unicode_left == 0:
                    self._escape = False
                return
            if ch == "u":
                self._unicode_left = 4
                return
            if ch in _ESCAPES:
                self._string.append("\\" + ch)
                self._escape = False
                return
            self._fail(f"invalid escape \\{ch}")
            return

        if ch == '"':
            value = "".join(self._string)
            if self._string_role == "key":
                self.state = "object_colon"
            else:
                self._after_value()
            return
        if ch == "\\":
            self._escape = True
            return
        if ord(ch) < 0x20:
            self._fail("unescaped control character in string")
            return
        self._string.append(ch)

    def _step_number(self, ch: str) -> None:
        if not self._number:
            if ch == "-":
                self._number = "-"
                return
            if ch.isdigit():
                self._number = ch
                return
            self._fail(f"invalid JSON number start {ch!r}")
            return
        if ch in _WS or ch in ",}]":
            if not _NUMBER_RE.match(self._number):
                self._fail(f"invalid JSON number {self._number!r}")
                return
            self._number = ""
            self._after_value()
            if ch not in _WS:
                # Reprocess the structural delimiter in the new state.
                self.text = self.text[:-1]
                self._step(ch)
            return
        candidate = self._number + ch
        if ch.isdigit():
            self._number = candidate
            return
        if ch == ".":
            if "." in self._number or "e" in self._number.lower():
                self._fail("misplaced decimal point in number")
                return
            if not self._number or self._number == "-":
                self._fail("decimal point must follow an integer part")
                return
            self._number = candidate
            return
        if ch in "eE":
            if "e" in self._number.lower():
                self._fail("duplicate exponent in number")
                return
            if not self._number or self._number in ("-",) or self._number.endswith("."):
                self._fail("exponent must follow digits")
                return
            self._number = candidate
            return
        if ch in "+-":
            if not self._number or self._number[-1] not in "eE":
                self._fail("sign is only allowed after an exponent marker")
                return
            self._number = candidate
            return
        self._fail(f"invalid character {ch!r} in number")

    def _step_literal(self, ch: str) -> None:
        if self._literal_pos >= len(self._literal):
            self._finish_literal()
            self.text = self.text[:-1]
            self._step(ch)
            return
        if ch != self._literal[self._literal_pos]:
            self._fail(f"invalid literal; expected {self._literal!r}")
            return
        self._literal_pos += 1
        if self._literal_pos == len(self._literal):
            self._finish_literal()

    def _finish_literal(self) -> None:
        self._literal = ""
        self._literal_pos = 0
        self._after_value()

    def _after_value(self) -> None:
        if not self.stack:
            self.state = "done"
            return
        top = self.stack[-1]
        top["count"] += 1
        self.state = "object_comma_or_close" if top["type"] == "obj" else "array_comma_or_close"

    def _close_container(self, expected: str) -> None:
        if not self.stack or self.stack[-1]["type"] != expected:
            self._fail("mismatched closing delimiter")
            return
        self.stack.pop()
        self._after_value()

    # -- introspection ---------------------------------------------------
    def allowed_chars(self) -> Set[str]:
        """Curated set of characters that keep the machine in a valid prefix state."""

        state = self.state
        if state == "error":
            return set()
        if state == "done":
            return set(_WS)
        if state in ("value", "array_value"):
            return set(_VALUE_START) | _WS
        if state == "object_key_or_close":
            return {'"', "}"} | _WS
        if state == "object_key":
            return {'"'} | _WS
        if state == "object_colon":
            return {":"} | _WS
        if state == "object_comma_or_close":
            return {",", "}"} | _WS
        if state == "array_value_or_close":
            return _VALUE_START | {"]"} | _WS
        if state == "array_comma_or_close":
            return {",", "]"} | _WS
        if state == "string":
            if self._escape:
                if self._unicode_left:
                    return set(_HEX)
                return set(_ESCAPES) | {"u"}
            return set(chr(c) for c in range(0x20, 0x7F)) | {'"', "\\"}
        if state == "number":
            candidates = set("0123456789") | {".", "e", "E", "+", "-"} | {",", "}", "]"} | _WS
            return {ch for ch in candidates if self._probe(ch)}
        if state == "literal":
            if self._literal_pos < len(self._literal):
                return {self._literal[self._literal_pos]}
            return set()
        return set()

    def _probe(self, ch: str) -> bool:
        probe = self.clone()
        return probe.step(ch)

    def allowed_chars_probed(self, alphabet: Optional[Iterable[str]] = None) -> Set[str]:
        """Compute the exact allowed set by trial stepping (used to verify the fast path)."""

        if alphabet is None:
            alphabet = [chr(c) for c in range(0x20, 0x7F)] + ["\n", "\t", "\r"]
        allowed: Set[str] = set()
        for ch in alphabet:
            probe = self.clone()
            if probe.step(ch):
                allowed.add(ch)
        return allowed

    def key_prefix_allowed(self, prefix: str, keys: Sequence[str]) -> Set[str]:
        """Next characters that keep ``prefix`` a prefix of at least one allowed key."""

        out: Set[str] = set()
        for key in keys:
            if key.startswith(prefix) and len(key) > len(prefix):
                out.add(key[len(prefix)])
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "depth": len(self.stack),
            "containers": [frame["type"] for frame in self.stack],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# token masking
# ---------------------------------------------------------------------------

@dataclass
class MaskStats:
    prefix_length: int = 0
    vocabulary: int = 0
    allowed: int = 0
    masked: int = 0
    eos_allowed: bool = False


class TokenMasker:
    """Exact token-level grammar mask over a supplied vocabulary.

    Tokens are strings (the text a token decodes to).  A token is admissible when the
    current prefix plus that token is still a valid JSON prefix.  The ``eos`` token is
    admissible only when the document is complete.
    """

    def __init__(
        self,
        vocabulary: Sequence[str],
        *,
        eos_token: Optional[int] = None,
        cache_size: int = 8192,
    ):
        self.vocabulary = list(vocabulary)
        self.eos_token = eos_token
        self.cache_size = max(16, int(cache_size))
        self._cache: Dict[str, List[bool]] = {}
        self._order: List[str] = []
        self.stats = MaskStats(vocabulary=len(self.vocabulary))

    def _state_for(self, prefix: str) -> JSONPushdownAutomaton:
        machine = JSONPushdownAutomaton()
        if prefix and not machine.feed(prefix):
            raise JSONSyntaxError(f"invalid JSON prefix: {machine.error}")
        return machine

    def valid_mask(self, prefix: str) -> List[bool]:
        cached = self._cache.get(prefix)
        if cached is not None:
            return cached
        base = self._state_for(prefix)
        mask: List[bool] = []
        for text in self.vocabulary:
            probe = base.clone()
            ok = probe.feed(text) and not probe.is_error()
            mask.append(ok)
        if self.eos_token is not None:
            while len(mask) <= self.eos_token:
                mask.append(False)
            mask[self.eos_token] = base.accepts()
        self._cache[prefix] = mask
        self._order.append(prefix)
        if len(self._order) > self.cache_size:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)
        self.stats = MaskStats(
            prefix_length=len(prefix),
            vocabulary=len(self.vocabulary),
            allowed=sum(1 for v in mask if v),
            masked=sum(1 for v in mask if not v),
            eos_allowed=base.accepts(),
        )
        return mask

    def apply(
        self,
        prefix: str,
        logits: Any,
        *,
        negative: float = -math.inf,
    ) -> Any:
        """Mask logits in place-compatible fashion.

        Accepts ``Mapping[int, float]``, a mutable sequence, or a NumPy array.  Returns a
        new object for mappings; mutates sequences/arrays and returns them.
        """

        mask = self.valid_mask(prefix)
        if isinstance(logits, Mapping):
            out = dict(logits)
            for idx in range(len(self.vocabulary)):
                if idx < len(mask) and not mask[idx]:
                    out[idx] = negative
            if self.eos_token is not None and not mask[self.eos_token]:
                out[self.eos_token] = negative
            return out
        # sequences / numpy arrays
        n = len(logits)
        for idx in range(min(n, len(mask))):
            if not mask[idx]:
                logits[idx] = negative
        if self.eos_token is not None and self.eos_token < n and not mask[self.eos_token]:
            logits[self.eos_token] = negative
        return logits

    def clear_cache(self) -> None:
        self._cache.clear()
        self._order.clear()


# ---------------------------------------------------------------------------
# canonical rendering (used by 0-DoF Jump-Forward)
# ---------------------------------------------------------------------------

def json_number(value: Any, precision: Optional[int] = None) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"non-finite number {value!r} cannot be emitted as JSON")
        rounded = round(value, precision) if precision is not None else value
        if rounded == int(rounded) and abs(rounded) < 1e16:
            return str(int(rounded))
        return repr(float(rounded))
    raise TypeError(f"{type(value).__name__} is not a number")


def json_value(value: Any, precision: Optional[int] = None) -> str:
    if isinstance(value, bool) or value is None:
        return "true" if value is True else ("false" if value is False else "null")
    if isinstance(value, (int, float)):
        return json_number(value, precision)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return "{" + ", ".join(
            f"{json.dumps(str(k), ensure_ascii=False)}: {json_value(v, precision)}"
            for k, v in items
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(json_value(v, precision) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return "[" + ", ".join(json_value(v, precision) for v in sorted(value, key=str)) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def render_object(
    values: Mapping[str, Any],
    field_order: Optional[Sequence[str]] = None,
    precision: Optional[Mapping[str, int]] = None,
) -> str:
    order = list(field_order) if field_order else [k for k in values.keys()]
    seen: Set[str] = set()
    parts: List[str] = []
    for key in order:
        if key in seen or key not in values:
            continue
        seen.add(key)
        prec = precision.get(key) if precision else None
        parts.append(f"{json.dumps(str(key), ensure_ascii=False)}: {json_value(values[key], prec)}")
    for key in values:
        if key in seen:
            continue
        prec = precision.get(key) if precision else None
        parts.append(f"{json.dumps(str(key), ensure_ascii=False)}: {json_value(values[key], prec)}")
    return "{" + ", ".join(parts) + "}"


def render_continuation(
    values: Mapping[str, Any],
    keys: Sequence[str],
    precision: Optional[Mapping[str, int]] = None,
) -> str:
    """Render ``, "k": v`` fragments for the supplied keys (Jump-Forward splice payload)."""

    parts: List[str] = []
    for key in keys:
        if key not in values:
            continue
        prec = precision.get(key) if precision else None
        parts.append(f", {json.dumps(str(key), ensure_ascii=False)}: {json_value(values[key], prec)}")
    return "".join(parts)


def parse_json(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(JSONSyntaxError(f"invalid constant {c}")))
    except json.JSONDecodeError as exc:
        raise JSONSyntaxError(str(exc)) from exc


def is_complete_json(text: str) -> bool:
    machine = JSONPushdownAutomaton()
    return machine.feed(text) and machine.accepts()


__all__ = [
    "JSONPushdownAutomaton",
    "JSONSyntaxError",
    "MaskStats",
    "TokenMasker",
    "is_complete_json",
    "json_number",
    "json_value",
    "parse_json",
    "render_continuation",
    "render_object",
]
