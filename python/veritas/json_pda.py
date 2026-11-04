"""Incremental RFC 8259 JSON recognition, strict parsing, and vocabulary masking."""
from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_NUMBER_RE = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$")
_WS = set(" \t\n\r")
_HEX = set("0123456789abcdefABCDEF")
_ESCAPES = set('"\\/bfnrt')
_VALUE_START = set('{["-tfn0123456789')


class JSONSyntaxError(ValueError):
    pass


class JSONPushdownAutomaton:
    """Deterministic JSON prefix recognizer with an explicit container stack.

    ``last_event`` exposes clause boundaries without reparsing the accumulated text.
    A number/literal is complete only when its grammar is complete; a delimiter is
    reprocessed in the parent state, never used to excuse a malformed exponent.
    """
    def __init__(self, max_depth: int = 64):
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        self.max_depth = max_depth
        self.stack: list[dict[str, Any]] = []
        self.state = "value"
        self.error: str | None = None
        self.text = ""
        self._number = ""
        self._string: list[str] = []
        self._string_role = "value"
        self._escape = False
        self._unicode_left = 0
        self._literal = ""
        self._literal_pos = 0
        self._value_start = 0
        self.last_event: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []

    def clone(self):
        other = JSONPushdownAutomaton(self.max_depth)
        other.stack = [dict(frame) for frame in self.stack]
        for name in ("state", "error", "text", "_number", "_string_role", "_escape",
                     "_unicode_left", "_literal", "_literal_pos", "_value_start"):
            setattr(other, name, getattr(self, name))
        other._string = list(self._string)
        other.last_event = dict(self.last_event) if self.last_event else None
        other.events = [dict(e) for e in self.events]
        return other

    @property
    def depth(self):
        return len(self.stack)

    def signature(self) -> tuple[Any, ...] | None:
        """Canonical signature of everything that affects the future of the parse.

        Two prefixes with the same signature accept exactly the same set of future tokens,
        regardless of the text processed so far.  Used by the mask cache as a key much
        cheaper than string comparison over the whole document; ``hash()`` of the seen-key
        sets is run-local, which is fine for an in-process LRU.
        """

        if self.state == "error":
            return None
        stack_sig = tuple(
            (frame["type"], len(frame["keys"]), hash(frozenset(frame["keys"])))
            for frame in self.stack
        )
        string_meta = (self._string_role, len(self._string), self._escape, self._unicode_left)
        return (self.state, stack_sig, self._number, string_meta, self._literal_pos, self._literal)

    def reset(self):
        self.__init__(self.max_depth)

    def is_error(self):
        return self.state == "error"

    def finish(self):
        """Complete a pending number/literal at end-of-input (RFC 8259 allows EOF)."""
        if self.state == "number" and _NUMBER_RE.fullmatch(self._number):
            self._finish_value(len(self.text))
        elif self.state == "literal" and self._literal_pos == len(self._literal):
            self._finish_value(len(self.text))
        return self.state != "error"

    def accepts(self):
        return self.state == "done" and not self.stack

    def in_string(self):
        return self.state == "string"

    def current_key_role(self):
        return self._string_role if self.state == "string" else None

    def feed(self, text: str) -> bool:
        if not isinstance(text, str):
            return self._fail("JSON input must be text")
        return all(self.step(ch) for ch in text)

    def step(self, ch: str) -> bool:
        if self.state == "error":
            return False
        if not isinstance(ch, str) or len(ch) != 1:
            return self._fail("expected one Unicode character")
        self.last_event = None
        self.events = []
        offset = len(self.text)
        self.text += ch
        # At most one delimiter/literal boundary needs reprocessing. A delimiter can
        # complete a pending number AND close its parent; expose both events in order.
        for _ in range(2):
            consumed = self._consume(ch, offset)
            if self.last_event is not None:
                self.events.append(self.last_event)
            if consumed:
                return self.state != "error"
            if self.state == "error":
                return False
            self.last_event = None
        return self._fail("internal boundary loop")

    def _consume(self, ch, offset):
        """Return True when consumed; False requests one boundary reprocess."""
        state = self.state
        if state == "string":
            self._step_string(ch, offset)
            return True
        if state == "number":
            if ch in _WS or ch in ",}]":
                if not _NUMBER_RE.fullmatch(self._number):
                    self._fail(f"invalid JSON number {self._number!r}", offset)
                    return True
                self._finish_value(offset)
                return False
            self._step_number(ch, offset)
            return True
        if state == "literal":
            if self._literal_pos >= len(self._literal):
                self._finish_value(offset)
                return False
            if ch != self._literal[self._literal_pos]:
                self._fail(f"invalid literal; expected {self._literal!r}", offset)
                return True
            self._literal_pos += 1
            if self._literal_pos == len(self._literal):
                self._finish_value(offset + 1)
            return True
        if ch in _WS:
            return True
        if state == "done":
            self._fail("trailing characters after JSON document", offset)
        elif state in ("value", "array_value", "array_value_or_close"):
            if state == "array_value_or_close" and ch == "]":
                self._close_container("arr", offset)
            else:
                self._start_value(ch, offset)
        elif state in ("object_key_or_close", "object_key"):
            if ch == "}" and state == "object_key_or_close":
                self._close_container("obj", offset)
            elif ch == '"':
                self._start_string("key")
            else:
                self._fail("expected object key" + " or '}'" if state == "object_key_or_close" else "expected object key", offset)
        elif state == "object_colon":
            if ch == ":":
                self.state = "value"
            else:
                self._fail("expected ':' after object key", offset)
        elif state in ("object_comma_or_close", "array_comma_or_close"):
            obj = state.startswith("object")
            if ch == ",":
                self.state = "object_key" if obj else "array_value"
            elif ch == ("}" if obj else "]"):
                self._close_container("obj" if obj else "arr", offset)
            else:
                self._fail("expected ',' or closing delimiter", offset)
        else:
            self._fail(f"unreachable state {state!r}", offset)
        return True

    def _fail(self, reason, offset=None):
        self.state = "error"
        self.error = f"{reason} at char {len(self.text) - 1 if offset is None else offset}"
        return False

    def _start_value(self, ch, offset):
        self._value_start = offset
        container = {"{": "obj", "[": "arr"}.get(ch)
        if container:
            if len(self.stack) >= self.max_depth:
                self._fail("maximum JSON depth exceeded", offset)
                return
            self.stack.append({"type": container, "count": 0, "start": offset, "keys": set()})
            self.state = "object_key_or_close" if container == "obj" else "array_value_or_close"
            self.last_event = {"kind": "container_start", "container": container, "start": offset, "end": offset + 1}
            return
        self.last_event = {"kind": "value_start", "start": offset, "char": ch}
        if ch == '"':
            self._start_string("value")
        elif ch == "-" or ch in "0123456789":
            self.state = "number"
            self._number = ch
        elif ch in "tfn":
            self._literal = next(word for word in ("true", "false", "null") if word.startswith(ch))
            self._literal_pos = 1
            self.state = "literal"
        else:
            self._fail(f"expected a JSON value, got {ch!r}", offset)

    def _start_string(self, role):
        self.state = "string"
        self._string = []
        self._string_role = role
        self._escape = False
        self._unicode_left = 0

    def _step_string(self, ch, offset):
        if self._escape:
            if self._unicode_left:
                if ch not in _HEX:
                    self._fail("expected hex digit in unicode escape", offset)
                    return
                self._string.append(ch)
                self._unicode_left -= 1
                if not self._unicode_left:
                    self._escape = False
            elif ch == "u":
                self._unicode_left = 4
            elif ch in _ESCAPES:
                self._string.append("\\" + ch)
                self._escape = False
            else:
                self._fail(f"invalid escape \\{ch}", offset)
            return
        if ch == '"':
            raw = '"' + "".join(self._string) + '"'
            if self._string_role == "key":
                key = json.loads(raw)
                top = self.stack[-1]
                self.last_event = {"kind": "key", "key": key, "duplicate": key in top["keys"], "start": self._value_start, "end": offset + 1}
                top["keys"].add(key)
                self.state = "object_colon"
            else:
                self._finish_value(offset + 1)
            return
        if ch == "\\":
            self._escape = True
        elif ord(ch) < 0x20:
            self._fail("unescaped control character in string", offset)
        else:
            self._string.append(ch)

    def _step_number(self, ch, offset):
        candidate = self._number + ch
        # Permit precisely the prefixes of the JSON number grammar.
        if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?", candidate):
            self._number = candidate
        else:
            self._fail(f"invalid character {ch!r} in number {self._number!r}", offset)

    def _finish_value(self, end):
        start = self._value_start
        self.last_event = {"kind": "value", "start": start, "end": end, "text": self.text[start:end], "container": self.stack[-1]["type"] if self.stack else None}
        self._number = ""
        self._after_value()

    def _after_value(self):
        if not self.stack:
            self.state = "done"
            return
        top = self.stack[-1]
        top["count"] += 1
        self.state = "object_comma_or_close" if top["type"] == "obj" else "array_comma_or_close"

    def _close_container(self, expected, offset):
        if not self.stack or self.stack[-1]["type"] != expected:
            self._fail("mismatched closing delimiter", offset)
            return
        frame = self.stack.pop()
        self._value_start = frame["start"]
        self.last_event = {"kind": "container_end", "container": expected, "start": frame["start"],
                           "end": offset + 1, "text": self.text[frame["start"]:offset + 1], "count": frame["count"]}
        self._after_value()

    def allowed_chars(self):
        return self.allowed_chars_probed()

    def allowed_chars_probed(self, alphabet=None):
        if alphabet is None:
            alphabet = [chr(c) for c in range(0x20, 0x7F)] + ["\n", "\t", "\r"]
        return {ch for ch in alphabet if self.clone().step(ch)}

    def key_prefix_allowed(self, prefix, keys):
        return {key[len(prefix)] for key in keys if key.startswith(prefix) and len(key) > len(prefix)}

    def as_dict(self):
        return {"state": self.state, "depth": self.depth, "containers": [f["type"] for f in self.stack], "error": self.error}


@dataclass
class MaskStats:
    prefix_length: int = 0
    vocabulary: int = 0
    allowed: int = 0
    masked: int = 0
    eos_allowed: bool = False


class TokenMasker:
    """Exact prefix mask over validated token text, optionally schema-aware."""
    def __init__(self, vocabulary: Sequence[str], *, eos_token=None, cache_size=8192, schema=None):
        self.vocabulary = list(vocabulary)
        if any(not isinstance(t, str) for t in self.vocabulary):
            raise TypeError("vocabulary entries must be decoded token strings")
        if eos_token is not None and (isinstance(eos_token, bool) or not isinstance(eos_token, int) or eos_token < 0):
            raise ValueError("eos_token must be a non-negative integer")
        self.eos_token = eos_token
        if eos_token is not None and eos_token < len(self.vocabulary) and self.vocabulary[eos_token] != "":
            raise ValueError("EOS index collides with a text token")
        self.cache_size = max(16, int(cache_size))
        self.schema = schema
        self._cache = OrderedDict()
        self.stats = MaskStats(vocabulary=len(self.vocabulary))

    def _state_for(self, prefix):
        if self.schema is not None:
            from .schema_pda import SchemaJSONAutomaton
            machine = SchemaJSONAutomaton(self.schema)
        else:
            machine = JSONPushdownAutomaton()
        if prefix and not machine.feed(prefix):
            raise JSONSyntaxError(f"invalid JSON prefix: {machine.error}")
        return machine

    def _cache_key(self, prefix, machine):
        """Canonical cache key: the PDA signature when available, else the raw prefix."""

        if self.schema is None:
            try:
                sig = machine.signature()
            except Exception:
                sig = None
            if sig is not None:
                return sig
        return prefix

    def valid_mask(self, prefix):
        """Boolean mask aligned with the vocabulary; every non-JSON-continuation is False."""

        machine = self._state_for(prefix)
        key = self._cache_key(prefix, machine)
        if key in self._cache:
            self._cache.move_to_end(key)
            return list(self._cache[key])
        native_mask = None
        if self.schema is None:
            from . import native as _native
            native_mask = _native.mask_or_none(prefix, self.vocabulary, self.eos_token)
        if native_mask is not None:
            self._cache[key] = tuple(native_mask)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            eos_ok = bool(
                self.eos_token is not None
                and self.eos_token < len(native_mask)
                and native_mask[self.eos_token]
            )
            self.stats = MaskStats(
                len(prefix), len(self.vocabulary), sum(native_mask), len(native_mask) - sum(native_mask), eos_ok
            )
            return native_mask
        base = machine
        # Empty text tokens cannot make progress and must never be sampled as grammar.
        mask = [bool(text) and base.clone().feed(text) for text in self.vocabulary]
        if self.eos_token is not None:
            mask.extend([False] * max(0, self.eos_token + 1 - len(mask)))
            eos_ok = base.accepts()
            if not eos_ok:
                probe = base.clone()
                if hasattr(probe, "finish"):
                    probe.finish()
                    eos_ok = probe.accepts()
            mask[self.eos_token] = eos_ok
        self._cache[key] = tuple(mask)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        self.stats = MaskStats(
            len(prefix), len(self.vocabulary), sum(mask), len(mask) - sum(mask),
            bool(self.eos_token is not None and mask[self.eos_token]),
        )
        return mask

    def apply(self, prefix, logits, *, negative=-math.inf):
        mask = self.valid_mask(prefix)
        if isinstance(logits, Mapping):
            out = dict(logits)
            for idx in out:
                if idx >= len(mask) or not mask[idx]:
                    out[idx] = negative
            return out
        if len(logits) < len(self.vocabulary):
            raise ValueError("logits are smaller than the tokenizer vocabulary")
        for idx in range(len(logits)):
            if idx >= len(mask) or not mask[idx]:
                logits[idx] = negative
        return logits

    def clear_cache(self):
        self._cache.clear()


def json_number(value, precision=None):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number {value!r} cannot be emitted as JSON")
        rounded = round(value, precision) if precision is not None else value
        if not math.isfinite(rounded):
            raise ValueError("rounding overflow")
        if rounded == int(rounded) and abs(rounded) < 1e16:
            return str(int(rounded))
        return repr(float(rounded))
    raise TypeError(f"{type(value).__name__} is not a number")


def json_value(value, precision=None):
    if value is None or isinstance(value, bool):
        return "null" if value is None else ("true" if value else "false")
    if isinstance(value, (int, float)):
        return json_number(value, precision)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("JSON object keys must be strings")
        return "{" + ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {json_value(v, precision)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(json_value(v, precision) for v in value) + "]"
    rendered = _scalar_json(value)
    if rendered is not None:
        return rendered
    raise TypeError(f"cannot serialize {type(value).__name__} as JSON")


def _scalar_json(value):
    """JSON rendering for the scalar types Pydantic models actually produce.

    ``verify_values`` and ``verify_instance`` compare *typed* values (a model field may be
    a ``datetime``, ``UUID``, ``Decimal`` or IP address), while JSON Schema validation needs
    their JSON form.  Rendering them here keeps one canonical spelling for both the decoder
    and the verifier; unfamiliar objects still fail loudly.
    """

    from datetime import date, datetime, time, timedelta
    from decimal import Decimal
    from enum import Enum
    from pathlib import PurePath
    from uuid import UUID

    if isinstance(value, Decimal):
        if value.is_finite():
            return json_number(float(value))
        return json.dumps(str(value))
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, datetime):
        text = value.isoformat()
        return json.dumps(text[:-6] + "Z" if text.endswith("+00:00") else text)
    if isinstance(value, (date, time)):
        return json.dumps(value.isoformat())
    if isinstance(value, timedelta):
        return json.dumps(_iso_duration(value))
    if isinstance(value, (UUID, PurePath)):
        return json.dumps(str(value))
    # ``ipaddress`` objects are not part of a small stable base class; duck-type them.
    module = type(value).__module__
    if module == "ipaddress":
        return json.dumps(str(value))
    # Pydantic's ``Url``/``SecretStr`` wrappers: JSON sees their string payload.
    if hasattr(value, "get_secret_value"):
        secret = value.get_secret_value()
        if isinstance(secret, (str, int, float, bytes)):
            return json_value(secret.decode() if isinstance(secret, bytes) else secret)
    if module.startswith("pydantic") and str(type(value).__name__) in {
        "Url",
        "MultiHostUrl",
        "AnyUrl",
        "HttpUrl",
        "HttpsUrl",
        "AnyHttpUrl",
        "FileUrl",
        "PostgresDsn",
        "RedisDsn",
        "WebsocketUrl",
    }:
        return json.dumps(str(value))
    return None


def _iso_duration(delta):
    """ISO-8601 duration, matching Pydantic's JSON spelling for ``timedelta``."""

    total = delta.total_seconds()
    sign = "-" if total < 0 else ""
    total = abs(total)
    days, remainder = divmod(total, 86400.0)
    hours, remainder = divmod(remainder, 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    body = "T"
    if days:
        body = f"{int(days)}D" + body
    if seconds or not (days or hours or minutes):
        body += f"{seconds:g}S"
    else:
        if hours:
            body += f"{int(hours)}H"
        if minutes:
            body += f"{int(minutes)}M"
    return f"{sign}P{body}"


def render_object(values, field_order=None, precision=None):
    order = list(dict.fromkeys([*(field_order or values.keys()), *values.keys()]))
    parts = []
    for key in order:
        if key in values:
            prec = precision.get(key) if precision else None
            parts.append(f"{json_value(str(key))}: {json_value(values[key], prec)}")
    return "{" + ", ".join(parts) + "}"


def render_continuation(values, keys, precision=None):
    return "".join(f", {json_value(str(k))}: {json_value(values[k], precision.get(k) if precision else None)}"
                   for k in keys if k in values)


def parse_json(text: str, *, max_depth=64, max_bytes=8_388_608):
    """Strict JSON: no NaN/Infinity, duplicate keys, or unbounded nesting."""
    if not isinstance(text, str):
        raise JSONSyntaxError("JSON input must be text")
    if len(text.encode("utf-8")) > max_bytes:
        raise JSONSyntaxError("JSON document exceeds byte limit")
    machine = JSONPushdownAutomaton(max_depth)
    if not (machine.feed(text) and machine.finish() and machine.accepts()):
        raise JSONSyntaxError(machine.error or "incomplete JSON")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise JSONSyntaxError(f"duplicate object key {key!r}")
            result[key] = value
        return result
    def constant(value):
        raise JSONSyntaxError(f"invalid JSON constant {value}")
    def real(value):
        number = float(value)
        if not math.isfinite(number):
            raise JSONSyntaxError("number outside finite IEEE-754 range")
        return number
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=real)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise JSONSyntaxError(str(exc)) from exc


def is_complete_json(text):
    machine = JSONPushdownAutomaton()
    return isinstance(text, str) and machine.feed(text) and machine.accepts()


__all__ = ["JSONPushdownAutomaton", "JSONSyntaxError", "MaskStats", "TokenMasker", "is_complete_json",
           "json_number", "json_value", "parse_json", "render_continuation", "render_object"]