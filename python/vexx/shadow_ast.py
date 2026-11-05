"""Incremental clause-boundary detection for structured output.

``ShadowParser`` feeds bytes as the model emits them and reports *completed top-level
key/value clauses*, which is the signal the Tier-2 verifier needs.  It also tracks the
active semantic slot (the key currently being emitted), used for numeric interval masking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Clause:
    key: str
    value: str
    start: int
    end: int
    complete: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value, "start": self.start, "end": self.end}


class ShadowParser:
    def __init__(self, language: str = "json"):
        self.language = language
        self.buffer = ""
        self.depth = 0
        self.completed_clauses: list[Clause] = []
        self._reported = 0
        self._current_key: str | None = None
        self._key_start = 0

    # -- incremental feeding -------------------------------------------
    def feed_bytes(self, data: bytes | str, token_idx: int = 0, entropy: float = 0.0) -> list[Clause]:
        text = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
        self.buffer += text
        return self._scan()

    def _scan(self) -> list[Clause]:
        clauses, current_key, key_start = _extract_top_level(self.buffer)
        self.depth = _depth(self.buffer)
        self._current_key = current_key
        self._key_start = key_start
        newly: list[Clause] = []
        for clause in clauses[self._reported:]:
            self.completed_clauses.append(clause)
            newly.append(clause)
        self._reported = len(clauses)
        return newly

    def current_semantic_slot(self) -> str | None:
        return self._current_key

    def current_key_span(self) -> tuple:
        return self._key_start, len(self.buffer)

    def parsed_value(self, key: str) -> Any:
        for clause in reversed(self.completed_clauses):
            if clause.key == key:
                try:
                    return json.loads(clause.value)
                except json.JSONDecodeError:
                    return None
        return None

    def reset(self) -> None:
        self.buffer = ""
        self.depth = 0
        self.completed_clauses = []
        self._reported = 0
        self._current_key = None
        self._key_start = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "depth": self.depth,
            "current_key": self._current_key,
            "clauses": [clause.as_dict() for clause in self.completed_clauses],
            "buffer_length": len(self.buffer),
        }


def _depth(text: str) -> int:
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth = max(0, depth - 1)
    return depth


def _extract_top_level(text: str) -> tuple:
    """Return ``(completed clauses, current key, key start)`` for a JSON object prefix."""

    clauses: list[Clause] = []
    i = 0
    n = len(text)
    depth = 0
    in_string = False
    escape = False
    current_key: str | None = None
    key_start = 0
    value_start = -1
    awaiting_colon = False

    while i < n:
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                if depth == 1 and current_key is None:
                    # key string ended
                    raw = text[key_start:i + 1]
                    try:
                        current_key = json.loads(raw)
                    except json.JSONDecodeError:
                        current_key = None
                    awaiting_colon = True
            i += 1
            continue
        if ch == '"':
            in_string = True
            if depth == 1 and current_key is None:
                key_start = i
            i += 1
            continue
        if ch in "[{":
            depth += 1
            if depth == 2 and value_start < 0:
                value_start = i
            i += 1
            continue
        if ch in "]}":
            depth = max(0, depth - 1)
            if depth == 1 and current_key is not None and value_start >= 0:
                # value container closed; clause completes at the comma/brace
                value_start = min(value_start, i)
            elif depth == 0 and current_key is not None and value_start >= 0:
                clauses.append(Clause(current_key, text[value_start:i].strip(), value_start, i))
                current_key = None
                value_start = -1
            i += 1
            continue
        if depth == 1 and awaiting_colon and ch == ":":
            awaiting_colon = False
            value_start = i + 1
            i += 1
            continue
        if depth == 1 and ch == "," and current_key is not None and value_start >= 0:
            clauses.append(Clause(current_key, text[value_start:i].strip(), value_start, i))
            current_key = None
            value_start = -1
            i += 1
            continue
        i += 1

    # A scalar value at the end of the current prefix is not complete until ',' or '}'.
    return clauses, current_key, key_start


ShadowAst = ShadowParser
ShadowAstParser = ShadowParser
IncrementalParser = ShadowParser
TreeSitterShadow = ShadowParser

__all__ = [
    "Clause",
    "IncrementalParser",
    "ShadowAst",
    "ShadowAstParser",
    "ShadowParser",
    "TreeSitterShadow",
]
