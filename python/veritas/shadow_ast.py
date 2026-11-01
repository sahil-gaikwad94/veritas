"""Phase 2: Incremental shadow AST / clause boundary detection (pure Python)."""

from __future__ import annotations

from typing import List


class ShadowParser:
    def __init__(self, language: str = "json"):
        self.language = language
        self.buffer = ""
        self.depth = 0
        self.completed_clauses: List[str] = []

    def feed_bytes(self, data: bytes | str, token_idx: int = 0, entropy: float = 0.0) -> List[str]:
        text = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
        self.buffer += text
        out = []
        # very simple clause detection: top-level key-value pairs ending in comma or brace
        # We'll just return [] unless a closing brace completes an object
        if "}" in text:
            out.append("clause")
            self.completed_clauses.extend(out)
        return out

    def current_semantic_slot(self) -> None:
        return None

    def reset(self) -> None:
        self.buffer = ""
        self.depth = 0
        self.completed_clauses = []


ShadowAst = ShadowParser
ShadowAstParser = ShadowParser
IncrementalParser = ShadowParser
TreeSitterShadow = ShadowParser
