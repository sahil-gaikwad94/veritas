"""Tier-0 public PDA facade.

The real recognizer lives in :mod:`veritas.json_pda`; this module exposes the historical
``PushdownAutomaton`` / ``AutomatonEngine`` API and a token-bitmask path for callers that
work with a fixed vocabulary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from .json_pda import JSONPushdownAutomaton, TokenMasker


class PushdownAutomaton:
    """Byte/char-level JSON PDA with optional vocabulary bitmasks."""

    def __init__(self, vocab_size: int = 0, vocabulary: Optional[Sequence[str]] = None,
                 language: str = "json"):
        self.vocab_size = vocab_size
        self.vocabulary = list(vocabulary) if vocabulary else None
        self.language = language
        self.machine = JSONPushdownAutomaton()
        self.masker = (
            TokenMasker(self.vocabulary, eos_token=self.vocab_size - 1)
            if self.vocabulary and self.vocab_size > 0
            else None
        )

    @property
    def state(self) -> str:
        return self.machine.state

    @property
    def stack(self) -> List[Dict[str, Any]]:
        return self.machine.stack

    def feed(self, text: str) -> bool:
        return self.machine.feed(text)

    def get_valid_token_bitmask(self, state: Any = None, prefix: Optional[str] = None) -> int:
        """Return a bitmask; with no vocabulary every bit is valid (legacy behaviour)."""

        if not self.vocabulary or self.vocab_size <= 0:
            return -1 if self.vocab_size <= 0 else (1 << self.vocab_size) - 1
        text = prefix if prefix is not None else self.machine.text
        mask = self.masker.valid_mask(text) if self.masker is not None else []
        bits = 0
        for index, valid in enumerate(mask[: self.vocab_size]):
            if valid:
                bits |= 1 << index
        return bits

    def anchor_prefix(self, prefix: str) -> None:
        self.machine = JSONPushdownAutomaton()
        if prefix:
            self.machine.feed(prefix)

    def step(self, token_id: int) -> str:
        text = self.vocabulary[token_id] if self.vocabulary and token_id < len(self.vocabulary) else ""
        if text:
            self.machine.feed(text)
        return self.machine.state

    def accepts(self, token_ids: Optional[Sequence[int]] = None) -> bool:
        if token_ids is None:
            return self.machine.accepts()
        target = JSONPushdownAutomaton()
        if not target.feed(self.machine.text):
            return False
        for token_id in token_ids:
            if self.vocabulary is None or token_id >= len(self.vocabulary):
                return False
            if not target.feed(self.vocabulary[token_id]):
                return False
        return target.accepts()

    def allowed_chars(self) -> Set[str]:
        return self.machine.allowed_chars()


# aliases retained by the public API
PDA = PushdownAutomaton
AutomatonEngine = PushdownAutomaton
TokenBitmasker = PushdownAutomaton
BytePDA = PushdownAutomaton

__all__ = [
    "AutomatonEngine",
    "BytePDA",
    "PDA",
    "PushdownAutomaton",
    "TokenBitmasker",
]
