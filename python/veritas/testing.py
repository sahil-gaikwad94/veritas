"""Deterministic reference models for exercising the decode engine without a GPU.

These are *test doubles*, not language models: they expose
``next_logits(prefix_text) -> Mapping[str, float]`` with known preferences so the effect
of grammar masks, interval masks, residual accumulators and 0-DoF splices can be measured
exactly.  Real models are adapted through :mod:`veritas.bridges`.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, Mapping, Optional


class MockCharModel:
    """Prefers a configurable digit order, so unconstrained decoding would fail."""

    name = "veritas.testing.MockCharModel"

    def __init__(
        self,
        digit_order: str = "9876543210",
        *,
        stop_logit: float = 6.0,
        char_base: float = 5.0,
        punctuation_logit: float = 2.0,
        other_logit: float = -8.0,
        noise: float = 0.0,
        seed: int = 0,
    ):
        self.digit_order = digit_order
        self.stop_logit = stop_logit
        self.char_base = char_base
        self.punctuation_logit = punctuation_logit
        self.other_logit = other_logit
        self.noise = noise
        self._rng = random.Random(seed)
        self.seen_prefixes: list = []

    def next_logits(self, prefix: str) -> Dict[str, float]:
        self.seen_prefixes.append(prefix)
        scores: Dict[str, float] = {}
        for rank, digit in enumerate(self.digit_order):
            scores[digit] = self.char_base - rank * 0.05
        scores["."] = self.punctuation_logit
        scores["-"] = self.punctuation_logit - 1.0
        scores["<stop>"] = self.stop_logit
        for code in range(0x20, 0x7F):
            ch = chr(code)
            scores.setdefault(ch, self.other_logit)
        if self.noise:
            for ch in scores:
                scores[ch] += self._rng.uniform(-self.noise, self.noise)
        return scores


class RandomCharModel:
    """Pseudo-random but *deterministic per prefix* logits (for fuzz/property tests)."""

    name = "veritas.testing.RandomCharModel"

    def __init__(self, seed: int = 0, stop_logit: float = 0.0):
        self.seed = seed
        self.stop_logit = stop_logit

    def next_logits(self, prefix: str) -> Dict[str, float]:
        digest = hashlib.sha256(f"{self.seed}:{prefix}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        scores = {chr(code): rng.uniform(-2.0, 2.0) for code in range(0x20, 0x7F)}
        scores["<stop>"] = self.stop_logit
        return scores


class StructuredReasoningModel(MockCharModel):
    """Emits a ``reasoning`` scratchpad note, mirroring the CRANE idea for tests."""

    name = "veritas.testing.StructuredReasoningModel"

    def __init__(self, reasoning: str = "let me reason about the constraints", **kwargs):
        super().__init__(**kwargs)
        self.reasoning = reasoning


def mock_model(**kwargs) -> MockCharModel:
    return MockCharModel(**kwargs)


__all__ = [
    "MockCharModel",
    "RandomCharModel",
    "StructuredReasoningModel",
    "mock_model",
]
