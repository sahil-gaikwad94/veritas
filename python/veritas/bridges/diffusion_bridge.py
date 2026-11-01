"""Diffusion language-model bridge: Gumbel-max sampling and confidence unmasking."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


def gumbel_max_sample(logits: Any, temperature: float = 1.0, rng: random.Random | None = None):
    """Sample an index/key from logits using the Gumbel-max trick.

    Mapping logits return their key; sequences return an integer index.  ``temperature``
    scales the logits (``0`` means deterministic argmax).
    """

    rng = rng or random
    if isinstance(logits, Mapping):
        items = [(key, float(value)) for key, value in logits.items() if value == value]
        if not items:
            raise ValueError("logits are empty")
        if temperature <= 0:
            return max(items, key=lambda item: item[1])[0]
        best_key, best_score = None, -math.inf
        for key, value in items:
            if value == -math.inf:
                continue
            score = value / temperature - math.log(-math.log(max(rng.random(), 1e-12)))
            if score > best_score:
                best_key, best_score = key, score
        return best_key
    values = [float(value) for value in logits]
    if not values:
        raise ValueError("logits are empty")
    if temperature <= 0:
        return max(range(len(values)), key=lambda index: values[index])
    best_index, best_score = 0, -math.inf
    for index, value in enumerate(values):
        if value == -math.inf:
            continue
        score = value / temperature - math.log(-math.log(max(rng.random(), 1e-12)))
        if score > best_score:
            best_index, best_score = index, score
    return best_index


def unmask_order(confidences: Sequence[float]) -> list:
    """Indices sorted from most to least confident (stable on ties)."""

    return [index for index, _ in sorted(enumerate(confidences), key=lambda pair: (-pair[1], pair[0]))]


class DiffusionBridge:
    def __init__(self, model: Any = None, spec: Any = None, *, seed: int | None = None):
        self.model = model
        self.spec = spec
        self.rng = random.Random(seed)

    def sample_token(self, logits: Any, temperature: float = 1.0) -> Any:
        return gumbel_max_sample(logits, temperature, self.rng)

    def step(self, token_logits: Sequence[Any], mask: Sequence[bool], temperature: float = 1.0) -> int | None:
        """Choose the masked position with the highest sampled confidence and its token."""

        candidates = [i for i, masked in enumerate(mask) if masked]
        if not candidates:
            return None
        positions = [gumbel_max_sample(token_logits[i], temperature, self.rng) for i in candidates]
        # Gumbel-max over per-position max log-probabilities selects an unmask order.
        scores = []
        for position, token in zip(candidates, positions):
            values = token_logits[position]
            if isinstance(values, Mapping):
                scores.append(float(values.get(token, -math.inf)))
            else:
                scores.append(float(values[token]))
        return candidates[max(range(len(candidates)), key=lambda i: scores[i])]


gumbel_max = gumbel_max_sample

__all__ = ["DiffusionBridge", "gumbel_max", "gumbel_max_sample", "unmask_order"]
