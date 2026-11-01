"""Diffusion LLM bridge (shim)."""

from __future__ import annotations

class DiffusionBridge:
    def __init__(self, model=None, spec=None):
        self.model = model
        self.spec = spec

def gumbel_max_sample(logits, temperature=1.0):
    return logits
