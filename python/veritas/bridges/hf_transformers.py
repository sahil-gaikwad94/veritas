"""HuggingFace transformers bridge (shim)."""

from __future__ import annotations

class VeritasLogitsProcessor:
    def __call__(self, input_ids, scores):
        return scores

    def __post_init__(self):
        pass

class HFTransformersBridge:
    def __init__(self, model=None, spec=None):
        self.model = model
        self.spec = spec
