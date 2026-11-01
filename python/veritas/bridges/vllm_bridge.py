"""vLLM bridge (shim)."""

from __future__ import annotations

class VeritasVLLMBridge:
    def __init__(self, model=None, spec=None):
        self.model = model
        self.spec = spec
        self.logits_processor = None

    def patch(self, engine):
        return engine


def patch_vllm(engine, spec=None):
    return engine


class VLLMLogitsProcessor:
    def __call__(self, input_ids, scores):
        return scores
