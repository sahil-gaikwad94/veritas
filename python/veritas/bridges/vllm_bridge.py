"""vLLM bridge — attach an exact JSON grammar processor to a vLLM engine."""

from __future__ import annotations

from typing import Any, List, Optional

from .hf_transformers import VeritasLogitsProcessor


class VLLMLogitsProcessor(VeritasLogitsProcessor):
    """Same exact masking semantics, named for vLLM's ``logits_processors`` list."""

    def __init__(self, tokenizer: Any = None, spec: Any = None, vocabulary=None, eos_token_id=None, **kwargs):
        super().__init__(tokenizer=tokenizer, spec=spec, vocabulary=vocabulary,
                         eos_token_id=eos_token_id, **kwargs)


class VeritasVLLMBridge:
    def __init__(self, model: Any = None, spec: Any = None, tokenizer: Any = None):
        self.model = model
        self.spec = spec
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.logits_processor = VLLMLogitsProcessor(self.tokenizer, spec)

    def patch(self, engine: Any = None) -> Any:
        target = engine if engine is not None else self.model
        if target is None:
            return target
        try:
            existing = getattr(target, "logits_processor", None)
            if existing is None:
                setattr(target, "logits_processor", [self.logits_processor])
            elif isinstance(existing, list):
                if self.logits_processor not in existing:
                    existing.append(self.logits_processor)
            else:
                setattr(target, "veritas_processor", self.logits_processor)
            setattr(target, "veritas_spec", self.spec)
        except Exception:
            pass
        return target


def patch_vllm(engine: Any, spec: Any = None, tokenizer: Any = None) -> Any:
    return VeritasVLLMBridge(engine, spec, tokenizer).patch(engine)


__all__ = ["VLLMLogitsProcessor", "VeritasVLLMBridge", "patch_vllm"]
