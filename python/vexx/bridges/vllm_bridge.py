"""vLLM bridge — attach an exact JSON grammar processor to a vLLM engine."""

from __future__ import annotations

from typing import Any

from .hf_transformers import VexxLogitsProcessor


class VLLMLogitsProcessor(VexxLogitsProcessor):
    """Same exact masking semantics, named for vLLM's ``logits_processors`` list."""

    def __init__(self, tokenizer: Any = None, spec: Any = None, vocabulary=None, eos_token_id=None, **kwargs):
        super().__init__(tokenizer=tokenizer, spec=spec, vocabulary=vocabulary,
                         eos_token_id=eos_token_id, **kwargs)


class VexxVLLMBridge:
    def __init__(self, model: Any = None, spec: Any = None, tokenizer: Any = None):
        self.model = model
        self.spec = spec
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.logits_processor = VLLMLogitsProcessor(self.tokenizer, spec)

    def sampling_params(self) -> dict:
        """Pass these to vLLM ``SamplingParams`` / ``LLM.generate`` to enforce masking."""
        return {"logits_processors": [self.logits_processor]}

    def patch(self, engine: Any = None) -> Any:
        target = engine if engine is not None else self.model
        if target is None:
            return target
        existing = getattr(target, "logits_processors", None)
        legacy = getattr(target, "logits_processor", None)
        if isinstance(existing, list):
            if self.logits_processor not in existing:
                existing.append(self.logits_processor)
        elif isinstance(legacy, list):
            if self.logits_processor not in legacy:
                legacy.append(self.logits_processor)
        else:
            raise TypeError("vLLM host exposes no logits_processors list; pass bridge.sampling_params() to SamplingParams")
        target.vexx_spec = self.spec
        return target


def patch_vllm(engine: Any, spec: Any = None, tokenizer: Any = None) -> Any:
    return VexxVLLMBridge(engine, spec, tokenizer).patch(engine)


__all__ = ["VLLMLogitsProcessor", "VexxVLLMBridge", "patch_vllm"]
