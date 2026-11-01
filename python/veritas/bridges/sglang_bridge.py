"""SGLang bridge — the same exact processor, attached to an SGLang runtime."""

from __future__ import annotations

from typing import Any

from .hf_transformers import VeritasLogitsProcessor


class SGLangLogitsProcessor(VeritasLogitsProcessor):
    pass


class SGLangBridge:
    def __init__(self, model: Any = None, spec: Any = None, tokenizer: Any = None):
        self.model = model
        self.spec = spec
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.logits_processor = SGLangLogitsProcessor(self.tokenizer, spec)

    def patch(self, engine: Any = None) -> Any:
        target = engine if engine is not None else self.model
        if target is None:
            return target
        try:
            existing = getattr(target, "logits_processor", None)
            if isinstance(existing, list):
                existing.append(self.logits_processor)
            else:
                setattr(target, "veritas_logits_processor", self.logits_processor)
            setattr(target, "veritas_spec", self.spec)
        except Exception:
            pass
        return target


def patch_sglang(engine: Any, spec: Any = None, tokenizer: Any = None) -> Any:
    return SGLangBridge(engine, spec, tokenizer).patch(engine)


__all__ = ["SGLangBridge", "SGLangLogitsProcessor", "patch_sglang"]
