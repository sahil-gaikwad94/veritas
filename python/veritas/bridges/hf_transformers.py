"""HuggingFace ``transformers`` bridge.

``VeritasLogitsProcessor`` is a drop-in HF ``LogitsProcessor``: it decodes the current
prefix, computes the exact JSON token mask and sets every inadmissible logit to ``-inf``.
The vocabulary is derived from the tokenizer's ``get_vocab`` mapping (or supplied
explicitly), so the mask is exact for the model actually being sampled.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..json_pda import TokenMasker, JSONSyntaxError


def _report_ok(report: Any) -> bool:
    if report is None:
        return True
    ok = getattr(report, "ok", None)
    if ok is not None:
        return bool(ok)
    return getattr(report, "status", "") == "PASS"


class VeritasLogitsProcessor:
    def __init__(
        self,
        tokenizer: Any = None,
        spec: Any = None,
        vocabulary: Optional[Sequence[str]] = None,
        eos_token_id: Optional[int] = None,
        *,
        cache_size: int = 8192,
    ):
        self.tokenizer = tokenizer
        self.spec = spec
        self.vocabulary = list(vocabulary) if vocabulary is not None else self._vocabulary_from_tokenizer(tokenizer)
        self.eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
        self.masker = (
            TokenMasker(self.vocabulary, eos_token=self.eos_token_id, cache_size=cache_size)
            if self.vocabulary
            else None
        )

    @staticmethod
    def _vocabulary_from_tokenizer(tokenizer: Any) -> Optional[List[str]]:
        if tokenizer is None:
            return None
        vocab = None
        if hasattr(tokenizer, "get_vocab"):
            try:
                vocab = tokenizer.get_vocab()
            except Exception:
                vocab = None
        if vocab is None:
            vocab = getattr(tokenizer, "vocab", None)
        if not isinstance(vocab, Mapping) or not vocab:
            return None
        size = max(int(index) for index in vocab.values()) + 1
        out = [""] * size
        for token, index in vocab.items():
            index = int(index)
            if 0 <= index < size and not out[index]:
                out[index] = str(token)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(list(ids))
        if self.vocabulary:
            return "".join(self.vocabulary[i] for i in ids if 0 <= i < len(self.vocabulary))
        return ""

    def verify(self, text: str) -> Any:
        if self.spec is None:
            return None
        from ..instance import verify_json

        return verify_json(self.spec, text)

    def _rows(self, scores: Any) -> List[Any]:
        ndim = getattr(scores, "ndim", None)
        if ndim is not None:
            return [scores[i] for i in range(scores.shape[0])] if ndim > 1 else [scores]
        if isinstance(scores, (list, tuple)) and scores and isinstance(scores[0], (list, tuple)):
            return list(scores)
        return [scores]

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.masker is None:
            return scores
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[-1]
        prefix = self.decode(ids)
        rows = self._rows(scores)
        for row in rows:
            self.masker.apply(prefix, row)
        return scores

    @classmethod
    def from_spec(cls, spec: Any, tokenizer: Any = None, **kwargs: Any) -> "VeritasLogitsProcessor":
        return cls(tokenizer=tokenizer, spec=spec, **kwargs)


class HFTransformersBridge:
    def __init__(self, model: Any = None, spec: Any = None, tokenizer: Any = None):
        self.model = model
        self.spec = spec
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.logits_processor = VeritasLogitsProcessor(self.tokenizer, spec)

    def patch(self, model: Any = None) -> Any:
        target = model if model is not None else self.model
        try:
            setattr(target, "veritas_logits_processor", self.logits_processor)
        except Exception:
            pass
        return target

    def generate_kwargs(self) -> Dict[str, Any]:
        return {"logits_processor": [self.logits_processor]}

    def verify(self, text: str) -> Any:
        return self.logits_processor.verify(text)


__all__ = ["HFTransformersBridge", "VeritasLogitsProcessor", "JSONSyntaxError"]
