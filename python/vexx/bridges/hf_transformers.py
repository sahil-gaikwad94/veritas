"""HuggingFace ``transformers`` bridge.

``VexxLogitsProcessor`` is a drop-in HF ``LogitsProcessor``: it decodes the current
prefix, computes the exact JSON token mask and sets every inadmissible logit to ``-inf``.
The vocabulary is derived from the tokenizer's ``get_vocab`` mapping (or supplied
explicitly), so the mask is exact for the model actually being sampled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..json_pda import JSONSyntaxError, TokenMasker


def _report_ok(report: Any) -> bool:
    if report is None:
        return True
    ok = getattr(report, "ok", None)
    if ok is not None:
        return bool(ok)
    return getattr(report, "status", "") == "PASS"


class VexxLogitsProcessor:
    def __init__(
        self,
        tokenizer: Any = None,
        spec: Any = None,
        vocabulary: Sequence[str] | None = None,
        eos_token_id: int | None = None,
        *,
        cache_size: int = 8192,
        prompt_len: int = 0,
    ):
        self.tokenizer = tokenizer
        self.spec = spec
        self.prompt_len = int(prompt_len)
        self.vocabulary = list(vocabulary) if vocabulary is not None else self._vocabulary_from_tokenizer(tokenizer)
        self.eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
        schema = None
        if spec is not None:
            vir = getattr(spec, "spec", spec)
            schema = getattr(vir, "model_schema", None)
        self.masker = (
            TokenMasker(self.vocabulary, eos_token=self.eos_token_id, cache_size=cache_size, schema=schema)
            if self.vocabulary
            else None
        )

    @staticmethod
    def _vocabulary_from_tokenizer(tokenizer: Any) -> list[str] | None:
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
            if not (0 <= index < size):
                continue
            text = None
            if hasattr(tokenizer, "decode"):
                try:
                    text = tokenizer.decode([index], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                except TypeError:
                    try:
                        text = tokenizer.decode([index])
                    except Exception:
                        text = None
                except Exception:
                    text = None
            if text is None and hasattr(tokenizer, "convert_ids_to_tokens") and hasattr(tokenizer, "convert_tokens_to_string"):
                try:
                    text = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(index)])
                except Exception:
                    text = None
            out[index] = str(text if text is not None else token)
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

    def _rows(self, scores: Any) -> list[Any]:
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
        batched = bool(ids) and isinstance(ids[0], (list, tuple))
        rows = self._rows(scores)
        if batched:
            if len(ids) != len(rows):
                raise ValueError("input_ids batch size does not match scores batch size")
            for row_ids, row in zip(ids, rows):
                prefix = self.decode(list(row_ids)[self.prompt_len:])
                self._apply_row(prefix, row)
        else:
            prefix = self.decode(list(ids)[self.prompt_len:])
            for row in rows:
                self._apply_row(prefix, row)
        return scores

    def _apply_row(self, prefix: str, row: Any) -> None:
        try:
            self.masker.apply(prefix, row)
        except JSONSyntaxError:
            # Dead-end prefix: mask every continuation, including EOS.
            if isinstance(row, dict):
                for key in row:
                    row[key] = float("-inf")
            else:
                for i in range(len(row)):
                    row[i] = float("-inf")

    @classmethod
    def from_spec(cls, spec: Any, tokenizer: Any = None, **kwargs: Any) -> VexxLogitsProcessor:
        return cls(tokenizer=tokenizer, spec=spec, **kwargs)


class HFTransformersBridge:
    def __init__(self, model: Any = None, spec: Any = None, tokenizer: Any = None):
        self.model = model
        self.spec = spec
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.logits_processor = VexxLogitsProcessor(self.tokenizer, spec)

    def patch(self, model: Any = None) -> Any:
        target = model if model is not None else self.model
        try:
            target.vexx_logits_processor = self.logits_processor
        except Exception:
            pass
        return target

    def generate_kwargs(self) -> dict[str, Any]:
        return {"logits_processor": [self.logits_processor]}

    def verify(self, text: str) -> Any:
        return self.logits_processor.verify(text)


__all__ = ["HFTransformersBridge", "VexxLogitsProcessor", "JSONSyntaxError"]
