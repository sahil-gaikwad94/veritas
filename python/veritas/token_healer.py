"""BPE boundary healing and Jump-Forward splice construction.

Constrained decoding naively appends a deterministic string (``", "sharpe_ratio": 0.6437"``)
to a token prefix.  At the byte level that is a *different* tokenization than the model
would have produced, so the KV-cache becomes inconsistent with the emitted text.  Token
healing backs up a small number of trailing tokens, re-tokenizes the joined byte span with
the model's own tokenizer, and reports the token ids to prefill in one batch.

The implementation is tokenizer-agnostic (``encode(list[str]) -> list[int]`` /
``decode(list[int]) -> str``); :class:`ByteTokenizer` provides a dependency-free default
used by tests and by the reference decode engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> List[int]:  # pragma: no cover - protocol
        ...

    def decode(self, ids: Sequence[int]) -> str:  # pragma: no cover - protocol
        ...


@dataclass
class JumpForwardPrefill:
    """Instruction for a batched KV-cache extension.

    ``backup_tokens`` tokens are rewound from the KV cache; ``new_token_ids`` are then
    appended.  Because healing re-tokenizes the rewound tail together with the splice,
    ``new_token_ids`` normally decodes to ``tail + splice_string`` (not the splice alone),
    while the *net new text* is exactly ``splice_string``.
    """

    backup_tokens: int = 0
    new_token_ids: List[Any] = field(default_factory=list)
    splice_string: str = ""
    healed_text: str = ""
    offset: int = 0

    @property
    def num_new_tokens(self) -> int:
        return len(self.new_token_ids)

    def __repr__(self) -> str:
        return (
            f"JumpForwardPrefill(backup={self.backup_tokens}, "
            f"new_tokens={len(self.new_token_ids)}, offset={self.offset})"
        )


class ByteTokenizer:
    """UTF-8 byte tokenizer: exact, dependency-free, and stable for healing tests."""

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        raw = bytes(int(i) & 0xFF for i in ids)
        return raw.decode("utf-8", "replace")


class TokenHealer:
    def __init__(self, tokenizer: Optional[Any] = None, max_backup: int = 4):
        self.tokenizer = tokenizer
        self.max_backup = max(1, int(max_backup))

    # -- helpers ---------------------------------------------------------
    def _tok(self, tokenizer: Any = None) -> Any:
        tok = tokenizer if tokenizer is not None else self.tokenizer
        if tok is None:
            return ByteTokenizer()
        encode = getattr(tok, "encode", None)
        decode = getattr(tok, "decode", None)
        if not callable(encode) or not callable(decode):
            raise TypeError("tokenizer must implement encode(text) and decode(ids)")
        return tok

    def build_splice(self, parts: Sequence[str]) -> str:
        return "".join(parts)

    # -- core ------------------------------------------------------------
    def heal(
        self,
        prefix_text: str,
        splice_string: str,
        tokenizer: Any = None,
        *,
        max_backup: Optional[int] = None,
    ) -> JumpForwardPrefill:
        """Find the minimal token backup that makes ``splice_string`` tokenize exactly.

        Returns token ids that extend the KV cache after ``backup_tokens`` are rewound.
        If no exact boundary exists within the budget, the splice is encoded directly and
        ``backup_tokens == 0`` — callers should treat that as "text-level splice".
        """

        tok = self._tok(tokenizer)
        if not splice_string:
            return JumpForwardPrefill(0, [], "", prefix_text, len(prefix_text))
        prefix_ids = list(tok.encode(prefix_text))
        full_text = prefix_text + splice_string
        full_ids = list(tok.encode(full_text))
        budget = self.max_backup if max_backup is None else max(0, int(max_backup))

        for backup in range(0, min(budget, len(prefix_ids)) + 1):
            if backup == 0:
                cut_text = prefix_text
                kept_ids = prefix_ids
            else:
                cut_text = tok.decode(prefix_ids[:-backup])
                if not prefix_text.startswith(cut_text):
                    continue
                kept_ids = list(tok.encode(cut_text))
                # The retained prefix must tokenize identically to avoid another mismatch.
                if prefix_ids[: len(kept_ids)] != kept_ids:
                    continue
            # Re-tokenize the *whole* text; the rewound tail is re-emitted together with
            # the splice so the model's tokenizer controls the boundary, not us.
            if full_ids[: len(kept_ids)] != kept_ids:
                continue
            new_ids = full_ids[len(kept_ids):]
            expected_text = prefix_text[len(cut_text):] + splice_string
            if tok.decode(new_ids) == expected_text:
                return JumpForwardPrefill(
                    backup_tokens=backup,
                    new_token_ids=new_ids,
                    splice_string=splice_string,
                    healed_text=full_text,
                    offset=len(cut_text),
                )

        # No exact boundary: still emit a usable prefill, but never claim healing.
        return JumpForwardPrefill(
            backup_tokens=0,
            new_token_ids=list(tok.encode(splice_string)),
            splice_string=splice_string,
            healed_text=full_text,
            offset=len(prefix_text),
        )

    def heal_and_splice(self, prefix_tokens: List[Any], splice_string: str) -> JumpForwardPrefill:
        """Character-level compatibility API (no tokenizer required)."""

        if self.tokenizer is not None:
            prefix_text = self.tokenizer.decode(prefix_tokens)
            return self.heal(prefix_text, splice_string, self.tokenizer)
        backup = 0
        if prefix_tokens and isinstance(splice_string, str) and splice_string:
            last = prefix_tokens[-1]
            if isinstance(last, str) and last and splice_string.startswith(last):
                backup = 1
        return JumpForwardPrefill(
            backup_tokens=backup,
            new_token_ids=list(splice_string),
            splice_string=splice_string,
            healed_text="".join(str(t) for t in prefix_tokens) + splice_string,
            offset=len(prefix_tokens),
        )

    def compute_token_healed_splice(
        self, zero_dof_values: List[Any], last_token_id: Any = None
    ) -> JumpForwardPrefill:
        splice = "".join(str(v) for v in zero_dof_values)
        prefix = [last_token_id] if last_token_id is not None else []
        return self.heal_and_splice(prefix, splice)


# aliases kept for the public API surface
TokenHealerEngine = TokenHealer
BpeTokenHealer = TokenHealer
JumpForwardSplicer = TokenHealer
TokenHealingSplicer = TokenHealer

__all__ = [
    "BpeTokenHealer",
    "ByteTokenizer",
    "JumpForwardPrefill",
    "JumpForwardSplicer",
    "TokenHealer",
    "TokenHealerEngine",
    "TokenHealingSplicer",
    "TokenizerProtocol",
]
