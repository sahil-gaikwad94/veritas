"""Phase 3: BPE token healing and Jump-Forward prefill splice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class JumpForwardPrefill:
    backup_tokens: int = 0
    new_token_ids: List[Any] = field(default_factory=list)
    splice_string: str = ""


class TokenHealer:
    def __init__(self, tokenizer: Any = None):
        self.tokenizer = tokenizer

    def build_splice(self, parts: List[str]) -> str:
        return "".join(parts)

    def heal_and_splice(self, prefix_tokens: List[Any], splice_string: str) -> JumpForwardPrefill:
        # Simplified: no actual BPE; return splice as new tokens list (chars)
        backup = 0
        # if last token is partial prefix of splice, back up 1
        if prefix_tokens and isinstance(splice_string, str) and splice_string:
            # heuristic: if last token string is prefix of splice_string, backup
            last = prefix_tokens[-1]
            if isinstance(last, str) and splice_string.startswith(last):
                backup = 1
        new_ids = list(splice_string)
        return JumpForwardPrefill(backup_tokens=backup, new_token_ids=new_ids, splice_string=splice_string)

    def compute_token_healed_splice(self, zero_dof_values: List[Any], last_token_id: Any = None) -> JumpForwardPrefill:
        splice = ""
        for v in zero_dof_values:
            splice += str(v)
        return self.heal_and_splice([last_token_id] if last_token_id is not None else [], splice)


# aliases
TokenHealerEngine = TokenHealer
BpeTokenHealer = TokenHealer
JumpForwardSplicer = TokenHealer
TokenHealingSplicer = TokenHealer
