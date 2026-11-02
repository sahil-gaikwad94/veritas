"""llama.cpp bridge.

llama.cpp exposes grammar through GBNF rather than arbitrary logits callbacks.  This
adapter attaches the Python processor when the host supports ``set_logits_processor`` and
exposes a GBNF document for hosts that only accept grammar strings.  When a Veritas spec
is provided, a schema-specific GBNF is generated to enforce the spec constraints.
"""

from __future__ import annotations

from typing import Any

from .gbnf import generate_gbnf_from_schema
from .hf_transformers import VeritasLogitsProcessor

GBNF_JSON = r'''
root   ::= value
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (member (ws "," ws member)*)? ws "}"
member ::= string ws ":" ws value
array  ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= [ \t\n\r]*
'''


class LlamaCppBridge:
    def __init__(self, model: Any = None, spec: Any = None, tokenizer: Any = None):
        self.model = model
        self.spec = spec
        self.logits_processor = VeritasLogitsProcessor(tokenizer, spec)
        # Generate schema-specific GBNF if spec is provided, otherwise use generic JSON
        if spec is not None:
            self.grammar = generate_gbnf_from_schema(spec)
        else:
            self.grammar = GBNF_JSON

    def patch(self, model: Any = None) -> Any:
        target = model if model is not None else self.model
        if target is None:
            return target
        setter = getattr(target, "set_logits_processor", None)
        if callable(setter):
            setter(self.logits_processor)
            target.veritas_logits_processor = self.logits_processor
        elif hasattr(target, "grammar"):
            if self.spec is not None:
                raise TypeError("llama.cpp host only exposes grammar; cannot enforce spec without set_logits_processor")
            target.grammar = self.grammar
        else:
            raise TypeError("llama.cpp host exposes neither set_logits_processor nor grammar; pass bridge.grammar to Llama(grammar=...)")
        target.veritas_grammar = self.grammar
        return target


def patch_llamacpp(model: Any, spec: Any = None, tokenizer: Any = None) -> Any:
    return LlamaCppBridge(model, spec, tokenizer).patch(model)


__all__ = ["GBNF_JSON", "LlamaCppBridge", "patch_llamacpp", "generate_gbnf_from_schema"]
