"""llama.cpp bridge (shim)."""

from __future__ import annotations

class LlamaCppBridge:
    def __init__(self, model=None, spec=None):
        self.model = model
        self.spec = spec

def patch_llamacpp(model, spec=None):
    return model
