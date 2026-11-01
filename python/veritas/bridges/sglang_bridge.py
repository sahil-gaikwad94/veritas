"""SGLang bridge (shim)."""

from __future__ import annotations

class SGLangBridge:
    def __init__(self, model=None, spec=None):
        self.model = model
        self.spec = spec

def patch_sglang(engine, spec=None):
    return engine
