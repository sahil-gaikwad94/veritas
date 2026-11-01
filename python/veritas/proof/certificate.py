"""Proof certificate and VerifiedResult (shim)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProofCertificate:
    smt2: str = ""
    sha256: str = ""


@dataclass
class VerifiedResult:
    value: Any = None
    reasoning: str = ""
    proof: Optional[ProofCertificate] = None
    telemetry: Any = None


def generate(model=None, prompt=None, spec=None, return_proof=False, **kwargs):
    # Minimal shim: run preflight if spec provided
    if spec is not None and hasattr(spec, "preflight") and spec.preflight is not None:
        if not getattr(spec.preflight, "is_sat", True):
            raise RuntimeError("Infeasible spec")
    # Try to instantiate model
    value = None
    try:
        if spec is not None and getattr(spec, "model", None) is not None:
            value = spec.model()
    except Exception:
        value = None
    proof = ProofCertificate() if return_proof else None
    return VerifiedResult(value=value, reasoning="", proof=proof, telemetry={})
