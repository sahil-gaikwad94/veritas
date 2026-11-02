"""Optional native accelerator: ``crates/veritas-py`` (module name ``veritas_py``).

The pure-Python engine stays authoritative. This module only *prefers* the native answer
when the compiled module is importable, and every wrapper falls back to the Python
implementation on any error. That ordering matters:

* an install without a compiler (or on an unsupported platform) behaves exactly like the
  pure-Python engine, so the test suite and the semantics never depend on the wheel;
* a native bug can only make a mask narrower or a verdict "unknown", never silently accept a
  document the Python automaton rejects — ``tests/test_native_backend.py`` pins the parity.

``veritas.doctor()`` reports :func:`available` and :func:`load_error`, because "the extension
did not load" is the single most common support question for a package with an optional
native path.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "DISABLE_ENV",
    "available",
    "disabled_by_env",
    "load_error",
    "info",
    "wire_version",
    "prefix_status",
    "canonical_document",
    "mask",
    "mask_or_none",
    "numeric_prefix",
    "numeric_prefix_or_none",
    "contract",
    "solve_linear",
]

_NATIVE: Any | None = None
_LOAD_ERROR: str | None = None

#: Set to a truthy value to force the pure-Python engine. This is a debugging and measurement
#: switch, not a feature flag: an operator who suspects the accelerator, or a benchmark that
#: wants a controlled A/B of the *same* call sites, needs a way to pin the reference path
#: without uninstalling anything.
DISABLE_ENV = "VERITAS_NO_NATIVE"

try:  # pragma: no cover - exercised by the wheels job, not by the pure-Python suite
    _NATIVE = importlib.import_module("veritas_py")
except Exception as exc:  # pragma: no cover
    _LOAD_ERROR = f"{type(exc).__name__}: {exc}"


def disabled_by_env() -> bool:
    """True when :data:`DISABLE_ENV` asks for the pure-Python engine."""
    value = os.environ.get(DISABLE_ENV, "")
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def available() -> bool:
    """True when the compiled extension is loaded, enabled and its wire format matches."""
    if disabled_by_env():
        return False
    if _NATIVE is None:
        return False
    expected = 1
    try:
        return int(getattr(_NATIVE, "WIRE_VERSION", expected)) == expected
    except Exception:  # pragma: no cover
        return False


def load_error() -> str | None:
    """Why the extension is unavailable, or ``None`` when it loaded."""
    if disabled_by_env():
        return f"disabled by {DISABLE_ENV}"
    return _LOAD_ERROR


def info() -> dict[str, Any]:
    """Engine description, with the pure-Python fallback reported explicitly."""
    if not available():
        return {
            "name": "python",
            "version": "pure",
            "wire_version": None,
            "native_available": False,
            "load_error": load_error(),
        }
    import json

    try:
        described = json.loads(_NATIVE.engine_info())
    except Exception:  # pragma: no cover
        described = {"name": "veritas_py"}
    described["native_available"] = True
    described["load_error"] = None
    return described


def wire_version() -> int | None:
    if not available():
        return None
    return int(getattr(_NATIVE, "WIRE_VERSION", 1))


def prefix_status(text: str) -> str:
    """``"complete"``, ``"incomplete"`` or ``"invalid"``, matching ``json_pda``."""
    if available():
        try:
            return str(_NATIVE.prefix_status(text))
        except Exception:  # pragma: no cover - fall back, never raise from a probe
            pass
    from .json_pda import JSONPushdownAutomaton

    machine = JSONPushdownAutomaton()
    if not machine.feed(text):
        return "invalid"
    probe = machine.clone()
    probe.finish()
    return "complete" if probe.accepts() else "incomplete"


def canonical_document(text: str) -> str | None:
    """Canonical rendering of a valid document, or ``None``."""
    if available():
        try:
            return _NATIVE.canonical_document(text)
        except Exception:  # pragma: no cover
            pass
    from .json_pda import JSONSyntaxError, parse_json

    try:
        value = parse_json(text)
    except JSONSyntaxError:
        return None
    import json

    return json.dumps(value, ensure_ascii=False)


def mask_or_none(
    prefix: str,
    vocabulary: Sequence[str],
    eos_token: int | None = None,
) -> list[bool] | None:
    """Native mask, or ``None`` when it cannot be trusted for this call.

    The Python rules are preserved exactly: an empty token never makes progress, and the
    end-of-sequence token is legal only once a pending number or literal can be closed.
    """
    if not available():
        return None
    try:
        allowed = _NATIVE.prefix_status(prefix)
        if allowed == "invalid":
            return None
        indices = set(_NATIVE.allowed_tokens(prefix, list(vocabulary), None))
        mask = [bool(text) and index in indices for index, text in enumerate(vocabulary)]
        if eos_token is not None:
            while len(mask) <= eos_token:
                mask.append(False)
            mask[eos_token] = allowed == "complete"
        return mask
    except Exception:  # pragma: no cover - any native failure falls back to Python
        return None


def mask(
    prefix: str,
    vocabulary: Sequence[str],
    eos_token: int | None = None,
) -> list[bool]:
    """Native mask with a pure-Python fallback (never ``None``)."""
    native_mask = mask_or_none(prefix, vocabulary, eos_token)
    if native_mask is not None:
        return native_mask
    from .json_pda import TokenMasker

    return TokenMasker(vocabulary, eos_token=eos_token).valid_mask(prefix)


def numeric_prefix_or_none(raw_candidate: str, domain_bounds, allow_exp: bool = False):
    """Native numeric prefix interval, or ``None`` when the extension cannot answer."""
    from .interval import Interval

    if not available():
        return None
    try:
        lo, hi = _NATIVE.numeric_prefix(
            str(raw_candidate), float(domain_bounds.lo), float(domain_bounds.hi), allow_exp
        )
    except Exception:  # pragma: no cover - fall back to Python
        return None
    return Interval(lo, hi)


def numeric_prefix(raw_candidate: str, domain_bounds, allow_exp: bool = False):
    """Interval of reachable values for a partially typed number (with fallback)."""
    from .interval import Interval

    span = numeric_prefix_or_none(raw_candidate, domain_bounds, allow_exp)
    if span is not None:
        return span
    return Interval._from_numeric_prefix_python(raw_candidate, domain_bounds, allow_exp)


def contract(domains: dict[Any, tuple[float, float]], constraints: Iterable[dict], *, rounds: int = 8):
    """Run the native HC4 contractor; ``None`` when unavailable."""
    if not available():
        return None
    import json

    wire_domains = {str(int(key)): [float(lo), float(hi)] for key, (lo, hi) in domains.items()}
    try:
        answer = _NATIVE.contract(
            json.dumps(wire_domains), json.dumps(list(constraints)), int(rounds)
        )
    except Exception:  # pragma: no cover
        return None
    return json.loads(answer)


def solve_linear(constraints: Sequence[dict]):
    """Exact feasibility for linear atoms; ``None`` when the engine cannot decide."""
    if not available():
        return None
    import json

    try:
        answer = json.loads(_NATIVE.solve_linear(json.dumps(list(constraints))))
    except Exception:  # pragma: no cover
        return None
    if answer.get("status") == "unknown":
        return None
    return answer
