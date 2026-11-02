"""Concrete witnesses for JSON-Schema ``format`` strings.

A constrained decoder must be able to *produce* a value for a formatted string field:
``format: "date-time"``, ``"uuid"``, ``"email"`` and friends are validated by the concrete
schema checker, so sampling arbitrary characters cannot work.  This module supplies a
canonical witness per supported format, and re-checks every candidate with
``jsonschema``'s own ``FormatChecker`` before returning it — the same checker the verifier
uses, so the witness cannot drift away from the validation semantics.

Formats outside this table are *not* claimed: callers fall back to length masking and
record the fallback, leaving the concrete validator authoritative.
"""

from __future__ import annotations

from functools import lru_cache

#: Canonical candidates per format, most preferred first.  Every entry is re-validated by
#: :func:`witness`, so a typo here is a test failure rather than a silent bad value.
FORMAT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "date-time": ("2024-01-01T00:00:00Z", "2024-06-15T12:30:45+00:00"),
    "date": ("2024-01-01",),
    "time": ("00:00:00", "01:02:03"),
    "duration": ("PT1H", "PT0S", "P1DT2H"),
    "uuid": (
        "00000000-0000-4000-8000-000000000000",
        "123e4567-e89b-42d3-a456-426614174000",
    ),
    "email": ("user@example.com", "user.name+tag@sub.example.org"),
    "idn-email": ("user@example.com",),
    "uri": ("https://example.com/", "https://example.com/a/b?c=d"),
    "url": ("https://example.com/",),
    "iri": ("https://example.com/",),
    "uri-reference": ("https://example.com/", "/a/b"),
    "ipv4": ("192.0.2.1",),
    "ipv6": ("2001:db8::1",),
    "hostname": ("example.com", "host.example.org"),
    "idn-hostname": ("example.com",),
    "json-pointer": ("", "/a", "/a/0"),
    "relative-json-pointer": ("0", "1/a"),
    "regex": ("a", "a+" ),
    "byte": ("AA==", "aGVsbG8="),
    "password": ("secret",),
}

#: Formats whose JSON value is the string itself.  ``"duration"`` is ISO-8601 in JSON
#: Schema, which is what Pydantic's ``timedelta`` serialises to as well.
SUPPORTED_FORMATS: frozenset[str] = frozenset(FORMAT_CANDIDATES)


@lru_cache(maxsize=1)
def _checker():
    from jsonschema import FormatChecker

    return FormatChecker()


def supported_formats() -> frozenset[str]:
    return SUPPORTED_FORMATS


def is_supported(fmt: str | None) -> bool:
    return bool(fmt) and str(fmt) in SUPPORTED_FORMATS


def validate(fmt: str, text: str) -> bool | None:
    """``True``/``False`` when the format has a registered checker, else ``None``."""

    if not is_supported(fmt):
        return None
    try:
        _checker().check(text, fmt)
        return True
    except Exception:
        return False


def witness(
    fmt: str | None,
    *,
    min_len: int = 0,
    max_len: int | None = None,
) -> str | None:
    """A concrete value that satisfies ``format`` (and the length window), if one exists.

    Candidates are checked with the concrete validator, so the result is a *verified*
    witness rather than an example that merely looks right.
    """

    if not is_supported(fmt):
        return None
    floor = max(0, int(min_len or 0))
    ceiling = int(max_len) if max_len else None
    for candidate in FORMAT_CANDIDATES[str(fmt)]:
        if len(candidate) < floor:
            continue
        if ceiling is not None and len(candidate) > ceiling:
            continue
        if validate(str(fmt), candidate):
            return candidate
    return None


__all__ = [
    "FORMAT_CANDIDATES",
    "SUPPORTED_FORMATS",
    "is_supported",
    "supported_formats",
    "validate",
    "witness",
]
