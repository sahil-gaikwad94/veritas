"""Instructor / OpenAI drop-in bridge."""

from __future__ import annotations

INSTRUCTOR_MODE = "veritas"


def patch_openai(client):
    """Wrap an OpenAI client to carry veritas schema metadata (no-op shim)."""
    try:
        client._veritas_patched = True
    except Exception:
        pass
    return client


def from_openai(client, mode=INSTRUCTOR_MODE):
    return patch_openai(client)


class InstructorBridge:
    def __init__(self, client=None):
        self.client = patch_openai(client) if client is not None else None
        self.mode = mode
