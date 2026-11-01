"""Black-box async stream supervisor (shim)."""

from __future__ import annotations

import asyncio

class AsyncStreamSupervisor:
    def __init__(self, spec=None):
        self.spec = spec

    async def supervise(self, stream):
        async for chunk in stream:
            yield chunk

def abort_stream():
    return None
