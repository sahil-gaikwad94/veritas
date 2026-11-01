"""Black-box async stream supervision.

Works with raw text chunks, SSE ``data:`` frames and OpenAI/Anthropic-style delta objects.
The supervisor aborts as soon as a chunk makes the JSON prefix malformed, and optionally
performs full spec verification when the document completes.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Any

from ..json_pda import JSONPushdownAutomaton


class StreamViolation(RuntimeError):
    def __init__(self, message: str, *, report: Any = None, kind: str = "grammar"):
        super().__init__(message)
        self.report = report
        self.kind = kind


def _extract_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", "replace")
    if isinstance(chunk, str):
        text = chunk
        if text.startswith("data:"):
            payload = text[5:].strip()
            if payload == "[DONE]":
                return ""
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return payload
            return _extract_text(data)
        return text
    if isinstance(chunk, dict):
        choices = chunk.get("choices")
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or choice.get("message") or {}
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
            text = choice.get("text")
            if isinstance(text, str):
                return text
        delta = chunk.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            return delta["text"]
        for key in ("content", "text", "token"):
            value = chunk.get(key)
            if isinstance(value, str):
                return value
        return ""
    for name in ("content", "text", "token"):
        value = getattr(chunk, name, None)
        if isinstance(value, str):
            return value
    choices = getattr(chunk, "choices", None)
    if choices:
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        text = getattr(delta, "content", None) if delta is not None else None
        if isinstance(text, str):
            return text
    return ""


def _report_ok(report: Any) -> bool:
    ok = getattr(report, "ok", None)
    if ok is not None:
        return bool(ok)
    return getattr(report, "status", "") == "PASS"


class AsyncStreamSupervisor:
    def __init__(
        self,
        spec: Any = None,
        *,
        require_complete: bool = True,
        on_violation: Callable[[StreamViolation], Any] | None = None,
    ):
        self.spec = spec
        self.require_complete = require_complete
        self.on_violation = on_violation

    def _notify(self, violation: StreamViolation) -> None:
        if self.on_violation is None:
            return
        result = self.on_violation(violation)
        if inspect.isawaitable(result):
            # Callers that need async notification can pass a coroutine function; the
            # exception path below still aborts deterministically.
            try:
                import asyncio

                asyncio.get_running_loop().create_task(result)
            except RuntimeError:
                pass

    async def supervise(self, stream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        """Supervise a stream, withholding the final chunk until verification passes.

        Earlier chunks are released as soon as the next chunk proves the prefix is still
        grammatical. The last chunk is buffered so a downstream consumer never receives
        a complete-looking document that later fails instance verification.
        """
        machine = JSONPushdownAutomaton()
        pieces: list[str] = []
        pending: Any = None
        have_pending = False
        async for chunk in stream:
            text = _extract_text(chunk)
            if text:
                if not machine.feed(text):
                    violation = StreamViolation(f"invalid JSON stream: {machine.error}", kind="grammar")
                    self._notify(violation)
                    raise violation
                pieces.append(text)
            if have_pending:
                yield pending
            pending = chunk
            have_pending = True
        text = "".join(pieces)
        if self.require_complete and not (machine.finish() and machine.accepts()):
            violation = StreamViolation(
                f"stream ended before JSON was complete: {machine.error or machine.state}",
                kind="incomplete",
            )
            self._notify(violation)
            raise violation
        if self.spec is not None and text:
            from ..instance import verify_json

            try:
                report = verify_json(self.spec, text)
            except Exception as exc:
                violation = StreamViolation(f"stream verification failed: {exc}", kind="instance")
                self._notify(violation)
                raise violation from exc
            if not _report_ok(report):
                violation = StreamViolation(
                    f"stream violated the spec ({getattr(report, 'status', 'FAIL')})",
                    report=report,
                    kind="instance",
                )
                self._notify(violation)
                raise violation
        if have_pending:
            yield pending

    async def collect(self, stream: AsyncIterable[Any]) -> list[Any]:
        return [chunk async for chunk in self.supervise(stream)]


def abort_stream(reason: str = "aborted by veritas") -> StreamViolation:
    return StreamViolation(reason, kind="aborted")


__all__ = ["AsyncStreamSupervisor", "StreamViolation", "abort_stream"]
