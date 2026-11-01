"""OpenAI / Instructor drop-in bridge with post-generation verification.

``patch_openai(client, spec)`` returns a proxy that behaves like the original client but
attaches the VIR specification to requests and verifies the returned JSON before handing
the response back.  Invalid output raises :class:`VerificationError` instead of silently
passing through — the caller can still inspect ``error.report``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..instance import verify_json

INSTRUCTOR_MODE = "veritas"


class VerificationError(RuntimeError):
    def __init__(self, message: str, *, report: Any = None, response: Any = None):
        super().__init__(message)
        self.report = report
        self.response = response


def _coerce_spec(candidate: Any) -> Any:
    if candidate is None:
        return None
    from ..compiler.vir import VirSpec

    if isinstance(candidate, VirSpec):
        return candidate
    spec = getattr(candidate, "spec", None)
    if isinstance(spec, VirSpec):
        return spec
    return None


def _spec_schema(spec: Any) -> dict[str, Any] | None:
    if spec is None:
        return None
    schema = getattr(spec, "model_schema", None)
    if isinstance(schema, dict):
        return schema
    schema = getattr(spec, "json_schema", None)
    if callable(schema):
        try:
            return schema()
        except Exception:
            return None
    return None


def _response_text(response: Any) -> str | None:
    if response is None:
        return None
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if choices:
            choice = choices[0]
            message = choice.get("message") or {}
            content = message.get("content", choice.get("text"))
            if isinstance(content, str):
                return content
        for key in ("output_text", "content", "text"):
            if isinstance(response.get(key), str):
                return response[key]
        return None
    choices = getattr(response, "choices", None)
    if choices:
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content is None:
            content = getattr(choice, "text", None)
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            )
        if isinstance(content, str):
            return content
    for name in ("output_text", "content", "text"):
        value = getattr(response, name, None)
        if isinstance(value, str):
            return value
    return None


def _report_ok(report: Any) -> bool:
    ok = getattr(report, "ok", None)
    if ok is not None:
        return bool(ok)
    return getattr(report, "status", "") == "PASS"


def _attach(response: Any, report: Any) -> None:
    try:
        response.veritas_report = report
    except Exception:
        try:
            response.__dict__["veritas_report"] = report
        except Exception:
            pass


class _CompletionsProxy:
    def __init__(self, parent: VeritasOpenAIProxy):
        self._parent = parent

    def create(self, **kwargs: Any) -> Any:
        return self._parent._create(**kwargs)


class _ChatProxy:
    def __init__(self, parent: VeritasOpenAIProxy):
        self.completions = _CompletionsProxy(parent)


class VeritasOpenAIProxy:
    def __init__(self, client: Any, spec: Any = None, mode: str = INSTRUCTOR_MODE):
        self._client = client
        self.spec = _coerce_spec(spec)
        self.mode = mode
        self.chat = _ChatProxy(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _create(self, **kwargs: Any) -> Any:
        response_format = kwargs.get("response_format")
        spec = self.spec or _coerce_spec(response_format)
        if spec is not None:
            schema = _spec_schema(spec)
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "veritas_output", "schema": schema or {"type": "object"}},
            }
            extra = dict(kwargs.get("extra_body") or {})
            try:
                from ..serde import spec_to_dict

                extra.setdefault("veritas_vir", spec_to_dict(spec))
            except Exception:
                extra.setdefault("veritas_spec", True)
            kwargs["extra_body"] = extra
        create = self._client.chat.completions.create
        response = create(**kwargs)
        text = _response_text(response)
        if spec is not None and text is not None:
            try:
                report = verify_json(spec, text)
            except Exception as exc:
                raise VerificationError(f"response was not valid JSON: {exc}", response=response) from exc
            _attach(response, report)
            if not _report_ok(report):
                raise VerificationError(
                    f"response violated the specification ({getattr(report, 'status', 'FAIL')})",
                    report=report,
                    response=response,
                )
        return response


def patch_openai(client: Any, spec: Any = None, mode: str = INSTRUCTOR_MODE) -> Any:
    if client is None:
        return None
    if isinstance(client, VeritasOpenAIProxy):
        return client
    return VeritasOpenAIProxy(client, spec=spec, mode=mode)


def from_openai(client: Any, mode: str = INSTRUCTOR_MODE, spec: Any = None) -> Any:
    if mode != INSTRUCTOR_MODE:
        return client
    return patch_openai(client, spec=spec, mode=mode)


class InstructorBridge:
    def __init__(self, client: Any = None, spec: Any = None, mode: str = INSTRUCTOR_MODE):
        self.mode = mode
        self.client = patch_openai(client, spec=spec, mode=mode) if client is not None else None

    def create(self, **kwargs: Any) -> Any:
        if self.client is None:
            raise RuntimeError("InstructorBridge has no client")
        return self.client.chat.completions.create(**kwargs)


__all__ = [
    "INSTRUCTOR_MODE",
    "InstructorBridge",
    "VerificationError",
    "VeritasOpenAIProxy",
    "from_openai",
    "patch_openai",
]
