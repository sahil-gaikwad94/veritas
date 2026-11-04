"""``veritas serve`` — OpenAI-compatible verification reverse proxy.

Any MCP-ready / OpenAI-shaped client can point at Veritas instead of an upstream LLM and get
a hard firewall for structured outputs.  Two enforcement levels, chosen per request:

* **non-streaming** (``stream: false``): the proxy buffers the upstream completion, re-verifies
  the decoded JSON against the configured ``veritas_spec`` and either returns the completion
  *plus* a ``veritas_certificate`` (PASS) or HTTP 422 with the concrete violations (FAIL).
  Nothing unverified ever reaches the client.  Use ``enforce`` (default) or ``monitor``.

* **streaming** (``stream: true``): token chunks flow straight through (the proxy cannot
  un-emit tokens).  When the upstream finishes, the proxy verifies the accumulated document
  and injects a final ``data: {"veritas_certificate": {...}}`` event *before* ``[DONE]``;
  the response carries ``X-Veritas-Verification: pass|fail`` so stream consumers can gate
  on it programmatically.

Endpoints: ``POST /v1/chat/completions`` (spec-aware), ``GET /v1/models`` (proxy),
``GET /healthz``.

Run it::

    veritas serve --upstream http://127.0.0.1:11434/v1 --spec-module ./schemas.py \
        --spec portfolio --port 8080

Specs are ``@vt.verified_schema()`` objects in ``--spec-module``; the client selects one per
request with ``extra_body: {"veritas_spec": "portfolio"}`` (also accepted as a top-level
``veritas_spec`` key).  The default spec (first schema in the module) applies otherwise.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import sys
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

from .instance import verify_values
from .proof.certificate import build_certificate
from .serde import spec_digest

_SSE_CONTENT = "text/event-stream"
_JSON_CONTENT = "application/json"


# ---------------------------------------------------------------------------
# spec registry
# ---------------------------------------------------------------------------

def _load_module(target: str):
    spec = importlib.util.spec_from_file_location("_veritas_serve_specs", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec module {target!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpecRegistry:
    """Loads ``--spec-module`` once and dispatches schemas by name."""

    def __init__(self, module_path: str, default: str | None = None):
        self.module_path = module_path
        self.specs: dict[str, Any] = {}
        if module_path:
            from pathlib import Path

            from .decorators import VerifiedSchema

            if not Path(module_path).exists():
                raise ImportError(f"cannot load spec module {module_path!r}: file does not exist")
            module = _load_module(module_path)
            self.specs = {
                name: value for name, value in vars(module).items()
                if isinstance(value, VerifiedSchema)
            }
            if not self.specs:
                raise ValueError(f"no @vt.verified_schema objects found in {module_path!r}")
        self.default = default or next(iter(self.specs), None)
        if default and default not in self.specs:
            raise ValueError(f"spec {default!r} not found in {module_path!r}; have: {sorted(self.specs)}")

    def get(self, name: str | None):
        if name and name in self.specs:
            return name, self.specs[name]
        return self.default, self.specs.get(self.default)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def _extract_content(parsed: Any) -> tuple[str, Mapping[str, Any] | None]:
    try:
        choices = parsed.get("choices") if isinstance(parsed, Mapping) else None
        if not choices:
            return "", None
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None and isinstance(choice.get("delta"), Mapping):
            content = choice["delta"].get("content")
        if not isinstance(content, str):
            return "", parsed if isinstance(parsed, Mapping) else None
        return content, parsed if isinstance(parsed, Mapping) else None
    except Exception:
        return "", None


def _parse_json_candidate(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    # Tolerate a json-fenced code block around the document.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 2:
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _verify_completion(spec: Any, content: str) -> dict[str, Any]:
    """Verify a completion string against a compiled schema; returns certificate dict."""

    stamp = {"verified_at": int(time.time()), "spec_digest": spec_digest(spec.spec) if hasattr(spec, "spec") else ""}
    candidate = _parse_json_candidate(content)
    if candidate is None:
        return {
            **stamp,
            "status": "FAIL",
            "errors": ["completion content is not valid JSON"],
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    report = verify_values(spec.spec, candidate, derive_missing=True)
    digest_source = report.spec_digest or stamp["spec_digest"]
    cert = {
        **stamp,
        "status": report.status,
        "spec_digest": digest_source,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "values_sha256": hashlib.sha256(
            json.dumps(report.values, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "backtracks": None,  # streaming/proxy path does not backtrack; the verifier signs, not repairs
        "violations": [o.text for o in report.violations] if report.violations else [],
        "unsat_core": list(report.unsat_core),
    }
    if report.status == "PASS":
        try:
            proof = build_certificate(spec.spec, report.values, metadata={"proxy": "veritas serve"})
            cert["proof_sha256"] = proof.sha256
        except Exception as exc:  # proof building must not break the response
            cert["proof_error"] = str(exc)
    return cert


# ---------------------------------------------------------------------------
# upstream helpers
# ---------------------------------------------------------------------------

def _upstream_parts(upstream: str) -> tuple[str, int, str]:
    parsed = urlparse(upstream if "://" in upstream else f"http://{upstream}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    return host, port, base_path


def _stream_chat(handler: _Handler, upstream: str, payload: dict[str, Any], api_key: str | None, spec_name: str | None):
    """Forward an SSE request, buffer the document, inject the certificate before DONE.

    The response uses HTTP/1.1 chunked transfer encoding — without it the client has no way
    to delimit the streamed body and blocks forever.
    """

    host, port, base_path = _upstream_parts(upstream)
    connection = http.client.HTTPConnection(host, port, timeout=300)
    path = f"{base_path}/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": _SSE_CONTENT}
    effective_key = api_key or payload.get("api_key_hint")
    auth = None
    if effective_key:
        headers["Authorization"] = f"Bearer {effective_key}"
    else:
        auth = handler.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
    body = json.dumps(payload).encode("utf-8")
    connection.request("POST", path, body=body, headers=headers)
    upstream_resp = connection.getresponse()

    if upstream_resp.status != 200:
        # Buffer the upstream error verbatim; it carries a length we do know.
        error_body = upstream_resp.read()
        handler.send_response(upstream_resp.status)
        handler.send_header("Content-Type", _JSON_CONTENT)
        handler.send_header("X-Veritas-Verification", "upstream-error")
        handler.send_header("Content-Length", str(len(error_body)))
        handler.end_headers()
        handler.wfile.write(error_body)
        connection.close()
        return

    handler.send_response(200)
    handler.send_header("Content-Type", _SSE_CONTENT)
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()

    def _write_chunk(data: bytes) -> None:
        handler.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
        handler.wfile.flush()

    accumulated = ""
    buffer = b""
    while True:
        chunk = upstream_resp.read1(65536) if hasattr(upstream_resp, "read1") else upstream_resp.read(65536)
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            stripped = line.decode("utf-8", errors="replace").rstrip("\r")
            if stripped.startswith("data:") and stripped[5:].strip() == "[DONE]":
                # Hold the terminator back: the certificate is appended before it.
                continue
            _write_chunk(line + b"\n")
            if not stripped.startswith("data:"):
                continue
            data = stripped[5:].strip()
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = (parsed.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str):
                accumulated += piece
    if buffer:
        _write_chunk(buffer)

    registry = handler.server.veritas_registry  # type: ignore[attr-defined]
    cert = None
    if spec_name and registry.specs:
        _, spec = registry.get(spec_name)
        cert = _verify_completion(spec, accumulated) if spec is not None else None
    if cert is not None:
        _write_chunk(f"data: {json.dumps({'veritas_certificate': cert}, default=str)}\n\n".encode())
    # One terminator, always after the certificate.
    _write_chunk(b"data: [DONE]\n\n")
    handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()
    connection.close()


def _buffered_chat(handler: _Handler, upstream: str, payload: dict[str, Any], api_key: str | None, spec_name: str | None, enforce: bool):
    host, port, base_path = _upstream_parts(upstream)
    connection = http.client.HTTPConnection(host, port, timeout=300)
    path = f"{base_path}/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": _JSON_CONTENT}
    effective_key = api_key
    if effective_key:
        headers["Authorization"] = f"Bearer {effective_key}"
    else:
        auth = handler.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
    if "stream" in payload:
        payload = {**payload, "stream": False}
    connection.request("POST", path, body=json.dumps(payload).encode("utf-8"), headers=headers)
    upstream_resp = connection.getresponse()
    raw = upstream_resp.read()
    connection.close()
    try:
        parsed = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        parsed = None

    registry = handler.server.veritas_registry  # type: ignore[attr-defined]
    if spec_name and registry.specs:
        _, spec = registry.get(spec_name)
        content, _ = _extract_content(parsed) if isinstance(parsed, Mapping) else ("", None)
        cert = _verify_completion(spec, content or "") if spec is not None else None
        if cert is not None:
            if cert["status"] == "PASS":
                if isinstance(parsed, dict):
                    parsed = {**parsed, "veritas_certificate": cert}
                    raw = json.dumps(parsed).encode("utf-8")
                handler.send_response(200)
                handler.send_header("Content-Type", _JSON_CONTENT)
                handler.send_header("X-Veritas-Verification", "pass")
                handler.send_header("Content-Length", str(len(raw)))
                handler.end_headers()
                handler.wfile.write(raw)
                return
            if enforce:
                error = {
                    "error": {
                        "type": "veritas_verification_failed",
                        "message": "completion failed schema verification",
                        "spec": spec_name,
                        "violations": cert.get("violations", []),
                        "unsat_core": cert.get("unsat_core", []),
                    },
                    "veritas_certificate": cert,
                }
                payload_bytes = json.dumps(error, default=str).encode("utf-8")
                handler.send_response(422)
                handler.send_header("Content-Type", _JSON_CONTENT)
                handler.send_header("X-Veritas-Verification", "fail")
                handler.send_header("Content-Length", str(len(payload_bytes)))
                handler.end_headers()
                handler.wfile.write(payload_bytes)
                return
            parsed = {**parsed, "veritas_certificate": cert}
            raw = json.dumps(parsed).encode("utf-8")
    handler.send_response(upstream_resp.status if parsed is None else 200)
    handler.send_header("Content-Type", _JSON_CONTENT)
    handler.send_header("X-Veritas-Verification", "monitor" if spec_name else "disabled")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep stderr quiet-ish; tests opt in
        if getattr(self.server, "veritas_verbose", False):
            sys.stderr.write(fmt % args + "\n")

    def _json_reply(self, status: int, payload: Any, extra_headers: Mapping[str, str] | None = None):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", _JSON_CONTENT)
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (http.server API)
        if self.path == "/healthz":
            registry: SpecRegistry = self.server.veritas_registry  # type: ignore[attr-defined]
            self._json_reply(200, {
                "status": "ok",
                "specs": sorted(registry.specs),
                "default": registry.default,
                "upstream": getattr(self.server, "veritas_upstream", None),
            })
            return
        if self.path.startswith("/v1/models"):
            host, port, base_path = _upstream_parts(self.server.veritas_upstream)  # type: ignore[attr-defined]
            connection = http.client.HTTPConnection(host, port, timeout=30)
            try:
                connection.request("GET", f"{base_path}/models", headers=self._forward_headers())
                upstream_resp = connection.getresponse()
                raw = upstream_resp.read()
                self.send_response(upstream_resp.status)
                self.send_header("Content-Type", _JSON_CONTENT)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except OSError as exc:
                self._json_reply(502, {"error": {"type": "upstream_unavailable", "message": str(exc)}})
            finally:
                connection.close()
            return
        self._json_reply(404, {"error": {"type": "not_found"}})

    def _forward_headers(self) -> dict[str, str]:
        headers = {}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        return headers

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json_reply(404, {"error": {"type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body) if raw_body else {}
        except (ValueError, json.JSONDecodeError):
            self._json_reply(400, {"error": {"type": "invalid_json"}})
            return
        if not isinstance(payload, dict):
            self._json_reply(400, {"error": {"type": "invalid_request"}})
            return

        # Spec selection: top-level or extra_body.veritas_spec.
        extra = payload.get("extra_body") if isinstance(payload.get("extra_body"), Mapping) else {}
        spec_name = payload.get("veritas_spec") or extra.get("veritas_spec") or getattr(self.server, "veritas_default_spec", None)
        enforce = bool(getattr(self.server, "veritas_enforce", True))
        api_key = payload.pop("veritas_api_key", None) or getattr(self.server, "veritas_api_key", None)
        # Never forward the Veritas-specific keys upstream.
        for key in ("veritas_spec", "veritas_api_key"):
            payload.pop(key, None)
        if isinstance(payload.get("extra_body"), Mapping):
            payload["extra_body"] = {k: v for k, v in payload["extra_body"].items() if k != "veritas_spec"}

        stream = bool(payload.get("stream"))
        upstream = self.server.veritas_upstream  # type: ignore[attr-defined]
        try:
            if stream:
                _stream_chat(self, upstream, payload, api_key, spec_name)
            else:
                _buffered_chat(self, upstream, payload, api_key, spec_name, enforce)
        except OSError as exc:
            self._json_reply(502, {"error": {"type": "upstream_unavailable", "message": str(exc)}})


# NOTE on threading: the historical default (one thread per connection) works fine, but
# single-threaded handling is dramatically more robust on constrained interpreters (a hard
# kill in a background thread loses buffered responses).  Structured-output verification is
# a short-lived request/response workload, so requests are handled in order.  Embedding in a
# threaded server is as simple as passing `server_class=ThreadingHTTPServer`.
class VeritasServeServer(HTTPServer):
    def __init__(self, server_address, handler):
        self.veritas_upstream = None
        self.veritas_registry = None
        self.veritas_enforce = True
        self.veritas_api_key = None
        self.veritas_verbose = False
        self.veritas_default_spec = None
        super().__init__(server_address, handler)

    daemon_threads = True


def serve(
    upstream: str,
    spec_module: str | None = None,
    spec: str | None = None,
    port: int = 8080,
    host: str = "127.0.0.1",
    *,
    enforce: bool = True,
    api_key: str | None = None,
    verbose: bool = False,
) -> VeritasServeServer:
    """Create and return a configured server (caller calls ``serve_forever``)."""

    registry = SpecRegistry(spec_module or "", default=spec)
    server = VeritasServeServer((host, port), _Handler)
    server.veritas_upstream = upstream  # type: ignore[attr-defined]
    server.veritas_registry = registry  # type: ignore[attr-defined]
    server.veritas_enforce = enforce  # type: ignore[attr-defined]
    server.veritas_api_key = api_key  # type: ignore[attr-defined]
    server.veritas_verbose = verbose  # type: ignore[attr-defined]
    server.veritas_default_spec = registry.default  # type: ignore[attr-defined]
    return server
