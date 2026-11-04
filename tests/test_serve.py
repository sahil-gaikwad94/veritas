"""Live tests for `veritas serve` — a verification reverse proxy in front of an upstream.

The fixture spawns the upstream stub *and* the proxy as one subprocess (the proxy is a
service, after all), waits for its ``READY`` marker, exercises it over HTTP, then terminates
it.  Running the service out-of-process keeps these tests independent of interpreter-teardown
behavior on constrained platforms.
"""

from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from veritas.serve import _parse_json_candidate, SpecRegistry  # noqa: E402

SPEC_MODULE = Path(__file__).parent / "serve_spec_fixture.py"

_RUNNER = r"""
import json
import socket
import sys
import threading
import http.server
import http.client


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        content = self.server.content
        if b'"stream": true' in body or b'"stream":true' in body:
            sse = (
                f'data: {{"id":"x","object":"chat.completion.chunk","choices":[{{"index":0,'
                f'"delta":{{"role":"assistant","content":{json.dumps(content)}}},'
                f'"finish_reason":null}}]}}\n\n'
                'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,'
                '"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)
            return
        reply = json.dumps({
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main():
    upstream_port = _free_port()
    proxy_port = _free_port()
    content = sys.argv[2]
    spec_module = sys.argv[3]
    spec_name = sys.argv[4]

    up = http.server.HTTPServer(("127.0.0.1", upstream_port), _Handler)
    up.content = content
    threading.Thread(target=up.serve_forever, daemon=True).start()

    from veritas.serve import serve
    proxy = serve(
        f"http://127.0.0.1:{upstream_port}/v1",
        spec_module=spec_module,
        spec=spec_name,
        port=proxy_port,
        enforce=True,
    )
    print(f"READY {proxy_port}", flush=True)
    proxy.serve_forever()


if __name__ == "__main__":
    main()
"""


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _LiveServe:
    def __init__(self, content: str, spec_module: Path, spec: str):
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _RUNNER, "x", content, str(spec_module), spec],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1] / "python"),
        )
        line = self._proc.stdout.readline().strip()
        assert line.startswith("READY "), f"runner did not become ready: {line!r}"
        self.port = int(line.split()[1])

    def url(self, path: str = "/v1/chat/completions") -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def post(self, body: dict, headers: dict | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        response = conn.getresponse()
        data = response.read()
        return response.status, dict(response.getheaders()), data

    def close(self):
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


@pytest.fixture(scope="module")
def spec_file(tmp_path_factory) -> Path:
    target = tmp_path_factory.mktemp("veritas-serve") / "serve_spec_fixture.py"
    target.write_text(
        "import veritas as vt\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class Followup(BaseModel):\n"
        "    a: float = Field(ge=0.0, le=100.0)\n"
        "    b: float = Field(ge=0.0, le=100.0)\n"
        "\n"
        "@vt.verified_schema()\n"
        "def followup(p: Followup):\n"
        "    vt.require(p.b == p.a + 1.0, tol=1e-6)\n",
        encoding="utf-8",
    )
    return target


@pytest.fixture(scope="module")
def passing_stack(spec_file):
    stack = _LiveServe('{"a": 5.0, "b": 6.0}', spec_file, "followup")
    yield stack
    stack.close()


@pytest.fixture(scope="module")
def failing_stack(spec_file):
    stack = _LiveServe('{"a": 5.0, "b": 9.0}', spec_file, "followup")
    yield stack
    stack.close()


def _body(spec: str = "followup"):
    return {
        "model": "stub",
        "messages": [{"role": "user", "content": "answer as json"}],
        "extra_body": {"veritas_spec": spec},
    }


def test_healthz_reports_specs(passing_stack):
    conn = http.client.HTTPConnection("127.0.0.1", passing_stack.port, timeout=10)
    conn.request("GET", "/healthz")
    response = conn.getresponse()
    payload = json.loads(response.read())
    assert response.status == 200
    assert payload["status"] == "ok"
    assert "followup" in payload["specs"]


def test_nonstreaming_pass_attaches_certificate(passing_stack):
    status, headers, body = passing_stack.post(_body())
    assert status == 200
    assert headers.get("X-Veritas-Verification") == "pass"
    parsed = json.loads(body)
    certificate = parsed.get("veritas_certificate")
    assert certificate and certificate["status"] == "PASS"
    assert len(certificate["sha256"]) == 64
    assert certificate["spec_digest"]


def test_nonstreaming_fail_returns_422_with_violations(failing_stack):
    status, headers, body = failing_stack.post(_body())
    assert status == 422
    assert headers.get("X-Veritas-Verification") == "fail"
    parsed = json.loads(body)
    assert parsed["error"]["type"] == "veritas_verification_failed"
    assert parsed["veritas_certificate"]["status"] == "FAIL"


def test_streaming_injects_certificate_before_done(passing_stack):
    status, headers, raw = passing_stack.post({**_body(), "stream": True})
    assert status == 200
    text = raw.decode()
    assert "chat.completion.chunk" in text
    assert "veritas_certificate" in text
    assert text.rstrip().endswith("[DONE]")


def test_parse_json_candidate_accepts_fenced_document():
    assert _parse_json_candidate('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_candidate("not json") is None
    assert _parse_json_candidate('{"a": 1}') == {"a": 1}


def test_spec_registry_rejects_missing_module(tmp_path):
    with pytest.raises((ImportError, ValueError)):
        SpecRegistry(str(tmp_path / "missing.py"))
