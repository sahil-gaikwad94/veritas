"""Bridge tests: exact masking, stream aborts, Gumbel sampling and OpenAI verification."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import veritas as vt
from veritas.bridges.async_stream_proxy import AsyncStreamSupervisor, StreamViolation
from veritas.bridges.diffusion_bridge import DiffusionBridge, gumbel_max_sample, unmask_order
from veritas.bridges.hf_transformers import VeritasLogitsProcessor
from veritas.bridges.instructor_bridge import VerificationError, patch_openai
from veritas.bridges.llamacpp_bridge import patch_llamacpp
from veritas.bridges.sglang_bridge import patch_sglang
from veritas.bridges.vllm_bridge import patch_vllm


class Linear(BaseModel):
    a: float
    b: float


@vt.verified_schema()
def linear(p: Linear):
    vt.require(p.a >= 0)
    vt.require(p.a + p.b == 1.0, tol=1e-9)


class FakeTokenizer:
    eos_token_id = 6

    def __init__(self):
        self.vocab = {"{": 0, '"a"': 1, ":": 2, "1": 3, "}": 4, "x": 5, "": 6}

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, ids):
        return "".join(self.vocab_inv().get(int(i), "") for i in ids)

    def vocab_inv(self):
        if not hasattr(self, "_inv"):
            self._inv = {index: token for token, index in self.vocab.items()}
        return self._inv


def test_hf_processor_masks_invalid_tokens_and_allows_eos():
    tokenizer = FakeTokenizer()
    processor = VeritasLogitsProcessor(tokenizer, spec=linear)
    scores = [1.0] * 7
    processor([], scores)
    assert scores[0] > float("-inf")  # '{'
    assert scores[5] == float("-inf")  # 'x'
    assert scores[6] == float("-inf")  # EOS before completion

    # Schema-aware masking blocks EOS until all required fields are present.
    partial = [1.0] * 7
    processor([0, 1, 2, 3, 4], partial)  # {"a": 1}
    assert partial[6] == float("-inf")
    grammar_only = VeritasLogitsProcessor(tokenizer, spec=None)
    done = [1.0] * 7
    grammar_only([0, 1, 2, 3, 4], done)  # {"a": 1}
    assert done[6] == 1.0
    assert processor.verify('{"a": 0.25, "b": 0.75}').status == "PASS"


def test_host_adapters_attach_processors():
    tokenizer = FakeTokenizer()
    engine = SimpleNamespace(logits_processor=[])
    assert patch_vllm(engine, linear, tokenizer) is engine
    assert engine.logits_processor

    attached = {}
    model = SimpleNamespace(set_logits_processor=lambda p: attached.setdefault("p", p))
    patch_llamacpp(model, linear, tokenizer)
    assert attached["p"] is not None
    assert model.veritas_grammar.startswith("root")

    sglang = SimpleNamespace(logits_processor=[])
    patch_sglang(sglang, linear, tokenizer)
    assert sglang.logits_processor


def test_async_stream_supervisor_accepts_and_aborts():
    async def good():
        yield '{"a": '
        yield '0.25, "b": 0.75}'

    supervisor = AsyncStreamSupervisor(spec=linear)
    chunks = asyncio.run(supervisor.collect(good()))
    assert len(chunks) == 2

    async def bad():
        yield '{"a": '
        yield "oops"

    with pytest.raises(StreamViolation) as error:
        asyncio.run(AsyncStreamSupervisor(spec=linear).collect(bad()))
    assert error.value.kind == "grammar"

    async def incomplete():
        yield '{"a": 1'

    with pytest.raises(StreamViolation) as error:
        asyncio.run(AsyncStreamSupervisor().collect(incomplete()))
    assert error.value.kind == "incomplete"

    async def violating():
        yield '{"a": 0.2, "b": 0.2}'

    with pytest.raises(StreamViolation) as error:
        asyncio.run(AsyncStreamSupervisor(spec=linear).collect(violating()))
    assert error.value.kind == "instance"
    assert error.value.report.status == "FAIL"


def test_async_stream_handles_openai_sse_frames():
    async def stream():
        yield {"choices": [{"delta": {"content": '{"a": '}}]}
        yield "data: " + json.dumps({"choices": [{"delta": {"content": '0.25, "b": 0.75}'}}]} )

    chunks = asyncio.run(AsyncStreamSupervisor(spec=linear).collect(stream()))
    assert len(chunks) == 2


def test_diffusion_gumbel_sampling_and_unmasking():
    assert gumbel_max_sample({"a": 1.0, "b": 2.0}, temperature=0) == "b"
    assert gumbel_max_sample([0.0, 5.0, 1.0], temperature=0) == 1
    assert unmask_order([0.1, 0.9, 0.5]) == [1, 2, 0]
    bridge = DiffusionBridge(seed=7)
    sampled = bridge.sample_token({"left": 0.0, "right": 1.0})
    assert sampled in {"left", "right"}
    assert bridge.step([{"a": 1.0}, {"b": 1.0}], [True, True]) in (0, 1)


class _FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _FakeClient:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_openai_proxy_verifies_valid_response_and_injects_spec():
    client = _FakeClient('{"a": 0.25, "b": 0.75}')
    proxy = patch_openai(client, spec=linear)
    response = proxy.chat.completions.create(model="fake", messages=[])
    assert response.veritas_report.status == "PASS"
    call = client.completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert "veritas_vir" in call["extra_body"]


def test_openai_proxy_rejects_invalid_response():
    client = _FakeClient('{"a": 0.2, "b": 0.2}')
    proxy = patch_openai(client, spec=linear)
    with pytest.raises(VerificationError) as error:
        proxy.chat.completions.create(model="fake", messages=[])
    assert error.value.report.status == "FAIL"
    assert error.value.response is not None
