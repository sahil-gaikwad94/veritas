"""Tier-0 JSON pushdown automaton, token masking and canonical rendering."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pytest

from veritas.json_pda import (
    JSONPushdownAutomaton,
    JSONSyntaxError,
    TokenMasker,
    is_complete_json,
    json_value,
    parse_json,
    render_continuation,
    render_object,
)


def accepts(text: str) -> bool:
    machine = JSONPushdownAutomaton()
    return machine.feed(text) and machine.accepts()


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        "[]",
        '{"a": 1}',
        '{"a": [1, 2.5, -3e-2, true, false, null], "b": {"c": "x\\ny"}}',
        '[{"a": 0}, {"b": [{}]}]',
        '  {"a" : "\\u00e9"}  ',
    ],
)
def test_valid_documents(text):
    assert accepts(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{",
        '{"a"}',
        '{"a": }',
        '{"a": 1,}',
        "[1,]",
        '{"a": 01}',
        '{"a": .5}',
        '{"a": 1.}',
        '{"a": "unterminated}',
        '{"a": "bad\\q"}',
        '{"a": 1} extra',
        "[1 2]",
        '{"a": "line\nbreak"}',
    ],
)
def test_invalid_documents(text):
    assert not accepts(text)


def test_prefix_acceptance_and_allowed_chars():
    machine = JSONPushdownAutomaton()
    assert machine.feed('{"a": ')
    assert not machine.accepts()
    assert machine.state == "value"
    allowed = machine.allowed_chars()
    assert '"' in allowed and "1" in allowed and "}" not in allowed
    # Curated fast set must agree with trial stepping on the printable alphabet.
    probed = machine.allowed_chars_probed()
    assert allowed == probed


def test_invalid_prefix_reports_error():
    machine = JSONPushdownAutomaton()
    assert machine.feed('{"a": }') is False
    assert machine.is_error()
    assert machine.error


def test_depth_limit():
    machine = JSONPushdownAutomaton(max_depth=3)
    assert machine.feed("[[[")
    assert machine.feed("[") is False


def test_token_masker_blocks_invalid_tokens():
    vocab = ["0", ".", "5", ",", ":", "{", "}", '"', "a", "1", "x", " "]
    masker = TokenMasker(vocab, eos_token=len(vocab))
    mask = masker.valid_mask('{"a": ')
    allowed = {vocab[i] for i, ok in enumerate(mask) if ok}
    assert "x" not in allowed
    assert "1" in allowed
    assert "}" not in allowed  # cannot close before a value

    final_mask = masker.valid_mask('{"a": 1}')
    assert final_mask[-1] is True  # EOS allowed only on a complete document
    assert masker.valid_mask('{"a": 1,')[-1] is False


def test_masker_apply_mapping_and_sequence():
    vocab = ["1", "x"]
    masker = TokenMasker(vocab)
    masked = masker.apply('{"a": ', {0: 1.0, 1: 2.0})
    assert masked[1] == -math.inf
    assert masked[0] == 1.0

    arr = [1.0, 2.0]
    masker.apply('{"a": ', arr)
    assert arr[1] == -math.inf
    assert arr[0] == 1.0


def test_masker_rejects_invalid_prefix():
    masker = TokenMasker(["1"])
    with pytest.raises(JSONSyntaxError):
        masker.valid_mask("{bad")


def test_render_and_parse_roundtrip():
    values = {"b": 0.1 + 0.2, "a": {"z": 1, "y": [True, None, "s"]}, "c": 3}
    text = render_object(values, field_order=["a", "b", "c"], precision={"b": 6})
    parsed = parse_json(text)
    assert list(parsed.keys()) == ["a", "b", "c"]
    assert abs(parsed["b"] - 0.3) < 1e-12
    assert parsed["a"] == {"y": [True, None, "s"], "z": 1}
    assert is_complete_json(text)


def test_render_continuation():
    text = render_continuation(
        {"expected_return": 0.165, "volatility": 0.19417},
        ["expected_return", "volatility"],
        precision={"expected_return": 5, "volatility": 5},
    )
    assert text.startswith(', "expected_return": 0.165')
    assert ", " in text and "volatility" in text
    assert accepts('{"weights": {}' + text + "}")


def test_json_value_rejects_non_finite():
    with pytest.raises(ValueError):
        json_value(float("inf"))
