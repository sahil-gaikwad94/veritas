"""BPE boundary healing against a deterministic fake tokenizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from veritas.token_healer import ByteTokenizer, JumpForwardPrefill, TokenHealer


class FakeBPE:
    """Greedy longest-match tokenizer over an explicit vocabulary + byte fallback."""

    def __init__(self, vocab):
        self.vocab = sorted(set(vocab), key=len, reverse=True)
        self._inverse = {token: i for i, token in enumerate(self.vocab)}
        self._offset = len(self.vocab)

    def encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            for token in self.vocab:
                if token and text.startswith(token, i):
                    ids.append(self._inverse[token])
                    i += len(token)
                    break
            else:
                ids.append(self._offset + ord(text[i]))
                i += 1
        return ids

    def decode(self, ids):
        out = []
        for token_id in ids:
            if token_id < self._offset:
                out.append(self.vocab[token_id])
            else:
                out.append(chr(token_id - self._offset))
        return "".join(out)


def test_byte_tokenizer_is_exact():
    healer = TokenHealer(ByteTokenizer())
    prefill = healer.heal('{"a": 1', ', "b": 2}')
    assert isinstance(prefill, JumpForwardPrefill)
    assert prefill.splice_string == ', "b": 2}'
    assert prefill.healed_text == '{"a": 1, "b": 2}'
    assert prefill.new_token_ids == list(b', "b": 2}')
    assert prefill.backup_tokens == 0


def test_healing_repairs_a_merged_boundary():
    # Without healing, "0" + "." + "5" would not match the model's own tokenization of
    # the spliced text: the tokenizer prefers the merged token "0.5".
    tok = FakeBPE(["0.5", "0", ".", "5", '"x"', ":", ",", " ", "{", "}"])
    healer = TokenHealer(tok, max_backup=3)
    prefix = '{"x": 0.'
    splice = '5}'
    baseline = tok.encode(prefix + splice)
    prefill = healer.heal(prefix, splice, tok)
    assert prefill.backup_tokens >= 1
    assert prefill.healed_text == prefix + splice
    # New tokens cover the rewound tail plus the splice, so the joined text is exact.
    tail = prefix[prefill.offset:]
    assert tok.decode(prefill.new_token_ids) == tail + splice
    kept = baseline[: len(baseline) - len(prefill.new_token_ids)]
    assert kept == tok.encode(prefix[: prefill.offset])
    assert tok.encode(prefill.healed_text) == baseline


def test_healing_never_loses_text():
    tok = FakeBPE(['"', ":", ",", " ", "{", "}", "true", "false", "null", "0.5"])
    healer = TokenHealer(tok)
    for prefix, splice in [
        ('{"a": ', '0.5}'),
        ('{"a": 0.', '5}'),
        ('{"a": tr', "ue}"),
        ("", '{"a": 1}'),
    ]:
        prefill = healer.heal(prefix, splice, tok)
        assert prefill.healed_text == prefix + splice
        tail = prefix[prefill.offset:]
        assert tok.decode(prefill.new_token_ids) == tail + splice


def test_empty_splice():
    healer = TokenHealer(ByteTokenizer())
    prefill = healer.heal("abc", "")
    assert prefill.new_token_ids == []
    assert prefill.backup_tokens == 0


def test_character_compatibility_api():
    healer = TokenHealer()
    prefill = healer.heal_and_splice(["tok"], "abc")
    assert prefill.splice_string == "abc"
    assert prefill.new_token_ids == list("abc")
