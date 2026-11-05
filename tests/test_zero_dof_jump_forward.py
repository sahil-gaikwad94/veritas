"""Phase 3 tests: zero-dof jump forward."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vexx.token_healer import JumpForwardPrefill, TokenHealer
from vexx.zero_dof import ZeroDofEngine, ZeroDofRule


def test_cascade_evaluation():
    eng = ZeroDofEngine()
    eng.add_rule(ZeroDofRule(target_var="er", input_vars=["w"], rhs_eval=lambda vals: vals["w"] * 2))
    eng.add_rule(ZeroDofRule(target_var="sharpe", input_vars=["er"], rhs_eval=lambda vals: vals["er"] / 0.5))
    out = eng.evaluate_cascade({"w": 3.0})
    assert out["er"] == 6.0
    assert out["sharpe"] == 12.0


def test_token_healed_splice():
    healer = TokenHealer()
    jp = healer.heal_and_splice(["tok"], "abc")
    assert isinstance(jp, JumpForwardPrefill)
    assert jp.splice_string == "abc"
    assert jp.new_token_ids == list("abc")
