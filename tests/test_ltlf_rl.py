"""Finite-trace temporal logic and the RL data flywheel."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import vexx as vt
from vexx.ltlf import Trace, at_most, before, eventually, never, occurred, responds
from vexx.rl.collector import StepRecord, VexxRLCollector


def _trace():
    trace = Trace()
    trace.append("verify_kyc")
    trace.append("check_balance")
    trace.append("transfer", {"amount": 100})
    return trace


def test_before_and_responds():
    trace = _trace()
    assert before("verify_kyc", "transfer").evaluate(trace)
    assert before("check_balance", "transfer").evaluate(trace)
    assert not before("transfer", "verify_kyc").evaluate(trace)
    assert responds("verify_kyc", "transfer").evaluate(trace)
    assert not responds("transfer", "verify_kyc").evaluate(trace)


def test_always_eventually_never_at_most():
    trace = _trace()
    assert always_holds(trace)
    assert eventually(occurred("transfer")).evaluate(trace)
    assert never("delete_account").evaluate(trace)
    assert at_most("transfer", 1).evaluate(trace)
    assert not at_most("transfer", 0).evaluate(trace)


def always_holds(trace):
    # Every prefix satisfies "at most one transfer".
    return vt.always(lambda t: t.count("transfer") <= 1).evaluate(trace)


def test_formula_composition():
    trace = _trace()
    formula = occurred("verify_kyc") & (~occurred("delete_account"))
    assert formula.evaluate(trace)
    assert (~formula).evaluate(Trace())


def test_world_state_tool_gating():
    state = vt.WorldState()
    state.declare_tool("verify_kyc", effects=["kyc_verified"])
    state.declare_tool("transfer", requires=["kyc_verified"])
    assert not state.can_call("transfer")
    state.record_call("verify_kyc")
    assert state.can_call("transfer")
    state.record_call("transfer", amount=100)
    assert state.trace.count("transfer") == 1


def test_rl_exports_and_manifest(tmp_path):
    collector = VexxRLCollector()
    collector.add_preference(
        prompt="allocate",
        chosen='{"a": 0.5, "b": 0.5}',
        rejected='{"a": 0.9, "b": 0.9}',
        unsat_core=[2],
        margin=0.8,
    )
    collector.record_step(
        StepRecord(prompt="allocate", completion='{"a": 0.5}', rewards=[1.0, 0.1], metadata={"backtracks": 1})
    )
    dpo = collector.export_step_dpo(tmp_path)
    grpo = collector.export_grpo(tmp_path)
    assert dpo["num_records"] == 1
    assert grpo["num_records"] == 1
    rows = [json.loads(line) for line in Path(dpo["file"]).read_text().splitlines()]
    assert rows[0]["chosen"] and rows[0]["rejected"]
    manifest = json.loads(Path(dpo["manifest"]).read_text())
    assert manifest["rows"] == 1
    assert dpo["sha256"] == collector.export_step_dpo(tmp_path)["sha256"]


def test_process_reward_is_dense_and_bounded():
    rewards = VexxRLCollector.process_reward(
        verified=True, backtracks=2, jump_forward_tokens=10, final_margin=0.0, max_backtracks=8
    )
    assert rewards[0] == 1.0
    assert 0 < rewards[1] < 0.25
    assert all(-2.0 <= r <= 2.0 for r in rewards)
