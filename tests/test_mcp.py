"""Tests for the stateful MCP firewall (``vexx.mcp`` / ``vt.VexxMcpSession``)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pydantic import BaseModel

import vexx as vt
from vexx import mcp
from vexx.cartridges.sec import PolicyViolation
from vexx.mcp import McpState, UntrustedString, VexxMcpSession


class RefundArgs(BaseModel):
    order_id: str
    refund_amount: float
    reason_code: str


def _refund_session() -> VexxMcpSession:
    session = VexxMcpSession(
        temporal_policy=vt.before("lookup_order", "issue_refund"),
        max_session_spend=100.0,
    )

    @session.verified_tool(name="lookup_order", cost=1.0)
    def lookup_order(args, state: McpState):
        return {"order": {"id": "ORD-99", "total": 142.50}}

    @session.verified_tool(name="issue_refund", cost=5.0, requires_taint_free=True)
    def issue_refund(args: RefundArgs, state: McpState):
        mcp.mcp_require(args.order_id == state.env["last_order_id"], "order mismatch")
        mcp.mcp_require(args.refund_amount <= state.env["last_order_total"], "refund exceeds order total")
        mcp.mcp_require(mcp.is_untainted(args.reason_code), "tainted reason code")
        return {"refunded": args.refund_amount, "order_id": args.order_id}

    return session


def test_temporal_gate_blocks_out_of_order_call_fail_closed():
    session = _refund_session()
    assert session.allowed_tools() == ["lookup_order"]
    with pytest.raises(Exception) as excinfo:
        session._tools["issue_refund"]["fn"](
            {"order_id": "ORD-99", "refund_amount": 1.0, "reason_code": "damaged"}
        )
    assert "temporal policy" in str(excinfo.value)


def test_allowed_tools_after_lookup():
    session = _refund_session()
    session._tools["lookup_order"]["fn"]({})
    assert session.allowed_tools() == ["lookup_order", "issue_refund"]


def test_env_binding_and_accumulation():
    session = _refund_session()
    session._tools["lookup_order"]["fn"]({})
    assert session.state.env["last_order_id"] == "ORD-99"
    assert session.state.env["last_order_total"] == 142.50
    assert session.state.accumulated["total"] == 142.50
    out = session._tools["issue_refund"]["fn"](
        {"order_id": "ORD-99", "refund_amount": 42.5, "reason_code": "damaged"}
    )
    assert out["refunded"] == 42.5
    assert session.state.accumulated["refunded"] == 42.5
    assert session.spend == 6.0


def test_require_enforces_env_bound():
    session = _refund_session()
    session._tools["lookup_order"]["fn"]({})
    with pytest.raises(PolicyViolation) as excinfo:
        session._tools["issue_refund"]["fn"](
            {"order_id": "ORD-99", "refund_amount": 999.0, "reason_code": "ok"}
        )
    assert "refund exceeds order total" in str(excinfo.value)


def test_tainted_input_rejected():
    session = _refund_session()
    session._tools["lookup_order"]["fn"]({})
    tainted = vt.taint("refund me please", ["email"])
    with pytest.raises(Exception) as excinfo:
        session._tools["issue_refund"]["fn"](
            {"order_id": "ORD-99", "refund_amount": 1.0, "reason_code": tainted}
        )
    assert "tainted" in str(excinfo.value).lower()


def test_tainted_nested_input_rejected():
    session = VexxMcpSession(max_session_spend=10.0)

    @session.verified_tool(name="webhook_ingest", requires_taint_free=True)
    def ingest(args, state):
        return {"ok": True}

    with pytest.raises(Exception) as excinfo:
        session._tools["webhook_ingest"]["fn"]({"body": vt.taint("click me", ["email"])})
    assert "tainted" in str(excinfo.value).lower()


def test_spend_cap():
    session = VexxMcpSession(max_session_spend=2.0)

    @session.verified_tool(name="report", cost=10.0)
    def report(args, state):
        return {"ok": True}

    with pytest.raises(Exception) as excinfo:
        session._tools["report"]["fn"]({})
    assert "spend" in str(excinfo.value).lower()


def test_on_violation_return_mode():
    session = VexxMcpSession(
        temporal_policy=vt.before("lookup_order", "issue_refund"),
        on_violation="return",
    )

    @session.verified_tool(name="issue_refund")
    def issue_refund(args, state):
        return {"ok": True}

    result = session._tools["issue_refund"]["fn"]({})
    assert result["ok"] is False
    assert result["policy"] == "temporal"


def test_untrusted_string_survives_pydantic_validation():
    class StrictArgs(BaseModel):
        note: UntrustedString

    tainted = vt.taint("hello", ["email"])
    model = StrictArgs(note=tainted)
    assert isinstance(model.note, UntrustedString)
    assert model.note.sources == frozenset({"email"})
    assert not mcp.is_untainted(model)
    assert model.model_json_schema()["properties"]["note"]["type"] == "string"


def test_record_observation_binds_env():
    session = VexxMcpSession()
    session.record_observation("lookup_order", {"order_id": "ORD-77", "total": 55.0})
    assert session.state.env["last_order_id"] == "ORD-77"
    assert session.state.env["last_total"] == 55.0


def test_require_inside_tool_blocks_call():
    session = VexxMcpSession()

    @session.verified_tool(name="guarded")
    def guarded(args, state):
        mcp.mcp_require(False, "guard says no")
        return {"ok": True}

    with pytest.raises(Exception) as excinfo:
        session._tools["guarded"]["fn"]({})
    assert "guard says no" in str(excinfo.value)
