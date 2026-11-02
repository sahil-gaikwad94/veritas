"""Formatted JSON strings: witnesses, decoding and pre-flight.

JSON Schema ``format`` values (``date-time``, ``uuid``, ``email``, ``ipv4`` ...) are checked
by the concrete validator, so a constrained decoder must be able to *produce* a conforming
value.  These tests pin the type lowering, the witness table (every candidate is
re-validated), the decode path, and the honest fallback for unknown formats.
"""

import ipaddress
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel, Field, HttpUrl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import veritas as vt
from veritas import textformats
from veritas.compiler.schema_lowering import lower_pydantic_model
from veritas.decode import DecodeConfig, VeritasSession


class Formatted(BaseModel):
    ts: datetime
    day: date
    at: time
    dur: timedelta
    uid: UUID
    amount: Decimal = Field(ge=0, le=1000)
    url: HttpUrl
    ip: ipaddress.IPv4Address
    contact: str = Field(json_schema_extra={"format": "email", "maxLength": 40})


@vt.verified_schema()
def formatted_spec(p: Formatted):
    vt.require(p.amount >= 0)


def test_types_lower_to_formatted_strings():
    lowered = lower_pydantic_model(Formatted)
    kinds = {
        field["name"]: (field["type"].kind, field["type"].text_format)
        for field in lowered["fields"]
    }
    assert kinds["ts"] == ("string", "date-time")
    assert kinds["day"] == ("string", "date")
    assert kinds["at"] == ("string", "time")
    assert kinds["dur"] == ("string", "duration")
    assert kinds["uid"] == ("string", "uuid")
    assert kinds["url"][1] == "uri"
    assert kinds["ip"] == ("string", "ipv4")
    assert kinds["contact"] == ("string", "email")
    # Decimal is numeric: JSON Schema renders it as number-or-string and we use the
    # contractible branch.
    assert kinds["amount"] == ("real", None)


@pytest.mark.parametrize("fmt", sorted(textformats.SUPPORTED_FORMATS))
def test_every_declared_candidate_is_really_valid(fmt):
    """The witness table is self-checking: no candidate may fail the format checker."""

    for candidate in textformats.FORMAT_CANDIDATES[fmt]:
        assert textformats.validate(fmt, candidate) is True, (fmt, candidate)
    assert textformats.witness(fmt) is not None


def test_witness_honours_the_length_window():
    assert textformats.witness("email", max_len=5) is None
    assert textformats.witness("uuid", min_len=40) is None
    found = textformats.witness("uuid", min_len=36, max_len=36)
    assert found is not None and len(found) == 36


def test_unknown_format_has_no_witness():
    assert textformats.witness("colour") is None
    assert textformats.validate("colour", "red") is None


def test_generation_emits_a_conforming_value_for_every_format():
    result = vt.generate(model=vt.MockCharModel(), spec=formatted_spec)
    assert isinstance(result.value.ts, datetime)
    assert isinstance(result.value.day, date)
    assert isinstance(result.value.at, time)
    assert isinstance(result.value.dur, timedelta)
    assert isinstance(result.value.uid, UUID)
    assert isinstance(result.value.amount, Decimal)
    assert str(result.value.url).startswith("https://")
    assert isinstance(result.value.ip, ipaddress.IPv4Address)
    assert "@" in result.value.contact
    assert result.telemetry["format_witnesses"] >= 8
    assert result.telemetry["format_fallbacks"] == []


def test_format_witnesses_are_deterministic_and_off_the_sampling_path():
    session = VeritasSession(formatted_spec, vt.MockCharModel(), DecodeConfig())
    first = session.run(vt.MockCharModel())
    second = session.run(vt.MockCharModel())
    assert first.text == second.text
    assert first.report.status == "PASS"


def test_preflight_finds_a_witness_for_formatted_schemas():
    result = vt.check_spec(formatted_spec.spec)
    assert result.status == "SAT", result.details
    report = vt.verify_values(formatted_spec.spec, result.witness, derive_missing=False)
    assert report.status == "PASS", report.summary_line()


def test_unsupported_format_falls_back_and_stays_honest():
    class Unknown(BaseModel):
        colour: str = Field(min_length=1, json_schema_extra={"format": "colour"})

    @vt.verified_schema()
    def unknown_spec(p: Unknown):
        vt.require(p.colour != "")

    session = VeritasSession(unknown_spec, vt.MockCharModel(), DecodeConfig())
    result = session.run(vt.MockCharModel())
    assert result.telemetry.format_fallbacks, "unknown formats must be reported"
    assert "colour" in result.telemetry.format_fallbacks[0]
    # No format witness was invented; the ordinary string mask ran instead.
    assert result.telemetry.format_witnesses == 0
    assert result.report.status == "PASS"


def test_pattern_wins_when_it_disagrees_with_the_format_witness():
    """A pattern is conjunctive with the format; the canonical witness must yield to it."""

    class Pinned(BaseModel):
        ts: str = Field(
            pattern=r"^1999-01-01T00:00:00Z$",
            json_schema_extra={"format": "date-time"},
        )

    @vt.verified_schema()
    def pinned_spec(p: Pinned):
        vt.require(p.ts != "")

    session = VeritasSession(pinned_spec, vt.MockCharModel(), DecodeConfig())
    result = session.run(vt.MockCharModel())
    assert result.ok, result.report.summary_line()
    assert result.values["ts"] == "1999-01-01T00:00:00Z"
    # The canonical witness ("2024-01-01T00:00:00Z") does not match, so the format path
    # was abandoned rather than emitting a value that violates the pattern.
    assert result.telemetry.format_witnesses == 0
    assert any("pattern" in entry for entry in result.telemetry.format_fallbacks)


def test_verify_instance_accepts_typed_model_values():
    """The audit bug: typed values (datetime/UUID/Decimal/Url) had no JSON rendering, so
    verifying a real model instance failed as a *structural* error."""

    instance = Formatted(
        ts=datetime(2024, 1, 2, 3, 4, 5, tzinfo=None),
        day=date(2024, 1, 2),
        at=time(3, 4, 5),
        dur=timedelta(minutes=90),
        uid=UUID("123e4567-e89b-42d3-a456-426614174000"),
        amount=Decimal("12.5"),
        url="https://example.org/x",
        ip=ipaddress.IPv4Address("203.0.113.9"),
        contact="a@b.example",
    )
    report = vt.verify_instance(formatted_spec, instance)
    assert report.status == "PASS", report.summary_line()

    bad = instance.model_copy(update={"amount": Decimal("-1")})
    failed = vt.verify_instance(formatted_spec, bad)
    assert failed.status == "FAIL"
    assert failed.violations and "amount" in failed.violations[0].text


def test_json_rendering_matches_pydantic_json_mode():
    from veritas.json_pda import json_value, parse_json

    rendered = json_value(
        {
            "ts": datetime(2024, 1, 1, tzinfo=__import__("datetime").timezone.utc),
            "dur": timedelta(hours=2),
            "uid": UUID(int=0),
            "amount": Decimal("9.25"),
            "host": ipaddress.IPv6Address("2001:db8::1"),
        }
    )
    payload = parse_json(rendered)
    assert payload == {
        "ts": "2024-01-01T00:00:00Z",
        "dur": "PT2H",
        "uid": "00000000-0000-0000-0000-000000000000",
        "amount": 9.25,
        "host": "2001:db8::1",
    }
