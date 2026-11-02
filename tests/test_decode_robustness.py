"""Regression suite for the hardened decode/verify paths.

Every test here pins a defect that was found by probing the shipped engine:

* a structural (JSON-Schema) failure used to produce a report with **no** outcomes, so
  ``summary_line()`` read ``0 satisfied, 0 violated, 0 unevaluated``;
* ``gt``/``lt`` open bounds were accepted at the excluded endpoint;
* two contradictory open bounds were reported ``UNKNOWN`` instead of ``INFEASIBLE``;
* container element bounds ignored open bounds entirely;
* ``vt.prefer`` was verified but never steered generation;
* the CRANE scratchpad, token-healing splice plans and soft-preference repair loop were
  documented but unimplemented.
"""

import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import veritas as vt
from veritas.decode import DecodeConfig, VeritasSession

HIGH_FIRST = vt.MockCharModel()
LOW_FIRST = vt.MockCharModel(digit_order="0123456789")


# ---------------------------------------------------------------------------
# structural diagnostics
# ---------------------------------------------------------------------------

class Required(BaseModel):
    a: int
    b: int


@vt.verified_schema()
def required_spec(p: Required):
    vt.require(p.a > 3)


def test_missing_field_reports_an_unevaluated_outcome():
    report = vt.verify_values(required_spec.spec, {"a": 5})
    assert report.status == "INCOMPLETE"
    assert report.missing == ["b"]
    assert report.outcomes, "structural failures must not produce an empty outcome list"
    assert [o.satisfied for o in report.outcomes] == [None]
    assert "b" in report.outcomes[0].text
    assert "1 unevaluated" in report.summary_line()


def test_type_error_reports_a_violated_outcome_naming_the_field():
    report = vt.verify_values(required_spec.spec, {"a": "not-a-number", "b": 1})
    assert report.status == "FAIL"
    assert report.outcomes
    assert all(o.satisfied is False for o in report.outcomes)
    assert any("a" in o.text for o in report.outcomes)
    assert report.unsat_core == [o.index for o in report.outcomes]


def test_failure_text_names_the_offending_field():
    report = vt.verify_values(required_spec.spec, {"a": 5})
    message = str(vt.VerificationError(report))
    assert "b" in message
    assert "0 satisfied, 0 violated, 0 unevaluated" not in message


# ---------------------------------------------------------------------------
# open (strict) bounds
# ---------------------------------------------------------------------------

class OpenFloat(BaseModel):
    a: float = Field(gt=0.0)
    b: float = Field(gt=0.0)


@vt.verified_schema()
def open_float_spec(p: OpenFloat):
    vt.require(p.a + p.b == 1.0)


class OpenInt(BaseModel):
    a: int = Field(gt=0, lt=3)
    b: int = Field(ge=0, le=2)


@vt.verified_schema()
def open_int_spec(p: OpenInt):
    vt.require(p.a + p.b >= 1)


@pytest.mark.parametrize("model", [HIGH_FIRST, LOW_FIRST])
def test_open_float_bounds_are_never_satisfied_at_the_endpoint(model):
    result = vt.generate(model=model, spec=open_float_spec)
    assert result.value.a > 0.0 and result.value.b > 0.0
    assert abs(result.value.a + result.value.b - 1.0) <= 1e-6


@pytest.mark.parametrize("model", [HIGH_FIRST, LOW_FIRST])
def test_open_integer_bounds_generate_inside_the_open_interval(model):
    result = vt.generate(model=model, spec=open_int_spec)
    assert result.value.a in (1, 2)
    assert 0 <= result.value.b <= 2


def test_contradictory_open_real_bounds_fail_preflight():
    with pytest.raises(vt.InfeasibleSpecError):
        @vt.verified_schema()
        def contradiction(p: OpenFloat):
            vt.require(p.a <= 0.0)


def test_contradictory_integer_open_bounds_fail_preflight():
    class Narrow(BaseModel):
        n: int = Field(ge=0)

    with pytest.raises(vt.InfeasibleSpecError):
        @vt.verified_schema()
        def contradiction_int(p: Narrow):
            vt.require(p.n >= 1)
            vt.require(p.n < 1)


def test_strict_real_inequality_pair_fails_preflight():
    class Free(BaseModel):
        x: float

    with pytest.raises(vt.InfeasibleSpecError):
        @vt.verified_schema()
        def contradiction_real(p: Free):
            vt.require(p.x >= 1.0)
            vt.require(p.x < 1.0)


# ---------------------------------------------------------------------------
# container element bounds
# ---------------------------------------------------------------------------

class OpenList(BaseModel):
    xs: list[float] = Field(min_length=2, max_length=2)


@vt.verified_schema()
def open_list_spec(p: OpenList):
    vt.require(vt.forall(p.xs, lambda v: (0.2 < v) & (v < 0.4)))


class OpenIntList(BaseModel):
    xs: list[int] = Field(min_length=2, max_length=2)


@vt.verified_schema()
def open_int_list_spec(p: OpenIntList):
    vt.require(vt.forall(p.xs, lambda v: (1 < v) & (v < 4)))


@pytest.mark.parametrize("model", [HIGH_FIRST, LOW_FIRST])
def test_open_element_bounds_are_enforced_for_floats(model):
    result = vt.generate(model=model, spec=open_list_spec)
    for value in result.value.xs:
        assert 0.2 < value < 0.4


def test_open_element_bounds_are_enforced_for_integers():
    result = vt.generate(model=LOW_FIRST, spec=open_int_list_spec)
    assert all(value in (2, 3) for value in result.value.xs)


def test_element_exclusivity_reaches_the_container_analysis():
    session = VeritasSession(open_list_spec, LOW_FIRST, DecodeConfig())
    info = session.containers["xs"]
    assert info.elem_min == pytest.approx(0.2)
    assert info.elem_max == pytest.approx(0.4)
    assert info.elem_min_exclusive and info.elem_max_exclusive


# ---------------------------------------------------------------------------
# string masking (length + pattern)
# ---------------------------------------------------------------------------

class Coded(BaseModel):
    code: str = Field(pattern=r"^[A-Z]{3}$", min_length=3, max_length=3)


@vt.verified_schema()
def coded_spec(p: Coded):
    vt.require(p.code != "")


class Choice(BaseModel):
    code: str = Field(pattern=r"^(foo|ba[rz])$")


@vt.verified_schema()
def choice_spec(p: Choice):
    vt.require(p.code != "")
    vt.require(p.code != "foobar")


def test_pattern_masking_emits_a_matching_string():
    result = vt.generate(model=HIGH_FIRST, spec=coded_spec)
    assert len(result.value.code) == 3
    assert result.value.code.isupper()
    assert result.telemetry["string_mask_prunes"] >= 0


def test_pattern_masking_handles_alternation():
    result = vt.generate(model=LOW_FIRST, spec=choice_spec)
    assert result.value.code in {"foo", "bar", "baz"}


def test_unsupported_pattern_falls_back_honestly():
    class Insensitive(BaseModel):
        s: str = Field(pattern=r"(?i)abc", min_length=3, max_length=3)

    @vt.verified_schema()
    def insensitive_spec(p: Insensitive):
        vt.require(p.s != "")

    session = VeritasSession(insensitive_spec, LOW_FIRST, DecodeConfig())
    result = session.run(LOW_FIRST)
    assert result.telemetry.pattern_fallbacks
    assert any("case-insensitive" in entry for entry in result.telemetry.pattern_fallbacks)
    # The mask is weaker, so the verifier must reject the mismatch, not hide it.
    assert result.report.status == "FAIL"
    assert any("does not match" in outcome.text for outcome in result.report.outcomes)


def test_vir_round_trips_pattern_and_format():
    from veritas.serde import spec_from_dict, spec_to_dict

    restored = spec_from_dict(spec_to_dict(coded_spec.spec))
    vtype = restored.variables[restored.name_to_id["code"]]["type"]
    assert vtype.pattern == r"^[A-Z]{3}$"


# ---------------------------------------------------------------------------
# soft preferences
# ---------------------------------------------------------------------------

class Soft(BaseModel):
    x: float


@vt.verified_schema()
def soft_spec(p: Soft):
    vt.require(0.0 <= p.x)
    vt.require(p.x <= 1.0)
    vt.prefer(p.x >= 0.8, weight=2.0, tag="high")


@vt.verified_schema()
def two_soft_spec(p: Soft):
    vt.require(0.0 <= p.x)
    vt.require(p.x <= 1.0)
    vt.prefer(p.x >= 0.5, weight=1.0)
    vt.prefer(p.x <= 0.6, weight=5.0)


def test_preference_repair_steers_a_disagreeing_model():
    # The model wants 0; the soft bound has to pull it up to the preferred region.
    result = vt.generate(model=LOW_FIRST, spec=soft_spec)
    assert result.value.x >= 0.8 - 1e-9
    assert result.telemetry["preference_score"] == pytest.approx(1.0)
    assert result.telemetry["preference_repairs"] >= 1


def test_preference_repair_intersects_multiple_soft_bounds():
    result = vt.generate(model=LOW_FIRST, spec=two_soft_spec)
    assert 0.5 - 1e-9 <= result.value.x <= 0.6 + 1e-9
    assert all(outcome.satisfied for outcome in result.report.preferences)


def test_preferences_remain_soft_and_never_break_hard_constraints():
    result = vt.generate(model=HIGH_FIRST, spec=soft_spec)
    assert 0.0 <= result.value.x <= 1.0
    assert result.telemetry["preference_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# reasoning scratchpad and token healing
# ---------------------------------------------------------------------------

class Small(BaseModel):
    a: float = Field(ge=0.0, le=1.0)


@vt.verified_schema()
def small_spec(p: Small):
    vt.require(p.a + p.a <= 1.0)


class Tokenizer:
    """Minimal tokenizer protocol: only ``encode``/``decode`` are required."""

    def __init__(self):
        self.vocabulary = ['{"a": ', "0.5", "}"]
        self._ids = {token: index for index, token in enumerate(self.vocabulary)}

    def encode(self, text):
        return [self._ids.get(text, len(self.vocabulary))]

    def decode(self, ids):
        return "".join(
            self.vocabulary[i] if i < len(self.vocabulary) else "?" for i in ids
        )


def test_crane_scratchpad_stays_out_of_the_json():
    model = vt.StructuredReasoningModel(reasoning="let a be small; a + a <= 1")
    result = vt.generate(model=model, spec=small_spec, reasoning_mode="crane_cot")
    assert result.reasoning
    assert result.reasoning not in result.raw
    assert result.telemetry["reasoning_chars"] == len(result.reasoning)
    json.loads(result.raw)  # the structured span is still pure JSON


def test_strict_reasoning_mode_disables_the_scratchpad():
    model = vt.StructuredReasoningModel()
    result = vt.generate(model=model, spec=small_spec, reasoning_mode="strict")
    assert result.reasoning == ""
    assert result.telemetry["reasoning_chars"] == 0


def test_schema_level_reasoning_mode_is_honoured():
    # ``@vt.verified_schema`` defaults to ``reasoning_mode="crane_cot"``; the config value
    # ``auto`` must defer to it rather than silently disabling the scratchpad.
    model = vt.StructuredReasoningModel(reasoning="scratch note")
    result = vt.generate(model=model, spec=small_spec, reasoning_mode="auto")
    assert result.reasoning == "scratch note"


def test_unknown_reasoning_mode_is_rejected():
    with pytest.raises(ValueError):
        vt.generate(model=HIGH_FIRST, spec=small_spec, reasoning_mode="chain-of-thought")


class Derived(BaseModel):
    a: float = Field(ge=0.0, le=1.0)
    double: float


@vt.verified_schema()
def derived_spec(p: Derived):
    vt.derive(p.double == p.a * 2)


def test_zero_dof_field_is_spliced_without_sampling():
    session = VeritasSession(derived_spec, HIGH_FIRST, DecodeConfig())
    result = session.run(HIGH_FIRST)
    assert result.ok, result.report.summary_line()
    assert result.telemetry.jump_forward_fields == 1
    assert result.values["double"] == pytest.approx(2 * result.values["a"])


def test_token_healing_plans_a_splice_when_a_tokenizer_is_present():
    result = vt.generate(model=HIGH_FIRST, spec=derived_spec, tokenizer=Tokenizer())
    assert result.splices, "a tokenizer-enabled decode should plan token-healed splices"
    splice = result.splices[0]
    assert splice["field"] == "double"
    assert splice["token_ids"]
    assert splice["text"].endswith(str(splice["value"]))
    assert result.telemetry["jump_forward_tokens"] == len(splice["token_ids"])


def test_token_healing_is_disabled_without_a_tokenizer():
    result = vt.generate(model=HIGH_FIRST, spec=derived_spec)
    assert result.splices == []


# ---------------------------------------------------------------------------
# numeric search and telemetry honesty
# ---------------------------------------------------------------------------

def test_number_search_backtracks_within_a_field():
    """A greedy literal (0.4) must be abandoned for a feasible sibling (0.3)."""

    session = VeritasSession(open_list_spec, HIGH_FIRST, DecodeConfig())
    result = session.run(HIGH_FIRST)
    assert result.ok, result.report.summary_line()
    assert all(0.2 < value < 0.4 for value in result.values["xs"])
    assert result.telemetry.number_expansions >= 2


class HalfBox(BaseModel):
    x: float = Field(gt=0.0, lt=1.0)


@vt.verified_schema()
def half_box_spec(p: HalfBox):
    vt.require(p.x * 2 <= 1.0)


@pytest.mark.parametrize("model", [HIGH_FIRST, LOW_FIRST])
def test_no_phantom_domain_cut_for_feasible_specs(model):
    session = VeritasSession(half_box_spec, model, DecodeConfig())
    result = session.run(model)
    assert result.ok, result.report.summary_line()
    assert 0.0 < result.values["x"] < 1.0
    assert result.telemetry.domain_cuts == []
    assert result.telemetry.backtrack_fields == []


def test_domain_cut_is_recorded_with_culprit_and_bounds():
    class Product(BaseModel):
        a: float = Field(ge=0.0, le=1.0)
        b: float = Field(ge=0.0, le=1.0)

    @vt.verified_schema()
    def product_spec(p: Product):
        vt.require(p.a * p.b <= 0.1, tol=1e-6)

    session = VeritasSession(product_spec, HIGH_FIRST, DecodeConfig())
    result = session.run(HIGH_FIRST)
    assert result.ok, result.report.summary_line()
    assert result.telemetry.backtracks >= 1
    assert result.telemetry.domain_cuts, "a nonlinear violation should cut a domain"
    cut = result.telemetry.domain_cuts[0]
    assert cut["field"] in {"a", "b"}
    assert cut["field"] in result.telemetry.backtrack_fields
    assert cut["bounds"][0] <= cut["bounds"][1]
    # Cuts only ever narrow the declared box, so the accepted value must respect every
    # bound that was recorded for its field.
    for recorded in result.telemetry.domain_cuts:
        if recorded["field"] == cut["field"]:
            low, high = recorded["bounds"]
            assert low - 1e-9 <= result.values[cut["field"]] <= high + 1e-9


# ---------------------------------------------------------------------------
# pattern witnesses in pre-flight
# ---------------------------------------------------------------------------

class Ticket(BaseModel):
    code: str = Field(pattern=r"^[A-Z]{2}-\d{3}$", min_length=6, max_length=6)
    size: str = Field(pattern=r"^(small|large)$")
    ratio: float = Field(gt=0.0, lt=1.0)


@vt.verified_schema()
def ticket_spec(p: Ticket):
    vt.require(p.ratio * 2 <= 1.0)


def test_preflight_produces_a_checked_witness_for_pattern_strings():
    result = vt.check_spec(ticket_spec.spec)
    assert result.status == "SAT", result.details
    assert result.witness is not None
    report = vt.verify_values(ticket_spec.spec, result.witness, derive_missing=False)
    assert report.status == "PASS", report.summary_line()


def test_pattern_witness_respects_the_length_window():
    from veritas.pattern import PatternConstraint

    constraint = PatternConstraint.compile(r"^[a-z]+$")
    witness = constraint.witness(min_length=6, max_length=8)
    assert witness is not None and 6 <= len(witness) <= 8
    assert constraint.satisfied(witness)


def test_preflight_still_reports_unknown_without_a_pattern_witness():
    class Impossible(BaseModel):
        code: str = Field(pattern=r"^[A-Z]{5}$", min_length=2, max_length=3)

    @vt.verified_schema()
    def impossible_spec(p: Impossible):
        vt.require(p.code != "")

    result = vt.check_spec(impossible_spec.spec)
    # A pattern that cannot fit the declared length window has no witness; pre-flight must
    # say so instead of inventing a value.
    assert result.status in {"UNKNOWN", "UNSAT"}
