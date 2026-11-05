"""Solver-guided decoding: the reference neuro-symbolic decode engine.

This module is the runtime spine the specification describes:

1. **Tier-0** — the canonical JSON writer and grammar (keys are emitted deterministically;
   values are validated against the JSON pushdown automaton).
2. **Tier-1** — numeric interval masking via :class:`~vexx.interval.Interval` prefix
   analysis, cross-field contraction through :class:`Hc4AffineContractor`, and AGDecode
   residual lookahead for dynamic containers.
3. **0-DoF Jump-Forward** — derived fields are computed and spliced without sampling.
4. **Tier-2** — every completed object is verified by :func:`verify_values`; failures are
   attributed to a culprit field via the CDCL engine, which applies a domain cut and a
   tabu value before regenerating.

The engine is deliberately model-agnostic: a model only has to expose
``next_logits(prefix_text) -> Mapping[str, float]`` over *text units*, which makes the
whole system deterministically testable without a GPU.  Production bridges adapt token
logits to this interface.
"""

from __future__ import annotations

import math
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import textformats
from .cdcl_jumpback import CdclJumpbackEngine, ClauseCheckpoint
from .compiler.vir import ContainerInfo, VirSpec, VirType
from .crane_mode import CraneModeController
from .evaluate import as_number, collect_var_ids
from .instance import VerificationError, VerificationReport, _coerce_spec, verify_values
from .interval import Hc4AffineContractor, Interval
from .json_pda import (
    JSONPushdownAutomaton,
    is_complete_json,
    json_number,
    json_value,
    render_object,
)
from .linear import compile_clauses, simplify_clause
from .pattern import PatternConstraint, UnsupportedPattern
from .proof.certificate import ProofCertificate, VerifiedResult, build_certificate
from .rl.collector import ConflictBranchRecord, VexxRLCollector
from .serde import spec_digest
from .token_healer import TokenHealer
from .zero_dof import ZeroDofEngine

_NUMBER_COMPLETE = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$")
_CONTAINER_KINDS = {"dynamic_map", "list", "dict", "set"}
_STOP = "<stop>"
_DIGITS = "0123456789"


class DecodeError(RuntimeError):
    """Base class for decode failures."""


class ModelProtocolError(DecodeError):
    """The supplied model does not implement the text-logit protocol."""


class DeadEnd(DecodeError):
    def __init__(self, field_name: str | None, reason: str):
        self.field_name = field_name
        self.reason = reason
        super().__init__(f"dead end at {field_name!r}: {reason}")


@dataclass
class DecodeConfig:
    temperature: float = 0.0  # 0 => deterministic argmax
    max_backtracks: int = 8
    max_steps: int = 4096
    max_integer_digits: int = 4
    max_fraction_digits: int = 6
    max_string_length: int = 64
    allow_exponent: bool = False
    volume_twist_beta: float = 0.0
    #: ``auto`` honours the schema's ``reasoning_mode``; ``strict`` never opens a
    #: scratchpad; ``crane_cot`` forces one when the model can supply reasoning text.
    reasoning_mode: str = "auto"
    reasoning_max_chars: int = 8192
    max_preference_repairs: int = 2
    #: Node budget for the per-field backtracking search over number literals.
    max_number_nodes: int = 512
    seed: int | None = 0
    derive_missing: bool = True
    enforce_grammar: bool = True
    #: If True, automatically compute integer/fraction digit limits from field bounds
    #: when the configured limits would be too restrictive.
    auto_scale_numeric_limits: bool = True
    #: Default numeric tolerance for cartridge validators and constraint checks.
    numeric_tolerance: float = 1e-9
    #: Optional path to a JSONL trace of conflict-branch records for ``vexx export-rl``.
    rl_trace_path: str | None = None
    #: If True, on CDCL backtrack the engine re-opens a bounded CRANE scratchpad seeded
    #: with the algebraic conflict certificate (field, reason, domain cut).  The notes are
    #: appended to ``result.reasoning`` — never to the decoded JSON — and keep their
    #: ``<|solver_feedback|>`` markers so a downstream tool can lift them verbatim.
    reflex_on_backtrack: bool = False
    #: Cap on the characters of the adaptive feedback span per backtrack.
    reflex_max_chars: int = 512


@dataclass
class StepTrace:
    step: int
    attempt: int
    field_name: str
    char: str
    kind: str = "sample"  # sample | splice | close | backtrack | error
    allowed: int = 0
    candidates: int = 0
    masked: int = 0
    entropy: float = 0.0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "attempt": self.attempt,
            "field": self.field_name,
            "text": self.char,
            "kind": self.kind,
            "allowed": self.allowed,
            "masked": self.masked,
            "entropy": round(self.entropy, 6),
            "note": self.note,
        }


@dataclass
class DecodeTelemetry:
    steps: int = 0
    sampled_units: int = 0
    masked_candidates: int = 0
    solver_checks: int = 0
    contractor_prunes: int = 0
    string_mask_prunes: int = 0
    format_witnesses: int = 0
    number_expansions: int = 0
    jump_forward_fields: int = 0
    jump_forward_chars: int = 0
    jump_forward_tokens: int = 0
    token_heal_exact: int = 0
    twist_adjustments: int = 0
    reasoning_chars: int = 0
    mined_equations: int = 0
    preference_repairs: int = 0
    preference_score: float | None = None
    backtracks: int = 0
    #: Attempts consumed by the engine before producing the returned text.
    attempts: int = 1
    #: Sum of ``-log(accepted probability mass)`` accumulated per field (Horizon 4.2):
    #: the exact KL distortion introduced by hard masking, per JSON field.
    field_kl_divergence: dict[str, float] = field(default_factory=dict)
    kl_distortion_tokens: int = 0
    backtrack_fields: list[str] = field(default_factory=list)
    domain_cuts: list[dict[str, Any]] = field(default_factory=list)
    pattern_fallbacks: list[str] = field(default_factory=list)
    format_fallbacks: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    final_status: str = ""
    spec_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["backtrack_fields"] = list(self.backtrack_fields)
        payload["domain_cuts"] = list(self.domain_cuts)
        payload["elapsed_ms"] = round(self.elapsed_ms, 3)
        return payload


@dataclass
class DecodeResult:
    text: str
    values: dict[str, Any]
    report: VerificationReport
    telemetry: DecodeTelemetry
    trace: list[StepTrace]
    attempts: int = 1
    model_name: str = ""
    prompt: str = ""
    reasoning: str = ""
    mined_equations: list[str] = field(default_factory=list)
    splices: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.report.status == "PASS"

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "values": self.values,
            "status": self.report.status,
            "telemetry": self.telemetry.as_dict(),
            "attempts": self.attempts,
            "model": self.model_name,
            "reasoning": self.reasoning,
            "report": self.report.as_dict(),
        }

    def receipt(self) -> dict[str, Any]:
        """Self-contained verification receipt: what was produced, by whom, at what cost.

        Contains the content hashes (text + spec), the verification outcome, budget
        counters (attempts, backtracks, jump-forward fields) and the per-field KL
        distortion introduced by masking.  For an ed25519 signature attach a
        :func:`vexx.proof.certificate.build_certificate` proof alongside this receipt.
        """

        _sha = _sha256_text
        return {
            "sha256": _sha(self.text),
            "spec_digest": self.telemetry.spec_digest,
            "status": self.report.status,
            "model": self.model_name,
            "prompt_sha256": _sha(self.prompt or ""),
            "attempts": self.attempts,
            "backtracks": self.telemetry.backtracks,
            "jump_forward_fields": self.telemetry.jump_forward_fields,
            "jump_forward_tokens": self.telemetry.jump_forward_tokens,
            "field_kl_divergence": dict(self.telemetry.field_kl_divergence),
            "kl_distortion_tokens": self.telemetry.kl_distortion_tokens,
            "elapsed_ms": round(self.telemetry.elapsed_ms, 3),
            "verified": bool(self.ok),
        }


class VexxSession:
    """One constrained generation over one compiled spec."""

    def __init__(
        self,
        schema_or_spec: Any,
        model: Any = None,
        config: DecodeConfig | None = None,
        *,
        tokenizer: Any = None,
    ):
        self.spec: VirSpec = _coerce_spec(schema_or_spec)
        self.model = model
        self.config = config or DecodeConfig()
        self.rng = random.Random(self.config.seed)
        self.telemetry = DecodeTelemetry(spec_digest=spec_digest(self.spec))
        self.trace: list[StepTrace] = []
        self.prompt = ""
        self.tokenizer = tokenizer
        self.healer = TokenHealer(tokenizer) if _has_tokenizer(tokenizer) else None
        self.splices: list[dict[str, Any]] = []
        self.reasoning = ""
        self.mined_equations: list[str] = []
        self.zero = ZeroDofEngine.from_spec(self.spec)
        self.cdcl = CdclJumpbackEngine()
        self.rl = VexxRLCollector()
        self._pending_conflicts: list[ConflictBranchRecord] = []

        names = list(self.spec.field_order)
        for name in self.spec.name_to_id:
            if name not in names:
                names.append(name)
        self.fields: list[str] = [n for n in names if n in self.spec.name_to_id]
        self.vid: dict[str, int] = {n: self.spec.name_to_id[n] for n in self.fields}
        self.types: dict[str, Any] = {
            n: (self.spec.variables.get(self.vid[n]) or {}).get("type") for n in self.fields
        }
        self.rules = {r.target_json_key: r for r in self.spec.zero_dof_rules if r.target_json_key}
        self.containers: dict[str, ContainerInfo] = {
            info.name: info for info in self.spec.containers.values() if info.name
        }
        self.clauses = compile_clauses(self.spec)
        self._field_clause_index: dict[str, int] = {}
        for clause in self.clauses:
            for vid in collect_var_ids(clause.expr):
                name = self.spec.id_to_name.get(vid)
                if name and name not in self._field_clause_index:
                    self._field_clause_index[name] = clause.index

        # Per-field provenance: real character spans and measured entropies make CDCL
        # culprit attribution meaningful instead of a fixed last-field fallback.
        self._field_index: dict[str, int] = {name: index for index, name in enumerate(self.fields)}
        self._field_span: dict[str, tuple[int, int]] = {}
        self._field_entropy: dict[str, list[float]] = {}
        self._field_tokens: dict[str, str] = {}
        self._conflict_counts: dict[str, int] = {}
        self._field_kl: dict[str, float] = {}
        self._kl_steps = 0
        self._clause_fields: dict[int, list[str]] = {}
        for clause in self.clauses:
            names: list[str] = []
            for vid in collect_var_ids(clause.expr):
                name = self.spec.id_to_name.get(vid)
                if name in self._field_index and name not in names:
                    names.append(name)
            self._clause_fields[clause.index] = names
        self._exclusive: dict[str, tuple[bool, bool]] = {
            name: (bool(getattr(self.types.get(name), "min_exclusive", False)),
                   bool(getattr(self.types.get(name), "max_exclusive", False)))
            for name in self.fields
        }
        self._patterns: dict[str, PatternConstraint | None] = {}
        self._soft_overrides: dict[str, tuple[float, float]] = {}
        self.values: dict[str, Any] = {}
        self.text = ""
        self._attempt = 0
        self._step = 0
        self._partial_numeric: dict[str, float] = {}
        self._attempt_checkpoints: list[ClauseCheckpoint] = []
        self._overrides: dict[str, tuple[float, float]] = {}
        self._elem_overrides: dict[tuple[str, str], tuple[float, float]] = {}
        self._tabu: dict[Any, set[float]] = {}
        self._cut_direction: dict[str, int] = {}
        self._commit_order: list[str] = []
        self.contractor = self._build_contractor()
        # Snapshot of the feasible box *before* any assignment; domain cuts are computed
        # against this, never against a committed singleton.
        self._initial_domains: dict[int, Interval] = {
            key: Interval(bound.lo, bound.hi) for key, bound in self.contractor.domains.items()
        }

    def _compute_numeric_limits(self, name: str) -> tuple[int, int]:
        """Compute reasonable integer/fraction digit limits from field bounds.

        Returns (max_integer_digits, max_fraction_digits) that can represent
        the field's declared range without being overly restrictive.
        """
        vtype = self.types.get(name)
        if vtype is None:
            return self.config.max_integer_digits, self.config.max_fraction_digits

        lo = getattr(vtype, "min", None)
        hi = getattr(vtype, "max", None)
        if lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi):
            return self.config.max_integer_digits, self.config.max_fraction_digits

        # Compute max integer digits needed to represent the range
        max_abs = max(abs(lo), abs(hi))
        if max_abs == 0:
            int_digits = 1
        else:
            int_digits = max(1, int(math.floor(math.log10(max_abs))) + 1)
        # Add 1 for sign if negative values possible
        if lo < 0:
            int_digits += 1

        # For fraction digits, use the configured default unless we can infer
        # from the precision needed to distinguish values in the range
        frac_digits = self.config.max_fraction_digits

        # If the range is very small, we might need more fraction digits
        span = hi - lo
        if span > 0 and span < 1:
            # Estimate digits needed to represent the span
            needed_frac = max(0, int(math.ceil(-math.log10(span))) + 1)
            frac_digits = max(frac_digits, min(needed_frac, 15))  # Cap at 15

        return int_digits, frac_digits

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_contractor(self) -> Hc4AffineContractor:
        contractor = Hc4AffineContractor()
        for name in self.fields:
            vtype = self.types.get(name)
            kind = getattr(vtype, "kind", "any")
            if kind in _CONTAINER_KINDS:
                continue
            lo = getattr(vtype, "min", None)
            hi = getattr(vtype, "max", None)
            if kind == "bool":
                lo, hi = 0.0, 1.0
            lo = float(lo) if lo is not None else float("-inf")
            hi = float(hi) if hi is not None else float("inf")
            # ``gt``/``lt`` are open bounds.  The adjacent representable value is exactly
            # the open interval over floats, so the contractor stays sound and the
            # decoder never accepts the excluded endpoint.
            if getattr(vtype, "min_exclusive", False) and math.isfinite(lo):
                lo = math.nextafter(lo, math.inf)
            if getattr(vtype, "max_exclusive", False) and math.isfinite(hi):
                hi = math.nextafter(hi, math.inf * -1)
            contractor.add_variable(self.vid[name], Interval(lo, hi))
        for clause in self.clauses:
            simple = simplify_clause(clause)
            if simple is not None:
                op, coeffs, rhs = simple
                if not coeffs or any(key not in contractor.domains for key in coeffs):
                    continue
                try:
                    contractor.add_constraint(("linear", dict(coeffs), float(rhs), op))
                except Exception:
                    continue
            # Quadratic constraints: compare-nodes with a quad_form operand are pushed to the
            # contractor as (difference-of-)PSD Cholesky constraints.  The negative part of an
            # indefinite matrix is bounded soundly over the current box (u₋/l₊); we never add
            # sign-flipped independent PSD constraints (they refute every real box).
            if getattr(clause.expr, "kind", None) == "compare":
                spec_constraint = self._quad_demand(clause.expr)
                if spec_constraint is not None:
                    var_ids, l_plus, l_minus, rhs, op = spec_constraint
                    try:
                        if l_minus is None:
                            contractor.add_constraint((
                                "quadratic_form", var_ids, l_plus, rhs, op,
                            ))
                        else:
                            contractor.add_constraint((
                                "dc_quadratic_form", var_ids, (l_plus, l_minus), rhs, op,
                            ))
                    except Exception:
                        continue
        contractor.propagate()
        return contractor

    @staticmethod
    def _quad_demand(expr: Any) -> tuple[list[int], Any, Any, float, str] | None:
        """Extract a DC contract for ``quad_form OP const`` comparisons.

        Returns ``(var_ids, l_plus, l_minus, rhs, op)`` when ``expr`` is exactly a quadratic
        form compared against a numeric constant; returns ``None`` for anything more complex
        (nested ``sqrt`` etc.), which the contractor and verifier then handle soundly by staying
        silent rather than guessing a bound.
        """

        lhs = getattr(expr, "lhs", None)
        rhs_expr = getattr(expr, "rhs", None)
        op = getattr(expr, "op", None)
        if op is None:
            return None
        flip = {"le": "ge", "lt": "gt", "ge": "le", "gt": "lt", "eq": "eq"}
        quad_node = None
        const_num = None
        if getattr(lhs, "kind", None) in ("quad_form_cholesky", "quad_form_split"):
            quad_node = lhs
            const_num = as_number(rhs_expr)
        elif getattr(rhs_expr, "kind", None) in ("quad_form_cholesky", "quad_form_split"):
            if flip is not None:
                quad_node = rhs_expr
                const_num = as_number(lhs)
                op = flip.get(op, op)
        if quad_node is None or const_num is None:
            return None
        var_ids = [
            arg.var_id for arg in (getattr(quad_node, "args", []) or [])
            if getattr(arg, "kind", None) == "var" and isinstance(arg.var_id, int)
        ]
        if not var_ids:
            return None
        if getattr(quad_node, "kind", None) == "quad_form_cholesky":
            l_plus = getattr(quad_node, "cholesky_l", None)
            l_minus = None
        else:
            l_plus = getattr(quad_node, "cholesky_l_plus", None)
            l_minus = getattr(quad_node, "cholesky_l_minus", None)
        l_plus = _as_factor(l_plus)
        l_minus = _as_factor(l_minus) if l_minus is not None else None
        if l_plus is None or (l_minus is None and getattr(quad_node, "kind", None) == "quad_form_split"):
            return None
        if l_minus is not None and len(l_minus) != len(var_ids):
            return None
        if len(l_plus) != len(var_ids):
            return None
        return var_ids, l_plus, l_minus, float(const_num), str(op)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, model: Any = None, *, prompt: str = "") -> DecodeResult:
        model = model if model is not None else self.model
        if model is None:
            raise ModelProtocolError(
                "no model supplied; pass a model exposing next_logits(prefix) -> Mapping[str, float], "
                "or use vexx.testing.mock_model() for a deterministic reference model"
            )
        self.prompt = prompt or ""
        start = time.perf_counter()
        self._prepare_reasoning(model)
        best = self._run_attempts(model, start)
        if best.ok and self.config.max_preference_repairs > 0:
            best = self._repair_preferences(model, best, start)
        best.reasoning = self.reasoning
        best.mined_equations = list(self.mined_equations)
        best.splices = list(self.splices)
        if self.config.rl_trace_path:
            self._write_rl_trace()
        self.telemetry.preference_score = _preference_score(best.report)
        self.telemetry.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return best

    def _write_rl_trace(self) -> None:
        """Append conflict-branch records (one per backtrack) to the configured JSONL trace.

        Chosen completions are attached when this run ultimately verified; rejected branches
        are recorded immediately on backtrack so preference pairs exist even on failed runs.
        """

        path = self.config.rl_trace_path
        if not path:
            return
        import json as _json
        import os

        records: list[dict[str, Any]] = []
        for record in self.rl.records:
            payload = record.as_dict()
            payload["kind"] = "conflict"
            payload["spec_digest"] = self.telemetry.spec_digest
            if not payload.get("prompt"):
                payload["prompt"] = self.prompt
            records.append(payload)
        if not records:
            return
        try:
            parent = os.path.dirname(path) or "."
            os.makedirs(parent, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(_json.dumps(record, sort_keys=True, default=str) + "\n")
        except OSError:
            # Trace capture must never break a verified generation.
            return

    def _reflex_feedback(self, model: Any, exc: DeadEnd) -> None:
        """Re-open a bounded scratchpad seeded with the algebraic conflict certificate.

        The feedback never lands in ``self.text``: it is an ephemeral reasoning span for the
        model's next attempt, surfaced on ``DecodeResult.reasoning`` with explicit
        ``<|solver_feedback|>`` markers so a caller (or a downstream agent loop) can lift it.
        """

        cut: dict[str, Any] | None = None
        if self.telemetry.domain_cuts:
            cut = self.telemetry.domain_cuts[-1]
        cut_describe = ""
        if cut:
            bounds = cut.get("bounds") or cut.get("bounds", [])
            cut_describe = f" domain cut to [{bounds[0]:.6g}, {bounds[1]:.6g}] for {cut.get('field')!r};" if bounds else ""
        feedback = (
            f"Backtrack at field `{exc.field_name}`: {exc.reason}.{cut_describe} "
            f"Admissible interval is unchanged by the splice; resample a different value."
        )
        text = _model_reasoning(model, f"{self.prompt}\n<|solver_feedback|>{feedback}</|solver_feedback|>",
                                self.config.reflex_max_chars)
        if not text:
            text = "acknowledging backtrack; will sample away from the cut branch"
        span = f"<|solver_feedback|>{feedback}</|solver_feedback|>\n{text}\n"
        self.reasoning = (self.reasoning + "\n" + span).strip() if self.reasoning else span
        self.telemetry.reasoning_chars += len(span)
        self.trace.append(
            StepTrace(
                step=self._next_step(),
                attempt=self._attempt,
                field_name=str(exc.field_name or "<reflex>"),
                char="",
                kind="reflex",
                note=f"solver feedback reflex on backtrack ({len(text)} chars)",
            )
        )

    # ------------------------------------------------------------------
    # CRANE scratchpad
    # ------------------------------------------------------------------
    def _reasoning_mode(self) -> str:
        mode = str(self.config.reasoning_mode or "auto")
        if mode == "auto":
            mode = str(self.spec.config.get("reasoning_mode", "strict") or "strict")
        return mode

    def _prepare_reasoning(self, model: Any) -> None:
        """Open the CRANE scratchpad window and passively mine equations.

        The scratchpad is never part of the decoded JSON: it is exposed on
        :attr:`DecodeResult.reasoning` so a caller can log or search it, while the
        structured span stays under full Tier-0/1/2 enforcement.
        """

        self.reasoning = ""
        self.mined_equations = []
        if self._reasoning_mode() != "crane_cot":
            return
        text = _model_reasoning(model, self.prompt, self.config.reasoning_max_chars)
        if not text:
            return
        controller = CraneModeController()
        controller.feed(text)
        self.reasoning = text
        self.mined_equations = list(dict.fromkeys(controller.mined_equations))
        self.telemetry.reasoning_chars = len(text)
        self.telemetry.mined_equations = len(self.mined_equations)
        self.trace.append(
            StepTrace(
                step=self._next_step(),
                attempt=-1,
                field_name="<think>",
                char="",
                kind="reasoning",
                note=f"{len(text)} chars, {len(self.mined_equations)} mined equations",
            )
        )

    # ------------------------------------------------------------------
    # attempts
    # ------------------------------------------------------------------
    def _run_attempts(self, model: Any, start: float) -> DecodeResult:
        best: DecodeResult | None = None
        attempts = 0
        for attempt in range(self.config.max_backtracks + 1):
            attempts = attempt + 1
            self._attempt = attempt
            self._reset_attempt()
            try:
                text = self._generate_once(model)
            except DeadEnd as exc:
                self.telemetry.backtracks += 1
                self.telemetry.backtrack_fields.append(str(exc.field_name))
                self.trace.append(
                    StepTrace(self._next_step(), attempt, str(exc.field_name), "", kind="backtrack",
                              note=exc.reason)
                )
                if self.config.reflex_on_backtrack:
                    self._reflex_feedback(model, exc)
                if not self._cut_for_field(exc.field_name):
                    break
                continue

            report = verify_values(self.spec, self.values, derive_missing=self.config.derive_missing)
            self.telemetry.solver_checks += 1
            result = self._finalize(text, report, attempts, model, start)
            if report.status == "PASS":
                for conflict in self._pending_conflicts:
                    if not conflict.chosen:
                        self.rl.record_repair(conflict, result.text)
                self._pending_conflicts.clear()
                return result
            if best is None or len(report.violations) < len(best.report.violations):
                best = result
            self.telemetry.backtracks += 1
            self.trace.append(
                StepTrace(self._next_step(), attempt, "", "", kind="backtrack",
                          note=f"clause verification: {report.summary_line()}")
            )
            if not self._apply_report_cut(report, text):
                break
        if best is None:
            # Never return a truncated document: a decode result must always be parseable
            # JSON, even when the engine exhausted its backtrack budget mid-number.
            text = self.text if is_complete_json(self.text) else render_object(
                self.values, self.fields
            )
            report = verify_values(self.spec, self.values, derive_missing=self.config.derive_missing)
            best = self._finalize(text, report, attempts, model, start)
        best.telemetry.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return best

    # ------------------------------------------------------------------
    # soft preferences (MaxSMT-style steering, never hard constraints)
    # ------------------------------------------------------------------
    def _preference_bounds(self, values: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
        """Derive *soft* domain tightenings from currently violated preferences."""

        bounds: dict[str, tuple[float, float]] = {}
        for pref in self.spec.soft_preferences:
            expr = pref.expr
            if expr is None or getattr(expr, "kind", None) != "compare":
                continue
            name, op, const = _unary_preference(expr, self.spec)
            if name is None or name in self.rules or name not in self._field_index:
                continue
            vtype = self.types.get(name)
            if getattr(vtype, "kind", "real") not in ("real", "int", "float", "number"):
                continue
            lo, hi = self._declared_bounds(name)
            previous = bounds.get(name)
            if previous is not None:
                lo, hi = max(lo, previous[0]), min(hi, previous[1])
            if op in ("ge", "gt"):
                lo = max(lo, const)
            elif op in ("le", "lt"):
                hi = min(hi, const)
            elif op == "eq":
                lo, hi = max(lo, const), min(hi, const)
            else:
                continue
            if lo <= hi:
                bounds[name] = (lo, hi)
        return bounds

    def _repair_preferences(self, model: Any, best: DecodeResult, start: float) -> DecodeResult:
        score = _preference_score(best.report)
        self.telemetry.preference_score = score
        if score is None or score >= 1.0 - 1e-12:
            return best
        saved_overrides = dict(self._overrides)
        saved_elems = dict(self._elem_overrides)
        saved_tabu = {key: set(value) for key, value in self._tabu.items()}
        for _ in range(self.config.max_preference_repairs):
            bounds = self._preference_bounds(best.values)
            changed = False
            for name, (lo, hi) in bounds.items():
                current = self._soft_overrides.get(name)
                merged = (max(current[0], lo), min(current[1], hi)) if current else (lo, hi)
                if merged[0] <= merged[1] and merged != current:
                    if merged != self._declared_bounds(name):
                        self._soft_overrides[name] = merged
                        changed = True
            if not changed:
                break
            self.telemetry.preference_repairs += 1
            candidate = self._run_attempts(model, start)
            candidate_score = _preference_score(candidate.report)
            if not candidate.ok or candidate_score is None or candidate_score <= score + 1e-12:
                # Soft tightening cannot make a verified result worse; it simply stops
                # paying off. Restore the learning state of the accepted result.
                self._overrides = saved_overrides
                self._elem_overrides = saved_elems
                self._tabu = saved_tabu
                break
            best, score = candidate, candidate_score
            saved_overrides = dict(self._overrides)
            saved_elems = dict(self._elem_overrides)
            saved_tabu = {key: set(value) for key, value in self._tabu.items()}
            if score >= 1.0 - 1e-12:
                break
        self.telemetry.preference_score = score
        return best

    # ------------------------------------------------------------------
    # attempt lifecycle
    # ------------------------------------------------------------------
    def _reset_attempt(self) -> None:
        self.values = {}
        self.text = ""
        self._step = 0
        self._partial_numeric = {}
        self._attempt_checkpoints = []
        self._pending_conflicts = []
        self._field_span = {}
        self._field_entropy = {}
        self._field_tokens = {}
        self._commit_order = []
        self._field_kl = {}
        self._kl_steps = 0
        self.contractor = self._build_contractor()

    def _next_step(self) -> int:
        self._step += 1
        self.telemetry.steps += 1
        return self._step

    def _generate_once(self, model: Any) -> str:
        self.text = "{"
        for index, name in enumerate(self.fields):
            if index:
                self.text += ", "
            self.text += json_value(name) + ": "
            value = self._emit_field(name, model)
            self.values[name] = value
        self.text += "}"
        if self.config.enforce_grammar:
            machine = JSONPushdownAutomaton()
            if not (machine.feed(self.text) and machine.accepts()):
                raise DeadEnd(self.fields[-1] if self.fields else None,
                              f"emitted text is not valid JSON: {machine.error}")
        return self.text

    def _finalize(
        self,
        text: str,
        report: VerificationReport,
        attempts: int,
        model: Any,
        start: float,
    ) -> DecodeResult:
        self.telemetry.final_status = report.status
        self.telemetry.attempts = attempts
        self.telemetry.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DecodeResult(
            text=text,
            values=dict(report.values if report.values else self.values),
            report=report,
            telemetry=self.telemetry,
            trace=list(self.trace),
            attempts=attempts,
            model_name=getattr(model, "name", None) or type(model).__name__,
            prompt=self.prompt,
        )

    # ------------------------------------------------------------------
    # field emission
    # ------------------------------------------------------------------
    def _emit_field(self, name: str, model: Any) -> Any:
        start = len(self.text)
        self._field_entropy[name] = []
        try:
            if name in self.rules:
                return self._emit_derived(name, model)
            if name in self.containers:
                return self._emit_container(name, model)
            return self._emit_scalar(name, model)
        finally:
            self._field_span[name] = (start, len(self.text))

    def _emit_derived(self, name: str, model: Any) -> Any:
        value = self.zero.evaluate(name, self.values)
        if value is None:
            ready = self.zero.is_ready(name, self.values)
            culprit = self._upstream_free_field(name)
            raise DeadEnd(culprit, f"derived field {name!r} is not evaluable (ready={ready})")
        precision = max(0, int(getattr(self.rules[name], "precision_digits", 6) or 0))
        rendered = json_number(value, precision) if isinstance(value, (int, float)) else json_value(value)
        self.text += rendered
        self._field_tokens[name] = rendered
        self.telemetry.jump_forward_fields += 1
        self.telemetry.jump_forward_chars += len(rendered)
        self._record(StepTrace, name, rendered, kind="splice",
                     note=f"0-DoF: computed {rendered}")
        self._plan_splice(name, rendered)
        # A derived value outside the contracted domain is a genuine conflict in an
        # upstream free choice, not a reason to reject the derived field itself.
        vid = self.vid.get(name)
        if vid is not None and vid in self.contractor.domains:
            bound = self.contractor.get_bounds(vid)
            number = as_number(value)
            if number is not None and not bound.is_empty():
                if number < bound.lo - 1e-9 or number > bound.hi + 1e-9:
                    culprit = self._upstream_free_field(name)
                    raise DeadEnd(
                        culprit,
                        f"derived {name}={number:g} violates contracted domain "
                        f"[{bound.lo:g}, {bound.hi:g}]",
                    )
        return value

    def _plan_splice(self, name: str, rendered: str) -> None:
        """Emit a token-healed prefill plan for a deterministic continuation.

        The reference session samples characters, so the splice is already in the text.
        A production bridge drives the same ``ZeroDofEngine`` and consumes this plan to
        extend the KV cache in one batched pass, so the plan is recorded with the model's
        own tokenizer whenever one was supplied.
        """

        if self.healer is None:
            return
        prefix = json_value(name) + ": "
        span = self._field_span.get(name)
        value_start = span[0] if span is not None else len(self.text) - len(rendered)
        # The deterministic continuation starts at this key's opening quote, i.e. after
        # the previously committed field.
        cut = max(0, value_start - len(prefix))
        tail = self.text[:cut]
        suffix = ", " + prefix + rendered
        try:
            prefill = self.healer.heal(tail, suffix)
        except Exception:  # pragma: no cover - tokenizer protocol violation
            return
        self.telemetry.jump_forward_tokens += len(prefill.new_token_ids)
        if prefill.exact:
            self.telemetry.token_heal_exact += 1
        self.splices.append(
            {
                "field": name,
                "value": rendered,
                "text": suffix,
                "backup_tokens": prefill.backup_tokens,
                "token_ids": list(prefill.new_token_ids),
                "exact": bool(prefill.exact),
                "offset": prefill.offset,
            }
        )

    def _upstream_free_field(self, name: str) -> str | None:
        rule = self.rules.get(name)
        upstream: list[str] = []
        if rule is not None:
            for dep in rule.input_vars:
                dep_name = self.spec.id_to_name.get(dep) if isinstance(dep, int) else str(dep)
                if dep_name in self.fields and dep_name not in self.rules:
                    upstream.append(dep_name)
        if upstream:
            # The latest upstream free field is the most specific culprit.
            return upstream[-1]
        free = [f for f in self.fields if f not in self.rules]
        return free[-1] if free else None

    def _emit_scalar(self, name: str, model: Any) -> Any:
        vtype = self.types.get(name)
        kind = getattr(vtype, "kind", "real")
        vid = self.vid[name]
        if kind == "int":
            lo, hi = self._bounds(name)
            text, value = self._emit_number(
                name, model, (lo, hi), int_only=True, tabu_key=name
            )
            value = int(round(value))
            self._commit_scalar(name, vid, float(value))
            return value
        if kind in ("real", "float", "int", "number", "any"):
            lo, hi = self._bounds(name)
            text, value = self._emit_number(name, model, (lo, hi), tabu_key=name)
            self._commit_scalar(name, vid, value)
            return value
        if kind == "bool":
            choice = self._pick_strings(name, model, ["true", "false"])
            value = choice == "true"
            self.text += choice
            self._record(StepTrace, name, choice)
            self._commit_scalar(name, vid, 1.0 if value else 0.0)
            return value
        if kind == "finite_set":
            universe = list(getattr(vtype, "universe", None) or [])
            if not universe:
                raise DeadEnd(name, "finite-set field has an empty universe")
            actions = [json_value(v) for v in universe]
            mapping = {json_value(v): v for v in universe}
            choice = self._pick_strings(name, model, actions)
            self.text += choice
            self._record(StepTrace, name, choice)
            return mapping[choice]
        if kind == "object":
            return self._emit_object(name, vtype, model)
        if kind in _CONTAINER_KINDS:
            return self._emit_container(name, model)
        if kind == "null":
            self.text += "null"
            return None
        if kind == "string":
            value = self._emit_string(name, model, vtype)
            self.text += json_value(value)
            return value
        raise DeadEnd(name, f"unsupported field type {kind!r} for constrained decoding")

    def _pattern_for(self, name: str, vtype: Any) -> PatternConstraint | None:
        """Compile (once) the field's JSON-Schema ``pattern`` into a prefix automaton."""

        if name in self._patterns:
            return self._patterns[name]
        constraint: PatternConstraint | None = None
        raw = getattr(vtype, "pattern", None)
        if isinstance(raw, str) and raw:
            try:
                constraint = PatternConstraint.compile(raw)
            except UnsupportedPattern as exc:
                self._patterns[name] = None
                if raw not in self.telemetry.pattern_fallbacks:
                    self.telemetry.pattern_fallbacks.append(f"{name}: {exc}")
                return None
        self._patterns[name] = constraint
        return constraint

    def _string_alphabet(self) -> list[str]:
        return [chr(c) for c in range(0x20, 0x7F) if chr(c) not in {'"', "\\"}]

    def _emit_string(self, name: str, model: Any, vtype: Any = None) -> str:
        vtype = vtype if vtype is not None else self.types.get(name)
        min_len = max(0, int(getattr(vtype, "min_len", 0) or 0))
        declared_max = int(getattr(vtype, "max_len", 0) or 0)
        limit = self.config.max_string_length
        if declared_max:
            limit = min(limit, declared_max)
        if limit < min_len:
            raise DeadEnd(
                name,
                f"string length bounds are unsatisfiable (min_length={min_len} > "
                f"max_length={declared_max or self.config.max_string_length})",
            )
        constraint = self._pattern_for(name, vtype)
        # Formatted strings (``date-time``, ``uuid``, ``email``, ...) are validated by the
        # concrete schema checker, so sampling characters cannot work.  Emit a witness for
        # the format instead -- deterministically, in zero sampled tokens -- and let the
        # verifier confirm it.  Anything unsupported degrades to length/pattern masking
        # with an explicit telemetry entry.
        fmt = getattr(vtype, "text_format", None)
        if fmt:
            candidate = textformats.witness(fmt, min_len=min_len, max_len=limit)
            if candidate is not None and (constraint is None or constraint.satisfied(candidate)):
                self.telemetry.format_witnesses += 1
                self._record(
                    StepTrace,
                    name,
                    candidate,
                    kind="format",
                    note=f"format {fmt!r} witness",
                )
                return candidate
            reason = (
                "witness violates the declared pattern"
                if candidate is not None
                else f"no witness inside length window for format {fmt!r}"
            )
            entry = f"{name}: {reason}"
            if entry not in self.telemetry.format_fallbacks:
                self.telemetry.format_fallbacks.append(entry)
        alphabet = self._string_alphabet()
        chars: list[str] = []
        while True:
            prefix = "".join(chars)
            actions: list[str] = []
            if len(chars) < limit:
                if constraint is None:
                    actions = list(alphabet)
                else:
                    actions = constraint.allowed_next(prefix, alphabet)
                    self.telemetry.string_mask_prunes += max(0, len(alphabet) - len(actions))
            can_stop = len(chars) >= min_len and (
                constraint is None or constraint.can_stop(prefix)
            )
            if can_stop:
                actions.insert(0, _STOP)
            if not actions:
                raise DeadEnd(
                    name,
                    f"string prefix {prefix!r} has no admissible continuation "
                    f"(min_length={min_len}, pattern={getattr(vtype, 'pattern', None)!r})",
                )
            choice = self._pick(
                name, model, self.text + prefix, actions, candidates=len(alphabet) + 1
            )
            if choice == _STOP:
                return prefix
            chars.append(choice)
            self._record(StepTrace, name, choice)
        # The *caller* appends ``json_value(value)`` once the literal is complete.  Writing
        # the raw characters into the document here as well duplicated every string field
        # (``"name": 99"99"``), which made the JSON PDA reject the attempt and silently
        # fall back to canonical rendering.  Keep the sampled prefix local instead.

    def _exclusive_for(self, name: str) -> tuple[bool, bool]:
        """Open-bound flags that still matter after integer promotion."""

        if getattr(self.types.get(name), "kind", None) == "int":
            return (False, False)
        return self._exclusive.get(name, (False, False))

    def _upstream_culprit(self, name: str) -> str:
        """The most recent free field committed before ``name``."""

        for other in reversed(self._commit_order):
            if other != name and other not in self.rules:
                return other
        return name

    def _open_domain_conflict(self) -> str | None:
        """Pending field whose contracted domain admits no *strictly interior* value."""

        for pending in self.fields:
            if pending in self.values or pending in self.rules:
                continue
            vid = self.vid.get(pending)
            if vid is None or vid not in self.contractor.domains:
                continue
            bound = self.contractor.get_bounds(vid)
            if bound.is_empty():
                return pending
            lo_ex, hi_ex = self._exclusive_for(pending)
            if not (lo_ex or hi_ex):
                continue
            declared_lo, declared_hi = self._declared_bounds(pending)
            if bound.lo == bound.hi:
                value = bound.lo
                if (lo_ex and value <= declared_lo + 1e-12) or (
                    hi_ex and value >= declared_hi - 1e-12
                ):
                    return pending
        return None

    def _commit_scalar(self, name: str, vid: int, value: float) -> None:
        # Record the assignment before propagation so a conflict can name this field as
        # the culprit and cut its domain on the next attempt.
        self.values[name] = value
        self._commit_order.append(name)
        if vid not in self.contractor.domains:
            return
        # Propagating a rejected assignment must not poison the contractor: restore the
        # pre-assignment domains so the next attempt still has a usable feasible box.
        saved = {key: Interval(bound.lo, bound.hi) for key, bound in self.contractor.domains.items()}
        # The literal was chosen from a domain that was contracted with *outward-rounded*
        # arithmetic, so an exact point assignment can sit up to a few ULPs outside the
        # intersection the contractor can re-prove.  Widening the point by a handful of
        # ULPs keeps float noise from masquerading as infeasibility; a genuinely
        # infeasible assignment still fails on every attempt.
        slack = max(abs(value) * 8 * 2.220446049250313e-16, 1e-12)
        attempts = [Interval(value, value), Interval(value - slack, value + slack)]
        propagated = False
        for domain in attempts:
            self.contractor.domains = {k: Interval(v.lo, v.hi) for k, v in saved.items()}
            self.contractor.set_domain(vid, domain)
            if self.contractor.propagate():
                propagated = True
                break
        if not propagated:
            self.contractor.domains = saved
            raise DeadEnd(
                self._upstream_culprit(name),
                f"contractor became infeasible after {name}={value:g}",
            )
        blocked = self._open_domain_conflict()
        if blocked is not None:
            # ``name`` consumed the room that ``blocked`` needed (e.g. an open bound).
            self.contractor.domains = saved
            raise DeadEnd(
                name,
                f"after {name}={value:g}, open-bounded field {blocked!r} has no admissible value",
            )

    # ------------------------------------------------------------------
    # numeric emission
    # ------------------------------------------------------------------
    def _declared_bounds(self, name: str) -> tuple[float, float]:
        """Type bounds plus domain cuts, *without* the current contractor assignment.

        Cuts must be computed against the declared box: once a field has been assigned,
        its contractor domain is a singleton and would make a cut impossible.
        """

        vid = self.vid.get(name)
        initial = self._initial_domains.get(vid) if vid is not None else None
        if initial is not None and not initial.is_empty():
            lo, hi = initial.lo, initial.hi
        else:
            vtype = self.types.get(name)
            lo = getattr(vtype, "min", None)
            hi = getattr(vtype, "max", None)
            lo = float(lo) if lo is not None else float("-inf")
            hi = float(hi) if hi is not None else float("inf")
        vtype = self.types.get(name)
        # ``_initial_domains`` comes from the contractor, whose bounds are rounded
        # outwards by up to one ULP.  Clamping back onto the declared type bounds keeps
        # "is this value the excluded endpoint?" decidable (``lt=1`` must reject 1 even
        # when the contractor advertises 1.0000000000000004).
        declared_lo = getattr(vtype, "min", None)
        declared_hi = getattr(vtype, "max", None)
        if declared_lo is not None and math.isfinite(float(declared_lo)):
            lo = max(lo, float(declared_lo))
        if declared_hi is not None and math.isfinite(float(declared_hi)):
            hi = min(hi, float(declared_hi))
        for source in (self._soft_overrides, self._overrides):
            override = source.get(name)
            if override is not None:
                lo = max(lo, override[0])
                hi = min(hi, override[1])
        if getattr(self.types.get(name), "kind", None) == "int":
            # Open integer bounds have a closed integer equivalent: (0, 5) == [1, 4].
            lo_ex, hi_ex = self._exclusive.get(name, (False, False))
            if lo_ex and math.isfinite(lo):
                lo = math.floor(lo) + 1
            if hi_ex and math.isfinite(hi):
                hi = math.ceil(hi) - 1
        return lo, hi

    def _bounds(self, name: str) -> tuple[float, float]:
        lo, hi = self._declared_bounds(name)
        vid = self.vid.get(name)
        if vid is not None and vid in self.contractor.domains:
            bound = self.contractor.get_bounds(vid)
            if not bound.is_empty():
                lo = max(lo, bound.lo)
                hi = min(hi, bound.hi)
        return lo, hi

    def _emit_number(
        self,
        name: str,
        model: Any,
        bounds: tuple[float, float],
        *,
        int_only: bool = False,
        tabu_key: Any = None,
        exclusive: tuple[bool, bool] | None = None,
        declared: tuple[float, float] | None = None,
    ) -> tuple[str, float]:
        lo, hi = bounds
        if declared is None and name in self.types:
            declared = self._declared_bounds(name)
        if exclusive is None:
            exclusive = self._exclusive_for(name)
        if lo > hi + 1e-12:
            raise DeadEnd(
                self._upstream_culprit(name),
                f"empty numeric domain [{lo:g}, {hi:g}] for {name!r}",
            )
        interval = Interval(lo, hi)
        budget = [max(1, int(getattr(self.config, "max_number_nodes", 512) or 1))]

        def search(prefix: str, depth: int) -> tuple[str, float] | None:
            # A number literal is short; the depth cap keeps the recursion finite even for
            # malformed configurations, and the node budget bounds the total work so a
            # hostile model cannot turn a single field into an unbounded search.
            if depth > 40 or self._step + depth > self.config.max_steps:
                return None
            actions: list[str] = []
            for ch in "0123456789.-":
                candidate = prefix + ch
                if not self._number_prefix_ok(candidate, int_only, name):
                    continue
                if Interval.from_numeric_prefix(
                    candidate, interval, self.config.allow_exponent
                ).is_empty():
                    self.telemetry.contractor_prunes += 1
                    continue
                actions.append(ch)
            stop_ok = self._number_stop_ok(
                prefix, interval, int_only, tabu_key, exclusive, name, declared
            )
            choices = ([_STOP] if stop_ok else []) + actions
            # The local grammar has twelve numeric characters plus the stop token; every
            # one that is absent from ``choices`` was masked by the reachable interval.
            self.telemetry.masked_candidates += max(0, 13 - len(choices))
            if not choices:
                return None
            for choice in self._ordered_actions(name, model, self.text + prefix, choices):
                budget[0] -= 1
                if budget[0] <= 0:
                    # Out of budget: stop exploring and let the deterministic constructor
                    # below pick a feasible literal instead of failing the attempt.
                    return None
                if choice == _STOP:
                    value = float(prefix)
                    return prefix, (float(int(round(value))) if int_only else value)
                self.telemetry.number_expansions += 1
                found = search(prefix + choice, depth + 1)
                if found is not None:
                    return found
            return None

        found = search("", 0)
        if found is None:
            found = self._canonical_number(
                interval, int_only, tabu_key, exclusive, name, declared
            )
        if found is None:
            raise DeadEnd(
                self._upstream_culprit(name),
                f"no representable number in [{lo:g}, {hi:g}] for {name!r} "
                f"(digits, stop and backtracking all exhausted)",
            )
        literal, value = found
        for ch in literal:
            self.text += ch
            self.telemetry.sampled_units += 1
            self._record(StepTrace, name, ch, kind="sample")
        self._partial_numeric[name] = value
        return literal, value

    def _canonical_number(
        self,
        interval: Interval,
        int_only: bool,
        tabu_key: Any,
        exclusive: tuple[bool, bool] | None,
        name: str,
        declared: tuple[float, float] | None,
    ) -> tuple[str, float] | None:
        """Construct one feasible literal directly when the search gives up.

        Model-preferred digits can descend into a decimal dead end that is exponentially
        large to exhaust (``0.000000`` and siblings while the only feasible values sit
        near the middle of the interval).  A bounded number of deterministic candidates --
        the interior of the reachable interval, formatted with the configured precision --
        keeps the field time-bounded *without* ever emitting an unverified value: every
        candidate still passes ``_number_stop_ok``.
        """

        lo, hi = interval.lo, interval.hi
        digits = max(0, int(self.config.max_fraction_digits))
        candidates: list[float] = []
        if int_only:
            low = math.ceil(lo) if math.isfinite(lo) else -10**6
            high = math.floor(hi) if math.isfinite(hi) else 10**6
            if low > high:
                return None
            candidates = [(low + high) // 2, low, high]
        elif math.isfinite(lo) and math.isfinite(hi):
            span = hi - lo
            candidates = [lo + span / 2, lo + span / 4, lo + span * 3 / 4, lo, hi]
        elif math.isfinite(hi):
            candidates = [hi - 1.0, hi - max(1.0, abs(hi)) / 2, hi]
        elif math.isfinite(lo):
            candidates = [lo + 1.0, lo + max(1.0, abs(lo)) / 2, lo]
        else:
            candidates = [0.0, 0.5, 1.0, -0.5]
        for value in candidates:
            if not math.isfinite(value):
                continue
            if int_only:
                text = str(int(value))
            else:
                text = f"{value:.{digits}f}"
                if "." in text:
                    text = text.rstrip("0").rstrip(".")
                if text in ("", "-"):
                    text = "0"
            if not self._number_prefix_ok(text, int_only, name):
                continue
            if self._number_stop_ok(
                text, interval, int_only, tabu_key, exclusive, name, declared
            ):
                return text, float(text)
        return None

    def _ordered_actions(
        self, name: str, model: Any, prefix: str, actions: Sequence[str]
    ) -> list[str]:
        """Order candidate continuations by model preference (best first)."""

        if len(actions) <= 1:
            return list(actions)
        raw = self._logits(model, prefix)
        logits = raw
        distortion = _mask_kl(raw, actions)
        if distortion > 0.0:
            self._field_kl[name] = self._field_kl.get(name, 0.0) + distortion
            self._kl_steps += 1
            self.telemetry.field_kl_divergence = dict(sorted(self._field_kl.items()))
            self.telemetry.kl_distortion_tokens = self._kl_steps
        scores: list[tuple[float, int, str]] = []
        for position, action in enumerate(actions):
            score = logits.get(action)
            score = float(score) if isinstance(score, (int, float)) else float("-inf")
            if self.config.temperature and self.config.temperature > 0:
                score /= float(self.config.temperature)
            scores.append((score, -position, action))
        scores.sort(reverse=True)
        ordered = [action for _, _, action in scores]
        self._last_entropy = _entropy(_softmax([score for score, _, _ in scores]))
        return ordered

    def _number_prefix_ok(self, prefix: str, int_only: bool, name: str | None = None) -> bool:
        if prefix == "":
            return False
        if prefix == "-":
            return not int_only or True
        body = prefix[1:] if prefix.startswith("-") else prefix
        if not body:
            return False
        if len(prefix) > 24:
            return False
        if int_only:
            if not body.isdigit():
                return False
            return not (len(body) > 1 and body[0] == "0")
        if body.count(".") > 1:
            return False
        if body.startswith("."):
            return False
        integer, dot, fraction = body.partition(".")
        if integer and not integer.isdigit():
            return False

        # Determine limits: use dynamic scaling if enabled and field name provided
        max_int_digits = self.config.max_integer_digits
        max_frac_digits = self.config.max_fraction_digits
        if self.config.auto_scale_numeric_limits and name is not None:
            dyn_int, dyn_frac = self._compute_numeric_limits(name)
            max_int_digits = max(max_int_digits, dyn_int)
            max_frac_digits = max(max_frac_digits, dyn_frac)

        if integer and len(integer) > max_int_digits:
            return False
        if integer and len(integer) > 1 and integer[0] == "0":
            return False
        if dot and len(fraction) > max_frac_digits:
            return False
        return True

    def _number_stop_ok(
        self,
        prefix: str,
        interval: Interval,
        int_only: bool,
        tabu_key: Any,
        exclusive: tuple[bool, bool] = (False, False),
        name: str | None = None,
        declared: tuple[float, float] | None = None,
    ) -> bool:
        if not prefix or prefix == "-" or prefix.endswith("."):
            return False
        if not _NUMBER_COMPLETE.match(prefix):
            return False
        if int_only and "." in prefix:
            return False
        value = float(prefix)
        if value < interval.lo - 1e-12 or value > interval.hi + 1e-12:
            return False
        # ``gt``/``lt`` declare an open bound: the exact declared endpoint is never a
        # valid value, even when the contracted interval has been rounded outwards.
        if exclusive[0] or exclusive[1]:
            # The reference range is the *declared* one (type bounds for scalar
            # fields, the analysed element range for container slots), never the
            # contracted interval: a constraint may legitimately narrow a domain to a
            # point that is still interior to the open bound.
            lower: float | None = None
            upper: float | None = None
            if declared is not None:
                lower, upper = declared
            else:
                vtype = self.types.get(name) if name else None
                raw_lo = getattr(vtype, "min", None)
                raw_hi = getattr(vtype, "max", None)
                lower = None if raw_lo is None else float(raw_lo)
                upper = None if raw_hi is None else float(raw_hi)
            if exclusive[0] and lower is not None and value <= lower + 1e-12:
                return False
            if exclusive[1] and upper is not None and value >= upper - 1e-12:
                return False
        if tabu_key is not None and self._is_tabu(tabu_key, value):
            return False
        return True

    def _is_tabu(self, key: Any, value: float) -> bool:
        values = self._tabu.get(key)
        if not values:
            return False
        return any(abs(value - other) <= 1e-12 for other in values)

    # ------------------------------------------------------------------
    # container emission (AGDecode residual accumulator)
    # ------------------------------------------------------------------
    def _emit_object(self, name: str, vtype: Any, model: Any) -> dict[str, Any]:
        properties = getattr(vtype, "properties", {}) or {}
        required = list(getattr(vtype, "required", []) or list(properties))
        self.text += "{"
        result: dict[str, Any] = {}
        for index, key in enumerate(required):
            if key not in properties:
                raise DeadEnd(name, f"required property {key!r} has no type information")
            if index:
                self.text += ", "
            self.text += json_value(key) + ": "
            result[key] = self._emit_value(f"{name}.{key}", properties[key], model)
        self.text += "}"
        self._record(StepTrace, name, "{}", kind="close", note=f"object with {len(result)} properties")
        return result

    def _emit_value(self, path: str, vtype: Any, model: Any) -> Any:
        kind = getattr(vtype, "kind", "any")
        if getattr(vtype, "nullable", False) and kind not in ("null",):
            # Deterministic reference policy: prefer a concrete value when one is
            # representable; null remains available to the final verifier.
            pass
        if kind == "object":
            return self._emit_object(path, vtype, model)
        if kind in _CONTAINER_KINDS:
            return self._emit_container(path, model, vtype=vtype)
        if kind in ("real", "float", "number", "int"):
            lo = getattr(vtype, "min", None)
            hi = getattr(vtype, "max", None)
            lo = float(lo) if lo is not None else float("-inf")
            hi = float(hi) if hi is not None else float("inf")
            _text, value = self._emit_number(path, model, (lo, hi), int_only=(kind == "int"), tabu_key=path)
            return int(round(value)) if kind == "int" else value
        if kind == "bool":
            choice = self._pick_strings(path, model, ["true", "false"])
            self.text += choice
            return choice == "true"
        if kind == "finite_set":
            universe = list(getattr(vtype, "universe", None) or [])
            if not universe:
                raise DeadEnd(path, "finite-set field has an empty universe")
            actions = [json_value(v) for v in universe]
            choice = self._pick_strings(path, model, actions)
            self.text += choice
            return universe[actions.index(choice)]
        if kind == "null":
            self.text += "null"
            return None
        if kind == "string":
            value = self._emit_string(path, model, vtype)
            self.text += json_value(value)
            return value
        raise DeadEnd(path, f"unsupported nested type {kind!r}")

    def _emit_list(self, name: str, model: Any, vtype: Any) -> list[Any]:
        item_type = getattr(vtype, "item_type", None)
        min_len = int(getattr(vtype, "min_len", 0) or 0)
        max_len = getattr(vtype, "max_len", None)
        length = min_len if max_len is None else min(min_len, int(max_len))
        if max_len is not None and min_len > int(max_len):
            raise DeadEnd(name, "list min_items exceeds max_items")
        self.text += "["
        result: list[Any] = []
        info = self.containers.get(name)
        numeric_item = item_type is None or getattr(item_type, "kind", "real") in ("real", "int", "float", "number", "any")
        total = 0.0
        for index in range(length):
            if index:
                self.text += ", "
            if numeric_item and info is not None:
                lo, hi = self._element_bounds(name, str(index), info, total, index)
                _text, value = self._emit_number(
                    name,
                    model,
                    (lo, hi),
                    int_only=getattr(item_type, "kind", None) == "int",
                    exclusive=(info.elem_min_exclusive, info.elem_max_exclusive),
                    declared=(lo, hi),
                    tabu_key=(name, index),
                )
                total += value
                result.append(value)
            else:
                result.append(self._emit_value(f"{name}[{index}]", item_type or VirType(), model))
        self.text += "]"
        self._record(StepTrace, name, "[]", kind="close", note=f"list with {len(result)} items")
        return result

    def _emit_container(self, name: str, model: Any, vtype: Any = None) -> Any:
        info = self.containers.get(name)
        if vtype is None and info is not None:
            vtype = self.types.get(name)
        container_kind = getattr(vtype, "container_kind", "dict") if vtype is not None else "dict"
        if container_kind == "list":
            return self._emit_list(name, model, vtype)
        if info is None:
            # Nested dict without analysis: emit an empty object only when allowed.
            min_len = int(getattr(vtype, "min_len", 0) or 0)
            if min_len:
                raise DeadEnd(name, "cannot synthesize required keys for an unconstrained dictionary")
            self.text += "{}"
            return {}
        keys = list(info.universe or [])
        item_type = getattr(vtype, "item_type", None)
        numeric_item = item_type is None or getattr(item_type, "kind", "real") in ("real", "int", "float", "number", "any")
        if not keys:
            min_len = max(0, info.min_len)
            if min_len:
                raise DeadEnd(name, "cannot synthesize required keys for an unconstrained dictionary")
            self.text += "{}"
            self._record(StepTrace, name, "{}", kind="close",
                         note="no key universe declared; emitting an empty container")
            return {}
        if not numeric_item:
            self.text += "{"
            result: dict[str, Any] = {}
            for index, key in enumerate(keys):
                if index:
                    self.text += ", "
                self.text += json_value(key) + ": "
                result[key] = self._emit_value(f"{name}.{key}", item_type, model)
            self.text += "}"
            return result
        self.text += "{"
        total = 0.0
        emitted = 0
        first = True
        result: dict[str, Any] = {}
        for key in keys:
            if emitted >= max(0, info.min_len) and self._can_close(info, total, emitted):
                break
            lo, hi = self._element_bounds(name, key, info, total, emitted)
            if lo > hi + 1e-12:
                raise DeadEnd(name, f"residual bounds are empty for {key!r} at sum={total:g}")
            if not first:
                self.text += ", "
            self.text += json_value(key) + ": "
            _text, value = self._emit_number(
                name,
                model,
                (lo, hi),
                int_only=getattr(item_type, "kind", None) == "int",
                exclusive=(info.elem_min_exclusive, info.elem_max_exclusive),
                declared=(lo, hi),
                tabu_key=(name, key),
            )
            first = False
            result[key] = value
            self.values[name] = dict(result)
            total += value
            emitted += 1
            if not info.residual_feasible(total, emitted):
                raise DeadEnd(
                    name,
                    f"residual sum {total:g} after {emitted} elements cannot reach "
                    f"target {info.target_sum:g}",
                )
        self.text += "}"
        if not self._can_close(info, total, emitted):
            raise DeadEnd(name, f"container {name!r} cannot close at sum {total:g}")
        self._record(StepTrace, name, "{}", kind="close",
                     note=f"{emitted} elements, sum={total:g}")
        return result

    def _element_bounds(
        self,
        name: str,
        key: str,
        info: ContainerInfo,
        total: float,
        emitted: int,
    ) -> tuple[float, float]:
        if info.target_sum is None:
            lo, hi = info.elem_min, info.elem_max
        else:
            lo, hi = info.residual_interval(total, emitted)
            lo = max(lo, info.elem_min)
            hi = min(hi, info.elem_max)
        override = self._elem_overrides.get((name, key))
        if override is not None:
            lo = max(lo, override[0])
            hi = min(hi, override[1])
        lo = lo if math.isfinite(lo) else float("-inf")
        hi = hi if math.isfinite(hi) else float("inf")
        return lo, hi

    def _can_close(self, info: ContainerInfo, total: float, emitted: int) -> bool:
        lo, hi = info.cardinality_bounds()
        if emitted < lo or (hi is not None and emitted > hi):
            return False
        if info.target_sum is None:
            return True
        tolerance = max(info.sum_tol, 1e-9)
        if info.sum_op == "eq":
            return abs(info.target_sum - total) <= tolerance
        if info.sum_op in ("le", "lt"):
            return total <= info.target_sum + tolerance
        if info.sum_op in ("ge", "gt"):
            return total >= info.target_sum - tolerance
        return False

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def _volume_twist(
        self, field_name: str, prefix: str, actions: Sequence[str]
    ) -> list[float]:
        """SHIM-style future-volume correction over numeric continuations.

        ``log Vol(D_{t+1} | prefix + action) - log Vol(D_t)`` is estimated by intersecting
        the field's domain with the prefix interval and running one contractor pass.  When
        the field has no contracted domain (containers, unconstrained numbers) the 1-D
        interval width is used instead, which is the exact analogue for a single axis.
        """

        adjustments = [0.0] * len(actions)
        beta = float(self.config.volume_twist_beta)
        if not beta or field_name not in self.vid:
            return adjustments
        kind = getattr(self.types.get(field_name), "kind", None)
        if kind not in ("real", "float", "number", "int"):
            return adjustments
        vid = self.vid[field_name]
        lo, hi = self._bounds(field_name)
        base_width = max(hi - lo, 1e-12)
        use_contractor = (
            vid in self.contractor.domains and len(self.contractor.constraints) <= 64
        )
        base_log = self.contractor.current_log_volume() if use_contractor else None
        for index, action in enumerate(actions):
            if action == _STOP:
                continue
            interval = Interval.from_numeric_prefix(
                prefix + action, Interval(lo, hi), self.config.allow_exponent
            )
            if interval.is_empty():
                continue
            adjustment = 0.0
            if use_contractor and base_log is not None and math.isfinite(base_log):
                remaining = self.contractor.estimate_log_volume_after(vid, interval)
                if math.isfinite(remaining):
                    adjustment = beta * (remaining - base_log)
            else:
                width = interval.width()
                if math.isfinite(width):
                    adjustment = beta * math.log(max(width, 1e-12) / base_width)
            if adjustment:
                adjustments[index] = adjustment
                self.telemetry.twist_adjustments += 1
        return adjustments

    def _pick(self, field_name: str, model: Any, prefix: str, actions: Sequence[str], candidates: int = 0) -> str:
        raw = self._logits(model, prefix)
        logits = raw
        scores = [float(logits.get(action, 0.0)) for action in actions]
        if self.config.volume_twist_beta:
            adjustments = self._volume_twist(field_name, prefix, actions)
            if any(adjustments):
                scores = [score + delta for score, delta in zip(scores, adjustments)]
        entropy, choice = self._select(field_name, actions, scores, raw=raw)
        self.telemetry.masked_candidates += max(0, candidates - len(actions)) if candidates else 0
        self.telemetry.sampled_units += 1
        self._last_entropy = entropy
        return choice

    def _pick_strings(self, field_name: str, model: Any, actions: Sequence[str]) -> str:
        raw = self._logits(model, self.text)
        logits = raw
        scores = [float(logits.get(action, 0.0)) for action in actions]
        entropy, choice = self._select(field_name, actions, scores, raw=raw)
        self._last_entropy = entropy
        return choice

    def _select(self, field_name: str, actions: Sequence[str], scores: Sequence[float],
                raw: Mapping[str, float] | None = None) -> tuple[float, str]:
        probabilities = _softmax(scores)
        entropy = _entropy(probabilities)
        distortion = _mask_kl(raw, actions)
        if distortion > 0.0 and field_name:
            self._field_kl[field_name] = self._field_kl.get(field_name, 0.0) + distortion
            self._kl_steps += 1
            self.telemetry.field_kl_divergence = dict(sorted(self._field_kl.items()))
            self.telemetry.kl_distortion_tokens = self._kl_steps
        if self.config.temperature <= 0:
            best = max(range(len(actions)), key=lambda i: (scores[i], -i))
            return entropy, actions[best]
        weights = _softmax([s / max(self.config.temperature, 1e-6) for s in scores])
        choice = self.rng.choices(list(range(len(actions))), weights=weights, k=1)[0]
        return entropy, actions[choice]

    def _logits(self, model: Any, prefix: str) -> Mapping[str, float]:
        caller = getattr(model, "next_logits", None)
        if not callable(caller):
            if callable(model):
                caller = model
            else:
                raise ModelProtocolError(
                    "model must expose next_logits(prefix) or be callable(prefix)"
                )
        try:
            raw = caller((self.prompt + "\n" if self.prompt else "") + prefix)
        except Exception as exc:
            raise ModelProtocolError(f"model.next_logits failed: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ModelProtocolError(
                f"model.next_logits must return a Mapping[str, float], got {type(raw).__name__}"
            )
        return raw

    def _record(
        self,
        cls: Any,
        field_name: str,
        text: str,
        *,
        kind: str = "sample",
        allowed: int = 0,
        note: str = "",
    ) -> None:
        entropy = float(getattr(self, "_last_entropy", 0.0) or 0.0)
        if field_name:
            self._field_entropy.setdefault(field_name, []).append(entropy)
        if cls is StepTrace:
            self.trace.append(
                StepTrace(
                    step=self._next_step(),
                    attempt=self._attempt,
                    field_name=field_name,
                    char=text,
                    kind=kind,
                    allowed=allowed,
                    entropy=entropy,
                    note=note,
                )
            )

    # ------------------------------------------------------------------
    # backtracking / domain cuts
    # ------------------------------------------------------------------
    def _attempt_checkpoints_for_report(self) -> list[ClauseCheckpoint]:
        """One checkpoint per free field, carrying real spans, entropy and conflict counts."""

        checkpoints: list[ClauseCheckpoint] = []
        for name in self.fields:
            if name in self.rules:
                continue
            value = self.values.get(name)
            number = as_number(value) if not isinstance(value, (dict, list)) else None
            start, end = self._field_span.get(name, (0, 0))
            entropies = self._field_entropy.get(name) or [0.0]
            committed = float(number) if number is not None else 0.0
            checkpoints.append(
                ClauseCheckpoint(
                    clause_id=self._field_index[name],
                    var_id=name,
                    committed_val=committed,
                    start_token_idx=start,
                    end_token_idx=end,
                    mean_shannon_entropy=sum(entropies) / len(entropies),
                    token_sequence=[self.text[start:end]] if end > start else [],
                    conflict_count=self._conflict_counts.get(name, 0),
                )
            )
        return checkpoints

    def _core_fields(self, report: VerificationReport) -> list[str]:
        """Free fields referenced by the violated constraints, latest first."""

        names: list[str] = []
        for index in report.unsat_core:
            for name in self._clause_fields.get(index, []):
                if name not in self.rules and name not in names:
                    names.append(name)
        if not names:
            names = [name for name in self.fields if name not in self.rules]
        order = self._field_index
        return sorted(names, key=lambda name: order.get(name, -1))

    def _apply_report_cut(self, report: VerificationReport, text: str) -> bool:
        self._attempt_checkpoints = self._attempt_checkpoints_for_report()
        self.cdcl.checkpoints.extend(self._attempt_checkpoints)
        candidates = self._core_fields(report)
        culprit_name: str | None = None
        if candidates and self._attempt_checkpoints:
            core_ids = [self._field_index[name] for name in candidates]
            self.cdcl.jumpback(core_ids)
            culprit = self.cdcl.select_culprit(core_ids)
            if culprit is not None and culprit.var_id in self._field_index:
                culprit_name = str(culprit.var_id)
        if culprit_name is None:
            culprit_name = self._culprit_from_report(report)
        margin = 0.0
        for outcome in report.violations:
            if outcome.residual is not None:
                margin = max(margin, abs(outcome.residual))
        span = self._field_span.get(culprit_name or "", (0, len(text)))
        record = ConflictBranchRecord(
            prefix_tokens=[text[: span[0]]],
            rejected_clause_tokens=[text[span[0]: span[1]]] if span[1] > span[0] else [],
            unsat_core_ids=list(report.unsat_core),
            violation_margin=margin,
        )
        record.prompt = self.prompt
        record.rejected = text
        record.metadata.setdefault("spec_digest", self.telemetry.spec_digest)
        record.metadata.setdefault("culprit_field", culprit_name)
        record.metadata.setdefault("status", report.status)
        self._pending_conflicts.append(record)
        self.rl.record_conflict_branch(None, None, report.unsat_core, record=record)
        if culprit_name is not None:
            self._conflict_counts[culprit_name] = self._conflict_counts.get(culprit_name, 0) + 1
        return self._cut_for_field(culprit_name)

    def _culprit_from_report(self, report: VerificationReport) -> str | None:
        free = [field for field in self.fields if field not in self.rules]
        candidates: list[str] = []
        by_index = {clause.index: clause for clause in self.clauses}
        for outcome in report.violations:
            clause = by_index.get(outcome.index)
            if clause is None:
                # A structural (JSON-Schema) failure names the offending field(s) in its
                # message; prefer the earliest such field in emission order.
                for name in free:
                    if name in outcome.text and name not in candidates:
                        candidates.append(name)
                continue
            for vid in collect_var_ids(clause.expr):
                name = self.spec.id_to_name.get(vid)
                if name in free and name not in candidates:
                    candidates.append(name)
        if not candidates:
            candidates = free
        if not candidates:
            return None
        return candidates[-1]

    def _cut_for_field(self, name: str | None) -> bool:
        if name is None:
            return False
        if name in self.rules:
            name = self._upstream_free_field(name)
            if name is None:
                return False
        if name in self.containers:
            result = self.values.get(name)
            if isinstance(result, Mapping) and result:
                key = list(result.keys())[-1]
                return self._cut_element(name, key, result[key])
            return False
        value = self.values.get(name)
        number = as_number(value)
        if number is None:
            number = self._partial_numeric.get(name)
        if number is None:
            return False
        lo, hi = self._declared_bounds(name)
        # Unbounded declared axes still need a finite cut base around the rejected value.
        if not math.isfinite(lo):
            lo = number - max(1.0, abs(number))
        if not math.isfinite(hi):
            hi = number + max(1.0, abs(number))
        direction = self._cut_direction.get(name, 0)
        self._cut_direction[name] = 1 - direction
        # Bisection cuts are deliberately aggressive: a near-miss on a nonlinear clause
        # (e.g. a*b <= 0.1) cannot be fixed by excluding a 1e-9 sliver.  Removing half of
        # the infeasible half-space is still a valid domain cut and converges quickly.
        upper_cut = lo + (number - lo) * 0.5 if number > lo + 1e-15 else None
        lower_cut = hi - (hi - number) * 0.5 if number < hi - 1e-15 else None
        attempts = []
        if upper_cut is not None:
            attempts.append((lo, upper_cut))
        if lower_cut is not None:
            attempts.append((lower_cut, hi))
        if direction:
            attempts.reverse()
        for new_lo, new_hi in attempts:
            if new_lo <= new_hi and (new_lo > lo + 1e-15 or new_hi < hi - 1e-15):
                self._overrides[name] = (new_lo, new_hi)
                self._tabu.setdefault(name, set()).add(number)
                self.telemetry.domain_cuts.append(
                    {"field": name, "value": number, "bounds": [new_lo, new_hi]}
                )
                if name not in self.telemetry.backtrack_fields:
                    self.telemetry.backtrack_fields.append(name)
                return True
        return False

    def _cut_element(self, name: str, key: str, value: Any) -> bool:
        number = as_number(value)
        info = self.containers[name]
        if number is None:
            return False
        current_lo, current_hi = self._elem_overrides.get(
            (name, key), (info.elem_min, info.elem_max)
        )
        lo = info.elem_min if not math.isfinite(info.elem_min) else info.elem_min
        hi = info.elem_max if not math.isfinite(info.elem_max) else info.elem_max
        upper = lo + (number - lo) * 0.5 if math.isfinite(lo) and number > lo else None
        lower = hi - (hi - number) * 0.5 if math.isfinite(hi) and number < hi else None
        if upper is not None:
            self._elem_overrides[(name, key)] = (lo, upper)
        elif lower is not None:
            self._elem_overrides[(name, key)] = (lower, hi)
        else:
            return False
        self._tabu.setdefault((name, key), set()).add(float(value))
        self.telemetry.domain_cuts.append(
            {"field": f"{name}[{key!r}]", "value": value,
             "bounds": list(self._elem_overrides[(name, key)])}
        )
        label = f"{name}[{key!r}]"
        if label not in self.telemetry.backtrack_fields:
            self.telemetry.backtrack_fields.append(label)
        return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_factor(value: Any) -> list[list[float]] | None:
    """Validate/coerce a Cholesky factor matrix (ragged rows rejected)."""

    try:
        rows = list(value) if value is not None else []
        out: list[list[float]] = []
        for row in rows:
            items = list(row)
            out.append([float(v) for v in items])
    except (TypeError, ValueError):
        return None
    if not out:
        return None
    n = len(out)
    # Full or lower-triangular factors are both accepted; entries must be finite.
    if any(not math.isfinite(v) for row in out for v in row):
        return None
    if any(len(row) > n for row in out):
        return None
    return out


def _has_tokenizer(tokenizer: Any) -> bool:
    if tokenizer is None:
        return False
    return callable(getattr(tokenizer, "encode", None)) and callable(
        getattr(tokenizer, "decode", None)
    )


def _model_reasoning(model: Any, prompt: str, max_chars: int) -> str:
    """Ask the model for a CRANE scratchpad, if it can produce one.

    Two protocols are accepted: ``next_reasoning(prompt) -> str`` and a plain
    ``reasoning`` attribute holding pre-generated text.  Anything else yields no
    scratchpad instead of inventing one.
    """

    getter = getattr(model, "next_reasoning", None)
    text: Any = None
    if callable(getter):
        try:
            text = getter(prompt)
        except TypeError:
            try:
                text = getter()
            except Exception:
                text = None
        except Exception:
            text = None
    if not isinstance(text, str):
        candidate = getattr(model, "reasoning", None)
        text = candidate if isinstance(candidate, str) else None
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.strip()
    limit = max(0, int(max_chars or 0))
    return text[:limit] if limit else text


def _preference_score(report: VerificationReport | None) -> float | None:
    """Weighted fraction of soft preferences satisfied (``None`` when there are none)."""

    if report is None or not report.preferences:
        return None
    total = 0.0
    got = 0.0
    for outcome in report.preferences:
        weight = float(outcome.weight if outcome.weight and outcome.weight > 0 else 1.0)
        total += weight
        if outcome.satisfied is True:
            got += weight
    if total <= 0:
        return None
    return got / total


def _unary_preference(expr: Any, spec: VirSpec) -> tuple[str | None, str, float]:
    """Extract ``(field, op, constant)`` from ``p.field OP const`` preferences."""

    from .compiler.analysis import _const_number  # constant folding helper

    lhs = getattr(expr, "lhs", None)
    rhs = getattr(expr, "rhs", None)
    op = str(getattr(expr, "op", ""))
    flip = {"le": "ge", "lt": "gt", "ge": "le", "gt": "lt", "eq": "eq"}
    for target, other, direction in ((lhs, rhs, False), (rhs, lhs, True)):
        if target is None or getattr(target, "kind", None) != "var":
            continue
        const = _const_number(other)
        if const is None:
            continue
        name = spec.id_to_name.get(getattr(target, "var_id", None))
        if name is None:
            continue
        return name, (flip.get(op, op) if direction else op), float(const)
    return None, op, 0.0


def _mask_kl(raw: Mapping[str, float] | None, actions: Sequence[str]) -> float:
    """``-log(accepted probability mass)`` for the raw model distribution.

    Let ``P_raw`` be the model's softmax over the emitted logits and ``A`` the allowed
    actions after masking.  The distortion introduced by hard masking at this step is
    exactly ``-log(P_raw(A))``: the lost probability mass.  Models that only emit the
    allowed candidates see zero distortion; full-vocabulary logits measure true drift.
    """

    if not raw or not actions:
        return 0.0
    finite = [float(v) for v in raw.values() if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not finite:
        return 0.0
    peak = max(finite)
    total = sum(math.exp(v - peak) for v in finite)
    accepted = 0.0
    for action in actions:
        value = raw.get(action)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            accepted += math.exp(float(value) - peak)
    if total <= 0.0 or accepted <= 0.0:
        return 0.0
    mass = min(1.0, accepted / total)
    return -math.log(mass)


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return [1.0 / len(scores)] * len(scores)
    peak = max(finite)
    exps = [math.exp(s - peak) if math.isfinite(s) else 0.0 for s in scores]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


def _entropy(probabilities: Sequence[float]) -> float:
    total = 0.0
    for p in probabilities:
        if p > 0:
            total -= p * math.log(p)
    return total


# ---------------------------------------------------------------------------
# top-level generate
# ---------------------------------------------------------------------------

def generate(
    model: Any = None,
    prompt: str | None = None,
    spec: Any = None,
    return_proof: bool = False,
    *,
    tokenizer: Any = None,
    config: DecodeConfig | None = None,
    max_backtracks: int | None = None,
    temperature: float | None = None,
    volume_twist_beta: float | None = None,
    reasoning_mode: str | None = None,
    max_preference_repairs: int | None = None,
    seed: int | None = None,
    token_healing: bool = True,
    rl_trace_path: str | None = None,
    **kwargs: Any,
) -> VerifiedResult:
    """Run constrained decoding for ``spec`` and return a verified result.

    ``model`` may be any object exposing ``next_logits(prefix_text)``; see
    :mod:`vexx.testing` for a deterministic reference model.  GPU bridges in
    :mod:`vexx.bridges` adapt token-level logits to this protocol.
    """

    if spec is None:
        raise ValueError("generate() requires spec=<@vt.verified_schema function or VirSpec>")
    if model is None:
        raise ValueError(
            "generate() requires a model adapter; use vexx.testing.mock_model() for a "
            "deterministic reference run, or a vexx.bridges adapter for a real model"
        )
    if kwargs:
        raise TypeError(f"generate() got unexpected keyword arguments: {sorted(kwargs)}")
    cfg = config or DecodeConfig()
    if max_backtracks is not None:
        cfg.max_backtracks = int(max_backtracks)
    if temperature is not None:
        cfg.temperature = float(temperature)
    if volume_twist_beta is not None:
        beta = float(volume_twist_beta)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("volume_twist_beta must be a finite non-negative number")
        cfg.volume_twist_beta = beta
    if reasoning_mode is not None:
        if reasoning_mode not in ("auto", "strict", "crane_cot"):
            raise ValueError("reasoning_mode must be auto, strict or crane_cot")
        cfg.reasoning_mode = reasoning_mode
    if max_preference_repairs is not None:
        cfg.max_preference_repairs = max(0, int(max_preference_repairs))
    if rl_trace_path is not None:
        cfg.rl_trace_path = str(rl_trace_path)
    if seed is not None:
        cfg.seed = seed
    session = VexxSession(
        spec, model, cfg, tokenizer=tokenizer if token_healing else None
    )
    result = session.run(model, prompt=prompt or "")
    proof: ProofCertificate | None = None
    if return_proof and result.report.status == "PASS":
        proof = build_certificate(
            spec,
            result.values,
            metadata={
                "model": result.model_name,
                "prompt_sha256": _sha256_text(prompt or ""),
                "attempts": result.attempts,
            },
        )
    if result.report.status != "PASS":
        raise VerificationError(result.report)
    value: Any = result.report.model if result.report.model is not None else result.values
    return VerifiedResult(
        value=value,
        reasoning=result.reasoning,
        proof=proof,
        splices=list(result.splices),
        mined_equations=list(result.mined_equations),
        telemetry=result.telemetry.as_dict(),
        report=result.report,
        raw=result.text,
        spec_name=getattr(spec, "name", None) or "VirSpec",
    )


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# aliases expected by the API surface
DecodeEngine = VexxSession
SolverGuidedDecoder = VexxSession

__all__ = [
    "DeadEnd",
    "DecodeConfig",
    "DecodeEngine",
    "DecodeError",
    "DecodeResult",
    "DecodeTelemetry",
    "ModelProtocolError",
    "SolverGuidedDecoder",
    "StepTrace",
    "VexxSession",
    "generate",
]
