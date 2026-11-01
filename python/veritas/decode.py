"""Solver-guided decoding: the reference neuro-symbolic decode engine.

This module is the runtime spine the specification describes:

1. **Tier-0** — the canonical JSON writer and grammar (keys are emitted deterministically;
   values are validated against the JSON pushdown automaton).
2. **Tier-1** — numeric interval masking via :class:`~veritas.interval.Interval` prefix
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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .cdcl_jumpback import CdclJumpbackEngine, ClauseCheckpoint, ConflictBranchRecord
from .compiler.vir import ContainerInfo, VirExpr, VirSpec
from .evaluate import as_number, collect_var_ids, eval_expr, expr_to_str, format_number
from .instance import VerificationReport, _coerce_spec, describe_spec, verify_values
from .interval import Hc4AffineContractor, Interval
from .json_pda import (
    JSONPushdownAutomaton,
    is_complete_json,
    json_number,
    json_value,
    render_object,
)
from .linear import compile_clauses, simplify_clause
from .proof.certificate import ProofCertificate, VerifiedResult, build_certificate
from .rl.collector import VeritasRLCollector
from .serde import spec_digest
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
    def __init__(self, field_name: Optional[str], reason: str):
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
    seed: Optional[int] = 0
    derive_missing: bool = True
    enforce_grammar: bool = True


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

    def as_dict(self) -> Dict[str, Any]:
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
    jump_forward_fields: int = 0
    jump_forward_chars: int = 0
    backtracks: int = 0
    backtrack_fields: List[str] = field(default_factory=list)
    domain_cuts: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    final_status: str = ""
    spec_digest: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload["backtrack_fields"] = list(self.backtrack_fields)
        payload["domain_cuts"] = list(self.domain_cuts)
        payload["elapsed_ms"] = round(self.elapsed_ms, 3)
        return payload


@dataclass
class DecodeResult:
    text: str
    values: Dict[str, Any]
    report: VerificationReport
    telemetry: DecodeTelemetry
    trace: List[StepTrace]
    attempts: int = 1
    model_name: str = ""
    prompt: str = ""

    @property
    def ok(self) -> bool:
        return self.report.status == "PASS"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "values": self.values,
            "status": self.report.status,
            "telemetry": self.telemetry.as_dict(),
            "attempts": self.attempts,
            "model": self.model_name,
            "report": self.report.as_dict(),
        }


class VeritasSession:
    """One constrained generation over one compiled spec."""

    def __init__(
        self,
        schema_or_spec: Any,
        model: Any = None,
        config: Optional[DecodeConfig] = None,
    ):
        self.spec: VirSpec = _coerce_spec(schema_or_spec)
        self.model = model
        self.config = config or DecodeConfig()
        self.rng = random.Random(self.config.seed)
        self.telemetry = DecodeTelemetry(spec_digest=spec_digest(self.spec))
        self.trace: List[StepTrace] = []
        self.prompt = ""
        self.zero = ZeroDofEngine.from_spec(self.spec)
        self.cdcl = CdclJumpbackEngine()
        self.rl = VeritasRLCollector()

        names = list(self.spec.field_order)
        for name in self.spec.name_to_id:
            if name not in names:
                names.append(name)
        self.fields: List[str] = [n for n in names if n in self.spec.name_to_id]
        self.vid: Dict[str, int] = {n: self.spec.name_to_id[n] for n in self.fields}
        self.types: Dict[str, Any] = {
            n: (self.spec.variables.get(self.vid[n]) or {}).get("type") for n in self.fields
        }
        self.rules = {r.target_json_key: r for r in self.spec.zero_dof_rules if r.target_json_key}
        self.containers: Dict[str, ContainerInfo] = {
            info.name: info for info in self.spec.containers.values() if info.name
        }
        self.clauses = compile_clauses(self.spec)
        self._field_clause_index: Dict[str, int] = {}
        for clause in self.clauses:
            for vid in collect_var_ids(clause.expr):
                name = self.spec.id_to_name.get(vid)
                if name and name not in self._field_clause_index:
                    self._field_clause_index[name] = clause.index

        self.values: Dict[str, Any] = {}
        self.text = ""
        self._attempt = 0
        self._step = 0
        self._partial_numeric: Dict[str, float] = {}
        self._attempt_checkpoints: List[ClauseCheckpoint] = []
        self._overrides: Dict[str, Tuple[float, float]] = {}
        self._elem_overrides: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._tabu: Dict[Any, Set[float]] = {}
        self._cut_direction: Dict[str, int] = {}
        self.contractor = self._build_contractor()
        # Snapshot of the feasible box *before* any assignment; domain cuts are computed
        # against this, never against a committed singleton.
        self._initial_domains: Dict[int, Interval] = {
            key: Interval(bound.lo, bound.hi) for key, bound in self.contractor.domains.items()
        }

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
            contractor.add_variable(
                self.vid[name],
                Interval(float(lo) if lo is not None else float("-inf"),
                         float(hi) if hi is not None else float("inf")),
            )
        for clause in self.clauses:
            simple = simplify_clause(clause)
            if simple is None:
                continue
            op, coeffs, rhs = simple
            if not coeffs or any(key not in contractor.domains for key in coeffs):
                continue
            try:
                contractor.add_constraint(("linear", dict(coeffs), float(rhs), op))
            except Exception:
                continue
        contractor.propagate()
        return contractor

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, model: Any = None, *, prompt: str = "") -> DecodeResult:
        model = model if model is not None else self.model
        if model is None:
            raise ModelProtocolError(
                "no model supplied; pass a model exposing next_logits(prefix) -> Mapping[str, float], "
                "or use veritas.testing.mock_model() for a deterministic reference model"
            )
        self.prompt = prompt or ""
        start = time.perf_counter()
        best: Optional[DecodeResult] = None
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
                if not self._cut_for_field(exc.field_name):
                    break
                continue

            report = verify_values(self.spec, self.values, derive_missing=self.config.derive_missing)
            self.telemetry.solver_checks += 1
            result = self._finalize(text, report, attempts, model, start)
            if report.status == "PASS":
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
    # attempt lifecycle
    # ------------------------------------------------------------------
    def _reset_attempt(self) -> None:
        self.values = {}
        self.text = ""
        self._step = 0
        self._partial_numeric = {}
        self._attempt_checkpoints = []
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
        if name in self.rules:
            return self._emit_derived(name, model)
        if name in self.containers:
            return self._emit_container(name, model)
        return self._emit_scalar(name, model)

    def _emit_derived(self, name: str, model: Any) -> Any:
        value = self.zero.evaluate(name, self.values)
        if value is None:
            ready = self.zero.is_ready(name, self.values)
            culprit = self._upstream_free_field(name)
            raise DeadEnd(culprit, f"derived field {name!r} is not evaluable (ready={ready})")
        precision = max(0, int(getattr(self.rules[name], "precision_digits", 6) or 0))
        rendered = json_number(value, precision) if isinstance(value, (int, float)) else json_value(value)
        self.text += rendered
        self.telemetry.jump_forward_fields += 1
        self.telemetry.jump_forward_chars += len(rendered)
        self._record(StepTrace, name, rendered, kind="splice",
                     note=f"0-DoF: computed {rendered}")
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

    def _upstream_free_field(self, name: str) -> Optional[str]:
        rule = self.rules.get(name)
        upstream: List[str] = []
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
            text, value = self._emit_number(name, model, (lo, hi), int_only=True, tabu_key=name)
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
        if kind in _CONTAINER_KINDS:
            return self._emit_container(name, model)
        # string / unsupported scalar types: a bounded-free string is safe and honest.
        value = self._emit_string(name, model)
        self.text += json_value(value)
        return value

    def _emit_string(self, name: str, model: Any) -> str:
        chars: List[str] = []
        for _ in range(self.config.max_string_length + 1):
            actions = [chr(c) for c in range(0x20, 0x7F) if chr(c) not in {'"', "\\"}]
            stop_ok = len(chars) >= 1
            if stop_ok:
                actions = [_STOP] + actions
            if len(chars) >= self.config.max_string_length:
                if not stop_ok:
                    raise DeadEnd(name, "string exceeds maximum length")
                actions = [_STOP]
            choice = self._pick(name, model, self.text, actions, candidates=96)
            if choice == _STOP:
                return "".join(chars)
            chars.append(choice)
            self.text += choice
            self._record(StepTrace, name, choice)
        raise DeadEnd(name, "string did not terminate")

    def _commit_scalar(self, name: str, vid: int, value: float) -> None:
        # Record the assignment before propagation so a conflict can name this field as
        # the culprit and cut its domain on the next attempt.
        self.values[name] = value
        if vid not in self.contractor.domains:
            return
        # Propagating a rejected assignment must not poison the contractor: restore the
        # pre-assignment domains so the next attempt still has a usable feasible box.
        saved = {key: Interval(bound.lo, bound.hi) for key, bound in self.contractor.domains.items()}
        self.contractor.set_domain(vid, Interval(value, value))
        if not self.contractor.propagate():
            self.contractor.domains = saved
            raise DeadEnd(name, f"contractor became infeasible after {name}={value:g}")

    # ------------------------------------------------------------------
    # numeric emission
    # ------------------------------------------------------------------
    def _declared_bounds(self, name: str) -> Tuple[float, float]:
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
        override = self._overrides.get(name)
        if override is not None:
            lo = max(lo, override[0])
            hi = min(hi, override[1])
        return lo, hi

    def _bounds(self, name: str) -> Tuple[float, float]:
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
        bounds: Tuple[float, float],
        *,
        int_only: bool = False,
        tabu_key: Any = None,
    ) -> Tuple[str, float]:
        lo, hi = bounds
        if lo > hi + 1e-12:
            raise DeadEnd(name, f"empty numeric domain [{lo:g}, {hi:g}]")
        interval = Interval(lo, hi)
        prefix = ""
        local_steps = 0
        while True:
            local_steps += 1
            if local_steps > self.config.max_steps or self._step > self.config.max_steps:
                raise DeadEnd(name, "step budget exhausted")
            actions: List[str] = []
            for ch in "0123456789.-":
                candidate = prefix + ch
                if not self._number_prefix_ok(candidate, int_only):
                    continue
                if Interval.from_numeric_prefix(candidate, interval, self.config.allow_exponent).is_empty():
                    self.telemetry.contractor_prunes += 1
                    continue
                actions.append(ch)
            stop_ok = self._number_stop_ok(prefix, interval, int_only, tabu_key)
            if stop_ok:
                actions.insert(0, _STOP)
            if not actions:
                raise DeadEnd(name, f"no admissible continuation for number prefix {prefix!r} in [{lo:g}, {hi:g}]")

            choice = self._pick(name, model, self.text + prefix, actions, candidates=13)
            if choice != _STOP:
                prefix += choice
                try:
                    self._partial_numeric[name] = float(prefix)
                except ValueError:
                    pass
            if choice == _STOP:
                value = float(prefix)
                if int_only:
                    value = float(int(round(value)))
                return prefix, value
            self.text += choice
            self._record(StepTrace, name, choice, allowed=len(actions))

    def _number_prefix_ok(self, prefix: str, int_only: bool) -> bool:
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
        if integer and len(integer) > self.config.max_integer_digits:
            return False
        if integer and len(integer) > 1 and integer[0] == "0":
            return False
        if dot and len(fraction) > self.config.max_fraction_digits:
            return False
        return True

    def _number_stop_ok(
        self,
        prefix: str,
        interval: Interval,
        int_only: bool,
        tabu_key: Any,
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
    def _emit_container(self, name: str, model: Any) -> Dict[str, Any]:
        info = self.containers[name]
        keys = list(info.universe or [])
        if not keys:
            self.text += "{}"
            self._record(StepTrace, name, "{}", kind="close",
                         note="no key universe declared; emitting an empty container")
            return {}
        self.text += "{"
        total = 0.0
        emitted = 0
        first = True
        result: Dict[str, Any] = {}
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
                name, model, (lo, hi), tabu_key=(name, key)
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
    ) -> Tuple[float, float]:
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
        if emitted < max(0, info.min_len):
            return False
        if info.target_sum is None:
            return emitted >= max(1, info.min_len)
        tolerance = max(info.sum_tol, 1e-9)
        return abs(info.target_sum - total) <= tolerance

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def _pick(self, field_name: str, model: Any, prefix: str, actions: Sequence[str], candidates: int = 0) -> str:
        logits = self._logits(model, prefix)
        scores = [float(logits.get(action, 0.0)) for action in actions]
        if self.config.volume_twist_beta and field_name in self.vid:
            # 1-D SHIM-style adjustment: prefer continuations that leave the widest
            # feasible interval.  This is the interval analogue of polytope volume.
            lo, hi = self._bounds(field_name)
            for index, action in enumerate(actions):
                if action == _STOP:
                    continue
                candidate = prefix + action
                interval = Interval.from_numeric_prefix(candidate, Interval(lo, hi))
                width = interval.width()
                base = max(hi - lo, 1e-12)
                if math.isfinite(width):
                    scores[index] += self.config.volume_twist_beta * math.log(max(width, 1e-12) / base)
        entropy, choice = self._select(actions, scores)
        self.telemetry.masked_candidates += max(0, candidates - len(actions)) if candidates else 0
        self.telemetry.sampled_units += 1
        self._last_entropy = entropy
        return choice

    def _pick_strings(self, field_name: str, model: Any, actions: Sequence[str]) -> str:
        logits = self._logits(model, self.text)
        scores = [float(logits.get(action, 0.0)) for action in actions]
        entropy, choice = self._select(actions, scores)
        self._last_entropy = entropy
        return choice

    def _select(self, actions: Sequence[str], scores: Sequence[float]) -> Tuple[float, str]:
        probabilities = _softmax(scores)
        entropy = _entropy(probabilities)
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
        if cls is StepTrace:
            self.trace.append(
                StepTrace(
                    step=self._next_step(),
                    attempt=self._attempt,
                    field_name=field_name,
                    char=text,
                    kind=kind,
                    allowed=allowed,
                    entropy=getattr(self, "_last_entropy", 0.0),
                    note=note,
                )
            )

    # ------------------------------------------------------------------
    # backtracking / domain cuts
    # ------------------------------------------------------------------
    def _apply_report_cut(self, report: VerificationReport, text: str) -> bool:
        self._attempt_checkpoints = []
        for name in self.fields:
            if name in self.rules:
                continue
            value = self.values.get(name)
            number = as_number(value) if not isinstance(value, dict) else None
            self._attempt_checkpoints.append(
                ClauseCheckpoint(
                    clause_id=self._field_clause_index.get(name, 0),
                    var_id=name,
                    committed_val=number if number is not None else 0.0,
                    start_token_idx=0,
                    end_token_idx=len(text),
                    mean_shannon_entropy=1.0,
                    token_sequence=[json_value(value)] if value is not None else [],
                    conflict_count=0,
                )
            )
        self.cdcl.checkpoints.extend(self._attempt_checkpoints)
        culprit_name = None
        if report.unsat_core:
            self.cdcl.jumpback(report.unsat_core)
            culprit = self.cdcl.select_culprit(report.unsat_core)
            if culprit is not None:
                culprit_name = str(culprit.var_id)
        if culprit_name is None:
            culprit_name = self._culprit_from_report(report)
        record = ConflictBranchRecord(
            prefix_tokens=[text],
            rejected_clause_tokens=[],
            unsat_core_ids=list(report.unsat_core),
            violation_margin=abs(report.violations[0].residual) if report.violations and report.violations[0].residual is not None else 0.0,
        )
        self.rl.record_conflict_branch(None, None, report.unsat_core, record=record)
        return self._cut_for_field(culprit_name)

    def _culprit_from_report(self, report: VerificationReport) -> Optional[str]:
        by_index = {clause.index: clause for clause in self.clauses}
        candidates: List[str] = []
        for outcome in report.violations:
            clause = by_index.get(outcome.index)
            if clause is None:
                continue
            for vid in collect_var_ids(clause.expr):
                name = self.spec.id_to_name.get(vid)
                if name in self.fields and name not in self.rules:
                    candidates.append(name)
        if not candidates:
            candidates = [f for f in self.fields if f not in self.rules]
        if not candidates:
            return None
        return candidates[-1]

    def _cut_for_field(self, name: Optional[str]) -> bool:
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
        return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _softmax(scores: Sequence[float]) -> List[float]:
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
    prompt: Optional[str] = None,
    spec: Any = None,
    return_proof: bool = False,
    *,
    tokenizer: Any = None,
    config: Optional[DecodeConfig] = None,
    max_backtracks: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs: Any,
) -> VerifiedResult:
    """Run constrained decoding for ``spec`` and return a verified result.

    ``model`` may be any object exposing ``next_logits(prefix_text)``; see
    :mod:`veritas.testing` for a deterministic reference model.  GPU bridges in
    :mod:`veritas.bridges` adapt token-level logits to this protocol.
    """

    if spec is None:
        raise ValueError("generate() requires spec=<@vt.verified_schema function or VirSpec>")
    if model is None:
        raise ValueError(
            "generate() requires a model adapter; use veritas.testing.mock_model() for a "
            "deterministic reference run, or a veritas.bridges adapter for a real model"
        )
    cfg = config or DecodeConfig()
    if max_backtracks is not None:
        cfg.max_backtracks = int(max_backtracks)
    if temperature is not None:
        cfg.temperature = float(temperature)
    session = VeritasSession(spec, model, cfg)
    result = session.run(model, prompt=prompt or "")
    proof: Optional[ProofCertificate] = None
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
    value: Any = result.report.model if result.report.model is not None else result.values
    return VerifiedResult(
        value=value,
        reasoning="",
        proof=proof,
        telemetry=result.telemetry.as_dict(),
        report=result.report,
        raw=result.text,
        spec_name=getattr(spec, "name", None) or "VirSpec",
    )


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# aliases expected by the API surface
DecodeEngine = VeritasSession
SolverGuidedDecoder = VeritasSession

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
    "VeritasSession",
    "generate",
]
