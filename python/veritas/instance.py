"""Concrete instance verification and 0-DoF derivation.

This is the part of ``veritas`` that makes a *result* trustworthy: given a schema plus a
concrete value (a Pydantic model, a dict, or raw JSON), decide whether every hard
constraint holds, fill in deterministic derived fields, and report precisely which clause
failed and by how much.

Everything here is pure Python and total: a missing value yields ``INCOMPLETE``, never a
false ``PASS``.  That property is what the proof-certificate layer and the decode engine
both rely on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .compiler.vir import VirExpr, VirSpec, ZeroDofRule
from .evaluate import (
    Comparison,
    as_number,
    eval_compare,
    eval_expr,
    expr_to_str,
    format_number,
)

PASS = "PASS"
FAIL = "FAIL"
INCOMPLETE = "INCOMPLETE"


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class VerificationError(Exception):
    """Raised when a concrete instance violates a schema's hard constraints."""

    def __init__(self, report: "VerificationReport"):
        self.report = report
        super().__init__(report.summary_line())


class InstanceFormatError(ValueError):
    """Raised when an instance cannot be interpreted as a mapping of fields."""


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

@dataclass
class ConstraintOutcome:
    index: int
    kind: str  # "require" | "derive" | "prefer"
    text: str
    satisfied: Optional[bool]
    tol: float = 1e-6
    residual: Optional[float] = None
    lhs: Any = None
    rhs: Any = None
    detail: str = ""
    weight: float = 1.0
    tag: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "index": self.index,
            "kind": self.kind,
            "constraint": self.text,
            "satisfied": self.satisfied,
            "tolerance": self.tol,
            "residual": self.residual,
            "detail": self.detail,
        }
        if self.kind == "prefer":
            payload["weight"] = self.weight
            payload["tag"] = self.tag
        return payload


@dataclass
class VerificationReport:
    """Result of checking one concrete instance against a compiled spec."""

    status: str = INCOMPLETE
    outcomes: List[ConstraintOutcome] = field(default_factory=list)
    preferences: List[ConstraintOutcome] = field(default_factory=list)
    unsat_core: List[int] = field(default_factory=list)
    derived: Dict[str, Any] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)
    model: Any = None
    model_error: Optional[str] = None
    relaxation: List[str] = field(default_factory=list)
    spec_digest: Optional[str] = None

    # -- convenience ------------------------------------------------------
    @property
    def ok(self) -> bool:
        return self.status == PASS

    @property
    def violations(self) -> List[ConstraintOutcome]:
        return [o for o in self.outcomes if o.satisfied is False]

    @property
    def unknown(self) -> List[ConstraintOutcome]:
        return [o for o in self.outcomes if o.satisfied is None]

    def summary_line(self) -> str:
        counts = {PASS: 0, FAIL: 0, INCOMPLETE: 0}
        for outcome in self.outcomes:
            if outcome.satisfied is True:
                counts[PASS] += 1
            elif outcome.satisfied is False:
                counts[FAIL] += 1
            else:
                counts[INCOMPLETE] += 1
        return (
            f"{self.status}: {counts[PASS]} satisfied, {counts[FAIL]} violated, "
            f"{counts[INCOMPLETE]} unevaluated"
        )

    def raise_if_failed(self) -> "VerificationReport":
        if self.status != PASS:
            raise VerificationError(self)
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary_line(),
            "outcomes": [o.as_dict() for o in self.outcomes],
            "preferences": [o.as_dict() for o in self.preferences],
            "unsat_core": list(self.unsat_core),
            "derived": dict(self.derived),
            "unresolved": list(self.unresolved),
            "missing": list(self.missing),
            "relaxation": list(self.relaxation),
            "model_error": self.model_error,
            "spec_digest": self.spec_digest,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return self.summary_line()


# ---------------------------------------------------------------------------
# instance lowering
# ---------------------------------------------------------------------------

def extract_data(instance: Any) -> Dict[str, Any]:
    """Normalize a Pydantic model / mapping / JSON string into a plain dict."""

    if instance is None:
        raise InstanceFormatError("instance is None")
    if isinstance(instance, Mapping):
        return dict(instance)
    if isinstance(instance, (str, bytes, bytearray)):
        text = instance.decode("utf-8") if isinstance(instance, (bytes, bytearray)) else instance
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InstanceFormatError(f"instance is not valid JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise InstanceFormatError("instance JSON must be an object")
        return dict(payload)
    # Pydantic v2 BaseModel (duck-typed to avoid a hard import cycle at module load)
    dump = getattr(instance, "model_dump", None)
    if callable(dump):
        data = dump(mode="python")
        if isinstance(data, Mapping):
            return dict(data)
    as_dict = getattr(instance, "dict", None)
    if callable(as_dict):  # pydantic v1 compatibility
        data = as_dict()
        if isinstance(data, Mapping):
            return dict(data)
    raise InstanceFormatError(
        f"unsupported instance type {type(instance).__name__}; expected a mapping, "
        "JSON string, or Pydantic model"
    )


def lower_instance(spec: VirSpec, instance: Any) -> Tuple[Dict[str, Any], Dict[Any, Any]]:
    """Return ``(data, env)`` where ``env`` is keyed by both var id and field name."""

    data = extract_data(instance)
    env: Dict[Any, Any] = {}
    for vid, info in spec.variables.items():
        name = info.get("name") or spec.id_to_name.get(vid)
        if name is None:
            continue
        if name in data and data[name] is not None:
            env[vid] = data[name]
            env[name] = data[name]
    return data, env


def _rule_targets(spec: VirSpec) -> Dict[str, ZeroDofRule]:
    out: Dict[str, ZeroDofRule] = {}
    for rule in spec.zero_dof_rules:
        name = rule.target_json_key
        if name is None and isinstance(rule.target_var, int):
            name = spec.id_to_name.get(rule.target_var)
        if name:
            out[name] = rule
    return out


def _evaluate_rule(rule: ZeroDofRule, env: Mapping[Any, Any]) -> Any:
    if callable(rule.rhs_eval):
        try:
            return rule.rhs_eval(env)
        except Exception:
            return None
    expr = rule.rhs_expr
    if expr is None:
        return None
    if getattr(expr, "kind", None) == "compare":
        # A derive rule is stored as `target == rhs`; compute the rhs side.
        expr = expr.rhs if expr.rhs is not None else expr.lhs
    return eval_expr(expr, env)


def fill_derived(
    spec: VirSpec,
    data: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """Evaluate 0-DoF rules in topological order.

    Returns ``(values, unresolved, derived)``.  ``derived`` only contains fields that
    were actually computed during this call, which the decode engine uses to count
    Jump-Forward savings.
    """

    values = dict(data)
    rules = _rule_targets(spec)
    unresolved: List[str] = []
    derived: Dict[str, Any] = {}
    order = list(spec.field_order) or list(spec.name_to_id.keys())
    # Any derived field missing from the topological order is appended at the end.
    for name in rules:
        if name not in order:
            order.append(name)

    for name in order:
        rule = rules.get(name)
        if rule is None:
            continue
        present = name in values and values[name] is not None
        if present and not overwrite:
            continue
        env: Dict[Any, Any] = {}
        for vid, info in spec.variables.items():
            field_name = info.get("name") or spec.id_to_name.get(vid)
            if field_name and field_name in values and values[field_name] is not None:
                env[vid] = values[field_name]
                env[field_name] = values[field_name]
        value = _evaluate_rule(rule, env)
        if value is None:
            unresolved.append(name)
            continue
        precision = max(0, int(getattr(rule, "precision_digits", 6) or 0))
        if isinstance(value, float) and precision:
            value = round(value, precision)
        values[name] = value
        derived[name] = value
    return values, unresolved, derived


def build_model(spec: VirSpec, values: Mapping[str, Any]) -> Tuple[Any, Optional[str]]:
    model_cls = spec.model
    if model_cls is None:
        return None, None
    validator = getattr(model_cls, "model_validate", None)
    try:
        if callable(validator):
            return validator(dict(values)), None
        return model_cls(**dict(values)), None
    except Exception as exc:  # pydantic raises ValidationError
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def _outcome_for(
    index: int,
    kind: str,
    expr: VirExpr,
    env: Mapping[Any, Any],
    names: Mapping[int, str],
    *,
    weight: float = 1.0,
    tag: str = "",
) -> ConstraintOutcome:
    text = expr_to_str(expr, names)
    if getattr(expr, "kind", None) == "compare":
        comparison: Comparison = eval_compare(expr, env)
        return ConstraintOutcome(
            index=index,
            kind=kind,
            text=text,
            satisfied=comparison.satisfied,
            tol=comparison.tol,
            residual=comparison.residual,
            lhs=comparison.lhs,
            rhs=comparison.rhs,
            detail=comparison.detail,
            weight=weight,
            tag=tag,
        )
    value = eval_expr(expr, env)
    satisfied = None if value is None else bool(value)
    residual = None
    a = as_number(value)
    if a is not None:
        residual = a - 1.0 if satisfied is False else 0.0
    return ConstraintOutcome(
        index=index,
        kind=kind,
        text=text,
        satisfied=satisfied,
        residual=residual,
        detail="" if satisfied is not None else "incomplete",
        weight=weight,
        tag=tag,
    )


def _relaxation_hint(outcome: ConstraintOutcome) -> Optional[str]:
    if outcome.satisfied is not False:
        return None
    if outcome.residual is not None:
        residual = outcome.residual
        direction = "increase" if residual < 0 else "decrease"
        return (
            f"constraint #{outcome.index} `{outcome.text}` is off by {residual:+.6g}; "
            f"{direction} the target/constant by {abs(residual):.6g} (or widen tol to "
            f"{abs(residual):.3g}) to make it satisfiable"
        )
    return f"constraint #{outcome.index} `{outcome.text}` is false"


def _collect(
    spec: VirSpec, env: Mapping[Any, Any]
) -> Tuple[List[ConstraintOutcome], List[ConstraintOutcome]]:
    names = spec.id_to_name
    outcomes: List[ConstraintOutcome] = []
    for i, expr in enumerate(spec.requires):
        outcomes.append(_outcome_for(i, "require", expr, env, names))
    for i, expr in enumerate(spec.derives, start=len(spec.requires)):
        outcomes.append(_outcome_for(i, "derive", expr, env, names))
    preferences: List[ConstraintOutcome] = []
    for i, pref in enumerate(spec.soft_preferences):
        if pref.expr is None:
            continue
        preferences.append(
            _outcome_for(
                i,
                "prefer",
                pref.expr,
                env,
                names,
                weight=getattr(pref, "weight", 1.0),
                tag=getattr(pref, "tag", "") or "",
            )
        )
    return outcomes, preferences


def _minimal_core(spec: VirSpec, env: Mapping[Any, Any], violated: Sequence[int]) -> List[int]:
    """Deletion-based minimal conflict set over the hard constraints."""

    if not violated:
        return []
    outcomes, _ = _collect(spec, env)
    by_index = {o.index: o for o in outcomes}
    core: List[int] = []
    for candidate in violated:
        remaining = [i for i in violated if i != candidate]
        if not remaining:
            core.append(candidate)
            continue
        if all(by_index.get(i) is not None and by_index[i].satisfied is True for i in remaining):
            core.append(candidate)
    return core or list(violated)


def verify_values(
    spec: VirSpec,
    values: Mapping[str, Any],
    *,
    derive_missing: bool = True,
    check_preferences: bool = True,
) -> VerificationReport:
    """Verify an already-lowered mapping of field values against ``spec``."""

    data = dict(values)
    derived: Dict[str, Any] = {}
    unresolved: List[str] = []
    if derive_missing:
        data, unresolved, derived = fill_derived(spec, data, overwrite=False)

    env: Dict[Any, Any] = {}
    missing: List[str] = []
    for vid, info in spec.variables.items():
        name = info.get("name") or spec.id_to_name.get(vid)
        if name is None:
            continue
        if name in data and data[name] is not None:
            env[vid] = data[name]
            env[name] = data[name]
        else:
            missing.append(name)

    outcomes, preferences = _collect(spec, env)
    violated = [o.index for o in outcomes if o.satisfied is False]
    unknown = [o.index for o in outcomes if o.satisfied is None]

    if violated:
        status = FAIL
    elif missing or unresolved or unknown:
        status = INCOMPLETE
    else:
        status = PASS

    core = _minimal_core(spec, env, violated) if violated else []
    hints = [h for h in (_relaxation_hint(o) for o in outcomes) if h]

    model = None
    model_error = None
    if status == PASS:
        model, model_error = build_model(spec, data)

    spec_hash = None
    try:
        from .serde import spec_digest

        spec_hash = spec_digest(spec)
    except Exception:  # pragma: no cover - digest is best effort here
        spec_hash = None

    return VerificationReport(
        status=status,
        outcomes=outcomes,
        preferences=preferences if check_preferences else [],
        unsat_core=core,
        derived=derived,
        unresolved=unresolved,
        missing=missing,
        values=data,
        model=model,
        model_error=model_error,
        relaxation=hints,
        spec_digest=spec_hash,
    )


def verify_instance(
    schema_or_spec: Any,
    instance: Any,
    *,
    derive_missing: bool = True,
) -> VerificationReport:
    """Verify a concrete instance against a ``@vt.verified_schema`` or ``VirSpec``."""

    spec = _coerce_spec(schema_or_spec)
    data, _ = lower_instance(spec, instance)
    return verify_values(spec, data, derive_missing=derive_missing)


def verify_json(schema_or_spec: Any, text: str) -> VerificationReport:
    spec = _coerce_spec(schema_or_spec)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InstanceFormatError(f"invalid JSON instance: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise InstanceFormatError("JSON instance must be an object")
    return verify_values(spec, dict(payload))


def _coerce_spec(schema_or_spec: Any) -> VirSpec:
    if isinstance(schema_or_spec, VirSpec):
        return schema_or_spec
    spec = getattr(schema_or_spec, "spec", None)
    if isinstance(spec, VirSpec):
        return spec
    if callable(schema_or_spec):
        # Allow passing a decorated function whose wrapper carries a spec.
        spec = getattr(schema_or_spec, "spec", None)
        if isinstance(spec, VirSpec):
            return spec
    raise TypeError(
        "expected a VirSpec or a @vt.verified_schema-decorated object, got "
        f"{type(schema_or_spec).__name__}"
    )


def describe_spec(spec: VirSpec) -> Dict[str, Any]:
    """JSON-able summary used by the CLI and error messages."""

    containers = []
    for cid, info in spec.containers.items():
        containers.append(
            {
                "name": info.name,
                "universe": info.universe,
                "cardinality": list(info.cardinality_bounds()),
                "element_bounds": [info.elem_min, info.elem_max],
                "target_sum": info.target_sum,
            }
        )
    return {
        "fields": list(spec.field_order),
        "free_fields": [f for f in spec.field_order if not spec.is_derived(f)],
        "derived_fields": spec.derived_names(),
        "requires": [expr_to_str(e, spec.id_to_name) for e in spec.requires],
        "derives": [expr_to_str(e, spec.id_to_name) for e in spec.derives],
        "prefers": [
            {
                "expr": expr_to_str(p.expr, spec.id_to_name),
                "weight": p.weight,
                "tag": p.tag,
            }
            for p in spec.soft_preferences
        ],
        "containers": containers,
        "zero_dof_rules": [
            {
                "target": r.target_json_key,
                "inputs": [
                    spec.id_to_name.get(v, str(v)) if isinstance(v, int) else str(v)
                    for v in r.input_vars
                ],
                "precision_digits": r.precision_digits,
            }
            for r in spec.zero_dof_rules
        ],
        "config": dict(spec.config),
        "digest": _safe_digest(spec),
    }


def _safe_digest(spec: VirSpec) -> Optional[str]:
    try:
        from .serde import spec_digest

        return spec_digest(spec)
    except Exception:  # pragma: no cover
        return None


__all__ = [
    "ConstraintOutcome",
    "FAIL",
    "INCOMPLETE",
    "InstanceFormatError",
    "PASS",
    "VerificationError",
    "VerificationReport",
    "build_model",
    "describe_spec",
    "extract_data",
    "fill_derived",
    "lower_instance",
    "verify_instance",
    "verify_json",
    "verify_values",
]
