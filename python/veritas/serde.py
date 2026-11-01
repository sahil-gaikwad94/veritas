"""Serialization of the Veritas Intermediate Representation (VIR).

Proof certificates must be verifiable *offline*, with no access to the model, the
process that produced the output, or the original Python source.  That requires the
semantic content of a compiled schema to travel with the certificate.  This module
provides the canonical, versioned JSON encoding used for exactly that, plus the
content-addressed digest that binds a certificate to a spec.

The encoding is deliberately explicit (tagged dicts for sets/tuples/real dicts) so it
survives JSON round-trips without silently changing key types — a covariance matrix
keyed by ``("AAPL", "NVDA")`` tuples must not come back as a plain string key.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from .compiler.vir import (
    ContainerInfo,
    SoftPreference,
    VirExpr,
    VirSpec,
    VirType,
    ZeroDofRule,
)

SERDE_VERSION = 3


# ---------------------------------------------------------------------------
# typing helpers
# ---------------------------------------------------------------------------

def _type_to_dict(vtype: VirType | None) -> dict[str, Any] | None:
    if vtype is None:
        return None
    return {
        "kind": vtype.kind,
        "min": vtype.min,
        "max": vtype.max,
        "tol": vtype.tol,
        "width": vtype.width,
        "universe": list(vtype.universe) if vtype.universe is not None else None,
        "min_len": vtype.min_len,
        "max_len": vtype.max_len,
        "val_min": vtype.val_min,
        "val_max": vtype.val_max,
        "nullable": vtype.nullable,
        "min_exclusive": vtype.min_exclusive,
        "max_exclusive": vtype.max_exclusive,
        "item_type": _type_to_dict(vtype.item_type),
        "properties": {k: _type_to_dict(v) for k, v in vtype.properties.items()},
        "required": list(vtype.required),
        "additional_properties": vtype.additional_properties,
        "model_name": vtype.model_name,
        "container_kind": vtype.container_kind,
    }


def _type_from_dict(payload: Mapping[str, Any] | None) -> VirType | None:
    if not payload:
        return None
    return VirType(
        kind=payload.get("kind", "any"),
        min=payload.get("min"),
        max=payload.get("max"),
        tol=payload.get("tol", 1e-6),
        width=payload.get("width"),
        universe=payload.get("universe"),
        min_len=payload.get("min_len", 0),
        max_len=payload.get("max_len", 0),
        val_min=payload.get("val_min", float("-inf")),
        val_max=payload.get("val_max", float("inf")),
        nullable=bool(payload.get("nullable", False)),
        min_exclusive=bool(payload.get("min_exclusive", False)),
        max_exclusive=bool(payload.get("max_exclusive", False)),
        item_type=_type_from_dict(payload.get("item_type")),
        properties={k: _type_from_dict(v) for k, v in (payload.get("properties") or {}).items()},
        required=list(payload.get("required") or []),
        additional_properties=payload.get("additional_properties", True),
        model_name=payload.get("model_name"),
        container_kind=payload.get("container_kind", "dict"),
    )


def _encode_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__float__": "inf" if value == math.inf else "-inf" if value == -math.inf else "nan"}
    if isinstance(value, list):
        return [_encode_value(v) for v in value]
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_value(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        items = [_encode_value(v) for v in value]
        items.sort(key=lambda v: repr(v))
        return {"__set__": items}
    if isinstance(value, dict):
        pairs = [[_encode_value(k), _encode_value(v)] for k, v in value.items()]
        pairs.sort(key=lambda kv: repr(kv[0]))
        return {"__pairs__": pairs}
    raise TypeError(f"cannot serialize {type(value).__name__} into VIR")


def _decode_value(payload: Any, _depth: int = 0) -> Any:
    if _depth > 128:
        raise ValueError("VIR value nesting exceeds 128")
    if isinstance(payload, list):
        return [_decode_value(v, _depth + 1) for v in payload]
    if isinstance(payload, dict):
        tags = {"__float__", "__tuple__", "__set__", "__pairs__", "__repr__"} & payload.keys()
        if tags:
            if len(payload) != 1:
                raise ValueError("tagged VIR value has unexpected sibling keys")
            tag = next(iter(tags))
            inner = payload[tag]
            if tag == "__float__":
                if inner not in ("inf", "-inf", "nan"):
                    raise ValueError("invalid tagged float")
                return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}[inner]
            if tag == "__repr__":
                raise ValueError("opaque Python objects are not portable VIR")
            if not isinstance(inner, list):
                raise ValueError(f"{tag} payload must be a list")
            if tag == "__tuple__":
                return tuple(_decode_value(v, _depth + 1) for v in inner)
            if tag == "__set__":
                return set(_decode_value(v, _depth + 1) for v in inner)
            out = {}
            for pair in inner:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("__pairs__ entries must be [key, value]")
                key = _decode_value(pair[0], _depth + 1)
                if key in out:
                    raise ValueError("duplicate key in tagged VIR mapping")
                out[key] = _decode_value(pair[1], _depth + 1)
            return out
        return {k: _decode_value(v, _depth + 1) for k, v in payload.items()}
    return payload


# ---------------------------------------------------------------------------
# expression helpers
# ---------------------------------------------------------------------------

_EXPR_FIELDS = (
    "kind",
    "var_id",
    "name",
    "op",
    "tol",
    "container_id",
    "target_sum",
    "elem_min",
    "elem_max",
    "min_remaining",
    "max_remaining",
    "weights_vars",
    "cholesky_l",
    "weight",
    "tag",
)


def expr_to_dict(expr: VirExpr | None) -> dict[str, Any] | None:
    if expr is None:
        return None
    payload: dict[str, Any] = {field: getattr(expr, field, None) for field in _EXPR_FIELDS}
    payload["value"] = _encode_value(expr.value)
    payload["args"] = [expr_to_dict(a) for a in expr.args]
    payload["lhs"] = expr_to_dict(expr.lhs)
    payload["rhs"] = expr_to_dict(expr.rhs)
    payload["iterable"] = expr_to_dict(expr.iterable) if hasattr(expr.iterable, "kind") else _encode_value(expr.iterable)
    payload["predicate"] = expr_to_dict(expr.predicate)
    return payload


def expr_from_dict(payload: Mapping[str, Any] | None) -> VirExpr | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or not isinstance(payload.get("kind"), str) or not payload["kind"]:
        raise ValueError("VIR expression must be an object with a non-empty kind")
    for key in ("args", "weights_vars", "cholesky_l"):
        if payload.get(key) is not None and not isinstance(payload[key], list):
            raise ValueError(f"VIR expression {key} must be a list")
    tol = payload.get("tol", 0.0)
    if tol is not None and (isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol < 0):
        raise ValueError("VIR expression tolerance must be finite and non-negative")
    kwargs: dict[str, Any] = {field: payload.get(field) for field in _EXPR_FIELDS}
    kwargs["tol"] = payload.get("tol", 0.0)
    if kwargs["tol"] is None:
        kwargs["tol"] = 0.0
    kwargs["weight"] = payload.get("weight", 1.0) if payload.get("weight") is not None else 1.0
    kwargs["tag"] = payload.get("tag", "") or ""
    kwargs["value"] = _decode_value(payload.get("value"))
    kwargs["args"] = [expr_from_dict(a) for a in payload.get("args") or []]
    if any(a is None for a in kwargs["args"]):
        raise ValueError("VIR expression args may not contain null")
    kwargs["lhs"] = expr_from_dict(payload.get("lhs"))
    kwargs["rhs"] = expr_from_dict(payload.get("rhs"))
    iterable = payload.get("iterable")
    if isinstance(iterable, dict) and "kind" in iterable:
        kwargs["iterable"] = expr_from_dict(iterable)
    else:
        kwargs["iterable"] = _decode_value(iterable)
    kwargs["predicate"] = expr_from_dict(payload.get("predicate"))
    return VirExpr(**kwargs)


def _rule_to_dict(rule: ZeroDofRule) -> dict[str, Any]:
    return {
        "target_var": rule.target_var,
        "target_json_key": rule.target_json_key,
        "input_vars": list(rule.input_vars),
        "rhs_expr": expr_to_dict(rule.rhs_expr),
        "precision_digits": rule.precision_digits,
    }


def _rule_from_dict(payload: Mapping[str, Any]) -> ZeroDofRule:
    return ZeroDofRule(
        target_var=payload.get("target_var"),
        target_json_key=payload.get("target_json_key"),
        input_vars=list(payload.get("input_vars") or []),
        rhs_expr=expr_from_dict(payload.get("rhs_expr")),
        rhs_eval=None,
        precision_digits=payload.get("precision_digits", 6),
    )


def _preference_to_dict(pref: SoftPreference) -> dict[str, Any]:
    return {
        "id": pref.id,
        "tag": pref.tag,
        "weight": pref.weight,
        "expr": expr_to_dict(pref.expr),
    }


def _preference_from_dict(payload: Mapping[str, Any]) -> SoftPreference:
    return SoftPreference(
        id=payload.get("id", 0),
        tag=payload.get("tag", "") or "",
        weight=payload.get("weight", 1.0),
        expr=expr_from_dict(payload.get("expr")),
    )


def _container_to_dict(info: ContainerInfo) -> dict[str, Any]:
    return {
        "var_id": info.var_id,
        "name": info.name,
        "universe": list(info.universe) if info.universe is not None else None,
        "min_len": info.min_len,
        "max_len": info.max_len,
        "elem_min": info.elem_min,
        "elem_max": info.elem_max,
        "target_sum": info.target_sum,
        "sum_tol": info.sum_tol,
        "sum_op": info.sum_op,
        "container_kind": info.container_kind,
    }


def _container_from_dict(payload: Mapping[str, Any]) -> ContainerInfo:
    return ContainerInfo(
        var_id=payload.get("var_id", -1),
        name=payload.get("name", "") or "",
        universe=payload.get("universe"),
        min_len=payload.get("min_len", 0),
        max_len=payload.get("max_len"),
        elem_min=payload.get("elem_min", float("-inf")),
        elem_max=payload.get("elem_max", float("inf")),
        target_sum=payload.get("target_sum"),
        sum_tol=payload.get("sum_tol", 1e-6),
        sum_op=payload.get("sum_op", "eq"),
        container_kind=payload.get("container_kind", "dict"),
    )


# ---------------------------------------------------------------------------
# spec <-> dict
# ---------------------------------------------------------------------------

def spec_to_dict(spec: VirSpec, *, include_model_name: bool = True) -> dict[str, Any]:
    model_name = None
    if include_model_name and spec.model is not None:
        model_name = getattr(spec.model, "__name__", None) or str(spec.model)
    return {
        "version": SERDE_VERSION,
        "model": model_name,
        "variables": {
            str(vid): {"name": info.get("name"), "type": _type_to_dict(info.get("type"))}
            for vid, info in spec.variables.items()
        },
        "id_to_name": {str(k): v for k, v in spec.id_to_name.items()},
        "name_to_id": dict(spec.name_to_id),
        "field_order": list(spec.field_order),
        "requires": [expr_to_dict(e) for e in spec.requires],
        "derives": [expr_to_dict(e) for e in spec.derives],
        "prefers": [expr_to_dict(e) for e in spec.prefers],
        "zero_dof_rules": [_rule_to_dict(r) for r in spec.zero_dof_rules],
        "soft_preferences": [_preference_to_dict(p) for p in spec.soft_preferences],
        "containers": [_container_to_dict(c) for c in spec.containers.values()],
        "config": _encode_value(spec.config),
        "model_schema": spec.model_schema,
        "field_aliases": dict(spec.field_aliases),
        "portable_model": spec.portable_model,
        "compiler_diagnostics": list(spec.compiler_diagnostics),
    }


def _require_mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _validate_spec_payload(payload):
    _require_mapping(payload, "spec payload")
    if payload.get("version") != SERDE_VERSION:
        raise ValueError(f"unsupported VIR version {payload.get('version')!r}; expected {SERDE_VERSION}")
    variables = _require_mapping(payload.get("variables"), "variables")
    ids = set()
    names = set()
    for raw_id, info in variables.items():
        try:
            vid = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid variable id {raw_id!r}") from exc
        if vid in ids:
            raise ValueError(f"duplicate variable id {vid}")
        ids.add(vid)
        info = _require_mapping(info, f"variable {vid}")
        name = info.get("name")
        if not isinstance(name, str) or name in names:
            raise ValueError(f"variable {vid} has invalid/duplicate name {name!r}")
        names.add(name)
        if info.get("type") is not None:
            _require_mapping(info["type"], f"variable {vid} type")
    id_to_name = _require_mapping(payload.get("id_to_name"), "id_to_name")
    name_to_id = _require_mapping(payload.get("name_to_id"), "name_to_id")
    if {int(k) for k in id_to_name} != ids or set(name_to_id) != names:
        raise ValueError("variable id/name tables are inconsistent")
    for key, name in id_to_name.items():
        if name_to_id.get(name) != int(key) or variables[str(int(key))]["name"] != name:
            raise ValueError("variable id/name bijection is inconsistent")
    order = _require_list(payload.get("field_order"), "field_order")
    if len(set(order)) != len(order) or not set(order) <= names:
        raise ValueError("field_order contains unknown/duplicate fields")
    for key in ("requires", "derives", "prefers", "zero_dof_rules", "soft_preferences", "containers", "compiler_diagnostics"):
        _require_list(payload.get(key), key)
    for rule in payload["zero_dof_rules"]:
        rule = _require_mapping(rule, "zero_dof_rule")
        if rule.get("target_var") not in ids or not set(rule.get("input_vars") or []) <= ids:
            raise ValueError("derivation rule references unknown variables")
        if rule.get("target_json_key") not in names:
            raise ValueError("derivation rule has unknown target field")
        expr_from_dict(rule.get("rhs_expr"))
    for pref in payload["soft_preferences"]:
        pref = _require_mapping(pref, "soft_preference")
        weight = pref.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("preference weight must be finite and non-negative")
        expr_from_dict(pref.get("expr"))
    for container in payload["containers"]:
        container = _require_mapping(container, "container")
        if container.get("var_id") not in ids:
            raise ValueError("container references unknown variable")
    for key in ("requires", "derives", "prefers"):
        for expr in payload[key]:
            expr_from_dict(expr)
    aliases = _require_mapping(payload.get("field_aliases"), "field_aliases")
    if not set(aliases) <= names or any(not isinstance(v, str) for v in aliases.values()):
        raise ValueError("field_aliases is invalid")
    if not isinstance(payload.get("portable_model"), bool):
        raise ValueError("portable_model must be boolean")
    schema = payload.get("model_schema")
    if schema is not None:
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(schema)
    config = _decode_value(payload.get("config") or {})
    if not isinstance(config, dict):
        raise ValueError("config must decode to an object")


def spec_from_dict(payload: Mapping[str, Any]) -> VirSpec:
    _validate_spec_payload(payload)
    def _int_key_dict(raw: Mapping[Any, Any]) -> dict[int, Any]:
        out: dict[int, Any] = {}
        for key, value in (raw or {}).items():
            out[int(key)] = value
        return out

    variables: dict[int, dict[str, Any]] = {}
    for vid, info in _int_key_dict(payload.get("variables") or {}).items():
        variables[vid] = {
            "name": info.get("name"),
            "type": _type_from_dict(info.get("type")),
        }
    containers: dict[int, ContainerInfo] = {}
    for entry in payload.get("containers") or []:
        info = _container_from_dict(entry)
        containers[info.var_id] = info
    return VirSpec(
        model=None,
        variables=variables,
        id_to_name={int(k): v for k, v in (payload.get("id_to_name") or {}).items()},
        name_to_id=dict(payload.get("name_to_id") or {}),
        field_order=list(payload.get("field_order") or []),
        requires=[expr_from_dict(e) for e in payload.get("requires") or []],
        derives=[expr_from_dict(e) for e in payload.get("derives") or []],
        prefers=[expr_from_dict(e) for e in payload.get("prefers") or []],
        zero_dof_rules=[_rule_from_dict(r) for r in payload.get("zero_dof_rules") or []],
        soft_preferences=[_preference_from_dict(p) for p in payload.get("soft_preferences") or []],
        containers=containers,
        config=_decode_value(payload.get("config") or {}),
        model_schema=payload.get("model_schema"),
        field_aliases=dict(payload.get("field_aliases") or {}),
        portable_model=bool(payload.get("portable_model", False)),
        compiler_diagnostics=list(payload.get("compiler_diagnostics") or []),
    )


# ---------------------------------------------------------------------------
# canonical hashing
# ---------------------------------------------------------------------------

def canonical_json(payload: Any) -> str:
    """Stable JSON encoding used for every digest in the project."""

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return {"__float__": "inf" if value == math.inf else "-inf" if value == -math.inf else "nan"}
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(v) for v in value]
        return value
    return json.dumps(finite(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spec_digest(spec: VirSpec) -> str:
    """Content-addressed digest of a spec's *semantics*.

    ``config`` is included because decode policy (e.g. ``zero_dof_collapse``) changes
    what a certificate means.  The Python model class name is intentionally **excluded**:
    a certificate must stay verifiable after the producing codebase is gone, and a VIR
    round-trip (which cannot recreate the class) must produce the same digest.
    """

    payload = spec_to_dict(spec)
    payload["model"] = None
    return sha256_hex(canonical_json(payload))


def digest_of_instance(values: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(_encode_value(dict(values))))


__all__ = [
    "SERDE_VERSION",
    "canonical_json",
    "digest_of_instance",
    "expr_from_dict",
    "expr_to_dict",
    "sha256_hex",
    "spec_digest",
    "spec_from_dict",
    "spec_to_dict",
]
