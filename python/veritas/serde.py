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
from typing import Any, Dict, List, Mapping, Optional

from .compiler import vir as vir_module
from .compiler.vir import (
    ContainerInfo,
    SoftPreference,
    VirExpr,
    VirSpec,
    VirType,
    ZeroDofRule,
)

SERDE_VERSION = 2


# ---------------------------------------------------------------------------
# typing helpers
# ---------------------------------------------------------------------------

def _type_to_dict(vtype: Optional[VirType]) -> Optional[Dict[str, Any]]:
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
    }


def _type_from_dict(payload: Optional[Mapping[str, Any]]) -> Optional[VirType]:
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
    )


def _encode_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return value
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
    # Last resort: keep a readable, stable representation instead of failing serialization.
    return {"__repr__": repr(value)}


def _decode_value(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_decode_value(v) for v in payload]
    if isinstance(payload, dict):
        if "__tuple__" in payload:
            return tuple(_decode_value(v) for v in payload["__tuple__"])
        if "__set__" in payload:
            return set(_decode_value(v) for v in payload["__set__"])
        if "__pairs__" in payload:
            return {_decode_value(k): _decode_value(v) for k, v in payload["__pairs__"]}
        if "__repr__" in payload:
            return payload["__repr__"]
        # plain dict (older payloads)
        return {k: _decode_value(v) for k, v in payload.items()}
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


def expr_to_dict(expr: Optional[VirExpr]) -> Optional[Dict[str, Any]]:
    if expr is None:
        return None
    payload: Dict[str, Any] = {field: getattr(expr, field, None) for field in _EXPR_FIELDS}
    payload["value"] = _encode_value(expr.value)
    payload["args"] = [expr_to_dict(a) for a in expr.args]
    payload["lhs"] = expr_to_dict(expr.lhs)
    payload["rhs"] = expr_to_dict(expr.rhs)
    payload["iterable"] = expr_to_dict(expr.iterable) if hasattr(expr.iterable, "kind") else _encode_value(expr.iterable)
    payload["predicate"] = expr_to_dict(expr.predicate)
    return payload


def expr_from_dict(payload: Optional[Mapping[str, Any]]) -> Optional[VirExpr]:
    if payload is None:
        return None
    kwargs: Dict[str, Any] = {field: payload.get(field) for field in _EXPR_FIELDS}
    kwargs["tol"] = payload.get("tol", 1e-6) or 1e-6
    kwargs["weight"] = payload.get("weight", 1.0) if payload.get("weight") is not None else 1.0
    kwargs["tag"] = payload.get("tag", "") or ""
    kwargs["value"] = _decode_value(payload.get("value"))
    kwargs["args"] = [expr_from_dict(a) for a in payload.get("args") or [] if a is not None]
    kwargs["lhs"] = expr_from_dict(payload.get("lhs"))
    kwargs["rhs"] = expr_from_dict(payload.get("rhs"))
    iterable = payload.get("iterable")
    if isinstance(iterable, dict) and "kind" in iterable:
        kwargs["iterable"] = expr_from_dict(iterable)
    else:
        kwargs["iterable"] = _decode_value(iterable)
    kwargs["predicate"] = expr_from_dict(payload.get("predicate"))
    return VirExpr(**kwargs)


def _rule_to_dict(rule: ZeroDofRule) -> Dict[str, Any]:
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


def _preference_to_dict(pref: SoftPreference) -> Dict[str, Any]:
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


def _container_to_dict(info: ContainerInfo) -> Dict[str, Any]:
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
    )


# ---------------------------------------------------------------------------
# spec <-> dict
# ---------------------------------------------------------------------------

def spec_to_dict(spec: VirSpec, *, include_model_name: bool = True) -> Dict[str, Any]:
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
    }


def spec_from_dict(payload: Mapping[str, Any]) -> VirSpec:
    def _int_key_dict(raw: Mapping[Any, Any]) -> Dict[int, Any]:
        out: Dict[int, Any] = {}
        for key, value in (raw or {}).items():
            try:
                out[int(key)] = value
            except (TypeError, ValueError):
                continue
        return out

    variables: Dict[int, Dict[str, Any]] = {}
    for vid, info in _int_key_dict(payload.get("variables") or {}).items():
        variables[vid] = {
            "name": info.get("name"),
            "type": _type_from_dict(info.get("type")),
        }
    containers: Dict[int, ContainerInfo] = {}
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
    )


# ---------------------------------------------------------------------------
# canonical hashing
# ---------------------------------------------------------------------------

def canonical_json(payload: Any) -> str:
    """Stable JSON encoding used for every digest in the project."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True, default=str)


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
