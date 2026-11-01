"""Lower Pydantic v2 field types without discarding validation semantics."""
from __future__ import annotations

import enum
import types
import typing
from typing import Any

from pydantic import BaseModel

from .dependency_graph import DependencyGraph
from .vir import VirType


def _python_type_to_vir_type(annotation: Any, seen=frozenset()) -> VirType:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Annotated:
        result = _python_type_to_vir_type(args[0], seen)
        for meta in args[1:]:
            _apply_metadata(result, meta)
        return result
    if origin in (typing.Union, types.UnionType):
        candidates = [a for a in args if a is not type(None)]
        result = _python_type_to_vir_type(candidates[0], seen) if len(candidates) == 1 else VirType(kind="complex")
        result.nullable = len(candidates) != len(args)
        return result
    if annotation is float:
        return VirType.real()
    if annotation is int:
        return VirType.int()
    if annotation is bool:
        return VirType.bool()
    if annotation is str:
        return VirType(kind="string")
    if annotation is type(None):
        return VirType(kind="null", nullable=True)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return VirType(kind="finite_set", universe=[e.value for e in annotation])
    if origin is typing.Literal:
        return VirType(kind="finite_set", universe=list(args))
    if annotation is dict or origin is dict:
        item = _python_type_to_vir_type(args[1], seen) if len(args) == 2 else VirType()
        return VirType(kind="dynamic_map", item_type=item)
    if annotation is list or origin is list:
        item = _python_type_to_vir_type(args[0], seen) if args else VirType()
        return VirType(kind="dynamic_map", container_kind="list", item_type=item)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in seen:
            raise TypeError(f"recursive model {annotation.__name__!r} is not supported by the decoder")
        nested = lower_pydantic_model(annotation, seen | {annotation})
        return VirType(kind="object", properties={f["name"]: f["type"] for f in nested["fields"]},
                       required=[f["name"] for f in nested["fields"] if f["field"].is_required()],
                       additional_properties=annotation.model_config.get("extra") != "forbid",
                       model_name=annotation.__name__)
    return VirType(kind="complex")


def _apply_metadata(vtype: VirType, meta: Any) -> None:
    for attr, target, exclusive in (("ge", "min", False), ("gt", "min", True),
                                    ("le", "max", False), ("lt", "max", True)):
        value = getattr(meta, attr, None)
        if value is None:
            continue
        value = float(value)
        current = getattr(vtype, target)
        if current is None or (value > current if target == "min" else value < current):
            setattr(vtype, target, value)
            setattr(vtype, target + "_exclusive", exclusive)
        elif value == current and exclusive:
            setattr(vtype, target + "_exclusive", True)
    if getattr(meta, "min_length", None) is not None:
        vtype.min_len = int(meta.min_length)
    if getattr(meta, "max_length", None) is not None:
        vtype.max_len = int(meta.max_length)


def lower_pydantic_model(model: Any, seen=frozenset()) -> dict[str, Any]:
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError("lower_pydantic_model expects a Pydantic BaseModel subclass")
    graph = DependencyGraph()
    fields, var_table, aliases = [], {}, {}
    for name, field in model.model_fields.items():
        vtype = _python_type_to_vir_type(field.annotation, seen | {model})
        for meta in field.metadata or []:
            _apply_metadata(vtype, meta)
        if vtype.kind == "dynamic_map" and vtype.item_type and vtype.item_type.kind in ("real", "int"):
            vtype.val_min = vtype.item_type.min if vtype.item_type.min is not None else float("-inf")
            vtype.val_max = vtype.item_type.max if vtype.item_type.max is not None else float("inf")
        vid = graph.add_variable(name, vtype)
        fields.append({"name": name, "type": vtype, "var_id": vid, "field": field})
        var_table[vid] = {"name": name, "type": vtype, "field": field}
        alias = field.validation_alias if isinstance(field.validation_alias, str) else (field.alias or name)
        aliases[name] = alias
    decorators = getattr(model, "__pydantic_decorators__", None)
    portable = True
    if decorators is not None:
        portable = not any(getattr(decorators, key, {}) for key in
                           ("field_validators", "model_validators", "field_serializers", "model_serializers"))
    return {"model": model, "fields": fields, "var_table": var_table, "graph": graph,
            "model_schema": model.model_json_schema(), "field_aliases": aliases,
            "portable_model": portable}


def build_json_key_order(graph):
    return graph.topological_order()


def model_field_names(model):
    return list(model.model_fields.keys())