"""Lower Pydantic v2 models into a symbolic variable table and JSON schema ordering."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .vir import VirType
from .dependency_graph import DependencyGraph


def _python_type_to_vir_type(annotation: Any) -> VirType:
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", None)

    # unwrap Annotated
    try:
        import typing
        if origin is not None and getattr(origin, "__name__", "") == "Annotated" and args:
            annotation = args[0]
            origin = getattr(annotation, "__origin__", None)
            args = getattr(annotation, "__args__", None)
    except Exception:
        pass

    # Optional[T] / Union[T, None] -> treat as T (nullability is a JSON concern, not a
    # numeric-domain one).
    try:
        import types
        import typing
        origin_now = typing.get_origin(annotation)
        if origin_now in (typing.Union, types.UnionType):
            candidates = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(candidates) == 1:
                return _python_type_to_vir_type(candidates[0])
    except Exception:
        pass

    if annotation is float:
        return VirType.real()
    if annotation is int:
        return VirType.int()
    if annotation is bool:
        return VirType.bool()
    if annotation is str:
        return VirType(kind="string")
    # Enum -> finite set
    try:
        import enum
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            vals = [e.value for e in annotation]
            return VirType(kind="finite_set", universe=vals)
    except Exception:
        pass
    # Literal -> finite set
    try:
        import typing
        if origin is not None and getattr(origin, "__name__", "") == "Literal":
            return VirType(kind="finite_set", universe=list(args) if args else [])
    except Exception:
        pass
    if annotation is dict or origin is dict:
        # Bare `dict` / `dict[str, float]` with no declared element bounds is left
        # unconstrained: inventing [0, 1] here would make unrelated schemas look
        # infeasible.  `vt.forall(...)` supplies real element bounds during analysis.
        return VirType.dynamic_map(val_min=float("-inf"), val_max=float("inf"))
    if annotation is list or origin is list:
        return VirType.dynamic_map(val_min=float("-inf"), val_max=float("inf"))
    # Optional / Union fallback
    if origin is not None:
        return VirType(kind="complex")
    return VirType(kind=str(annotation))


def lower_pydantic_model(model: Any) -> Dict[str, Any]:
    """Lower a Pydantic v2 model into a variable table + dependency graph."""
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError("lower_pydantic_model expects a Pydantic BaseModel subclass")

    graph = DependencyGraph()
    fields: List[Dict[str, Any]] = []
    var_table: Dict[int, Dict[str, Any]] = {}

    for name, field in model.model_fields.items():
        vtype = _python_type_to_vir_type(field.annotation)
        # apply pydantic Field constraints if present
        for m in getattr(field, "metadata", []) or []:
            if hasattr(m, "ge") and m.ge is not None:
                if vtype.min is None or m.ge > vtype.min:
                    vtype.min = float(m.ge) if not isinstance(m.ge, int) or vtype.kind != "int" else m.ge
            if hasattr(m, "gt") and m.gt is not None:
                val = m.gt
                # represent as min exclusive by nudging? keep as min
                if vtype.min is None or val >= (vtype.min or val):
                    vtype.min = val
            if hasattr(m, "le") and m.le is not None:
                if vtype.max is None or m.le < vtype.max:
                    vtype.max = m.le
            if hasattr(m, "lt") and m.lt is not None:
                if vtype.max is None or m.lt <= (vtype.max or m.lt):
                    vtype.max = m.lt
            if hasattr(m, "min_length") and m.min_length is not None:
                vtype.min_len = int(m.min_length)
            if hasattr(m, "max_length") and m.max_length is not None:
                vtype.max_len = int(m.max_length)
        vid = graph.add_variable(name, vtype)
        fields.append({"name": name, "type": vtype, "var_id": vid, "field": field})
        var_table[vid] = {"name": name, "type": vtype, "field": field}

    return {
        "model": model,
        "fields": fields,
        "var_table": var_table,
        "graph": graph,
    }


def build_json_key_order(graph: DependencyGraph) -> List[str]:
    """Topological order of JSON keys (free vars before derived vars)."""
    return graph.topological_order()


def attach_derives_to_graph(graph: DependencyGraph, derives: List[Any]) -> None:
    """Helper: attaches parsed derive rules (target + inputs) to graph."""
    for d in derives:
        target = getattr(d, "target_name", None) or getattr(d, "name", None)
        inputs = getattr(d, "input_names", None) or []
        if target is not None:
            graph.add_derive(target, list(inputs), expr=d)


def model_field_names(model: Any) -> List[str]:
    return list(model.model_fields.keys())
