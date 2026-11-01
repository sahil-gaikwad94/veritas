"""Schema synthesis from concrete examples (``veritas synth --from-json``).

The handbook's long-term vision is natural-language-to-spec synthesis.  Without a model
adapter that is not implementable honestly, so this module does the deterministic half:
infer a Pydantic model plus numeric/key/range invariants from example JSON objects, and
emit a ready-to-run ``@vt.verified_schema`` module.  It never invents constraints that the
examples do not support.
"""

from __future__ import annotations

import json
import keyword
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _load_samples(samples: Any) -> list[dict[str, Any]]:
    if samples is None:
        raise ValueError("synth requires at least one example")
    if isinstance(samples, (str, Path)):
        text = Path(samples).read_text(encoding="utf-8") if Path(str(samples)).exists() else str(samples)
        payload = json.loads(text)
    else:
        payload = samples
    if isinstance(payload, Mapping):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError("examples must be a JSON object or a non-empty list of objects")
    out: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("every example must be a JSON object")
        out.append(dict(item))
    return out


def _infer_type(values: Sequence[Any]) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "Optional[float]"
    if all(isinstance(v, bool) for v in present):
        return "bool"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in present):
        return "int"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return "float"
    if all(isinstance(v, str) for v in present):
        return "str"
    if all(isinstance(v, Mapping) for v in present):
        keys: list[str] = []
        nested: list[Any] = []
        for mapping in present:
            for key, value in mapping.items():
                if str(key) not in keys:
                    keys.append(str(key))
                nested.append(value)
        value_type = _infer_type(nested)
        if value_type.startswith("Optional["):
            value_type = value_type[len("Optional["):-1]
        return f"dict[str, {value_type}]"
    if all(isinstance(v, (list, tuple)) for v in present):
        nested = [item for seq in present for item in seq]
        element = _infer_type(nested) if nested else "float"
        return f"list[{element}]"
    return "Any"


def _field_annotation(name: str, values: Sequence[Any], kinds: Mapping[str, str]) -> str:
    annotation = _infer_type(values)
    if any(v is None for v in values) and not annotation.startswith("Optional["):
        annotation = f"Optional[{annotation}]"
    return annotation


def _default_literal(annotation: str, value: Any) -> str:
    if value is None:
        return "None"
    if annotation.startswith("Optional["):
        return "None"
    if annotation == "bool":
        return "True" if value else "False"
    if annotation == "int":
        return repr(int(value))
    if annotation == "float":
        return repr(float(value))
    if annotation == "str":
        return repr(str(value))
    if annotation.startswith("dict"):
        return "Field(default_factory=dict)"
    if annotation.startswith("list"):
        return "Field(default_factory=list)"
    return "None"


def _snake(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
    cleaned = re.sub(r"(?<!^)(?=[A-Z])", "_", cleaned).lower()
    if not cleaned:
        cleaned = "synthesized"
    if cleaned[0].isdigit():
        cleaned = "m_" + cleaned
    if keyword.iskeyword(cleaned):
        cleaned = "field_" + cleaned
    return cleaned


def _identifiers(fields: Sequence[str]) -> dict[str, str]:
    used: set[str] = set()
    out: dict[str, str] = {}
    for field in fields:
        base = _snake(field)
        ident = base
        counter = 2
        while ident in used:
            ident = f"{base}_{counter}"
            counter += 1
        used.add(ident)
        out[field] = ident
    return out


def _pascal(name: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p) or "Synthesized"


def synth_schema(
    samples: Any = None,
    *,
    name: str = "SynthesizedModel",
    from_json: Any = None,
    output: str | None = None,
    require_ranges: bool = True,
) -> str:
    """Return a Python module (source text) inferred from example objects."""

    examples = _load_samples(from_json if from_json is not None else samples)
    fields: list[str] = []
    for example in examples:
        for key in example:
            if str(key) not in fields:
                fields.append(str(key))

    columns: dict[str, list[Any]] = {field: [ex.get(field) for ex in examples] for field in fields}
    annotations = {field: _field_annotation(field, columns[field], {}) for field in fields}
    identifiers = _identifiers(fields)
    class_name = _pascal(name)
    function_name = _snake(name)
    if function_name in {"model", "schema", "base_model"}:
        function_name = function_name + "_spec"

    lines: list[str] = [
        '"""Synthesized by veritas from concrete examples. Review before production use."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any, Optional",
        "",
        "import veritas as vt",
        "from pydantic import BaseModel, Field",
        "",
        "",
        f"class {class_name}(BaseModel):",
    ]
    for field in fields:
        annotation = annotations[field]
        ident = identifiers[field]
        default = _default_literal(annotation, columns[field][0])
        if ident != field:
            if default.startswith("Field("):
                default = default[:-1] + f", alias={field!r})"
            else:
                default = f"Field({default}, alias={field!r})"
        lines.append(f"    {ident}: {annotation} = {default}")
    lines += ["", "", "@vt.verified_schema()", f"def {function_name}(p: {class_name}):"]

    body: list[str] = []
    for field in fields:
        values = [v for v in columns[field] if v is not None]
        annotation = annotations[field]
        if annotation in ("int", "float") and require_ranges and len(values) >= 2:
            lo, hi = min(values), max(values)
            if lo == hi:
                continue
            if annotation == "int":
                body.append(f"    vt.require(({lo} <= p.{identifiers[field]}) & (p.{identifiers[field]} <= {hi}))")
            else:
                padding = (hi - lo) * 0.1 or abs(hi) * 0.01 or 0.01
                body.append(
                    f"    vt.require(({lo - padding:.12g} <= p.{identifiers[field]}) & "
                    f"(p.{identifiers[field]} <= {hi + padding:.12g}))"
                )
        if annotation.startswith("dict"):
            universe: list[str] = []
            numeric: list[float] = []
            for mapping in (ex.get(field) for ex in examples):
                if isinstance(mapping, Mapping):
                    for key, value in mapping.items():
                        if str(key) not in universe:
                            universe.append(str(key))
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            numeric.append(float(value))
            if universe:
                body.append(f"    vt.require(vt.keys_subset(p.{identifiers[field]}, {sorted(set(universe))!r}))")
            if require_ranges and len(numeric) >= 2 and len(universe) == len(set(universe)):
                lo, hi = min(numeric), max(numeric)
                if lo != hi:
                    padding = (hi - lo) * 0.1 or 0.01
                    body.append(
                        f"    vt.require(vt.forall(p.{identifiers[field]}.values(), "
                        f"lambda v: ({lo - padding:.12g} <= v) & (v <= {hi + padding:.12g})))"
                    )
    if not body:
        body.append("    # No invariants were inferable from the examples.")
        body.append("    pass")
    lines += body
    source = "\n".join(lines) + "\n"
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return source


def synth_to_file(samples: Any, output: str, **kwargs: Any) -> str:
    return synth_schema(samples, output=output, **kwargs)


__all__ = ["synth_schema", "synth_to_file"]
