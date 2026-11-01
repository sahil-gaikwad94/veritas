"""Incremental JSON Schema supervision for a documented, non-recursive subset.

The JSON PDA owns syntax. This layer tracks *independent* schema candidates at each
container, so ``anyOf``/``oneOf`` branches never share mutable partial data. Completed
containers are validated against the original schema (including ``oneOf`` cardinality),
not just the branch used for navigation. Local ``$ref`` is resolved against the root.
Unsupported constructs fail loudly instead of silently weakening a mask.
"""
from __future__ import annotations

import copy
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .json_pda import JSONPushdownAutomaton, JSONSyntaxError, parse_json

_UNSUPPORTED = {
    "allOf", "not", "if", "then", "else", "dependentSchemas", "dependentRequired",
    "patternProperties", "prefixItems", "unevaluatedProperties", "unevaluatedItems",
    "$dynamicRef", "$recursiveRef", "contains",
}
_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_CHILD_KEYS = ("properties", "$defs", "definitions")
_CHILD_NODE_KEYS = ("items", "additionalProperties", "propertyNames", "additionalItems")


def _pointer(root, ref):
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise JSONSyntaxError(f"only local JSON-pointer $ref is supported: {ref!r}")
    target = root
    try:
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
    except (KeyError, TypeError, IndexError) as exc:
        raise JSONSyntaxError(f"unresolved schema reference {ref!r}") from exc
    return target


def _check_supported(node, root, active=frozenset()):
    if isinstance(node, bool):
        return
    if not isinstance(node, dict):
        raise JSONSyntaxError("schema nodes must be objects or booleans")
    bad = _UNSUPPORTED & node.keys()
    if bad:
        raise JSONSyntaxError(f"unsupported schema keywords for incremental masking: {sorted(bad)}")
    if "$ref" in node:
        ref = node["$ref"]
        if ref in active:
            raise JSONSyntaxError(f"recursive schema reference {ref!r} is not supported")
        _check_supported(_pointer(root, ref), root, active | {ref})
    for key, value in node.items():
        if key in _CHILD_KEYS and isinstance(value, dict):
            for child in value.values():
                _check_supported(child, root, active)
        elif key in _CHILD_NODE_KEYS:
            _check_supported(value, root, active)
        elif key in ("anyOf", "oneOf") and isinstance(value, list):
            for child in value:
                _check_supported(child, root, active)


class SchemaJSONAutomaton:
    def __init__(self, schema, max_depth=64):
        Draft202012Validator.check_schema(schema)
        _check_supported(schema, schema)
        self.schema = schema
        self._validator = Draft202012Validator(schema, format_checker=FormatChecker())
        self.machine = JSONPushdownAutomaton(max_depth)
        self.frames = []
        self.root_candidates = [{"schema": schema, "branch": alt, "data": None}
                                for alt in self._alternatives(schema)]
        self.error = None
        self._root_done = False

    def clone(self):
        other = object.__new__(SchemaJSONAutomaton)
        other.schema = self.schema
        other._validator = self._validator
        other.machine = self.machine.clone()
        other.frames = copy.deepcopy(self.frames)
        other.root_candidates = copy.deepcopy(self.root_candidates)
        other.error = self.error
        other._root_done = self._root_done
        return other

    @property
    def state(self):
        return self.machine.state

    def is_error(self):
        return self.error is not None or self.machine.is_error()

    def accepts(self):
        return self.machine.accepts() and not self.is_error() and self._root_done

    def feed(self, text):
        return all(self.step(ch) for ch in text)

    def finish(self):
        before = len(self.machine.events)
        if not self.machine.finish():
            self.error = self.error or self.machine.error
            return False
        try:
            for event in self.machine.events[before:]:
                if not self._on_event(event):
                    return False
            return self.accepts()
        except (JSONSyntaxError, ValueError, TypeError, KeyError) as exc:
            return self._fail(str(exc))

    def allowed_chars(self, alphabet=None):
        alphabet = alphabet if alphabet is not None else [chr(c) for c in range(0x20, 0x7F)] + ["\n", "\t", "\r"]
        return {ch for ch in alphabet if self.clone().step(ch)}

    def _resolve(self, schema, seen=frozenset()):
        if isinstance(schema, bool) or "$ref" not in schema:
            return schema
        ref = schema["$ref"]
        if ref in seen:
            raise JSONSyntaxError(f"recursive schema reference {ref!r}")
        resolved = self._resolve(_pointer(self.schema, ref), seen | {ref})
        if isinstance(resolved, dict):
            merged = dict(resolved)
            merged.update({k: v for k, v in schema.items() if k != "$ref"})
            return merged
        return resolved

    def _alternatives(self, schema):
        schema = self._resolve(schema)
        if isinstance(schema, bool):
            return [schema]
        options = schema.get("anyOf", schema.get("oneOf"))
        if options is None:
            return [schema]
        base = {k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")}
        out = []
        for option in options:
            resolved = self._resolve(option)
            if isinstance(resolved, dict):
                merged = dict(base)
                merged.update(resolved)
                resolved = merged
            out.append(resolved)
        return out

    @staticmethod
    def _types(schema):
        if isinstance(schema, bool):
            return set() if schema is False else _TYPES
        kind = schema.get("type")
        if kind:
            return set(kind if isinstance(kind, list) else [kind])
        if "enum" in schema:
            return {("null" if v is None else "boolean" if isinstance(v, bool) else
                     "string" if isinstance(v, str) else "array" if isinstance(v, list) else
                     "object" if isinstance(v, dict) else "number") for v in schema["enum"]}
        return _TYPES

    def _valid(self, schema, value):
        return self._validator.evolve(schema=schema).is_valid(value)

    def _fail(self, message):
        self.error = message
        self.machine.state = "error"
        self.machine.error = message
        return False

    def step(self, ch):
        if self.is_error() or not self.machine.step(ch):
            self.error = self.error or self.machine.error
            return False
        try:
            for event in self.machine.events:
                if not self._on_event(event):
                    return False
            return True
        except (JSONSyntaxError, ValueError, TypeError, KeyError) as exc:
            return self._fail(str(exc))

    def _pending_schema(self, candidate):
        if not self.frames:
            return candidate["schema"]
        frame = self.frames[-1]
        if frame["container"] == "arr":
            branch = candidate["branch"]
            return branch.get("items", True) if isinstance(branch, dict) else True
        return candidate.get("pending", True)

    def _on_event(self, event):
        kind = event["kind"]
        if kind == "key":
            if not self.frames or self.frames[-1]["container"] != "obj":
                return self._fail("object key outside object")
            if event["duplicate"]:
                return self._fail(f"duplicate object key {event['key']!r}")
            key = event["key"]
            kept = []
            for candidate in self.frames[-1]["candidates"]:
                branch = candidate["branch"]
                if isinstance(branch, bool):
                    child = branch
                else:
                    props = branch.get("properties", {})
                    if key in props:
                        child = props[key]
                    else:
                        extra = branch.get("additionalProperties", True)
                        if extra is False:
                            continue
                        child = extra
                    names = branch.get("propertyNames")
                    if names is not None and not self._valid(names, key):
                        continue
                    maximum = branch.get("maxProperties")
                    if isinstance(maximum, int) and len(candidate["data"]) >= maximum:
                        continue
                candidate["pending"] = child
                candidate["pending_key"] = key
                kept.append(candidate)
            if not kept:
                return self._fail(f"property {key!r} is not allowed by the schema")
            self.frames[-1]["candidates"] = kept
            return True

        if kind == "container_start":
            actual = "object" if event["container"] == "obj" else "array"
            parents = self.frames[-1]["candidates"] if self.frames else self.root_candidates
            candidates = []
            for parent in parents:
                pending = self._pending_schema(parent)
                for branch in self._alternatives(pending):
                    if actual in self._types(branch):
                        candidates.append({"schema": pending, "branch": branch,
                                           "data": {} if actual == "object" else []})
            if not candidates:
                return self._fail(f"{actual} is not allowed at this position")
            self.frames.append({"container": event["container"], "candidates": candidates})
            return True

        if kind == "value_start":
            if self._root_done and not self.frames:
                return self._fail("multiple root values")
            ch = event["char"]
            actual = {"{": "object", "[": "array", '"': "string", "t": "boolean", "f": "boolean", "n": "null"}.get(ch, "number")
            parents = self.frames[-1]["candidates"] if self.frames else self.root_candidates
            allowed = set()
            for parent in parents:
                types = set().union(*(self._types(o) for o in self._alternatives(self._pending_schema(parent))))
                if "integer" in types:
                    types.add("number")
                allowed |= types
            if actual not in allowed:
                return self._fail(f"{actual} is not allowed at this position")
            return True

        if kind in ("value", "container_end"):
            value = parse_json(event["text"])
            if kind == "container_end":
                frame = self.frames.pop()
                if not any(self._valid(c["schema"], c["data"]) for c in frame["candidates"]):
                    return self._fail("container does not satisfy required/arity/schema constraints")
            self._accept_value(value)
            return True
        return True

    def _accept_value(self, value):
        if self.frames:
            frame = self.frames[-1]
            kept = []
            for candidate in frame["candidates"]:
                schema = self._pending_schema(candidate)
                if not any(self._valid(o, value) for o in self._alternatives(schema)):
                    continue
                if frame["container"] == "obj":
                    candidate["data"][candidate["pending_key"]] = value
                else:
                    branch = candidate["branch"]
                    maximum = branch.get("maxItems") if isinstance(branch, dict) else None
                    if isinstance(maximum, int) and len(candidate["data"]) >= maximum:
                        continue
                    candidate["data"].append(value)
                kept.append(candidate)
            if not kept:
                raise JSONSyntaxError("value does not satisfy the current schema branch")
            frame["candidates"] = kept
        else:
            candidates = [c for c in self.root_candidates if self._valid(c["schema"], value)]
            if not candidates:
                raise JSONSyntaxError("root value does not satisfy the schema")
            self.root_candidates = candidates
            self._root_done = True


def validate_schema(instance: Any, schema: dict) -> None:
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(instance)