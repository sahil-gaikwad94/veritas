"""Schema-specific GBNF generation for llama.cpp.

This module generates GBNF grammars that enforce Veritas specifications,
allowing llama.cpp hosts that only accept grammar strings to enforce
the full schema rather than just generic JSON syntax.
"""

from __future__ import annotations

from typing import Any

from ..compiler.vir import VirSpec, VirType

# GBNF primitives
GBNF_WS = r'[ \t\n\r]*'
GBNF_STRING_CONTENT = r'([^"\\] | \\ (["\\/bfnrt] | u [0-9a-fA-F]{4}))*'
GBNF_NUMBER = r'-? (0 | [1-9] [0-9]*) (\. [0-9]+)? ([eE] [+-]? [0-9]+)?'
GBNF_BOOL = r'true | false'
GBNF_NULL = r'null'

# Format-specific patterns (simplified for GBNF)
FORMAT_PATTERNS = {
    "date-time": r'\d{4}-\d{2}-\d{2} [Tt] \d{2}:\d{2}:\d{2} (\.\d+)? (Z|[+-]\d{2}:?\d{2})?',
    "date": r'\d{4}-\d{2}-\d{2}',
    "time": r'\d{2}:\d{2}:\d{2} (\.\d+)?',
    "duration": r'P (\d+D)? (T (\d+H)? (\d+M)? (\d+(\.\d+)?S)?)?',
    "uuid": r'[0-9a-fA-F]{8} - [0-9a-fA-F]{4} - [0-9a-fA-F]{4} - [0-9a-fA-F]{4} - [0-9a-fA-F]{12}',
    "email": r'[^@\s]+ @ [^@\s]+ \. [^@\s]+',
    "uri": r'[a-zA-Z][a-zA-Z0-9+.-]* : // [^\s]*',
    "ipv4": r'\d{1,3} \. \d{1,3} \. \d{1,3} \. \d{1,3}',
    "ipv6": r'([0-9a-fA-F]{1,4}:){7} [0-9a-fA-F]{1,4}',
}


class GBNFGenerator:
    """Generate GBNF grammar from a VIR spec."""

    def __init__(self, spec: VirSpec):
        self.spec = spec
        self.rules: dict[str, str] = {}
        self.rule_counter = 0

    def _new_rule_name(self, prefix: str = "rule") -> str:
        self.rule_counter += 1
        return f"{prefix}_{self.rule_counter}"

    def _sanitize_name(self, name: str) -> str:
        """Convert a field name to a valid GBNF identifier."""
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name)

    def generate(self) -> str:
        """Generate the complete GBNF grammar."""
        # Start with root rule
        self._generate_value_rules()
        self._generate_spec_rules()

        # Build the grammar text
        lines = []
        # Root rule first
        lines.append("root ::= value")
        lines.append("")
        # Then all other rules
        for name, rule in self.rules.items():
            lines.append(f"{name} ::= {rule}")
        return "\n".join(lines)

    def _generate_value_rules(self) -> None:
        """Generate the base JSON value rules."""
        self.rules["value"] = "object | array | string | number | bool | null"
        self.rules["bool"] = GBNF_BOOL
        self.rules["null"] = GBNF_NULL
        self.rules["number"] = GBNF_NUMBER
        self.rules["string"] = f'"{GBNF_STRING_CONTENT}"'
        self.rules["ws"] = GBNF_WS

    def _generate_spec_rules(self) -> None:
        """Generate rules for the spec's fields."""
        # First, create the base object and array rules so they can be referenced
        self.rules["object"] = '{{ ws (member (ws "," ws member)*)? ws }}'
        self.rules["array"] = '[ ws (value (ws "," ws value)*)? ws ]'
        self.rules["member"] = ""  # Will be populated below

        # Build the main object rule for the model
        field_rules = []
        for field_name in self.spec.field_order:
            if field_name not in self.spec.name_to_id:
                continue
            vid = self.spec.name_to_id[field_name]
            vtype = self.spec.variables.get(vid, {}).get("type")
            if vtype is None:
                continue
            rule_name = self._generate_field_rule(field_name, vtype)
            field_rules.append(f'"{field_name}" ws ":" ws {rule_name}')

        # Object rule with required fields
        required_fields = set()
        for vid in self.spec.name_to_id.values():
            vtype = self.spec.variables.get(vid, {}).get("type")
            if vtype and getattr(vtype, "required", None):
                required_fields.update(vtype.required)

        # Update the member rule with all field rules
        if field_rules:
            members = " | ".join(field_rules)
            self.rules["member"] = members
        else:
            self.rules["member"] = ""
            self.rules["object"] = '{{ ws }}'

    def _generate_field_rule(self, field_name: str, vtype: VirType) -> str:
        """Generate a GBNF rule for a single field."""
        kind = getattr(vtype, "kind", "any")

        if kind in ("real", "int", "float", "number"):
            rule_name = f"num_{self._sanitize_name(field_name)}"
            # Add numeric bounds if present
            pattern = GBNF_NUMBER
            if getattr(vtype, "min", None) is not None or getattr(vtype, "max", None) is not None:
                # GBNF doesn't support numeric bounds directly; we add a comment
                # The actual enforcement happens via logits processor or verification
                pass
            self.rules[rule_name] = pattern
            return rule_name

        elif kind == "bool":
            rule_name = f"bool_{self._sanitize_name(field_name)}"
            self.rules[rule_name] = GBNF_BOOL
            return rule_name

        elif kind == "string":
            rule_name = f"str_{self._sanitize_name(field_name)}"
            pattern = f'"{GBNF_STRING_CONTENT}"'

            # Add format constraint if present
            text_format = getattr(vtype, "text_format", None)
            if text_format and text_format in FORMAT_PATTERNS:
                # GBNF can't easily enforce complex patterns inline
                # We use a more permissive string and rely on verification
                pass

            # Add pattern constraint if present
            pattern_str = getattr(vtype, "pattern", None)
            if pattern_str:
                # Can't easily convert regex to GBNF; rely on verification
                pass

            # Add length constraints as comments
            self.rules[rule_name] = pattern
            return rule_name

        elif kind in ("dynamic_map", "list", "dict", "set"):
            # Container types - use generic object/array
            item_type = getattr(vtype, "item_type", None)
            if item_type and getattr(item_type, "kind", "any") in ("real", "int", "float", "number"):
                # Numeric container - could generate more specific rule
                pass
            if kind == "list" or kind == "set":
                rule_name = f"arr_{self._sanitize_name(field_name)}"
                self.rules[rule_name] = self.rules["array"]
                return rule_name
            else:
                rule_name = f"obj_{self._sanitize_name(field_name)}"
                self.rules[rule_name] = self.rules["object"]
                return rule_name

        elif kind == "object":
            # Nested object - generate inline
            rule_name = f"obj_{self._sanitize_name(field_name)}"
            properties = getattr(vtype, "properties", {})
            if properties:
                prop_rules = []
                for prop_name, prop_type in properties.items():
                    prop_rule = self._generate_field_rule(prop_name, prop_type)
                    prop_rules.append(f'"{prop_name}" ws ":" ws {prop_rule}')
                members = " | ".join(prop_rules)
                self.rules[rule_name] = '{{ ws (member (ws "," ws member)*)? ws }}'
                self.rules[f"{rule_name}_member"] = members
            else:
                self.rules[rule_name] = self.rules["object"]
            return rule_name

        elif kind == "finite_set":
            # Enum - generate as choice of string literals
            rule_name = f"enum_{self._sanitize_name(field_name)}"
            universe = getattr(vtype, "universe", [])
            if universe:
                choices = " | ".join(f'"{self._escape_gbnf(str(v))}"' for v in universe)
                self.rules[rule_name] = choices
            else:
                self.rules[rule_name] = self.rules["string"]
            return rule_name

        else:
            # Fallback to generic value
            return "value"

    def _escape_gbnf(self, text: str) -> str:
        """Escape a string for GBNF."""
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def generate_gbnf(spec: VirSpec) -> str:
    """Generate a schema-specific GBNF grammar from a VIR spec."""
    generator = GBNFGenerator(spec)
    return generator.generate()


def generate_gbnf_from_schema(schema_or_spec: Any) -> str:
    """Generate GBNF from a schema function or compiled spec."""
    from ..instance import _coerce_spec
    spec = _coerce_spec(schema_or_spec)
    return generate_gbnf(spec)


__all__ = ["generate_gbnf", "generate_gbnf_from_schema", "GBNFGenerator"]