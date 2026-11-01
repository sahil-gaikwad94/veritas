"""0-DoF deterministic evaluation, cascaded scheduling and Jump-Forward splicing.

A schema field is *0-DoF* when every input of its derivation is already bound: the model
has nothing left to decide, so ``veritas`` computes the value exactly, formats it, and
splices the whole syntactic fragment into the token stream in one batched prefill.

This module owns the evaluator/scheduler.  It deliberately delegates expression semantics
to :mod:`veritas.evaluate` so there is exactly one definition of what a constraint means
across verification, decoding and proof checking.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .evaluate import MISSING, eval_expr
from .json_pda import render_continuation


@dataclass
class ZeroDofRule:
    target_var: Any
    target_json_key: str | None = None
    input_vars: list[Any] = field(default_factory=list)
    rhs_eval: Callable[[dict[Any, Any]], Any] | None = None
    rhs_expr: Any = None
    precision_digits: int = 6


def _lookup(values: Mapping[Any, Any], key: Any, aliases: Mapping[Any, Any]) -> tuple[bool, Any]:
    if key in values:
        return True, values[key]
    alias = aliases.get(key)
    if alias is not None and alias in values:
        return True, values[alias]
    return False, None


class ZeroDofEngine:
    def __init__(self, rules: list[ZeroDofRule] | None = None):
        self.rules: list[ZeroDofRule] = list(rules or [])
        self.cache: dict[Any, Any] = {}
        # Optional id/name aliasing used when rules come from a compiled spec.
        self._aliases: dict[Any, Any] = {}
        self.last_splice: str = ""
        self.jump_forward_fields: list[str] = []

    # -- construction ----------------------------------------------------
    @classmethod
    def from_spec(cls, spec: Any) -> ZeroDofEngine:
        engine = cls()
        aliases: dict[Any, Any] = {}
        for rule in getattr(spec, "zero_dof_rules", []) or []:
            target = rule.target_json_key
            if target is None and isinstance(rule.target_var, int):
                target = spec.id_to_name.get(rule.target_var)
            inputs: list[Any] = []
            for item in rule.input_vars:
                if isinstance(item, int):
                    name = spec.id_to_name.get(item)
                    if name is not None:
                        inputs.append(name)
                        aliases[item] = name
                        aliases[name] = item
                    else:
                        inputs.append(item)
                else:
                    inputs.append(item)
                    vid = spec.name_to_id.get(str(item))
                    if vid is not None:
                        aliases[item] = vid
                        aliases[vid] = item
            engine.add_rule(
                ZeroDofRule(
                    target_var=rule.target_var,
                    target_json_key=target,
                    input_vars=inputs,
                    rhs_eval=rule.rhs_eval,
                    rhs_expr=rule.rhs_expr,
                    precision_digits=getattr(rule, "precision_digits", 6),
                )
            )
        engine._aliases = aliases
        return engine

    def add_rule(self, rule: ZeroDofRule) -> None:
        self.rules.append(rule)

    # -- readiness -------------------------------------------------------
    def _rule_for(self, target_var: Any) -> ZeroDofRule | None:
        for rule in self.rules:
            if rule.target_var == target_var or rule.target_json_key == target_var:
                return rule
        return None

    def is_ready(self, target_var: Any, values: Mapping[Any, Any]) -> bool:
        rule = self._rule_for(target_var)
        if rule is None:
            return False
        return all(_lookup(values, dep, self._aliases)[0] for dep in rule.input_vars)

    def pending(self, values: Mapping[Any, Any]) -> list[ZeroDofRule]:
        out: list[ZeroDofRule] = []
        for rule in self.rules:
            target = rule.target_json_key if rule.target_json_key is not None else rule.target_var
            if _lookup(values, target, self._aliases)[0]:
                continue
            if all(_lookup(values, dep, self._aliases)[0] for dep in rule.input_vars):
                out.append(rule)
        return out

    # -- evaluation ------------------------------------------------------
    def evaluate(self, target_var: Any, values: Mapping[Any, Any]) -> Any:
        rule = self._rule_for(target_var)
        if rule is None:
            return None
        if not all(_lookup(values, dep, self._aliases)[0] for dep in rule.input_vars):
            return None
        if callable(rule.rhs_eval):
            try:
                value = rule.rhs_eval(dict(values))
            except Exception:
                return None
        else:
            expr = rule.rhs_expr
            if expr is None:
                return None
            if getattr(expr, "kind", None) == "compare":
                expr = expr.rhs if expr.rhs is not None else expr.lhs
            value = eval_expr(expr, values)
        if value is None or value is MISSING:
            return None
        precision = max(0, int(getattr(rule, "precision_digits", 6) or 0))
        if isinstance(value, float) and precision:
            value = round(value, precision)
        self.cache[rule.target_json_key or rule.target_var] = value
        return value

    def evaluate_cascade(self, values: Mapping[Any, Any]) -> dict[Any, Any]:
        result = dict(values)
        changed = True
        passes = 0
        while changed and passes < 100:
            changed = False
            passes += 1
            for rule in self.rules:
                target = rule.target_json_key if rule.target_json_key is not None else rule.target_var
                present, _ = _lookup(result, target, self._aliases)
                if present:
                    continue
                value = self.evaluate(rule.target_var, result)
                if value is None:
                    continue
                result[target] = value
                if rule.target_var is not None:
                    result[rule.target_var] = value
                changed = True
        return result

    def compute_ready(self, values: Mapping[Any, Any]) -> dict[str, Any]:
        """Evaluate every currently-ready rule; return only newly computed fields."""

        computed: dict[str, Any] = {}
        result = dict(values)
        for _ in range(100):
            progress = False
            for rule in self.pending(result):
                target = rule.target_json_key if rule.target_json_key is not None else rule.target_var
                value = self.evaluate(rule.target_var, result)
                if value is None:
                    continue
                computed[str(target)] = value
                result[target] = value
                if rule.target_var is not None:
                    result[rule.target_var] = value
                progress = True
            if not progress:
                break
        self.jump_forward_fields = list(computed.keys())
        return computed

    # -- splicing --------------------------------------------------------
    def format_value(self, value: Any, precision_digits: int = 6) -> str:
        from .json_pda import json_number

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return json_number(value, precision_digits)
        return str(value)

    def splice_json(
        self,
        values: Mapping[Any, Any],
        keys: Sequence[str] | None = None,
        precision: Mapping[str, int] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Return the Jump-Forward continuation text plus the fields it computed."""

        computed = self.compute_ready(values)
        if not computed:
            self.last_splice = ""
            return "", {}
        precision_map = dict(precision or {})
        for rule in self.rules:
            name = rule.target_json_key
            if name and name not in precision_map:
                precision_map[name] = max(0, int(rule.precision_digits or 0))
        order = [k for k in (keys or []) if k in computed] or list(computed.keys())
        text = render_continuation(computed, order, precision_map)
        self.last_splice = text
        return text, computed


# aliases
ZeroDof = ZeroDofEngine
DeterministicCollapse = ZeroDofEngine
ZeroDofScheduler = ZeroDofEngine
CascadedEvaluator = ZeroDofEngine


def _eval_expr(expr: Any, values: Mapping[Any, Any]) -> Any:
    """Backwards-compatible free function delegating to the canonical evaluator."""

    return eval_expr(expr, values)


__all__ = [
    "CascadedEvaluator",
    "DeterministicCollapse",
    "ZeroDof",
    "ZeroDofEngine",
    "ZeroDofRule",
    "ZeroDofScheduler",
    "_eval_expr",
]
