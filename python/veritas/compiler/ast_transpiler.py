"""Strict Python-AST to data-only VIR compiler.

Unsupported syntax is an error, never an ignored constraint. DSL identity is resolved
through captured Python objects, not through a function-name suffix.
"""
from __future__ import annotations

import ast
import builtins
import copy
import inspect
import math
import operator
import textwrap
import typing

from .analysis import analyze_containers
from .schema_lowering import lower_pydantic_model
from .vir import SoftPreference, VirExpr, VirSpec, ZeroDofRule


class TranspileError(ValueError):
    """A schema uses syntax outside the supported, checkable DSL."""


def _extract_simple_bounds(expr: VirExpr) -> list[tuple[int, str, float, bool]]:
    """Extract simple variable bounds from a require expression.
    
    Returns list of (var_id, op, value, exclusive) for constraints like:
    - var >= value (op='ge', exclusive=False)
    - var > value (op='gt', exclusive=True)
    - var <= value (op='le', exclusive=False)
    - var < value (op='lt', exclusive=True)
    - var == value (op='eq', exclusive=False)
    """
    bounds = []
    
    def visit(node: VirExpr):
        if node is None:
            return
        kind = getattr(node, "kind", None)
        if kind == "and":
            for arg in getattr(node, "args", []) or []:
                visit(arg)
        elif kind == "compare":
            lhs = getattr(node, "lhs", None)
            rhs = getattr(node, "rhs", None)
            op = getattr(node, "op", "eq")
            # Check if lhs is a var and rhs is a constant
            if lhs is not None and getattr(lhs, "kind", None) == "var":
                bound = _const_number(rhs)
                if bound is not None:
                    bounds.append((lhs.var_id, op, bound, op in ("gt", "lt")))
            # Check if rhs is a var and lhs is a constant
            elif rhs is not None and getattr(rhs, "kind", None) == "var":
                bound = _const_number(lhs)
                if bound is not None:
                    # Flip the operator
                    flipped = {"le": "ge", "lt": "gt", "ge": "le", "gt": "lt", "eq": "eq", "ne": "ne"}.get(op, op)
                    bounds.append((rhs.var_id, flipped, bound, flipped in ("gt", "lt")))
        # Recurse into other expression types that might contain compares
        for child in (
            getattr(node, "lhs", None),
            getattr(node, "rhs", None),
            getattr(node, "predicate", None),
            getattr(node, "iterable", None),
        ):
            visit(child)
        for arg in getattr(node, "args", []) or []:
            visit(arg)
    
    visit(expr)
    return bounds


def _const_number(expr: VirExpr | None) -> float | None:
    if expr is None:
        return None
    if getattr(expr, "kind", None) in ("const_real", "const_int"):
        try:
            return float(expr.value)
        except (TypeError, ValueError):
            return None
    if getattr(expr, "kind", None) == "const_value" and isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool):
        return float(expr.value)
    if getattr(expr, "kind", None) == "neg":
        inner = _const_number(getattr(expr, "lhs", None))
        return -inner if inner is not None else None
    return None


class AstTranspiler:
    _CALLS = {"require", "derive", "prefer", "forall", "sum", "dot", "quad_form", "sqrt", "keys_subset"}
    _BUILTINS = {"abs": builtins.abs, "min": builtins.min, "max": builtins.max,
                 "len": builtins.len, "round": builtins.round}

    def __init__(self, func, model, config=None):
        self.func = inspect.unwrap(func)
        self.model = model
        self.config = dict(config or {})
        params = list(inspect.signature(self.func).parameters.values())
        if len(params) != 1 or params[0].kind not in (inspect.Parameter.POSITIONAL_ONLY,
                                                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise TranspileError("schema function must take exactly one positional model parameter")
        self.param_name = params[0].name
        self.globals = dict(getattr(self.func, "__globals__", {}))
        code = getattr(self.func, "__code__", None)
        for name, cell in zip(getattr(code, "co_freevars", ()), getattr(self.func, "__closure__", ()) or ()):
            try:
                self.globals[name] = cell.cell_contents
            except ValueError:
                raise TranspileError(f"unbound closure constant {name!r}") from None
        try:
            hints = typing.get_type_hints(self.func, localns=self.globals)
        except Exception:
            hints = {}
        if hints.get(self.param_name, params[0].annotation) is not model:
            raise TranspileError("the resolved model annotation does not match the supplied model")
        lowered = lower_pydantic_model(model)
        self.graph = lowered["graph"]
        self.var_table = lowered["var_table"]
        self.lowered = lowered
        self.name_to_id = dict(self.graph.name_to_id)
        self.id_to_name = dict(self.graph.id_to_name)
        self.requires, self.derives, self.prefers = [], [], []
        self.zero_dof_rules, self.soft_preferences = [], []
        self._lambda_names: list[str] = []
        self._source_name = "<schema>"

    def _error(self, node, message):
        line = getattr(node, "lineno", 0)
        raise TranspileError(f"{self._source_name}:{line}: {message}")

    def transpile(self) -> VirSpec:
        try:
            file_name = inspect.getsourcefile(self.func) or "<schema>"
            source, start = inspect.getsourcelines(self.func)
        except (OSError, TypeError) as exc:
            raise TranspileError(f"schema source is unavailable (define it in a Python module): {exc}") from exc
        self._source_name = f"{file_name}:{start}"
        try:
            tree = ast.parse(textwrap.dedent("".join(source)))
        except SyntaxError as exc:
            raise TranspileError(f"invalid schema source: {exc}") from exc
        matches = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == self.func.__name__]
        if not matches:
            raise TranspileError("cannot locate the schema function in its source")
        node = matches[0]
        if isinstance(node, ast.AsyncFunctionDef):
            self._error(node, "schema declarations must be synchronous")
        
        # Pass 1: Extract simple bounds from require() calls to propagate to variable types
        self._collect_and_apply_bounds(node)
        
        # Pass 2: Full transpilation
        for stmt in node.body:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                self._error(stmt, "only vt.require/derive/prefer calls are allowed; arbitrary Python is not compiled")
            self._handle_top_level(stmt.value)
        self._build_zero_dof_rules()
        spec = VirSpec(
            model=self.model,
            variables={vid: {"name": info["name"], "type": info["type"]} for vid, info in self.var_table.items()},
            id_to_name=dict(self.id_to_name), name_to_id=dict(self.name_to_id),
            field_order=self.graph.topological_order(), requires=self.requires, derives=self.derives,
            prefers=self.prefers, zero_dof_rules=self.zero_dof_rules,
            soft_preferences=self.soft_preferences, config=dict(self.config),
            model_schema=self.lowered["model_schema"], field_aliases=self.lowered["field_aliases"],
            portable_model=self.lowered["portable_model"],
        )
        spec.containers = analyze_containers(spec)
        return spec

    def _collect_and_apply_bounds(self, node: ast.AST) -> None:
        """First pass: extract simple variable bounds from require() calls and apply to variable types."""
        for stmt in node.body:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            dsl_name = self._dsl_name(call)
            if dsl_name != "require":
                continue
            # Parse the argument to extract simple bounds
            arg = call.args[0] if call.args else None
            if arg is None:
                continue
            # Parse the expression to VirExpr
            expr = self.parse_expr(arg)
            # Extract simple bounds
            bounds = _extract_simple_bounds(expr)
            # Apply bounds to variable types
            for var_id, op, value, exclusive in bounds:
                info = self.var_table.get(var_id)
                if info is None:
                    continue
                vtype = info["type"]
                if op in ("ge", "gt"):
                    current = getattr(vtype, "min", None)
                    if current is None or value > current:
                        vtype.min = value
                        vtype.min_exclusive = exclusive
                    elif value == current and exclusive:
                        vtype.min_exclusive = True
                elif op in ("le", "lt"):
                    current = getattr(vtype, "max", None)
                    if current is None or value < current:
                        vtype.max = value
                        vtype.max_exclusive = exclusive
                    elif value == current and exclusive:
                        vtype.max_exclusive = True
                elif op == "eq":
                    # Pin to exact value
                    vtype.min = value
                    vtype.max = value
                    vtype.min_exclusive = False
                    vtype.max_exclusive = False

    def _resolve(self, node):
        if isinstance(node, ast.Name):
            return self.globals.get(node.id, getattr(builtins, node.id, None))
        if isinstance(node, ast.Attribute):
            base = self._resolve(node.value)
            return getattr(base, node.attr, None)
        return None

    def _dsl_name(self, call):
        obj = self._resolve(call.func)
        name = getattr(obj, "_veritas_dsl", None)
        if name in self._CALLS:
            return name
        # Direct imports from the DSL module are also identity checked.
        module = getattr(obj, "__module__", None)
        short = getattr(obj, "__name__", None)
        if module == "veritas.decorators" and short in self._CALLS:
            return short
        return None

    def _kw(self, call, allowed):
        seen = set()
        for kw in call.keywords:
            if kw.arg not in allowed or kw.arg in seen:
                self._error(call, f"unexpected/duplicate keyword {kw.arg!r}")
            seen.add(kw.arg)
        return {kw.arg: self._resolve_constant(kw.value) for kw in call.keywords}

    def _arity(self, call, count, name):
        if len(call.args) != count:
            self._error(call, f"{name} expects {count} positional arguments")

    @staticmethod
    def _tol(node, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise TranspileError(f"{getattr(node, 'lineno', 0)}: tolerance must be finite and non-negative")
        return float(value)

    def _handle_top_level(self, call):
        name = self._dsl_name(call)
        if name not in {"require", "derive", "prefer"}:
            self._error(call, "top-level call is not a Veritas constraint declaration")
        self._arity(call, 1, name)
        kw = self._kw(call, {"tol", "weight", "tag"} if name == "prefer" else {"tol"})
        tol = self._tol(call, kw.pop("tol", 1e-6 if name == "derive" else 0.0))
        expr = self.parse_expr(call.args[0])
        stack = [expr]
        while stack:
            node = stack.pop()
            if node.kind == "compare":
                node.tol = tol
            stack.extend(node.args)
            stack.extend(child for child in (node.lhs, node.rhs, node.predicate) if child is not None)
        if name == "require":
            self.requires.append(expr)
        elif name == "derive":
            if expr.kind != "compare" or expr.op != "eq":
                self._error(call, "derive requires target == expression")
            target, rhs = self._derive_target(expr)
            inputs = rhs.collect_var_ids()
            if target in inputs:
                self._error(call, "a derivation may not depend on itself")
            if any(r.target_var == target for r in self.zero_dof_rules):
                self._error(call, "duplicate derivation target")
            self.derives.append(expr)
            self.graph.add_derive(self.id_to_name[target], [self.id_to_name[v] for v in inputs], expr=expr, tol=tol)
        else:
            weight = kw.pop("weight", 1.0)
            tag = kw.pop("tag", "")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                self._error(call, "preference weight must be finite and non-negative")
            if not isinstance(tag, str):
                self._error(call, "preference tag must be a string")
            expr.weight, expr.tag = float(weight), tag
            self.prefers.append(expr)
            self.soft_preferences.append(SoftPreference(len(self.soft_preferences) + 1, tag, float(weight), expr))

    def parse_expr(self, node) -> VirExpr:
        if isinstance(node, ast.Constant):
            if node.value is None:
                return VirExpr(kind="const_none")
            kind = {bool: "const_bool", int: "const_int", float: "const_real", str: "const_str"}.get(type(node.value))
            if kind is None:
                self._error(node, f"unsupported constant {type(node.value).__name__}")
            if kind == "const_real" and not math.isfinite(node.value):
                self._error(node, "non-finite constant")
            return VirExpr(kind=kind, value=node.value)
        if isinstance(node, ast.Name):
            if node.id == self.param_name:
                return VirExpr(kind="model_ref", name=node.id)
            if node.id in self._lambda_names:
                return VirExpr(kind="elem", name=node.id)
            if node.id in self.globals:
                return VirExpr(kind="const_value", value=self._resolve_constant(node))
            self._error(node, f"unknown symbol {node.id!r}")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == self.param_name:
                if node.attr not in self.name_to_id:
                    self._error(node, f"unknown model field {node.attr!r}")
                return VirExpr(kind="var", var_id=self.name_to_id[node.attr], name=node.attr)
            self._error(node, "only direct model-field attributes are supported")
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "values"
                    and not node.args and not node.keywords):
                return VirExpr(kind="values", args=[self.parse_expr(node.func.value)])
            name = self._dsl_name(node)
            if name in {"sum", "sqrt", "dot", "quad_form", "keys_subset", "forall"}:
                counts = {"sum": 1, "sqrt": 1, "dot": 2, "quad_form": 2, "keys_subset": 2, "forall": 2}
                self._arity(node, counts[name], name)
                if node.keywords:
                    self._error(node, f"{name} does not accept keywords")
                if name == "forall":
                    lam = node.args[1]
                    if (not isinstance(lam, ast.Lambda) or len(lam.args.args) != 1 or lam.args.defaults
                            or lam.args.vararg or lam.args.kwarg or lam.args.kwonlyargs or lam.args.posonlyargs):
                        self._error(node, "forall requires a one-argument lambda")
                    self._lambda_names.append(lam.args.args[0].arg)
                    try:
                        pred = self.parse_expr(lam.body)
                    finally:
                        self._lambda_names.pop()
                    return VirExpr(kind="forall", iterable=self.parse_expr(node.args[0]), predicate=pred)
                args = [self.parse_expr(a) for a in node.args]
                if name == "quad_form":
                    from ..cartridges.quant import (
                        cholesky,
                        cholesky_with_jitter,
                        covariance_labels,
                        covariance_matrix,
                        eigendecomposition_split,
                    )
                    cov = self._resolve_constant(node.args[1])
                    if not isinstance(cov, dict):
                        self._error(node, "quad_form covariance must be a constant mapping")
                    labels = covariance_labels(cov)
                    matrix = covariance_matrix(labels, cov)
                    factor = cholesky_with_jitter(matrix)
                    if factor is None:
                        # Non-PSD matrix: split into Q+ - Q- via eigendecomposition
                        q_plus, q_minus = eigendecomposition_split(matrix)
                        factor_plus = cholesky(q_plus)
                        factor_minus = cholesky(q_minus)
                        if factor_plus is None or factor_minus is None:
                            self._error(node, "covariance eigendecomposition failed")
                        # Return a special marker for split quadratic form
                        return VirExpr(
                            kind="quad_form_split",
                            args=args,
                            value=cov,
                            cholesky_l_plus=factor_plus,
                            cholesky_l_minus=factor_minus,
                        )
                    return VirExpr(kind="quad_form_cholesky", args=args, value=cov, cholesky_l=factor)
                if name in {"dot", "keys_subset"}:
                    return VirExpr(kind="quad_form_cholesky" if name == "quad_form" else name,
                                   args=args, value=self._resolve_constant(node.args[1]))
                return VirExpr(kind=name, args=args)
            obj = self._resolve(node.func)
            short = next((k for k, v in self._BUILTINS.items() if obj is v), None)
            if short and not node.keywords:
                return VirExpr(kind="call", name=short, args=[self.parse_expr(a) for a in node.args])
            self._error(node, f"unsupported function {ast.unparse(node.func)!r}")
        if isinstance(node, ast.BinOp):
            kinds = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div",
                     ast.Pow: "pow", ast.Mod: "mod", ast.BitAnd: "and", ast.BitOr: "or"}
            kind = kinds.get(type(node.op))
            if kind is None:
                self._error(node, f"unsupported operator {type(node.op).__name__}")
            left, right = self.parse_expr(node.left), self.parse_expr(node.right)
            return VirExpr(kind=kind, lhs=left, rhs=right, args=[left, right] if kind in ("and", "or") else [])
        if isinstance(node, ast.UnaryOp):
            operand = self.parse_expr(node.operand)
            if isinstance(node.op, ast.USub):
                return VirExpr(kind="neg", lhs=operand, args=[operand])
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.Not):
                return VirExpr(kind="not", args=[operand])
            self._error(node, "unsupported unary operator")
        if isinstance(node, ast.Compare):
            ops = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le",
                   ast.Gt: "gt", ast.GtE: "ge", ast.In: "in", ast.NotIn: "not_in"}
            parts = []
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                if type(op) not in ops:
                    self._error(node, f"unsupported comparison {type(op).__name__}")
                parts.append(VirExpr(kind="compare", op=ops[type(op)], lhs=self.parse_expr(left), rhs=self.parse_expr(right)))
                left = right
            return parts[0] if len(parts) == 1 else VirExpr(kind="and", args=parts)
        if isinstance(node, ast.BoolOp):
            return VirExpr(kind="and" if isinstance(node.op, ast.And) else "or",
                           args=[self.parse_expr(v) for v in node.values])
        if isinstance(node, ast.IfExp):
            return VirExpr(kind="ifexp", args=[self.parse_expr(node.test), self.parse_expr(node.body), self.parse_expr(node.orelse)])
        if isinstance(node, ast.Subscript):
            return VirExpr(kind="subscript", args=[self.parse_expr(node.value)], value=self._resolve_constant(node.slice))
        if isinstance(node, (ast.List, ast.Set, ast.Tuple, ast.Dict)):
            return VirExpr(kind={ast.List: "const_list", ast.Set: "const_set", ast.Tuple: "const_tuple", ast.Dict: "const_dict"}[type(node)],
                           value=self._resolve_constant(node))
        self._error(node, f"unsupported syntax {type(node).__name__}")

    def _resolve_constant(self, node):
        """Capture constants by value; never evaluate source text."""
        try:
            if isinstance(node, ast.Name):
                if node.id not in self.globals:
                    raise ValueError(f"unknown constant {node.id!r}")
                value = self.globals[node.id]
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                value = self._resolve_constant(node.operand)
                value = -value if isinstance(node.op, ast.USub) else +value
            elif isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow):
                fn = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                      ast.Div: operator.truediv, ast.Pow: operator.pow}[type(node.op)]
                value = fn(self._resolve_constant(node.left), self._resolve_constant(node.right))
            else:
                value = ast.literal_eval(node)
            value = copy.deepcopy(value)
            self._check_constant(value)
            return value
        except Exception as exc:
            self._error(node, f"expected an immutable JSON-compatible constant: {exc}")

    def _check_constant(self, value, depth=0):
        if depth > 32:
            raise ValueError("constant nesting exceeds 32")
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("non-finite constant")
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                self._check_constant(item, depth + 1)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool, tuple)) and key is not None:
                    raise ValueError("unsupported mapping key")
                self._check_constant(key, depth + 1)
                self._check_constant(item, depth + 1)
            return
        raise ValueError(f"unsupported constant type {type(value).__name__}")

    def _derive_target(self, expr):
        if expr.lhs.kind == "var" and not any(v == expr.lhs.var_id for v in expr.rhs.collect_var_ids()):
            return expr.lhs.var_id, expr.rhs
        if expr.rhs.kind == "var" and not any(v == expr.rhs.var_id for v in expr.lhs.collect_var_ids()):
            expr.lhs, expr.rhs = expr.rhs, expr.lhs
            return expr.lhs.var_id, expr.rhs
        self._error(expr, "derive requires one isolated target field")

    def _build_zero_dof_rules(self):
        for expr in self.derives:
            target, rhs = self._derive_target(expr)
            precision = min(15, max(0, math.ceil(-math.log10(expr.tol)))) if expr.tol > 0 else 15
            self.zero_dof_rules.append(ZeroDofRule(target, self.id_to_name[target], rhs.collect_var_ids(),
                                                   expr, precision_digits=precision))