"""AST transpiler: Python function + Pydantic model -> VirSpec."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None

from .vir import VirExpr, VirSpec, VirType, ZeroDofRule, SoftPreference
from .dependency_graph import DependencyGraph
from .schema_lowering import lower_pydantic_model, _python_type_to_vir_type


class TranspileError(Exception):
    pass


class AstTranspiler:
    def __init__(self, func, model, config: Optional[Dict[str, Any]] = None):
        self.func = func
        self.model = model
        self.config = config or {}
        self.param_name = self._first_param_name(func)
        self.globals = dict(getattr(func, "__globals__", {}))
        # capture closure (free) variables when available
        code = getattr(func, "__code__", None)
        closure = getattr(func, "__closure__", None)
        if code is not None and closure is not None:
            for name, cell in zip(getattr(code, "co_freevars", ()), closure):
                try:
                    self.globals[name] = cell.cell_contents
                except ValueError:
                    pass
        lowered = lower_pydantic_model(model)
        self.graph: DependencyGraph = lowered["graph"]
        self.var_table = lowered["var_table"]
        self.var_ids: Dict[str, int] = {name: info["var_id"] for name, info in zip(
            [f["name"] for f in lowered["fields"]],
            lowered["fields"],
        )} if False else {f["name"]: f["var_id"] for f in lowered["fields"]}
        # name->id map convenience
        self.name_to_id: Dict[str, int] = dict(self.graph.name_to_id)
        self.id_to_name: Dict[int, str] = dict(self.graph.id_to_name)

        self.requires: List[VirExpr] = []
        self.derives: List[VirExpr] = []
        self.prefers: List[VirExpr] = []
        self.zero_dof_rules: List[ZeroDofRule] = []
        self.soft_preferences: List[SoftPreference] = []
        self._soft_counter = 0
        self._current_lambda_param: Optional[str] = None

    # ---------- public ----------
    def transpile(self) -> VirSpec:
        try:
            source = textwrap.dedent(inspect.getsource(self.func))
        except (OSError, TypeError) as e:
            raise TranspileError(f"Cannot get source for {self.func}: {e}")
        tree = ast.parse(source)
        func_nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not func_nodes:
            func_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not func_nodes:
            raise TranspileError("No function definition found")
        func_node = func_nodes[0]

        for stmt in func_node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self._handle_top_level(stmt.value)

        # Build zero-dof rules from derives where target is a derived var
        self._build_zero_dof_rules()

        variables: Dict[int, Dict[str, Any]] = {}
        for vid, info in self.var_table.items():
            variables[vid] = {"name": info["name"], "type": info["type"]}

        spec = VirSpec(
            model=self.model,
            variables=variables,
            id_to_name=dict(self.id_to_name),
            name_to_id=dict(self.name_to_id),
            field_order=[],  # filled by caller / dependency graph
            requires=self.requires,
            derives=self.derives,
            prefers=self.prefers,
            zero_dof_rules=self.zero_dof_rules,
            soft_preferences=self.soft_preferences,
            config=dict(self.config),
        )
        # topological order
        spec.field_order = self.graph.topological_order()
        return spec

    # ---------- top-level handlers ----------
    def _handle_top_level(self, call: ast.Call):
        name = self._call_name(call)
        short = name.split(".")[-1]
        if short == "require":
            expr = self.parse_expr(call.args[0]) if call.args else None
            tol = self._kw(call, "tol", 1e-6)
            if expr is not None:
                expr.tol = tol
                self.requires.append(expr)
        elif short == "derive":
            expr = self.parse_expr(call.args[0]) if call.args else None
            tol = self._kw(call, "tol", 1e-6)
            if expr is not None:
                expr.tol = tol
                self.derives.append(expr)
                target, inputs = self._derive_target_inputs(expr)
                if target is not None and target in self.name_to_id:
                    self.graph.add_derive(target, inputs, expr=expr, tol=tol)
        elif short == "prefer":
            expr = self.parse_expr(call.args[0]) if call.args else None
            weight = self._kw(call, "weight", 1.0)
            tag = self._kw(call, "tag", "")
            if expr is not None:
                expr.weight = weight
                expr.tag = tag
                self.prefers.append(expr)
                self._soft_counter += 1
                self.soft_preferences.append(
                    SoftPreference(id=self._soft_counter, tag=tag, weight=weight, expr=expr)
                )
        else:
            # ignore unknown top-level calls
            pass

    # ---------- expression parsing ----------
    def parse_expr(self, node: ast.AST) -> VirExpr:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return VirExpr(kind="const_bool", value=node.value)
            if isinstance(node.value, int):
                return VirExpr(kind="const_int", value=node.value)
            if isinstance(node.value, float):
                return VirExpr(kind="const_real", value=node.value)
            if isinstance(node.value, str):
                return VirExpr(kind="const_str", value=node.value)
            if isinstance(node.value, (set, frozenset)):
                return VirExpr(kind="const_set", value=sorted(node.value))
            if isinstance(node.value, (list, tuple)):
                return VirExpr(kind="const_list", value=list(node.value))
            if node.value is None:
                return VirExpr(kind="const_none", value=None)
            return VirExpr(kind="const", value=node.value)

        if isinstance(node, ast.Name):
            if node.id == self.param_name:
                # bare model reference; rare
                return VirExpr(kind="model_ref", name=node.id)
            # lambda-bound element variable
            if self._current_lambda_param and node.id == self._current_lambda_param:
                return VirExpr(kind="elem", name=node.id)
            # global constant
            if node.id in self.globals:
                val = self.globals[node.id]
                if isinstance(val, (int, float, bool, str, dict, list, tuple, set, frozenset)):
                    return VirExpr(kind="const_value", value=val)
            return VirExpr(kind="name", name=node.id)

        if isinstance(node, ast.Attribute):
            # model field access: p.weights
            if isinstance(node.value, ast.Name) and node.value.id == self.param_name:
                fname = node.attr
                if fname in self.name_to_id:
                    return VirExpr(kind="var", var_id=self.name_to_id[fname], name=fname)
            # something.values()
            if node.attr == "values" and isinstance(node.value, ast.Attribute):
                inner = self.parse_expr(node.value)
                return VirExpr(kind="values", args=[inner])
            if node.attr == "values" and isinstance(node.value, ast.Name) and node.value.id == self.param_name:
                # p.values() unlikely
                pass
            return VirExpr(kind="attr", name=ast.unparse(node))

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "values" and not node.args:
                inner = self.parse_expr(node.func.value)
                return VirExpr(kind="values", args=[inner])
            cname = self._call_name(node)
            short = cname.split(".")[-1]
            if short in ("sqrt",):
                return VirExpr(kind="sqrt", args=[self.parse_expr(node.args[0])])
            if short in ("sum",):
                arg = self.parse_expr(node.args[0]) if node.args else VirExpr(kind="none")
                return VirExpr(kind="sum", args=[arg])
            if short in ("dot",):
                args = [self.parse_expr(a) for a in node.args]
                # resolve constants for second arg if global
                extra = {}
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Name):
                    g = self.globals.get(node.args[1].id)
                    if g is not None:
                        extra["value"] = g
                e = VirExpr(kind="dot", args=args)
                if "value" in extra:
                    e.value = extra["value"]
                return e
            if short in ("quad_form",):
                # args: (weights, cov)
                w_expr = self.parse_expr(node.args[0])
                cov = self._resolve_constant(node.args[1]) if len(node.args) > 1 else None
                w_name = w_expr.name if w_expr.kind == "var" else None
                w_var_ids = [w_expr.var_id] if w_expr.var_id is not None else []
                # if weights is a dict var, still single var id
                cholesky_l = self._cholesky(cov)
                return VirExpr(
                    kind="quad_form_cholesky",
                    args=[w_expr],
                    weights_vars=w_var_ids,
                    cholesky_l=cholesky_l,
                    value=cov,
                )
            if short in ("keys_subset",):
                d_expr = self.parse_expr(node.args[0]) if node.args else None
                allowed = self._resolve_constant(node.args[1]) if len(node.args) > 1 else None
                return VirExpr(kind="keys_subset", args=[d_expr] if d_expr else [], value=allowed)
            if short in ("forall",):
                # forall(iterable, lambda)
                iterable_expr = self.parse_expr(node.args[0]) if node.args else None
                lam = node.args[1] if len(node.args) > 1 else None
                pred = None
                if isinstance(lam, ast.Lambda):
                    old = self._current_lambda_param
                    self._current_lambda_param = lam.args.args[0].arg if lam.args.args else None
                    pred = self.parse_expr(lam.body)
                    self._current_lambda_param = old
                return VirExpr(kind="forall", iterable=iterable_expr, predicate=pred)
            # vt.require/derive/prefer shouldn't be nested; but handle as expr pass-through
            if short in ("require", "derive", "prefer"):
                if node.args:
                    return self.parse_expr(node.args[0])
            # generic function call
            args = [self.parse_expr(a) for a in node.args]
            return VirExpr(kind="call", name=cname, args=args)

        if isinstance(node, ast.BinOp):
            left = self.parse_expr(node.left)
            right = self.parse_expr(node.right)
            if isinstance(node.op, ast.Add):
                return VirExpr(kind="add", lhs=left, rhs=right, args=[left, right])
            if isinstance(node.op, ast.Sub):
                return VirExpr(kind="sub", lhs=left, rhs=right, args=[left, right])
            if isinstance(node.op, ast.Mult):
                return VirExpr(kind="mul", lhs=left, rhs=right, args=[left, right])
            if isinstance(node.op, ast.Div):
                return VirExpr(kind="div", lhs=left, rhs=right, args=[left, right])
            if isinstance(node.op, ast.Pow):
                return VirExpr(kind="pow", lhs=left, rhs=right, args=[left, right])
            if isinstance(node.op, ast.Mod):
                return VirExpr(kind="mod", lhs=left, rhs=right, args=[left, right])
            if isinstance(node.op, ast.BitAnd):
                return VirExpr(kind="and", args=[left, right])
            if isinstance(node.op, ast.BitOr):
                return VirExpr(kind="or", args=[left, right])
            return VirExpr(kind="binop", lhs=left, rhs=right, args=[left, right])

        if isinstance(node, ast.UnaryOp):
            operand = self.parse_expr(node.operand)
            if isinstance(node.op, ast.USub):
                return VirExpr(kind="neg", args=[operand], lhs=operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            return VirExpr(kind="unary", args=[operand])

        if isinstance(node, ast.Compare):
            if len(node.ops) == 1:
                left = self.parse_expr(node.left)
                right = self.parse_expr(node.comparators[0])
                op = self._cmp_op(node.ops[0])
                e = VirExpr(kind="compare", op=op, lhs=left, rhs=right, tol=1e-6)
                return e
            # chained compares -> and
            parts = []
            cur = node.left
            for op, comp in zip(node.ops, node.comparators):
                parts.append(ast.Compare(left=cur, ops=[op], comparators=[comp]))
                cur = comp
            # parse each as compare
            parsed = [self.parse_expr(p) for p in parts]  # won't work: p is ast.Compare already okay
            # Actually parse_expr accepts ast.Compare
            and_expr = VirExpr(kind="and", args=parsed)
            return and_expr

        if isinstance(node, ast.BoolOp):
            parsed = [self.parse_expr(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return VirExpr(kind="and", args=parsed)
            if isinstance(node.op, ast.Or):
                return VirExpr(kind="or", args=parsed)
            return VirExpr(kind="boolop", args=parsed)

        if isinstance(node, ast.IfExp):
            return VirExpr(kind="ifexp", args=[self.parse_expr(node.test), self.parse_expr(node.body), self.parse_expr(node.orelse)])

        if isinstance(node, ast.Subscript):
            base = self.parse_expr(node.value)
            sl = node.slice
            if isinstance(sl, ast.Constant):
                key = sl.value
            elif isinstance(sl, ast.Index):  # py<3.9
                key = self._resolve_constant(sl.value)
            else:
                key = self._resolve_constant(sl)
            return VirExpr(kind="subscript", args=[base], value=key, name=str(key))

        if isinstance(node, ast.Lambda):
            old = self._current_lambda_param
            if node.args.args:
                self._current_lambda_param = node.args.args[0].arg
            body = self.parse_expr(node.body)
            self._current_lambda_param = old
            return VirExpr(kind="lambda", predicate=body)

        if isinstance(node, ast.List):
            return VirExpr(kind="const_list", value=[self._resolve_constant(e) for e in node.elts])
        if isinstance(node, ast.Set):
            return VirExpr(kind="const_set", value=sorted(self._resolve_constant(e) for e in node.elts))
        if isinstance(node, ast.Tuple):
            return VirExpr(kind="const_tuple", value=tuple(self._resolve_constant(e) for e in node.elts))
        if isinstance(node, ast.Dict):
            keys = [self._resolve_constant(k) if k is not None else None for k in node.keys]
            vals = [self._resolve_constant(v) for v in node.values]
            return VirExpr(kind="const_dict", value=dict(zip(keys, vals)))

        return VirExpr(kind="unknown", name=ast.unparse(node) if hasattr(ast, "unparse") else str(node))

    # ---------- helpers ----------
    def _first_param_name(self, func) -> str:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params:
            raise TranspileError("verified_schema function must have at least one parameter")
        return params[0].name

    def _call_name(self, call: ast.Call) -> str:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return ast.unparse(call.func) if hasattr(ast, "unparse") else f"{call.func.value.id}.{call.func.attr}"
        return ""

    def _kw(self, call: ast.Call, key: str, default: Any) -> Any:
        for kw in call.keywords:
            if kw.arg == key:
                return self._resolve_constant(kw.value)
        return default

    def _cmp_op(self, op: ast.cmpop) -> str:
        mapping = {
            ast.Eq: "eq",
            ast.NotEq: "ne",
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
        }
        return mapping.get(type(op), "eq")

    def _resolve_constant(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.globals:
                return self.globals[node.id]
            return node.id
        if isinstance(node, ast.List):
            return [self._resolve_constant(e) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._resolve_constant(e) for e in node.elts)
        if isinstance(node, ast.Set):
            return {self._resolve_constant(e) for e in node.elts}
        if isinstance(node, ast.Dict):
            ks = [self._resolve_constant(k) if k is not None else None for k in node.keys]
            vs = [self._resolve_constant(v) for v in node.values]
            return dict(zip(ks, vs))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = self._resolve_constant(node.operand)
            if isinstance(v, (int, float)):
                return -v
        # fallback: try ast.unparse + eval (best effort, safe-ish on globals)
        try:
            src = ast.unparse(node)
            return eval(src, {"__builtins__": {}}, self.globals)
        except Exception:
            return None

    def _cholesky(self, cov: Any) -> List[List[float]]:
        if cov is None:
            return []
        if np is None:
            return []
        try:
            if isinstance(cov, dict):
                keys = sorted({k[0] if isinstance(k, tuple) else k for k in cov.keys()})
                n = len(keys)
                idx = {k: i for i, k in enumerate(keys)}
                mat = [[0.0] * n for _ in range(n)]
                for (a, b), v in cov.items():
                    if a in idx and b in idx:
                        mat[idx[a]][idx[b]] = float(v)
                        mat[idx[b]][idx[a]] = float(v)
                cov = mat
            arr = np.array(cov, dtype=float)
            L = np.linalg.cholesky(arr)
            return L.tolist()
        except Exception:
            return []

    def _derive_target_inputs(self, expr: VirExpr):
        target = None
        inputs: List[str] = []
        if expr.kind == "compare" and expr.op == "eq":
            if expr.lhs is not None and expr.lhs.kind == "var" and expr.lhs.name is not None:
                target = expr.lhs.name
            elif expr.lhs is not None and expr.lhs.kind == "var" and expr.lhs.var_id is not None:
                target = self.id_to_name.get(expr.lhs.var_id)
        # collect var names from rhs
        rhs = expr.rhs if expr.kind == "compare" else expr
        ids = rhs.collect_var_ids() if hasattr(rhs, "collect_var_ids") else []
        for vid in ids:
            nm = self.id_to_name.get(vid)
            if nm is not None and nm not in inputs:
                inputs.append(nm)
        # Also resolve names from const_value? no
        # If target None, try any var on lhs of compare
        if target is None and expr.kind == "compare" and expr.lhs is not None and expr.lhs.kind == "var":
            target = expr.lhs.name or self.id_to_name.get(expr.lhs.var_id)
        return target, inputs

    def _build_zero_dof_rules(self):
        for d in self.derives:
            target, inputs = self._derive_target_inputs(d)
            if target is None:
                continue
            # only derived fields (not free) become zero-dof
            if target not in self.name_to_id:
                continue
            target_vid = self.name_to_id[target]
            input_vids = [self.name_to_id[x] for x in inputs if x in self.name_to_id]
            # precision digits from tol? default 6
            prec = 6
            if getattr(d, "tol", None):
                try:
                    prec = max(1, int(-np.log10(d.tol))) if d.tol > 0 else 6
                except Exception:
                    prec = 6
            rule = ZeroDofRule(
                target_var=target_vid,
                target_json_key=target,
                input_vars=input_vids,
                rhs_expr=d,
                precision_digits=prec,
            )
            self.zero_dof_rules.append(rule)
            # attach to graph
            self.graph.add_derive(target, inputs, expr=d, tol=getattr(d, "tol", 1e-6))
