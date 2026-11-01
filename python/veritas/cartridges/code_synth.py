"""``veritas.code`` — static guardrails for synthesized Python.

Checks the properties that matter for generated code before it is executed: undefined
identifiers (scope-aware), literal division by zero, and literal out-of-bounds indexing.
It is intentionally conservative: checks it cannot complete are reported as warnings, never
as passes.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class CodeIssue:
    kind: str
    message: str
    line: int = 0
    column: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "message": self.message, "line": self.line, "column": self.column}


@dataclass
class CodeReport:
    errors: List[CodeIssue] = field(default_factory=list)
    warnings: List[CodeIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, kind: str, message: str, node: Optional[ast.AST] = None) -> None:
        self.errors.append(CodeIssue(kind, message, getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))

    def add_warning(self, kind: str, message: str, node: Optional[ast.AST] = None) -> None:
        self.warnings.append(CodeIssue(kind, message, getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
        }


class SymbolTable:
    def __init__(self):
        self.scopes: List[Dict[str, Any]] = [{}]

    def push(self) -> None:
        self.scopes.append({})

    def pop(self) -> None:
        if len(self.scopes) > 1:
            self.scopes.pop()

    def define(self, name: str, value: Any = None) -> None:
        self.scopes[-1][name] = value

    def resolve(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self.scopes))


_BUILTINS = set(dir(builtins))


class _Checker(ast.NodeVisitor):
    def __init__(self):
        self.symbols = SymbolTable()
        for name in _BUILTINS | {"__name__", "__file__", "__doc__", "__package__"}:
            self.symbols.define(name)
        self.report = CodeReport()

    # -- scopes ----------------------------------------------------------
    def visit_Module(self, node: ast.Module) -> None:
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.symbols.define(statement.name)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                self._define_imports(statement)
        self.generic_visit(node)

    def _define_imports(self, node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.symbols.define(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    self.symbols.define(alias.asname or alias.name)

    def visit_Import(self, node: ast.Import) -> None:
        self._define_imports(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._define_imports(node)

    def _visit_function(self, node: Any) -> None:
        self.symbols.define(node.name)
        self.symbols.push()
        arguments = node.args
        for arg in list(arguments.posonlyargs) + list(arguments.args) + list(arguments.kwonlyargs):
            self.symbols.define(arg.arg)
        if arguments.vararg:
            self.symbols.define(arguments.vararg.arg)
        if arguments.kwarg:
            self.symbols.define(arguments.kwarg.arg)
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.symbols.define(statement.name)
        for statement in node.body:
            self.visit(statement)
        self.symbols.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.define(node.name)
        self.symbols.push()
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.symbols.define(statement.name)
        for statement in node.body:
            self.visit(statement)
        self.symbols.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.symbols.push()
        for arg in node.args.args:
            self.symbols.define(arg.arg)
        self.visit(node.body)
        self.symbols.pop()

    # -- names / assignments --------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self.symbols.resolve(node.id):
            self.report.add_error("undefined_identifier", f"undefined identifier {node.id!r}", node)

    def _define_target(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self.symbols.define(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                self._define_target(element)
        elif isinstance(node, ast.Starred):
            self._define_target(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._define_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self._define_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._define_target(node.target)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._define_target(node.target)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)  # type: ignore[arg-type]

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._define_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.symbols.define(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.visit(node.iter)
        self._define_target(node.target)
        for condition in node.ifs:
            self.visit(condition)

    def visit_ListComp(self, node: Any) -> None:
        # Generators bind their targets before the element expression is evaluated.
        self.symbols.push()
        for generator in node.generators:
            self.visit(generator)
        self.visit(node.elt)
        self.symbols.pop()

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.symbols.push()
        for generator in node.generators:
            self.visit(generator)
        self.visit(node.key)
        self.visit(node.value)
        self.symbols.pop()

    # -- numeric safety --------------------------------------------------
    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Div, ast.Mod, ast.FloorDiv)):
            divisor = node.right
            if isinstance(divisor, ast.Constant) and divisor.value == 0:
                self.report.add_error("division_by_zero", "literal division by zero", node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.generic_visit(node)
        base = node.value
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, int):
            if isinstance(base, (ast.List, ast.Tuple)) and base.elts:
                if index.value >= len(base.elts) or index.value < -len(base.elts):
                    self.report.add_error(
                        "index_out_of_bounds",
                        f"index {index.value} is out of bounds for a literal of length {len(base.elts)}",
                        node,
                    )
            if isinstance(base, ast.Constant) and isinstance(base.value, str):
                if index.value >= len(base.value) or index.value < -len(base.value):
                    self.report.add_error(
                        "index_out_of_bounds",
                        f"index {index.value} is out of bounds for a string of length {len(base.value)}",
                        node,
                    )


def verify_python_source(source: str, *, filename: str = "<synthesized>") -> CodeReport:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        report = CodeReport()
        report.add_error("syntax_error", str(exc))
        return report
    checker = _Checker()
    checker.visit(tree)
    return checker.report


# ---------------------------------------------------------------------------
# compatibility helpers
# ---------------------------------------------------------------------------

def scope_safe(identifiers: Iterable[str], used: Iterable[str]) -> bool:
    return set(used).issubset(set(identifiers))


def no_division_by_zero(divisors: Iterable[float]) -> bool:
    return all(divisor != 0 for divisor in divisors)


def array_bounds_ok(index: int, size: int) -> bool:
    return 0 <= index < size


def type_compatible(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    numeric = {"int", "float"}
    return expected in numeric and actual in numeric


__all__ = [
    "CodeIssue",
    "CodeReport",
    "SymbolTable",
    "array_bounds_ok",
    "no_division_by_zero",
    "scope_safe",
    "type_compatible",
    "verify_python_source",
]
