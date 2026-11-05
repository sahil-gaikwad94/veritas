"""``vexx.phys`` — dimensional analysis with an AST checker.

Every quantity carries rational exponents over the seven SI base dimensions.  Addition and
equality demand equal dimensions; multiplication adds exponents; transcendental functions
require dimensionless arguments.  A small expression parser checks whole equations without
evaluating them numerically, which is exactly the invariant ``vt.phys`` promises.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

# SI base dimensions: L, M, T, I, Theta, N, J
Dimension = tuple[Fraction, ...]
ZERO_DIM: Dimension = tuple(Fraction(0) for _ in range(7))


def _dim(*values: int | Fraction) -> Dimension:
    return tuple(Fraction(v) for v in values)


LENGTH = _dim(1, 0, 0, 0, 0, 0, 0)
MASS = _dim(0, 1, 0, 0, 0, 0, 0)
TIME = _dim(0, 0, 1, 0, 0, 0, 0)
CURRENT = _dim(0, 0, 0, 1, 0, 0, 0)
TEMPERATURE = _dim(0, 0, 0, 0, 1, 0, 0)
AMOUNT = _dim(0, 0, 0, 0, 0, 1, 0)
LUMINOUS = _dim(0, 0, 0, 0, 0, 0, 1)

VELOCITY = _dim(1, 0, -1, 0, 0, 0, 0)
ACCELERATION = _dim(1, 0, -2, 0, 0, 0, 0)
FORCE = _dim(1, 1, -2, 0, 0, 0, 0)
ENERGY = _dim(2, 1, -2, 0, 0, 0, 0)
POWER = _dim(2, 1, -3, 0, 0, 0, 0)
PRESSURE = _dim(-1, 1, -2, 0, 0, 0, 0)
MOMENTUM = _dim(1, 1, -1, 0, 0, 0, 0)


def add_dim(a: Dimension, b: Dimension) -> Dimension:
    return tuple(x + y for x, y in zip(a, b))


def sub_dim(a: Dimension, b: Dimension) -> Dimension:
    return tuple(x - y for x, y in zip(a, b))


def mul_dim(a: Dimension, b: Dimension) -> Dimension:
    return add_dim(a, b)


def div_dim(a: Dimension, b: Dimension) -> Dimension:
    return sub_dim(a, b)


def pow_dim(a: Dimension, exponent: int | float | Fraction) -> Dimension:
    factor = Fraction(exponent).limit_denominator(1000)
    return tuple(x * factor for x in a)


def is_dimensionless(a: Dimension) -> bool:
    return all(x == 0 for x in a)


def homogeneous(lhs: Dimension, rhs: Dimension) -> bool:
    return tuple(Fraction(x) for x in lhs) == tuple(Fraction(x) for x in rhs)


def dim_to_str(dim: Dimension) -> str:
    names = "LMTIΘNJ"
    parts = []
    for name, exponent in zip(names, dim):
        if exponent == 0:
            continue
        parts.append(name if exponent == 1 else f"{name}^{exponent}")
    return "·".join(parts) if parts else "1"


@dataclass(frozen=True)
class Dim:
    exponents: Dimension = ZERO_DIM

    def __mul__(self, other: Dim) -> Dim:
        return Dim(add_dim(self.exponents, other.exponents))

    def __truediv__(self, other: Dim) -> Dim:
        return Dim(sub_dim(self.exponents, other.exponents))

    def __pow__(self, exponent: int | float | Fraction) -> Dim:
        return Dim(pow_dim(self.exponents, exponent))

    @property
    def dimensionless(self) -> bool:
        return is_dimensionless(self.exponents)

    def __repr__(self) -> str:
        return f"Dim({dim_to_str(self.exponents)})"


@dataclass
class Quantity:
    value: float
    dim: Dim = Dim()

    def __mul__(self, other: Quantity) -> Quantity:
        return Quantity(self.value * other.value, self.dim * other.dim)

    def __truediv__(self, other: Quantity) -> Quantity:
        return Quantity(self.value / other.value, self.dim / other.dim)

    def __add__(self, other: Quantity) -> Quantity:
        if self.dim != other.dim:
            raise ValueError(f"cannot add {self.dim!r} and {other.dim!r}")
        return Quantity(self.value + other.value, self.dim)

    def __sub__(self, other: Quantity) -> Quantity:
        if self.dim != other.dim:
            raise ValueError(f"cannot subtract {self.dim!r} and {other.dim!r}")
        return Quantity(self.value - other.value, self.dim)


class DimensionError(ValueError):
    pass


def _dim_of(node: ast.AST, symbols: Mapping[str, Dimension]) -> Dimension:
    if isinstance(node, ast.Expression):
        return _dim_of(node.body, symbols)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise DimensionError(f"non-numeric constant {node.value!r}")
        return ZERO_DIM
    if isinstance(node, ast.Name):
        if node.id not in symbols:
            raise DimensionError(f"unknown symbol {node.id!r}")
        return symbols[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _dim_of(node.operand, symbols)
    if isinstance(node, ast.BinOp):
        left = _dim_of(node.left, symbols)
        right = _dim_of(node.right, symbols)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if not homogeneous(left, right):
                raise DimensionError(
                    f"incompatible dimensions in +/-: {dim_to_str(left)} vs {dim_to_str(right)}"
                )
            return left
        if isinstance(node.op, ast.Mult):
            return add_dim(left, right)
        if isinstance(node.op, ast.Div):
            return sub_dim(left, right)
        if isinstance(node.op, ast.Pow):
            exponent = _constant_exponent(node.right)
            return pow_dim(left, exponent)
        raise DimensionError(f"unsupported operator {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise DimensionError("unsupported call target")
        name = node.func.id
        if len(node.args) != 1:
            raise DimensionError(f"{name}() expects exactly one argument")
        inner = _dim_of(node.args[0], symbols)
        if name == "sqrt":
            return pow_dim(inner, Fraction(1, 2))
        if name in {"sin", "cos", "tan", "exp", "log", "log10"}:
            if not is_dimensionless(inner):
                raise DimensionError(f"{name}() requires a dimensionless argument, got {dim_to_str(inner)}")
            return ZERO_DIM
        if name == "abs":
            return inner
        raise DimensionError(f"unsupported function {name}()")
    raise DimensionError(f"unsupported syntax {type(node).__name__}")


def _constant_exponent(node: ast.AST) -> Fraction:
    try:
        value = ast.literal_eval(node)
    except Exception as exc:  # pragma: no cover - defensive
        raise DimensionError("exponent must be a numeric literal") from exc
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DimensionError("exponent must be numeric")
    return Fraction(value).limit_denominator(1000)


def dimension_of_expression(expression: str, symbols: Mapping[str, Dimension]) -> Dim:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise DimensionError(f"invalid expression: {exc}") from exc
    return Dim(_dim_of(tree, symbols))


def check_equation(lhs: str, rhs: str, symbols: Mapping[str, Dimension]) -> tuple[bool, str]:
    try:
        left = dimension_of_expression(lhs, symbols)
        right = dimension_of_expression(rhs, symbols)
    except DimensionError as exc:
        return False, str(exc)
    if left != right:
        return False, f"dimension mismatch: LHS is {dim_to_str(left.exponents)}, RHS is {dim_to_str(right.exponents)}"
    return True, f"dimensionally consistent: {dim_to_str(left.exponents)}"


def check_expression(expression: str, symbols: Mapping[str, Dimension]) -> tuple[bool, str]:
    try:
        dim = dimension_of_expression(expression, symbols)
    except DimensionError as exc:
        return False, str(exc)
    return True, dim_to_str(dim.exponents)


__all__ = [
    "ACCELERATION",
    "AMOUNT",
    "CURRENT",
    "Dim",
    "Dimension",
    "DimensionError",
    "ENERGY",
    "FORCE",
    "LENGTH",
    "LUMINOUS",
    "MASS",
    "MOMENTUM",
    "POWER",
    "PRESSURE",
    "Quantity",
    "TEMPERATURE",
    "TIME",
    "VELOCITY",
    "ZERO_DIM",
    "add_dim",
    "check_equation",
    "check_expression",
    "dim_to_str",
    "dimension_of_expression",
    "div_dim",
    "homogeneous",
    "is_dimensionless",
    "mul_dim",
    "pow_dim",
    "sub_dim",
]
