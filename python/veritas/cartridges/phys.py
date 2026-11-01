"""veritas.phys cartridge."""

from __future__ import annotations

from typing import Tuple

# SI base dimensions: L, M, T, I, Theta, N, J
Dimension = Tuple[int, int, int, int, int, int, int]


def add_dim(a: Dimension, b: Dimension) -> Dimension:
    return tuple(a[i] + b[i] for i in range(7))


def sub_dim(a: Dimension, b: Dimension) -> Dimension:
    return tuple(a[i] - b[i] for i in range(7))


def mul_dim(a: Dimension, b: Dimension) -> Dimension:
    return tuple(a[i] + b[i] for i in range(7))


def div_dim(a: Dimension, b: Dimension) -> Dimension:
    return tuple(a[i] - b[i] for i in range(7))


def homogeneous(lhs: Dimension, rhs: Dimension) -> bool:
    return lhs == rhs


ZERO_DIM = (0, 0, 0, 0, 0, 0, 0)
LENGTH = (1, 0, 0, 0, 0, 0, 0)
MASS = (0, 1, 0, 0, 0, 0, 0)
TIME = (0, 0, 1, 0, 0, 0, 0)
VELOCITY = (1, 0, -1, 0, 0, 0, 0)
ACCELERATION = (1, 0, -2, 0, 0, 0, 0)
FORCE = (1, 1, -2, 0, 0, 0, 0)
ENERGY = (2, 1, -2, 0, 0, 0, 0)
