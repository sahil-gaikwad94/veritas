"""``veritas.stat`` — probability axioms, Fréchet bounds and graphical-model checks."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


def probability_simplex(probs: Sequence[float], tol: float = 1e-9) -> bool:
    if not probs:
        return False
    if any(not isinstance(p, (int, float)) or not math.isfinite(p) for p in probs):
        return False
    return all(p >= -tol for p in probs) and abs(sum(probs) - 1.0) <= tol


def frechet_lower(p_a: float, p_b: float) -> float:
    return max(0.0, p_a + p_b - 1.0)


def frechet_upper(p_a: float, p_b: float) -> float:
    return min(p_a, p_b)


def frechet_ok(p_a: float, p_b: float, p_joint: float, tol: float = 1e-9) -> bool:
    return frechet_lower(p_a, p_b) - tol <= p_joint <= frechet_upper(p_a, p_b) + tol


def covariance_psd(eigenvalues: Sequence[float], tol: float = 1e-9) -> bool:
    return all(ev >= -tol for ev in eigenvalues)


def matrix_psd(matrix: Sequence[Sequence[float]], tol: float = 1e-9) -> bool:
    try:
        from .quant import is_positive_semidefinite

        return is_positive_semidefinite(matrix, tol=tol)
    except Exception:
        return False


def bayes_consistency(p_a_given_b: float, p_b: float, p_b_given_a: float, p_a: float, tol: float = 1e-9) -> bool:
    """Check ``P(A|B)P(B) == P(B|A)P(A)`` for a valid joint."""

    values = (p_a_given_b, p_b, p_b_given_a, p_a)
    if any(not math.isfinite(v) or v < -tol or v > 1 + tol for v in values):
        return False
    return abs(p_a_given_b * p_b - p_b_given_a * p_a) <= tol


def joint_table_valid(table: Sequence[Sequence[float]], tol: float = 1e-9) -> bool:
    flat = [float(v) for row in table for v in row]
    return bool(flat) and probability_simplex(flat, tol)


def _parents(graph: Mapping[str, Iterable[str]]) -> Dict[str, Set[str]]:
    parents: Dict[str, Set[str]] = {node: set() for node in graph}
    for child, deps in graph.items():
        for parent in deps:
            parents.setdefault(child, set()).add(parent)
            parents.setdefault(parent, set())
    return parents


def _ancestors(parents: Dict[str, Set[str]], seeds: Set[str]) -> Set[str]:
    found: Set[str] = set()
    stack = list(seeds)
    while stack:
        node = stack.pop()
        if node in found:
            continue
        found.add(node)
        stack.extend(parents.get(node, set()))
    return found


def is_d_separated(
    graph: Mapping[str, Iterable[str]],
    x: Iterable[str] | str,
    y: Iterable[str] | str,
    z: Iterable[str] | str = (),
) -> bool:
    """d-separation via the ancestral moral graph.

    ``graph`` maps each node to its parents (a DAG).  Returns ``True`` when every path
    between ``x`` and ``y`` is blocked by ``z``.
    """

    def _nodes(value):
        return {value} if isinstance(value, str) else set(value)

    xs, ys, zs = _nodes(x), _nodes(y), _nodes(z)
    parents = _parents(graph)
    nodes = set(parents)
    if not (xs | ys | zs).issubset(nodes):
        return False
    ancestral = _ancestors(parents, xs | ys | zs)
    adjacency: Dict[str, Set[str]] = {node: set() for node in ancestral}
    for child in ancestral:
        ps = [p for p in parents.get(child, set()) if p in ancestral]
        for parent in ps:
            adjacency[child].add(parent)
            adjacency[parent].add(child)
        # moralize: connect co-parents
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                adjacency[ps[i]].add(ps[j])
                adjacency[ps[j]].add(ps[i])
    for node in zs:
        adjacency.pop(node, None)
    starts = [node for node in xs if node in adjacency]
    seen = set(starts)
    stack = list(starts)
    while stack:
        node = stack.pop()
        if node in ys:
            return False
        for neighbour in adjacency.get(node, set()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return True


def mutually_exclusive(p_joint: float, tol: float = 1e-12) -> bool:
    return abs(p_joint) <= tol


def independence_implies_joint(p_a: float, p_b: float, p_joint: float, tol: float = 1e-9) -> bool:
    return abs(p_joint - p_a * p_b) <= tol


__all__ = [
    "bayes_consistency",
    "covariance_psd",
    "frechet_lower",
    "frechet_ok",
    "frechet_upper",
    "independence_implies_joint",
    "is_d_separated",
    "joint_table_valid",
    "matrix_psd",
    "mutually_exclusive",
    "probability_simplex",
]
