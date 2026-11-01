"""veritas.stat cartridge."""

from __future__ import annotations

from typing import List


def probability_simplex(probs: List[float], tol: float = 1e-9) -> bool:
    return all(p >= -tol for p in probs) and abs(sum(probs) - 1.0) <= tol


def frechet_lower(p_a: float, p_b: float) -> float:
    return max(0.0, p_a + p_b - 1.0)


def frechet_upper(p_a: float, p_b: float) -> float:
    return min(p_a, p_b)


def covariance_psd(eigenvalues: List[float]) -> bool:
    return all(ev >= -1e-9 for ev in eigenvalues)
