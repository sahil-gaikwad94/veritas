"""``veritas.quant`` — portfolio mathematics, accounting identities and option bounds.

All helpers are deterministic and dependency-free (NumPy is optional).  They are the
concrete evaluators used by schemas that mix ``vt.derive`` (expected return, volatility,
Sharpe) with ``vt.require`` (simplex, concentration limits, no-arbitrage bounds).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# covariance / quadratic forms
# ---------------------------------------------------------------------------

def covariance_labels(cov: Mapping[Any, float]) -> list[Any]:
    labels: list[Any] = []
    for key in cov:
        first = key[0] if isinstance(key, tuple) and len(key) == 2 else key
        if first not in labels:
            labels.append(first)
    return labels


def covariance_matrix(labels: Sequence[Any], cov: Mapping[Any, float]) -> list[list[float]]:
    """Build a symmetric matrix over ``labels`` from a triangular covariance map."""

    index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0.0] * n for _ in range(n)]
    for key, value in cov.items():
        a, b = key if isinstance(key, tuple) and len(key) == 2 else (key, key)
        if a not in index or b not in index:
            continue
        i, j = index[a], index[b]
        matrix[i][j] = float(value)
        if i != j:
            matrix[j][i] = float(value)
    return matrix


def quad_form(weights: Mapping[str, float], cov: Mapping[Any, float]) -> float:
    """``wᵀΣw`` with subset support and symmetrization (consistent with ``vt.quad_form``)."""

    labels = list(weights.keys())
    vector = [float(weights[k]) for k in labels]
    matrix = covariance_matrix(labels, cov)
    total = 0.0
    for i in range(len(vector)):
        for j in range(len(vector)):
            total += vector[i] * matrix[i][j] * vector[j]
    return total


def cholesky(matrix: Sequence[Sequence[float]], jitter: float = 1e-12) -> list[list[float]] | None:
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                value = matrix[i][i] - total
                # Tikhonov jitter for near-PSD matrices
                if value <= 0:
                    if value > -jitter:
                        value = jitter
                    else:
                        return None
                lower[i][j] = math.sqrt(value)
            else:
                if lower[j][j] == 0:
                    return None
                lower[i][j] = (matrix[i][j] - total) / lower[j][j]
    return lower


def cholesky_with_jitter(matrix: Sequence[Sequence[float]], jitter: float = 1e-12) -> list[list[float]] | None:
    """Cholesky with Tikhonov jitter for near-PSD matrices."""
    return cholesky(matrix, jitter)


def eigendecomposition_split(matrix: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """Split symmetric matrix into positive and negative semidefinite parts via eigendecomposition.
    
    Returns (Q_plus, Q_minus) where matrix = Q_plus - Q_minus, both PSD.
    Uses power iteration for simplicity (for small matrices in contractor).
    """
    import numpy as np
    a = np.array(matrix, dtype=np.float64)
    if a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square")
    # Ensure symmetric
    a = (a + a.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    # Split into positive and negative parts
    pos_evals = np.maximum(eigenvalues, 0)
    neg_evals = np.maximum(-eigenvalues, 0)
    q_plus = eigenvectors @ np.diag(pos_evals) @ eigenvectors.T
    q_minus = eigenvectors @ np.diag(neg_evals) @ eigenvectors.T
    return q_plus.tolist(), q_minus.tolist()


def is_positive_semidefinite(matrix: Sequence[Sequence[float]], tol: float = 1e-9) -> bool:
    """Symmetric PSD test via an LDL^T-style factorization (no ridge laundering).

    Singular PSD matrices are accepted; a negative pivot beyond ``tol`` is rejected.
    """
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        return False
    try:
        a = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
    except (TypeError, ValueError):
        return False
    if any(not math.isfinite(x) for row in a for x in row):
        return False
    for i in range(n):
        for j in range(i):
            if abs(a[i][j] - a[j][i]) > tol:
                return False
    lower = [[0.0] * n for _ in range(n)]
    diag = [0.0] * n
    for i in range(n):
        value = a[i][i] - sum(lower[i][k] * lower[i][k] * diag[k] for k in range(i))
        if value < -tol:
            return False
        if abs(value) <= tol:
            # Zero pivot: the remaining Schur-complement row must also be zero.
            for j in range(i + 1, n):
                residual = a[j][i] - sum(lower[j][k] * lower[i][k] * diag[k] for k in range(i))
                if abs(residual) > tol:
                    return False
            diag[i] = 0.0
            continue
        diag[i] = value
        for j in range(i + 1, n):
            residual = a[j][i] - sum(lower[j][k] * lower[i][k] * diag[k] for k in range(i))
            lower[j][i] = residual / value
    return True


# ---------------------------------------------------------------------------
# portfolio statistics
# ---------------------------------------------------------------------------

@dataclass
class PortfolioStats:
    expected_return: float
    volatility: float
    sharpe_ratio: float
    variance: float
    risk_free: float

    def as_dict(self) -> dict[str, float]:
        return {
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "variance": self.variance,
            "risk_free": self.risk_free,
        }


def portfolio_weights_valid(
    weights: Mapping[str, float],
    allowed: Iterable[str] | None = None,
    lo: float = 0.05,
    hi: float = 0.50,
    total: float = 1.0,
    tol: float = 1e-6,
) -> bool:
    if not isinstance(weights, Mapping) or not weights:
        return False
    values = list(weights.values())
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return False
    if allowed is not None and not set(weights).issubset(set(allowed)):
        return False
    if abs(sum(values) - total) > tol:
        return False
    return all(lo - tol <= v <= hi + tol for v in values)


def expected_return(weights: Mapping[str, float], returns: Mapping[str, float]) -> float:
    return sum(float(weights[key]) * float(returns[key]) for key in weights if key in returns)


def portfolio_stats(
    weights: Mapping[str, float],
    returns: Mapping[str, float],
    cov: Mapping[Any, float],
    risk_free: float = 0.04,
) -> PortfolioStats:
    variance = quad_form(weights, cov)
    if variance < -1e-12:
        raise ValueError(f"negative variance {variance!r}: covariance is not PSD")
    volatility = math.sqrt(max(0.0, variance))
    expected = expected_return(weights, returns)
    sharpe = float("inf") if volatility == 0 else (expected - risk_free) / volatility
    return PortfolioStats(expected, volatility, sharpe, variance, risk_free)


def concentration_ok(weights: Mapping[str, float], max_weight: float, tol: float = 1e-9) -> bool:
    return all(float(w) <= max_weight + tol for w in weights.values())


def long_only_ok(weights: Mapping[str, float], tol: float = 1e-12) -> bool:
    return all(float(w) >= -tol for w in weights.values())


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def black_scholes_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def implied_volatility(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    *,
    is_call: bool = True,
    tol: float = 1e-10,
    max_iterations: int = 200,
) -> float | None:
    """Bisection on a monotone Black–Scholes price; ``None`` if no root exists."""

    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    upper_bound = S if is_call else K * math.exp(-r * T)
    if price < intrinsic - 1e-12 or price > upper_bound + 1e-12:
        return None
    fn = black_scholes_call if is_call else black_scholes_put
    lo, hi = 1e-8, 10.0
    if fn(S, K, r, hi, T) < price:
        return None
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        value = fn(S, K, r, mid, T)
        if abs(value - price) <= tol:
            return mid
        if value < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def no_arbitrage_bounds(price: float, S: float, K: float, r: float, T: float, *, is_call: bool = True) -> bool:
    lower = max(0.0, (S - K * math.exp(-r * T)) if is_call else (K * math.exp(-r * T) - S))
    upper = S if is_call else K * math.exp(-r * T)
    return lower - 1e-9 <= price <= upper + 1e-9


# ---------------------------------------------------------------------------
# accounting identities
# ---------------------------------------------------------------------------

def gaap_balance(assets: float, liabilities: float, equity: float, tol: float = 1e-6) -> bool:
    return abs(assets - (liabilities + equity)) <= tol


def cash_flow_reconciles(
    opening: float,
    operating: float,
    investing: float,
    financing: float,
    closing: float,
    tol: float = 1e-6,
) -> bool:
    return abs(opening + operating + investing + financing - closing) <= tol


def double_entry_balanced(debits: Sequence[float], credits: Sequence[float], tol: float = 1e-6) -> bool:
    return abs(sum(debits) - sum(credits)) <= tol


# legacy alias used by the handbook
def sharpe_ratio(expected_return, volatility, rf=0.04):
    return float("inf") if volatility == 0 else (expected_return - rf) / volatility


__all__ = [
    "PortfolioStats",
    "black_scholes_call",
    "black_scholes_put",
    "cash_flow_reconciles",
    "cholesky",
    "concentration_ok",
    "covariance_labels",
    "covariance_matrix",
    "double_entry_balanced",
    "expected_return",
    "gaap_balance",
    "implied_volatility",
    "is_positive_semidefinite",
    "long_only_ok",
    "no_arbitrage_bounds",
    "portfolio_stats",
    "portfolio_weights_valid",
    "quad_form",
    "sharpe_ratio",
]
