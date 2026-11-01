"""veritas.quant cartridge."""

from __future__ import annotations

import math
from typing import Any, Dict, List

def portfolio_weights_valid(weights: Dict[str, float], allowed=None, lo=0.05, hi=0.50, total=1.0, tol=1e-6) -> bool:
    if allowed is not None and not set(weights).issubset(set(allowed)):
        return False
    vals = list(weights.values())
    if abs(sum(vals) - total) > tol:
        return False
    return all(lo - tol <= w <= hi + tol for w in vals)

def quad_form(weights: Dict[str, float], cov: Dict) -> float:
    # reuse veritas quad_form if available
    try:
        from veritas.decorators import quad_form as _qf
        return _qf(weights, cov)
    except Exception:
        keys = list(weights.keys())
        vec = [weights[k] for k in keys]
        n = len(keys)
        mat = [[0.0]*n for _ in range(n)]
        for (a,b),v in cov.items():
            if a in keys and b in keys:
                i=keys.index(a); j=keys.index(b)
                mat[i][j]=float(v)
        total=0.0
        for i in range(n):
            for j in range(n):
                total += vec[i]*mat[i][j]*vec[j]
        return total

def sharpe_ratio(expected_return: float, volatility: float, rf: float = 0.04) -> float:
    if volatility == 0:
        return float("inf")
    return (expected_return - rf) / volatility

def black_scholes_call(S, K, r, sigma, T) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    def N(x):
        return 0.5*(1+math.erf(x/math.sqrt(2)))
    return S*N(d1) - K*math.exp(-r*T)*N(d2)

def gaap_balance(assets, liabilities, equity, tol=1e-6) -> bool:
    return abs(assets - (liabilities + equity)) <= tol
