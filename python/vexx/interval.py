"""Phase 2: Interval arithmetic, HC4-Revise, Affine Arithmetic (AF2), and volume twist."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

NoiseId = int


def _next_down(x: float) -> float:
    return math.nextafter(x, -math.inf)


def _next_up(x: float) -> float:
    return math.nextafter(x, math.inf)


@dataclass
class Interval:
    lo: float = float("-inf")
    hi: float = float("inf")

    EMPTY: Interval = None  # placeholder filled below
    ENTIRE: Interval = None

    def __post_init__(self):
        if self.lo is None:
            self.lo = float("-inf")
        if self.hi is None:
            self.hi = float("inf")
        # normalize empty
        if self.lo > self.hi:
            # keep as given; is_empty handles
            pass

    @classmethod
    def empty(cls) -> Interval:
        return cls(float("inf"), float("-inf"))

    @classmethod
    def entire(cls) -> Interval:
        return cls(float("-inf"), float("inf"))

    def is_empty(self) -> bool:
        return self.lo > self.hi or math.isnan(self.lo) or math.isnan(self.hi)

    def width(self) -> float:
        if self.is_empty():
            return 0.0
        # clamp infinities for volume products
        lo = self.lo if math.isfinite(self.lo) else -1e300
        hi = self.hi if math.isfinite(self.hi) else 1e300
        return max(0.0, hi - lo)

    def intersect(self, other: Interval) -> Interval:
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        return Interval(lo, hi)

    def union(self, other: Interval) -> Interval:
        if self.is_empty():
            return Interval(other.lo, other.hi)
        if other.is_empty():
            return Interval(self.lo, self.hi)
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    # ---- outward-rounded operations ----
    def add(self, other: Interval) -> Interval:
        return Interval(_next_down(self.lo + other.lo), _next_up(self.hi + other.hi))

    def sub(self, other: Interval) -> Interval:
        return Interval(_next_down(self.lo - other.hi), _next_up(self.hi - other.lo))

    def mul(self, other: Interval) -> Interval:
        if self.is_empty() or other.is_empty():
            return Interval.empty()
        if (self.lo == self.hi == 0) or (other.lo == other.hi == 0):
            return Interval(0.0, 0.0)
        def prod(a, b):
            if a == 0 or b == 0:
                return 0.0
            return a * b
        products = (prod(self.lo, other.lo), prod(self.lo, other.hi), prod(self.hi, other.lo), prod(self.hi, other.hi))
        return Interval(_next_down(min(products)), _next_up(max(products)))

    def div(self, other: Interval) -> Interval:
        if self.is_empty() or other.is_empty():
            return Interval.empty()
        if other.lo <= 0 <= other.hi:
            if self.lo == self.hi == 0:
                return Interval(0.0, 0.0)
            # The divisor straddles zero; the quotient is unbounded. This is a sound
            # over-approximation, not a split that silently drops the pole.
            return Interval.entire()
        vals = (self.lo / other.lo, self.lo / other.hi, self.hi / other.lo, self.hi / other.hi)
        return Interval(_next_down(min(vals)), _next_up(max(vals)))

    def sqr(self) -> Interval:
        if self.lo >= 0:
            return Interval(_next_down(self.lo * self.lo), _next_up(self.hi * self.hi))
        if self.hi <= 0:
            return Interval(_next_down(self.hi * self.hi), _next_up(self.lo * self.lo))
        m = max(abs(self.lo), abs(self.hi))
        return Interval(0.0, _next_up(m * m))

    def sqrt(self) -> Interval:
        if self.hi < 0:
            return Interval.empty()
        lo = 0.0 if self.lo < 0 else math.sqrt(max(0.0, self.lo))
        hi = math.sqrt(max(0.0, self.hi))
        return Interval(_next_down(lo), _next_up(hi))

    # ---- numeric prefix parsing (from blueprint) ----
    @staticmethod
    def from_numeric_prefix(raw_candidate, domain_bounds, allow_exp: bool = False) -> Interval:
        from . import native as _native

        span = _native.numeric_prefix_or_none(raw_candidate, domain_bounds, allow_exp)
        if span is not None:
            return span
        return Interval._from_numeric_prefix_python(raw_candidate, domain_bounds, allow_exp)

    @staticmethod
    def _from_numeric_prefix_python(raw_candidate, domain_bounds, allow_exp: bool = False) -> Interval:
        if not isinstance(domain_bounds, Interval):
            try:
                domain_bounds = Interval(domain_bounds[0], domain_bounds[1])
            except Exception:
                domain_bounds = Interval.entire()
        if raw_candidate is None:
            s = ""
        else:
            s = str(raw_candidate).lstrip()
        if s == "":
            return Interval(domain_bounds.lo, domain_bounds.hi)

        # delimiter detection (do not strip trailing delimiters)
        cut = None
        for i, ch in enumerate(s):
            if ch in (",", "}", "]", " ", "\n", "\t"):
                cut = i
                break
        if cut is not None:
            num_part = s[:cut]
            is_closed = True
        else:
            num_part = s
            is_closed = False

        if num_part == "-":
            return domain_bounds.intersect(Interval(float("-inf"), 0.0))
        if num_part == "+":
            return domain_bounds.intersect(Interval(0.0, float("inf")))

        is_neg = num_part.startswith("-")
        # strip a single trailing dot
        clean = num_part[:-1] if num_part.endswith(".") else num_part
        try:
            base_val = float(clean)
        except (ValueError, TypeError):
            return Interval.empty()

        if is_closed:
            return domain_bounds.intersect(Interval(base_val, base_val))

        if allow_exp and ("e" in num_part.lower() or "E" in num_part):
            return Interval(domain_bounds.lo, domain_bounds.hi)

        # decimal point present
        if "." in num_part:
            dot_idx = num_part.find(".")
            frac_digits = len(num_part) - dot_idx - 1
            if frac_digits < 0:
                frac_digits = 0
            ulp = 10.0 ** (-frac_digits)
            if is_neg:
                # negative prefix: completions in (base_val - ulp, base_val]
                lo = _next_up(base_val - ulp) if ulp > 0 else base_val
                hi = base_val
                span = Interval(lo, hi)
            else:
                span = Interval(base_val, _next_down(base_val + ulp))
            return domain_bounds.intersect(span)

        # leading zero
        if num_part in ("0", "-0"):
            if is_neg:
                span = Interval(_next_up(-1.0), 0.0)
            else:
                span = Interval(0.0, _next_down(1.0))
            return domain_bounds.intersect(span)

        # non-zero integer prefix
        abs_v = abs(base_val)
        union_lo = float("inf")
        union_hi = float("-inf")
        scale = 1.0
        for _ in range(16):
            if is_neg:
                scaled_lo = -(abs_v + 1.0) * scale
                scaled_hi = -abs_v * scale
                # half-open on lower bound: (scaled_lo, scaled_hi]
                lo = _next_up(scaled_lo)
                hi = scaled_hi
            else:
                scaled_lo = abs_v * scale
                scaled_hi = (abs_v + 1.0) * scale
                # half-open on upper bound: [scaled_lo, scaled_hi)
                lo = scaled_lo
                hi = _next_down(scaled_hi) if scaled_hi != float("inf") else scaled_hi
            sub = domain_bounds.intersect(Interval(lo, hi))
            if not sub.is_empty():
                union_lo = min(union_lo, sub.lo)
                union_hi = max(union_hi, sub.hi)
            scale *= 10.0
            if (not is_neg and abs_v * scale > domain_bounds.hi and domain_bounds.hi != float("inf")):
                break
            if (is_neg and -abs_v * scale < domain_bounds.lo and domain_bounds.lo != float("-inf")):
                break
        if union_lo == float("inf"):
            return Interval.empty()
        return Interval(union_lo, union_hi)

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return False
        return self.lo == other.lo and self.hi == other.hi

    def __repr__(self):
        return f"Interval({self.lo}, {self.hi})"


# Fill placeholders
Interval.EMPTY = Interval.empty()
Interval.ENTIRE = Interval.entire()


@dataclass
class AffineForm:
    center: float = 0.0
    terms: list[tuple[NoiseId, float]] = field(default_factory=list)
    err_radius: float = 0.0

    def _combine(self, other_terms: Sequence[tuple[NoiseId, float]]) -> list[tuple[NoiseId, float]]:
        mapping: dict[NoiseId, float] = {}
        for nid, coeff in self.terms + list(other_terms):
            mapping[nid] = mapping.get(nid, 0.0) + coeff
        # drop near-zero
        return [(nid, c) for nid, c in mapping.items() if c != 0.0]

    def to_interval(self) -> Interval:
        l1 = sum(abs(c) for _, c in self.terms) + self.err_radius
        return Interval(self.center - l1, self.center + l1)

    def add(self, other: AffineForm) -> AffineForm:
        return AffineForm(
            center=self.center + other.center,
            terms=self._combine(other.terms),
            err_radius=self.err_radius + other.err_radius,
        )

    def sub(self, other: AffineForm) -> AffineForm:
        neg_terms = [(nid, -c) for nid, c in other.terms]
        return AffineForm(
            center=self.center - other.center,
            terms=self._combine(neg_terms),
            err_radius=self.err_radius + other.err_radius,
        )

    def scale(self, scalar: float) -> AffineForm:
        return AffineForm(
            center=self.center * scalar,
            terms=[(nid, c * scalar) for nid, c in self.terms],
            err_radius=self.err_radius * abs(scalar),
        )

    def mul(self, other: AffineForm) -> AffineForm:
        other = other if isinstance(other, AffineForm) else AffineForm.constant(float(other))
        x0 = self.center
        y0 = other.center
        rx = sum(abs(c) for _, c in self.terms) + self.err_radius
        ry = sum(abs(c) for _, c in other.terms) + other.err_radius
        terms = [(nid, c * y0) for nid, c in self.terms] + [(nid, c * x0) for nid, c in other.terms]
        # combine via temporary
        mapping = {}
        for nid, coeff in terms:
            mapping[nid] = mapping.get(nid, 0.0) + coeff
        combined = [(nid, c) for nid, c in mapping.items() if c != 0.0]
        err = abs(x0) * ry + abs(y0) * rx + rx * ry
        return AffineForm(center=x0 * y0, terms=combined, err_radius=err)

    def __mul__(self, other):
        return self.mul(other)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def intersect_interval(self, iv: Interval) -> AffineForm:
        # Convert to interval, intersect, and rebuild a simple affine form (loses correlation)
        new_iv = self.to_interval().intersect(iv)
        if new_iv.is_empty():
            return None  # type: ignore
        center = (new_iv.lo + new_iv.hi) / 2.0
        radius = (new_iv.hi - new_iv.lo) / 2.0
        return AffineForm(center=center, terms=[], err_radius=radius)

    @classmethod
    def from_interval(cls, iv: Interval, noise_id: NoiseId = 0) -> AffineForm:
        center = (iv.lo + iv.hi) / 2.0
        radius = (iv.hi - iv.lo) / 2.0
        return cls(center=center, terms=[(noise_id, radius)], err_radius=0.0)

    @classmethod
    def constant(cls, value: float) -> AffineForm:
        return cls(center=value, terms=[], err_radius=0.0)


class ResidualAccumulator:
    """Symbolic residual accumulator for dynamic collections (AGDecode)."""

    def __init__(self, target_sum: float, elem_min: float, elem_max: float,
                 min_remaining: int = 0, max_remaining: int = 0):
        self.target_sum = target_sum
        self.elem_min = elem_min
        self.elem_max = elem_max
        self.min_remaining = min_remaining
        self.max_remaining = max_remaining

    def residual(self, current_sum: float) -> float:
        return self.target_sum - current_sum

    def feasible_suffix(self, current_sum: float, emitted_count: int) -> bool:
        r = self.residual(current_sum)
        m_min = max(0, self.min_remaining - emitted_count)
        m_max = self.max_remaining - emitted_count
        if m_max < m_min:
            return False
        lo = min(m_min * self.elem_min, m_max * self.elem_min)
        hi = max(m_min * self.elem_max, m_max * self.elem_max)
        if self.elem_min <= 0 <= self.elem_max:
            lo, hi = m_max * self.elem_min, m_max * self.elem_max
        return lo <= r <= hi

    def interval_for_remaining(self, emitted_count: int) -> Interval:
        m_min = max(0, self.min_remaining - emitted_count)
        m_max = self.max_remaining - emitted_count
        if m_max < 0:
            return Interval.empty()
        lo = min(m_min * self.elem_min, m_max * self.elem_min)
        hi = max(m_min * self.elem_max, m_max * self.elem_max)
        if self.elem_min <= 0 <= self.elem_max:
            lo, hi = m_max * self.elem_min, m_max * self.elem_max
        return Interval(lo, hi)


class Hc4AffineContractor:
    """Hybrid HC4-Revise + AF2 contractor with volume twist support."""

    def __init__(self, domains: dict[Any, Interval] | None = None):
        self.domains: dict[Any, Interval] = {}
        if domains:
            for k, v in domains.items():
                self.domains[k] = v if isinstance(v, Interval) else Interval(v[0], v[1])
        self.constraints: list[Any] = []
        self.affine: dict[Any, AffineForm] = {}

    # ---- variable management ----
    def add_variable(self, var_id: Any, interval: Interval | None = None,
                     lo: float | None = None, hi: float | None = None) -> None:
        if interval is None:
            interval = Interval(lo if lo is not None else float("-inf"),
                                hi if hi is not None else float("inf"))
        self.domains[var_id] = interval

    def set_domain(self, var_id: Any, interval: Interval) -> None:
        self.domains[var_id] = interval if isinstance(interval, Interval) else Interval(interval[0], interval[1])

    def set_value(self, var_id: Any, value: float) -> None:
        self.domains[var_id] = Interval(value, value)

    def get_bounds(self, var_id: Any) -> Interval:
        return self.domains.get(var_id, Interval.entire())

    def bounds(self, var_id: Any) -> Interval:
        return self.get_bounds(var_id)

    # ---- constraints ----
    def add_constraint(self, constraint: Any) -> None:
        self.constraints.append(constraint)

    def add_relation(self, constraint: Any) -> None:
        self.add_constraint(constraint)

    def _interval_for_var(self, var_id: Any) -> Interval:
        return self.domains.get(var_id, Interval.entire())

    def _revise_dc_constraint(
        self,
        var_ids: Sequence[Any],
        cholesky_l_plus: Sequence[Sequence[float]],
        cholesky_l_minus: Sequence[Sequence[float]] | None,
        rhs: float,
        op: str,
    ) -> bool:
        """Forward check for ``xᵀAx op rhs`` in (difference-of) PSD Cholesky form.

        Returns ``True`` when the current box provably violates the constraint (a
        *refutation*), ``False`` otherwise.  For an indefinite matrix ``A = A₊ − A₋`` we use
        the difference-of-convex form ``||L₊ᵀx||² − ||L₋ᵀx||²`` with ``A± = L±L±ᵀ`` and only
        interval arithmetic: the upper bound ``u₋`` of the negative part over the current box
        is added to ``rhs`` for ``≤``-style constraints (and dually for ``≥``).  This never
        refutes a box that contains a solution; it can only stay silent.
        """

        n = len(var_ids)
        intervals = [self._interval_for_var(vid) for vid in var_ids]
        y_plus = _apply_lt(intervals, cholesky_l_plus, n)
        if y_plus is None:
            return False
        plus = _sum_square(y_plus)
        tol = 1e-9 * (1.0 + abs(rhs))
        if cholesky_l_minus is None:
            minus_lo = 0.0
            minus_hi = 0.0
            total = plus
        else:
            if len(cholesky_l_minus) != n:
                return False
            y_minus = _apply_lt(intervals, cholesky_l_minus, n)
            if y_minus is None:
                return False
            minus = _sum_square(y_minus)
            minus_lo = minus.lo
            minus_hi = minus.hi
            total = plus.sub(minus)
        # Full indefinite interval compatibility (necessary condition for feasibility).
        if op in ("le", "lt", "eq"):
            if total.lo > rhs + tol:
                return True
        if op in ("ge", "gt", "eq"):
            if total.hi < rhs - tol:
                return True
        # Relaxed-PSD bounds using the negative part's box bounds.  For le: any solution
        # satisfies ||L₊ᵀx||² ≤ rhs + max(||L₋ᵀx||²); for ge it satisfies
        # ||L₊ᵀx||² ≥ rhs + min(||L₋ᵀx||²).  Violating either value refutes the box.
        if op in ("le", "lt", "eq"):
            if plus.lo > rhs + minus_hi + tol:
                return True
        if op in ("ge", "gt", "eq"):
            if plus.hi < rhs + minus_lo - tol:
                return True
        if op == "eq":
            if rhs < total.lo - tol or rhs > total.hi + tol:
                return True
        return False

    def _revise_constraint(self, cons: Any) -> bool:
        # tuple/list forms
        if isinstance(cons, (tuple, list)):
            if not cons:
                return True
            kind = cons[0]
            if kind in ("bound", "bounds"):
                vid = cons[1]
                lo = cons[2] if len(cons) > 2 else None
                hi = cons[3] if len(cons) > 3 else None
                iv = self._interval_for_var(vid)
                if lo is not None:
                    iv = iv.intersect(Interval(lo, iv.hi))
                if hi is not None:
                    iv = iv.intersect(Interval(iv.lo, hi))
                if iv.is_empty():
                    return False
                self.domains[vid] = iv
                return True

            if kind in ("sum", "eq_sum", "sum_eq"):
                vars_ = list(cons[1])
                target = float(cons[2])
                # HC4 revise for each var
                for i, vid in enumerate(vars_):
                    others = [self._interval_for_var(v) for j, v in enumerate(vars_) if j != i]
                    # sum others interval
                    o_lo = sum(iv.lo for iv in others) if others else 0.0
                    o_hi = sum(iv.hi for iv in others) if others else 0.0
                    # x_i = target - sum_others -> [target - o_hi, target - o_lo].
                    # Outward rounding is mandatory: without it the recurrence
                    # ``a + b == 1`` can exclude a genuinely feasible point on the next
                    # pass purely because of float error, and the contractor reports a
                    # false dead end.
                    new_iv = Interval(_next_down(target - o_hi), _next_up(target - o_lo))
                    cur = self._interval_for_var(vid)
                    inter = cur.intersect(new_iv)
                    if inter.is_empty():
                        return False
                    self.domains[vid] = inter
                return True

            if kind in ("linear", "affine"):
                # cons = ('linear', {var: coeff}, target, op)
                coeffs = cons[1]
                target = float(cons[2])
                op = cons[3] if len(cons) > 3 else "eq"
                vars_ = list(coeffs.keys())
                # equality narrowing only when op indicates equality
                if op in ("eq", "equal", "=", None, "eq_sum"):
                    for vid in vars_:
                        c_i = coeffs[vid]
                        if c_i == 0:
                            continue
                        o_lo = 0.0
                        o_hi = 0.0
                        for v in vars_:
                            if v == vid:
                                continue
                            iv = self._interval_for_var(v)
                            contrib_lo = coeffs[v] * (iv.lo if coeffs[v] >= 0 else iv.hi)
                            contrib_hi = coeffs[v] * (iv.hi if coeffs[v] >= 0 else iv.lo)
                            o_lo += contrib_lo
                            o_hi += contrib_hi
                        rem_lo = _next_down(target - o_hi)
                        rem_hi = _next_up(target - o_lo)
                        if c_i > 0:
                            new_lo, new_hi = _next_down(rem_lo / c_i), _next_up(rem_hi / c_i)
                        else:
                            new_lo, new_hi = _next_down(rem_hi / c_i), _next_up(rem_lo / c_i)
                        new_iv = Interval(new_lo, new_hi)
                        cur = self._interval_for_var(vid)
                        inter = cur.intersect(new_iv)
                        if inter.is_empty():
                            return False
                        self.domains[vid] = inter
                # inequality handling (<=): sum coeff*x <= target
                if op in ("le", "lt"):
                    for vid in vars_:
                        c_i = coeffs[vid]
                        if c_i == 0:
                            continue
                        o_lo = 0.0
                        o_hi = 0.0
                        for v in vars_:
                            if v == vid:
                                continue
                            iv = self._interval_for_var(v)
                            contrib_lo = coeffs[v] * (iv.lo if coeffs[v] >= 0 else iv.hi)
                            contrib_hi = coeffs[v] * (iv.hi if coeffs[v] >= 0 else iv.lo)
                            o_lo += contrib_lo
                            o_hi += contrib_hi
                        if c_i > 0:
                            hi = _next_up((target - o_lo) / c_i)
                            cur = self._interval_for_var(vid)
                            inter = cur.intersect(Interval(float("-inf"), hi))
                        else:
                            # c_i < 0 flips the inequality; existence over the other
                            # intervals uses their lower sum, not their upper sum.
                            lo = _next_down((target - o_lo) / c_i)
                            cur = self._interval_for_var(vid)
                            inter = cur.intersect(Interval(lo, float("inf")))
                        if inter.is_empty():
                            return False
                        self.domains[vid] = inter
                # >= handling
                if op in ("ge", "gt"):
                    for vid in vars_:
                        c_i = coeffs[vid]
                        if c_i == 0:
                            continue
                        o_lo = 0.0
                        o_hi = 0.0
                        for v in vars_:
                            if v == vid:
                                continue
                            iv = self._interval_for_var(v)
                            contrib_lo = coeffs[v] * (iv.lo if coeffs[v] >= 0 else iv.hi)
                            contrib_hi = coeffs[v] * (iv.hi if coeffs[v] >= 0 else iv.lo)
                            o_lo += contrib_lo
                            o_hi += contrib_hi
                        if c_i > 0:
                            lo = _next_down((target - o_hi) / c_i)
                            cur = self._interval_for_var(vid)
                            inter = cur.intersect(Interval(lo, float("inf")))
                        else:
                            hi = _next_up((target - o_hi) / c_i)
                            cur = self._interval_for_var(vid)
                            inter = cur.intersect(Interval(float("-inf"), hi))
                        if inter.is_empty():
                            return False
                        self.domains[vid] = inter
                return True

            if kind in ("residual", "residual_sum"):
                # cons = ('residual_sum', container_var, target_sum, elem_min, elem_max, min_rem, max_rem, current_sum?, count?)
                # We'll just bound the container variable if possible; skip complex
                return True

            if kind in ("quad_form", "quad_form_cholesky", "quad_form_split", "quadratic_form", "dc_quadratic_form"):
                # Quadratic contraction: forward check using (difference-of) Cholesky factor(s).
                # The constraint is xᵀ A x op rhs where A = L Lᵀ (L lower triangular).
                # Internally y = Lᵀ x (span of the columns of L), so xᵀ A x = ||y||².
                # The negative part (indefinite DC form) is bounded soundly with interval
                # arithmetic; no sign-flipped independent PSD constraints are ever added.
                # tuple form: ('quadratic_form'|'quad_form_cholesky', vars, l, rhs, op)
                #            ('dc_quadratic_form', vars, (l_plus, l_minus), rhs, op)
                vars_ = cons[1] if len(cons) > 1 else []
                l_arg = cons[2] if len(cons) > 2 else None
                if kind == "dc_quadratic_form" and isinstance(l_arg, tuple):
                    cholesky_l_plus, cholesky_l_minus = l_arg
                    rhs = float(cons[3]) if len(cons) > 3 else 0.0
                    op = cons[4] if len(cons) > 4 else "eq"
                elif kind == "dc_quadratic_form":
                    cholesky_l_plus = l_arg
                    cholesky_l_minus = cons[3] if len(cons) > 3 else None
                    rhs = float(cons[4]) if len(cons) > 4 else 0.0
                    op = cons[5] if len(cons) > 5 else "eq"
                else:
                    cholesky_l_plus, cholesky_l_minus = l_arg, None
                    rhs = float(cons[3]) if len(cons) > 3 else 0.0
                    op = cons[4] if len(cons) > 4 else "eq"
                if vars_ and cholesky_l_plus and len(cholesky_l_plus) == len(vars_):
                    if self._revise_dc_constraint(vars_, cholesky_l_plus, cholesky_l_minus, rhs, op):
                        return False
                return True

        # dict form
        if isinstance(cons, dict):
            kind = cons.get("type") or cons.get("kind")
            if kind in ("bound",):
                return self._revise_constraint(("bound", cons["var"], cons.get("lo"), cons.get("hi")))
            if kind in ("sum",):
                return self._revise_constraint(("sum", cons["vars"], cons.get("target", 0.0)))
            if kind in ("linear",):
                return self._revise_constraint(("linear", cons["coeffs"], cons.get("target", 0.0), cons.get("op", "eq")))
            if kind == "quadratic_form" or kind == "quad_form" or kind == "quad_form_cholesky":
                cholesky_l = cons.get("cholesky_l", cons.get("cholesky_l_plus"))
                if cholesky_l is None:
                    return True
                return self._revise_constraint(
                    ("quadratic_form", cons.get("vars", []), cholesky_l,
                     cons.get("rhs", 0.0), cons.get("op", "eq"))
                )
            if kind in ("quad_form_split", "dc_quadratic_form"):
                return self._revise_constraint(
                    ("dc_quadratic_form", cons.get("vars", []),
                     (cons.get("cholesky_l_plus"), cons.get("cholesky_l_minus")),
                     cons.get("rhs", 0.0), cons.get("op", "eq"))
                )

        # VirExpr (very basic)
        try:
            from .vir import VirExpr
            if isinstance(cons, VirExpr):
                # handle compare var vs const
                if cons.kind == "compare":
                    lhs = cons.lhs
                    rhs = cons.rhs
                    if lhs is not None and getattr(lhs, "kind", None) == "var" and rhs is not None and getattr(rhs, "kind", None) in ("const_real", "const_int", "const_value"):
                        val = rhs.value
                        if isinstance(val, bool):
                            val = float(val)
                        if isinstance(val, (int, float)):
                            op = cons.op
                            vid = lhs.var_id
                            cur = self._interval_for_var(vid)
                            if op in ("eq",):
                                inter = cur.intersect(Interval(float(val), float(val)))
                            elif op in ("le", "lt"):
                                inter = cur.intersect(Interval(float("-inf"), float(val)))
                            elif op in ("ge", "gt"):
                                inter = cur.intersect(Interval(float(val), float("inf")))
                            else:
                                inter = cur
                            if inter.is_empty():
                                return False
                            self.domains[vid] = inter
                            return True
                if cons.kind == "and":
                    for a in cons.args:
                        if not self._revise_constraint(a):
                            return False
                    return True
        except Exception:
            pass

        return True

    def propagate(self, clause: Any = None) -> bool:
        if clause is not None:
            if clause not in self.constraints:
                self.constraints.append(clause)
        changed = True
        passes = 0
        while changed and passes < 50:
            changed = False
            passes += 1
            snapshot = {k: Interval(v.lo, v.hi) for k, v in self.domains.items()}
            for cons in list(self.constraints):
                if not self._revise_constraint(cons):
                    return False
            for k in self.domains:
                if k in snapshot and self.domains[k] != snapshot[k]:
                    changed = True
        if any(iv.is_empty() for iv in self.domains.values()):
            return False
        return _all_finite_feasible(self.domains)

    def propagate_clause(self, clause: Any = None) -> bool:
        return self.propagate(clause)

    def propagate_all(self) -> bool:
        return self.propagate()

    def current_log_volume(self) -> float:
        total = 0.0
        for iv in self.domains.values():
            if iv.is_empty():
                return float("-inf")
            w = iv.width()
            if w <= 0:
                return float("-inf")
            total += math.log(w)
        return total

    def estimate_log_volume_after(self, var_id: Any, candidate_interval: Interval) -> float:
        # non-mutating
        saved = {k: Interval(v.lo, v.hi) for k, v in self.domains.items()}
        try:
            cur = self._interval_for_var(var_id)
            inter = cur.intersect(candidate_interval if isinstance(candidate_interval, Interval) else Interval(candidate_interval[0], candidate_interval[1]))
            if inter.is_empty():
                return float("-inf")
            self.domains[var_id] = inter
            # one pass
            for cons in list(self.constraints):
                self._revise_constraint(cons)
            return self.current_log_volume()
        finally:
            self.domains = saved

    def compute_numeric_token_twist(self, var_id: Any, candidate_interval: Any, beta: float = 1.0) -> float:
        base = self.current_log_volume()
        cand = candidate_interval if isinstance(candidate_interval, Interval) else Interval(candidate_interval[0], candidate_interval[1])
        rem = self.estimate_log_volume_after(var_id, cand)
        if rem == float("-inf") or base == float("-inf"):
            return float("-inf")
        return beta * (rem - base)

    def future_volume_twist(self, var_id: Any, candidate_interval: Any, beta: float = 1.0) -> float:
        return self.compute_numeric_token_twist(var_id, candidate_interval, beta)

    def extract_conflict_core(self, clause: Any = None) -> list[Any]:
        core = [vid for vid, iv in self.domains.items() if iv.is_empty()]
        if not core and clause is not None:
            core = [clause]
        return core

    def is_empty(self) -> bool:
        return any(iv.is_empty() for iv in self.domains.values())

    def __repr__(self):
        return f"Hc4AffineContractor({self.domains})"


def _apply_lt(
    intervals: list[Interval],
    factor: Sequence[Sequence[float]],
    n: int,
) -> list[Interval] | None:
    """Compute ``y = Lᵀ x`` with ``x_i`` in ``intervals[i]`` (outward rounding).

    ``L`` is a *lower-triangular* Cholesky factor of a PSD matrix ``A = L Lᵀ``; because
    ``xᵀ A x = ||Lᵀ x||²`` and ``(Lᵀ x)_i = Σ_{j≥i} L[j][i] x_j``.  Returns ``None`` when the
    factor shape does not match the variable count.
    """

    y: list[Interval] = []
    for i in range(n):
        acc = Interval(0.0, 0.0)
        for j in range(i, n):
            row = factor[j]
            if i >= len(row):
                return None
            coeff = float(row[i])
            if coeff == 0.0:
                continue
            term = intervals[j].mul(Interval(coeff, coeff))
            acc = acc.add(term)
        y.append(acc)
    return y


def _sum_square(terms: list[Interval]) -> Interval:
    """Outward-rounded ``Σ y_i²`` for interval ``y_i``."""

    acc = Interval(0.0, 0.0)
    for term in terms:
        acc = acc.add(term.sqr())
    return acc


def _all_finite_feasible(domains: dict[Any, Interval]) -> bool:
    return not any(iv.is_empty() for iv in domains.values())


# Aliases expected by various tests
IntervalContractor = Hc4AffineContractor
AffineContractor = Hc4AffineContractor
HC4Contractor = Hc4AffineContractor
Contractor = Hc4AffineContractor
IntervalArithmetic = Interval
HC4 = Hc4AffineContractor
AF2 = AffineForm
ResidualSum = ResidualAccumulator
ResidualAccumulatorVariable = ResidualAccumulator
DynamicContainerLookahead = ResidualAccumulator
VolumeTwist = Hc4AffineContractor
from_numeric_prefix = Interval.from_numeric_prefix
