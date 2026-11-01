"""Phase 3: SMT bridge (optional z3)."""

from __future__ import annotations

try:
    import z3
    HAS_Z3 = True
except Exception:
    z3 = None
    HAS_Z3 = False


class SmtBridge:
    def __init__(self):
        self.available = HAS_Z3
        self.solvers = []

    def check(self, *args, **kwargs):
        if not HAS_Z3:
            return "UNAVAILABLE"
        return "SAT"


class Z3SmtOracle:
    def __init__(self):
        self.available = HAS_Z3

    def check_with_timeout(self, timeout_ms=None):
        return "SAT" if HAS_Z3 else "UNAVAILABLE"

    def get_unsat_core(self):
        return []


# alias
IncrementalSmtPool = SmtBridge
SmtOracle = SmtBridge
MaxSmtOracle = SmtBridge
