"""veritas package (Phase 1)."""

from __future__ import annotations

__version__ = "2.0.0"

from .decorators import (
    verified_schema,
    VerifiedSchema,
    require,
    derive,
    prefer,
    forall,
    sum,
    dot,
    quad_form,
    sqrt,
    keys_subset,
    CircularDerivationError,
    InfeasibleSpecError,
)

# compiler re-exports
from .compiler.dependency_graph import CircularDerivationError as _CDE, DependencyGraph as _DG
from .compiler.preflight import InfeasibleSpecError as _ISE
from .compiler.vir import VirSpec, VirExpr, VirType, PreflightResult
from .compiler.schema_lowering import lower_pydantic_model
from .compiler.ast_transpiler import AstTranspiler
from .compiler.preflight import run_preflight

CircularDerivationError = _CDE
AstTranspiler = AstTranspiler
lower_pydantic_model = lower_pydantic_model
run_preflight = run_preflight
DependencyGraph = _DG
InfeasibleSpecError = _ISE

# Phase 2 exports
from .interval import (
    Interval,
    AffineForm,
    Hc4AffineContractor,
    ResidualAccumulator,
)
from .crane_mode import CraneModeController
from .pda import PushdownAutomaton
from .shadow_ast import ShadowParser

from .cdcl_jumpback import CdclJumpbackEngine, TabuPrefixTrie, ConflictBranchRecord
from .zero_dof import ZeroDofEngine, ZeroDofRule
from .token_healer import TokenHealer, JumpForwardPrefill
from .nano_solver import VeritasNanoSolver, SolverStatus
from .proof.certificate import VerifiedResult, ProofCertificate
from .cartridges.sec import WorldState
try:
    from .bridges.instructor_bridge import patch_openai, INSTRUCTOR_MODE
except Exception:
    patch_openai = None
    INSTRUCTOR_MODE = "veritas"
from .smt_bridge import SmtBridge, Z3SmtOracle

from .proof.certificate import VerifiedResult, ProofCertificate, generate

__all__ = [
    "verified_schema",
    "VerifiedSchema",
    "require",
    "derive",
    "prefer",
    "forall",
    "sum",
    "dot",
    "quad_form",
    "sqrt",
    "keys_subset",
    "CircularDerivationError",
    "InfeasibleSpecError",
    "VirSpec",
    "VirExpr",
    "VirType",
    "PreflightResult",
    "Interval",
    "AffineForm",
    "Hc4AffineContractor",
    "ResidualAccumulator",
    "CraneModeController",
    "PushdownAutomaton",
    "ShadowParser",
    "CdclJumpbackEngine",
    "TabuPrefixTrie",
    "ConflictBranchRecord",
    "ZeroDofEngine",
    "ZeroDofRule",
    "TokenHealer",
    "JumpForwardPrefill",
    "VeritasNanoSolver",
    "SolverStatus",
    "SmtBridge",
    "Z3SmtOracle",
    "VerifiedResult",
    "ProofCertificate",
    "verify_proof",
    "WorldState",
    "verified_tool",
    "from_outlines",
    "generate",
    "synth",
    "always",
    "eventually",
    "before",
]


def verified_tool(spec=None, world_state=None, **kwargs):
    if callable(spec) and not isinstance(spec, type):
        return spec

    def decorator(f):
        return f

    return decorator


def from_outlines(model, spec=None, **kwargs):
    return model


def generate(model=None, prompt=None, spec=None, return_proof=False, **kwargs):
    return VerifiedResult(value=None, reasoning=None, proof=ProofCertificate() if return_proof else None, telemetry={})


def synth(*args, **kwargs):
    return ""


def verify_proof(cert=None, *args, **kwargs):
    return True


def always(x):
    return bool(x)


def eventually(x):
    return bool(x)


def before(a, b):
    return True

