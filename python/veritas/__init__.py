"""veritas — neuro-symbolic solver-guided decoding & formal verification engine.

The public surface follows the product handbook:

* ``@vt.verified_schema`` — compile Python/Pydantic constraints into VIR.
* ``vt.require / derive / prefer / forall / sum / dot / quad_form / keys_subset`` — the
  authoring DSL.
* ``spec.verify(instance)`` / ``vt.verify`` — check a concrete result (never a false PASS).
* ``vt.generate(model=..., spec=...)`` — solver-guided decoding with 0-DoF Jump-Forward.
* ``vt.build_certificate`` / ``vt.verify_certificate`` — self-contained proof artifacts.
* ``vt.WorldState`` + ``vt.always/eventually/before/responds`` — finite-trace policies.
"""

from __future__ import annotations

import functools
from typing import Any

from . import native

__version__ = "2.0.0"

# --- authoring layer -------------------------------------------------------
from .cartridges.sec import WorldState

# --- jump-forward / backtracking -----------------------------------------
from .cdcl_jumpback import (
    CdclJumpbackEngine,
    ClauseCheckpoint,
    ConflictBranchRecord,
    TabuPrefixTrie,
)
from .compiler.analysis import analyze_containers, container_for_name
from .compiler.ast_transpiler import AstTranspiler, TranspileError
from .compiler.dependency_graph import DependencyGraph
from .compiler.preflight import InfeasibleSpecError, check_spec, run_preflight
from .compiler.schema_lowering import build_json_key_order, lower_pydantic_model
from .compiler.vir import (
    ContainerInfo,
    PreflightResult,
    SoftPreference,
    VirExpr,
    VirSpec,
    VirType,
    ZeroDofRule,
)

# --- tier 0 / 1 / 2 engines ----------------------------------------------
from .crane_mode import CraneModeController

# --- decoding / proof -----------------------------------------------------
from .decode import (
    DeadEnd,
    DecodeConfig,
    DecodeEngine,
    DecodeError,
    DecodeResult,
    DecodeTelemetry,
    ModelProtocolError,
    StepTrace,
    VeritasSession,
    generate,
)
from .decorators import (
    CircularDerivationError,
    VerifiedSchema,
    derive,
    dot,
    forall,
    keys_subset,
    prefer,
    quad_form,
    require,
    sqrt,
    sum,
    verified_schema,
)

# --- concrete verification ------------------------------------------------
from .instance import (
    FAIL,
    INCOMPLETE,
    PASS,
    ConstraintOutcome,
    InstanceFormatError,
    VerificationError,
    VerificationReport,
    describe_spec,
    extract_data,
    fill_derived,
    lower_instance,
    verify_instance,
    verify_json,
    verify_values,
)
from .interval import (
    AffineForm,
    Hc4AffineContractor,
    Interval,
    ResidualAccumulator,
)
from .json_pda import (
    JSONPushdownAutomaton,
    JSONSyntaxError,
    TokenMasker,
    is_complete_json,
    json_value,
    parse_json,
    render_object,
)
from .linear import Clause, LinearForm, compile_clauses, linearize

# --- temporal logic / ecosystem ------------------------------------------
from .ltlf import (
    Always as Always,
)
from .ltlf import (
    Event,
    Formula,
    Trace,
    always,
    at_most,
    before,
    eventually,
    never,
    occurred,
    responds,
)
from .ltlf import (
    evaluate as evaluate_formula,
)
from .nano_solver import SolverStatus, VeritasNanoSolver
from .pattern import PatternConstraint, UnsupportedPattern
from .pattern import matches as pattern_matches
from .pda import PushdownAutomaton
from .proof.certificate import (
    CertificateCheck,
    CertificateError,
    CertificateVerification,
    ProofCertificate,
    VerifiedResult,
    build_certificate,
    render_smt2,
    verify_certificate,
)
from .rl.collector import (
    StepRecord,
    VeritasRLCollector,
    export_step_dpo,
)
from .serde import digest_of_instance, spec_digest, spec_from_dict, spec_to_dict
from .shadow_ast import ShadowParser
from .smt_bridge import SmtBridge, Z3SmtOracle
from .testing import MockCharModel, RandomCharModel, StructuredReasoningModel, mock_model
from .token_healer import (
    ByteTokenizer,
    JumpForwardPrefill,
    TokenHealer,
    TokenizerProtocol,
)
from .zero_dof import ZeroDofEngine

# ---------------------------------------------------------------------------
# top-level convenience
# ---------------------------------------------------------------------------

def verify(spec: Any, instance: Any, **kwargs: Any) -> VerificationReport:
    """Verify ``instance`` against ``spec`` (a schema object or a ``VirSpec``)."""

    return verify_instance(spec, instance, **kwargs)


def verify_proof(certificate: Any, expected_spec: Any = None) -> CertificateVerification:
    """Verify a proof certificate offline; returns the full check report."""

    return verify_certificate(certificate, expected_spec=expected_spec)


def _wrap_tool(function, spec, world_state, return_proof: bool):
    import inspect

    def _finish(output):
        if spec is None:
            if world_state is not None:
                world_state.record_call(function.__name__)
            return output
        report = verify_instance(spec, output)
        wrapper.last_report = report
        report.raise_if_failed()
        if return_proof:
            wrapper.last_proof = build_certificate(spec, report.values, metadata={"tool": function.__name__})
        if world_state is not None:
            world_state.record_call(function.__name__)
        return report.model if report.model is not None else report.values

    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def async_wrapper(*args, **kwargs):
            if world_state is not None:
                world_state.before_tool_call(function.__name__, kwargs)
            output = await function(*args, **kwargs)
            return _finish(output)
        wrapper = async_wrapper
    else:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if world_state is not None:
                world_state.before_tool_call(function.__name__, kwargs)
            output = function(*args, **kwargs)
            return _finish(output)

    wrapper.spec = spec
    wrapper.last_report = None
    wrapper.last_proof = None
    return wrapper


def verified_tool(spec: Any = None, world_state: Any = None, return_proof: bool = False, **kwargs: Any):
    """Wrap a tool so its return value is checked against ``spec`` before use."""

    if spec is not None and not isinstance(spec, (VerifiedSchema, VirSpec)) and callable(spec):
        # Bare @verified_tool usage: no schema to enforce.
        return _wrap_tool(spec, None, world_state, return_proof)

    def decorator(function):
        return _wrap_tool(function, spec, world_state, return_proof)

    return decorator


class OutlinesAdapter:
    """Minimal adapter exposing the handbook's ``vt.from_outlines`` entry point.

    It does not reimplement Outlines; it routes generation through ``vt.generate`` with
    whatever model adapter is supplied, so a tokenizer-backed model can be wrapped.
    """

    def __init__(self, model: Any, spec: Any = None, **options: Any):
        self.model = model
        self.spec = spec
        self.options = options

    def generate(self, prompt: str, **kwargs: Any):
        spec = kwargs.pop("spec", self.spec)
        return generate(model=self.model, prompt=prompt, spec=spec, **kwargs)


def from_outlines(model: Any, spec: Any = None, **options: Any) -> OutlinesAdapter:
    return OutlinesAdapter(model, spec, **options)


def synth(samples: Any = None, **kwargs: Any):
    """Infer a schema from example JSON objects; see :mod:`veritas.synth`."""

    from .synth import synth_schema

    return synth_schema(samples, **kwargs)


__all__ = [
    "__version__",
    "doctor",
    "native",
    # authoring
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
    "TranspileError",
    "VirSpec",
    "VirExpr",
    "VirType",
    "ZeroDofRule",
    "SoftPreference",
    "ContainerInfo",
    "PreflightResult",
    "DependencyGraph",
    "AstTranspiler",
    "lower_pydantic_model",
    "build_json_key_order",
    "run_preflight",
    "check_spec",
    "analyze_containers",
    "container_for_name",
    # verification
    "verify",
    "verify_instance",
    "verify_values",
    "verify_json",
    "VerificationReport",
    "VerificationError",
    "ConstraintOutcome",
    "InstanceFormatError",
    "describe_spec",
    "extract_data",
    "fill_derived",
    "lower_instance",
    "spec_digest",
    "spec_to_dict",
    "spec_from_dict",
    "digest_of_instance",
    "PASS",
    "FAIL",
    "INCOMPLETE",
    # engine
    "Interval",
    "AffineForm",
    "Hc4AffineContractor",
    "ResidualAccumulator",
    "PushdownAutomaton",
    "ShadowParser",
    "CraneModeController",
    "SmtBridge",
    "Z3SmtOracle",
    "VeritasNanoSolver",
    "SolverStatus",
    "LinearForm",
    "Clause",
    "compile_clauses",
    "linearize",
    # tier 0
    "JSONPushdownAutomaton",
    "JSONSyntaxError",
    "TokenMasker",
    "render_object",
    "json_value",
    "parse_json",
    "is_complete_json",
    # jump forward
    "ZeroDofEngine",
    "ZeroDofRule",
    "PatternConstraint",
    "TokenHealer",
    "pattern_matches",
    "UnsupportedPattern",
    "ByteTokenizer",
    "JumpForwardPrefill",
    "TokenizerProtocol",
    "CdclJumpbackEngine",
    "ClauseCheckpoint",
    "ConflictBranchRecord",
    "TabuPrefixTrie",
    # decoding
    "generate",
    "VeritasSession",
    "DecodeEngine",
    "DecodeConfig",
    "DecodeResult",
    "DecodeTelemetry",
    "DecodeError",
    "ModelProtocolError",
    "DeadEnd",
    "StepTrace",
    # proofs
    "ProofCertificate",
    "VerifiedResult",
    "build_certificate",
    "verify_certificate",
    "verify_proof",
    "CertificateCheck",
    "CertificateVerification",
    "CertificateError",
    "render_smt2",
    # temporal / ecosystem
    "WorldState",
    "Formula",
    "Trace",
    "Event",
    "always",
    "eventually",
    "never",
    "before",
    "responds",
    "at_most",
    "occurred",
    "evaluate_formula",
    "VeritasRLCollector",
    "StepRecord",
    "export_step_dpo",
    "MockCharModel",
    "RandomCharModel",
    "StructuredReasoningModel",
    "mock_model",
    "verified_tool",
    "from_outlines",
    "OutlinesAdapter",
    "synth",
]


def doctor() -> dict[str, Any]:
    """Describe the loaded engine, including whether the native extension is in use.

    ``native.available()`` is false on any install without a compiler, and the pure-Python
    engine then does all the work; ``native.load_error()`` says why. That answer belongs in
    one place, because "the extension did not load" is the first thing to check when a
    deployment is slower than expected.
    """
    return {
        "version": __version__,
        "python": "pure" if not native.available() else "accelerated",
        "engine": native.info(),
        "wire_version": native.wire_version(),
    }


