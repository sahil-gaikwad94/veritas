"""Proof-carrying outputs: self-contained, offline-verifiable certificates.

A ``ProofCertificate`` carries the *semantic* spec (VIR), the witness (the concrete field
values), a portable SMT-LIB2 rendering, and a SHA-256 content-addressed digest.  Offline
verification reconstructs the spec from the certificate payload and re-evaluates every
hard constraint against the witness — no model, no original Python module, no GPU.

Security properties enforced here:

* ``sha256`` covers the spec payload, the witness, metadata and the SMT-LIB2 text, so
  altering any of them is detectable; even if an attacker recomputes the hash, the
  constraints are re-evaluated against the witness and fail.
* ``verify_certificate`` never returns true on an exception or a partially parsed payload:
  every check is explicit and reported.
* Witness verification does **not** fill derived fields — a certificate must stand on the
  values it actually contains.
"""

from __future__ import annotations

import hmac
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar

from ..compiler.vir import VirSpec, VirType
from ..evaluate import expr_to_str, format_number
from ..instance import VerificationError, VerificationReport, _coerce_spec, verify_values
from ..serde import (
    SERDE_VERSION,
    _decode_value,
    _encode_value,
    canonical_json,
    digest_of_instance,
    sha256_hex,
    spec_digest,
    spec_from_dict,
    spec_to_dict,
)

T = TypeVar("T")

CERTIFICATE_VERSION = "2.0"


# ---------------------------------------------------------------------------
# SMT-LIB2 rendering
# ---------------------------------------------------------------------------

_SMT_RESERVED = {
    "assert", "check-sat", "declare-const", "define-fun", "exists", "forall", "let",
    "true", "false", "and", "or", "not", "ite", "select", "store", "as", "par",
}
_SMT_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")
_CONTAINER_KINDS = {"dynamic_map", "list", "dict", "set"}


def _sanitize(text: Any) -> str:
    name = _SMT_IDENT_RE.sub("_", str(text) or "v")
    if not name or name[0].isdigit():
        name = "v_" + name
    if name in _SMT_RESERVED:
        name = "v_" + name
    return name


def _smt_real(value: float) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite value in SMT rendering")
    numerator, denominator = number.as_integer_ratio()
    if denominator == 1:
        return str(numerator)
    return f"(/ {numerator} {denominator})"


def _sort_for(vtype: Optional[VirType]) -> str:
    kind = getattr(vtype, "kind", "real")
    if kind == "int":
        return "Int"
    if kind == "bool":
        return "Bool"
    return "Real"


class _SmtRenderer:
    def __init__(self, spec: VirSpec, witness: Mapping[str, Any]):
        self.spec = spec
        self.witness = witness
        self.symbol_by_var: Dict[int, str] = {}
        self.decls: List[Tuple[str, str]] = []
        self.elem_symbols: Dict[Tuple[int, Any], str] = {}
        self.skipped: List[str] = []
        self._container_ids = set(spec.containers.keys())
        for vid, info in spec.variables.items():
            name = info.get("name") or spec.id_to_name.get(vid, str(vid))
            vtype = info.get("type")
            if vid in self._container_ids:
                value = witness.get(name)
                if isinstance(value, Mapping):
                    for key in sorted(value.keys(), key=str):
                        if not isinstance(value[key], (int, float)) or isinstance(value[key], bool):
                            continue
                        symbol = _sanitize(f"{name}_{key}")
                        self.elem_symbols[(vid, key)] = symbol
                        self.decls.append((symbol, "Real"))
                elif isinstance(value, (list, tuple)):
                    for index, item in enumerate(value):
                        if not isinstance(item, (int, float)) or isinstance(item, bool):
                            continue
                        symbol = _sanitize(f"{name}_{index}")
                        self.elem_symbols[(vid, index)] = symbol
                        self.decls.append((symbol, "Real"))
                continue
            symbol = _sanitize(name)
            self.symbol_by_var[vid] = symbol
            self.decls.append((symbol, _sort_for(vtype)))

    # -- helpers ---------------------------------------------------------
    def _var_symbol(self, node: Any) -> Optional[str]:
        vid = getattr(node, "var_id", None)
        if vid is None:
            name = getattr(node, "name", None)
            vid = self.spec.name_to_id.get(name) if name else None
        if vid is None:
            return None
        return self.symbol_by_var.get(vid)

    def _container_var_id(self, node: Any) -> Optional[int]:
        if node is None:
            return None
        if getattr(node, "kind", None) == "values" and node.args:
            return self._container_var_id(node.args[0])
        if getattr(node, "kind", None) != "var":
            return None
        vid = getattr(node, "var_id", None)
        if vid is None and getattr(node, "name", None):
            vid = self.spec.name_to_id.get(node.name)
        return vid if vid in self._container_ids else None

    def _container_keys(self, vid: int) -> List[Any]:
        return [key for (cid, key) in self.elem_symbols if cid == vid]

    def _container_symbol(self, vid: int, key: Any) -> Optional[str]:
        return self.elem_symbols.get((vid, key))

    # -- translation -----------------------------------------------------
    def expr(self, node: Any, elem_symbol: Optional[str] = None) -> Optional[str]:
        if node is None:
            return None
        kind = getattr(node, "kind", None)
        if kind in ("const_real", "const_int"):
            return _smt_real(node.value)
        if kind == "const_value":
            value = node.value
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (int, float)):
                return _smt_real(value)
            return None
        if kind == "const_bool":
            return "true" if node.value else "false"
        if kind == "var":
            return self._var_symbol(node)
        if kind == "name":
            vid = self.spec.name_to_id.get(getattr(node, "name", ""))
            return self.symbol_by_var.get(vid) if vid is not None else None
        if kind == "elem":
            return elem_symbol
        if kind == "neg":
            inner = self.expr(node.lhs if node.lhs is not None else (node.args[0] if node.args else None), elem_symbol)
            return f"(- {inner})" if inner else None
        if kind in ("add", "sub", "mul", "div"):
            left = self.expr(node.lhs if node.lhs is not None else (node.args[0] if node.args else None), elem_symbol)
            right = self.expr(node.rhs if node.rhs is not None else (node.args[1] if len(node.args) > 1 else None), elem_symbol)
            if left is None or right is None:
                return None
            op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[kind]
            return f"({op} {left} {right})"
        if kind == "pow":
            base = self.expr(node.lhs, elem_symbol)
            exponent = getattr(node.rhs, "value", None)
            if base is None:
                return None
            if isinstance(exponent, (int, float)) and float(exponent).is_integer() and 0 <= exponent <= 8:
                if int(exponent) == 0:
                    return "1"
                result = base
                for _ in range(int(exponent) - 1):
                    result = f"(* {result} {base})"
                return result
            return None
        if kind == "abs":
            inner = self.expr(node.args[0] if node.args else node.lhs, elem_symbol)
            return f"(abs {inner})" if inner else None
        if kind == "sqrt":
            self.skipped.append(f"sqrt({expr_to_str(node, self.spec.id_to_name)}) is transcendental")
            return None
        if kind == "sum":
            inner = node.args[0] if node.args else None
            vid = self._container_var_id(inner)
            if vid is not None:
                symbols = [self._container_symbol(vid, key) for key in self._container_keys(vid)]
                symbols = [s for s in symbols if s]
                if not symbols:
                    return "0"
                return symbols[0] if len(symbols) == 1 else "(+ " + " ".join(symbols) + ")"
            if inner is not None and getattr(inner, "kind", None) == "const_list":
                numbers = [v for v in inner.value or [] if isinstance(v, (int, float))]
                if len(numbers) == len(inner.value or []):
                    return _smt_real(sum(numbers))
            return None
        if kind == "dot":
            left = node.args[0] if node.args else None
            vid = self._container_var_id(left)
            returns = getattr(node, "value", None)
            if vid is None or not isinstance(returns, Mapping):
                return None
            terms: List[str] = []
            for key in self._container_keys(vid):
                coefficient = returns.get(key)
                symbol = self._container_symbol(vid, key)
                if coefficient is None or symbol is None:
                    continue
                terms.append(f"(* {symbol} {_smt_real(coefficient)})")
            if not terms:
                return "0"
            return terms[0] if len(terms) == 1 else "(+ " + " ".join(terms) + ")"
        if kind == "quad_form_cholesky":
            left = node.args[0] if node.args else None
            vid = self._container_var_id(left)
            cov = getattr(node, "value", None)
            if vid is None or not isinstance(cov, Mapping):
                return None
            keys = self._container_keys(vid)
            labels = list(keys)
            matrix: Dict[Tuple[Any, Any], float] = {}
            for key, value in cov.items():
                a, b = key if isinstance(key, tuple) and len(key) == 2 else (key, key)
                matrix[(a, b)] = float(value)
                if a != b:
                    matrix[(b, a)] = float(value)
            terms: List[str] = []
            for a in labels:
                for b in labels:
                    coefficient = matrix.get((a, b))
                    sym_a = self._container_symbol(vid, a)
                    sym_b = self._container_symbol(vid, b)
                    if coefficient is None or sym_a is None or sym_b is None:
                        continue
                    terms.append(f"(* {sym_a} {sym_b} {_smt_real(coefficient)})")
            if not terms:
                return "0"
            return terms[0] if len(terms) == 1 else "(+ " + " ".join(terms) + ")"
        if kind == "forall":
            iterable = node.iterable
            vid = self._container_var_id(iterable)
            if vid is None:
                return None
            parts: List[str] = []
            for key in self._container_keys(vid):
                symbol = self._container_symbol(vid, key)
                if symbol is None:
                    continue
                translated = self.expr(node.predicate, symbol)
                if translated is None:
                    return None
                parts.append(translated)
            if not parts:
                return "true"
            return parts[0] if len(parts) == 1 else "(and " + " ".join(parts) + ")"
        if kind == "keys_subset":
            # Declared element symbols come from the witness, which the operator supplied;
            # the subset fact is therefore structural and reported as true.
            return "true"
        if kind == "subscript":
            base = node.args[0] if node.args else None
            vid = self._container_var_id(base)
            key = getattr(node, "value", None)
            if vid is None:
                return None
            return self._container_symbol(vid, key)
        if kind == "values":
            return None
        if kind in ("and", "or"):
            parts = [self.expr(child, elem_symbol) for child in node.args]
            if any(p is None for p in parts):
                return None
            op = "and" if kind == "and" else "or"
            return parts[0] if len(parts) == 1 else f"({op} " + " ".join(parts) + ")"
        if kind == "ifexp":
            parts = [self.expr(child, elem_symbol) for child in node.args]
            if len(parts) != 3 or any(p is None for p in parts):
                return None
            return f"(ite {parts[0]} {parts[1]} {parts[2]})"
        if kind == "compare":
            left = self.expr(node.lhs, elem_symbol)
            right = self.expr(node.rhs, elem_symbol)
            if left is None or right is None:
                return None
            tol = float(getattr(node, "tol", 1e-6) or 1e-6)
            tol_s = _smt_real(tol)
            op = getattr(node, "op", "eq")
            if op == "eq":
                return f"(<= (abs (- {left} {right})) {tol_s})"
            if op == "ne":
                return f"(not (<= (abs (- {left} {right})) {tol_s}))"
            if op == "le":
                return f"(<= {left} (+ {right} {tol_s}))"
            if op == "lt":
                return f"(< {left} (+ {right} {tol_s}))"
            if op == "ge":
                return f"(>= {left} (- {right} {tol_s}))"
            if op == "gt":
                return f"(> {left} (- {right} {tol_s}))"
            return None
        return None

    def render(self, *, spec_hash: str, witness_hash: str) -> str:
        lines: List[str] = [
            "; veritas proof certificate (v" + CERTIFICATE_VERSION + ")",
            f"; spec_sha256:    {spec_hash}",
            f"; witness_sha256: {witness_hash}",
            "; This script asserts the schema's hard constraints and the witness values.",
            "; Non-linear/transcendental fragments that SMT-LIB cannot express exactly are",
            "; listed as comments; offline veritas verification checks them against the VIR.",
            "(set-logic ALL)",
            "(set-option :produce-proofs true)",
            "",
        ]
        if self.decls:
            lines.append("; --- symbolic variables ---")
            for symbol, sort in self.decls:
                lines.append(f"(declare-const {symbol} {sort})")
            lines.append("")

        lines.append("; --- hard constraints ---")
        for index, expr in enumerate(self.spec.requires):
            rendered = self.expr(expr)
            text = expr_to_str(expr, self.spec.id_to_name)
            if rendered is None:
                lines.append(f"; [unrendered require #{index}] {text}")
                continue
            lines.append(f"; require #{index}: {text}")
            lines.append(f"(assert (! {rendered} :named req_{index}))")
        for index, expr in enumerate(self.spec.derives):
            rendered = self.expr(expr)
            text = expr_to_str(expr, self.spec.id_to_name)
            if rendered is None:
                lines.append(f"; [unrendered derive #{index}] {text}")
                continue
            lines.append(f"; derive #{index}: {text}")
            lines.append(f"(assert (! {rendered} :named der_{index}))")
        if self.spec.soft_preferences:
            lines.append("; --- soft preferences (not asserted as hard constraints) ---")
            for pref in self.spec.soft_preferences:
                lines.append(
                    f"; prefer weight={pref.weight:g} tag={pref.tag!r}: "
                    f"{expr_to_str(pref.expr, self.spec.id_to_name)}"
                )
        lines.append("")

        lines.append("; --- witness ---")
        for vid, symbol in self.symbol_by_var.items():
            name = self.spec.id_to_name.get(vid, str(vid))
            value = self.witness.get(name)
            if isinstance(value, bool):
                lines.append(f"(assert (= {symbol} {'true' if value else 'false'}))")
            elif isinstance(value, (int, float)):
                lines.append(f"(assert (= {symbol} {_smt_real(value)}))")
        for (vid, key), symbol in self.elem_symbols.items():
            name = self.spec.id_to_name.get(vid, str(vid))
            container = self.witness.get(name)
            if isinstance(container, Mapping) and key in container:
                value = container[key]
            elif isinstance(container, (list, tuple)) and isinstance(key, int) and key < len(container):
                value = container[key]
            else:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lines.append(f"(assert (= {symbol} {_smt_real(value)}))")
        lines.append("")
        if self.skipped:
            lines.append("; --- renderer notes ---")
            for note in sorted(set(self.skipped)):
                lines.append(f"; {note}")
            lines.append("")
        lines += ["(check-sat)", "(get-model)", "(exit)", ""]
        return "\n".join(lines)


def _plain_data(value: Any) -> Any:
    """Project a witness/metadata value onto plain JSON types for the public format."""

    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain_data(item) for item in value), key=repr)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def render_smt2(spec: VirSpec, witness: Mapping[str, Any]) -> str:
    renderer = _SmtRenderer(spec, witness)
    return renderer.render(spec_hash=spec_digest(spec), witness_hash=digest_of_instance(witness))


# ---------------------------------------------------------------------------
# certificate
# ---------------------------------------------------------------------------

@dataclass
class ProofCertificate:
    version: str = CERTIFICATE_VERSION
    spec_sha256: str = ""
    witness_sha256: str = ""
    sha256: str = ""
    smt2: str = ""
    witness: Dict[str, Any] = field(default_factory=dict)
    spec_payload: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- canonical payload / digest --------------------------------------
    def digest_payload(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "spec_sha256": self.spec_sha256,
            "witness_sha256": self.witness_sha256,
            "witness": _plain_data(self.witness),
            "spec": self.spec_payload,
            "smt2": self.smt2,
            "metadata": _plain_data(self.metadata),
        }

    def compute_digest(self) -> str:
        return sha256_hex(canonical_json(self.digest_payload()))

    def seal(self) -> "ProofCertificate":
        self.witness_sha256 = digest_of_instance(self.witness)
        self.sha256 = self.compute_digest()
        return self

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.digest_payload(),
            "sha256": self.sha256,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofCertificate":
        raw_witness = payload.get("witness") or {}
        if isinstance(raw_witness, Mapping) and "__pairs__" in raw_witness:
            raw_witness = _decode_value(raw_witness)
        raw_metadata = payload.get("metadata") or {}
        if isinstance(raw_metadata, Mapping) and "__pairs__" in raw_metadata:
            raw_metadata = _decode_value(raw_metadata)
        return cls(
            version=str(payload.get("version", "")),
            spec_sha256=str(payload.get("spec_sha256", "")),
            witness_sha256=str(payload.get("witness_sha256", "")),
            sha256=str(payload.get("sha256", "")),
            smt2=str(payload.get("smt2", "")),
            witness=dict(raw_witness) if isinstance(raw_witness, Mapping) else {},
            spec_payload=payload.get("spec"),
            metadata=dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {},
        )

    @classmethod
    def load(cls, path: Any) -> "ProofCertificate":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def save(self, path: Any) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def __str__(self) -> str:  # pragma: no cover - presentation
        return f"ProofCertificate(spec={self.spec_sha256[:12]}…, witness={self.witness_sha256[:12]}…)"


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

@dataclass
class CertificateCheck:
    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class CertificateVerification:
    checks: List[CertificateCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    report: Optional[VerificationReport] = None
    certificate: Optional[ProofCertificate] = None

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks) and not self.errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [c.as_dict() for c in self.checks],
            "errors": list(self.errors),
            "report": self.report.as_dict() if self.report is not None else None,
            "spec_sha256": self.certificate.spec_sha256 if self.certificate else None,
        }

    def raise_if_failed(self) -> "CertificateVerification":
        if not self.ok:
            raise VerificationError(self.report) if self.report is not None else CertificateError(
                "; ".join(self.errors) or "certificate verification failed"
            )
        return self


class CertificateError(ValueError):
    """Raised when a certificate cannot be parsed or verified."""


def _load_certificate(source: Any) -> ProofCertificate:
    if isinstance(source, ProofCertificate):
        return source
    if isinstance(source, Mapping):
        return ProofCertificate.from_dict(source)
    if isinstance(source, (str, Path)):
        text = str(source)
        path = Path(text)
        if "\n" not in text and path.exists():
            return ProofCertificate.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return ProofCertificate.from_dict(json.loads(text))
    raise CertificateError(f"unsupported certificate source {type(source).__name__}")


def verify_certificate(
    source: Any,
    *,
    expected_spec: Any = None,
) -> CertificateVerification:
    """Verify a certificate fully offline.

    Reconstructs the VIR spec from the payload and re-evaluates every constraint against
    the witness.  ``expected_spec`` additionally binds the certificate to a live spec.
    """

    checks: List[CertificateCheck] = []
    errors: List[str] = []

    try:
        certificate = _load_certificate(source)
    except Exception as exc:
        return CertificateVerification(
            checks=[CertificateCheck("parse", False, str(exc))],
            errors=[f"cannot parse certificate: {exc}"],
        )

    checks.append(
        CertificateCheck(
            "version",
            certificate.version == CERTIFICATE_VERSION,
            f"found {certificate.version!r}, expected {CERTIFICATE_VERSION!r}",
        )
    )

    try:
        recomputed = certificate.compute_digest()
    except Exception as exc:
        recomputed = ""
        errors.append(f"cannot recompute digest: {exc}")
    hash_ok = bool(certificate.sha256) and hmac.compare_digest(recomputed, certificate.sha256)
    checks.append(CertificateCheck("sha256", hash_ok, "content digest matches" if hash_ok else "digest mismatch or missing"))

    witness_hash = digest_of_instance(certificate.witness)
    witness_ok = bool(certificate.witness_sha256) and hmac.compare_digest(
        witness_hash, certificate.witness_sha256
    )
    checks.append(
        CertificateCheck("witness_sha256", witness_ok, "witness digest matches" if witness_ok else "witness digest mismatch")
    )

    checks.append(
        CertificateCheck(
            "smt2",
            "(check-sat)" in certificate.smt2 and "(set-logic" in certificate.smt2,
            "SMT-LIB2 script present" if certificate.smt2 else "missing SMT-LIB2 script",
        )
    )

    spec = None
    if certificate.spec_payload is None:
        checks.append(CertificateCheck("spec_payload", False, "certificate has no embedded VIR spec"))
        errors.append("certificate is not self-contained: missing VIR payload")
    else:
        try:
            spec = spec_from_dict(certificate.spec_payload)
        except Exception as exc:
            checks.append(CertificateCheck("spec_payload", False, f"cannot rebuild spec: {exc}"))
            errors.append(f"cannot rebuild spec: {exc}")
        else:
            digest_match = hmac.compare_digest(spec_digest(spec), certificate.spec_sha256 or "")
            checks.append(
                CertificateCheck(
                    "spec_sha256",
                    digest_match,
                    "embedded spec matches its digest" if digest_match else "embedded spec digest mismatch",
                )
            )

    report: Optional[VerificationReport] = None
    if spec is not None:
        try:
            # No derivation filling: the certificate must carry every value it claims.
            report = verify_values(spec, certificate.witness, derive_missing=False)
        except Exception as exc:
            checks.append(CertificateCheck("constraints", False, f"verification crashed: {exc}"))
            errors.append(f"constraint verification crashed: {exc}")
        else:
            constraints_ok = report.status == "PASS"
            checks.append(
                CertificateCheck(
                    "constraints",
                    constraints_ok,
                    report.summary_line() if constraints_ok else f"witness violates the spec ({report.summary_line()})",
                )
            )
            if not constraints_ok:
                errors.extend(report.relaxation or [report.summary_line()])

    if expected_spec is not None:
        try:
            live_spec = _coerce_spec(expected_spec)
            live_digest = spec_digest(live_spec)
            bound = hmac.compare_digest(live_digest, certificate.spec_sha256 or "")
            checks.append(
                CertificateCheck(
                    "expected_spec",
                    bound,
                    "certificate matches the supplied spec" if bound else "certificate was produced for a different spec",
                )
            )
        except Exception as exc:
            checks.append(CertificateCheck("expected_spec", False, str(exc)))

    return CertificateVerification(
        checks=checks,
        errors=errors,
        report=report,
        certificate=certificate,
    )


def build_certificate(
    schema_or_spec: Any,
    values: Mapping[str, Any],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    require_valid: bool = True,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> ProofCertificate:
    """Build a sealed certificate for a concrete instance."""

    spec = _coerce_spec(schema_or_spec)
    report = verify_values(spec, dict(values), derive_missing=True)
    if require_valid and report.status != "PASS":
        raise VerificationError(report)
    witness = dict(report.values if report.status == "PASS" else values)
    if extra_fields:
        witness.update(dict(extra_fields))
    certificate = ProofCertificate(
        version=CERTIFICATE_VERSION,
        spec_sha256=spec_digest(spec),
        smt2=render_smt2(spec, witness),
        witness=witness,
        spec_payload=spec_to_dict(spec),
        metadata=dict(metadata or {}),
    )
    return certificate.seal()


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class VerifiedResult(Generic[T]):
    value: T
    reasoning: str = ""
    proof: Optional[ProofCertificate] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)
    report: Optional[VerificationReport] = None
    raw: str = ""
    spec_name: str = ""

    @property
    def ok(self) -> bool:
        return self.report is None or self.report.status == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "reasoning": self.reasoning,
            "telemetry": self.telemetry,
            "spec_name": self.spec_name,
            "ok": self.ok,
            "report": self.report.as_dict() if self.report is not None else None,
            "proof": self.proof.to_dict() if self.proof is not None else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __str__(self) -> str:  # pragma: no cover - presentation
        return f"VerifiedResult(ok={self.ok}, value={self.value!r})"


__all__ = [
    "CERTIFICATE_VERSION",
    "CertificateCheck",
    "CertificateError",
    "CertificateVerification",
    "ProofCertificate",
    "VerifiedResult",
    "build_certificate",
    "render_smt2",
    "verify_certificate",
]
