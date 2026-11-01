"""Self-contained proof certificates: build, verify, tamper detection, SMT2."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pytest
from pydantic import BaseModel

import veritas as vt

from veritas.proof.certificate import (
    ProofCertificate,
    build_certificate,
    verify_certificate,
)


class Pair(BaseModel):
    a: float
    b: float


@vt.verified_schema()
def pair(p: Pair):
    vt.require(p.a >= 0)
    vt.require(p.b >= 0)
    vt.require(p.a + p.b == 1.0, tol=1e-9)
    vt.require(p.b <= 0.8)


def test_certificate_roundtrip_and_verification(tmp_path):
    certificate = build_certificate(pair, {"a": 0.25, "b": 0.75})
    assert certificate.sha256
    path = certificate.save(tmp_path / "proof.json")
    loaded = ProofCertificate.load(path)
    verification = verify_certificate(loaded)
    assert verification.ok, verification.as_dict()

    payload = json.loads(path.read_text())
    assert payload["spec_sha256"] == certificate.spec_sha256
    assert payload["witness"]["a"] == 0.25


def test_tampered_witness_is_rejected():
    certificate = build_certificate(pair, {"a": 0.25, "b": 0.75})
    data = certificate.to_dict()
    data["witness"]["b"] = 0.5  # violates a + b == 1
    verification = verify_certificate(data)
    assert not verification.ok
    assert any(check.name == "constraints" and not check.ok for check in verification.checks)


def test_tampered_witness_with_recomputed_hash_is_still_rejected():
    certificate = build_certificate(pair, {"a": 0.25, "b": 0.75})
    data = certificate.to_dict()
    data["witness"]["b"] = 0.5
    forged = ProofCertificate.from_dict(data)
    forged.seal()  # attacker recomputes the hash
    verification = verify_certificate(forged)
    assert not verification.ok
    assert any(check.name == "constraints" and not check.ok for check in verification.checks)


def test_tampered_spec_payload_is_rejected():
    certificate = build_certificate(pair, {"a": 0.25, "b": 0.75})
    data = certificate.to_dict()
    # Remove the b <= 0.8 requirement from the embedded spec.
    requires = data["spec"]["requires"]
    data["spec"]["requires"] = requires[:-1]
    forged = ProofCertificate.from_dict(data)
    forged.seal()
    verification = verify_certificate(forged)
    assert not verification.ok
    assert any(check.name == "spec_sha256" and not check.ok for check in verification.checks)


def test_certificate_bound_to_expected_spec():
    certificate = build_certificate(pair, {"a": 0.25, "b": 0.75})
    assert verify_certificate(certificate, expected_spec=pair).ok

    class Other(BaseModel):
        x: float

    @vt.verified_schema()
    def other(p: Other):
        vt.require(p.x >= 0)

    assert not verify_certificate(certificate, expected_spec=other).ok


def test_smt2_contains_constraints_and_witness():
    certificate = build_certificate(pair, {"a": 0.25, "b": 0.75})
    smt2 = certificate.smt2
    assert "(set-logic ALL)" in smt2
    assert "(declare-const a Real)" in smt2
    assert "(check-sat)" in smt2
    assert "(assert (= a (/ 1 4)))" in smt2
    assert "req_2" in smt2  # named constraint for a + b == 1
    assert "; witness_sha256:" in smt2


def test_certificate_rejects_invalid_instance():
    with pytest.raises(vt.VerificationError):
        build_certificate(pair, {"a": 0.1, "b": 0.2})


def test_malformed_certificate_is_reported_not_crashing():
    verification = verify_certificate({"version": "2.0"})
    assert not verification.ok
    assert any(not check.ok for check in verification.checks)
