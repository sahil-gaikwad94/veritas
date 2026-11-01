"""``veritas`` command line: check-spec, verify, verify-proof, inspect, synth.

Every command is real: specs are imported and compiled, instances are verified, proof
certificates are re-checked offline, traces are rendered to a standalone HTML report, and
schema synthesis emits runnable Python.  Failures return non-zero exit codes rather than
printing success.
"""

from __future__ import annotations

import argparse
import html
import importlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..compiler.preflight import run_preflight
from ..decorators import VerifiedSchema
from ..instance import verify_instance
from ..proof.certificate import verify_certificate
from ..synth import synth_schema

# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load_module(target: str):
    path = Path(target)
    if path.exists() and path.suffix == ".py":
        module_name = f"_veritas_target_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def load_schemas(target: str) -> list[VerifiedSchema]:
    module_part, _, attribute = target.partition(":")
    module = _load_module(module_part or target)
    if attribute:
        obj = getattr(module, attribute, None)
        if obj is None:
            raise AttributeError(f"{module_part!r} has no attribute {attribute!r}")
        if isinstance(obj, VerifiedSchema):
            return [obj]
        schemas = getattr(obj, "schemas", None)
        if schemas:
            return list(schemas)
        raise TypeError(f"{target!r} is not a @vt.verified_schema object")
    schemas = [value for value in vars(module).values() if isinstance(value, VerifiedSchema)]
    # Deduplicate by identity while preserving declaration order.
    seen = set()
    unique = []
    for schema in schemas:
        if id(schema) in seen:
            continue
        seen.add(id(schema))
        unique.append(schema)
    if not unique:
        raise LookupError(f"no @vt.verified_schema objects found in {module_part!r}")
    return unique


def _select(schemas: Sequence[VerifiedSchema], name: str | None, target: str) -> VerifiedSchema:
    if name:
        for schema in schemas:
            if schema.name == name:
                return schema
        raise LookupError(f"no schema named {name!r} in {target!r}")
    if len(schemas) == 1:
        return schemas[0]
    raise LookupError(
        f"{target!r} defines {len(schemas)} schemas; select one with --name "
        + ", ".join(s.name for s in schemas)
    )


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_check_spec(args: argparse.Namespace) -> int:
    schemas = load_schemas(args.target)
    payloads = []
    ok = True
    for schema in schemas:
        described = schema.describe()
        preflight = schema.preflight
        if preflight is None:
            preflight = run_preflight(schema.spec)
        entry = {
            "name": schema.name,
            "fields": described["fields"],
            "derived": described["derived_fields"],
            "requires": described["requires"],
            "derives": described["derives"],
            "containers": described["containers"],
            "zero_dof_rules": described["zero_dof_rules"],
            "preflight": {
                "status": preflight.status,
                "conflicts": list(preflight.conflicts),
                "relaxation": preflight.relaxation,
            },
            "digest": described["digest"],
        }
        payloads.append(entry)
        if preflight.status != "SAT":
            ok = False
    if args.json:
        print(json.dumps(payloads, indent=2, default=str))
    else:
        for entry in payloads:
            print(f"schema: {entry['name']}  ({entry['preflight']['status']})")
            print(f"  fields:   {', '.join(entry['fields'])}")
            print(f"  derived:  {', '.join(entry['derived']) or '-'}")
            print(f"  requires: {len(entry['requires'])}  derives: {len(entry['derives'])}")
            print(f"  digest:   {entry['digest']}")
            if entry["preflight"]["status"] != "SAT":
                for hint in (entry["preflight"].get("relaxation") or {}).get("hints", []):
                    print(f"  ! {hint}")
    return 0 if ok else 1


def cmd_verify(args: argparse.Namespace) -> int:
    schemas = load_schemas(args.target)
    schema = _select(schemas, args.name, args.target)
    if args.instance == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.instance).read_text(encoding="utf-8")
    report = verify_instance(schema, text, derive_missing=not args.no_derive)
    payload = {"schema": schema.name, **report.as_dict()}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"{schema.name}: {report.status}")
        for outcome in report.outcomes:
            state = "ok" if outcome.satisfied is True else ("FAIL" if outcome.satisfied is False else "?")
            print(f"  [{state}] {outcome.kind}: {outcome.text}")
        for hint in report.relaxation:
            print(f"  ! {hint}")
        print(f"  digest: {report.spec_digest}")
    return 0 if report.ok else 1


def cmd_verify_proof(args: argparse.Namespace) -> int:
    verification = verify_certificate(args.proof)
    payload = verification.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("certificate: " + ("VALID" if verification.ok else "INVALID"))
        for check in verification.checks:
            print(f"  [{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
        for error in verification.errors:
            print(f"  ! {error}")
    return 0 if verification.ok else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    report = _render_trace_html(payload, title=args.trace)
    target = Path(args.out) if args.out else Path(args.trace).with_suffix(".html")
    target.write_text(report, encoding="utf-8")
    telemetry = payload.get("telemetry") or {}
    print(f"trace: {args.trace}")
    print(f"  status: {payload.get('status', telemetry.get('final_status', 'unknown'))}")
    if telemetry:
        print(f"  steps: {telemetry.get('steps')}  backtracks: {telemetry.get('backtracks')}  "
              f"jump-forward fields: {telemetry.get('jump_forward_fields')}")
    print(f"  report: {target}")
    if args.json:
        print(json.dumps({"trace": str(args.trace), "html": str(target),
                          "telemetry": telemetry}, indent=2, default=str))
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    if args.from_doc:
        print(
            "error: --from-doc requires a model adapter (not available in this build). "
            "Use --from-json with example objects instead.",
            file=sys.stderr,
        )
        return 2
    if not args.from_json:
        print("error: --from-json <examples.json> is required", file=sys.stderr)
        return 2
    examples = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    source = synth_schema(
        examples,
        name=args.name,
        output=args.out,
        require_ranges=not args.no_ranges,
    )
    if args.out:
        print(f"wrote {args.out}")
    else:
        print(source)
    return 0


# ---------------------------------------------------------------------------
# trace inspector
# ---------------------------------------------------------------------------

def _render_trace_html(payload: Any, *, title: str = "veritas trace") -> str:
    if isinstance(payload, list):
        trace = payload
        telemetry: dict[str, Any] = {}
        status = "unknown"
    else:
        trace = payload.get("trace") or payload.get("steps") or []
        telemetry = payload.get("telemetry") or {}
        status = payload.get("status") or telemetry.get("final_status") or "unknown"
    rows = []
    for index, step in enumerate(trace):
        if not isinstance(step, dict):
            rows.append((index, "", "", 0.0, "sample", ""))
            continue
        text = step.get("text") or step.get("char") or ""
        entropy = float(step.get("entropy") or 0.0)
        kind = step.get("kind") or "sample"
        rows.append(
            (
                index,
                step.get("field") or step.get("field_name") or "",
                text,
                entropy,
                kind,
                step.get("note") or "",
            )
        )
    max_entropy = max((row[3] for row in rows), default=1.0) or 1.0
    body_parts = []
    for index, field, text, entropy, kind, note in rows:
        percent = max(0.02, min(1.0, entropy / max_entropy))
        klass = "splice" if kind == "splice" else ("backtrack" if kind == "backtrack" else "sample")
        body_parts.append(
            f"<tr class='{klass}'><td>{index}</td><td>{html.escape(str(field))}</td>"
            f"<td class='tok'>{html.escape(str(text)) or '&nbsp;'}</td>"
            f"<td><span class='bar' style='width:{percent * 100.0:.1f}%'></span></td>"
            f"<td>{entropy:.3f}</td><td>{html.escape(str(note))}</td></tr>"
        )
    cards = "".join(
        f"<div class='card'><div class='k'>{html.escape(str(k))}</div>"
        f"<div class='v'>{html.escape(str(v))}</div></div>"
        for k, v in sorted(telemetry.items())
        if not isinstance(v, (list, dict))
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{html.escape(title)} — veritas inspect</title>
<style>
:root {{ color-scheme: dark; }}
body {{ font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; margin: 0;
       background: #0b1020; color: #dce3f0; }}
header {{ padding: 24px 32px; border-bottom: 1px solid #1e2a44; }}
h1 {{ font-size: 20px; margin: 0 0 6px; }}
.sub {{ color: #8fa3c8; }}
.cards {{ display: flex; gap: 12px; flex-wrap: wrap; padding: 16px 32px; }}
.card {{ background: #121a30; border: 1px solid #1e2a44; border-radius: 10px; padding: 10px 14px; }}
.card .k {{ color: #7f93b8; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
.card .v {{ font-size: 16px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #17203a; }}
th {{ color: #8fa3c8; font-weight: 500; position: sticky; top: 0; background: #0b1020; }}
tr.splice {{ background: rgba(34, 211, 238, .10); }}
tr.backtrack {{ background: rgba(244, 63, 94, .14); }}
.tok {{ color: #7dd3fc; max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.bar {{ display: inline-block; height: 8px; border-radius: 999px;
        background: linear-gradient(90deg, #22d3ee, #a78bfa); }}
.legend {{ padding: 8px 32px; color: #8fa3c8; }}
</style></head><body>
<header><h1>{html.escape(title)}</h1>
<div class="sub">status: {html.escape(str(status))} · generated by <code>veritas inspect</code></div></header>
<div class="cards">{cards or "<div class='card'><div class='k'>telemetry</div><div class='v'>none</div></div>"}</div>
<div class="legend">cyan = 0-DoF Jump-Forward splice · crimson = CDCL backtrack · bar = Shannon entropy</div>
<table><thead><tr><th>#</th><th>field</th><th>token</th><th>entropy</th><th>H</th><th>note</th></tr></thead>
<tbody>{"".join(body_parts) or "<tr><td colspan='6'>empty trace</td></tr>"}</tbody></table>
</body></html>
"""


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veritas", description="neuro-symbolic verification engine")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("check-spec", help="compile and pre-flight a schema")
    p.add_argument("target", help="path.py[:name] or module[:name]")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check_spec)

    p = sub.add_parser("verify", help="verify a JSON instance against a schema")
    p.add_argument("target")
    p.add_argument("instance", help="JSON file path or '-' for stdin")
    p.add_argument("--name")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-derive", action="store_true", help="do not fill missing derived fields")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("verify-proof", help="verify a proof certificate offline")
    p.add_argument("proof")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify_proof)

    p = sub.add_parser("inspect", help="render a decode trace to standalone HTML")
    p.add_argument("trace")
    p.add_argument("--out")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("synth", help="infer a schema module from JSON examples")
    p.add_argument("--from-json")
    p.add_argument("--from-doc")
    p.add_argument("--name", default="Synthesized")
    p.add_argument("--out")
    p.add_argument("--no-ranges", action="store_true")
    p.set_defaults(func=cmd_synth)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
