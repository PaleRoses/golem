"""Command-line entry point for the body compiler."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import assert_never

from golem.kernel.body.canonical import _canonical_json
from golem.kernel.body.compile import Compiler
from golem.kernel.body.project import rejected_body_text
from golem.kernel.body.report import assertion_records_pass, run_asserts
from golem.kernel.body.types import CompiledBody, RejectedBody
from golem.senses.model import RejectedSenses
from golem.senses.project import rejected_senses_text


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        from golem.kernel import body as _body_pkg
        print(_body_pkg.__doc__)
        return 0
    args = list(argv)
    spec_path = None
    out_path = None
    receipt_path = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-o", "--out"):
            out_path = args[i + 1]
            i += 2
        elif a == "--receipt":
            receipt_path = args[i + 1]
            i += 2
        else:
            spec_path = a
            i += 1
    if spec_path is None:
        print("usage: body.py <spec.json> -o <out.json> [--receipt <r.json>]",
              file=sys.stderr)
        return 2

    p = Path(spec_path)
    spec = json.loads(p.read_text())
    comp = Compiler(spec, spec_dir=p.resolve().parent)
    result = comp.compile()
    match result:
        case CompiledBody(graph, receipt, _):
            pass
        case RejectedBody() as rejected:
            print(rejected_body_text(rejected), file=sys.stderr)
            return 1
        case _ as unreachable:
            assert_never(unreachable)

    if out_path:
        Path(out_path).write_text(_canonical_json(graph))
    if receipt_path:
        Path(receipt_path).write_text(_canonical_json(receipt))

    # run the embedded contract + print a compact receipt with skeleton addrs.
    assertion_result = run_asserts(graph, graph["intent"])
    match assertion_result:
        case RejectedSenses() as rejected:
            sys.stderr.write(rejected_senses_text(rejected))
            return 2
        case (senses, records):
            pass
        case _ as unreachable:
            assert_never(unreachable)
    fails = [r for r in records if r["status"] == "fail"]
    unmeas = [r for r in records if r["status"] == "unmeasurable"]
    passes = [r for r in records if r["status"] == "pass"]
    print(f"BODY {graph['name']} -> {len(graph['parts'])} parts | "
          f"target {comp.target} | ground_lift {receipt['solved']['ground_lift']:+.4f}")
    print(f"  components(predicted) {senses.n_components} | "
          f"solved goals {len(comp.solved_goals)} | "
          f"limit_violations {len(comp.limit_violations)} | "
          f"compile violations {len(comp.violations)}")
    for g in comp.solved_goals:
        tag = "reach" if g.get("reachable") else "CLAMP"
        extra = f" flexion {g.get('flexion_deg')}deg" if g.get("reachable") else \
            f" gap {g.get('gap')}"
        print(f"  goal[{g['id']}] {tag} residual {g['residual']}{extra}")
    print(f"ASSERT {len(fails)} FAIL / {len(passes)} pass"
          + (f" / {len(unmeas)} unmeasurable" if unmeas else ""))
    for r in fails + unmeas:
        print("  " + r["human"])
    for v in comp.violations:
        print(f"  VIOLATION [{v['rule']}] {v['address']}: {v['detail']}")
    return (
        0
        if assertion_records_pass(records)
        and not comp.violations
        and not comp.limit_violations
        else 1
    )
