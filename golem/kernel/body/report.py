"""Contract evaluation over compiled senses + the deterministic compile digest."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from golem.senses.model import RejectedSenses, Senses

if TYPE_CHECKING:
    from golem.contracts.model import Verdict


# --------------------------------------------------------------------------- #
# Skeleton-space provenance display is owned by ``asserts.py`` (its guarded     #
# provenance passthrough reads ``senses["provenance"]``); body.py does not      #
# re-implement it, keeping a single source of truth for the address round-trip. #
# --------------------------------------------------------------------------- #
def run_typed_asserts(
    graph: dict, intent: dict
) -> RejectedSenses | tuple[Senses, tuple["Verdict", ...]]:
    from golem.contracts.verdicts import evaluate_verdict_pack
    from golem.senses import proprio

    senses_result = proprio.build_senses(graph, intent)
    match senses_result:
        case RejectedSenses() as rejected:
            return rejected
        case Senses() as senses:
            pass
        case _ as unreachable:
            assert_never(unreachable)
    clauses = (intent.get("asserts") or {}).get("clauses", [])
    return senses, evaluate_verdict_pack(clauses, senses)


def run_asserts(
    graph: dict, intent: dict
) -> RejectedSenses | tuple[Senses, list[dict[str, object]]]:
    from golem.contracts.views import verdict_record

    typed_result = run_typed_asserts(graph, intent)
    match typed_result:
        case RejectedSenses() as rejected:
            return rejected
        case (senses, verdicts):
            return senses, list(map(verdict_record, verdicts))
        case _ as unreachable:
            assert_never(unreachable)


def assertion_records_pass(records: list[dict[str, object]]) -> bool:
    return all(record["status"] == "pass" for record in records)


# --------------------------------------------------------------------------- #
# Deterministic compile digest (a platform-stable, byte-freezable summary).    #
# Floats are formatted at the precision the M2 golden proved libm-drift-free   #
# (<=3 decimals; flexion at 1); the EMIT block is pure provenance strings, so  #
# the whole digest is safe as a write-once golden. It is NOT the receipt (the  #
# receipt keeps full 6-decimal landmark coordinates for downstream senses).    #
# --------------------------------------------------------------------------- #
def render_digest(graph: dict, receipt: dict, senses: "Senses",
                  records: list[dict]) -> str:
    solved = receipt["solved"]
    fails = [r for r in records if r["status"] == "fail"]
    passes = [r for r in records if r["status"] == "pass"]
    unmeas = [r for r in records if r["status"] == "unmeasurable"]
    lines = [
        f"BODY {graph['name']} | target {receipt['target']} | "
        f"parts {len(graph['parts'])} | ground_lift {solved['ground_lift']:+.3f}",
        f"predicted_components {senses.n_components} | "
        f"goals {len(solved['goals'])} | "
        f"limit_violations {len(solved['limit_violations'])} | "
        f"compile_violations {len(receipt['violations'])}",
    ]
    for g in solved["goals"]:
        if g.get("reachable"):
            lines.append(f"goal {g['id']} reachable=yes "
                         f"flexion {g['flexion_deg']:.1f} "
                         f"residual {g['residual']:.3f}")
        else:
            lines.append(f"goal {g['id']} reachable=no "
                         f"gap {g['gap']:.3f} residual {g['residual']:.3f}")
    tail = f" / {len(unmeas)} unmeasurable" if unmeas else ""
    lines.append(f"ASSERT {len(fails)} FAIL / {len(passes)} pass{tail}")
    prov = receipt["provenance"]
    lines.append(f"EMIT ({len(prov)})")
    for pid in sorted(prov):
        lines.append(f"  {pid} <- {prov[pid]}")
    return "\n".join(lines) + "\n"
