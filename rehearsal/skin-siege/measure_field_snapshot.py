# V4 measurement: composed-field snapshot memo on the Sable Stride
# formed-flesh evaluation at the canonical assembly resolution (the
# entry-001 20.50s baseline path). Two arms against a fresh snapshot
# store: "miss" evaluates the full certified fold and stores; "hit"
# returns the stored snapshot. Writes one JSON receipt beside this
# script; touches nothing else. GOLEM_CIRCUIT_CACHE_COLD must be unset
# (this probes the warm authoring path; cold sealing is unaffected).
from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path

from golem.assembly.compile import pitch_res
from golem.kernel import body
from golem.kernel.engine import compile as engine_compile
from golem.kernel.engine.compile import _evaluate_certified_checked
from golem.kernel.engine.types import decode_graph, require_accepted

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "specs" / "sable_stride.json"
RECEIPT = Path(__file__).with_name("measure-field-snapshot.json")

THICKNESS = 0.004
MISS_SAMPLES = 2
HIT_SAMPLES = 5


def formed_spec() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for layer in spec["anatomy"]["overall"]["integument_layers"]:
        layer["formation"] = "static_implicit_relaxation"
        assert abs(layer["thickness"] - THICKNESS) < 1e-15
    return spec


def timed(action) -> float:
    start = time.perf_counter()
    result = action()
    elapsed = time.perf_counter() - start
    assert not hasattr(result, "obstructions"), f"rejected: {result!r}"
    return elapsed


def main() -> None:
    compiled = body.Compiler(formed_spec(), spec_dir=SPEC.parent).compile()
    assert isinstance(compiled, body.CompiledBody)
    graph = compiled.graph
    resolution = pitch_res(graph, 0.017)
    decoded = require_accepted(decode_graph(graph))

    store = Path(tempfile.mkdtemp(prefix="field-snapshot-probe-"))
    engine_compile._composed_snapshot_root = lambda: store

    def evaluate() -> object:
        return _evaluate_certified_checked(decoded, resolution)

    def clear() -> None:
        for path in store.glob("*.pkl"):
            path.unlink()

    miss_seconds = tuple(
        (clear(), timed(evaluate))[1] for _ in range(MISS_SAMPLES)
    )
    hit_seconds = tuple(timed(evaluate) for _ in range(HIT_SAMPLES))

    receipt = {
        "geometry": "sable_stride formed (static_implicit_relaxation)",
        "resolution": resolution,
        "miss_seconds": miss_seconds,
        "hit_seconds": hit_seconds,
        "miss_median_seconds": statistics.median(miss_seconds),
        "hit_median_seconds": statistics.median(hit_seconds),
        "snapshot_speedup": (
            statistics.median(miss_seconds) / statistics.median(hit_seconds)
        ),
        "entry_001_baseline_seconds": 20.50,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
