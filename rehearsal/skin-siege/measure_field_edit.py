# V4b measurement: the edit-loop transaction on the Sable Stride formed
# flesh field at the canonical assembly resolution. Four arms against
# fresh stores: (A) cold build populating both memos, (B) the commission
# target — first evaluation after a bounds-stable local edit (composed
# miss, every unchanged section hits, edited section symbolic), (C) the
# recompose floor — composed store cleared, every section hits, and
# (D) the composed hit. The edited part is chosen so graph bounds stay
# BITWISE identical: the authoring grid derives from bounds, so a
# hull-moving edit shifts every point and lawfully misses every section
# key — that caveat is part of the receipt, not a defect. Writes one
# JSON receipt beside this script. GOLEM_CIRCUIT_CACHE_COLD must be
# unset.
from __future__ import annotations

import copy
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from golem.assembly.compile import pitch_res
from golem.kernel import body
from golem.kernel.engine import compile as engine_compile
from golem.kernel.engine.compile import (
    _evaluate_certified_checked,
    graph_bounds_checked,
)
from golem.kernel.engine.types import (
    GeometryDecodeFailure,
    decode_graph,
    require_accepted,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "specs" / "sable_stride.json"
RECEIPT = Path(__file__).with_name("measure-field-edit.json")

THICKNESS = 0.004
SHRINK = 0.98


def formed_spec() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for layer in spec["anatomy"]["overall"]["integument_layers"]:
        layer["formation"] = "static_implicit_relaxation"
        assert abs(layer["thickness"] - THICKNESS) < 1e-15
    return spec


def shrunk(part: dict) -> dict:
    edited = copy.deepcopy(part)
    if "size" in edited:
        edited["size"] = [value * SHRINK for value in edited["size"]]
    elif "radii" in edited:
        edited["radii"] = [value * SHRINK for value in edited["radii"]]
    else:
        raise AssertionError(f"no shrinkable field on {part.get('id')}")
    return edited


def bounds_stable_edit(graph: dict) -> tuple[dict, str]:
    # Unmirrored parts first: a mirrored part is two instances, so its
    # edit re-samples two sections and doubles the store delta.
    base_lower, base_upper = graph_bounds_checked(
        require_accepted(decode_graph(graph))
    )
    candidates = sorted(
        enumerate(graph["parts"]), key=lambda pair: bool(pair[1].get("mirror"))
    )
    for index, part in candidates:
        edited_graph = copy.deepcopy(graph)
        edited_graph["parts"][index] = shrunk(part)
        try:
            decoded = require_accepted(decode_graph(edited_graph))
        except GeometryDecodeFailure:
            continue
        lower, upper = graph_bounds_checked(decoded)
        if np.array_equal(lower, base_lower) and np.array_equal(
            upper, base_upper
        ):
            return edited_graph, str(part.get("id"))
    raise AssertionError("no bounds-stable single-part edit exists")


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
    seal_resolution = pitch_res(graph, 0.017)
    resolution = int(sys.argv[1]) if len(sys.argv) > 1 else seal_resolution
    edited_graph, edited_part = bounds_stable_edit(graph)
    base = require_accepted(decode_graph(graph))
    edited = require_accepted(decode_graph(edited_graph))

    composed_store = Path(tempfile.mkdtemp(prefix="field-edit-composed-"))
    section_store = Path(tempfile.mkdtemp(prefix="field-edit-sections-"))
    engine_compile._composed_snapshot_root = lambda: composed_store
    engine_compile._section_sample_root = lambda: section_store

    def evaluate(decoded) -> object:
        return _evaluate_certified_checked(decoded, resolution)

    def clear_composed() -> None:
        for path in composed_store.glob("*.pkl"):
            path.unlink()

    try:
        base_miss_seconds = timed(lambda: evaluate(base))
        clear_composed()
        edit_warm_seconds = timed(lambda: evaluate(edited))
        recompose_seconds = tuple(
            (clear_composed(), timed(lambda: evaluate(edited)))[1]
            for _ in range(2)
        )
        hit_seconds = tuple(
            timed(lambda: evaluate(edited)) for _ in range(3)
        )
        section_entries = tuple(section_store.rglob("*.npz"))
        section_bytes = sum(path.stat().st_size for path in section_entries)
    finally:
        shutil.rmtree(composed_store, ignore_errors=True)
        shutil.rmtree(section_store, ignore_errors=True)

    receipt = {
        "geometry": "sable_stride formed (static_implicit_relaxation)",
        "resolution": resolution,
        "seal_resolution": seal_resolution,
        "edited_part": edited_part,
        "edit": f"uniform shrink x{SHRINK} (bounds-stable, bitwise)",
        "positive_sections": len(graph["parts"])
        + len(graph.get("webs", ())),
        "carves": len(graph.get("carves", ())),
        "section_store_entries": len(section_entries),
        "section_store_bytes": section_bytes,
        "base_miss_seconds": base_miss_seconds,
        "edit_warm_seconds": edit_warm_seconds,
        "recompose_floor_seconds": recompose_seconds,
        "recompose_floor_median_seconds": statistics.median(
            recompose_seconds
        ),
        "composed_hit_seconds": hit_seconds,
        "composed_hit_median_seconds": statistics.median(hit_seconds),
        "edit_speedup_vs_miss": base_miss_seconds / edit_warm_seconds,
        "v4a_miss_baseline_seconds": 25.36,
        "v4a_hit_baseline_seconds": 0.228,
        "entry_001_baseline_seconds": 20.50,
        "caveat": "wins apply to bounds-stable edits; a hull-moving edit "
        "shifts the derived grid and lawfully misses every section key",
    }
    suffix = "" if resolution == seal_resolution else f"-res{resolution}"
    receipt_path = RECEIPT.with_name(f"measure-field-edit{suffix}.json")
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
