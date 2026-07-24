# Skin-siege measurement: ray-root distribution on the Sable Stride
# formed-flesh field. Reproduces the sealed scaffold ray stencil
# (symmetric trilinear gradient normals, directional boundary extents)
# with a dense sample count and records, per flesh vertex, every
# Psi = F_flesh - thickness root along the outward ray:
#   (a) first-root distance / thickness   -> lower bound for K
#   (b) second-root distance / thickness  -> upper bound for K
# Receipts feed docs/plans/skin-siege-journal.md. Read-only w.r.t. the
# sealed formation-trial tree; writes only its own JSON receipt beside
# this script.
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from golem.assembly.compile import pitch_res
from golem.kernel import body
from golem.kernel.engine.compile import (
    _evaluate_certified_checked,
    _symmetric_trilinear_gradients,
    _trilinear_samples,
)
from golem.kernel.engine.types import decode_graph, require_accepted

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "specs" / "sable_stride.json"
RECEIPT = Path(__file__).with_name("measure-root-distribution.json")

THICKNESS = 0.004
SAMPLE_COUNT = 1024
VERTEX_CHUNK = 2048


def formed_spec() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    layers = spec["anatomy"]["overall"]["integument_layers"]
    for layer in layers:
        layer["formation"] = "static_implicit_relaxation"
        assert abs(layer["thickness"] - THICKNESS) < 1e-15
    return spec


def main() -> None:
    compiled = body.Compiler(formed_spec(), spec_dir=SPEC.parent).compile()
    assert isinstance(compiled, body.CompiledBody), "formed body must compile"
    graph = compiled.graph
    assert graph.get("skin") is not None, "formed token must reach the graph"

    resolution = pitch_res(graph, 0.017)
    decoded = require_accepted(decode_graph(graph))

    start = time.perf_counter()
    flesh = _evaluate_certified_checked(decoded, resolution)
    flesh_seconds = time.perf_counter() - start
    assert not hasattr(flesh, "obstructions"), f"flesh rejected: {flesh!r}"

    vertices = np.asarray(flesh.vertices, dtype=np.float64)
    lower = np.asarray(flesh.lower, dtype=np.float64)
    upper = np.asarray(flesh.upper, dtype=np.float64)
    pitch = np.asarray(flesh.world_pitch, dtype=np.float64)
    skin_field = np.asarray(flesh.field, dtype=np.float64) - THICKNESS

    gradients = _symmetric_trilinear_gradients(
        skin_field, lower, pitch, vertices
    )
    magnitudes = np.linalg.norm(gradients, axis=1)
    normals = gradients / magnitudes[:, None]

    directional = np.where(
        normals > 0.0,
        np.divide(
            upper - vertices,
            normals,
            out=np.full_like(vertices, np.inf),
            where=normals > 0.0,
        ),
        np.where(
            normals < 0.0,
            np.divide(
                lower - vertices,
                normals,
                out=np.full_like(vertices, np.inf),
                where=normals < 0.0,
            ),
            np.inf,
        ),
    )
    boundary_extents = np.min(directional, axis=1)

    vertex_count = len(vertices)
    first_roots = np.full(vertex_count, np.nan)
    second_roots = np.full(vertex_count, np.nan)
    root_counts = np.zeros(vertex_count, dtype=np.int64)

    start = time.perf_counter()
    for offset in range(0, vertex_count, VERTEX_CHUNK):
        stop = min(offset + VERTEX_CHUNK, vertex_count)
        count = stop - offset
        parameters = (
            boundary_extents[offset:stop, None]
            * np.linspace(0.0, 1.0, SAMPLE_COUNT)[None, :]
        )
        points = (
            vertices[offset:stop, None, :]
            + parameters[:, :, None] * normals[offset:stop, None, :]
        )
        values = _trilinear_samples(
            skin_field, lower, pitch, points.reshape((-1, 3))
        ).values.reshape((count, SAMPLE_COUNT))
        signs = np.signbit(values)
        crossings = signs[:, :-1] != signs[:, 1:]
        counts = np.count_nonzero(crossings, axis=1)
        root_counts[offset:stop] = counts
        indices = np.argsort(~crossings, axis=1, kind="stable")
        ordered = np.take_along_axis(crossings, indices, axis=1)
        positions = np.take_along_axis(parameters, indices, axis=1)
        has_first = ordered[:, 0] & (counts >= 1)
        has_second = ordered[:, 1] & (counts >= 2)
        first_roots[offset:stop] = np.where(
            has_first, positions[:, 0], np.nan
        )
        second_roots[offset:stop] = np.where(
            has_second, positions[:, 1], np.nan
        )
    stencil_seconds = time.perf_counter() - start

    rooted = ~np.isnan(first_roots)
    multi = ~np.isnan(second_roots)
    first_over_t = first_roots[rooted] / THICKNESS
    second_over_t = second_roots[multi] / THICKNESS

    receipt = {
        "resolution": int(resolution),
        "vertex_count": int(vertex_count),
        "thickness": THICKNESS,
        "sample_count": SAMPLE_COUNT,
        "flesh_eval_seconds": flesh_seconds,
        "stencil_seconds": stencil_seconds,
        "rooted_vertices": int(np.count_nonzero(rooted)),
        "rootless_vertices": int(np.count_nonzero(~rooted)),
        "multi_root_vertices": int(np.count_nonzero(multi)),
        "root_count_histogram": {
            str(count): int(np.count_nonzero(root_counts == count))
            for count in sorted(set(root_counts.tolist()))
        },
        "first_root_over_thickness": {
            "min": float(np.min(first_over_t)),
            "p50": float(np.percentile(first_over_t, 50)),
            "p95": float(np.percentile(first_over_t, 95)),
            "p99": float(np.percentile(first_over_t, 99)),
            "p999": float(np.percentile(first_over_t, 99.9)),
            "max": float(np.max(first_over_t)),
        },
        "second_root_over_thickness": {
            "min": float(np.min(second_over_t)),
            "p01": float(np.percentile(second_over_t, 1)),
            "p05": float(np.percentile(second_over_t, 5)),
            "p50": float(np.percentile(second_over_t, 50)),
            "max": float(np.max(second_over_t)),
        }
        if multi.any()
        else None,
        "multi_root_vertex_indices": [
            int(index) for index in np.flatnonzero(multi)
        ],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "multi_root_vertex_indices"}, indent=1))


if __name__ == "__main__":
    main()
